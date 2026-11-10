package serve

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Warm-model keep-alive: when a client (`run`) exits, it may ask this
// daemon to hold the loaded model for a bounded window so the next `run`
// is warm instead of paying a full reload (POST /-/hold {"seconds": N};
// the effective window is the config knob keepalive_seconds, overridden
// by `run --keepalive N`). Expiry unloads through the same
// /v1/model/unload path /-/unload uses, so VRAM is actually freed —
// nothing leaks. Semantics:
//
//	0 / absent   unload immediately (the DEFAULT — a held model, ~2.3 GiB
//	             or ~62% of the reference 4 GiB card, cannot coexist with
//	             anything else that wants the GPU, e.g. the owner's games)
//	N            hold for N seconds after the last client exits
//
// The daemon never starts a hold on its own: it cannot observe "the last
// client went away", so the client drives the window. markWanted at chat
// entry cancels a pending window; expiry never unloads a wanted model.

// markWanted records that a client is using the model right now: the
// hold window is cancelled until the client goes away again.
func (d *daemon) markWanted() {
	d.mu.Lock()
	d.holdUntil = time.Time{}
	d.mu.Unlock()
}

// handleHold serves POST /-/hold {"seconds": N}: ask the daemon to hold
// the loaded model N seconds after the last client goes away. N <= 0
// behaves exactly like /-/unload (immediate). With N > 0 a live foreign
// GPU contender refuses the hold and unloads instead — the fail-safe
// direction is always "free the card". On the llama backend the model
// is resident for the process lifetime (no unload endpoint), so no hold
// is recorded and the answer says so. Control-route authed (state
// token) like /-/unload.
func (d *daemon) handleHold(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, "use POST", "method_not_allowed")
		return
	}
	var req struct {
		Seconds int `json:"seconds"`
	}
	_ = json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req)
	if req.Seconds <= 0 {
		d.handleUnload(w, r)
		return
	}
	if _, isLlama := d.be.(llamaBackend); isLlama {
		// llama-server's model is resident for the process lifetime (no
		// /v1/model/unload — expiry would 404 and hold a deadline the
		// backend can never deliver). Recording a hold would be a lie:
		// accept the request, hold nothing, say so.
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"hold_until":0,"note":"llama backend: the model is resident for the server process lifetime; keep-alive cannot unload it"}`+"\n")
		return
	}
	if ce := d.contention(); ce != nil {
		d.handleUnload(w, r)
		fmt.Fprintf(d.opts.Stderr, "stone-llama: hold refused — %v\n", ce)
		return
	}
	d.mu.Lock()
	d.holdUntil = d.now().Add(time.Duration(req.Seconds) * time.Second)
	until := d.holdUntil
	d.mu.Unlock()
	d.holdWake(until)
	w.Header().Set("Content-Type", "application/json")
	_, _ = fmt.Fprintf(w, "{\"hold_until\":%d}\n", until.Unix())
}

// modelHoldUntil reports the unix second the hold ends (0 = none).
func (d *daemon) modelHoldUntil() int64 {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.holdUntil.IsZero() {
		return 0
	}
	return d.holdUntil.Unix()
}

// holdExpiry blocks until the hold window ends, the daemon stops, or a
// client touched the model again (holdUntil cleared). True = expired
// (caller unloads); false = superseded or shutting down.
func (d *daemon) holdExpiry(ctx context.Context) bool {
	d.mu.Lock()
	until := d.holdUntil
	d.mu.Unlock()
	if until.IsZero() {
		return false
	}
	timer := time.NewTimer(time.Until(until))
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		d.mu.Lock()
		same := d.holdUntil.Equal(until)
		d.mu.Unlock()
		return same
	}
}

// keepAliveMaintenance runs the expiry loop for the daemon's lifetime
// (started by Serve next to sessionMaintenance). A hold POSTed to
// /-/hold nudges holdTick; the loop waits out the window and unloads on
// expiry. Each wake re-checks contention: a foreign GPU holder appearing
// mid-hold (the owner's game) unloads immediately instead of waiting the
// window out — the card must be reclaimable while we merely hold.
func (d *daemon) keepAliveMaintenance(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-d.holdTick:
			for {
				if !d.modelLoaded() || d.modelHoldUntil() == 0 {
					break
				}
				if ce := d.contention(); ce != nil {
					fmt.Fprintf(d.opts.Stderr, "stone-llama: hold cut short — %v\n", ce)
					d.expireUnload()
					break
				}
				if d.holdExpiry(ctx) {
					d.expireUnload()
					break
				}
			}
		}
	}
}

// expireUnload performs the TTL unload with /-/unload's semantics:
// backend unload first, state cleared only on 2xx. Errors are logged;
// a failed unload leaves the model resident (ps shows it) — never
// silently claimed gone. loadMu serializes against handleLoad/
// handleUnload.
func (d *daemon) expireUnload() {
	d.loadMu.Lock()
	defer d.loadMu.Unlock()
	if !d.modelLoaded() {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		d.conn.BaseURL+"/v1/model/unload", strings.NewReader("{}"))
	if err != nil {
		fmt.Fprintf(d.opts.Stderr, "stone-llama: keepalive expiry: %v\n", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	if d.upKey != "" {
		req.Header.Set("Authorization", "Bearer "+d.upKey)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		fmt.Fprintf(d.opts.Stderr, "stone-llama: keepalive expiry: %v\n", err)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		fmt.Fprintf(d.opts.Stderr, "stone-llama: keepalive expiry: backend refused unload (HTTP %d) — model still resident\n", resp.StatusCode)
		return
	}
	d.clearLoaded()
	fmt.Fprintf(d.opts.Stderr, "stone-llama: model unloaded (keepalive expired)\n")
}
