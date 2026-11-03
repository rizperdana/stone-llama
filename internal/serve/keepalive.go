package serve

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// Warm-model keep-alive: after the last client goes away the daemon
// holds the loaded model for a bounded window (default 90 s) so the
// next `run` is warm instead of paying a full reload. Expiry unloads
// through the same /v1/model/unload path `/-/unload` uses, so VRAM is
// actually freed — nothing leaks. The knob (config keepalive_seconds /
// --keepalive SECONDS on serve+run):
//
//	0   unload immediately (the old behaviour — free the card now)
//	N   hold for N seconds after the last client exits
//
// A resident model holds ~62% of a 4 GB card, so the default is short:
// it covers "closed it, one more question" without pinning VRAM all day.

// defaultKeepAlive is the hold window when neither flag nor config set
// it. Short by design: instant warm restart beats reload, but a game
// (or anything else) must be able to reclaim the VRAM promptly.
const defaultKeepAlive = 90 * time.Second

// markWanted records that a client is using the model right now: the
// hold window is cancelled until the client goes away again.
func (d *daemon) markWanted() {
	d.mu.Lock()
	d.holdUntil = time.Time{}
	d.mu.Unlock()
}

// startHold begins the countdown (client went away). A non-positive
// keepAlive never holds (immediate-unload policy).
func (d *daemon) startHold() {
	d.mu.Lock()
	if d.keepAliveDur() <= 0 {
		d.mu.Unlock()
		return
	}
	d.holdUntil = d.now().Add(d.keepAliveDur())
	until := d.holdUntil
	d.mu.Unlock()
	d.holdWake(until)
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
// (started by Serve next to sessionMaintenance). startHold nudges
// holdTick; the loop waits out the window and unloads on expiry.
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
