package serve

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/rizperdana/stone-llama/internal/autofit"
	"github.com/rizperdana/stone-llama/internal/store"
)

// Status is the /-/status payload — ps and run both read it.
type Status struct {
	Mode       string `json:"mode"` // "supervised" | "attach"
	Upstream   string `json:"upstream"`
	Attach     string `json:"attach,omitempty"`
	Host       string `json:"host"`
	Port       int    `json:"port"`
	PID        int    `json:"pid"`
	ChildPID   int    `json:"child_pid,omitempty"`
	Model      string `json:"model,omitempty"`
	Ctx        int    `json:"ctx,omitempty"`
	CacheMode  string `json:"cache_mode,omitempty"`
	VRAMPeak   int    `json:"vram_peak_mib,omitempty"`
	UptimeS    int64  `json:"uptime_s"`
	Ready      bool   `json:"ready"`
	FitSummary string `json:"fit_summary,omitempty"` // last accepted autofit breakdown
}

// LoadRequest is the /-/load payload (run, tests, direct clients).
type LoadRequest struct {
	Model     string `json:"model"`
	Ctx       int    `json:"ctx,omitempty"`
	CacheMode string `json:"cache_mode,omitempty"`
	// NoAutofit bypasses the VRAM ladder: ctx/cache pass through as given
	// (empty = backend defaults).
	NoAutofit bool `json:"no_autofit,omitempty"`
}

// handleStatus serves GET /-/status from state plus one readiness probe
// (contract item 2: any answer from {BaseURL}/v1/models = serving).
func (d *daemon) handleStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, "use GET", "method_not_allowed")
		return
	}
	d.mu.Lock()
	st := d.st
	fitSum := ""
	if d.fit != nil {
		fitSum = d.fit.Summary()
	}
	d.mu.Unlock()

	model := st.Model
	if st.Attach != "" {
		// The upstream owns its model in attach mode; ask it (best
		// effort — non-standard endpoint, display-only, not contract).
		model = d.attachModel()
	}
	resp := Status{
		Mode:       "supervised",
		Upstream:   d.conn.BaseURL,
		Attach:     st.Attach,
		Host:       st.Host,
		Port:       st.Port,
		PID:        st.PID,
		ChildPID:   st.ChildPID,
		Model:      model,
		Ctx:        st.Ctx,
		CacheMode:  st.CacheMode,
		FitSummary: fitSum,
		VRAMPeak:   st.VRAMPeakMiB,
		UptimeS:    d.opts.Now().Unix() - st.StartedAt,
		Ready:      d.readyNow(),
	}
	if st.Attach != "" {
		resp.Mode = "attach"
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// readyNow probes the backend per the contract.
func (d *daemon) readyNow() bool {
	if d.conn.BaseURL == "" {
		return false
	}
	if d.port != 0 { // supervised: probe the child directly
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		return d.opts.probe(ctx, d.port) == nil
	}
	return probeAttach(context.Background(), d.conn.BaseURL, 2*time.Second) == nil
}

// attachModel returns the attach upstream's loaded model id, or "".
func (d *daemon) attachModel() string {
	ctx, cancel := context.WithTimeout(context.Background(), 700*time.Millisecond)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, d.conn.BaseURL+"/v1/model", nil)
	if err != nil {
		return ""
	}
	if d.upKey != "" {
		req.Header.Set("Authorization", "Bearer "+d.upKey)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return ""
	}
	var m struct {
		ID string `json:"id"`
	}
	if json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&m) != nil {
		return ""
	}
	return m.ID
}

// handleLoad loads a model through the backend (contract: the backend
// owns load orchestration; we own fit, progress streaming, and state).
// The stream is zero-buffer: every upstream event line is written and
// flushed the moment it arrives.
func (d *daemon) handleLoad(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, "use POST", "method_not_allowed")
		return
	}
	if d.opts.Attach != "" {
		writeJSON(w, http.StatusConflict,
			"attach mode: model loading is owned by the upstream server — restart serve without --attach to load models here",
			"attach_mode")
		return
	}
	d.loadMu.Lock()
	defer d.loadMu.Unlock()

	var req LoadRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, "decode /-/load body: "+err.Error(), "invalid_request")
		return
	}
	if req.Model == "" {
		writeJSON(w, http.StatusBadRequest, "load requires a model name — 'stone-llama list' for options", "invalid_request")
		return
	}
	if filepath.Base(req.Model) != req.Model || strings.HasPrefix(req.Model, ".") {
		writeJSON(w, http.StatusBadRequest, fmt.Sprintf("invalid model name %q", req.Model), "invalid_request")
		return
	}
	ctx := req.Ctx
	if ctx < 0 {
		writeJSON(w, http.StatusBadRequest, "ctx must be >= 0", "invalid_request")
		return
	}
	if ctx > 0 && ctx%256 != 0 {
		writeJSON(w, http.StatusBadRequest,
			fmt.Sprintf("ctx must be a multiple of 256 (got %d)", ctx), "invalid_request")
		return
	}
	mode := req.CacheMode
	if mode != "" {
		if _, err := autofit.BytesPerElement(mode); err != nil {
			writeJSON(w, http.StatusBadRequest,
				fmt.Sprintf("invalid cache_mode %q (want FP16, Q8..Q2, or a k,v pair like \"2,4\")", mode),
				"invalid_request")
			return
		}
	}

	dir := filepath.Join(d.opts.ModelsDir, req.Model)
	if _, err := filepath.EvalSymlinks(dir); err != nil {
		writeJSON(w, http.StatusNotFound,
			fmt.Sprintf("model %q not found in %s — 'stone-llama list' for options", req.Model, d.opts.ModelsDir),
			"not_found")
		return
	}
	var fitRes *autofit.Result
	if !req.NoAutofit {
		var res autofit.Result
		if err := d.fitModel(w, dir, req.Model, &ctx, &mode, &res); err != nil {
			return // fitModel already wrote the refusal envelope
		}
		d.mu.Lock()
		d.fit = &res
		d.mu.Unlock()
		fitRes = &res
	}

	// Load through the backend's native endpoint; the event stream is
	// forwarded verbatim except OOM/contention enrichment.
	payload := map[string]any{"model_name": req.Model, "gpu_split_auto": true}
	if ctx > 0 {
		payload["max_seq_len"] = ctx
		payload["cache_size"] = ctx
	}
	if mode != "" {
		payload["cache_mode"] = NormalizeForTabby(mode)
	}
	// The same autofit.Result that chose ctx/cache above decides load
	// tuning — payload and rendered config can never disagree.
	if fitRes != nil {
		for k, v := range fitLoadArgs(fitRes) {
			payload[k] = v
		}
	}
	body, _ := json.Marshal(payload)
	hreq, err := http.NewRequestWithContext(r.Context(), http.MethodPost,
		d.conn.BaseURL+"/v1/model/load", bytes.NewReader(body))
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, err.Error(), "internal")
		return
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "text/event-stream")
	if d.upKey != "" {
		hreq.Header.Set("Authorization", "Bearer "+d.upKey)
	}
	resp, err := http.DefaultClient.Do(hreq)
	if err != nil {
		writeJSON(w, http.StatusBadGateway,
			d.enrichLoadError("upstream load failed: "+err.Error()), "load_failed")
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrBody))
		code := resp.StatusCode
		if code >= 500 {
			code = http.StatusBadGateway
		}
		writeJSON(w, code, d.enrichLoadError(extractMessage(b)), "load_failed")
		return
	}

	// zero-buffer SSE: headers first, then every event line flushed on
	// arrival (ARCHITECTURE §7; first-token latency is the contract).
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	fl, _ := w.(http.Flusher)
	flush := func() {
		if fl != nil {
			fl.Flush()
		}
	}
	flush()
	br := bufio.NewReader(resp.Body)
	for {
		line, err := br.ReadString('\n')
		if len(line) > 0 {
			if out, changed := d.rewriteLoadEvent(line, req.Model, ctx, mode); changed {
				line = out
			}
			if _, werr := io.WriteString(w, line); werr != nil {
				return // client went away
			}
			flush()
		}
		if err != nil {
			return
		}
	}
}

// fitModel runs the VRAM ladder for a supervised load and mutates
// *ctx/*mode to the decision. On refusal it writes the 422 envelope
// (arithmetic + largest fitting ctx) and returns a sentinel error.
func (d *daemon) fitModel(w http.ResponseWriter, dir, model string, ctx *int, mode *string, res *autofit.Result) error {
	rep, err := d.opts.probeGPU()
	if err != nil || len(rep.GPUs) == 0 {
		note := rep.Note
		if note == "" {
			note = fmt.Sprintf("probe error: %v", err)
		}
		writeJSON(w, http.StatusUnprocessableEntity,
			fmt.Sprintf("no NVIDIA GPU available — stone-llama is CUDA-only ('stone-llama doctor' for detail): %s", note),
			"out_of_vram")
		return errFitRefused
	}
	spec, serr := store.SpecFromDir(dir)
	if serr != nil {
		writeJSON(w, http.StatusBadRequest,
			fmt.Sprintf("model %q has no readable config.json (unsupported or corrupt): %v", model, serr),
			"invalid_model")
		return errFitRefused
	}
	out, ferr := autofit.Fit(spec, store.WeightsBytes(dir), rep.GPUs[0].VRAMMiB, autofit.Options{
		HeadroomMiB:       d.opts.Fit.HeadroomMiB,
		WorkspaceMiB:      d.opts.Fit.WorkspaceMiB,
		CtxHeadroomMiB:    d.opts.Fit.CtxHeadroomMiB,
		OverheadMiB:       d.opts.Fit.OverheadMiB,
		MinCtx:            d.opts.Fit.MinCtx,
		ChunkSizeOverride: d.opts.Fit.ChunkSize,
		UserCtx:           *ctx,
		ForceMode:         *mode,
	})
	if ferr != nil {
		writeJSON(w, http.StatusUnprocessableEntity, ferr.Error(), "invalid_model")
		return errFitRefused
	}
	if !out.Fits {
		writeJSON(w, http.StatusUnprocessableEntity, out.Reason, "out_of_vram")
		return errFitRefused
	}
	*ctx, *mode = out.Ctx, out.Mode
	*res = out
	return nil
}

// fitLoadArgs maps an accepted autofit verdict onto TabbyAPI's
// ModelLoadRequest keys (chunk_size and warmup are both accepted fields
// at the pinned commit — endpoints/core/types/model.py). The verdict is
// the only source: nothing here computes, so payload and rendered YAML
// cannot disagree. Unset verdict fields emit no key (backend defaults).
func fitLoadArgs(res *autofit.Result) map[string]any {
	args := map[string]any{}
	if res.ChunkSize > 0 {
		args["chunk_size"] = res.ChunkSize
	}
	if res.Warmup {
		args["warmup"] = true
	}
	return args
}

// errFitRefused signals that fitModel already wrote the response.
var errFitRefused = fmt.Errorf("fit refused")

// rewriteLoadEvent post-processes one upstream SSE line: OOM messages
// gain the lever advice, load failures gain GPU-contention evidence, a
// "finished" event marks the model loaded in state. Returns the line
// to write and whether it changed.
func (d *daemon) rewriteLoadEvent(line, model string, ctx int, mode string) (string, bool) {
	if !strings.HasPrefix(line, "data:") {
		return line, false
	}
	payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
	var m map[string]any
	if err := json.Unmarshal([]byte(payload), &m); err != nil {
		return line, false
	}
	changed := false
	if em, ok := m["error"].(map[string]any); ok {
		if s, ok := em["message"].(string); ok {
			em["message"] = d.enrichLoadError(s)
			changed = true
		}
	}
	if s, _ := m["status"].(string); s == "finished" {
		d.markLoaded(model, ctx, mode)
	}
	if !changed {
		return line, false
	}
	b, err := json.Marshal(m)
	if err != nil {
		return line, false
	}
	return "data: " + string(b) + "\n", true
}

// enrichLoadError adds actionable evidence to a failed load: OOM gets
// the ctx/cache levers with this daemon's actual numbers; a foreign VRAM
// holder gets the doctor's contention error verbatim.
func (d *daemon) enrichLoadError(msg string) string {
	if isOOM([]byte(msg)) {
		msg += "\n" + d.oomAdvice()
	}
	if ce := d.contention(); ce != nil {
		msg += "; " + ce.Error()
	}
	return msg
}

// markLoaded records a successful load (peak resets with the model).
func (d *daemon) markLoaded(model string, ctx int, mode string) {
	d.mu.Lock()
	d.st.Model = model
	d.st.Ctx = ctx
	d.st.CacheMode = mode
	d.st.VRAMPeakMiB = 0
	err := WriteState(d.opts.DataDir, d.st)
	d.mu.Unlock()
	if err != nil {
		fmt.Fprintf(d.opts.Stderr, "stone-llama: persist state: %v\n", err)
	}
}

// clearLoaded wipes the loaded-model fields (unload; a crash-restart
// loses the model too).
func (d *daemon) clearLoaded() {
	d.mu.Lock()
	d.st.Model = ""
	d.st.Ctx = 0
	d.st.CacheMode = ""
	d.fit = nil
	err := WriteState(d.opts.DataDir, d.st)
	d.mu.Unlock()
	if err != nil {
		fmt.Fprintf(d.opts.Stderr, "stone-llama: persist state: %v\n", err)
	}
}

// handleUnload asks the backend to unload; 2xx clears our state.
func (d *daemon) handleUnload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, "use POST", "method_not_allowed")
		return
	}
	if d.opts.Attach != "" {
		writeJSON(w, http.StatusConflict,
			"attach mode: model loading is owned by the upstream server", "attach_mode")
		return
	}
	d.loadMu.Lock()
	defer d.loadMu.Unlock()

	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost,
		d.conn.BaseURL+"/v1/model/unload", strings.NewReader("{}"))
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, err.Error(), "internal")
		return
	}
	req.Header.Set("Content-Type", "application/json")
	if d.upKey != "" {
		req.Header.Set("Authorization", "Bearer "+d.upKey)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		writeJSON(w, http.StatusBadGateway,
			d.enrichLoadError("upstream unload failed: "+err.Error()), "unload_failed")
		return
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrBody))
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNoContent {
		code := resp.StatusCode
		if code >= 500 {
			code = http.StatusBadGateway
		}
		writeJSON(w, code, d.enrichLoadError(extractMessage(b)), "unload_failed")
		return
	}
	d.clearLoaded()
	w.Header().Set("Content-Type", "application/json")
	_, _ = io.WriteString(w, `{"status":"unloaded"}`+"\n")
}

// rewriteOOM is the proxy's ModifyResponse: completion failures that
// were OOM gain the lever advice (+ contention evidence); other 5xx gain
// contention evidence; everything else passes through untouched.
func (d *daemon) rewriteOOM(resp *http.Response) error {
	if resp.StatusCode < 400 {
		return nil
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, maxErrBody))
	resp.Body.Close()
	if err != nil {
		return fmt.Errorf("read upstream error body: %w", err)
	}
	msg := extractMessage(b)
	enriched := msg
	if isOOM([]byte(msg)) {
		enriched += "\n" + d.oomAdvice()
		if ce := d.contention(); ce != nil {
			enriched += "; " + ce.Error()
		}
	} else if resp.StatusCode >= 500 {
		if ce := d.contention(); ce != nil {
			enriched += "; " + ce.Error()
		}
	}
	if enriched == msg {
		resp.Body = io.NopCloser(bytes.NewReader(b))
		resp.ContentLength = int64(len(b))
		return nil
	}
	nb, _ := json.Marshal(map[string]any{
		"error": map[string]any{"message": enriched, "type": "server_error"},
	})
	resp.Body = io.NopCloser(bytes.NewReader(nb))
	resp.ContentLength = int64(len(nb))
	resp.Header.Set("Content-Length", strconv.Itoa(len(nb)))
	resp.Header.Set("Content-Type", "application/json")
	return nil
}

// isOOM reports the CUDA allocation failure signature (prefill OOM on a
// small card; max_tokens=1 does not help — allocation precedes generation).
func isOOM(b []byte) bool { return bytes.Contains(b, []byte("CUDA out of memory")) }

// extractMessage digs the human message out of an upstream error body:
// OpenAI {"error":{"message"}}, FastAPI {"detail"}, {"message"}, or raw.
func extractMessage(b []byte) string {
	var m struct {
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
		Detail  json.RawMessage `json:"detail"`
		Message string          `json:"message"`
	}
	if err := json.Unmarshal(b, &m); err == nil {
		if m.Error != nil && m.Error.Message != "" {
			return m.Error.Message
		}
		if len(m.Detail) > 0 {
			var s string
			if json.Unmarshal(m.Detail, &s) == nil && s != "" {
				return s
			}
			return string(m.Detail)
		}
		if m.Message != "" {
			return m.Message
		}
	}
	return strings.TrimSpace(string(b))
}

// writeJSON writes our error envelope: {"error":{"message","type"}}.
func writeJSON(w http.ResponseWriter, code int, msg, typ string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]string{"message": msg, "type": typ},
	})
}

// Load is the run-side client for /-/load: POST, consume the backend
// event stream to completion, surface any error event, return fresh
// status (ctx/cache/fit summary included).
func Load(ctx context.Context, dataDir string, req LoadRequest) (Status, error) {
	st, err := ReadState(dataDir)
	if err != nil {
		return Status{}, err
	}
	body, merr := json.Marshal(req)
	if merr != nil {
		return Status{}, merr
	}
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		"http://"+st.Addr()+loadPath, bytes.NewReader(body))
	if err != nil {
		return Status{}, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	if st.Token != "" {
		hreq.Header.Set("Authorization", "Bearer "+st.Token)
	}
	resp, err := http.DefaultClient.Do(hreq)
	if err != nil {
		return Status{}, err
	}
	defer resp.Body.Close()
	var streamErr error
	if resp.StatusCode == http.StatusOK {
		br := bufio.NewReader(resp.Body)
		for {
			line, rerr := br.ReadString('\n')
			if strings.HasPrefix(strings.TrimSpace(line), "data:") {
				var m struct {
					Error *struct {
						Message string `json:"message"`
					} `json:"error"`
				}
				payload := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(line), "data:"))
				if json.Unmarshal([]byte(payload), &m) == nil && m.Error != nil && m.Error.Message != "" {
					streamErr = fmt.Errorf("%s", m.Error.Message)
				}
			}
			if rerr != nil {
				break
			}
		}
	} else {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrBody))
		streamErr = fmt.Errorf("%s", envelopeMessage(b, resp.StatusCode))
	}
	if streamErr != nil {
		return Status{}, streamErr
	}
	return Query(dataDir)
}

// Unload is the run-side client for /-/unload.
func Unload(ctx context.Context, dataDir string) error {
	st, err := ReadState(dataDir)
	if err != nil {
		return err
	}
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		"http://"+st.Addr()+unloadPath, strings.NewReader("{}"))
	if err != nil {
		return err
	}
	hreq.Header.Set("Content-Type", "application/json")
	if st.Token != "" {
		hreq.Header.Set("Authorization", "Bearer "+st.Token)
	}
	resp, err := http.DefaultClient.Do(hreq)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusNoContent {
		return nil
	}
	b, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrBody))
	return fmt.Errorf("%s", envelopeMessage(b, resp.StatusCode))
}

// envelopeMessage pulls {"error":{"message"}} out of an error body.
func envelopeMessage(b []byte, code int) string {
	var env struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(b, &env) == nil && env.Error.Message != "" {
		return env.Error.Message
	}
	return fmt.Sprintf("HTTP %d: %s", code, strings.TrimSpace(string(b)))
}
