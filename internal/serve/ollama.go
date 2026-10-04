package serve

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/rizperdana/stone-llama/internal/store"
)

// Ollama compatibility shim: the four /api/* routes ollama-only clients
// expect, translated onto the OpenAI surface the backend serves. This
// file knows only the OpenAI protocol, our own store types, and our own
// control routes — no backend specifics (AGENTS.md seam rule).

// writeOllamaError answers an ollama route with ollama's own error shape
// {"error":"…"} — its clients read that string, not our control envelope.
func writeOllamaError(w http.ResponseWriter, code int, msg string) {
	b, _ := json.Marshal(msg)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	fmt.Fprintf(w, `{"error":%s}`+"\n", b)
}

// rawPresent: an absent or explicit-null JSON field counts as unset.
func rawPresent(raw json.RawMessage) bool {
	return len(raw) > 0 && !bytes.Equal(bytes.TrimSpace(raw), []byte("null"))
}

// optr boxes v for optional-field encoding.
func optr[T any](v T) *T { return &v }

// ollamaMessage is both the inbound ollama message and the outbound
// OpenAI message: role+content is the shared core; any other ollama
// message field is dropped (unknown fields are dropped by design).
type ollamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ollamaOptions is the ollama options block we can honestly map onto
// OpenAI sampling fields.
type ollamaOptions struct {
	Temperature *float64 `json:"temperature"`
	TopP        *float64 `json:"top_p"`
	NumPredict  int      `json:"num_predict"`
	// Seed is decoded but NEVER forwarded, on either route: neither
	// pinned request type has a seed field (TabbyAPI BaseSamplerRequest,
	// common/sampling.py @ f07131c — verified by grep, no seed anywhere
	// in the request types). Sending it would be a silently ignored
	// extra; dropping it here is documented divergence, not oversight.
	Seed *int `json:"seed"`
}

type ollamaChatRequest struct {
	Model     string          `json:"model"`
	Messages  []ollamaMessage `json:"messages"`
	Stream    *bool           `json:"stream"` // ollama default: true
	Options   ollamaOptions   `json:"options"`
	KeepAlive json.RawMessage `json:"keep_alive"`
	System    string          `json:"system"`
	Think     json.RawMessage `json:"think"`
	// rejected up front, never silently ignored:
	Tools    json.RawMessage `json:"tools"`
	Format   json.RawMessage `json:"format"`
	Template json.RawMessage `json:"template"`
	Raw      json.RawMessage `json:"raw"`
	Context  json.RawMessage `json:"context"`
}

// ollamaChatBody is the outbound OpenAI body: model, messages, stream,
// max_tokens, temperature, top_p, enable_thinking, plus
// reasoning_effort for ollama's think effort strings — both reasoning
// fields exist on the pinned ChatCompletionRequest (f07131c).
type ollamaChatBody struct {
	Model           string          `json:"model"`
	Messages        []ollamaMessage `json:"messages"`
	Stream          bool            `json:"stream"`
	MaxTokens       *int            `json:"max_tokens,omitempty"`
	Temperature     *float64        `json:"temperature,omitempty"`
	TopP            *float64        `json:"top_p,omitempty"`
	EnableThinking  *bool           `json:"enable_thinking,omitempty"`
	ReasoningEffort string          `json:"reasoning_effort,omitempty"`
}

// openaiUsage is the token-count block both OpenAI routes share.
type openaiUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

// openaiChunk consumes both OpenAI reply shapes: chat (message) and
// completions (text) sit side by side, stream deltas arrive in Delta.
type openaiChunk struct {
	Choices []struct {
		Message      ollamaMessage `json:"message"`
		Text         string        `json:"text"`
		Delta        ollamaMessage `json:"delta"`
		FinishReason string        `json:"finish_reason"`
	} `json:"choices"`
	Usage *openaiUsage `json:"usage"`
}

// ollamaChatResponse is the non-streaming /api/chat answer. Counts are
// filled only when the upstream sent usage; durations are only ever
// values we measured (or 0 — never invented).
type ollamaChatResponse struct {
	Model              string        `json:"model"`
	CreatedAt          string        `json:"created_at"`
	Message            ollamaMessage `json:"message"`
	Done               bool          `json:"done"`
	DoneReason         string        `json:"done_reason"`
	PromptEvalCount    *int          `json:"prompt_eval_count,omitempty"`
	EvalCount          *int          `json:"eval_count,omitempty"`
	TotalDuration      int64         `json:"total_duration"`
	LoadDuration       int64         `json:"load_duration"`
	PromptEvalDuration int64         `json:"prompt_eval_duration"`
	EvalDuration       int64         `json:"eval_duration"`
}

// ollamaChatChunk is one streaming /api/chat object; the terminating
// object flips Done and carries done_reason + counts (no durations —
// the shim measures only total request time, see the non-stream shape).
type ollamaChatChunk struct {
	Model           string        `json:"model"`
	CreatedAt       string        `json:"created_at"`
	Message         ollamaMessage `json:"message"`
	Done            bool          `json:"done"`
	DoneReason      string        `json:"done_reason,omitempty"`
	PromptEvalCount *int          `json:"prompt_eval_count,omitempty"`
	EvalCount       *int          `json:"eval_count,omitempty"`
}

// handleOllamaChat: POST /api/chat — ollama chat onto OpenAI chat.
func (d *daemon) handleOllamaChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeOllamaError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	var req ollamaChatRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		writeOllamaError(w, http.StatusBadRequest, "decode /api/chat body: "+err.Error())
		return
	}
	if req.Model == "" {
		writeOllamaError(w, http.StatusBadRequest, "model is required")
		return
	}
	for _, u := range []struct {
		name string
		raw  json.RawMessage
	}{{"tools", req.Tools}, {"format", req.Format}, {"template", req.Template},
		{"raw", req.Raw}, {"context", req.Context}} {
		if rawPresent(u.raw) {
			writeOllamaError(w, http.StatusBadRequest, fmt.Sprintf(
				"field %q is not supported by stone-llama's Ollama shim — it is rejected here, never silently ignored",
				u.name))
			return
		}
	}

	msgs := req.Messages
	if req.System != "" && !hasSystemMessage(msgs) {
		msgs = append([]ollamaMessage{{Role: "system", Content: req.System}}, msgs...)
	}
	stream := req.Stream == nil || *req.Stream
	body := ollamaChatBody{Model: req.Model, Messages: msgs, Stream: stream}
	if req.Options.NumPredict > 0 { // absent/0 → key omitted entirely
		body.MaxTokens = optr(req.Options.NumPredict)
	}
	body.Temperature = req.Options.Temperature
	body.TopP = req.Options.TopP
	think, effort, err := ollamaThink(req.Think)
	if err != nil {
		writeOllamaError(w, http.StatusBadRequest, "invalid \"think\": "+err.Error())
		return
	}
	// enable_thinking/reasoning_effort: field types verified on the
	// pinned ChatCompletionRequest (f07131c); live effect UNVERIFIED
	// pending a live-backend test (TabbyAPI 5002 down, left down).
	body.EnableThinking = think
	body.ReasoningEffort = effort

	payload, err := json.Marshal(body)
	if err != nil {
		writeOllamaError(w, http.StatusInternalServerError, err.Error())
		return
	}
	start := time.Now()
	resp, handled := d.ollamaUpstream(w, r, "/v1/chat/completions", payload, "chat")
	if handled {
		return
	}
	defer resp.Body.Close()
	created := time.Now().UTC().Format(time.RFC3339Nano)

	if !stream {
		var cr openaiChunk
		if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil {
			writeOllamaError(w, http.StatusBadGateway, "decode upstream chat reply: "+err.Error())
			return
		}
		d.ollamaKeepAliveUnload(r.Context(), req.KeepAlive)
		content, reason := "", ""
		if len(cr.Choices) > 0 {
			content, reason = cr.Choices[0].Message.Content, cr.Choices[0].FinishReason
		}
		out := ollamaChatResponse{
			Model:         req.Model,
			CreatedAt:     created,
			Message:       ollamaMessage{Role: "assistant", Content: content},
			Done:          true,
			DoneReason:    ollamaDoneReason(reason),
			TotalDuration: time.Since(start).Nanoseconds(),
			// load/prompt_eval/eval durations are not observable
			// through the OpenAI surface — sent as 0, never invented.
		}
		if cr.Usage != nil {
			out.PromptEvalCount = optr(cr.Usage.PromptTokens)
			out.EvalCount = optr(cr.Usage.CompletionTokens)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(out)
		return
	}

	// streaming: newline-delimited JSON per delta, terminating
	// done:true object. The upstream [DONE] sentinel is consumed, never
	// forwarded — this route is not SSE.
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-cache")
	fl, _ := w.(http.Flusher)
	flush := func() {
		if fl != nil {
			fl.Flush()
		}
	}
	emit := func(content string) bool {
		b, _ := json.Marshal(ollamaChatChunk{
			Model: req.Model, CreatedAt: created,
			Message: ollamaMessage{Role: "assistant", Content: content},
		})
		if _, err := w.Write(append(b, '\n')); err != nil {
			return false // client went away
		}
		flush()
		return true
	}
	scanOpenAIStream(resp, emit, func(reason string, usage *openaiUsage) {
		final := ollamaChatChunk{
			Model: req.Model, CreatedAt: created,
			Message:    ollamaMessage{Role: "assistant"},
			Done:       true,
			DoneReason: ollamaDoneReason(reason),
		}
		if usage != nil {
			final.PromptEvalCount = optr(usage.PromptTokens)
			final.EvalCount = optr(usage.CompletionTokens)
		}
		b, _ := json.Marshal(final)
		_, _ = w.Write(append(b, '\n'))
		flush()
	})
	d.ollamaKeepAliveUnload(r.Context(), req.KeepAlive)
}

// ollamaThink maps ollama's think field onto the backend's pinned
// reasoning fields (ChatCompletionRequest @ f07131c): bool →
// enable_thinking; effort string ("low"|"medium"|"high") →
// enable_thinking true PLUS reasoning_effort verbatim (the backend
// forwards it to the chat template; accepted values are the model's);
// absent/null → no keys. Anything else is an error — a client's
// request is never silently downgraded or dropped.
func ollamaThink(raw json.RawMessage) (*bool, string, error) {
	if !rawPresent(raw) {
		return nil, "", nil
	}
	s := strings.TrimSpace(string(raw))
	switch s {
	case "true":
		return optr(true), "", nil
	case "false":
		return optr(false), "", nil
	}
	if strings.HasPrefix(s, `"`) {
		var effort string
		if err := json.Unmarshal(raw, &effort); err != nil {
			return nil, "", fmt.Errorf("malformed string: %s", s)
		}
		return optr(true), effort, nil
	}
	return nil, "", fmt.Errorf("must be true, false, or an effort string like \"low\"/\"medium\"/\"high\" (got %s)", s)
}

// ollamaDoneReason maps an OpenAI finish_reason onto ollama's
// vocabulary. Only "length" has a direct ollama counterpart we can
// honestly claim; everything else (stop, empty, any flavour this shim
// never allows through — tools are rejected up front) reports "stop".
func ollamaDoneReason(finishReason string) string {
	if finishReason == "length" {
		return "length"
	}
	return "stop"
}

// hasSystemMessage reports whether the conversation already opens with
// a system message (top-level system is prepended only when absent).
func hasSystemMessage(msgs []ollamaMessage) bool {
	for _, m := range msgs {
		if m.Role == "system" {
			return true
		}
	}
	return false
}

// ollamaUpstream POSTs payload to path on the backend with the upstream
// key — the handleLoad outbound pattern. Upstream refusals become ollama
// error JSON on w and report handled=true; a non-2xx answer is never
// parsed as content.
func (d *daemon) ollamaUpstream(w http.ResponseWriter, r *http.Request, path string, payload []byte, what string) (*http.Response, bool) {
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost,
		d.conn.BaseURL+path, bytes.NewReader(payload))
	if err != nil {
		writeOllamaError(w, http.StatusInternalServerError, err.Error())
		return nil, true
	}
	req.Header.Set("Content-Type", "application/json")
	if d.upKey != "" {
		req.Header.Set("Authorization", "Bearer "+d.upKey)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		writeOllamaError(w, http.StatusBadGateway, what+" failed: "+err.Error())
		return nil, true
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		b, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrBody))
		code := resp.StatusCode
		if code >= 500 {
			code = http.StatusBadGateway
		}
		writeOllamaError(w, code, extractMessage(b))
		return nil, true
	}
	return resp, false
}

// scanOpenAIStream reads one upstream OpenAI SSE body: emit runs per
// content delta (false = client gone, stop), finish runs exactly once
// at the end with the accumulated finish reason and usage. The
// [DONE] sentinel is consumed here, never forwarded.
func scanOpenAIStream(resp *http.Response, emit func(content string) bool, finish func(finishReason string, usage *openaiUsage)) {
	var reason string
	var usage *openaiUsage
	br := bufio.NewReader(resp.Body)
	for {
		line, err := br.ReadString('\n')
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "data:") {
			payload := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
			if payload == "[DONE]" {
				break
			}
			var c openaiChunk
			if json.Unmarshal([]byte(payload), &c) == nil {
				if c.Usage != nil {
					usage = c.Usage
				}
				if len(c.Choices) > 0 {
					if c.Choices[0].FinishReason != "" {
						reason = c.Choices[0].FinishReason
					}
					// chat streams deltas in delta.content,
					// completions in text — both are content.
					txt := c.Choices[0].Delta.Content
					if txt == "" {
						txt = c.Choices[0].Text
					}
					if txt != "" && !emit(txt) {
						return
					}
				}
			}
		}
		if err != nil { // EOF without [DONE]: finish with what arrived
			break
		}
	}
	finish(reason, usage)
}

// ollamaKeepAliveUnload honours the one keep_alive value that means
// something here: 0 unloads. It POSTs our own /-/unload — the one place
// the shim touches a control route — with the state token read from
// daemon.json, because that route always demands it. Any other
// keep_alive value is accepted and ignored: this daemon is resident by
// design. Failures are logged, never invented into the response.
func (d *daemon) ollamaKeepAliveUnload(ctx context.Context, raw json.RawMessage) {
	if d.opts.DataDir == "" {
		return
	}
	ka := strings.Trim(strings.TrimSpace(string(raw)), `"`)
	if ka != "0" && ka != "0s" {
		return
	}
	logf := func(format string, args ...any) {
		fmt.Fprintf(d.opts.Stderr, "stone-llama: keep_alive 0: "+format+"\n", args...)
	}
	st, err := ReadState(d.opts.DataDir)
	if err != nil {
		logf("unload skipped: %v", err)
		return
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		"http://"+st.Addr()+unloadPath, strings.NewReader("{}"))
	if err != nil {
		logf("unload skipped: %v", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+st.Token)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		logf("unload failed: %v", err)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNoContent {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrBody))
		logf("unload failed: HTTP %d: %s", resp.StatusCode, extractMessage(b))
	}
}

// handleOllamaTags: GET /api/tags — the model store in ollama's shape.
// Fields without an honest source are omitted, never faked: no digest
// (our manifests carry per-file sha256, not a manifest digest), no
// parameter_size (no honest source in config.json/manifest).
func (d *daemon) handleOllamaTags(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeOllamaError(w, http.StatusMethodNotAllowed, "use GET")
		return
	}
	models, _, err := store.Scan(d.opts.ModelsDir)
	if err != nil {
		writeOllamaError(w, http.StatusInternalServerError, "scan models: "+err.Error())
		return
	}
	type tag struct {
		Model      string         `json:"model"`
		Name       string         `json:"name"`
		Size       int64          `json:"size"`
		ModifiedAt string         `json:"modified_at,omitempty"`
		Details    map[string]any `json:"details"`
	}
	out := struct {
		Models []tag `json:"models"`
	}{Models: []tag{}}
	for _, m := range models {
		t := tag{Model: m.Name, Name: m.Name, Size: m.SizeBytes, Details: ollamaDetails(m)}
		if fi, err := os.Stat(m.Path); err == nil {
			t.ModifiedAt = fi.ModTime().UTC().Format(time.RFC3339Nano)
		}
		out.Models = append(out.Models, t)
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

// ollamaDetails builds ollama's details block from what the model
// directory actually carries: family = config.json's model_type,
// quantization_level = manifest quant, format = the weights format
// visibly on disk. Anything else (parameter_size in particular) stays
// OUT — an uncomputable field is omitted, never faked.
func ollamaDetails(m store.Model) map[string]any {
	details := map[string]any{}
	if cfg, err := readModelConfig(m.Path); err == nil {
		if mt, ok := cfg["model_type"].(string); ok && mt != "" {
			details["family"] = mt
		}
	}
	if hasSafetensors(m.Path) {
		details["format"] = "safetensors"
	}
	if m.Manifest != nil && m.Manifest.Quant != "" && m.Manifest.Quant != "-" {
		details["quantization_level"] = m.Manifest.Quant
	}
	return details
}

// readModelConfig parses dir/config.json as generic JSON.
// Missing → (nil, nil); unreadable/corrupt → error (loud beats silent).
func readModelConfig(dir string) (map[string]any, error) {
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("config.json: %w", err)
	}
	return m, nil
}

// hasSafetensors reports whether the model dir visibly stores
// safetensors weights (top level only — the format ollama's details
// format field can honestly claim here).
func hasSafetensors(dir string) bool {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".safetensors") {
			return true
		}
	}
	return false
}

// showInfoKeys: the config.json keys /api/show is willing to echo as
// model_info — verbatim HF key names and values, no invented ollama
// dotted-key translation.
var showInfoKeys = []string{
	"model_type", "architectures", "num_hidden_layers", "hidden_size",
	"intermediate_size", "num_attention_heads", "num_key_value_heads",
	"vocab_size", "max_position_embeddings", "torch_dtype",
}

type ollamaShowResponse struct {
	Details   map[string]any `json:"details"`
	ModelInfo map[string]any `json:"model_info,omitempty"`
	Template  string         `json:"template,omitempty"`
}

// handleOllamaShow: POST /api/show — the answerable subset only:
// details, a limited verbatim model_info, and template only when a
// template file genuinely exists on disk. Everything else (modelfile,
// parameters, license, info, capabilities…) is omitted.
func (d *daemon) handleOllamaShow(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeOllamaError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	var req struct {
		Name  string `json:"name"`
		Model string `json:"model"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		writeOllamaError(w, http.StatusBadRequest, "decode /api/show body: "+err.Error())
		return
	}
	name := req.Name
	if name == "" {
		name = req.Model
	}
	if name == "" {
		writeOllamaError(w, http.StatusBadRequest, "model name is required")
		return
	}
	models, _, err := store.Scan(d.opts.ModelsDir)
	if err != nil {
		writeOllamaError(w, http.StatusInternalServerError, "scan models: "+err.Error())
		return
	}
	var found *store.Model
	for i := range models {
		if models[i].Name == name {
			found = &models[i]
			break
		}
	}
	if found == nil {
		writeOllamaError(w, http.StatusNotFound, fmt.Sprintf("model %q not found", name))
		return
	}
	cfg, err := readModelConfig(found.Path)
	if err != nil {
		writeOllamaError(w, http.StatusInternalServerError, err.Error())
		return
	}
	out := ollamaShowResponse{Details: ollamaDetails(*found)}
	if cfg != nil {
		info := map[string]any{}
		for _, k := range showInfoKeys {
			if v, ok := cfg[k]; ok {
				info[k] = v
			}
		}
		if len(info) > 0 {
			out.ModelInfo = info
		}
	}
	if f, err := os.Open(filepath.Join(found.Path, "template")); err == nil {
		b, _ := io.ReadAll(io.LimitReader(f, 1<<20))
		f.Close()
		out.Template = string(b)
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

type ollamaGenerateRequest struct {
	Model     string          `json:"model"`
	Prompt    string          `json:"prompt"`
	Stream    *bool           `json:"stream"` // ollama default: true
	Options   ollamaOptions   `json:"options"`
	KeepAlive json.RawMessage `json:"keep_alive"`
	Raw       json.RawMessage `json:"raw"` // accepted: /v1/completions is verbatim by design
	// rejected up front, never silently ignored:
	Format   json.RawMessage `json:"format"`
	Template json.RawMessage `json:"template"`
	Context  json.RawMessage `json:"context"`
	Suffix   json.RawMessage `json:"suffix"`
	System   json.RawMessage `json:"system"`
}

// ollamaCompletionsBody is the outbound OpenAI /v1/completions body.
type ollamaCompletionsBody struct {
	Model       string   `json:"model"`
	Prompt      string   `json:"prompt"`
	Stream      bool     `json:"stream"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
}

type ollamaGenerateResponse struct {
	Model              string `json:"model"`
	CreatedAt          string `json:"created_at"`
	Response           string `json:"response"`
	Done               bool   `json:"done"`
	DoneReason         string `json:"done_reason,omitempty"`
	PromptEvalCount    *int   `json:"prompt_eval_count,omitempty"`
	EvalCount          *int   `json:"eval_count,omitempty"`
	TotalDuration      int64  `json:"total_duration"`
	LoadDuration       int64  `json:"load_duration"`
	PromptEvalDuration int64  `json:"prompt_eval_duration"`
	EvalDuration       int64  `json:"eval_duration"`
}

type ollamaGenerateChunk struct {
	Model           string `json:"model"`
	CreatedAt       string `json:"created_at"`
	Response        string `json:"response"`
	Done            bool   `json:"done"`
	DoneReason      string `json:"done_reason,omitempty"`
	PromptEvalCount *int   `json:"prompt_eval_count,omitempty"`
	EvalCount       *int   `json:"eval_count,omitempty"`
}

// handleOllamaGenerate: POST /api/generate — ollama generate onto the
// backend's /v1/completions (verbatim prompt surface). No fallback to
// chat: if that route is unavailable the upstream refusal is reported
// as-is.
func (d *daemon) handleOllamaGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeOllamaError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	var req ollamaGenerateRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		writeOllamaError(w, http.StatusBadRequest, "decode /api/generate body: "+err.Error())
		return
	}
	if req.Model == "" {
		writeOllamaError(w, http.StatusBadRequest, "model is required")
		return
	}
	for _, u := range []struct {
		name string
		raw  json.RawMessage
	}{{"format", req.Format}, {"template", req.Template}, {"context", req.Context},
		{"suffix", req.Suffix}, {"system", req.System}} {
		if rawPresent(u.raw) {
			writeOllamaError(w, http.StatusBadRequest, fmt.Sprintf(
				"field %q is not supported by stone-llama's Ollama shim — it is rejected here, never silently ignored",
				u.name))
			return
		}
	}

	stream := req.Stream == nil || *req.Stream
	body := ollamaCompletionsBody{Model: req.Model, Prompt: req.Prompt, Stream: stream}
	if req.Options.NumPredict > 0 { // absent/0 → key omitted entirely
		body.MaxTokens = optr(req.Options.NumPredict)
	}
	body.Temperature = req.Options.Temperature
	body.TopP = req.Options.TopP
	// options.seed deliberately NOT forwarded — see ollamaOptions.Seed:
	// the pinned backend request types have no seed field.

	payload, err := json.Marshal(body)
	if err != nil {
		writeOllamaError(w, http.StatusInternalServerError, err.Error())
		return
	}
	start := time.Now()
	// UNVERIFIED pending a live-backend test: the route and
	// CompletionRequest exist in the pinned tree (OAI/types/
	// completion.py @ f07131c) but were never exercised live —
	// TabbyAPI 5002 is down and was left down.
	resp, handled := d.ollamaUpstream(w, r, "/v1/completions", payload, "completions")
	if handled {
		return
	}
	defer resp.Body.Close()
	created := time.Now().UTC().Format(time.RFC3339Nano)

	if !stream {
		var cr openaiChunk
		if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil {
			writeOllamaError(w, http.StatusBadGateway, "decode upstream completions reply: "+err.Error())
			return
		}
		d.ollamaKeepAliveUnload(r.Context(), req.KeepAlive)
		text, reason := "", ""
		if len(cr.Choices) > 0 {
			text, reason = cr.Choices[0].Text, cr.Choices[0].FinishReason
		}
		out := ollamaGenerateResponse{
			Model:         req.Model,
			CreatedAt:     created,
			Response:      text,
			Done:          true,
			DoneReason:    ollamaDoneReason(reason),
			TotalDuration: time.Since(start).Nanoseconds(),
			// load/prompt_eval/eval durations are not observable
			// through the OpenAI surface — sent as 0, never invented.
		}
		if cr.Usage != nil {
			out.PromptEvalCount = optr(cr.Usage.PromptTokens)
			out.EvalCount = optr(cr.Usage.CompletionTokens)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(out)
		return
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-cache")
	fl, _ := w.(http.Flusher)
	flush := func() {
		if fl != nil {
			fl.Flush()
		}
	}
	emit := func(delta string) bool {
		b, _ := json.Marshal(ollamaGenerateChunk{
			Model: req.Model, CreatedAt: created, Response: delta,
		})
		if _, err := w.Write(append(b, '\n')); err != nil {
			return false // client went away
		}
		flush()
		return true
	}
	scanOpenAIStream(resp, emit, func(reason string, usage *openaiUsage) {
		final := ollamaGenerateChunk{
			Model: req.Model, CreatedAt: created,
			Done: true, DoneReason: ollamaDoneReason(reason),
		}
		if usage != nil {
			final.PromptEvalCount = optr(usage.PromptTokens)
			final.EvalCount = optr(usage.CompletionTokens)
		}
		b, _ := json.Marshal(final)
		_, _ = w.Write(append(b, '\n'))
		flush()
	})
	d.ollamaKeepAliveUnload(r.Context(), req.KeepAlive)
}
