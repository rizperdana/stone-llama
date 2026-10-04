package serve

import (
	"bytes"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/rizperdana/stone-llama/internal/store"
)

// cannedUpstream is the fake backend for the Ollama shim: fixed OpenAI
// replies for /v1/chat/completions and /v1/completions (streaming and
// non-streaming), the upstream Bearer asserted on every call, and the
// last request body per route recorded for payload assertions.
func cannedUpstream(t *testing.T) (*httptest.Server, func(string) []byte) {
	t.Helper()
	var mu sync.Mutex
	got := map[string][]byte{}
	record := func(path string, b []byte) {
		mu.Lock()
		got[path] = b
		mu.Unlock()
	}
	last := func(path string) []byte {
		mu.Lock()
		defer mu.Unlock()
		return got[path]
	}
	check := func(w http.ResponseWriter, r *http.Request) bool {
		if r.Header.Get("Authorization") != "Bearer sekret" {
			t.Errorf("%s: upstream Authorization = %q, want Bearer sekret", r.URL.Path, r.Header.Get("Authorization"))
			w.WriteHeader(http.StatusUnauthorized)
			return false
		}
		return true
	}
	streamOf := func(w io.Writer, body []byte) bool {
		var req struct {
			Stream bool `json:"stream"`
		}
		return json.Unmarshal(body, &req) == nil && req.Stream
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		if !check(w, r) {
			return
		}
		b, _ := io.ReadAll(r.Body)
		record("/v1/chat/completions", b)
		if streamOf(w, b) {
			fl := w.(http.Flusher)
			io.WriteString(w, `data: {"choices":[{"delta":{"role":"assistant","content":"Hel"}}]}`+"\n\n")
			io.WriteString(w, `data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":7}}`+"\n\n")
			io.WriteString(w, "data: [DONE]\n\n")
			fl.Flush()
			return
		}
		io.WriteString(w, `{"id":"c1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Hi there"},"finish_reason":"length"}],"usage":{"prompt_tokens":11,"completion_tokens":7}}`)
	})
	mux.HandleFunc("/v1/completions", func(w http.ResponseWriter, r *http.Request) {
		if !check(w, r) {
			return
		}
		b, _ := io.ReadAll(r.Body)
		record("/v1/completions", b)
		if streamOf(w, b) {
			fl := w.(http.Flusher)
			io.WriteString(w, `data: {"choices":[{"text":"gen"}]}`+"\n\n")
			io.WriteString(w, `data: {"choices":[{"text":" output","finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}`+"\n\n")
			io.WriteString(w, "data: [DONE]\n\n")
			fl.Flush()
			return
		}
		io.WriteString(w, `{"id":"g1","object":"text_completion","choices":[{"index":0,"text":"gen output","finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}`)
	})
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	return ts, last
}

// ollamaDaemon: loopback daemon wired to a fake upstream — the shim's
// whole world is a store dir and a backend URL.
func ollamaDaemon(upstream, models, dataDir string) *daemon {
	return &daemon{
		opts:  Options{Stderr: io.Discard, Host: "127.0.0.1", ModelsDir: models, DataDir: dataDir},
		conn:  Conn{BaseURL: upstream},
		token: "ollama-test-token",
		upKey: "sekret",
	}
}

func ollamaTS(t *testing.T, d *daemon) *httptest.Server {
	t.Helper()
	ts := httptest.NewServer(d.authed(d.mux()))
	t.Cleanup(ts.Close)
	return ts
}

func postOllama(t *testing.T, ts *httptest.Server, path, body string) (int, []byte) {
	t.Helper()
	resp, err := http.Post(ts.URL+path, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return resp.StatusCode, b
}

// jsonKeys decodes body and returns its top-level keys, sorted.
func jsonKeys(t *testing.T, body []byte) ([]string, map[string]any) {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("decode reply %s: %v", body, err)
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys, m
}

func wantKeys(t *testing.T, what string, got, want []string) {
	t.Helper()
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("%s keys = [%s], want [%s]", what, strings.Join(got, ","), strings.Join(want, ","))
	}
}

// TestOllamaChatNonStreamShape: exact ollama reply shape for a canned
// upstream completion — done_reason maps from finish_reason "length",
// counts come from usage, durations are measured-or-zero.
func TestOllamaChatNonStreamShape(t *testing.T) {
	up, _ := cannedUpstream(t)
	ts := ollamaTS(t, ollamaDaemon(up.URL, t.TempDir(), t.TempDir()))

	code, body := postOllama(t, ts, "/api/chat",
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":false}`)
	if code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", code, body)
	}
	keys, got := jsonKeys(t, body)
	wantKeys(t, "chat reply", keys, []string{
		"created_at", "done", "done_reason", "eval_count", "eval_duration",
		"load_duration", "message", "model", "prompt_eval_count",
		"prompt_eval_duration", "total_duration"})
	if got["model"] != "m" || got["done"] != true || got["done_reason"] != "length" {
		t.Errorf("model/done/done_reason = %v/%v/%v, want m/true/length", got["model"], got["done"], got["done_reason"])
	}
	msg, ok := got["message"].(map[string]any)
	if !ok || msg["role"] != "assistant" || msg["content"] != "Hi there" {
		t.Errorf("message = %v, want assistant/Hi there", got["message"])
	}
	if got["prompt_eval_count"] != float64(11) || got["eval_count"] != float64(7) {
		t.Errorf("counts = %v/%v, want 11/7", got["prompt_eval_count"], got["eval_count"])
	}
	if got["total_duration"].(float64) <= 0 {
		t.Errorf("total_duration = %v, want measured > 0", got["total_duration"])
	}
	for _, k := range []string{"load_duration", "prompt_eval_duration", "eval_duration"} {
		if got[k] != float64(0) {
			t.Errorf("%s = %v, want 0 (unmeasurable)", k, got[k])
		}
	}
	if _, err := time.Parse(time.RFC3339Nano, got["created_at"].(string)); err != nil {
		t.Errorf("created_at %q not RFC3339Nano: %v", got["created_at"], err)
	}
}

// TestOllamaChatOmitsMaxTokensThinkSystem: num_predict 0 omits
// max_tokens entirely, think → enable_thinking, top-level system is
// prepended only when the messages carry no system message.
func TestOllamaChatOmitsMaxTokensThinkSystem(t *testing.T) {
	up, last := cannedUpstream(t)
	ts := ollamaTS(t, ollamaDaemon(up.URL, t.TempDir(), t.TempDir()))

	code, body := postOllama(t, ts, "/api/chat", `{
		"model":"m","stream":false,
		"options":{"temperature":0.7,"top_p":0.9,"num_predict":0},
		"think":true,
		"system":"Be terse.",
		"messages":[{"role":"user","content":"hi"}]
	}`)
	if code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", code, body)
	}
	var outbound map[string]any
	if err := json.Unmarshal(last("/v1/chat/completions"), &outbound); err != nil {
		t.Fatalf("decode outbound: %v", err)
	}
	if _, present := outbound["max_tokens"]; present {
		t.Errorf("num_predict 0 must omit max_tokens, got %v", outbound["max_tokens"])
	}
	if outbound["temperature"] != 0.7 || outbound["top_p"] != 0.9 {
		t.Errorf("sampling = %v/%v, want 0.7/0.9", outbound["temperature"], outbound["top_p"])
	}
	if outbound["enable_thinking"] != true {
		t.Errorf("think true → enable_thinking = %v, want true", outbound["enable_thinking"])
	}
	msgs := outbound["messages"].([]any)
	first := msgs[0].(map[string]any)
	if len(msgs) != 2 || first["role"] != "system" || first["content"] != "Be terse." {
		t.Errorf("messages = %v, want top-level system prepended", msgs)
	}

	// an existing system message wins: no prepend, no duplicate
	code, body = postOllama(t, ts, "/api/chat", `{
		"model":"m","stream":false,
		"system":"Be terse.",
		"messages":[{"role":"system","content":"Pirate speak"},{"role":"user","content":"hi"}]
	}`)
	if code != http.StatusOK {
		t.Fatalf("second status = %d, body = %s", code, body)
	}
	var second map[string]any
	if err := json.Unmarshal(last("/v1/chat/completions"), &second); err != nil {
		t.Fatalf("decode outbound: %v", err)
	}
	msgs = second["messages"].([]any)
	first = msgs[0].(map[string]any)
	if len(msgs) != 2 || first["content"] != "Pirate speak" {
		t.Errorf("messages = %v, want original system kept, no prepend", msgs)
	}
	if _, present := second["enable_thinking"]; present {
		t.Errorf("absent think must emit no enable_thinking key")
	}
	if _, present := second["max_tokens"]; present {
		t.Errorf("absent num_predict must omit max_tokens, got %v", second["max_tokens"])
	}
}

// TestOllamaChatThinkEffort: an effort string passes through as
// reasoning_effort (no silent downgrade to a boolean); a malformed
// think value is a clear 400, never dropped.
func TestOllamaChatThinkEffort(t *testing.T) {
	up, last := cannedUpstream(t)
	ts := ollamaTS(t, ollamaDaemon(up.URL, t.TempDir(), t.TempDir()))

	code, body := postOllama(t, ts, "/api/chat",
		`{"model":"m","stream":false,"think":"high","messages":[{"role":"user","content":"hi"}]}`)
	if code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", code, body)
	}
	var outbound map[string]any
	if err := json.Unmarshal(last("/v1/chat/completions"), &outbound); err != nil {
		t.Fatalf("decode outbound: %v", err)
	}
	if outbound["reasoning_effort"] != "high" || outbound["enable_thinking"] != true {
		t.Errorf("think high → reasoning_effort/enable_thinking = %v/%v, want high/true",
			outbound["reasoning_effort"], outbound["enable_thinking"])
	}

	code, body = postOllama(t, ts, "/api/chat",
		`{"model":"m","stream":false,"think":7,"messages":[]}`)
	if code != http.StatusBadRequest || !strings.Contains(string(body), "think") {
		t.Errorf("think 7 = %d %s, want 400 naming think", code, body)
	}
}

// TestOllamaChatStreamChunks: NDJSON delta sequence and the terminating
// done:true object; the upstream [DONE] sentinel must not leak through.
func TestOllamaChatStreamChunks(t *testing.T) {
	up, last := cannedUpstream(t)
	ts := ollamaTS(t, ollamaDaemon(up.URL, t.TempDir(), t.TempDir()))

	// no stream field: ollama's default is true, forwarded as such
	code, body := postOllama(t, ts, "/api/chat",
		`{"model":"m","messages":[{"role":"user","content":"hi"}]}`)
	if code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", code, body)
	}
	var outbound map[string]any
	if err := json.Unmarshal(last("/v1/chat/completions"), &outbound); err != nil {
		t.Fatalf("decode outbound: %v", err)
	}
	if outbound["stream"] != true {
		t.Errorf("absent stream must default true, got %v", outbound["stream"])
	}
	if strings.Contains(string(body), "[DONE]") || strings.Contains(string(body), "data:") {
		t.Errorf("stream body leaked OpenAI framing: %s", body)
	}
	lines := strings.Split(strings.TrimSpace(string(body)), "\n")
	if len(lines) != 3 {
		t.Fatalf("got %d lines, want 3: %s", len(lines), body)
	}
	_, c0 := jsonKeys(t, []byte(lines[0]))
	_, c1 := jsonKeys(t, []byte(lines[1]))
	k2, c2 := jsonKeys(t, []byte(lines[2]))
	if c0["done"] != false || c0["message"].(map[string]any)["content"] != "Hel" {
		t.Errorf("chunk0 = %v, want done:false content Hel", c0)
	}
	if c1["done"] != false || c1["message"].(map[string]any)["content"] != "lo" {
		t.Errorf("chunk1 = %v, want done:false content lo", c1)
	}
	wantKeys(t, "final chunk", k2, []string{
		"created_at", "done", "done_reason", "eval_count", "message", "model",
		"prompt_eval_count"})
	if c2["done"] != true || c2["done_reason"] != "stop" {
		t.Errorf("final done/done_reason = %v/%v, want true/stop", c2["done"], c2["done_reason"])
	}
	if c2["prompt_eval_count"] != float64(11) || c2["eval_count"] != float64(7) {
		t.Errorf("final counts = %v/%v, want 11/7", c2["prompt_eval_count"], c2["eval_count"])
	}
	if c2["message"].(map[string]any)["content"] != "" {
		t.Errorf("final content = %v, want empty", c2["message"])
	}
}

// TestOllamaChatRejectsTools: an unsupported field is a clear JSON
// error, never a silent drop, and never reaches the backend.
func TestOllamaChatRejectsTools(t *testing.T) {
	up, last := cannedUpstream(t)
	ts := ollamaTS(t, ollamaDaemon(up.URL, t.TempDir(), t.TempDir()))

	code, body := postOllama(t, ts, "/api/chat",
		`{"model":"m","stream":false,"messages":[],"tools":[{"type":"function","function":{"name":"f"}}]}`)
	if code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body = %s", code, body)
	}
	keys, got := jsonKeys(t, body)
	wantKeys(t, "error reply", keys, []string{"error"})
	msg := got["error"].(string)
	if !strings.Contains(msg, "tools") || !strings.Contains(msg, "not supported") {
		t.Errorf("error = %q, want it to name tools and not supported", msg)
	}
	if last("/v1/chat/completions") != nil {
		t.Errorf("rejected request must not reach the backend")
	}
}

// ollamaFixture writes one model dir: config.json, a safetensors file,
// and a manifest with a quant but NO manifest-level digest.
func ollamaFixture(t *testing.T) string {
	t.Helper()
	models := t.TempDir()
	dir := filepath.Join(models, "smollm3-exl3")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	cfg := `{"model_type":"smollm3","num_hidden_layers":36,"hidden_size":2048,` +
		`"intermediate_size":11008,"num_attention_heads":16,"num_key_value_heads":4,` +
		`"vocab_size":128256,"max_position_embeddings":65536}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	m := &store.Manifest{
		RepoID:   "org/smollm3-exl3",
		Quant:    "Q8_0",
		PulledAt: time.Now().UTC(),
		Files:    []store.FileEntry{{Path: "model.safetensors", Size: 1, SHA256: "deadbeef"}},
	}
	if err := store.SaveManifest(dir, m); err != nil {
		t.Fatal(err)
	}
	return models
}

// TestOllamaTagsShapeNoDigest: /api/tags shape, honest details only,
// and never a fabricated digest — the manifest has no sha256 at the
// manifest level, only per-file.
func TestOllamaTagsShapeNoDigest(t *testing.T) {
	ts := ollamaTS(t, ollamaDaemon("http://127.0.0.1:1", ollamaFixture(t), t.TempDir()))

	resp, err := http.Get(ts.URL + "/api/tags")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, body = %s", resp.StatusCode, body)
	}
	if bytes.Contains(body, []byte(`"digest"`)) {
		t.Errorf("reply fabricates a digest: %s", body)
	}
	if bytes.Contains(body, []byte(`"parameter_size"`)) {
		t.Errorf("reply fabricates parameter_size: %s", body)
	}
	keys, got := jsonKeys(t, body)
	wantKeys(t, "tags reply", keys, []string{"models"})
	models := got["models"].([]any)
	if len(models) != 1 {
		t.Fatalf("models = %d, want 1", len(models))
	}
	m := models[0].(map[string]any)
	wantKeys(t, "tags entry", sortedKeys(m), []string{
		"details", "model", "modified_at", "name", "size"})
	if m["name"] != "smollm3-exl3" || m["model"] != "smollm3-exl3" {
		t.Errorf("name/model = %v/%v", m["name"], m["model"])
	}
	if m["size"].(float64) <= 0 {
		t.Errorf("size = %v, want > 0", m["size"])
	}
	if _, err := time.Parse(time.RFC3339Nano, m["modified_at"].(string)); err != nil {
		t.Errorf("modified_at %v not RFC3339Nano: %v", m["modified_at"], err)
	}
	details := m["details"].(map[string]any)
	wantKeys(t, "details", sortedKeys(details), []string{
		"family", "format", "quantization_level"})
	if details["family"] != "smollm3" || details["format"] != "safetensors" ||
		details["quantization_level"] != "Q8_0" {
		t.Errorf("details = %v", details)
	}
	if _, present := details["parameter_size"]; present {
		t.Errorf("parameter_size has no honest source and must be omitted: %v", details)
	}
}

// sortedKeys: top-level keys of a decoded object, sorted.
func sortedKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// TestOllamaShowAnswerableFieldsOnly: details + a limited verbatim
// model_info; template only while a template file exists; no modelfile/
// parameters/license fantasy keys; unknown model → 404.
func TestOllamaShowAnswerableFieldsOnly(t *testing.T) {
	models := ollamaFixture(t)
	ts := ollamaTS(t, ollamaDaemon("http://127.0.0.1:1", models, t.TempDir()))

	code, body := postOllama(t, ts, "/api/show", `{"name":"smollm3-exl3"}`)
	if code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", code, body)
	}
	keys, got := jsonKeys(t, body)
	wantKeys(t, "show reply", keys, []string{"details", "model_info"})
	info := got["model_info"].(map[string]any)
	if info["model_type"] != "smollm3" || info["num_hidden_layers"] != float64(36) {
		t.Errorf("model_info = %v, want verbatim config subset", info)
	}
	if _, present := info["parameter_size"]; present {
		t.Errorf("parameter_size must be omitted: %v", info)
	}

	// template appears only once the file genuinely exists
	dir := filepath.Join(models, "smollm3-exl3")
	if err := os.WriteFile(filepath.Join(dir, "template"), []byte("{{ .Prompt }}"), 0o644); err != nil {
		t.Fatal(err)
	}
	code, body = postOllama(t, ts, "/api/show", `{"model":"smollm3-exl3"}`)
	if code != http.StatusOK {
		t.Fatalf("second status = %d, body = %s", code, body)
	}
	keys, got = jsonKeys(t, body)
	wantKeys(t, "show reply with template", keys, []string{"details", "model_info", "template"})
	if got["template"] != "{{ .Prompt }}" {
		t.Errorf("template = %v", got["template"])
	}
	if err := os.Remove(filepath.Join(dir, "template")); err != nil {
		t.Fatal(err)
	}
	noTmpl := mustPost(t, ts, "/api/show", `{"name":"smollm3-exl3"}`)
	if bytes.Contains(noTmpl, []byte(`"template"`)) {
		t.Errorf("template leaked without a template file: %s", noTmpl)
	}

	code, body = postOllama(t, ts, "/api/show", `{"name":"nope"}`)
	if code != http.StatusNotFound || !strings.Contains(string(body), "not found") {
		t.Errorf("unknown model = %d %s, want 404 naming not found", code, body)
	}
}

func mustPost(t *testing.T, ts *httptest.Server, path, body string) []byte {
	t.Helper()
	code, b := postOllama(t, ts, path, body)
	if code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", code, b)
	}
	return b
}

// TestOllamaGenerateResponseShape: response/done shape both modes, the
// verbatim prompt and num_predict → max_tokens upstream, and a clear
// error for unsupported fields (format here).
func TestOllamaGenerateResponseShape(t *testing.T) {
	up, last := cannedUpstream(t)
	ts := ollamaTS(t, ollamaDaemon(up.URL, t.TempDir(), t.TempDir()))

	code, body := postOllama(t, ts, "/api/generate",
		`{"model":"m","prompt":"Say hi","stream":false,"options":{"num_predict":32,"seed":7}}`)
	if code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", code, body)
	}
	keys, got := jsonKeys(t, body)
	wantKeys(t, "generate reply", keys, []string{
		"created_at", "done", "done_reason", "eval_count", "eval_duration",
		"load_duration", "model", "prompt_eval_count", "prompt_eval_duration",
		"response", "total_duration"})
	if got["response"] != "gen output" || got["done"] != true || got["done_reason"] != "stop" {
		t.Errorf("response/done/done_reason = %v/%v/%v", got["response"], got["done"], got["done_reason"])
	}
	if got["prompt_eval_count"] != float64(5) || got["eval_count"] != float64(2) {
		t.Errorf("counts = %v/%v, want 5/2", got["prompt_eval_count"], got["eval_count"])
	}
	var outbound map[string]any
	if err := json.Unmarshal(last("/v1/completions"), &outbound); err != nil {
		t.Fatalf("decode outbound: %v", err)
	}
	if outbound["prompt"] != "Say hi" || outbound["max_tokens"] != float64(32) {
		t.Errorf("prompt/max_tokens = %v/%v, want Say hi/32", outbound["prompt"], outbound["max_tokens"])
	}
	if _, present := outbound["messages"]; present {
		t.Errorf("/v1/completions body must not carry messages: %v", outbound)
	}
	if _, present := outbound["seed"]; present {
		t.Errorf("options.seed must be dropped (pinned backend has no seed field), got %v", outbound["seed"])
	}

	// streaming: NDJSON response deltas, no OpenAI framing
	code, body = postOllama(t, ts, "/api/generate",
		`{"model":"m","prompt":"Say hi","stream":true}`)
	if code != http.StatusOK {
		t.Fatalf("stream status = %d, body = %s", code, body)
	}
	if strings.Contains(string(body), "[DONE]") || strings.Contains(string(body), "data:") {
		t.Errorf("stream body leaked OpenAI framing: %s", body)
	}
	lines := strings.Split(strings.TrimSpace(string(body)), "\n")
	if len(lines) != 3 {
		t.Fatalf("got %d lines, want 3: %s", len(lines), body)
	}
	_, c0 := jsonKeys(t, []byte(lines[0]))
	_, c1 := jsonKeys(t, []byte(lines[1]))
	_, c2 := jsonKeys(t, []byte(lines[2]))
	if c0["response"] != "gen" || c0["done"] != false {
		t.Errorf("chunk0 = %v, want response gen done false", c0)
	}
	if c1["response"] != " output" || c1["done"] != false {
		t.Errorf("chunk1 = %v", c1)
	}
	if c2["done"] != true || c2["done_reason"] != "stop" || c2["response"] != "" {
		t.Errorf("final = %v, want done true, stop, empty response", c2)
	}
	if c2["prompt_eval_count"] != float64(5) || c2["eval_count"] != float64(2) {
		t.Errorf("final counts = %v/%v, want 5/2", c2["prompt_eval_count"], c2["eval_count"])
	}

	// unsupported field: clear error, backend untouched
	code, body = postOllama(t, ts, "/api/generate", `{"model":"m","prompt":"x","format":"json"}`)
	if code != http.StatusBadRequest || !strings.Contains(string(body), "format") ||
		!strings.Contains(string(body), "not supported") {
		t.Errorf("format = %d %s, want 400 naming format", code, body)
	}
}

// TestOllamaKeepAliveZeroUnloadsStateToken: keep_alive:0 is the one
// keep_alive value with an effect — an unload call through our own
// control route carrying the daemon.json state token; any other value
// is accepted and ignored.
func TestOllamaKeepAliveZeroUnloadsStateToken(t *testing.T) {
	up, _ := cannedUpstream(t)
	hits := make(chan string, 4)
	uts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != unloadPath {
			t.Errorf("unload route = %s %s, want POST %s", r.Method, r.URL.Path, unloadPath)
		}
		hits <- r.Header.Get("Authorization")
		io.WriteString(w, `{"status":"unloaded"}`)
	}))
	t.Cleanup(uts.Close)
	host, portStr, err := net.SplitHostPort(strings.TrimPrefix(uts.URL, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatal(err)
	}
	dataDir := t.TempDir()
	if err := WriteState(dataDir, State{PID: os.Getpid(), Host: host, Port: port, Token: "state-token"}); err != nil {
		t.Fatal(err)
	}
	d := ollamaDaemon(up.URL, t.TempDir(), dataDir)
	ts := ollamaTS(t, d)

	code, body := postOllama(t, ts, "/api/chat",
		`{"model":"m","stream":false,"keep_alive":0,"messages":[{"role":"user","content":"hi"}]}`)
	if code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", code, body)
	}
	select {
	case auth := <-hits:
		if auth != "Bearer state-token" {
			t.Errorf("unload Authorization = %q, want Bearer state-token", auth)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("keep_alive 0 did not trigger an unload call")
	}

	// any other keep_alive: accepted, ignored — no unload
	code, body = postOllama(t, ts, "/api/chat",
		`{"model":"m","stream":false,"keep_alive":"5m","messages":[{"role":"user","content":"hi"}]}`)
	if code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", code, body)
	}
	select {
	case auth := <-hits:
		t.Errorf("keep_alive 5m must not unload, got call with %q", auth)
	default:
	}
}
