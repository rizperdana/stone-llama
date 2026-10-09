package serve

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestBackendFor(t *testing.T) {
	for _, tc := range []struct {
		name string
		want string
	}{
		{"", "tabbyBackend"},
		{BackendTabby, "tabbyBackend"},
		{BackendLlama, "llamaBackend"},
	} {
		be, err := BackendFor(tc.name, 4096)
		if err != nil {
			t.Fatalf("BackendFor(%q): %v", tc.name, err)
		}
		got := fmt.Sprintf("%T", be)
		if got != "*serve."+tc.want && got != "serve."+tc.want {
			t.Errorf("BackendFor(%q) = %s, want %s", tc.name, got, tc.want)
		}
	}
	if _, err := BackendFor("cuda", 0); err == nil || !strings.Contains(err.Error(), "unknown backend") {
		t.Errorf("BackendFor(cuda) err = %v, want unknown backend", err)
	}
	// budget rides through to the llama implementation only
	be, err := BackendFor(BackendLlama, 7)
	if err != nil {
		t.Fatal(err)
	}
	if lb, ok := be.(llamaBackend); !ok || lb.deviceBudgetMiB != 7 {
		t.Errorf("llama backend = %#v, want deviceBudgetMiB=7", be)
	}
}

func TestResolveGGUF(t *testing.T) {
	dir := t.TempDir()
	write := func(name string) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte("x"), 0o600); err != nil {
			t.Fatal(err)
		}
		return p
	}
	mg := write("m.gguf")

	// explicit .gguf path
	if got, err := resolveGGUF(mg, ""); err != nil || got != mg {
		t.Errorf("explicit = %q, %v; want %q", got, err, mg)
	}
	// non-gguf file → refused, names what is supported
	bin := write("model.bin")
	if _, err := resolveGGUF(bin, ""); err == nil ||
		!strings.Contains(err.Error(), ".gguf") || !strings.Contains(err.Error(), "tabby") {
		t.Errorf("bin file err = %v, want GGUF-only guidance naming tabby", err)
	}
	// missing → refused
	if _, err := resolveGGUF(filepath.Join(dir, "nope.gguf"), ""); err == nil ||
		!strings.Contains(err.Error(), "not found") {
		t.Errorf("missing err = %v, want not found", err)
	}
	// dir with exactly one *.gguf
	sub := filepath.Join(dir, "modeldir")
	if err := os.Mkdir(sub, 0o700); err != nil {
		t.Fatal(err)
	}
	inner := filepath.Join(sub, "weights.gguf")
	if err := os.WriteFile(inner, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got, err := resolveGGUF(sub, ""); err != nil || got != inner {
		t.Errorf("dir single = %q, %v; want %q", got, err, inner)
	}
	// dir with two *.gguf → refused, lists both names
	if err := os.WriteFile(filepath.Join(sub, "other.gguf"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := resolveGGUF(sub, ""); err == nil ||
		!strings.Contains(err.Error(), "weights.gguf") || !strings.Contains(err.Error(), "other.gguf") {
		t.Errorf("two-gguf err = %v, want both names listed", err)
	}
	// dir with no *.gguf (EXL3-style store) → refused
	empty := filepath.Join(dir, "exl3dir")
	if err := os.Mkdir(empty, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(empty, "model.safetensors"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := resolveGGUF(empty, ""); err == nil || !strings.Contains(err.Error(), "no *.gguf") {
		t.Errorf("exl3 dir err = %v, want no *.gguf", err)
	}
	// bare name resolved against ModelsDir
	store := t.TempDir()
	storeGGUF := filepath.Join(store, "bare.gguf")
	if err := os.WriteFile(storeGGUF, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got, err := resolveGGUF("bare.gguf", store); err != nil || got != storeGGUF {
		t.Errorf("modelsdir lookup = %q, %v; want %q", got, err, storeGGUF)
	}
}

// TestLlamaPrepareRefusesNonGGUF is the GGUF routing contract: on the
// llama backend a non-GGUF model is a clear error — no Conn, no spawn,
// and no silent fallback to TabbyAPI. Runs before any llama-server
// binary check, so it works on machines without llama.cpp.
func TestLlamaPrepareRefusesNonGGUF(t *testing.T) {
	exl3 := t.TempDir()
	if err := os.WriteFile(filepath.Join(exl3, "model.safetensors"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	b := llamaBackend{}
	conn, cfg, err := b.Prepare(&Options{Model: exl3, RuntimeDir: t.TempDir()}, 59999)
	if err == nil {
		t.Fatal("non-GGUF on backend llama must fail, got success (silent fallback)")
	}
	for _, want := range []string{"GGUF", "tabby", "safetensorsdir"} {
		_ = want
	}
	if !strings.Contains(err.Error(), "GGUF") || !strings.Contains(err.Error(), "tabby") {
		t.Errorf("err = %v, must name GGUF support and the tabby backend", err)
	}
	if conn.BaseURL != "" || cfg != "" {
		t.Errorf("refusal leaked conn/cfg: %+v %q", conn, cfg)
	}
}

func TestLlamaPrepareGGUFAndKey(t *testing.T) {
	if _, err := exec.LookPath("llama-server"); err != nil {
		t.Skipf("llama-server not on PATH: %v", err)
	}
	dir := t.TempDir()
	gguf := filepath.Join(dir, "m.gguf")
	if err := os.WriteFile(gguf, []byte("gguf"), 0o600); err != nil {
		t.Fatal(err)
	}
	rt := t.TempDir()
	conn, cfg, err := (llamaBackend{}).Prepare(&Options{Model: gguf, RuntimeDir: rt}, 41234)
	if err != nil {
		t.Fatal(err)
	}
	if cfg != gguf {
		t.Errorf("cfgPath = %q, want %q", cfg, gguf)
	}
	if conn.BaseURL != "http://127.0.0.1:41234" {
		t.Errorf("BaseURL = %q", conn.BaseURL)
	}
	key := loadUpstreamKey(conn.UpKeyFile)
	if len(key) != 64 {
		t.Errorf("upstream key = %q (len %d), want 64 hex chars", key, len(key))
	}
	// proxy contract: same key from the file it injects from
	if loadUpstreamKey(llamaKeyFile(rt)) != key {
		t.Error("key file mismatch")
	}
}

func TestLlamaBudgetGate(t *testing.T) {
	dir := t.TempDir()
	gguf := filepath.Join(dir, "big.gguf")
	if err := os.WriteFile(gguf, make([]byte, 3<<20), 0o600); err != nil { // 3 MiB
		t.Fatal(err)
	}
	opts := &Options{Model: gguf, RuntimeDir: t.TempDir()}
	_, _, err := (llamaBackend{deviceBudgetMiB: 2}).Prepare(opts, 41235)
	if err == nil || !strings.Contains(err.Error(), "backend_device_budget_mib") {
		t.Fatalf("budget over err = %v, want backend_device_budget_mib refusal", err)
	}
	if !strings.Contains(err.Error(), "KV cache") {
		t.Errorf("refusal must say what is excluded: %v", err)
	}
	// budget 0 = no check (same path as TestLlamaPrepareGGUFAndKey)
}

// TestTabbyBackendPrepareSpawn pins tabbyBackend to the pre-seam
// behaviour: Prepare succeeds on a bare runtime dir (marker no-op,
// tokens + config written), Spawn refuses without the venv.
func TestTabbyBackendPrepareSpawn(t *testing.T) {
	rt := t.TempDir()
	conn, cfg, err := (tabbyBackend{}).Prepare(&Options{RuntimeDir: rt, ModelsDir: "m"}, 41236)
	if err != nil {
		t.Fatal(err)
	}
	if cfg != tabbyCfgPath(rt) {
		t.Errorf("cfgPath = %q, want %q", cfg, tabbyCfgPath(rt))
	}
	if conn.Name == "" || conn.UpKeyFile == "" || !strings.Contains(conn.BaseURL, "127.0.0.1:41236") {
		t.Errorf("conn = %+v", conn)
	}
	if loadUpstreamKey(conn.UpKeyFile) == "" {
		t.Error("upstream key missing")
	}
	for _, p := range []string{
		cfg,
		filepath.Join(rt, "tabbyAPI", "api_tokens.yml"),
		conn.UpKeyFile,
	} {
		if _, err := os.Stat(p); err != nil {
			t.Errorf("expected file %s: %v", p, err)
		}
	}
	if _, err := (tabbyBackend{}).Spawn(rt, cfg, filepath.Join(t.TempDir(), "x.log"), 41236, nil); err == nil ||
		!strings.Contains(err.Error(), "stone-llama setup") {
		t.Errorf("spawn without venv err = %v, want setup refusal", err)
	}
}

// TestLlamaBackendE2E spawns the REAL llama-server (spawn → /v1/models
// healthy → auth'd model listing → clean stop). Off by default so
// gates stay fast and llama.cpp stays optional:
//
//	STONE_LLAMA_E2E=1 STONE_LLAMA_E2E_GGUF=/path/to/m.gguf go test ./internal/serve -run LlamaBackendE2E
func TestLlamaBackendE2E(t *testing.T) {
	if os.Getenv("STONE_LLAMA_E2E") == "" {
		t.Skip("set STONE_LLAMA_E2E=1 (plus STONE_LLAMA_E2E_GGUF) to run")
	}
	gguf := os.Getenv("STONE_LLAMA_E2E_GGUF")
	if gguf == "" {
		t.Skip("STONE_LLAMA_E2E_GGUF not set")
	}
	if _, err := exec.LookPath("llama-server"); err != nil {
		t.Skipf("llama-server not on PATH: %v", err)
	}
	port, err := freePort() // ephemeral loopback port
	if err != nil {
		t.Fatal(err)
	}
	rt := t.TempDir()
	conn, cfg, err := (llamaBackend{}).Prepare(&Options{Model: gguf, RuntimeDir: rt}, port)
	if err != nil {
		t.Fatal(err)
	}
	logPath := filepath.Join(t.TempDir(), "llama.log")
	c, err := (llamaBackend{}).Spawn(rt, cfg, logPath, port, nil)
	if err != nil {
		t.Fatal(err)
	}
	died, err := awaitReady(context.Background(), c, conn.BaseURL, 180*time.Second, 500*time.Millisecond, probeHTTP)
	if err != nil || died {
		t.Fatalf("readiness: died=%v err=%v\n%s", died, err, tailLines(mustRead(t, logPath), 30))
	}
	// authed model listing: proxy contract (Bearer from the key file)
	req, _ := http.NewRequest(http.MethodGet, conn.BaseURL+"/v1/models", nil)
	req.Header.Set("Authorization", "Bearer "+loadUpstreamKey(conn.UpKeyFile))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	resp.Body.Close()
	var models struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if resp.StatusCode != http.StatusOK || json.Unmarshal(body, &models) != nil || len(models.Data) == 0 {
		t.Fatalf("GET /v1/models → HTTP %d body=%s (model should be loaded at spawn)", resp.StatusCode, body)
	}
	// clean stop: SIGTERM reaped, no orphan
	c.shutdownChild(killGrace)
	if c.cmd.ProcessState == nil {
		t.Error("llama-server not reaped after shutdownChild")
	}
}

func mustRead(t *testing.T, p string) []byte {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		return nil
	}
	return b
}
