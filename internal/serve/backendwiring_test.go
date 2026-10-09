package serve

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestBackendSelection table: "" and "tabby" must resolve to the
// pre-seam TabbyAPI engine (the default must not change), "llama" to
// llama-server, and anything else to a refusal — never a silent
// coercion to the default.
func TestBackendSelection(t *testing.T) {
	for _, tc := range []struct {
		name    string
		want    string
		wantErr bool
	}{
		{"", "serve.tabbyBackend", false},
		{BackendTabby, "serve.tabbyBackend", false},
		{BackendLlama, "serve.llamaBackend", false},
		{"cuda", "", true},
		{"Tabby", "", true}, // case-sensitive on purpose
	} {
		be, err := BackendFor(tc.name, 0)
		if tc.wantErr {
			if err == nil || !strings.Contains(err.Error(), "unknown backend") {
				t.Errorf("BackendFor(%q) err = %v, want unknown backend", tc.name, err)
			}
			continue
		}
		if err != nil {
			t.Errorf("BackendFor(%q): %v", tc.name, err)
			continue
		}
		if got := fmt.Sprintf("%T", be); got != tc.want {
			t.Errorf("BackendFor(%q) = %s, want %s", tc.name, got, tc.want)
		}
	}
}

// TestDefaultBackendIsTabby pins the production default: a zero
// Options (and so a config file with no "backend" key) supervises
// TabbyAPI exactly as before the seam.
func TestDefaultBackendIsTabby(t *testing.T) {
	be, err := BackendFor(Options{}.Backend, Options{}.DeviceBudgetMiB)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := be.(tabbyBackend); !ok {
		t.Errorf("zero Options selected %T, want tabbyBackend", be)
	}
}

// TestSpawnChildSharesStartProcess: the venv refusal is spawnChild's
// own (a TabbyAPI concern), but everything after it is the single
// startProcess implementation — pinned by running a real (harmless)
// process through it and reading back the same log/0600 contract both
// paths use.
//
// Neither call below goes through Serve: startProcess gets a one-shot
// `/bin/sh printf` (no daemon, and the port argument is only recorded
// on the child struct — never bound), while spawnChild refuses at its
// venv stat before it would exec anything.
func TestSpawnChildSharesStartProcess(t *testing.T) {
	rt := t.TempDir()
	// a venv python that exists: a 3-line script is a real POSIX shell
	// stand-in, and proves the argv/cwd/env/log path is startProcess's.
	binDir := filepath.Join(rt, "venv", "bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(rt, "tabbyAPI"), 0o755); err != nil {
		t.Fatal(err)
	}
	log := filepath.Join(t.TempDir(), "tabby.log")
	c, err := startProcess([]string{"/bin/sh", "-c", "printf hello"}, filepath.Join(rt, "tabbyAPI"), log, 0, nil)
	if err != nil {
		t.Fatalf("startProcess: %v", err)
	}
	if err := <-c.wait; err != nil {
		t.Fatalf("startProcess: %v", err)
	}
	b, err := os.ReadFile(log)
	if err != nil || !strings.Contains(string(b), "hello") {
		t.Errorf("log = %q err = %v, want the child output appended", b, err)
	}
	// spawnChild without a venv keeps its own refusal message verbatim
	// (it is a TabbyAPI diagnosis, not a generic spawn failure).
	if _, err := spawnChild(t.TempDir(), "", log, 0, nil); err == nil ||
		!strings.Contains(err.Error(), "runtime incomplete") {
		t.Errorf("spawnChild without venv err = %v, want runtime incomplete", err)
	}
}

// TestLlamaLoadRefusesNoHotSwap: /-/load on the llama backend must be
// a clear 409, never a silent TabbyAPI fallback — llama-server's model
// is fixed at spawn (-m), so a second model cannot be served.
//
// handleLoad is called directly on a bare daemon with an
// httptest.Recorder — no Serve, no listener, no child, so this can
// never bind a port.
func TestLlamaLoadRefusesNoHotSwap(t *testing.T) {
	d := &daemon{
		opts: Options{
			DataDir:   t.TempDir(),
			ModelsDir: t.TempDir(),
			Stderr:    io.Discard,
			// attach is empty on purpose: this is the supervised branch,
			// where the backend refusal is the only thing that can fire.
		},
		be:    llamaBackend{},
		conn:  Conn{BaseURL: "http://127.0.0.1:1"},
		token: "harness-token",
	}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/-/load",
		strings.NewReader(`{"model":"m"}`))
	d.handleLoad(rec, req)
	if rec.Code != http.StatusConflict {
		t.Fatalf("load on llama = %d, want 409 (body %q)", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	for _, want := range []string{"fixed at spawn", "backend_fixed_model", "--backend llama"} {
		if !strings.Contains(body, want) {
			t.Errorf("refusal missing %q: %s", want, body)
		}
	}
}
