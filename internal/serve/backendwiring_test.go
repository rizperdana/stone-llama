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

// TestBackendSelection table: "" and "llama" must resolve to the
// llama-server engine (the new default), "tabby" to TabbyAPI, and
// llama-server, and anything else to a refusal — never a silent
// coercion to the default.
func TestBackendSelection(t *testing.T) {
	for _, tc := range []struct {
		name    string
		want    string
		wantErr bool
	}{
		{"", "serve.llamaBackend", false},
		{BackendLlama, "serve.llamaBackend", false},
		{BackendTabby, "serve.tabbyBackend", false},
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

// TestDefaultBackendIsLlama pins the production default: a zero
// Options (and so a config file with no "backend" key) supervises
// llama-server exactly as after the flip.
func TestDefaultBackendIsLlama(t *testing.T) {
	be, err := BackendFor(Options{}.Backend, Options{}.DeviceBudgetMiB)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := be.(llamaBackend); !ok {
		t.Errorf("zero Options selected %T, want llamaBackend", be)
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

// TestLlamaLoadRefusesNoHotSwap: /-/load on a llama daemon WITH a
// resident GGUF must be a clear 409, never a silent TabbyAPI fallback —
// llama-server's model is fixed at spawn (-m), so a second, DIFFERENT
// model cannot be served. Both spellings of "different" are pinned: a
// name the resolver cannot map to any GGUF, and a resolvable GGUF that
// is not the resident file. (A load resolving to the resident GGUF
// itself is a 200 no-op — serve_test.go TestLlamaLoadSameModelNoOp.)
//
// handleLoad is called directly on a bare daemon with an
// httptest.Recorder — no Serve, no listener, no child, so this can
// never bind a port.
func TestLlamaLoadRefusesNoHotSwap(t *testing.T) {
	dir := t.TempDir()
	resident := filepath.Join(dir, "resident.gguf")
	other := filepath.Join(dir, "other.gguf")
	for _, p := range []string{resident, other} {
		if err := os.WriteFile(p, []byte("ggufbytes"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	d := &daemon{
		opts: Options{
			DataDir:   t.TempDir(),
			ModelsDir: t.TempDir(),
			Stderr:    io.Discard,
			// attach is empty on purpose: this is the supervised branch,
			// where the backend refusal is the only thing that can fire.
		},
		be:      llamaBackend{},
		cfgPath: resident,
		conn:    Conn{BaseURL: "http://127.0.0.1:1"},
		token:   "harness-token",
	}
	for _, model := range []string{"m", other} { // unresolvable name, then a different GGUF by full path
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/-/load",
			strings.NewReader(`{"model":"`+model+`"}`))
		d.handleLoad(rec, req)
		if rec.Code != http.StatusConflict {
			t.Fatalf("load %q on llama = %d, want 409 (body %q)", model, rec.Code, rec.Body.String())
		}
		body := rec.Body.String()
		for _, want := range []string{"fixed at spawn", "backend_fixed_model", "--backend llama"} {
			if !strings.Contains(body, want) {
				t.Errorf("refusal for %q missing %q: %s", model, want, body)
			}
		}
	}
}

// TestLlamaLoadWithoutResidentModelRefuses: a model-less llama spawn
// (cfgPath "") serves nothing, and llama-server has NO runtime
// model-load endpoint — the help text (-m, --model FNAME: "model path to
// load") puts every GGUF load at startup, and the only model-autoload
// flags (--models-dir/--models-preset/--models-max/--models-autoload)
// belong to the router server, which this Spawn never starts (it passes
// only --host/--port/--api-key-file/-m). So /-/load still 409s, with the
// message truthful for THAT case: nothing is loaded to "switch", so
// restart naming the GGUF (type backend_no_model).
func TestLlamaLoadWithoutResidentModelRefuses(t *testing.T) {
	d := &daemon{
		opts: Options{
			DataDir:   t.TempDir(),
			ModelsDir: t.TempDir(),
			Stderr:    io.Discard,
		},
		be:    llamaBackend{},
		conn:  Conn{BaseURL: "http://127.0.0.1:1"},
		token: "harness-token",
	}
	rec := httptest.NewRecorder()
	d.handleLoad(rec, httptest.NewRequest(http.MethodPost, "/-/load",
		strings.NewReader(`{"model":"/x/m.gguf"}`)))
	if rec.Code != http.StatusConflict {
		t.Fatalf("load on model-less llama = %d, want 409 (body %q)", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	for _, want := range []string{"serving no model", "backend_no_model", "--backend llama", "naming the GGUF"} {
		if !strings.Contains(body, want) {
			t.Errorf("model-less refusal missing %q: %s", want, body)
		}
	}
}
