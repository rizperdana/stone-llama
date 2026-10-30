package cli

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rizperdana/stone-llama/internal/config"
	"github.com/rizperdana/stone-llama/internal/serve"
)

// A live llama daemon + an EXL3 model must fail fast with the honest
// remedy (stop the daemon first): walking into the daemon's /-/load
// would replay the old "re-run 'stone-llama serve --backend llama
// <model.gguf>'" refusal, which no model can satisfy for EXL3.
func TestRunFailsFastOnLiveLlamaDaemonForEXL3(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	isolateConfig(t)
	dataDir := config.DataDir()
	mdir := filepath.Join(dataDir, "models", "Qwen3-EXL3")
	if err := os.MkdirAll(mdir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"quantization_config.json", "model.safetensors"} {
		if err := os.WriteFile(filepath.Join(mdir, f), []byte("{}"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	orig := queryStatusFn
	queryStatusFn = func(string) (serve.Status, error) {
		return serve.Status{Mode: "supervised", Host: "127.0.0.1", Port: 1, Backend: serve.BackendLlama}, nil
	}
	t.Cleanup(func() { queryStatusFn = orig })

	var out, errb bytes.Buffer
	if code := runRun([]string{"Qwen3-EXL3", "-p", "hi"}, bytes.NewReader(nil), &out, &errb); code != 1 {
		t.Fatalf("code = %d, want 1; err = %q", code, errb.String())
	}
	got := errb.String()
	for _, want := range []string{`backend "llama"`, "serves GGUF only", "'stone-llama stop'"} {
		if !strings.Contains(got, want) {
			t.Errorf("message missing %q: %s", want, got)
		}
	}
	if strings.Contains(got, "--backend llama") {
		t.Errorf("must not advise the unsatisfiable llama command: %s", got)
	}
}

// A legacy daemon whose /-/status predates the backend field must still
// fail fast (its engine is unknowable, so loading EXL3 is a gamble on a
// llama daemon) — with its own honest line, never the llama remedy.
func TestRunFailsFastOnLegacyDaemonWithoutBackendForEXL3(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	isolateConfig(t)
	dataDir := config.DataDir()
	mdir := filepath.Join(dataDir, "models", "Qwen3-EXL3")
	if err := os.MkdirAll(mdir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"quantization_config.json", "model.safetensors"} {
		if err := os.WriteFile(filepath.Join(mdir, f), []byte("{}"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	orig := queryStatusFn
	queryStatusFn = func(string) (serve.Status, error) {
		return serve.Status{Mode: "supervised", Host: "127.0.0.1", Port: 1}, nil
	}
	t.Cleanup(func() { queryStatusFn = orig })

	var out, errb bytes.Buffer
	if code := runRun([]string{"Qwen3-EXL3", "-p", "hi"}, bytes.NewReader(nil), &out, &errb); code != 1 {
		t.Fatalf("code = %d, want 1; err = %q", code, errb.String())
	}
	got := errb.String()
	if !strings.Contains(got, "no backend field") || !strings.Contains(got, "'stone-llama stop'") {
		t.Errorf("legacy-daemon refusal missing: %s", got)
	}
	if strings.Contains(got, "--backend llama") {
		t.Errorf("must not advise the unsatisfiable llama command: %s", got)
	}
}

// A tabby daemon + an EXL3 model is the CORRECT pairing: no fail-fast —
// run proceeds toward the load path (which errors on the dead port, not
// with the llama refusal).
func TestRunAllowsLiveTabbyDaemonForEXL3(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	isolateConfig(t)
	dataDir := config.DataDir()
	mdir := filepath.Join(dataDir, "models", "Qwen3-EXL3")
	if err := os.MkdirAll(mdir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"quantization_config.json", "model.safetensors"} {
		if err := os.WriteFile(filepath.Join(mdir, f), []byte("{}"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	orig := queryStatusFn
	queryStatusFn = func(string) (serve.Status, error) {
		return serve.Status{Mode: "supervised", Host: "127.0.0.1", Port: 1, Backend: serve.BackendTabby}, nil
	}
	t.Cleanup(func() { queryStatusFn = orig })

	var out, errb bytes.Buffer
	runRun([]string{"Qwen3-EXL3", "-p", "hi"}, bytes.NewReader(nil), &out, &errb)
	if strings.Contains(errb.String(), "serves GGUF only") || strings.Contains(errb.String(), "--backend tabby") {
		t.Errorf("tabby daemon must not trip the backend guard: %s", errb.String())
	}
}
