package serve

import (
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// llamaBackend supervises llama-server (llama.cpp) over GGUF weights.
// The model is chosen at SPAWN (-m): llama-server cannot hot-swap
// models, so /-/load refuses a hot-swap with a clear message (no
// silent TabbyAPI fallback) while initialLoad records the already-
// spawned model for /-/status (load.go).
type llamaBackend struct {
	// deviceBudgetMiB: pre-spawn weights gate, 0 = off. llama-server
	// has no MiB budget flag (its --fit already caps itself to device
	// memory), so the budget is OUR arithmetic: refuse before spawn,
	// print the numbers, rather than let the engine OOM-trim silently.
	deviceBudgetMiB int
}

// llamaKeyFile is the path cleanupGenerated already removes
// (runtime/upstream_key): llama-server reads it via --api-key-file —
// newline-separated keys per llama.cpp common/arg.cpp — so the key
// never appears in argv or logs (A7).
func llamaKeyFile(runtimeDir string) string {
	return filepath.Join(runtimeDir, "upstream_key")
}

// resolveGGUF maps a model onto a GGUF file: an explicit .gguf path, a
// directory containing exactly one *.gguf, or a bare name looked up in
// ModelsDir. Anything else is an ERROR naming what the llama backend
// supports — never a silent fallback to TabbyAPI.
func resolveGGUF(model, modelsDir string) (string, error) {
	supported := "supported: a path to a .gguf file, or a model directory containing exactly one *.gguf — " +
		`backend "llama" serves GGUF only; non-GGUF models (e.g. EXL3) load under the default engine ('stone-llama serve --backend tabby')`
	candidates := []string{model}
	if !filepath.IsAbs(model) && modelsDir != "" && filepath.Join(modelsDir, model) != model {
		candidates = append(candidates, filepath.Join(modelsDir, model))
	}
	var st os.FileInfo
	var path string
	for _, c := range candidates {
		if info, err := os.Stat(c); err == nil {
			st, path = info, c
			break
		}
	}
	if st == nil {
		return "", fmt.Errorf("backend %q: model %q not found — %s", BackendLlama, model, supported)
	}
	if !st.IsDir() {
		if !strings.EqualFold(filepath.Ext(path), ".gguf") {
			return "", fmt.Errorf("backend %q: %q is not a .gguf file — %s", BackendLlama, path, supported)
		}
		return path, nil
	}
	matches, err := filepath.Glob(filepath.Join(path, "*.gguf"))
	if err != nil {
		return "", err
	}
	switch len(matches) {
	case 1:
		return matches[0], nil
	case 0:
		return "", fmt.Errorf("backend %q: no *.gguf in model dir %q — %s", BackendLlama, path, supported)
	default:
		names := make([]string, len(matches))
		for i, m := range matches {
			names[i] = filepath.Base(m)
		}
		return "", fmt.Errorf("backend %q: %d GGUF files in %q (%s) — name one explicitly: %s",
			BackendLlama, len(matches), path, strings.Join(names, ", "), supported)
	}
}

// Prepare validates the spawn inputs in order — GGUF routing first
// (pure, so a non-GGUF model is refused with guidance even on machines
// without llama-server), then the binary, then the upstream key file.
// Returns the resolved GGUF path as cfgPath ("" = model-less start),
// which Spawn is the only consumer of.
func (b llamaBackend) Prepare(o *Options, port int) (Conn, string, error) {
	gguf := ""
	if o.Model != "" {
		var err error
		if gguf, err = resolveGGUF(o.Model, o.ModelsDir); err != nil {
			return Conn{}, "", err
		}
		if b.deviceBudgetMiB > 0 {
			st, serr := os.Stat(gguf)
			if serr != nil {
				return Conn{}, "", serr
			}
			if mib := st.Size() / (1 << 20); mib > int64(b.deviceBudgetMiB) {
				return Conn{}, "", fmt.Errorf("backend %q: gguf %q weights ≈ %d MiB exceed backend_device_budget_mib %d (weights only — KV cache and runtime overhead are excluded, so this is a floor on the weights, not the engine's memory use)",
					BackendLlama, filepath.Base(gguf), mib, b.deviceBudgetMiB)
			}
		}
	}
	if _, err := exec.LookPath("llama-server"); err != nil {
		return Conn{}, "", fmt.Errorf("backend %q needs llama-server on PATH (build llama.cpp and put llama-server on PATH, or run the default engine with 'stone-llama serve --backend tabby'): %w",
			BackendLlama, err)
	}
	key, err := generateToken()
	if err != nil {
		return Conn{}, "", err
	}
	keyFile := llamaKeyFile(o.RuntimeDir)
	if err := writeSecret(keyFile, key+"\n"); err != nil {
		return Conn{}, "", fmt.Errorf("write child keys: %w", err)
	}
	return Conn{
		BaseURL:   "http://" + net.JoinHostPort("127.0.0.1", strconv.Itoa(port)),
		Name:      "llama.cpp (llama-server)",
		UpKeyFile: keyFile,
	}, gguf, nil
}

// Spawn starts llama-server on loopback at the internal port; the GGUF
// llamaBackend.Prepare resolved rides -m. Model-less start (cfgPath
// "") is a legal llama-server mode. No secrets in argv (A7).
func (b llamaBackend) Spawn(runtimeDir, cfgPath, logPath string, port int, extraEnv map[string]string) (*child, error) {
	bin, err := exec.LookPath("llama-server")
	if err != nil {
		return nil, fmt.Errorf("backend %q needs llama-server on PATH (build llama.cpp and put llama-server on PATH, or run the default engine with 'stone-llama serve --backend tabby'): %w",
			BackendLlama, err)
	}
	argv := []string{bin, "--host", "127.0.0.1", "--port", strconv.Itoa(port),
		"--api-key-file", llamaKeyFile(runtimeDir)}
	if cfgPath != "" {
		argv = append(argv, "-m", cfgPath)
	}
	return startProcess(argv, "", logPath, port, extraEnv)
}
