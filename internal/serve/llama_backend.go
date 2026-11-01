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
// models, so /-/load refuses a DIFFERENT model with a clear message
// (no silent TabbyAPI fallback) while a load resolving to the resident
// GGUF itself is an idempotent no-op (load.go, sameGGUF) and
// initialLoad records the spawned model for /-/status (load.go).
type llamaBackend struct {
	// deviceBudgetMiB: pre-spawn weights gate, 0 = off. llama-server
	// has no MiB budget flag (its --fit already caps itself to device
	// memory), so the budget is OUR arithmetic: refuse before spawn,
	// print the numbers, rather than let the engine OOM-trim silently.
	deviceBudgetMiB int
	// ctx is the context window (tokens) passed to llama-server as -c.
	// 0 = llamaDefaultCtx. Larger ctx on this 4 GB card measurably
	// degrades decode (see llamaDefaultCtx), so the default is a
	// deliberate, visible trade: full-speed decode in exchange for a
	// context that fits the card's headroom rather than the model's
	// training maximum.
	ctx int
}

// llamaDefaultCtx is the measured safe context (tokens) for the reference
// RTX 3050 Laptop 4 GB card with a SmolLM3-class Q4_K_M GGUF. With -np 2
// llama-server's fit gives each slot half, so the pinned 8192 yields
// 2 x 4096-token slots that serve multi-KB prompts at ~53-55 tok/s while
// larger ctx (16384/32768) drops decode to 15-23 tok/s (measured
// 2026-11-01, see /home/anon/ai/llamafix). A smaller value (e.g. 3072)
// decodes slightly faster but caps the longest served prompt; a larger
// value widens it at a measurable decode cost. Users override via config
// `llama_ctx`.
// ponytail: card-specific constant; derive from llama.cpp's reported KV
// bytes (its startup log) instead if the reference card changes.
const llamaDefaultCtx = 8192

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
		`backend "llama" serves GGUF only; non-GGUF models (e.g. EXL3) load under the tabby engine ('stone-llama serve --backend tabby')`
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

// sameGGUF reports whether two resolved model paths name the SAME file:
// os.SameFile compares the underlying file, so a full path, a relative
// spelling, or a symlinked path of the resident GGUF all count as equal
// (the typed string and /-/status's reported basename never need to
// match textually). A stat failure on either side means NOT the same —
// the caller falls back to the hot-swap refusal.
func sameGGUF(a, b string) bool {
	ai, err := os.Stat(a)
	if err != nil {
		return false
	}
	bi, err := os.Stat(b)
	if err != nil {
		return false
	}
	return os.SameFile(ai, bi)
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
	if o.spawn == nil {
		if _, err := exec.LookPath("llama-server"); err != nil {
			return Conn{}, "", fmt.Errorf("backend %q needs llama-server on PATH (build llama.cpp and put llama-server on PATH, or run the tabby engine with `stone-llama serve --backend tabby`): %w",
				BackendLlama, err)
		}
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

// llamaArgv builds llama-server's argv. It owns the flag policy: -c (the
// pinned context), -fa on, -np 2 — every one justified by measurement (see
// the Spawn comment). ctx <= 0 means llamaDefaultCtx. Pure so the policy is
// unit-testable without spawning a process.
func llamaArgv(bin, runtimeDir, cfgPath string, port, ctx int) []string {
	argv := []string{bin, "--host", "127.0.0.1", "--port", strconv.Itoa(port),
		"--api-key-file", llamaKeyFile(runtimeDir)}
	if cfgPath != "" {
		argv = append(argv, "-m", cfgPath)
	}
	// Pin the flags that decide VRAM and decode speed (measured, 2026-11-01,
	// on the reference RTX 3050 Laptop 4 GB with a SmolLM3-class Q4_K_M GGUF;
	// see llamaDefaultCtx): -c fixes the context so an over-allocated KV
	// cache cannot silently degrade or evict (unpinned, llama-server follows
	// the model's n_ctx_train — 16384+ collapses decode to 15-23 tok/s);
	// -fa on because this llama.cpp build's `auto` leaves flash attention
	// OFF (~40 vs ~53 tok/s); -np 2 so each slot stays at 4096 ctx and can
	// serve multi-KB prompts at full speed while keeping 2-way concurrency
	// (with -np 4 the fit shrinks each slot to 2048 and 8 KB prompts are
	// refused; with -np 1 a single slot gets the full 8192 but decode sits
	// at ~40 tok/s — the 4096-token-slot shape is the measured sweet spot:
	// ~53-55 tok/s short AND long prompts). `-b 2048 -ub 512` are already
	// llama-server defaults and measured at full speed, so they stay implicit.
	if ctx <= 0 {
		ctx = llamaDefaultCtx
	}
	argv = append(argv, "-c", strconv.Itoa(ctx), "-fa", "on", "-np", "2")
	return argv
}

// Spawn starts llama-server on loopback at the internal port; the GGUF
// llamaBackend.Prepare resolved rides -m. Model-less start (cfgPath
// "") is a legal llama-server mode. No secrets in argv (A7).
func (b llamaBackend) Spawn(runtimeDir, cfgPath, logPath string, port int, extraEnv map[string]string) (*child, error) {
	bin, err := exec.LookPath("llama-server")
	if err != nil {
		return nil, fmt.Errorf("backend %q needs llama-server on PATH (build llama.cpp and put llama-server on PATH, or run the tabby engine with `stone-llama serve --backend tabby`): %w",
			BackendLlama, err)
	}
	return startProcess(llamaArgv(bin, runtimeDir, cfgPath, port, b.ctx), "", logPath, port, extraEnv)
}
