// Package config loads stone-llama's configuration.
// Precedence: defaults < config file < environment. Flags are per-command
// and handled by the CLI layer above this package.
package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type Autofit struct {
	Enabled        bool `json:"enabled"`
	HeadroomMiB    int  `json:"headroom_mib"`     // base fragmentation headroom
	WorkspaceMiB   int  `json:"workspace_mib"`    // cold prefill workspace reserve [est]
	CtxHeadroomMiB int  `json:"ctx_headroom_mib"` // extra headroom at 65536 ctx, linear [est]
	OverheadMiB    int  `json:"overhead_mib"`
	MinCtx         int  `json:"min_ctx"`
	ChunkSize      int  `json:"chunk_size"` // TabbyAPI load chunk override: 0 auto (fit decides), else 512-4096
	// Speculative drafting, all load-time tuning like ChunkSize. "" is the
	// backend's own default and renders no draft_model block at all.
	DraftMode      string `json:"draft_mode"`       // model | disabled | mtp | ngram
	NgramMatchMin  int    `json:"ngram_match_min"`  // ngram_match_min: 0 backend default (2), else >= 1
	DraftNumTokens int    `json:"draft_num_tokens"` // tokens drafted per iteration: 0 backend default, else >= 1
}

type Config struct {
	ModelsDir       string            `json:"models_dir"`
	Host            string            `json:"host"`
	Port            int               `json:"port"`
	Autofit         Autofit           `json:"autofit"`
	RuntimeDir      string            `json:"runtime_dir"`
	UpstreamKeyFile string            `json:"upstream_key_file"` // attach-mode upstream Bearer source (file path only; secrets are not stored in config.json)
	Upstream        string            `json:"upstream"`          // auto-attach probe address (default http://127.0.0.1:5002)
	EngineEnv       map[string]string `json:"engine_env"`        // extra environment for the supervised backend child process; nil = none
	// Backend selects the supervised inference engine: "" or "tabby" is
	// TabbyAPI + ExLlamaV3 (the default, unchanged), "llama" is
	// llama-server (llama.cpp) over GGUF. An unknown name is refused at
	// serve time, never coerced to the default.
	Backend string `json:"backend"`
	// DeviceBudgetMiB caps the GGUF weights a llama spawn may load
	// (0 = no check). Weights only — KV cache and runtime overhead are
	// excluded, so this is a pre-spawn floor, not the engine's memory
	// use. Tabby ignores it.
	DeviceBudgetMiB int `json:"backend_device_budget_mib"`
}

// Default returns the built-in configuration. Missing file fields keep
// these values (JSON unmarshals into the prefilled struct).
func Default() Config {
	return Config{
		ModelsDir: filepath.Join(DataDir(), "models"),
		Host:      "127.0.0.1",
		Port:      5111,
		Autofit: Autofit{
			Enabled:        true,
			HeadroomMiB:    512,
			WorkspaceMiB:   512,
			CtxHeadroomMiB: 512,
			OverheadMiB:    128,
			MinCtx:         4096,
			DraftMode:      "",
			NgramMatchMin:  2,
			DraftNumTokens: 0,
		},
		Upstream: "http://127.0.0.1:5002",
	}

}

// DataDir is XDG_DATA_HOME/stone-llama, falling back to ~/.local/share/stone-llama.
func DataDir() string {
	if x := os.Getenv("XDG_DATA_HOME"); x != "" {
		return filepath.Join(x, "stone-llama")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".local", "share", "stone-llama")
}

func configPath() (path string, explicit bool) {
	if p := os.Getenv("STONE_LLAMA_CONFIG"); p != "" {
		return p, true
	}
	if x := os.Getenv("XDG_CONFIG_HOME"); x != "" {
		return filepath.Join(x, "stone-llama", "config.json"), false
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "stone-llama", "config.json"), false
}

// Load resolves the effective configuration. A missing default config file
// is fine; a missing STONE_LLAMA_CONFIG target is an error.
func Load() (Config, error) {
	cfg := Default()
	path, explicit := configPath()
	data, err := os.ReadFile(path)
	switch {
	case err == nil:
		if uerr := json.Unmarshal(data, &cfg); uerr != nil {
			return Config{}, fmt.Errorf("config %s: %w", path, uerr)
		}
	case os.IsNotExist(err) && explicit:
		return Config{}, fmt.Errorf("config %s not found (set by STONE_LLAMA_CONFIG)", path)
	case os.IsNotExist(err):
		// defaults
	default:
		return Config{}, fmt.Errorf("config %s: %w", path, err)
	}

	if v := os.Getenv("STONE_LLAMA_HOST"); v != "" {
		cfg.Host = v
	}
	if v := os.Getenv("STONE_LLAMA_MODELS_DIR"); v != "" {
		cfg.ModelsDir = v
	}
	if v := os.Getenv("STONE_LLAMA_PORT"); v != "" {
		n, perr := strconv.Atoi(v)
		if perr != nil || n < 1 || n > 65535 {
			return Config{}, fmt.Errorf("invalid STONE_LLAMA_PORT %q (want 1-65535)", v)
		}
		cfg.Port = n
	}
	if v := os.Getenv("STONE_LLAMA_UPSTREAM"); v != "" {
		cfg.Upstream = v
	}
	if v := os.Getenv("STONE_LLAMA_UPSTREAM_KEY_FILE"); v != "" {
		cfg.UpstreamKeyFile = v
	}
	if v := os.Getenv("STONE_LLAMA_BACKEND"); v != "" {
		cfg.Backend = v
	}
	// schemeless upstream address (the documented attach form) wins a
	// default scheme here so the rest of the code can treat Upstream as a URL
	if cfg.Upstream != "" && !strings.Contains(cfg.Upstream, "://") {
		cfg.Upstream = "http://" + cfg.Upstream
	}
	if c := cfg.Autofit.ChunkSize; c != 0 && (c < 512 || c > 4096) {
		return Config{}, fmt.Errorf("autofit.chunk_size must be 0 (auto) or 512-4096 (got %d) — remove it to let the fit decide", c)
	}
	cfg.ModelsDir = expandHome(cfg.ModelsDir)
	cfg.RuntimeDir = expandHome(cfg.RuntimeDir)
	return cfg, nil
}

func expandHome(p string) string {
	if p == "~" || strings.HasPrefix(p, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, strings.TrimPrefix(strings.TrimPrefix(p, "~"), "/"))
		}
	}
	return p
}
