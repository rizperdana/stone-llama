package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// cleanEnv points HOME/XDG at temp dirs and clears every stone-llama env
// override so each test sees only what it sets.
func cleanEnv(t *testing.T) string {
	t.Helper()
	tmp := t.TempDir()
	t.Setenv("HOME", tmp)
	t.Setenv("XDG_DATA_HOME", "")
	t.Setenv("XDG_CONFIG_HOME", "")
	t.Setenv("STONE_LLAMA_CONFIG", "")
	t.Setenv("STONE_LLAMA_HOST", "")
	t.Setenv("STONE_LLAMA_PORT", "")
	t.Setenv("STONE_LLAMA_MODELS_DIR", "")
	return tmp
}

func writeConfig(t *testing.T, xdgHome, body string) {
	t.Helper()
	path := filepath.Join(xdgHome, "stone-llama", "config.json")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestDefaults(t *testing.T) {
	home := cleanEnv(t)
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Port != 5111 || cfg.Host != "127.0.0.1" {
		t.Errorf("defaults host/port = %s:%d, want 127.0.0.1:5111", cfg.Host, cfg.Port)
	}
	if !cfg.Autofit.Enabled || cfg.Autofit.HeadroomMiB != 512 ||
		cfg.Autofit.WorkspaceMiB != 512 || cfg.Autofit.CtxHeadroomMiB != 512 ||
		cfg.Autofit.OverheadMiB != 128 || cfg.Autofit.MinCtx != 4096 {
		t.Errorf("autofit defaults = %+v", cfg.Autofit)
	}
	want := filepath.Join(home, ".local", "share", "stone-llama", "models")
	if cfg.ModelsDir != want {
		t.Errorf("ModelsDir = %q, want %q", cfg.ModelsDir, want)
	}
}

func TestFileOverridesDefaultsWithPartialMerge(t *testing.T) {
	home := cleanEnv(t)
	xdg := filepath.Join(home, "cfg")
	t.Setenv("XDG_CONFIG_HOME", xdg)
	writeConfig(t, xdg, `{"port": 6000, "models_dir": "~/m", "autofit": {"enabled": false}}`)

	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Port != 6000 {
		t.Errorf("Port = %d, want 6000", cfg.Port)
	}
	if cfg.ModelsDir != filepath.Join(home, "m") {
		t.Errorf("ModelsDir = %q, want %q", cfg.ModelsDir, filepath.Join(home, "m"))
	}
	if cfg.Autofit.Enabled {
		t.Error("Autofit.Enabled = true, want false from file")
	}
	if cfg.Autofit.HeadroomMiB != 512 {
		t.Errorf("HeadroomMiB = %d, want default 512 (partial merge)", cfg.Autofit.HeadroomMiB)
	}
}

func TestEnvOverridesFile(t *testing.T) {
	home := cleanEnv(t)
	xdg := filepath.Join(home, "cfg")
	t.Setenv("XDG_CONFIG_HOME", xdg)
	writeConfig(t, xdg, `{"port": 6000, "host": "1.2.3.4"}`)

	t.Setenv("STONE_LLAMA_PORT", "7000")
	t.Setenv("STONE_LLAMA_HOST", "0.0.0.0")
	t.Setenv("STONE_LLAMA_MODELS_DIR", "/data/models")

	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Port != 7000 || cfg.Host != "0.0.0.0" || cfg.ModelsDir != "/data/models" {
		t.Errorf("env did not override file: %+v", cfg)
	}
}

func TestExplicitConfigMissingIsError(t *testing.T) {
	cleanEnv(t)
	t.Setenv("STONE_LLAMA_CONFIG", filepath.Join(t.TempDir(), "nope.json"))
	if _, err := Load(); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Errorf("err = %v, want 'not found'", err)
	}
}

func TestInvalidPortRejected(t *testing.T) {
	for _, v := range []string{"abc", "0", "70000"} {
		cleanEnv(t)
		t.Setenv("STONE_LLAMA_PORT", v)
		if _, err := Load(); err == nil || !strings.Contains(err.Error(), "STONE_LLAMA_PORT") {
			t.Errorf("port %q: err = %v, want STONE_LLAMA_PORT error", v, err)
		}
	}
}

func TestBadJSONIsError(t *testing.T) {
	home := cleanEnv(t)
	xdg := filepath.Join(home, "cfg")
	t.Setenv("XDG_CONFIG_HOME", xdg)
	writeConfig(t, xdg, `{port:}`)
	if _, err := Load(); err == nil || !strings.Contains(err.Error(), "config") {
		t.Errorf("err = %v, want config parse error", err)
	}
}

// Speculative-decoding keys are load-time tuning next to chunk_size: the
// defaults must survive a file that sets only one of them (partial-merge
// semantics), so a partial draft_mode does not zero ngram_match_min.
func TestDraftDefaultsSurvivePartialFile(t *testing.T) {
	home := cleanEnv(t)
	xdg := filepath.Join(home, "cfg")
	t.Setenv("XDG_CONFIG_HOME", xdg)
	writeConfig(t, xdg, `{"autofit": {"draft_mode": "ngram"}}`)

	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Autofit.DraftMode != "ngram" || cfg.Autofit.NgramMatchMin != 2 || cfg.Autofit.DraftNumTokens != 0 {
		t.Errorf("partial file zeroed a draft default: %+v", cfg.Autofit)
	}
}

func TestEngineEnvDefaultsNilAndLoadsFromFile(t *testing.T) {
	home := cleanEnv(t)
	xdg := filepath.Join(home, "cfg")
	t.Setenv("XDG_CONFIG_HOME", xdg)
	if cfg, err := Load(); err != nil {
		t.Fatal(err)
	} else if cfg.EngineEnv != nil {
		t.Errorf("EngineEnv default = %v, want nil", cfg.EngineEnv)
	}

	writeConfig(t, xdg, `{"engine_env": {"CUDA_MODULE_LOADING": "LAZY"}}`)
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.EngineEnv["CUDA_MODULE_LOADING"] != "LAZY" || len(cfg.EngineEnv) != 1 {
		t.Errorf("EngineEnv = %v, want exactly the one file key", cfg.EngineEnv)
	}
	// draft defaults still hold alongside the engine block
	if cfg.Autofit.NgramMatchMin != 2 || cfg.Autofit.DraftMode != "" {
		t.Errorf("engine_env file disturbed draft defaults: %+v", cfg.Autofit)
	}
}
