package cli

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func run(args ...string) (int, string, string) {
	var out, errb bytes.Buffer
	code := Run(args, "v9.9.9", &out, &errb)
	return code, out.String(), errb.String()
}

func TestVersion(t *testing.T) {
	code, out, _ := run("version")
	if code != 0 || !strings.Contains(out, "v9.9.9") {
		t.Errorf("code/out = %d/%q", code, out)
	}
}

func TestBareAndHelpPrintUsage(t *testing.T) {
	for _, args := range [][]string{nil, {"help"}} {
		code, out, _ := run(args...)
		if code != 0 || !strings.Contains(out, "Commands:") {
			t.Errorf("%v: code/out = %d/%q", args, code, out)
		}
	}
}

func TestPlannedCommandNamesMilestone(t *testing.T) {
	code, _, errOut := run("pull")
	if code != 2 || !strings.Contains(errOut, "M2") {
		t.Errorf("code/err = %d/%q, want 2/mentions M2", code, errOut)
	}
}

func TestUnknownCommand(t *testing.T) {
	code, _, errOut := run("bogus")
	if code != 2 || !strings.Contains(errOut, "unknown command") {
		t.Errorf("code/err = %d/%q", code, errOut)
	}
}

func TestDoctorRejectsArguments(t *testing.T) {
	code, _, _ := run("doctor", "extra")
	if code != 2 {
		t.Errorf("code = %d, want 2", code)
	}
}

func TestDoctorExitCodesFollowProbe(t *testing.T) {
	dir := t.TempDir()
	smi := filepath.Join(dir, "nvidia-smi")
	script := "#!/bin/sh\necho \"NVIDIA GeForce RTX 3050 Laptop GPU, 4096, 580.178.04\"\n"
	if err := os.WriteFile(smi, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir)
	t.Setenv("HOME", t.TempDir())
	t.Setenv("STONE_LLAMA_CONFIG", "")
	t.Setenv("XDG_CONFIG_HOME", "")

	code, out, _ := run("doctor")
	if code != 0 || !strings.Contains(out, "cu13") {
		t.Errorf("ready: code/out = %d/%q", code, out)
	}

	t.Setenv("PATH", t.TempDir())
	code, out, _ = run("doctor")
	if code != 1 || !strings.Contains(out, "CUDA-only") {
		t.Errorf("no gpu: code/out = %d/%q", code, out)
	}
}

func TestListEmptyShowsHint(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("STONE_LLAMA_CONFIG", "")
	t.Setenv("XDG_CONFIG_HOME", "")
	t.Setenv("STONE_LLAMA_MODELS_DIR", filepath.Join(t.TempDir(), "no-models"))

	code, out, _ := run("list")
	if code != 0 || !strings.Contains(out, "no models") {
		t.Errorf("code/out = %d/%q", code, out)
	}
}

func TestImportListRmWiring(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("STONE_LLAMA_CONFIG", "")
	t.Setenv("XDG_CONFIG_HOME", "")
	modelsDir := filepath.Join(t.TempDir(), "models")
	t.Setenv("STONE_LLAMA_MODELS_DIR", modelsDir)

	// build an external model dir
	ext := filepath.Join(t.TempDir(), "ext-model")
	if err := os.MkdirAll(ext, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ext, "config.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ext, "w.bin"), make([]byte, 2048), 0o644); err != nil {
		t.Fatal(err)
	}

	code, out, errOut := run("import", ext, "--name", "ext-model")
	if code != 0 {
		t.Fatalf("import: code/err = %d/%q", code, errOut)
	}
	if !strings.Contains(out, "imported ext-model") {
		t.Errorf("import out = %q", out)
	}

	code, out, errOut = run("list")
	if code != 0 {
		t.Fatalf("list: code/err = %d/%q", code, errOut)
	}
	for _, want := range []string{"ext-model", "imported", "KiB"} {
		if !strings.Contains(out, want) {
			t.Errorf("list missing %q:\n%s", want, out)
		}
	}

	code, _, errOut = run("rm", "ext-model")
	if code != 0 {
		t.Fatalf("rm: code/err = %d/%q", code, errOut)
	}
	// link gone, target intact
	if _, err := os.Lstat(filepath.Join(modelsDir, "ext-model")); !os.IsNotExist(err) {
		t.Errorf("link still present: %v", err)
	}
	if _, err := os.Stat(filepath.Join(ext, "config.json")); err != nil {
		t.Errorf("target touched: %v", err)
	}
}

func TestRmMissingModelFails(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("STONE_LLAMA_CONFIG", "")
	t.Setenv("XDG_CONFIG_HOME", "")
	t.Setenv("STONE_LLAMA_MODELS_DIR", filepath.Join(t.TempDir(), "models"))

	code, _, errOut := run("rm", "ghost")
	if code != 1 || !strings.Contains(errOut, "not found") {
		t.Errorf("code/err = %d/%q", code, errOut)
	}
}

func TestImportRequiresName(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("STONE_LLAMA_CONFIG", "")
	t.Setenv("XDG_CONFIG_HOME", "")
	t.Setenv("STONE_LLAMA_MODELS_DIR", filepath.Join(t.TempDir(), "models"))

	code, _, errOut := run("import", "/tmp/whatever")
	if code != 2 || !strings.Contains(errOut, "--name") {
		t.Errorf("code/err = %d/%q", code, errOut)
	}
}
