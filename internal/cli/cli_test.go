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
	// Ready GPU → 0
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

	// No GPU → 1
	t.Setenv("PATH", t.TempDir())
	code, out, _ = run("doctor")
	if code != 1 || !strings.Contains(out, "CUDA-only") {
		t.Errorf("no gpu: code/out = %d/%q", code, out)
	}
}
