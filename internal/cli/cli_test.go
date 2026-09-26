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
	code := Run(args, "v9.9.9", bytes.NewReader(nil), &out, &errb)
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
	// serve/ps/stop dispatch for real now (M5); only run stays planned.
	for _, c := range []string{"serve", "ps", "stop"} {
		if _, ok := planned[c]; ok {
			t.Errorf("%q still in the planned map", c)
		}
	}
	code, _, errOut := run("run")
	if code != 2 || !strings.Contains(errOut, "M6") {
		t.Errorf("code/err = %d/%q, want 2/mentions M6", code, errOut)
	}
}

func TestSetupRejectsUnknownFlag(t *testing.T) {
	// Usage path only — consent/download paths are covered offline in
	// internal/setup (seams injected); this must never touch the network.
	code, _, errOut := run("setup", "--bogus")
	if code != 2 || !strings.Contains(errOut, "usage: stone-llama setup") {
		t.Errorf("code/err = %d/%q", code, errOut)
	}
}

func TestUsageListsFrozenSurface(t *testing.T) {
	_, out, _ := run("help")
	for _, want := range []string{"fit ", "rank ", "--estimate", "--collection", "consent-gated"} {
		if !strings.Contains(out, want) {
			t.Errorf("usage missing %q", want)
		}
	}
}

func TestFitUsageWithoutRepo(t *testing.T) {
	code, _, errOut := run("fit")
	if code != 2 || !strings.Contains(errOut, "usage: stone-llama fit") {
		t.Errorf("code/err = %d/%q", code, errOut)
	}
}

func TestRankRequiresExactlyOneSource(t *testing.T) {
	for _, args := range [][]string{{"rank"}, {"rank", "--collection", "a/b", "--file", "f"}} {
		code, _, errOut := run(args...)
		if code != 2 || !strings.Contains(errOut, "usage: stone-llama rank") {
			t.Errorf("%v: code/err = %d/%q", args, code, errOut)
		}
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

func TestImportNameDefaultsToBasename(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("STONE_LLAMA_CONFIG", "")
	t.Setenv("XDG_CONFIG_HOME", "")
	t.Setenv("STONE_LLAMA_MODELS_DIR", filepath.Join(t.TempDir(), "models"))

	// no --name: defaults to the dir basename, then fails on the missing
	// source — a real attempt, not a usage error ([--name N] is optional).
	code, _, errOut := run("import", "/tmp/whatever")
	if code != 1 || !strings.Contains(errOut, "whatever") {
		t.Errorf("code/err = %d/%q", code, errOut)
	}
}

func runIn(stdin string, args ...string) (int, string, string) {
	var out, errb bytes.Buffer
	code := Run(args, "v9.9.9", strings.NewReader(stdin), &out, &errb)
	return code, out.String(), errb.String()
}

func TestPullUsageWithoutRef(t *testing.T) {
	code, _, errOut := run("pull")
	if code != 2 || !strings.Contains(errOut, "usage: stone-llama pull") {
		t.Errorf("code/err = %d/%q", code, errOut)
	}
}

func TestPullRequiresNVIDIAGPU(t *testing.T) {
	t.Setenv("PATH", "") // no nvidia-smi → doctor not ready
	code, out, errOut := run("pull", "org/model")
	if code != 1 {
		t.Errorf("code = %d, want 1", code)
	}
	if !strings.Contains(out, "NVIDIA GPU") {
		t.Errorf("doctor report not shown:\n%s", out)
	}
	if !strings.Contains(errOut, "needs a working NVIDIA GPU") {
		t.Errorf("err = %q", errOut)
	}
}

func TestLoginWritesTokenFile0600(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	t.Setenv("HOME", t.TempDir())
	code, out, errOut := runIn("hf_abcdef123456\n", "login")
	if code != 0 {
		t.Fatalf("code = %d, err = %q", code, errOut)
	}
	path := filepath.Join(os.Getenv("XDG_DATA_HOME"), "stone-llama", "hf_token")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(string(data)) != "hf_abcdef123456" {
		t.Errorf("file content = %q", data)
	}
	st, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if st.Mode().Perm() != 0o600 {
		t.Errorf("mode = %v, want 0600", st.Mode().Perm())
	}
	if !strings.Contains(out, "token saved") {
		t.Errorf("out = %q", out)
	}
	if strings.Contains(out+errOut, "hf_abcdef123456") {
		t.Error("token must never be echoed to any output stream")
	}
}

func TestLoginRejectsBadTokenAndArgs(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	t.Setenv("HOME", t.TempDir())
	code, _, errOut := runIn("not-a-token\n", "login")
	if code != 1 || !strings.Contains(errOut, "doesn't look like") {
		t.Errorf("code/err = %d/%q", code, errOut)
	}
	code, _, errOut = run("login", "extra-arg")
	if code != 2 || !strings.Contains(errOut, "never argv") {
		t.Errorf("args accepted: %d/%q", code, errOut)
	}
}

// /dev/null is a character device but not a terminal — the ModeCharDevice
// check used to make `setup </dev/null` prompt interactively.
func TestIsTerminalNotFooledByCharDevice(t *testing.T) {
	devNull, err := os.Open(os.DevNull)
	if err != nil {
		t.Fatal(err)
	}
	defer devNull.Close()
	if isTerminal(devNull) {
		t.Error("/dev/null must not count as a TTY")
	}
	pr, pw, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer pr.Close()
	defer pw.Close()
	if isTerminal(pr) || isTerminal(pw) {
		t.Error("pipe must not count as a TTY")
	}
	if isTerminal("stdin") { // non-*os.File
		t.Error("non-file must not count as a TTY")
	}
}
