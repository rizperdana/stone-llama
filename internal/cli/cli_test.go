package cli

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/rizperdana/stone-llama/internal/config"
	"github.com/rizperdana/stone-llama/internal/serve"
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
	// M6 complete: no planned stubs remain. `run` with no model is a
	// usage error BEFORE any daemon work (never auto-starts in tests).
	code, _, errOut := run("run")
	if code != 2 || !strings.Contains(errOut, "usage: stone-llama run") {
		t.Errorf("run: code/err = %d/%q, want 2/usage", code, errOut)
	}
	code, _, errOut = run("frobnicate")
	if code != 2 || !strings.Contains(errOut, "unknown command") {
		t.Errorf("unknown: code/err = %d/%q", code, errOut)
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

// TestRunOneShotStreams: attach-mode daemon over a fake upstream —
// run -p skips the load (model already reported), streams tokens as
// they arrive, and must not unload (attach leaves lifecycle upstream).
func TestRunOneShotStreams(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, `{"object":"list","data":[]}`)
	})
	mux.HandleFunc("/v1/model", func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, `{"id":"fake-model"}`)
	})
	mux.HandleFunc("/v1/completions", func(w http.ResponseWriter, _ *http.Request) {
		fl := w.(http.Flusher)
		io.WriteString(w, "data: {\"choices\":[{\"text\":\"Hi \"}]}\n\n")
		fl.Flush()
		io.WriteString(w, "data: {\"choices\":[{\"text\":\"there\"}]}\n\n")
		fl.Flush()
		io.WriteString(w, "data: [DONE]\n\n")
		fl.Flush()
	})
	ts := httptest.NewServer(mux)
	defer ts.Close()

	dataDir := config.DataDir()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errCh := make(chan error, 1)
	go func() {
		errCh <- serve.Serve(ctx, serve.Options{
			Host:    "127.0.0.1",
			DataDir: dataDir,
			Attach:  strings.TrimPrefix(ts.URL, "http://"),
		})
	}()
	deadline := time.Now().Add(5 * time.Second)
	for {
		select {
		case err := <-errCh:
			t.Fatalf("serve exited early: %v", err)
		default:
		}
		if st, err := serve.Query(dataDir); err == nil && st.Ready {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("daemon not ready within 5s")
		}
		time.Sleep(50 * time.Millisecond)
	}

	var out, errb bytes.Buffer
	code := runRun([]string{"fake-model", "-p", "hello"}, bytes.NewReader(nil), &out, &errb)
	if code != 0 {
		t.Fatalf("run -p: code=%d err=%q", code, errb.String())
	}
	if !strings.Contains(out.String(), "Hi there") {
		t.Errorf("streamed output missing tokens: %q", out.String())
	}
	if strings.Contains(out.String(), "unloaded.") {
		t.Errorf("attach mode must not unload the upstream model")
	}
	cancel()
	<-errCh
}

// freeLocalPort returns a loopback port that was free a moment ago —
// for tests that probe the configured port (A3 diagnosis).
func freeLocalPort(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	return strconv.Itoa(ln.Addr().(*net.TCPAddr).Port)
}

// isolateConfig points config at a file-less scratch location so no
// test can resolve a real machine config — never spawn against, or
// clean, a production runtime dir.
func isolateConfig(t *testing.T) {
	t.Helper()
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	if v, ok := os.LookupEnv("STONE_LLAMA_CONFIG"); ok {
		os.Unsetenv("STONE_LLAMA_CONFIG")
		t.Cleanup(func() { os.Setenv("STONE_LLAMA_CONFIG", v) })
	}
}

// H4: run must honour STONE_LLAMA_NO_AUTOSTART=1 (exactly "1", the
// same contract ps uses) — a missing daemon is reported, not
// auto-started. The base run hardcoded auto-start = true.
func TestRunHonoursNoAutostart(t *testing.T) {
	if os.Getenv("SL_H4_CHILD") == "1" {
		// The base implementation re-execs this test binary as `serve`
		// for auto-start; the env guard stops that child re-entering
		// the suite. The fixed tree never spawns here at all.
		t.Skip("recursive starter guard (base re-execs the test binary)")
	}
	t.Setenv("SL_H4_CHILD", "1")
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("STONE_LLAMA_NO_AUTOSTART", "1")
	t.Setenv("STONE_LLAMA_PORT", freeLocalPort(t))
	code, _, errb := run("run", "somemodel", "-p", "hi")
	if code != 1 {
		t.Errorf("exit code = %d, want 1 (no daemon, autostart off)", code)
	}
	if !strings.Contains(errb, "no stone-llama daemon is running") {
		t.Errorf("STONE_LLAMA_NO_AUTOSTART ignored by run: %q", errb)
	}
	if _, err := os.Stat(filepath.Join(config.DataDir(), "logs", "daemon.log")); err == nil {
		t.Errorf("auto-start attempted despite NO_AUTOSTART=1 (daemon.log created)")
	}
}

// Item 3: a supervised-backend start failure prints the cause once and
// exactly one remedy block (attach → adopt → setup), naming only flags
// that exist; a failed start keeps no secrets.
func TestServeStartFailurePrintsRemediesOnce(t *testing.T) {
	isolateConfig(t)
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	code, _, errb := run("serve", "--port", freeLocalPort(t))
	if code != 1 {
		t.Fatalf("exit code = %d, want 1; err = %q", code, errb)
	}
	if !strings.Contains(errb, "runtime incomplete") {
		t.Fatalf("want runtime-incomplete cause, got %q", errb)
	}
	for _, want := range []string{
		"stone-llama serve --attach",
		"--key-file",
		"stone-llama setup --adopt",
		"stone-llama setup",
	} {
		if !strings.Contains(errb, want) {
			t.Errorf("remedy %q missing from %q", want, errb)
		}
	}
	if n := strings.Count(errb, "remedies, least friction first"); n != 1 {
		t.Errorf("remedy block appears %d times, want exactly 1: %q", n, errb)
	}
	if n := strings.Count(errb, "runtime incomplete"); n != 1 {
		t.Errorf("cause line appears %d times, want exactly 1: %q", n, errb)
	}
	if _, err := os.Lstat(filepath.Join(config.DataDir(), "runtime", "upstream_key")); !os.IsNotExist(err) {
		t.Errorf("generated upstream key left behind by failed serve")
	}
}

// H3 Case 2 (CLI): stop works through corrupt state — salvage, then the
// identification gates — instead of exiting 1 without signalling
// anything (base behaviour).
func TestStopRecoversFromCorruptState(t *testing.T) {
	isolateConfig(t)
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	if runtime.GOOS == "windows" {
		t.Skip("stop is refused on windows")
	}
	proc := exec.Command("sleep", "30")
	proc.Args = []string{"stone-llama serve", "30"}
	if err := proc.Start(); err != nil {
		t.Skipf("needs sleep(1): %v", err)
	}
	exited := make(chan error, 1)
	go func() { exited <- proc.Wait() }() // owns the Wait: zombie-free
	t.Cleanup(func() { proc.Process.Kill() })
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-Stone-Llama", "1")
		io.WriteString(w, "stone-llama\n")
	})
	mux.HandleFunc("/-/status", func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, "{}")
	})
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	_, portStr, err := net.SplitHostPort(strings.TrimPrefix(ts.URL, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatal(err)
	}
	// corrupt state (truncated write) salvaging pid/host/port
	dataDir := config.DataDir()
	if err := os.MkdirAll(dataDir, 0o700); err != nil {
		t.Fatal(err)
	}
	corrupt := fmt.Sprintf(`{"child_pid":7,"pid":%d,"host":"127.0.0.1","port":%d,"token":"tok-abc"`, proc.Process.Pid, port)
	if err := os.WriteFile(filepath.Join(dataDir, "daemon.json"), []byte(corrupt), 0o600); err != nil {
		t.Fatal(err)
	}
	code, out, errb := run("stop")
	if code != 0 {
		t.Fatalf("stop through corrupt state: code=%d out=%q err=%q", code, out, errb)
	}
	select {
	case <-exited:
	case <-time.After(2 * time.Second):
		t.Error("daemon stand-in survived stop")
	}
	if !strings.Contains(out, "stopped") {
		t.Errorf("stop output: %q", out)
	}
	if _, err := os.Stat(filepath.Join(dataDir, "daemon.json")); !os.IsNotExist(err) {
		t.Errorf("corrupt state not cleared after stop: %v", err)
	}
}

// H3 Case 2 (CLI): ps explains corrupt state — the live daemon is
// still shown, the corruption is reported on stderr, and the file is
// neither deleted nor hidden.
func TestPsExplainsCorruptState(t *testing.T) {
	isolateConfig(t)
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	t.Setenv("STONE_LLAMA_NO_AUTOSTART", "1")
	t.Setenv("STONE_LLAMA_PORT", freeLocalPort(t))
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-Stone-Llama", "1")
		io.WriteString(w, "stone-llama\n")
	})
	mux.HandleFunc("/-/status", func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, "{}")
	})
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	_, portStr, err := net.SplitHostPort(strings.TrimPrefix(ts.URL, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatal(err)
	}
	dataDir := config.DataDir()
	if err := os.MkdirAll(dataDir, 0o700); err != nil {
		t.Fatal(err)
	}
	corrupt := fmt.Sprintf(`{"child_pid":7,"pid":%d,"host":"127.0.0.1","port":%d,"token":"tok-abc"`, os.Getpid(), port)
	if err := os.WriteFile(filepath.Join(dataDir, "daemon.json"), []byte(corrupt), 0o600); err != nil {
		t.Fatal(err)
	}
	code, out, errb := run("ps")
	if code != 0 {
		t.Fatalf("ps through corrupt state: code=%d out=%q err=%q", code, out, errb)
	}
	if !strings.Contains(out, "running") {
		t.Errorf("status missing: %q", out)
	}
	if !strings.Contains(errb, "corrupt") {
		t.Errorf("corruption not reported: %q", errb)
	}
	if _, err := os.Stat(filepath.Join(dataDir, "daemon.json")); err != nil {
		t.Errorf("ps deleted the corrupt state file: %v", err)
	}
}
