package cli

import (
	"bytes"
	"context"
	"encoding/json"
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
	"sync"
	"syscall"
	"testing"
	"time"
	"unsafe"

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

// One corrupt manifest must not hide every other model: list skips the
// bad entry, names it on stderr, and still shows the healthy ones (a
// single bad dir used to abort the whole listing with rc=1).
func TestListSkipsCorruptManifest(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("STONE_LLAMA_CONFIG", "")
	t.Setenv("XDG_CONFIG_HOME", "")
	modelsDir := filepath.Join(t.TempDir(), "models")
	t.Setenv("STONE_LLAMA_MODELS_DIR", modelsDir)

	good := filepath.Join(modelsDir, "good")
	if err := os.MkdirAll(good, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(good, "config.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	bad := filepath.Join(modelsDir, "bad")
	if err := os.MkdirAll(bad, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(bad, "manifest.json"), []byte("{oops"), 0o644); err != nil {
		t.Fatal(err)
	}

	code, out, errOut := run("list")
	if code != 0 {
		t.Fatalf("list with one corrupt manifest: code/err = %d/%q", code, errOut)
	}
	if !strings.Contains(out, "good") {
		t.Errorf("healthy model hidden by the corrupt one:\n%s", out)
	}
	if !strings.Contains(errOut, "bad") || !strings.Contains(errOut, "manifest") {
		t.Errorf("corrupt entry not named on stderr: %q", errOut)
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
	isolateConfig(t)
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, `{"object":"list","data":[]}`)
	})
	mux.HandleFunc("/v1/model", func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, `{"id":"fake-model"}`)
	})
	var mu sync.Mutex
	var lastBody []byte
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		lastBody = append(lastBody[:0], b...)
		mu.Unlock()
		if !strings.Contains(string(b), `"messages"`) || strings.Contains(string(b), `"prompt"`) {
			t.Errorf("run must speak the chat dialect (messages, not raw prompt): %s", b)
		}
		finish := `"stop"`
		if strings.Contains(string(b), "long") {
			finish = `"length"`
		}
		fl := w.(http.Flusher)
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Hi \"}}]}\n\n")
		fl.Flush()
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"there\"}}]}\n\n")
		fl.Flush()
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":"+finish+"}]}\n\n")
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
	body := func() []byte {
		mu.Lock()
		defer mu.Unlock()
		return append([]byte(nil), lastBody...)
	}
	type chatReq struct {
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
		Temperature    *float64 `json:"temperature"`
		TopP           *float64 `json:"top_p"`
		EnableThinking *bool    `json:"enable_thinking"`
		MaxTokens      *int     `json:"max_tokens"`
	}
	parse := func() chatReq {
		var c chatReq
		if err := json.Unmarshal(body(), &c); err != nil {
			t.Fatalf("bad request body %s: %v", body(), err)
		}
		return c
	}

	// 1) default: terse system message, bounded, thinking untouched, and
	// no sampling fields until metadata exists (backend defaults apply).
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
	if strings.Contains(errb.String(), "truncated") {
		t.Errorf("clean run must not warn about truncation: %q", errb.String())
	}
	c := parse()
	if len(c.Messages) != 2 || c.Messages[0].Role != "system" || c.Messages[0].Content != defaultSystemMsg {
		t.Errorf("default system message missing: %+v", c.Messages)
	}
	if c.Temperature != nil || c.TopP != nil || c.EnableThinking != nil {
		t.Errorf("no generation_config.json on disk → no tuned fields, got temp=%v top_p=%v thinking=%v", c.Temperature, c.TopP, c.EnableThinking)
	}
	if c.MaxTokens == nil || *c.MaxTokens != 2048 {
		t.Errorf("max_tokens = %v, want 2048", c.MaxTokens)
	}

	// 2) the model's own generation_config.json auto-tunes sampling.
	gc := filepath.Join(dataDir, "models", "fake-model", "generation_config.json")
	if err := os.MkdirAll(filepath.Dir(gc), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(gc, []byte(`{"temperature":0.6,"top_p":0.95,"do_sample":true}`), 0o644); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errb.Reset()
	if code := runRun([]string{"fake-model", "-p", "hello"}, bytes.NewReader(nil), &out, &errb); code != 0 {
		t.Fatalf("run -p profiled: code=%d err=%q", code, errb.String())
	}
	c = parse()
	if c.Temperature == nil || *c.Temperature != 0.6 || c.TopP == nil || *c.TopP != 0.95 {
		t.Errorf("generation_config.json not applied: temp=%v top_p=%v", c.Temperature, c.TopP)
	}

	// 3) explicit flags beat the profile; --no-system drops the system
	// message; --thinking sets the template toggle.
	out.Reset()
	errb.Reset()
	if code := runRun([]string{"fake-model", "--temperature", "0.3", "--top-p", "0.5", "--no-system", "--thinking", "-p", "hello"}, bytes.NewReader(nil), &out, &errb); code != 0 {
		t.Fatalf("run -p flags: code=%d err=%q", code, errb.String())
	}
	c = parse()
	if c.Temperature == nil || *c.Temperature != 0.3 || c.TopP == nil || *c.TopP != 0.5 {
		t.Errorf("explicit flags ignored: temp=%v top_p=%v", c.Temperature, c.TopP)
	}
	if len(c.Messages) != 1 || c.Messages[0].Role != "user" {
		t.Errorf("--no-system must send no system message: %+v", c.Messages)
	}
	if c.EnableThinking == nil || !*c.EnableThinking {
		t.Errorf("--thinking must set enable_thinking=true: %v", c.EnableThinking)
	}

	// 4) a caller-supplied system message replaces the default.
	out.Reset()
	errb.Reset()
	if code := runRun([]string{"fake-model", "--system", "Be brief.", "-p", "hello"}, bytes.NewReader(nil), &out, &errb); code != 0 {
		t.Fatalf("run -p custom system: code=%d err=%q", code, errb.String())
	}
	c = parse()
	if len(c.Messages) < 1 || c.Messages[0].Role != "system" || c.Messages[0].Content != "Be brief." {
		t.Errorf("--system must replace the default: %+v", c.Messages)
	}

	// 5) the REPL prints exactly one profile line before the first prompt.
	out.Reset()
	errb.Reset()
	if code := runRun([]string{"fake-model"}, strings.NewReader("/bye\n"), &out, &errb); code != 0 {
		t.Fatalf("repl: code=%d err=%q", code, errb.String())
	}
	if !strings.Contains(out.String(), "profile: max_tokens 2048") ||
		!strings.Contains(out.String(), "temperature 0.6 (generation_config.json)") ||
		!strings.Contains(out.String(), defaultSystemMsg) {
		t.Errorf("REPL profile line missing: %q", out.String())
	}

	// 6) A length-capped generation must say so instead of ending silently.
	out.Reset()
	errb.Reset()
	if code := runRun([]string{"fake-model", "--max-tokens", "8", "-p", "long story"}, bytes.NewReader(nil), &out, &errb); code != 0 {
		t.Fatalf("run -p bounded: code=%d err=%q", code, errb.String())
	}
	if !strings.Contains(errb.String(), "truncated at --max-tokens 8") {
		t.Errorf("truncation notice missing: %q", errb.String())
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

// H4: run must honour STONE_LLAMA_NO_AUTOSTART=1 (exactly "1") — a
// missing daemon is reported, not auto-started. The base run hardcoded
// auto-start = true. (ps never consults the env: it is read-only.)
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
	// zero-config: run now probes the configured upstream before the
	// (suppressed) supervised start, so isolate it from any live daemon —
	// otherwise the real process on 5002 would be auto-attached instead of
	// hitting the no-daemon error this test exercises.
	t.Setenv("STONE_LLAMA_UPSTREAM", "http://127.0.0.1:1")
	t.Setenv("STONE_LLAMA_PORT", freeLocalPort(t))
	code, _, errb := run("run", "somemodel", "-p", "hi")
	if code != 1 {
		t.Errorf("exit code = %d, want 1 (no daemon, autostart off)", code)
	}
	if !strings.Contains(errb, "no daemon is running") {
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
	waitExecCmdline(t, proc.Process.Pid, "stone-llama serve")
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

// waitExecCmdline blocks until /proc/<pid>/cmdline carries want. Right after
// Start returns the child may not have execve'd yet — /proc then shows OUR
// argv, Stop's identity gate reads the stand-in as a foreign pid, drops the
// state without signalling, and the test flakes under load (measured ~9%).
// Non-linux: best effort, no /proc.
func waitExecCmdline(t *testing.T, pid int, want string) {
	t.Helper()
	if runtime.GOOS != "linux" {
		return
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		b, err := os.ReadFile(fmt.Sprintf("/proc/%d/cmdline", pid))
		if err == nil && strings.Contains(string(b), want) {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Skipf("stand-in pid %d never showed cmdline %q", pid, want)
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

// Read-only contract: with no daemon, plain ps must not spawn — no
// re-exec (daemon.log), no daemon.json, no listener — and
// STONE_LLAMA_NO_AUTOSTART=1 must now be byte-identical: the env can no
// longer be the difference.
func TestPsNeverSpawns(t *testing.T) {
	isolateConfig(t)
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	t.Setenv("STONE_LLAMA_NO_AUTOSTART", "")
	port := freeLocalPort(t)
	t.Setenv("STONE_LLAMA_PORT", port)

	plainCode, plainOut, plainErr := run("ps")
	if plainCode != 1 {
		t.Fatalf("ps with no daemon: code=%d out=%q err=%q", plainCode, plainOut, plainErr)
	}
	if !strings.Contains(plainErr, "no stone-llama daemon is running (start one with 'stone-llama serve')") {
		t.Errorf("ps must state absence + remedy in one line: %q", plainErr)
	}
	dataDir := config.DataDir()
	if _, err := os.Stat(filepath.Join(dataDir, "daemon.json")); !os.IsNotExist(err) {
		t.Errorf("ps created daemon.json: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dataDir, "logs", "daemon.log")); !os.IsNotExist(err) {
		t.Errorf("ps spawned a daemon (daemon.log exists): %v", err)
	}
	if ln, lerr := net.Listen("tcp", "127.0.0.1:"+port); lerr != nil {
		t.Errorf("ps left a listener on %s: %v", port, lerr)
	} else {
		ln.Close()
	}

	t.Setenv("STONE_LLAMA_NO_AUTOSTART", "1")
	if code, out, errb := run("ps"); code != plainCode || out != plainOut || errb != plainErr {
		t.Errorf("NO_AUTOSTART=1 ps differs from plain ps:\n plain: %d %q %q\n env:   %d %q %q",
			plainCode, plainOut, plainErr, code, out, errb)
	}
}

// A live daemon is reported, read-only: daemon.json stays
// byte-identical and no daemon.log appears (ps started nothing).
func TestPsReportsLiveDaemonWithoutSpawning(t *testing.T) {
	isolateConfig(t)
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	t.Setenv("STONE_LLAMA_NO_AUTOSTART", "")
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-Stone-Llama", "1")
		io.WriteString(w, "stone-llama\n")
	})
	mux.HandleFunc("/-/status", func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, `{"mode":"supervised","pid":`+strconv.Itoa(os.Getpid())+`,"child_pid":77,"model":"qwen-test","uptime_s":12,"ready":true}`)
	})
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	_, portStr, err := net.SplitHostPort(strings.TrimPrefix(ts.URL, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	dataDir := config.DataDir()
	if err := os.MkdirAll(dataDir, 0o700); err != nil {
		t.Fatal(err)
	}
	state := fmt.Sprintf(`{"child_pid":77,"pid":%d,"host":"127.0.0.1","port":%s,"token":"tok-abc","started_at":%d}`, os.Getpid(), portStr, time.Now().Unix())
	statePath := filepath.Join(dataDir, "daemon.json")
	if err := os.WriteFile(statePath, []byte(state), 0o600); err != nil {
		t.Fatal(err)
	}

	code, out, errb := run("ps")
	if code != 0 {
		t.Fatalf("ps with live daemon: code=%d out=%q err=%q", code, out, errb)
	}
	for _, want := range []string{
		fmt.Sprintf("running (pid %d)", os.Getpid()),
		"supervised (child pid 77)",
		"model:    qwen-test",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("ps output missing %q: %q", want, out)
		}
	}
	if _, err := os.Stat(filepath.Join(dataDir, "logs", "daemon.log")); !os.IsNotExist(err) {
		t.Errorf("ps spawned a daemon (daemon.log exists): %v", err)
	}
	after, rerr := os.ReadFile(statePath)
	if rerr != nil || string(after) != state {
		t.Errorf("ps rewrote daemon.json: %q %v", after, rerr)
	}
}

// A foreign process on the configured port is named — ps may never
// claim "no daemon exists" while something else answers (A3), and it
// must not try to start one either.
func TestPsNamesForeignPortHolderWithoutSpawning(t *testing.T) {
	isolateConfig(t)
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	t.Setenv("STONE_LLAMA_NO_AUTOSTART", "")
	// answers HTTP but carries no stone-llama identity → probeForeign
	ts := httptest.NewServer(http.NotFoundHandler())
	t.Cleanup(ts.Close)
	_, portStr, err := net.SplitHostPort(strings.TrimPrefix(ts.URL, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("STONE_LLAMA_PORT", portStr)

	code, out, errb := run("ps")
	if code != 1 {
		t.Fatalf("ps with foreign holder: code=%d out=%q err=%q", code, out, errb)
	}
	if !strings.Contains(errb, "another process holds") {
		t.Errorf("ps must name the foreign holder: %q", errb)
	}
	if strings.Contains(errb, "no stone-llama daemon is running") {
		t.Errorf("ps claimed absence while a foreign process answered: %q", errb)
	}
	dataDir := config.DataDir()
	if _, err := os.Stat(filepath.Join(dataDir, "daemon.json")); !os.IsNotExist(err) {
		t.Errorf("ps created daemon.json: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dataDir, "logs", "daemon.log")); !os.IsNotExist(err) {
		t.Errorf("ps spawned a daemon (daemon.log exists): %v", err)
	}
}

// Defect 1: a stale runtime/upstream_key must not shadow the checkout's
// live key — the model query walks candidates while the upstream answers
// 401/403, so zero-flag run resolves the loaded model and chats against an
// auth-required upstream. Alias typed, loaded id used.
func TestRunWalksKeyCandidatesOn401(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	isolateConfig(t)
	t.Setenv("STONE_LLAMA_UPSTREAM_KEY", "")
	t.Setenv("STONE_LLAMA_UPSTREAM_KEY_FILE", "")
	const liveKey = "live-key"
	var mu sync.Mutex
	var chatAuth, chatModel string
	needKey := func(w http.ResponseWriter, r *http.Request) bool {
		if r.Header.Get("Authorization") == "Bearer "+liveKey {
			return true
		}
		w.WriteHeader(http.StatusUnauthorized)
		return false
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		if !needKey(w, r) {
			return
		}
		io.WriteString(w, `{"object":"list","data":[{"id":"live-model"}]}`)
	})
	mux.HandleFunc("/v1/model", func(w http.ResponseWriter, r *http.Request) {
		if !needKey(w, r) {
			return
		}
		io.WriteString(w, `{"id":"live-model"}`)
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		if !needKey(w, r) {
			return
		}
		b, _ := io.ReadAll(r.Body)
		var body struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(b, &body)
		mu.Lock()
		chatAuth = r.Header.Get("Authorization")
		chatModel = body.Model
		mu.Unlock()
		fl := w.(http.Flusher)
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"2\"}}]}\n\n")
		fl.Flush()
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		fl.Flush()
		io.WriteString(w, "data: [DONE]\n\n")
		fl.Flush()
	})
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	t.Setenv("STONE_LLAMA_UPSTREAM", ts.URL)

	// higher-precedence candidate: the stale state key (wrong on purpose)
	dataDir := config.DataDir()
	if err := os.MkdirAll(filepath.Join(dataDir, "runtime"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dataDir, "runtime", "upstream_key"), []byte("stale-key\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	// lower-precedence candidate: the discovered checkout's api_tokens.yml
	checkout := t.TempDir()
	if err := os.WriteFile(filepath.Join(checkout, "api_tokens.yml"),
		[]byte("api_key: live-key\nadmin_key: admin-key\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	origDiscover := discoverCheckoutFn
	discoverCheckoutFn = func(string) string { return checkout }
	t.Cleanup(func() { discoverCheckoutFn = origDiscover })

	code, out, errb := run("run", "alias-model", "-p", "1+1")
	if code != 0 {
		t.Fatalf("code = %d, out = %q, err = %q", code, out, errb)
	}
	for _, want := range []string{
		"using the TabbyAPI already running on",
		`note: using "live-model" (loaded upstream) — "alias-model" is a local alias`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("out missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out+errb, "no model is loaded") {
		t.Errorf("401 misread as no-model:\n%s%s", out, errb)
	}
	mu.Lock()
	defer mu.Unlock()
	if chatAuth != "Bearer "+liveKey {
		t.Error("chat request did not carry the live key (stale candidate was not retried past)")
	}
	if chatModel != "live-model" {
		t.Errorf("chat model = %q, want the upstream's loaded id (alias resolution)", chatModel)
	}
	if !strings.HasSuffix(strings.TrimSpace(out), "2") {
		t.Errorf("answer missing: %q", out)
	}
}

// Defect 1b: when every candidate is refused (401/403) the error must name
// the auth cause — never claim what is loaded.
func TestRunAuthRefusalNamesKeyNotModel(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	isolateConfig(t)
	t.Setenv("STONE_LLAMA_UPSTREAM_KEY", "")
	t.Setenv("STONE_LLAMA_UPSTREAM_KEY_FILE", "")
	mux := http.NewServeMux()
	unauthorized := func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}
	mux.HandleFunc("/v1/models", unauthorized)
	mux.HandleFunc("/v1/model", unauthorized)
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	t.Setenv("STONE_LLAMA_UPSTREAM", ts.URL)
	origDiscover := discoverCheckoutFn
	discoverCheckoutFn = func(string) string { return "" }
	t.Cleanup(func() { discoverCheckoutFn = origDiscover })

	code, out, errb := run("run", "some-model", "-p", "hi")
	if code != 1 {
		t.Errorf("code = %d, out = %q, err = %q", code, out, errb)
	}
	if !strings.Contains(errb, "401/403") {
		t.Errorf("auth refusal not named: %q", errb)
	}
	for _, bad := range []string{"no model is loaded", "answers /v1/models"} {
		if strings.Contains(out+errb, bad) {
			t.Errorf("refusal misreported as %q: %q", bad, out+errb)
		}
	}
}

// Defect 1c: an unreadable model query (500 here) must not hard-fail — run
// keeps the typed name and the upstream answers for it.
func TestRunUnreadableModelListKeepsTypedName(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	isolateConfig(t)
	t.Setenv("STONE_LLAMA_UPSTREAM_KEY", "")
	t.Setenv("STONE_LLAMA_UPSTREAM_KEY_FILE", "")
	var mu sync.Mutex
	var chatModel string
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, `{"object":"list","data":[]}`) // probe: serving
	})
	mux.HandleFunc("/v1/model", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError) // unreadable
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		var body struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(b, &body)
		mu.Lock()
		chatModel = body.Model
		mu.Unlock()
		fl := w.(http.Flusher)
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		fl.Flush()
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		fl.Flush()
		io.WriteString(w, "data: [DONE]\n\n")
		fl.Flush()
	})
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	t.Setenv("STONE_LLAMA_UPSTREAM", ts.URL)
	origDiscover := discoverCheckoutFn
	discoverCheckoutFn = func(string) string { return "" }
	t.Cleanup(func() { discoverCheckoutFn = origDiscover })

	code, out, errb := run("run", "typed-alias", "-p", "hi")
	if code != 0 {
		t.Fatalf("code = %d, out = %q, err = %q", code, out, errb)
	}
	if !strings.Contains(out, `asking for "typed-alias" directly`) {
		t.Errorf("no unreadable-note: %q", out)
	}
	mu.Lock()
	defer mu.Unlock()
	if chatModel != "typed-alias" {
		t.Errorf("chat model = %q, want the typed name", chatModel)
	}
}

// Defect 2: `serve <model>` accepts an optional positional. Attach mode
// prints one ignored-argument note (no fake load); flag errors keep usage.
func TestServePositionalModelAttachNoteAndUsage(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	isolateConfig(t)
	// (a) attach + model: note first, then attach attempt ends at the dead port
	code, out, errb := run("serve", "mymodel", "--attach", "127.0.0.1:9")
	if code != 1 {
		t.Errorf("attach-dead code = %d, out = %q, err = %q", code, out, errb)
	}
	if !strings.Contains(out, `note: model "mymodel" ignored in attach mode — the upstream owns loading`) {
		t.Errorf("missing attach-ignored note: %q", out)
	}
	if !strings.Contains(errb, "not reachable") {
		t.Errorf("attach failure missing: %q", errb)
	}
	// (b) unknown flag and a second positional still print usage
	code, _, errb = run("serve", "--bogus")
	if code != 2 || !strings.Contains(errb, "usage: stone-llama serve [<model>]") {
		t.Errorf("flag error: code = %d, err = %q", code, errb)
	}
	code, _, errb = run("serve", "a", "b")
	if code != 2 || !strings.Contains(errb, "usage: stone-llama serve") {
		t.Errorf("second positional: code = %d, err = %q", code, errb)
	}
}

// Defect 2, no-runtime branch: `serve <model>` with neither runtime nor
// upstream ends in exactly one remedy block (attach/adopt/setup) — and
// spawns nothing (spawn refuses at the missing venv python).
func TestServeModelNoRuntimeOneRemedyBlock(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	isolateConfig(t)
	code, out, errb := run("serve", "mymodel")
	if code != 1 {
		t.Fatalf("code = %d, out = %q, err = %q", code, out, errb)
	}
	if n := strings.Count(errb, "remedies, least friction first"); n != 1 {
		t.Errorf("remedy block count = %d, want 1:\n%s", n, errb)
	}
	for _, want := range []string{"--attach", "setup --adopt", "stone-llama setup"} {
		if !strings.Contains(errb, want) {
			t.Errorf("remedy missing %q:\n%s", want, errb)
		}
	}
}

// stop must never print "not running" while the configured port
// answers the healthz marker but daemon.json has not landed yet (the
// bind→write window): the holder is named and the exit code is 1.
func TestStopReportsMarkerHolderWithoutState(t *testing.T) {
	isolateConfig(t)
	t.Setenv("XDG_DATA_HOME", t.TempDir())
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
	t.Setenv("STONE_LLAMA_PORT", portStr)

	code, out, errb := run("stop")
	if code == 0 && strings.Contains(out, "not running") {
		t.Fatalf("stop claimed not-running while the port answers the marker: out=%q err=%q", out, errb)
	}
	if code != 1 {
		t.Fatalf("code = %d, want 1; out=%q err=%q", code, out, errb)
	}
	if !strings.Contains(errb, "serving") {
		t.Errorf("holder not named: %q", errb)
	}
}

// A daemon that spawned but has not bound yet (spawn lock held, port
// free) is reported as starting: stop exits 1 and says so — never
// "not running" while its boot is in flight.
func TestStopReportsStartingDaemon(t *testing.T) {
	isolateConfig(t)
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	t.Setenv("STONE_LLAMA_PORT", freeLocalPort(t))
	prev := stopStateWait
	stopStateWait = 50 * time.Millisecond
	t.Cleanup(func() { stopStateWait = prev })

	dataDir := config.DataDir()
	if err := os.MkdirAll(dataDir, 0o700); err != nil {
		t.Fatal(err)
	}
	lk, err := os.Create(filepath.Join(dataDir, "stone-llama.lock"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { lk.Close() })
	if err := syscall.Flock(int(lk.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		t.Fatalf("hold spawn lock: %v", err)
	}

	code, out, errb := run("stop")
	if code == 0 && strings.Contains(out, "not running") {
		t.Fatalf("stop claimed not-running during boot: out=%q err=%q", out, errb)
	}
	if code != 1 {
		t.Fatalf("code = %d, want 1; out=%q err=%q", code, out, errb)
	}
	if !strings.Contains(errb, "starting") {
		t.Errorf("starting daemon not named: %q", errb)
	}
}

// B: rm must refuse to delete the model the running daemon has loaded
// (serve.LoadedModel was a promised-but-dead guard); --force bypasses it.
func TestRmRefusesLoadedModelUnlessForced(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("STONE_LLAMA_CONFIG", "")
	t.Setenv("XDG_CONFIG_HOME", "")
	modelsDir := filepath.Join(t.TempDir(), "models")
	t.Setenv("STONE_LLAMA_MODELS_DIR", modelsDir)
	t.Setenv("XDG_DATA_HOME", t.TempDir())

	code, _, errOut := run("rm")
	if code != 2 || !strings.Contains(errOut, "usage: stone-llama rm <model> [--force]") {
		t.Errorf("usage: code/err = %d/%q", code, errOut)
	}

	// an installed model the daemon state reports as loaded
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
	if code, _, errOut := run("import", ext, "--name", "ext-model"); code != 0 {
		t.Fatalf("import: code/err = %d/%q", code, errOut)
	}
	dataDir := config.DataDir()
	if err := os.MkdirAll(dataDir, 0o700); err != nil {
		t.Fatal(err)
	}
	state := `{"pid":4242,"host":"127.0.0.1","port":5111,"model":"ext-model"}`
	if err := os.WriteFile(filepath.Join(dataDir, "daemon.json"), []byte(state), 0o600); err != nil {
		t.Fatal(err)
	}

	code, out, errOut := run("rm", "ext-model")
	if code != 1 {
		t.Fatalf("rm loaded model: code = %d, want 1; out/err = %q/%q", code, out, errOut)
	}
	for _, want := range []string{`"ext-model" is loaded by the running daemon`, "(pid 4242)", "--force"} {
		if !strings.Contains(errOut, want) {
			t.Errorf("refusal missing %q: %q", want, errOut)
		}
	}
	if _, err := os.Lstat(filepath.Join(modelsDir, "ext-model")); err != nil {
		t.Errorf("model removed despite the guard: %v", err)
	}

	code, out, errOut = run("rm", "ext-model", "--force")
	if code != 0 || !strings.Contains(out, "removed ext-model") {
		t.Errorf("force rm: code/out/err = %d/%q/%q", code, out, errOut)
	}
	if _, err := os.Lstat(filepath.Join(modelsDir, "ext-model")); !os.IsNotExist(err) {
		t.Errorf("link still present after --force: %v", err)
	}
}

// C: Ctrl-C during a streaming generation cancels the in-flight request
// and returns to the prompt — it must not kill the REPL (signal 2). The
// chat handler blocks until the client aborts, so the SIGINT lands with a
// request in flight and the handler registered. On the base tree there was
// NO signal.Notify around the ask (grep: cli.go's only signal use was
// runServe's NotifyContext) — this SIGINT killed the process.
func TestRunSigintMidStreamReturnsToPrompt(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("SIGINT semantics are unix-only")
	}
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	isolateConfig(t)
	t.Setenv("STONE_LLAMA_UPSTREAM_KEY", "")
	t.Setenv("STONE_LLAMA_UPSTREAM_KEY_FILE", "")
	started := make(chan struct{})
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, `{"object":"list","data":[{"id":"m"}]}`)
	})
	mux.HandleFunc("/v1/model", func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, `{"id":"m"}`)
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		fl := w.(http.Flusher)
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n")
		fl.Flush()
		close(started)
		<-r.Context().Done() // blocked until the CLI cancels (Ctrl-C)
	})
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	t.Setenv("STONE_LLAMA_UPSTREAM", ts.URL)

	type res struct {
		code        int
		out, errOut string
	}
	done := make(chan res, 1)
	go func() {
		var out, errb bytes.Buffer
		code := Run([]string{"run", "m"}, "v9.9.9", strings.NewReader("hi\n/bye\n"), &out, &errb)
		done <- res{code, out.String(), errb.String()}
	}()
	select {
	case <-started:
	case <-time.After(10 * time.Second):
		t.Fatal("chat request never arrived")
	}
	if err := syscall.Kill(os.Getpid(), syscall.SIGINT); err != nil {
		t.Fatal(err)
	}
	select {
	case r := <-done:
		if r.code != 0 {
			t.Errorf("code = %d, want 0 (REPL survived Ctrl-C); err = %q", r.code, r.errOut)
		}
		if strings.Contains(r.errOut, "stone-llama run: signal") {
			t.Errorf("SIGINT surfaced as an ask error: %q", r.errOut)
		}
		if n := strings.Count(r.out, ">>> "); n < 2 {
			t.Errorf("prompt not re-printed after the interrupt (%d prompts): %q", n, r.out)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("run did not return after SIGINT — Ctrl-C killed the flow")
	}
}

// D: login hides paste echo with stty; a SIGINT used to kill the process
// before the deferred restore ran, leaving the shell echo-less. The
// handler must restore echo then exit 130. A subprocess plus a fake stty
// on PATH proves the SIGINT path calls the restore (no PTY harness): the
// log must read -echo then echo, and the child exit status must be 130.
func TestLoginSigintRestoresEcho(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("stty is unix-only")
	}
	if os.Getenv("SL_LOGIN_SIGINT_CHILD") == "1" {
		// child: the parent holds the pipe open, so the token read blocks
		Run([]string{"login"}, "v9.9.9", os.Stdin, io.Discard, io.Discard)
		os.Exit(3) // the handler should have exited 130 first
	}
	bin := t.TempDir()
	if err := os.WriteFile(filepath.Join(bin, "stty"),
		[]byte("#!/bin/sh\necho \"$@\" >> \"$STTY_LOG\"\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	logPath := filepath.Join(t.TempDir(), "stty.log")
	pr, pw, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer pw.Close() // never written: the child's token read blocks
	cmd := exec.Command(os.Args[0], "-test.run=TestLoginSigintRestoresEcho")
	cmd.Env = append(os.Environ(),
		"SL_LOGIN_SIGINT_CHILD=1",
		"PATH="+bin+":"+os.Getenv("PATH"),
		"STTY_LOG="+logPath,
		"HOME="+t.TempDir(),
		"XDG_DATA_HOME="+t.TempDir(),
	)
	cmd.Stdin = pr
	cmd.Stdout, cmd.Stderr = io.Discard, io.Discard
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	pr.Close()
	// "-echo" in the log proves echo is hidden — and, with the handler
	// registered before -echo, that the SIGINT handler is live before we send it.
	waitLog := func(want string) error {
		deadline := time.Now().Add(10 * time.Second)
		for time.Now().Before(deadline) {
			if b, rerr := os.ReadFile(logPath); rerr == nil && strings.Contains(string(b), want) {
				return nil
			}
			time.Sleep(20 * time.Millisecond)
		}
		return fmt.Errorf("%q never appeared in %s", want, logPath)
	}
	if err := waitLog("-echo"); err != nil {
		_ = cmd.Process.Kill()
		cmd.Wait()
		t.Fatal(err)
	}
	if err := cmd.Process.Signal(syscall.SIGINT); err != nil {
		t.Fatal(err)
	}
	werr := make(chan error, 1)
	go func() { werr <- cmd.Wait() }()
	select {
	case err := <-werr:
		if err == nil || !strings.Contains(err.Error(), "exit status 130") {
			t.Errorf("child exit = %v, want exit status 130 (SIGINT restore path)", err)
		}
	case <-time.After(10 * time.Second):
		_ = cmd.Process.Kill()
		t.Fatal("child never exited after SIGINT — restore handler missing")
	}
	b, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Fields(string(b))
	if len(lines) != 2 || lines[0] != "-echo" || lines[1] != "echo" {
		t.Errorf("stty calls = %q, want [-echo echo] (hide, then restore on SIGINT)", lines)
	}
}

// D: real PTY test — proves ECHO bit is restored after SIGINT, not just
// that stty was called. Parent opens /dev/ptmx, spawns a child whose
// stdin is the slave side, watches TIOCGPTN for the slave's ECHO bit
// via TCGETS, sends SIGINT, asserts ECHO is back and exit status 130.
func TestLoginSigintRestoresEchoOnPTY(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("pty is not available on Windows")
	}
	if os.Getenv("STONE_LLAMA_PTY_CHILD") == "1" {
		// child: real login against a pty slave (blocks on token read)
		Run([]string{"login"}, "v9.9.9", os.Stdin, io.Discard, io.Discard)
		os.Exit(130)
	}
	// open master + slave
	master, err := os.OpenFile("/dev/ptmx", os.O_RDWR, 0)
	if err != nil {
		t.Skipf("open ptmx: %v", err)
	}
	defer master.Close()
	var unlock int32
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, master.Fd(),
		uintptr(syscall.TIOCSPTLCK), uintptr(unsafe.Pointer(&unlock))); errno != 0 {
		t.Fatalf("TIOCSPTLCK: %v", errno)
	}
	var ptn uint32
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, master.Fd(),
		uintptr(syscall.TIOCGPTN), uintptr(unsafe.Pointer(&ptn))); errno != 0 {
		t.Fatalf("TIOCGPTN: %v", errno)
	}
	slavePath := fmt.Sprintf("/dev/pts/%d", ptn)
	slave, err := os.OpenFile(slavePath, os.O_RDWR|syscall.O_NOCTTY, 0)
	if err != nil {
		t.Fatalf("open slave %s: %v", slavePath, err)
	}
	defer slave.Close()
	// parent keeps its own slave handle to read the ECHO bit
	readEcho := func() bool {
		var term syscall.Termios
		if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, slave.Fd(),
			uintptr(syscall.TCGETS), uintptr(unsafe.Pointer(&term))); errno != 0 {
			t.Fatalf("TCGETS: %v", errno)
		}
		return term.Lflag&syscall.ECHO != 0
	}
	// spawn child with slave as its stdin
	cmd := exec.Command(os.Args[0], "-test.run=TestLoginSigintRestoresEchoOnPTY")
	cmd.Env = append(os.Environ(), "STONE_LLAMA_PTY_CHILD=1")
	cmd.Stdin = slave
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	werr := make(chan error, 1)
	go func() { werr <- cmd.Wait() }()
	// wait for ECHO off (child reached stty -echo)
	deadline := time.After(10 * time.Second)
	for readEcho() {
		select {
		case <-deadline:
			_ = cmd.Process.Kill()
			t.Fatal("ECHO never went off — stty -echo not executed")
		default:
			time.Sleep(5 * time.Millisecond)
		}
	}
	// SIGINT mid-read: handler must restore ECHO then exit 130
	if err := cmd.Process.Signal(syscall.SIGINT); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-werr:
		if err == nil || !strings.Contains(err.Error(), "exit status 130") {
			t.Errorf("child exit = %v, want exit status 130", err)
		}
	case <-time.After(10 * time.Second):
		_ = cmd.Process.Kill()
		t.Fatal("child never exited after SIGINT")
	}
	if !readEcho() {
		t.Error("ECHO bit still off after SIGINT — termios not restored")
	}
}
