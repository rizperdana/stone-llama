package serve

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
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

	"github.com/rizperdana/stone-llama/internal/autofit"
	"github.com/rizperdana/stone-llama/internal/doctor"
)

// fakeUpstream answers the three backend-contract surfaces:
// GET /v1/models (ready), GET /v1/model (attach display), POST
// /v1/completions (slow SSE so buffering becomes visible). It also
// asserts the injected upstream Bearer on every call.
func fakeUpstream(t *testing.T, wantKey string, chunks int, delay time.Duration) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	check := func(w http.ResponseWriter, r *http.Request) bool {
		if got := r.Header.Get("Authorization"); got != "Bearer "+wantKey {
			t.Errorf("%s: upstream Authorization = %q, want Bearer %s (Conn.UpKeyFile injection)", r.URL.Path, got, wantKey)
			w.WriteHeader(http.StatusUnauthorized)
			return false
		}
		return true
	}
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"object":"list","data":[]}`)
	})
	mux.HandleFunc("/v1/model", func(w http.ResponseWriter, r *http.Request) {
		if check(w, r) {
			io.WriteString(w, `{"id":"fake-model"}`)
		}
	})
	mux.HandleFunc("/v1/completions", func(w http.ResponseWriter, r *http.Request) {
		if !check(w, r) {
			return
		}
		fl := w.(http.Flusher)
		for i := range chunks {
			fmt.Fprintf(w, "data: chunk%d\n\n", i)
			fl.Flush()
			time.Sleep(delay)
		}
		io.WriteString(w, "data: [DONE]\n\n")
		fl.Flush()
	})
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	return ts
}

func writeUpstreamKey(t *testing.T, key string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "upstream_key")
	if err := os.WriteFile(p, []byte(key+"\n"), 0600); err != nil {
		t.Fatal(err)
	}
	return p
}

func waitReady(t *testing.T, dataDir string, errCh <-chan error) Status {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	var last error
	for time.Now().Before(deadline) {
		select {
		case err := <-errCh:
			t.Fatalf("Serve exited before ready: %v", err)
		default:
		}
		st, err := Query(dataDir)
		if err == nil && st.Ready {
			return st
		}
		last = err
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("daemon not ready within 5s (last query error: %v)", last)
	return Status{}
}

// TestAttachStreamsZeroBuffer is the M5 attach E2E: schemeless host:port
// accepted (bug 3), upstream key injected from the file, status/mode/
// model correct, and chunks must arrive while the upstream is still
// producing — a buffering proxy collapses the arrival times.
func TestAttachStreamsZeroBuffer(t *testing.T) {
	const delay = 150 * time.Millisecond
	const chunks = 3
	ts := fakeUpstream(t, "sekret", chunks, delay)
	dataDir := t.TempDir()
	keyFile := writeUpstreamKey(t, "sekret")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errCh := make(chan error, 1)
	go func() {
		errCh <- Serve(ctx, Options{
			Host:            "127.0.0.1",
			DataDir:         dataDir,
			Attach:          strings.TrimPrefix(ts.URL, "http://"), // schemeless on purpose
			UpstreamKeyFile: keyFile,
		})
	}()

	st := waitReady(t, dataDir, errCh)
	if st.Mode != "attach" || st.Model != "fake-model" {
		t.Fatalf("status = %+v, want mode=attach model=fake-model", st)
	}

	req, err := http.NewRequest(http.MethodPost, "http://"+st.Addr()+"/v1/completions",
		strings.NewReader(`{"model":"m","stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("completions status = %d", resp.StatusCode)
	}

	t0 := time.Now()
	var first, last time.Time
	var body bytes.Buffer
	br := bufio.NewReader(resp.Body)
	for {
		line, err := br.ReadString('\n')
		body.WriteString(line)
		if strings.HasPrefix(line, "data:") {
			now := time.Now()
			if first.IsZero() {
				first = now
			}
			last = now
		}
		if err != nil || strings.Contains(line, "[DONE]") {
			break
		}
	}
	elapsed := time.Since(t0)
	if first.IsZero() {
		t.Fatalf("no streamed data arrived; body:\n%s", body.String())
	}
	if first.Sub(t0) > 400*time.Millisecond {
		t.Errorf("first chunk arrived %s after request — response buffered?", first.Sub(t0))
	}
	if last.Sub(first) < 300*time.Millisecond {
		t.Errorf("chunk span %s — arrivals collapsed (buffering)", last.Sub(first))
	}
	wantMin := time.Duration(chunks-1) * delay // 300ms between first and last flush
	if last.Sub(first) < wantMin-50*time.Millisecond {
		t.Errorf("chunk span %s < %s — chunks not passed through in real time", last.Sub(first), wantMin)
	}
	if elapsed < 400*time.Millisecond {
		t.Errorf("total %s < %s — stream completed too early", elapsed, 300*time.Millisecond)
	}
	for _, want := range []string{"chunk0", "chunk2", "[DONE]"} {
		if !strings.Contains(body.String(), want) {
			t.Errorf("stream missing %q; body:\n%s", want, body.String())
		}
	}

	cancel()
	if err := <-errCh; err != nil {
		t.Errorf("Serve returned %v after clean cancel, want nil", err)
	}
}

// TestStatusRequiresBearerOffLoopback: non-loopback bind → /-/status
// demands the bearer (loopback stays open), Query uses the state token.
func TestStatusRequiresBearerOffLoopback(t *testing.T) {
	ts := fakeUpstream(t, "sekret", 0, 0)
	dataDir := t.TempDir()
	keyFile := writeUpstreamKey(t, "sekret")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errCh := make(chan error, 1)
	go func() {
		errCh <- Serve(ctx, Options{
			Host:            "0.0.0.0",
			DataDir:         dataDir,
			Attach:          ts.URL,
			UpstreamKeyFile: keyFile,
		})
	}()

	st := waitReady(t, dataDir, errCh)
	state, err := ReadState(dataDir)
	if err != nil || state.Token == "" {
		t.Fatalf("no downstream token generated for non-loopback bind: state=%+v err=%v", state, err)
	}
	// no bearer → 401
	resp, err := http.Get("http://" + st.Addr() + statusPath)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("unauthenticated /-/status = %d, want 401", resp.StatusCode)
	}
	// with bearer (Query path) → 200, same values
	again, err := Query(dataDir)
	if err != nil {
		t.Fatalf("Query with token: %v", err)
	}
	if again.Mode != "attach" || !again.Ready {
		t.Errorf("Query = %+v, want mode=attach ready", again)
	}

	cancel()
	<-errCh
}

// TestRewriteOOMEnriches: an upstream OOM error gains the levers and
// keeps the original message; anything else passes through untouched.
func TestRewriteOOMEnriches(t *testing.T) {
	d := &daemon{opts: Options{Stderr: io.Discard, contenders: func() ([]doctor.Contender, error) { return nil, nil }}, conn: Conn{Name: "TabbyAPI test"}}
	d.fit = &autofit.Result{Ctx: 32768, Mode: "Q8", LargestCtx: 65536}
	newResp := func(status int, body string) *http.Response {
		return &http.Response{
			StatusCode: status,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(body)),
		}
	}

	oom := newResp(500, `{"error":{"message":"CUDA out of memory. Tried to allocate 512.00 MiB (GPU 0)","code":500}}`)
	if err := d.rewriteOOM(oom); err != nil {
		t.Fatalf("rewriteOOM: %v", err)
	}
	b, _ := io.ReadAll(oom.Body)
	for _, want := range []string{"CUDA out of memory", "--ctx", "--cache-mode", "largest"} {
		if !strings.Contains(string(b), want) {
			t.Errorf("OOM response missing %q:\n%s", want, b)
		}
	}

	plain := `{"error":{"message":"model not found"}}`
	other := newResp(404, plain)
	if err := d.rewriteOOM(other); err != nil {
		t.Fatalf("rewriteOOM(non-OOM): %v", err)
	}
	got, _ := io.ReadAll(other.Body)
	if string(got) != plain {
		t.Errorf("non-OOM body rewritten:\ngot  %s\nwant %s", got, plain)
	}
}

// TestContentionErrorVerbatim: a crash while a foreign process holds
// the GPU surfaces doctor's message verbatim, with the exit reason.
func TestContentionErrorVerbatim(t *testing.T) {
	contenders := []doctor.Contender{{PID: 4242, Name: "python", MiB: 1024}}
	want := doctor.ContentionError(contenders)
	if want == nil {
		t.Fatal("fixture should produce a contention error")
	}
	d := &daemon{
		opts: Options{Stderr: io.Discard, contenders: func() ([]doctor.Contender, error) { return contenders, nil }},
		conn: Conn{Name: "TabbyAPI test"},
	}
	d.st.ChildPID = 7 // contenders PID 4242 is foreign
	err := d.classifyCrash(errors.New("signal: killed"))
	if err == nil {
		t.Fatal("classifyCrash = nil, want fatal contention")
	}
	if !strings.Contains(err.Error(), want.Error()) {
		t.Errorf("contention text not verbatim:\ngot  %s\nwant %s", err.Error(), want.Error())
	}
	if !strings.Contains(err.Error(), "signal: killed") {
		t.Errorf("exit reason missing: %s", err.Error())
	}
}

// TestSuperviseFailsFastAfterLoad: post-load crash is fatal — no restart.
func TestSuperviseFailsFastAfterLoad(t *testing.T) {
	d := &daemon{
		opts:    Options{Stderr: io.Discard, contenders: func() ([]doctor.Contender, error) { return nil, nil }},
		conn:    Conn{Name: "TabbyAPI test"},
		logPath: filepath.Join(t.TempDir(), "none.log"),
	}
	d.st.Model = "m"
	c := &child{wait: make(chan error, 1)}
	c.wait <- errors.New("exit status 1")
	err := d.supervise(context.Background(), c)
	if err == nil || !strings.Contains(err.Error(), "crashed while a model was loaded") {
		t.Fatalf("supervise = %v, want fail-fast after load", err)
	}
}

// TestServeSpawnFailureNoState: supervised boot that cannot spawn fails
// fast and leaves no daemon.json behind.
func TestServeSpawnFailureNoState(t *testing.T) {
	p, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	dataDir := t.TempDir()
	serr := Serve(context.Background(), Options{
		Host:      "127.0.0.1",
		Port:      p,
		DataDir:   dataDir,
		ModelsDir: t.TempDir(),
		spawn: func(string, string, string, int) (*child, error) {
			return nil, errors.New("runtime missing")
		},
	})
	if serr == nil || !strings.Contains(serr.Error(), "runtime missing") {
		t.Fatalf("Serve = %v, want spawn failure propagated", serr)
	}
	if _, err := ReadState(dataDir); !errors.Is(err, ErrNoDaemon) {
		t.Errorf("daemon.json written despite failed start: %v", err)
	}
}

// TestServeForeignPortHint: a foreign process holding the configured
// port fails fatally naming the exact --port remedy with a probed-free
// port (ARCHITECTURE §2 "foreign process → fatal with --port hint").
func TestServeForeignPortHint(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	_, portStr, err := net.SplitHostPort(ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatal(err)
	}
	serr := Serve(context.Background(), Options{
		Host:      "127.0.0.1",
		Port:      port,
		DataDir:   t.TempDir(),
		ModelsDir: t.TempDir(),
	})
	if !errors.Is(serr, ErrPortInUse) {
		t.Fatalf("Serve = %v, want ErrPortInUse", serr)
	}
	msg := serr.Error()
	for _, want := range []string{
		fmt.Sprintf("127.0.0.1:%d", port),
		"stone-llama serve --port ",
	} {
		if !strings.Contains(msg, want) {
			t.Errorf("foreign-port refusal missing %q: %q", want, msg)
		}
	}
	if strings.Contains(msg, "stone-llama stop") {
		t.Errorf("foreign-port refusal suggests our own stop remedy: %q", msg)
	}
}

// TestServeOwnDaemonHint: our own daemon holding the port is a
// different refusal — reuse or stop, never a "pick another port"
// suggestion (the remedies must stay distinct).
func TestServeOwnDaemonHint(t *testing.T) {
	ts := markerHealthzServer(t)
	port := serverPort(t, ts)
	serr := Serve(context.Background(), Options{
		Host:      "127.0.0.1",
		Port:      port,
		DataDir:   t.TempDir(),
		ModelsDir: t.TempDir(),
	})
	if !errors.Is(serr, ErrAlreadyServing) {
		t.Fatalf("Serve = %v, want ErrAlreadyServing", serr)
	}
	msg := serr.Error()
	if !strings.Contains(msg, fmt.Sprintf("127.0.0.1:%d", port)) || !strings.Contains(msg, "stone-llama stop") {
		t.Errorf("own-daemon refusal missing addr or stop remedy: %q", msg)
	}
	if strings.Contains(msg, "--port") {
		t.Errorf("own-daemon refusal suggests the foreign-process remedy: %q", msg)
	}
}

// TestStopRefusesForeignPid: recycled pid is never signalled; stale
// state is dropped.
func TestStopRefusesForeignPid(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("stop is refused on windows")
	}
	proc := exec.Command("sleep", "30")
	if err := proc.Start(); err != nil {
		t.Skipf("needs sleep(1): %v", err)
	}
	t.Cleanup(func() {
		proc.Process.Kill()
		proc.Wait()
	})
	dataDir := t.TempDir()
	if err := WriteState(dataDir, State{PID: proc.Process.Pid, Host: "127.0.0.1", Port: 1}); err != nil {
		t.Fatal(err)
	}
	if err := Stop(dataDir, time.Second); err != nil {
		t.Fatalf("Stop(foreign pid): %v", err)
	}
	if err := proc.Process.Signal(syscall.Signal(0)); err != nil {
		t.Errorf("foreign process was signalled: %v", err)
	}
	if _, err := ReadState(dataDir); !errors.Is(err, ErrNoDaemon) {
		t.Errorf("stale state not dropped: %v", err)
	}
}

// TestEnsureDaemonNotRunning: start=false → plain not-running error.
func TestEnsureDaemonNotRunning(t *testing.T) {
	// The diagnosis now probes the configured port (A3: never claim
	// "not running" while a listener holds it) — pin it to a port
	// nothing listens on so the plain message is what's asserted.
	p, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("STONE_LLAMA_PORT", strconv.Itoa(p))
	_, err = EnsureDaemon(t.TempDir(), false, time.Second)
	if err == nil || !strings.Contains(err.Error(), "no stone-llama daemon is running") {
		t.Fatalf("EnsureDaemon = %v, want not-running error", err)
	}
}

// TestStartDetachedMissingExe: detached start seam reports failure
// instead of hanging.
func TestStartDetachedMissingExe(t *testing.T) {
	if _, err := StartDetached(filepath.Join(t.TempDir(), "nonexistent"), t.TempDir(), nil); err == nil {
		t.Fatal("StartDetached missing exe = nil, want error")
	}
}

// --- H1/H2/H3 + cleanup regressions (details in report.md) ---

// markerHealthzServer is a stand-in daemon listener: /healthz answers
// with the X-Stone-Llama marker and /-/status answers 200 — exactly
// what the identification gates in Stop and EnsureDaemon look for.
func markerHealthzServer(t *testing.T) *httptest.Server {
	t.Helper()
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
	return ts
}

// serverPort extracts the bound port of an httptest server.
func serverPort(t *testing.T, ts *httptest.Server) int {
	t.Helper()
	_, p, err := net.SplitHostPort(strings.TrimPrefix(ts.URL, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	n, err := strconv.Atoi(p)
	if err != nil {
		t.Fatal(err)
	}
	return n
}

// standIn is a sleep process whose cmdline reads as a given
// stone-llama invocation (sleep parses only its operands, argv0 free).
// It owns the process's Wait, so liveness checks are zombie-free:
// kill(pid, 0) still succeeds on an exited-but-unreaped child.
type standIn struct {
	pid    int
	exited <-chan error
}

func daemonStandIn(t *testing.T, argv0 string) *standIn {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("stop is refused on windows")
	}
	proc := exec.Command("sleep", "30")
	proc.Args = []string{argv0, "30"}
	if err := proc.Start(); err != nil {
		t.Skipf("needs sleep(1): %v", err)
	}
	waitExecCmdline(t, proc.Process.Pid, argv0)
	exited := make(chan error, 1)
	go func() { exited <- proc.Wait() }()
	t.Cleanup(func() { proc.Process.Kill() }) // Wait belongs to the goroutine above
	return &standIn{pid: proc.Process.Pid, exited: exited}
}

// waitExecCmdline blocks until /proc/<pid>/cmdline carries want. Right after
// Start returns, the child may not have execve'd yet — /proc then shows OUR
// argv, and Stop's identity gate would read the stand-in as a foreign pid,
// drop the state without signalling, and flake under load (measured: ~9% of
// starts read pre-exec). Other platforms: best effort, no /proc.
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

// writeCorruptState writes unparseable daemon.json whose bytes still
// carry pid/host/port/token — what a truncated or hand-mangled write
// leaves behind (WriteState's own atomic rename never does).
func writeCorruptState(t *testing.T, dataDir string, pid, port int) {
	t.Helper()
	if err := os.MkdirAll(dataDir, 0o700); err != nil {
		t.Fatal(err)
	}
	b := fmt.Sprintf(`{"child_pid":7,"pid":%d,"host":"127.0.0.1","port":%d,"token":"tok-abc"`, pid, port)
	if err := os.WriteFile(StatePath(dataDir), []byte(b), 0o600); err != nil {
		t.Fatal(err)
	}
}

// isolateConfig points config at a file-less scratch location so Stop's
// runtime-dir resolution can never reach a real machine config — a
// test must never run cleanup against a production runtime dir.
func isolateConfig(t *testing.T) {
	t.Helper()
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	if v, ok := os.LookupEnv("STONE_LLAMA_CONFIG"); ok {
		os.Unsetenv("STONE_LLAMA_CONFIG")
		t.Cleanup(func() { os.Setenv("STONE_LLAMA_CONFIG", v) })
	}
}

// H1: a child that exits 0 before readiness must fail startReady, and
// the consumed exit must be pushed back on c.wait. The old code
// returned (c, nil) for any rerr==nil — reporting success for a dead
// child — and dropped the already-read exit, hanging shutdownChild and
// supervise forever on their receive.
func TestStartReadyCleanExitBeforeReadinessFails(t *testing.T) {
	var kids []*child
	d := &daemon{
		opts: Options{
			Stderr:       io.Discard,
			ReadyTimeout: 50 * time.Millisecond,
			RetryDelay:   time.Millisecond,
			contenders:   func() ([]doctor.Contender, error) { return nil, nil },
			probe:        func(context.Context, int) error { return errors.New("never ready") },
			spawn: func(string, string, string, int) (*child, error) {
				c := &child{wait: make(chan error, 1), port: 1}
				kids = append(kids, c)
				go func() {
					time.Sleep(5 * time.Millisecond)
					c.wait <- nil // exit status 0 before readiness
				}()
				return c, nil
			},
		},
		conn:    Conn{Name: "TabbyAPI test"},
		logPath: filepath.Join(t.TempDir(), "none.log"),
	}
	c, err := d.startReady(context.Background())
	if err == nil {
		t.Fatalf("startReady = %v (child %v): clean exit before readiness must fail, not report success", err, c)
	}
	if strings.Contains(err.Error(), "(<nil>)") {
		t.Errorf("failure forensics print (nil): %v", err)
	}
	last := kids[len(kids)-1]
	select {
	case got := <-last.wait:
		if got == nil {
			t.Error("pushback value nil: supervise cannot tell an exit from nothing")
		}
	default:
		t.Fatal("consumed exit never pushed back: shutdownChild would block forever on c.wait")
	}
}

// H2: a process whose cmdline merely contains the substring
// "stone-llama" but is not `stone-llama serve` must never be signalled
// (the old gate was strings.Contains(cmd, "stone-llama")); its stale
// state is dropped instead.
func TestStopIgnoresNonServeStoneProcess(t *testing.T) {
	isolateConfig(t)
	proc := exec.Command("sleep", "30")
	proc.Args = []string{"stone-llama-sleeper", "30"}
	if err := proc.Start(); err != nil {
		t.Skipf("needs sleep(1): %v", err)
	}
	t.Cleanup(func() { proc.Process.Kill() }) // Wait belongs to the goroutine below
	dataDir := t.TempDir()
	if err := WriteState(dataDir, State{PID: proc.Process.Pid, Host: "127.0.0.1", Port: 1}); err != nil {
		t.Fatal(err)
	}
	exited := make(chan error, 1)
	go func() { exited <- proc.Wait() }() // reap: Signal(0) sees zombies
	if err := Stop(dataDir, time.Second); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	select {
	case <-exited:
		t.Error("non-serve process was signalled (bare substring gate)")
	case <-time.After(300 * time.Millisecond):
	}
	if _, err := ReadState(dataDir); !errors.Is(err, ErrNoDaemon) {
		t.Errorf("stale state not dropped: %v", err)
	}
}

// H2: even a perfect `stone-llama serve` cmdline is not enough — the
// recorded port must answer the healthz marker. Unconfirmed identity →
// refuse to signal, keep the state, say why.
func TestStopRefusesUnconfirmedDaemonCmdline(t *testing.T) {
	isolateConfig(t)
	s := daemonStandIn(t, "stone-llama serve")
	p, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	dataDir := t.TempDir()
	if err := WriteState(dataDir, State{PID: s.pid, Host: "127.0.0.1", Port: p}); err != nil {
		t.Fatal(err)
	}
	err = Stop(dataDir, time.Second)
	if err == nil || !strings.Contains(err.Error(), "refusing to signal") {
		t.Fatalf("Stop = %v, want refusal without confirmation", err)
	}
	if err := syscall.Kill(s.pid, 0); err != nil {
		t.Errorf("unconfirmed process was signalled: %v", err)
	}
	if _, rerr := ReadState(dataDir); rerr != nil {
		t.Errorf("state dropped despite refusal: %v", rerr)
	}
}

// H2 positive path: full identity (argv1 `serve` + healthz marker) →
// SIGTERM lands, exit observed, state dropped.
func TestStopSignalsConfirmedDaemon(t *testing.T) {
	isolateConfig(t)
	s := daemonStandIn(t, "stone-llama serve")
	ts := markerHealthzServer(t)
	dataDir := t.TempDir()
	if err := WriteState(dataDir, State{PID: s.pid, Host: "127.0.0.1", Port: serverPort(t, ts)}); err != nil {
		t.Fatal(err)
	}
	if err := Stop(dataDir, 3*time.Second); err != nil {
		t.Fatalf("Stop(confirmed daemon): %v", err)
	}
	select {
	case <-s.exited:
	case <-time.After(2 * time.Second):
		t.Error("daemon stand-in survived SIGTERM")
	}
	if _, err := ReadState(dataDir); !errors.Is(err, ErrNoDaemon) {
		t.Errorf("state not dropped after confirmed stop: %v", err)
	}
}

// H3 Case 1 (probe window): a listener that accepts but never answers
// is a daemon between net.Listen and srv.Serve — one failed probe must
// never delete its state. The old EnsureDaemon ran RemoveState after
// any single failed probe.
func TestEnsureDaemonKeepsStateWhenListenerUnconfirmed(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	port := ln.Addr().(*net.TCPAddr).Port
	dataDir := t.TempDir()
	if err := WriteState(dataDir, State{PID: os.Getpid(), Host: "127.0.0.1", Port: port}); err != nil {
		t.Fatal(err)
	}
	_, err = EnsureDaemon(dataDir, false, 2*time.Second)
	if err == nil {
		t.Fatal("EnsureDaemon = nil: an unconfirmed listener must not report a live daemon")
	}
	if !strings.Contains(err.Error(), "could not confirm") {
		t.Errorf("want could-not-confirm diagnosis, got: %v", err)
	}
	if _, rerr := ReadState(dataDir); rerr != nil {
		t.Fatalf("state deleted on a single failed probe (H3): %v", rerr)
	}
}

// H3 Case 2 (ps side): corrupt state that salvages a live daemon must
// still be reached — status shown, corruption reported, file intact.
// The old code reported "no stone-llama daemon is running" and hid the
// daemon behind the parse error.
func TestEnsureDaemonReachesLiveDaemonThroughCorruptState(t *testing.T) {
	ts := markerHealthzServer(t)
	port := serverPort(t, ts)
	dataDir := t.TempDir()
	writeCorruptState(t, dataDir, os.Getpid(), port)
	st, err := EnsureDaemon(dataDir, false, 2*time.Second)
	if err != nil {
		t.Fatalf("EnsureDaemon = %v: a live daemon behind corrupt state must be reached (H3)", err)
	}
	if st.Port != port {
		t.Errorf("salvaged state lost the port: got %d, want %d", st.Port, port)
	}
	if _, rerr := ReadState(dataDir); rerr == nil || !strings.Contains(rerr.Error(), "corrupt") {
		t.Errorf("corruption not reported: %v", rerr)
	}
	if _, serr := os.Stat(StatePath(dataDir)); serr != nil {
		t.Errorf("reader deleted the corrupt file: %v", serr)
	}
}

// H3: unparseable and unsalvageable state is diagnosed as corrupt and
// kept — never reported as a plain "not running", never deleted.
func TestEnsureDaemonReportsUnusableCorruptState(t *testing.T) {
	dataDir := t.TempDir()
	if err := os.WriteFile(StatePath(dataDir), []byte("not json at all"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err := EnsureDaemon(dataDir, false, time.Second)
	if err == nil || !strings.Contains(err.Error(), "corrupt") {
		t.Fatalf("EnsureDaemon = %v, want corrupt-state diagnosis", err)
	}
	if !strings.Contains(err.Error(), "kept") {
		t.Errorf("diagnosis must say the file was kept: %v", err)
	}
	if _, serr := os.Stat(StatePath(dataDir)); serr != nil {
		t.Errorf("corrupt file deleted: %v", serr)
	}
}

// H3 Case 2 (stop side): corrupt state + live listener + identified pid
// → stop still works (salvage, then the H2 gates) and clears the file
// after the verified shutdown. The old Stop bailed out of ReadState's
// parse error with exit 1 and never signalled anything.
func TestStopThroughCorruptState(t *testing.T) {
	isolateConfig(t)
	s := daemonStandIn(t, "stone-llama serve")
	ts := markerHealthzServer(t)
	dataDir := t.TempDir()
	writeCorruptState(t, dataDir, s.pid, serverPort(t, ts))
	if err := Stop(dataDir, 3*time.Second); err != nil {
		t.Fatalf("Stop through corrupt state: %v", err)
	}
	select {
	case <-s.exited:
	case <-time.After(2 * time.Second):
		t.Error("daemon stand-in survived stop-through-corrupt-state")
	}
	if _, err := ReadState(dataDir); !errors.Is(err, ErrNoDaemon) {
		t.Errorf("corrupt state not cleared after stop: %v", err)
	}
}

// Cleanup contract: a stop cycle removes exactly the generated files
// (secrets + lock + state) and never touches anything else under
// runtime/ — least of all an adopted runtime/venv symlink (its target
// and sentinel must survive verbatim) or a provisioned-style sibling
// venv dir. Fails if anyone ever adds a recursive wipe or a venv to
// the allow-list.
func TestStopCycleCleansGeneratedSparesVenvAndSiblings(t *testing.T) {
	isolateConfig(t)
	dataDir := t.TempDir()
	runtimeDir := filepath.Join(dataDir, "runtime")
	mk := func(parts ...string) string {
		p := filepath.Join(append([]string{runtimeDir}, parts...)...)
		if err := os.MkdirAll(filepath.Dir(p), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("s"), 0o600); err != nil {
			t.Fatal(err)
		}
		return p
	}
	upKey := mk("upstream_key")
	cfgYML := mk("tabby-config.yml")
	tokens := mk("tabbyAPI", "api_tokens.yml")
	realVenvPython := mk("real-venv", "bin", "python") // provisioned-style sibling
	if err := os.WriteFile(LockPath(dataDir), nil, 0o600); err != nil {
		t.Fatal(err)
	}

	// adopted venv: runtime/venv is a symlink into a user environment
	target := t.TempDir()
	sentinel := filepath.Join(target, "SENTINEL")
	if err := os.WriteFile(sentinel, []byte("keep"), 0o600); err != nil {
		t.Fatal(err)
	}
	venvLink := filepath.Join(runtimeDir, "venv")
	if err := os.Symlink(target, venvLink); err != nil {
		t.Skipf("needs symlink(2): %v", err)
	}

	// foreign pid → Stop's drop the stale state path runs cleanup
	proc := exec.Command("sleep", "30")
	proc.Args = []string{"unrelated-sleeper", "30"}
	if err := proc.Start(); err != nil {
		t.Skipf("needs sleep(1): %v", err)
	}
	t.Cleanup(func() {
		proc.Process.Kill()
		proc.Wait()
	})
	if err := WriteState(dataDir, State{PID: proc.Process.Pid, Host: "127.0.0.1", Port: 1}); err != nil {
		t.Fatal(err)
	}
	if err := Stop(dataDir, time.Second); err != nil {
		t.Fatalf("Stop: %v", err)
	}

	for _, p := range []string{upKey, cfgYML, tokens, LockPath(dataDir), StatePath(dataDir)} {
		if _, err := os.Lstat(p); !os.IsNotExist(err) {
			t.Errorf("generated file survives stop: %s", p)
		}
	}
	if _, err := os.Lstat(filepath.Join(runtimeDir, "tabbyAPI")); !os.IsNotExist(err) {
		t.Errorf("empty tabbyAPI dir not removed: %v", err)
	}
	fi, err := os.Lstat(venvLink)
	if err != nil {
		t.Fatalf("adopted venv link vanished: %v", err)
	}
	if fi.Mode()&os.ModeSymlink == 0 {
		t.Errorf("adopted venv link replaced: mode %v", fi.Mode())
	}
	if got, rerr := os.ReadFile(sentinel); rerr != nil || string(got) != "keep" {
		t.Errorf("adopted venv TARGET destroyed: %q %v", got, rerr)
	}
	if _, rerr := os.Lstat(realVenvPython); rerr != nil {
		t.Errorf("provisioned-style sibling venv lost: %v", rerr)
	}
	if fi, rerr := os.Lstat(runtimeDir); rerr != nil || !fi.IsDir() {
		t.Errorf("runtime dir lost: %v", rerr)
	}
}

// A failed supervised start must not keep the secrets it generated
// before the spawn (writeChildTokens runs first), nor the lock — while
// an adopted runtime/venv link stays untouched.
func TestServeSpawnFailureCleansGeneratedSecrets(t *testing.T) {
	p, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	dataDir := t.TempDir()
	runtimeDir := filepath.Join(dataDir, "runtime")
	if err := os.MkdirAll(runtimeDir, 0o700); err != nil {
		t.Fatal(err)
	}
	target := t.TempDir()
	sentinel := filepath.Join(target, "SENTINEL")
	if err := os.WriteFile(sentinel, []byte("keep"), 0o600); err != nil {
		t.Fatal(err)
	}
	venvLink := filepath.Join(runtimeDir, "venv")
	if err := os.Symlink(target, venvLink); err != nil {
		t.Skipf("needs symlink(2): %v", err)
	}
	serr := Serve(context.Background(), Options{
		Host:       "127.0.0.1",
		Port:       p,
		DataDir:    dataDir,
		ModelsDir:  t.TempDir(),
		RuntimeDir: runtimeDir,
		Stderr:     io.Discard,
		spawn: func(string, string, string, int) (*child, error) {
			return nil, errors.New("runtime missing")
		},
	})
	if serr == nil || !strings.Contains(serr.Error(), "runtime missing") {
		t.Fatalf("Serve = %v, want spawn failure propagated", serr)
	}
	for _, rel := range []string{"upstream_key", "tabby-config.yml", filepath.Join("tabbyAPI", "api_tokens.yml")} {
		if _, err := os.Lstat(filepath.Join(runtimeDir, rel)); !os.IsNotExist(err) {
			t.Errorf("generated file left by failed start: %s", rel)
		}
	}
	if _, err := os.Lstat(LockPath(dataDir)); !os.IsNotExist(err) {
		t.Errorf("lock file left by failed start")
	}
	if fi, err := os.Lstat(venvLink); err != nil || fi.Mode()&os.ModeSymlink == 0 {
		t.Errorf("adopted venv link damaged by failed start: %v %v", fi, err)
	}
	if got, rerr := os.ReadFile(sentinel); rerr != nil || string(got) != "keep" {
		t.Errorf("adopted venv TARGET destroyed: %q %v", got, rerr)
	}
}

// Defect 2 (supervised branch): `serve <model>` loads through our own
// /-/load — the same handler `run` uses — against the supervised backend,
// drains the event stream to the end, and marks the model loaded on the
// backend's "finished" event. Any failure would print once on stderr.
func TestInitialLoadThroughOwnLoadEndpoint(t *testing.T) {
	isolateConfig(t)
	var mu sync.Mutex
	var gotBody []byte
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/model/load" {
			http.NotFound(w, r)
			return
		}
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		gotBody = b
		mu.Unlock()
		fl, _ := w.(http.Flusher)
		io.WriteString(w, "data: {\"status\":\"finished\"}\n\n")
		if fl != nil {
			fl.Flush()
		}
	}))
	t.Cleanup(backend.Close)

	models := t.TempDir()
	if err := os.MkdirAll(filepath.Join(models, "m"), 0o755); err != nil {
		t.Fatal(err)
	}
	cfgJSON, err := os.ReadFile(filepath.Join("..", "autofit", "testdata", "smollm3-config.json"))
	if err != nil {
		t.Skipf("model config fixture: %v", err)
	}
	if err := os.WriteFile(filepath.Join(models, "m", "config.json"), cfgJSON, 0o644); err != nil {
		t.Fatal(err)
	}

	var errBuf bytes.Buffer
	d := &daemon{
		opts: Options{
			DataDir:   t.TempDir(),
			ModelsDir: models,
			Stderr:    &errBuf,
			probeGPU: func() (doctor.Report, error) {
				return doctor.Report{GPUs: []doctor.GPU{{VRAMMiB: 4096}}}, nil
			},
		},
		conn: Conn{BaseURL: backend.URL},
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	_, portStr, err := net.SplitHostPort(ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatal(err)
	}
	srv := &http.Server{Handler: d.authed(d.mux())}
	go srv.Serve(ln)
	t.Cleanup(func() { srv.Close() })
	d.st = State{Host: "127.0.0.1", Port: port}

	d.initialLoad(context.Background(), "m")

	mu.Lock()
	body := gotBody
	mu.Unlock()
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("load payload %q: %v", body, err)
	}
	if payload["model_name"] != "m" {
		t.Errorf("payload model_name = %v, want m (body %q)", payload["model_name"], body)
	}
	d.mu.Lock()
	model := d.st.Model
	d.mu.Unlock()
	if model != "m" {
		t.Errorf("st.Model = %q after the backend's finished event", model)
	}
	if errBuf.Len() != 0 {
		t.Errorf("initialLoad wrote stderr on success: %q", errBuf.String())
	}
}

// Bug: a fresh clone has no start_options.json, so start.py ran its
// self-installer (pip into whatever interpreter it found — "No virtual
// environment found for Python 3.11.10"), failed 3×, and the supervised
// child exited: no serve. The marker must be written in TabbyAPI's own
// shape — exactly what start.py persists itself (start.py:241).
func TestEnsureFirstRunMarkerWritesTabbyShape(t *testing.T) {
	runtimeDir := t.TempDir()
	checkout := filepath.Join(runtimeDir, "tabbyAPI")
	if err := os.MkdirAll(checkout, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := EnsureFirstRunMarker(runtimeDir); err != nil {
		t.Fatalf("EnsureFirstRunMarker: %v", err)
	}
	marker := filepath.Join(checkout, "start_options.json")
	fi, err := os.Lstat(marker)
	if err != nil {
		t.Fatalf("marker not written: %v", err)
	}
	if perm := fi.Mode().Perm(); perm != 0o600 {
		t.Errorf("marker mode = %o, want 600", perm)
	}
	b, err := os.ReadFile(marker)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("marker is not JSON start.py can load: %v (%s)", err, b)
	}
	if v, ok := got["first_run_done"].(bool); !ok || !v {
		t.Errorf("marker = %s, want first_run_done=true (start.py:201 reads this key)", b)
	}
	if len(got) != 1 {
		t.Errorf("marker carries keys start.py never wrote here: %s", b)
	}
}

// An existing marker is never rewritten: it may carry keys a user
// added (gpu_lib) or an explicit first_run_done=false choice.
func TestEnsureFirstRunMarkerKeepsExistingFile(t *testing.T) {
	runtimeDir := t.TempDir()
	checkout := filepath.Join(runtimeDir, "tabbyAPI")
	if err := os.MkdirAll(checkout, 0o700); err != nil {
		t.Fatal(err)
	}
	marker := filepath.Join(checkout, "start_options.json")
	original := `{"first_run_done":true,"gpu_lib":"cu12"}`
	if err := os.WriteFile(marker, []byte(original), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := EnsureFirstRunMarker(runtimeDir); err != nil {
		t.Fatalf("EnsureFirstRunMarker: %v", err)
	}
	b, err := os.ReadFile(marker)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != original {
		t.Errorf("existing marker rewritten: %q → %q", original, b)
	}
}

// A linked checkout is not ours (adoption links only runtime/venv; a
// linked tabbyAPI points into a user's own TabbyAPI): refuse instead of
// mutating it, target byte-identical.
func TestEnsureFirstRunMarkerRefusesLinkedCheckout(t *testing.T) {
	runtimeDir := t.TempDir()
	target := t.TempDir() // the user's own checkout
	if err := os.Symlink(target, filepath.Join(runtimeDir, "tabbyAPI")); err != nil {
		t.Skipf("needs symlink(2): %v", err)
	}
	err := EnsureFirstRunMarker(runtimeDir)
	if err == nil || !strings.Contains(err.Error(), "refusing") {
		t.Fatalf("linked checkout: err = %v, want refusal", err)
	}
	if !strings.Contains(err.Error(), "does not own") {
		t.Errorf("refusal must state ownership: %v", err)
	}
	if _, err := os.Lstat(filepath.Join(target, "start_options.json")); !os.IsNotExist(err) {
		t.Errorf("marker written into a checkout we do not own: %v", err)
	}
}

// No checkout: nothing to guard — and never fabricate a tabbyAPI dir
// (spawnChild's own stat owns that diagnosis).
func TestEnsureFirstRunMarkerWithoutCheckoutIsNoop(t *testing.T) {
	runtimeDir := t.TempDir()
	if err := EnsureFirstRunMarker(runtimeDir); err != nil {
		t.Fatalf("EnsureFirstRunMarker: %v", err)
	}
	if _, err := os.Lstat(filepath.Join(runtimeDir, "tabbyAPI")); !os.IsNotExist(err) {
		t.Errorf("phantom tabbyAPI dir created: %v", err)
	}
}

// The call-site contract: inside Serve the marker exists BEFORE spawn
// runs (the spy reads it mid-sequence), and a failed start keeps it —
// it is clone state, not a per-daemon secret.
func TestServeWritesFirstRunMarkerBeforeSpawn(t *testing.T) {
	p, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	dataDir := t.TempDir()
	runtimeDir := filepath.Join(dataDir, "runtime")
	checkout := filepath.Join(runtimeDir, "tabbyAPI")
	if err := os.MkdirAll(checkout, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(checkout, "start.py"), []byte("# fake checkout"), 0o600); err != nil {
		t.Fatal(err)
	}
	atSpawn := []byte("never read")
	serr := Serve(context.Background(), Options{
		Host:       "127.0.0.1",
		Port:       p,
		DataDir:    dataDir,
		ModelsDir:  t.TempDir(),
		RuntimeDir: runtimeDir,
		Stderr:     io.Discard,
		spawn: func(_, _, _ string, _ int) (*child, error) {
			atSpawn, _ = os.ReadFile(filepath.Join(checkout, "start_options.json"))
			return nil, errors.New("runtime missing")
		},
	})
	if serr == nil || !strings.Contains(serr.Error(), "runtime missing") {
		t.Fatalf("Serve = %v, want spawn failure propagated", serr)
	}
	if !strings.Contains(string(atSpawn), `"first_run_done": true`) {
		t.Errorf("spawn ran without the first-run marker (start.py would self-install): %q", atSpawn)
	}
	if _, err := os.Lstat(filepath.Join(checkout, "start_options.json")); err != nil {
		t.Errorf("failed start must keep the marker (clone state, not a secret): %v", err)
	}
}

// --- ProbeNoState: stop's missing-state gate (daemon.json lands only
// after the listener binds) ---

func TestProbeNoStateNotRunning(t *testing.T) {
	isolateConfig(t)
	p, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("STONE_LLAMA_PORT", strconv.Itoa(p))
	landed, perr := ProbeNoState(t.TempDir(), 500*time.Millisecond)
	if perr != nil || landed {
		t.Fatalf("ProbeNoState(down) = %v, %v; want false, nil", landed, perr)
	}
}

// Marker answers but no state exists in this data dir: name the
// holder, refuse to signal an unrecorded pid (never "not running").
func TestProbeNoStateNamesMarkerHolder(t *testing.T) {
	isolateConfig(t)
	ts := markerHealthzServer(t)
	t.Setenv("STONE_LLAMA_PORT", strconv.Itoa(serverPort(t, ts)))
	landed, perr := ProbeNoState(t.TempDir(), 500*time.Millisecond)
	if landed || perr == nil || !strings.Contains(perr.Error(), "serving") {
		t.Fatalf("ProbeNoState(marker) = %v, %v; want serving-holder error", landed, perr)
	}
}

// A listener that answers without our identity is reported as foreign —
// never signalled, never called "nothing".
func TestProbeNoStateNamesForeignHolder(t *testing.T) {
	isolateConfig(t)
	ts := httptest.NewServer(http.NotFoundHandler())
	t.Cleanup(ts.Close)
	t.Setenv("STONE_LLAMA_PORT", strconv.Itoa(serverPort(t, ts)))
	landed, perr := ProbeNoState(t.TempDir(), 500*time.Millisecond)
	if landed || perr == nil || !strings.Contains(perr.Error(), "another process") {
		t.Fatalf("ProbeNoState(foreign) = %v, %v; want foreign-holder error", landed, perr)
	}
}

// Pre-listen boot window: spawn lock held, port still free. A state
// file that lands during the wait hands the pid to the normal Stop.
func TestProbeNoStateWaitsForStartingDaemon(t *testing.T) {
	isolateConfig(t)
	p, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("STONE_LLAMA_PORT", strconv.Itoa(p))
	dataDir := t.TempDir()
	if err := os.MkdirAll(dataDir, 0o700); err != nil {
		t.Fatal(err)
	}
	lk, err := os.Create(LockPath(dataDir))
	if err != nil {
		t.Fatal(err)
	}
	defer lk.Close()
	if err := syscall.Flock(int(lk.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		t.Fatalf("hold spawn lock: %v", err)
	}
	go func() {
		time.Sleep(100 * time.Millisecond)
		WriteState(dataDir, State{PID: os.Getpid(), Host: "127.0.0.1", Port: p})
	}()
	landed, perr := ProbeNoState(dataDir, 5*time.Second)
	if perr != nil || !landed {
		t.Fatalf("ProbeNoState(booting) = %v, %v; want landed=true, nil", landed, perr)
	}
}

// A boot that never lands state must time out with an honest error —
// not a claim that nothing is running.
func TestProbeNoStateStartingTimeout(t *testing.T) {
	isolateConfig(t)
	p, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("STONE_LLAMA_PORT", strconv.Itoa(p))
	dataDir := t.TempDir()
	if err := os.MkdirAll(dataDir, 0o700); err != nil {
		t.Fatal(err)
	}
	lk, err := os.Create(LockPath(dataDir))
	if err != nil {
		t.Fatal(err)
	}
	defer lk.Close()
	if err := syscall.Flock(int(lk.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		t.Fatalf("hold spawn lock: %v", err)
	}
	landed, perr := ProbeNoState(dataDir, 50*time.Millisecond)
	if landed || perr == nil || !strings.Contains(perr.Error(), "starting the daemon") {
		t.Fatalf("ProbeNoState(stuck boot) = %v, %v; want starting-daemon error", landed, perr)
	}
}
