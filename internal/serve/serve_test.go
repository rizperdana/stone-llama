package serve

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
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
	_, err := EnsureDaemon(t.TempDir(), false, time.Second)
	if err == nil || !strings.Contains(err.Error(), "no stone-llama daemon is running") {
		t.Fatalf("EnsureDaemon = %v, want not-running error", err)
	}
}

// TestStartDetachedMissingExe: detached start seam reports failure
// instead of hanging.
func TestStartDetachedMissingExe(t *testing.T) {
	if _, err := StartDetached(filepath.Join(t.TempDir(), "nonexistent"), t.TempDir()); err == nil {
		t.Fatal("StartDetached missing exe = nil, want error")
	}
}
