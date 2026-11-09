package serve

// Keep-alive hold/expiry/contention contract, pinned with the existing
// seams only: opts.Now is the clock, opts.contenders fakes contention(),
// and the upstream is an httptest server answering POST
// /v1/model/unload with 200. No GPU, no real sleeping.

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rizperdana/stone-llama/internal/doctor"
)

// holdHarness: a daemon with a loaded model, a controllable clock, a
// contended-GPU seam, and an upstream whose /v1/model/unload counts
// calls and answers 200. handleHold/handleUnload are called through the
// recorder directly — no listener, no child, no GPU.
type holdHarness struct {
	d        *daemon
	unloads  atomic.Int64
	now      atomic.Int64 // unix seconds; advance explicitly
	contends atomic.Bool  // true = a foreign GPU holder is present
}

func newHoldHarness(t *testing.T) *holdHarness {
	t.Helper()
	h := &holdHarness{}
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/model/unload" && r.Method == http.MethodPost {
			h.unloads.Add(1)
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.URL.Path == "/v1/models" {
			io.WriteString(w, `{"object":"list","data":[{"id":"m"}]}`)
			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(up.Close)
	h.d = &daemon{
		opts: Options{
			DataDir: t.TempDir(),
			Stderr:  io.Discard,
			Host:    "127.0.0.1",
		},
		conn:  Conn{BaseURL: up.URL, Name: "test backend"},
		token: "tok",
	}
	h.d.opts.Now = func() time.Time { return time.Unix(h.now.Load(), 0).UTC() }
	h.d.opts.contenders = func() ([]doctor.Contender, error) {
		if h.contends.Load() {
			return []doctor.Contender{{PID: 4242, Name: "game.exe", MiB: 2600}}, nil
		}
		return nil, nil
	}
	h.d.markLoaded("m.gguf", 4096, "")
	return h
}

// post calls the handler directly (recorder, no listener).
func (h *holdHarness) post(path, body string) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	h.d.mux().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, path, strings.NewReader(body)))
	return rec
}

func TestHoldRouteContract(t *testing.T) {
	for _, tc := range []struct {
		name       string
		body       string
		contend    bool
		wantHeld   bool // hold window recorded afterwards
		wantUnload bool // backend unload POSTed
		wantCode   int
	}{
		{"zero unloads immediately", `{"seconds":0}`, false, false, true, http.StatusOK},
		{"negative unloads immediately", `{"seconds":-5}`, false, false, true, http.StatusOK},
		{"positive without contender holds", `{"seconds":300}`, false, true, false, http.StatusOK},
		{"positive with contender refuses and unloads", `{"seconds":300}`, true, false, true, http.StatusOK},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newHoldHarness(t)
			h.contends.Store(tc.contend)
			rec := h.post("/-/hold", tc.body)
			if rec.Code != tc.wantCode {
				t.Fatalf("/-/hold = %d, want %d (body %s)", rec.Code, tc.wantCode, rec.Body.String())
			}
			if got := h.d.modelHoldUntil() != 0; got != tc.wantHeld {
				t.Errorf("hold recorded = %v, want %v", got, tc.wantHeld)
			}
			if got := h.unloads.Load() > 0; got != tc.wantUnload {
				t.Errorf("unload issued = %v, want %v", got, tc.wantUnload)
			}
			if got := h.d.modelLoaded(); got != !tc.wantUnload {
				t.Errorf("model still loaded = %v, want %v", got, !tc.wantUnload)
			}
		})
	}
}

func TestHoldStatusUntilReportsDeadline(t *testing.T) {
	h := newHoldHarness(t)
	h.now.Store(1000)
	h.post("/-/hold", `{"seconds":300}`)
	st := h.d.currentStatus()
	want := time.Unix(1300, 0).UTC().Format(time.RFC3339)
	if st.Until != want {
		t.Errorf("status Until = %q, want %q (hold deadline)", st.Until, want)
	}
	// A client returns: the window is cancelled, Until stops reporting a
	// deadline.
	h.d.markWanted()
	if st := h.d.currentStatus(); st.Until != "Infinite" {
		t.Errorf("status Until after markWanted = %q, want \"Infinite\" (wanted model, no deadline)", st.Until)
	}
}

func TestHoldExpirySupersededByWantedClient(t *testing.T) {
	h := newHoldHarness(t)
	h.now.Store(1000)
	h.post("/-/hold", `{"seconds":1}`)
	// A client touches the model before the window ends.
	h.d.markWanted()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if h.d.holdExpiry(ctx) {
		t.Fatal("holdExpiry = true after the hold was cancelled by a client; a stale waiter must never unload a wanted model")
	}
	if !h.d.modelLoaded() {
		t.Error("model unloaded although a client is active")
	}
}

func TestHoldExpiryMaintenanceUnloadsOnce(t *testing.T) {
	h := newHoldHarness(t)
	h.d.holdTick = make(chan time.Time, 1)
	h.now.Store(1000)
	h.post("/-/hold", `{"seconds":2}`)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan struct{})
	go func() { h.d.keepAliveMaintenance(ctx); close(done) }()
	h.d.holdWake(time.Unix(h.d.modelHoldUntil(), 0)) // nudge: run the loop now
	// Advance the clock past the deadline and re-nudge.
	h.now.Store(1003)
	h.d.holdWake(time.Unix(h.d.modelHoldUntil(), 0))
	// The loop unloads, then blocks again (hold cleared → break → wait).
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if h.unloads.Load() >= 1 && !h.d.modelLoaded() {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if h.unloads.Load() != 1 {
		t.Fatalf("unload calls = %d, want 1 after expiry", h.unloads.Load())
	}
	if h.d.modelLoaded() {
		t.Error("model still loaded after the hold expired")
	}
	cancel()
	<-done
}

func TestHoldMaintenanceCutShortByContention(t *testing.T) {
	h := newHoldHarness(t)
	h.d.holdTick = make(chan time.Time, 1)
	h.now.Store(1000)
	h.post("/-/hold", `{"seconds":60}`) // long window
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan struct{})
	go func() { h.d.keepAliveMaintenance(ctx); close(done) }()
	// The clock never reaches the deadline — only the contender appears.
	h.contends.Store(true)
	h.d.holdWake(time.Unix(h.d.modelHoldUntil(), 0))
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if !h.d.modelLoaded() {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if h.unloads.Load() != 1 {
		t.Fatalf("unload calls = %d, want 1 (cut short, not waited out)", h.unloads.Load())
	}
	if h.d.modelLoaded() {
		t.Error("model still held while a foreign contender wants the GPU")
	}
	cancel()
	<-done
}
