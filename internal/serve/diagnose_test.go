package serve

// A3/A5 regression tests. These reference the startDetached seam, so
// they exist only in the fixed tree (the base checkout compiles without
// this file; failing-before evidence for A3/A5 is the live binary
// comparison in report.md).

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// A5: the auto-start failure carries only the bytes THIS attempt
// appended to daemon.log — history from past attempts must never
// repeat (the old code tail'd the whole file, restacking every prior
// failure line on each retry).
func TestAutoStartFailureScopedToThisAttempt(t *testing.T) {
	dataDir := t.TempDir()
	logPath := filepath.Join(dataDir, "logs", "daemon.log")
	if err := os.MkdirAll(filepath.Dir(logPath), 0o700); err != nil {
		t.Fatal(err)
	}
	const stale = "stone-llama serve: STALE-FROM-LAST-ATTEMPT\n"
	if err := os.WriteFile(logPath, []byte(stale), 0o600); err != nil {
		t.Fatal(err)
	}
	old := startDetached
	startDetached = func(string, string) (<-chan struct{}, error) {
		done := make(chan struct{})
		if f, err := os.OpenFile(logPath, os.O_APPEND|os.O_WRONLY, 0o600); err == nil {
			io.WriteString(f, "stone-llama serve: NEW-FAILURE\n")
			f.Close()
		}
		close(done)
		return done, nil
	}
	t.Cleanup(func() { startDetached = old })
	p, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("STONE_LLAMA_PORT", fmt.Sprintf("%d", p))
	if _, err := EnsureDaemon(dataDir, true, 3*time.Second); err == nil {
		t.Fatal("want auto-start failure")
	} else {
		msg := err.Error()
		if !strings.Contains(msg, "NEW-FAILURE") {
			t.Errorf("this attempt's output missing from failure: %s", msg)
		}
		if strings.Contains(msg, "STALE-FROM-LAST-ATTEMPT") {
			t.Errorf("stale history leaked into failure message: %s", msg)
		}
	}
}

// A3: a marker-answering daemon on the configured port with no state
// file here is diagnosed as exactly that — the competing daemon, its
// address, and this state dir — never the generic "auto-start failed"
// and never "not running".
func TestAutoStartFailureNamesCompetingDaemon(t *testing.T) {
	ts := markerHealthzServer(t)
	t.Setenv("STONE_LLAMA_PORT", fmt.Sprintf("%d", serverPort(t, ts)))
	dataDir := t.TempDir()
	old := startDetached
	startDetached = func(string, string) (<-chan struct{}, error) {
		done := make(chan struct{})
		close(done)
		return done, nil
	}
	t.Cleanup(func() { startDetached = old })
	_, err := EnsureDaemon(dataDir, true, 3*time.Second)
	if err == nil {
		t.Fatal("want auto-start failure")
	}
	msg := err.Error()
	if !strings.Contains(msg, "already serving on") {
		t.Errorf("competing-daemon diagnosis missing: %s", msg)
	}
	if !strings.Contains(msg, dataDir) {
		t.Errorf("state-dir fact missing: %s", msg)
	}
	if strings.Contains(msg, "no stone-llama daemon is running") {
		t.Errorf("misdiagnosed as not running: %s", msg)
	}
}
