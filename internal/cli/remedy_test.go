package cli

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/rizperdana/stone-llama/internal/config"
	"github.com/rizperdana/stone-llama/internal/serve"
)

// The auto-start failure message embeds the child's own output, which
// already carries one remedy block (live demo caught the double
// print). backendStartHelp must add nothing in that case; a fresh
// runtime-incomplete error with no block still gets one. This file
// references backendStartHelp (fixed tree only) — excluded when copying
// tests onto the base checkout, like serve's diagnose_test.go.
func TestRemedyBlockNeverPrintsTwice(t *testing.T) {
	msg := "auto-start failed:\n" +
		"stone-llama serve: runtime incomplete — run 'stone-llama setup' first (x missing)\n" +
		"remedies, least friction first:\n  1. attach to a running OpenAI-compatible server\n"
	if got := backendStartHelp(msg); got != "" {
		t.Errorf("second remedy block appended: %q", got)
	}
	fresh := "stone-llama serve: runtime incomplete — run 'stone-llama setup' first (x missing)"
	got := backendStartHelp(fresh)
	if !strings.Contains(got, "remedies, least friction first") {
		t.Errorf("fresh runtime-incomplete error got no remedies: %q", got)
	}
}

// run with auto-start disabled hits the no-daemon error directly: it
// must carry the same single remedy block (attach / adopt / setup).
func TestNoDaemonErrorCarriesRemedies(t *testing.T) {
	msg := "no stone-llama daemon is running (start one with 'stone-llama serve')"
	got := backendStartHelp(msg)
	for _, want := range []string{"remedies, least friction first", "serve --attach", "setup --adopt", "stone-llama setup"} {
		if !strings.Contains(got, want) {
			t.Errorf("remedies missing %q: %q", want, got)
		}
	}
	if strings.Count(got, "remedies, least friction first") != 1 {
		t.Errorf("want exactly one block: %q", got)
	}
}

// G1: the "nothing is available" error must not bake in the run prefix —
// runRun's print (stone-llama run: %v) is the single prefix. The base
// tree prefixed twice: "stone-llama run: stone-llama run: nothing is…".
func TestNothingAvailableSingleRunPrefix(t *testing.T) {
	err := nothingAvailableErr("http://127.0.0.1:5002", t.TempDir())
	got := fmt.Sprintf("stone-llama run: %v", err) // cli.go's print
	if !strings.HasPrefix(got, "stone-llama run: nothing is available") {
		t.Errorf("message start = %q", got)
	}
	if n := strings.Count(got, "stone-llama run:"); n != 1 {
		t.Errorf("stone-llama run: appears %d times, want 1:\n%s", n, got)
	}
	if !strings.Contains(got, "remedies, least friction first") {
		t.Errorf("remedy block missing:\n%s", got)
	}
}

// G2/K1: serve's startFailed already emits the single "auto-start failed:"
// prefix — ResolveBackend must not re-wrap it (the base tree printed it
// twice), and a busy failure still gets one remedy block leading with
// wait-and-retry (H2).
func TestAutoStartFailureSinglePrefix(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	t.Setenv("STONE_LLAMA_NO_AUTOSTART", "0")
	isolateConfig(t)
	dataDir := config.DataDir()
	// a provisioned runtime so ResolveBackend reaches the supervised start
	if err := os.MkdirAll(filepath.Join(dataDir, "runtime", "venv", "bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dataDir, "runtime", "venv", "bin", "python"),
		[]byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	origStatus, origProbe, origStart := queryStatusFn, probeUpstreamFn, startSupervisedFn
	queryStatusFn = func(string) (serve.Status, error) { return serve.Status{}, serve.ErrNoDaemon }
	probeUpstreamFn = func(context.Context, string, time.Duration) serve.ProbeResult {
		return serve.ProbeResult{Err: errors.New("connection refused")}
	}
	startSupervisedFn = func(string, time.Duration) (serve.State, error) {
		return serve.State{}, errors.New("auto-start failed: stone-llama serve: another stone-llama process is starting the daemon")
	}
	t.Cleanup(func() {
		queryStatusFn, probeUpstreamFn, startSupervisedFn = origStatus, origProbe, origStart
	})

	_, err := ResolveBackend(context.Background(), io.Discard, io.Discard, dataDir, config.Config{}, "", "", false)
	if err == nil {
		t.Fatal("want an auto-start failure")
	}
	got := fmt.Sprintf("stone-llama run: %v", err)
	if n := strings.Count(got, "auto-start failed:"); n != 1 {
		t.Errorf("auto-start failed: appears %d times, want 1:\n%s", n, got)
	}
	if n := strings.Count(got, "remedies, least friction first"); n != 1 {
		t.Errorf("remedy block appears %d times, want 1:\n%s", n, got)
	}
	if !strings.Contains(got, "wait a few seconds and retry") {
		t.Errorf("busy failure must lead with wait-and-retry:\n%s", got)
	}
}

// H1: remedy 1 must not claim a running server when the upstream does not
// answer — probe first, then tailor: live → attach, down → adopt/setup.
func TestBackendStartRemedyProbesUpstream(t *testing.T) {
	orig := probeUpstreamFn
	t.Cleanup(func() { probeUpstreamFn = orig })

	probeUpstreamFn = func(context.Context, string, time.Duration) serve.ProbeResult {
		return serve.ProbeResult{Err: errors.New("connect: connection refused")}
	}
	got := backendStartRemedy("auto-start failed: starter produced no output (log: /x/daemon.log)", "http://127.0.0.1:5002")
	if strings.Contains(got, "attach to the running server on") {
		t.Errorf("offered attach to a dead upstream: %q", got)
	}
	for _, want := range []string{"setup --adopt", "stone-llama setup", "if you have a server at 127.0.0.1:5002"} {
		if !strings.Contains(got, want) {
			t.Errorf("down-upstream remedy missing %q: %q", want, got)
		}
	}

	probeUpstreamFn = func(context.Context, string, time.Duration) serve.ProbeResult {
		return serve.ProbeResult{Ok: true}
	}
	got = backendStartRemedy("auto-start failed: starter produced no output (log: /x/daemon.log)", "http://127.0.0.1:5002")
	if !strings.Contains(got, "attach to the running server on 127.0.0.1:5002") {
		t.Errorf("live upstream must offer attach: %q", got)
	}
}

// H2: busy/starting failures lead with wait-and-retry instead of adopt/setup.
func TestBackendStartRemedyBusySaysRetry(t *testing.T) {
	for _, msg := range []string{
		"daemon still starting after 30s — check /x/logs/daemon.log",
		"auto-start failed: stone-llama serve: another stone-llama process is starting the daemon",
	} {
		got := backendStartRemedy(msg, "http://127.0.0.1:5002")
		if !strings.Contains(got, "1. wait a few seconds and retry") {
			t.Errorf("busy remedy missing wait-first for %q: %q", msg, got)
		}
	}
}

// K4/H3: the foreground never-ready error ("backend not ready" + addr +
// tabby.log tail) must get the standard remedy block from
// backendStartHelp; a message already carrying the block is never doubled.
func TestBackendStartHelpMatchesBackendNotReady(t *testing.T) {
	msg := "stone-llama serve: backend not ready after 30s at 127.0.0.1:5111 — tail of tabby.log: boom"
	got := backendStartHelp(msg)
	for _, want := range []string{"remedies, least friction first", "serve --attach", "setup --adopt", "stone-llama setup"} {
		if !strings.Contains(got, want) {
			t.Errorf("remedy missing %q: %q", want, got)
		}
	}
	if doubled := backendStartHelp(msg + "\nremedies, least friction first:\n  1. x"); doubled != "" {
		t.Errorf("second remedy block appended: %q", doubled)
	}
}
