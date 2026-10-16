package cli

import (
	"net"
	"strings"
	"testing"
)

// freeUnboundPort returns a port that is free RIGHT NOW and is released
// before use, so no test in this file can collide with a live daemon on
// the default 5111 — and so no assertion here depends on 5111 being
// empty. Serve only PROBES its downstream port before it binds (and the
// attach path below returns before it binds at all), so a probed port we
// never hold is exactly what we want.
func freeUnboundPort(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	_, port, err := net.SplitHostPort(ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	return port
}

// TestServeBackendFlagRejectsUnknown: --backend is a usage error (exit
// 2, the code every other bad serve flag uses), and the message names
// the real choices. A bogus value must never be coerced to the default
// engine. All three cases are refused INSIDE the flag loop, before Serve
// is called at all — no daemon, no port, no child.
func TestServeBackendFlagRejectsUnknown(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", t.TempDir()) // a leak here can never touch a real data dir
	isolateConfig(t)
	for _, args := range [][]string{
		{"serve", "--backend", "cuda"},
		{"serve", "--backend=vram"},
		{"serve", "--backend"}, // missing value
	} {
		code, _, errb := run(args...)
		if code != 2 {
			t.Errorf("%v: code = %d, want 2 (err %q)", args, code, errb)
		}
	}
	_, _, errb := run("serve", "--backend", "cuda")
	if !strings.Contains(errb, `invalid --backend "cuda" (want tabby|llama)`) {
		t.Errorf("bad value message = %q", errb)
	}
}

// TestServeBackendAcceptedNamesRealEngine: --backend must never be
// refused by the flag layer.
//
// --attach <dead port> is load-bearing, NOT decoration: without it this
// would take the SUPERVISED path and really start a `stone-llama serve`
// daemon (it supervises until readiness fails, ~120s, on the default
// port) — that is how an earlier version of this test leaked a daemon
// onto 127.0.0.1:5111. With --attach, Serve returns at probeAttach
// before it binds a listener or spawns anything. Do not "simplify" the
// flag away. --port additionally keeps the probe off the default 5111,
// so these assertions do not depend on 5111 being free.
func TestServeBackendAcceptedNamesRealEngine(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", t.TempDir()) // a leak here can never touch a real data dir
	isolateConfig(t)
	port := freeUnboundPort(t)
	if port == "5111" {
		t.Fatalf("test port must not be the default 5111 (got %s)", port)
	}
	for _, v := range []string{"tabby", "llama"} {
		code, _, errb := run("serve", "--backend", v, "--attach", "127.0.0.1:9", "--port", port)
		if code == 2 || strings.Contains(errb, "usage: stone-llama serve") {
			t.Errorf("--backend %s: code = %d, err = %q — the flag layer must accept it", v, code, errb)
		}
		if !strings.Contains(errb, "not reachable") {
			t.Errorf("--backend %s did not reach the attach probe: %q", v, errb)
		}
	}
	// the probe port was never bound by us: nothing may be listening now
	// that both runs returned.
	if ln, err := net.Listen("tcp", "127.0.0.1:"+port); err != nil {
		t.Errorf("test left port %s bound: %v", port, err)
	} else {
		ln.Close()
	}
}
