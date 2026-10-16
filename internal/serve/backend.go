package serve

import (
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
)

// Backend is the seam between Serve's supervision loop and the engine
// it supervises. Deliberately minimal: Serve needs (1) everything that
// must exist before the child starts — validation plus 0600 config/key
// files, and the Conn the proxy probes and injects the upstream Bearer
// from — and (2) one spawn entry per incarnation. Readiness probing
// (probeHTTP: any answer = serving), the child lifecycle, shutdown and
// supervision are engine-agnostic and stay shared.
type Backend interface {
	// Prepare validates this run's requirements and writes the
	// pre-spawn files. Returns the Conn for the proxy and the config
	// path Spawn receives for this incarnation.
	Prepare(o *Options, port int) (Conn, string, error)
	// Spawn starts one child incarnation — same signature as the
	// Options.spawn seam, so Serve's supervision calls it unchanged.
	Spawn(runtimeDir, cfgPath, logPath string, port int, extraEnv map[string]string) (*child, error)
}

const (
	// BackendTabby is the default: TabbyAPI + ExLlamaV3 (existing behaviour).
	BackendTabby = "tabby"
	// BackendLlama supervises llama-server (llama.cpp) over GGUF weights.
	BackendLlama = "llama"
)

// BackendFor resolves a configured backend name to its implementation.
// "" or BackendTabby → tabbyBackend — the default stays TabbyAPI.
// deviceBudgetMiB is the llama weights budget (0 = no check); tabby
// ignores it. Unknown names are refused, never coerced.
func BackendFor(name string, deviceBudgetMiB int) (Backend, error) {
	switch name {
	case "", BackendTabby:
		return tabbyBackend{}, nil
	case BackendLlama:
		return llamaBackend{deviceBudgetMiB: deviceBudgetMiB}, nil
	default:
		return nil, fmt.Errorf("unknown backend %q (want %q or %q)", name, BackendTabby, BackendLlama)
	}
}

// startProcess launches argv[0] with argv[1:], cwd dir, stdout+stderr
// appended to logPath (0600), env parent + extra. The generic half of
// every backend's Spawn: spawnChild and llamaBackend.Spawn both build
// argv and delegate here, so log/permission/wait-channel discipline
// exists once.
func startProcess(argv []string, dir, logPath string, port int, extraEnv map[string]string) (*child, error) {
	if err := os.MkdirAll(filepath.Dir(logPath), 0o700); err != nil {
		return nil, err
	}
	lf, err := os.OpenFile(logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Dir = dir
	cmd.Stdout, cmd.Stderr = lf, lf
	cmd.Env = childEnv(extraEnv)
	if err := cmd.Start(); err != nil {
		lf.Close()
		return nil, fmt.Errorf("spawn backend: %w", err)
	}
	c := &child{cmd: cmd, wait: make(chan error, 1), port: port, logf: lf}
	go func() {
		c.wait <- cmd.Wait()
		lf.Close()
	}()
	return c, nil
}

// tabbyBackend is TabbyAPI — the default engine, one implementation of
// Backend. ORCHESTRATION ONLY: every helper it calls keeps TabbyAPI's
// knowledge (YAML schema, venv/argv/cwd, token names, product strings)
// in tabby_backend.go / tabbyconf.go; nothing TabbyAPI-specific is
// spelled out here. Both methods are the pre-seam Serve call sites
// moved verbatim — behaviour unchanged.
type tabbyBackend struct{}

// Prepare is Serve's former pre-spawn block in order — first-run
// marker, 0600 child keys, rendered config — with the same error
// wrapping, so user-facing text is unchanged.
func (tabbyBackend) Prepare(o *Options, port int) (Conn, string, error) {
	if err := EnsureFirstRunMarker(o.RuntimeDir); err != nil {
		return Conn{}, "", err
	}
	keyFile, _, err := writeChildTokens(o.RuntimeDir)
	if err != nil {
		return Conn{}, "", fmt.Errorf("write child keys: %w", err)
	}
	cfg := tabbyCfgPath(o.RuntimeDir)
	if err := writeSecret(cfg, RenderTabbyYAML(bootTabbyConfig(*o, port))); err != nil {
		return Conn{}, "", fmt.Errorf("write child config: %w", err)
	}
	return Conn{
		BaseURL:   "http://" + net.JoinHostPort("127.0.0.1", strconv.Itoa(port)),
		Name:      tabbyConnName(o.PinnedCommit),
		UpKeyFile: keyFile,
	}, cfg, nil
}

// Spawn is spawnChild unchanged (venv python start.py --config <cfg>).
func (tabbyBackend) Spawn(runtimeDir, cfgPath, logPath string, port int, extraEnv map[string]string) (*child, error) {
	return spawnChild(runtimeDir, cfgPath, logPath, port, extraEnv)
}
