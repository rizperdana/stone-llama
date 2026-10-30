package serve

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"syscall"
	"time"
)

// child is a supervised TabbyAPI process. Its wait channel carries
// exactly one value per incarnation: awaitReady may consume it first
// when the child dies before readiness (startReady pushes it back so
// supervise/shutdownChild still get theirs).
type child struct {
	cmd  *exec.Cmd
	wait chan error
	port int
	logf *os.File
}

// spawnChild starts the pinned entrypoint:
//
//	<runtime>/venv/bin/python start.py --config <cfgPath>
//
// with CWD in the pinned checkout; stdout+stderr append to logPath (0600).
// Env is the parent's plus config's engine_env (see childEnv): PATH and
// venv discovery stay unaffected. The config path is the only
// stone-llama-specific argv content — tokens and keys never appear in
// argv (A7).
func spawnChild(runtimeDir, cfgPath, logPath string, port int, extraEnv map[string]string) (*child, error) {
	python := filepath.Join(runtimeDir, "venv", "bin", "python")
	tabbyDir := filepath.Join(runtimeDir, "tabbyAPI")
	if _, err := os.Stat(python); err != nil {
		return nil, fmt.Errorf("runtime incomplete — run 'stone-llama setup' first (%s missing)", python)
	}
	argv := []string{python, "start.py", "--config", cfgPath}
	return startProcess(argv, tabbyDir, logPath, port, extraEnv)
}

// childEnv builds the child's environment: os.Environ() as the base so
// PATH/venv discovery is unaffected, then config's engine_env as sorted
// KEY=value pairs — sorted so ordering is deterministic for tests. The
// extras append last, so exec's dedup keeps them over a same-named
// parent variable.
func childEnv(extra map[string]string) []string {
	env := os.Environ()
	keys := make([]string, 0, len(extra))
	for k := range extra {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		env = append(env, k+"="+extra[k])
	}
	return env
}

func (c *child) signalTerm() error {
	if c == nil || c.cmd == nil || c.cmd.Process == nil {
		return nil
	}
	return c.cmd.Process.Signal(syscall.SIGTERM)
}

func (c *child) kill() error {
	if c == nil || c.cmd == nil || c.cmd.Process == nil {
		return nil
	}
	return c.cmd.Process.Kill()
}

// shutdownChild terminates gracefully, escalating to Kill after grace.
func (c *child) shutdownChild(grace time.Duration) {
	if c == nil {
		return
	}
	if c.cmd == nil {
		return // no process (test seam): nothing to signal or reap
	}
	c.signalTerm()
	select {
	case <-c.wait:
	case <-time.After(grace):
		c.kill()
		<-c.wait
	}
}

// probeHTTP treats ANY HTTP response as "serving" — auth and model state
// are not this layer's concern.
func probeHTTP(ctx context.Context, port int) error {
	rctx, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()
	url := "http://" + net.JoinHostPort("127.0.0.1", strconv.Itoa(port)) + "/v1/models"
	req, err := http.NewRequestWithContext(rctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	return nil
}

// backendServing reports whether the child genuinely serves a model:
// GET {BaseURL}/v1/models answers 200 with a NON-EMPTY data array,
// authenticated with the upstream key. Distinct from probeHTTP (any
// answer = serving; readiness only) and from modelLoaded (state only):
// TabbyAPI's load "finished" event fires before warmup completes, and
// in that window /v1/models answers 200 {"data":[]} while chat 503s.
func (d *daemon) backendServing(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, d.conn.BaseURL+"/v1/models", nil)
	if err != nil {
		return false
	}
	if d.upKey != "" {
		req.Header.Set("Authorization", "Bearer "+d.upKey)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return false
	}
	var m struct {
		Data []json.RawMessage `json:"data"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&m); err != nil {
		return false
	}
	return len(m.Data) > 0
}

// awaitReady polls until the child answers, the child dies, or ctx/timeout
// ends. died=true means the incarnation exited (caller may respawn).
// addr is the daemon's configured PUBLIC address — the never-ready
// failure must name where the user's client connects, not the internal
// ephemeral child port (H), and carries the literal "backend not ready"
// the CLI's remedy gate keys on.
func awaitReady(ctx context.Context, c *child, addr string, timeout, delay time.Duration, probe func(context.Context, int) error) (died bool, err error) {
	deadline := time.NewTimer(timeout)
	defer deadline.Stop()
	tick := time.NewTicker(delay)
	defer tick.Stop()
	for {
		if probe(ctx, c.port) == nil {
			return false, nil
		}
		select {
		case <-ctx.Done():
			return false, ctx.Err()
		case err := <-c.wait:
			return true, err
		case <-deadline.C:
			return false, fmt.Errorf("backend not ready on %s within %s", addr, timeout)
		case <-tick.C:
		}
	}
}

// logTail returns the last n lines of path (crash forensics).
func logTail(path string, n int) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return tailLines(b, n)
}

// logTailSince returns only the lines appended after offset — auto-start
// forensics scoped to THIS attempt (A5: the full-file tail re-printed
// every past failure, stacking a duplicate wall one line per retry). A
// file that shrank or rotated below offset yields nothing.
func logTailSince(path string, offset int64, n int) string {
	b, err := os.ReadFile(path)
	if err != nil || offset >= int64(len(b)) {
		return ""
	}
	return tailLines(b[offset:], n)
}

// tailLines renders the last n lines of b, newline-prefixed ("" when b
// is empty).
func tailLines(b []byte, n int) string {
	if len(b) == 0 {
		return ""
	}
	lines := 0
	for i := len(b) - 1; i >= 0; i-- {
		if b[i] == '\n' {
			lines++
			if lines == n {
				return "\n" + string(b[i+1:])
			}
		}
	}
	return "\n" + string(b)
}

// writeChildTokens pre-creates api_tokens.yml in the child checkout's
// CWD before spawn, so it reads OUR keys instead of generating and
// logging its own (first-run prints both keys to the log — A7 forbids
// that), plus the raw upstream key file that Conn.UpKeyFile points the
// proxy at. Returns (upKeyFile, admin); keys stay in memory + these
// 0600 files, never argv/log/stdout.
func writeChildTokens(runtimeDir string) (string, string, error) {
	mkKey := func() (string, error) {
		b := make([]byte, 16)
		if _, err := rand.Read(b); err != nil {
			return "", err
		}
		return hex.EncodeToString(b), nil
	}
	api, err := mkKey()
	if err != nil {
		return "", "", err
	}
	admin, err := mkKey()
	if err != nil {
		return "", "", err
	}
	tokensPath := filepath.Join(runtimeDir, "tabbyAPI", "api_tokens.yml")
	if err := writeSecret(tokensPath, fmt.Sprintf("api_key: %s\nadmin_key: %s\n", api, admin)); err != nil {
		return "", "", err
	}
	upKeyFile := filepath.Join(runtimeDir, "upstream_key")
	if err := writeSecret(upKeyFile, admin+"\n"); err != nil {
		return "", "", err
	}
	return upKeyFile, admin, nil
}

// EnsureFirstRunMarker guarantees TabbyAPI's first-run marker exists in
// the checkout stone-llama owns, before anything spawns the child (Serve
// calls it pre-spawn; setup calls it right after the clone lands).
//
// start.py decides first-run purely from start_options.json in its CWD
// (runtime/tabbyAPI, because spawnChild sets cmd.Dir to the checkout):
//
//	start_options_path = pathlib.Path("start_options.json")        start.py:188
//	first_run = not start_options.get("first_run_done")            start.py:201
//	if first_run or args.update_deps:  → pip install -U .[cu12]    start.py:219
//
// On a fresh clone the file does not exist, so the self-installer ran
// against whatever interpreter it found ("error: No virtual environment
// found for Python 3.11.10"), failed 3×, and the supervised child exited
// with status 1 — the CLI could not serve. The tool owns the clone and
// supplies its own venv, so TabbyAPI's self-installer must never run: it
// would pip install into the wrong interpreter. Hence the marker,
// written BEFORE spawn so the install path is unreachable.
//
// Shape: exactly what start.py itself persists after a first run
// (start.py:241, start_options["first_run_done"] = True) — the single
// boolean {"first_run_done": true}; no other key is read unless
// --update-deps runs (never passed by spawnChild). Atomic 0600 write,
// same discipline as every file the daemon generates.
//
// Ownership: only the checkout under runtime/ is written. An existing
// marker is never touched (rewriting would drop keys a user added, e.g.
// gpu_lib); if runtime/tabbyAPI is a symlink the checkout is NOT ours —
// adoption links only runtime/venv, a linked checkout points into a
// user's own TabbyAPI — so we refuse instead of mutating it. A missing
// checkout is a no-op: there is nothing to guard and spawnChild's own
// stat reports the real problem.
func EnsureFirstRunMarker(runtimeDir string) error {
	dir := filepath.Join(runtimeDir, "tabbyAPI")
	fi, err := os.Lstat(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	if fi.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("%s is a symlink — refusing to write a first-run marker into a TabbyAPI checkout stone-llama does not own (remove the link and run 'stone-llama setup' to clone one)", dir)
	}
	marker := filepath.Join(dir, "start_options.json")
	if _, err := os.Lstat(marker); err == nil {
		return nil // already present — never rewritten
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return writeSecret(marker, `{"first_run_done": true}`)
}

// tabbyCfgPath is the generated child config — stone-llama owns this
// file and never touches a user's config.yml.
func tabbyCfgPath(runtimeDir string) string {
	return filepath.Join(runtimeDir, "tabby-config.yml")
}

// tabbyConnName is the Conn.Name banner string for the supervised
// backend (the literal backend name lives here, not in serve.go).
func tabbyConnName(commit string) string {
	if commit == "" {
		return "TabbyAPI"
	}
	return "TabbyAPI " + commit
}
