package serve

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"syscall"
	"time"
)

// child is a supervised TabbyAPI process. Only the supervisor consumes
// the wait channel (one receive per incarnation).
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
// The config path is the only stone-llama-specific argv content — tokens
// and keys never appear in argv (A7).
func spawnChild(runtimeDir, cfgPath, logPath string, port int) (*child, error) {
	python := filepath.Join(runtimeDir, "venv", "bin", "python")
	tabbyDir := filepath.Join(runtimeDir, "tabbyAPI")
	if _, err := os.Stat(python); err != nil {
		return nil, fmt.Errorf("runtime incomplete — run 'stone-llama setup' first (%s missing)", python)
	}
	if err := os.MkdirAll(filepath.Dir(logPath), 0o700); err != nil {
		return nil, err
	}
	lf, err := os.OpenFile(logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(python, "start.py", "--config", cfgPath)
	cmd.Dir = tabbyDir
	cmd.Stdout, cmd.Stderr = lf, lf
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

// awaitReady polls until the child answers, the child dies, or ctx/timeout
// ends. died=true means the incarnation exited (caller may respawn).
func awaitReady(ctx context.Context, c *child, timeout, delay time.Duration, probe func(context.Context, int) error) (died bool, err error) {
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
			return false, fmt.Errorf("backend not ready on port %d within %s", c.port, timeout)
		case <-tick.C:
		}
	}
}

// logTail returns the last n lines of path (crash forensics).
func logTail(path string, n int) string {
	b, err := os.ReadFile(path)
	if err != nil || len(b) == 0 {
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
