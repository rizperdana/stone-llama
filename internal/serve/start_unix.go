//go:build !windows

package serve

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
)

// StartDetached re-executes this binary as `stone-llama serve`, setsid'd
// with output appended to logs/daemon.log (0600). The returned channel
// closes when the starter exits — the auto-starter selects on it to
// surface failures instead of hanging.
func StartDetached(exe, dataDir string) (<-chan struct{}, error) {
	logs := filepath.Join(dataDir, "logs")
	if err := os.MkdirAll(logs, 0o700); err != nil {
		return nil, err
	}
	lf, err := os.OpenFile(filepath.Join(logs, "daemon.log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(exe, "serve")
	cmd.Stdout, cmd.Stderr = lf, lf
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := cmd.Start(); err != nil {
		lf.Close()
		return nil, fmt.Errorf("auto-start: %w", err)
	}
	done := make(chan struct{})
	go func() {
		cmd.Wait()
		lf.Close()
		close(done)
	}()
	return done, nil
}
