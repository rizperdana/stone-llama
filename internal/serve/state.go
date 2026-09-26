package serve

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

var ErrNoDaemon = errors.New("no stone-llama daemon is running")

// State is daemon.json (0600): who serves, where, and the downstream
// bearer token (A7 secret hygiene — file only, never argv/log).
type State struct {
	ChildPID    int    `json:"child_pid,omitempty"` // supervised TabbyAPI pid (VRAM attribution for ps)
	PID         int    `json:"pid"`
	Host        string `json:"host"`
	Port        int    `json:"port"`
	Token       string `json:"token,omitempty"`
	Model       string `json:"model,omitempty"`
	Ctx         int    `json:"ctx,omitempty"`
	CacheMode   string `json:"cache_mode,omitempty"`
	Attach      string `json:"attach,omitempty"`        // upstream base URL in attach mode
	VRAMPeakMiB int    `json:"vram_peak_mib,omitempty"` // sampled from the child's nvidia-smi row
	StartedAt   int64  `json:"started_at"`              // unix seconds
}

// Addr renders the /-/status bind address (ps/test client side).
func (s Status) Addr() string {
	return net.JoinHostPort(s.Host, strconv.Itoa(s.Port))
}

// Addr renders the downstream bind address (IPv6-safe).
func (s State) Addr() string {
	return net.JoinHostPort(s.Host, strconv.Itoa(s.Port))
}

func StatePath(dataDir string) string { return filepath.Join(dataDir, "daemon.json") }

// LockPath is the daemon-spawn flock target (ARCHITECTURE §2).
func LockPath(dataDir string) string { return filepath.Join(dataDir, "stone-llama.lock") }

// Query asks the running daemon for /-/status: readiness, attach model,
// and live state in one round trip (the ps/run client side).
func Query(dataDir string) (Status, error) {
	st, err := ReadState(dataDir)
	if err != nil {
		return Status{}, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://"+st.Addr()+statusPath, nil)
	if err != nil {
		return Status{}, err
	}
	if st.Token != "" {
		req.Header.Set("Authorization", "Bearer "+st.Token)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return Status{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return Status{}, fmt.Errorf("status endpoint: HTTP %d", resp.StatusCode)
	}
	var s Status
	if err := json.NewDecoder(resp.Body).Decode(&s); err != nil {
		return Status{}, err
	}
	return s, nil
}

// ReadState loads daemon.json. Missing/corrupt → ErrNoDaemon.
func ReadState(dataDir string) (State, error) {
	b, err := os.ReadFile(StatePath(dataDir))
	if err != nil {
		if os.IsNotExist(err) {
			return State{}, ErrNoDaemon
		}
		return State{}, err
	}
	var st State
	if err := json.Unmarshal(b, &st); err != nil {
		return State{}, fmt.Errorf("daemon.json: %w", err)
	}
	return st, nil
}

// WriteState persists state as 0600 via temp+fsync+rename.
func WriteState(dataDir string, st State) error {
	if err := os.MkdirAll(dataDir, 0o700); err != nil {
		return err
	}
	b, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		return err
	}
	final := StatePath(dataDir)
	tmp := final + ".tmp"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	if _, err := f.Write(b); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, final); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}

// RemoveState deletes daemon.json; absent is fine (idempotent cleanup).
func RemoveState(dataDir string) error {
	err := os.Remove(StatePath(dataDir))
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}
