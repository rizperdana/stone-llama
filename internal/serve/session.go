package serve

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/rizperdana/stone-llama/internal/config"
)

// Turn is one user/assistant exchange (server-side session).
type Turn struct {
	User      string `json:"user"`
	Assistant string `json:"assistant"`
}

// Session is one persisted server-side chat session.
type Session struct {
	ID           string `json:"id"`
	Model        string `json:"model,omitempty"`
	SystemPrompt string `json:"system,omitempty"`
	Turns        []Turn `json:"turns"`
	Created      int64  `json:"created"`
	LastAccessed int64  `json:"last_accessed"`
	Name         string `json:"name,omitempty"`

	mu sync.RWMutex `json:"-"`
}

type sessionSummary struct {
	ID           string `json:"id"`
	Name         string `json:"name,omitempty"`
	Model        string `json:"model,omitempty"`
	TurnCount    int    `json:"turn_count"`
	LastAccessed int64  `json:"last_accessed"`
}

func newSessionID() string {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		// ponytail: fall back to time-based id only if crypto/rand fails (never in practice)
		return fmt.Sprintf("%d", time.Now().UnixNano())
	}
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	h := hex.EncodeToString(b)
	return h[0:8] + "-" + h[8:12] + "-" + h[12:16] + "-" + h[16:20] + "-" + h[20:32]
}

func (d *daemon) sessionsDir() string {
	return filepath.Join(d.opts.DataDir, "sessions")
}

func (d *daemon) sessionPath(id string) string {
	return filepath.Join(d.sessionsDir(), id+".json")
}

func (d *daemon) createSession(model, system, name string) *Session {
	now := time.Now().Unix()
	s := &Session{
		ID:           newSessionID(),
		Model:        model,
		SystemPrompt: system,
		Turns:        []Turn{},
		Created:      now,
		LastAccessed: now,
		Name:         name,
	}
	d.sessions.Store(s.ID, s)
	_ = d.saveSession(s)
	return s
}

func (d *daemon) getSession(id string) (*Session, bool) {
	if v, ok := d.sessions.Load(id); ok {
		return v.(*Session), true
	}
	// try disk
	b, err := os.ReadFile(d.sessionPath(id))
	if err != nil {
		return nil, false
	}
	var s Session
	if err := json.Unmarshal(b, &s); err != nil {
		return nil, false
	}
	if s.ID == "" {
		s.ID = id
	}
	d.sessions.Store(s.ID, &s)
	return &s, true
}

func (d *daemon) deleteSession(id string) {
	d.sessions.Delete(id)
	_ = os.Remove(d.sessionPath(id))
}

func (d *daemon) touchSession(s *Session) {
	s.mu.Lock()
	s.LastAccessed = time.Now().Unix()
	s.mu.Unlock()
	_ = d.saveSession(s)
}

func (d *daemon) saveSession(s *Session) error {
	s.mu.RLock()
	b, err := json.Marshal(s)
	s.mu.RUnlock()
	if err != nil {
		return err
	}
	dir := d.sessionsDir()
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".session-*.tmp")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(b); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Chmod(tmp.Name(), 0o600); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), d.sessionPath(s.ID))
}

func (d *daemon) listSessions() []sessionSummary {
	out := []sessionSummary{}
	d.sessions.Range(func(key, value any) bool {
		s := value.(*Session)
		s.mu.RLock()
		out = append(out, sessionSummary{
			ID:           s.ID,
			Name:         s.Name,
			Model:        s.Model,
			TurnCount:    len(s.Turns),
			LastAccessed: s.LastAccessed,
		})
		s.mu.RUnlock()
		return true
	})
	// also disk sessions not loaded
	entries, _ := os.ReadDir(d.sessionsDir())
	for _, e := range entries {
		if e.IsDir() || !stringsHasSuffix(e.Name(), ".json") {
			continue
		}
		id := e.Name()[:len(e.Name())-5]
		if _, ok := d.sessions.Load(id); ok {
			continue
		}
		b, err := os.ReadFile(filepath.Join(d.sessionsDir(), e.Name()))
		if err != nil {
			continue
		}
		var s Session
		if err := json.Unmarshal(b, &s); err != nil {
			continue
		}
		if s.ID == "" {
			s.ID = id
		}
		out = append(out, sessionSummary{
			ID:           s.ID,
			Name:         s.Name,
			Model:        s.Model,
			TurnCount:    len(s.Turns),
			LastAccessed: s.LastAccessed,
		})
	}
	return out
}

func stringsHasSuffix(s, suf string) bool {
	return len(s) >= len(suf) && s[len(s)-len(suf):] == suf
}

// trimSession (server copy of cli.trimSession logic).
func trimSession(s *Session, nCtx, reserved int, count func(*Session) (int, bool)) (trimmed, before, after int, guarded, exact bool) {
	if nCtx <= 0 {
		return 0, 0, 0, false, false
	}
	budget := int(0.95*float64(nCtx)) - reserved
	if budget < 0 {
		budget = 0
	}
	var ok bool
	before, exact = count(s)
	after = before
	if before <= budget {
		return 0, before, before, true, exact
	}
	for len(s.Turns) > 0 {
		s.Turns = s.Turns[1:]
		trimmed++
		after, ok = count(s)
		exact = exact && ok
		if after <= budget {
			break
		}
	}
	if trimmed > 0 {
		s.LastAccessed = time.Now().Unix()
	}
	return trimmed, before, after, true, exact
}

func (d *daemon) sweepSessions(ttl time.Duration) {
	now := time.Now().Unix()
	d.sessions.Range(func(key, value any) bool {
		s := value.(*Session)
		s.mu.RLock()
		last := s.LastAccessed
		s.mu.RUnlock()
		if ttl > 0 && now-last > int64(ttl.Seconds()) {
			d.deleteSession(s.ID)
		}
		return true
	})
	// disk
	entries, _ := os.ReadDir(d.sessionsDir())
	for _, e := range entries {
		if e.IsDir() || !stringsHasSuffix(e.Name(), ".json") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(d.sessionsDir(), e.Name()))
		if err != nil {
			continue
		}
		var s Session
		if err := json.Unmarshal(b, &s); err != nil {
			continue
		}
		if ttl > 0 && now-s.LastAccessed > int64(ttl.Seconds()) {
			_ = os.Remove(filepath.Join(d.sessionsDir(), e.Name()))
		}
	}
}

func (d *daemon) sessionMaintenance(ctx context.Context) {
	if ctx == nil {
		ctx = context.Background()
	}
	ttl := d.sessionTTL()
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			d.sweepSessions(ttl)
		}
	}
}

func (d *daemon) sessionTTL() time.Duration {
	if d.sessionTTLValue > 0 {
		return d.sessionTTLValue
	}
	if d.opts.SessionTTL > 0 {
		return d.opts.SessionTTL
	}
	return time.Hour
}

func (d *daemon) keepAliveDur() time.Duration {
	if d.keepAliveValue > 0 {
		return d.keepAliveValue
	}
	if d.opts.KeepAlive > 0 {
		return d.opts.KeepAlive
	}
	return 5 * time.Minute
}

// ensure maintenance goroutine started once
var _ = config.Default
