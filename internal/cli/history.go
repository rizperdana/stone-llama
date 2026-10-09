package cli

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/rizperdana/stone-llama/internal/config"
	"github.com/rizperdana/stone-llama/internal/serve"
)

// History is persisted under config.DataDir()/chats/<model>/chat-<id>.json so
// that `run` (direct, daemon, SSE, and repl paths) can resume a conversation
// with --continue and so each turn survives a process restart. Files are
// 0600, the directory tree 0700, written via temp+rename (atomic on POSIX).
const historySubdir = "chats"

// errNoSession marks "no prior session for this model" — used by --continue
// and --history callers to decide whether to start fresh or print a notice.
var errNoSession = errors.New("no prior session")

// Turn is one user/assistant exchange.
type Turn struct {
	User      string `json:"user"`
	Assistant string `json:"assistant"`
}

// Session is one persisted chat. JSON layout:
// {"id","model","created","updated","system","turns":[{"user","assistant"}]}
type Session struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Created int64  `json:"created"`
	Updated int64  `json:"updated"`
	System  string `json:"system,omitempty"`
	Turns   []Turn `json:"turns"`
}

func newSession(model, system string) *Session {
	now := time.Now().Unix()
	return &Session{
		ID:      strconv.FormatInt(time.Now().UnixNano(), 10),
		Model:   model,
		Created: now,
		Updated: now,
		System:  system,
	}
}

func sessionRoot() string            { return filepath.Join(config.DataDir(), historySubdir) }
func sessionDir(model string) string { return filepath.Join(sessionRoot(), model) }
func sessionFile(model, id string) string {
	return filepath.Join(sessionDir(model), "chat-"+id+".json")
}

// listSessionIDs returns ids newest-first by file mtime (plan's
// greatest-mtime rule; unix-nano ids would tie for same-nanos writes).
// Missing dir → empty slice, nil (treated as "no session" by callers).
func listSessionIDs(model string) ([]string, error) {
	entries, err := os.ReadDir(sessionDir(model))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	type ent struct {
		id string
		mt time.Time
	}
	var list []ent
	for _, e := range entries {
		name := e.Name()
		if !strings.HasPrefix(name, "chat-") || !strings.HasSuffix(name, ".json") {
			continue
		}
		eid := strings.TrimSuffix(strings.TrimPrefix(name, "chat-"), ".json")
		mt := time.Time{}
		if info, ierr := e.Info(); ierr == nil {
			mt = info.ModTime()
		}
		list = append(list, ent{eid, mt})
	}
	sort.Slice(list, func(i, j int) bool {
		if !list[i].mt.Equal(list[j].mt) {
			return list[i].mt.After(list[j].mt)
		}
		return list[i].id > list[j].id // mtime tie → id order
	})
	ids := make([]string, 0, len(list))
	for _, l := range list {
		ids = append(ids, l.id)
	}
	return ids, nil
}

// lastSession loads the newest session for model; errNoSession if none exists.
func lastSession(model string) (*Session, error) {
	ids, err := listSessionIDs(model)
	if err != nil || len(ids) == 0 {
		return nil, errNoSession
	}
	return loadSession(model, ids[0])
}

func loadSession(model, id string) (*Session, error) {
	b, err := os.ReadFile(sessionFile(model, id))
	if err != nil {
		return nil, errNoSession
	}
	var s Session
	if err := json.Unmarshal(b, &s); err != nil {
		return nil, errNoSession
	}
	return &s, nil
}

// saveSession persists s atomically: MkdirAll(0700) -> CreateTemp ->
// Write+Sync+Close -> Rename. Errors are returned for the caller to report
// on stderr in the house style; a failed history save never aborts a run.
func saveSession(s *Session) error {
	dir := sessionDir(s.Model)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".chat-*.tmp")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name()) // best-effort; Rename wins or leaves the temp
	b, merr := json.Marshal(s)
	if merr != nil {
		tmp.Close()
		return merr
	}
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
	return os.Rename(tmp.Name(), sessionFile(s.Model, s.ID))
}

// buildMessages flattens a session into the chat completions message list:
// system (if any) + alternating turns + the current user prompt.
func buildMessages(s *Session, prof runProfile, prompt string) []map[string]string {
	n := 1
	if prof.System != "" {
		n++
	}
	if s != nil {
		n += len(s.Turns) * 2
	}
	msgs := make([]map[string]string, 0, n)
	if prof.System != "" {
		msgs = append(msgs, map[string]string{"role": "system", "content": prof.System})
	}
	if s != nil {
		for _, t := range s.Turns {
			msgs = append(msgs, map[string]string{"role": "user", "content": t.User})
			msgs = append(msgs, map[string]string{"role": "assistant", "content": t.Assistant})
		}
	}
	msgs = append(msgs, map[string]string{"role": "user", "content": prompt})
	return msgs
}

// appendTurn records a completed exchange (non-empty reply) on the session.
func appendTurn(s *Session, user, assistant string) {
	if assistant == "" {
		return
	}
	s.Turns = append(s.Turns, Turn{User: user, Assistant: assistant})
	s.Updated = time.Now().Unix()
}

// sessionTokens is the fallback token estimate: the marshaled payload length
// divided by four (UTF-8 ≈ 0.75 tokens/char, the heuristic the overflow guard
// uses when the runtime exposes no /v1/token/encode endpoint).
func sessionTokens(s *Session) int {
	b, _ := json.Marshal(s)
	return len(b) / 4
}

// trimSession drops the OLDEST turn pairs until count(s) fits the context
// budget, never the system message. Trimming is intentionally persistent:
// the caller saves the trimmed session, so dropped pairs leave the file too
// (--continue never sees them again — context overflow means they could not
// have been replayed anyway). nCtx <= 0 disables the guard (no context
// length known — e.g. direct mode with no reachable daemon). count is
// re-measured after every drop (the caller supplies the exact upstream
// encode count when reachable, else the len(json)/4 fallback), so the loop
// and the stderr figures always come from the same source. Returns pairs
// removed, before/after counts, guarded, and exact (every count came from
// the upstream encode endpoint).
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
		s.Updated = time.Now().Unix()
	}
	return trimmed, before, after, true, exact
}

// guard trims this session to fit the context window before the next chat
// request. nCtx is resolved once per run (upstreamCtx). The token count
// comes from POST {base}/v1/token/encode on the exact message array,
// re-measured after every drop; the len(json)/4 estimate is only the
// fallback when that endpoint never answers, and the stderr figures are
// labelled exact vs estimated accordingly. reserved defaults to 2048 (the
// REPL cap) when --max-tokens 0 (unbounded) is in effect.
func (s *Session) guard(nCtx int, base, token string, prof runProfile, prompt string, stderr io.Writer) {
	if nCtx <= 0 {
		return // no context length known: guard skipped
	}
	reserved := prof.MaxTokens
	if reserved <= 0 {
		reserved = 2048
	}
	count := func(sess *Session) (int, bool) {
		if n, ok := encodeTokens(base, token, buildMessages(sess, prof, prompt)); ok {
			return n, true
		}
		return sessionTokens(sess), false
	}
	trimmed, before, after, guarded, exact := trimSession(s, nCtx, reserved, count)
	if guarded && trimmed > 0 {
		label := "exact"
		if !exact {
			label = "estimated"
		}
		fmt.Fprintf(stderr, "stone-llama: history trimmed %d turns (tokens %d -> %d, ctx %d, %s)\n", trimmed, before, after, nCtx, label)
	}
}

// upstreamCtx resolves the chat context window once per run: GET
// {base}/props → n_ctx (also accepts max_seq_len), else our live daemon's
// serve.Query status, else 0 (guard skipped — e.g. direct mode with no
// daemon reachable).
func upstreamCtx(base, token string) int {
	if b, err := httpGet(base+"/props", token); err == nil {
		var p struct {
			NCtx      int `json:"n_ctx"`
			MaxSeqLen int `json:"max_seq_len"`
		}
		if json.Unmarshal(b, &p) == nil {
			if p.NCtx > 0 {
				return p.NCtx
			}
			if p.MaxSeqLen > 0 {
				return p.MaxSeqLen
			}
		}
	}
	if st, err := serve.Query(config.DataDir()); err == nil {
		return st.Ctx
	}
	return 0
}

// encodeTokens asks the upstream to count the message array (POST
// {base}/v1/token/encode). Any failure (endpoint absent, unexpected shape)
// returns ok=false so the caller falls back to sessionTokens.
func encodeTokens(base, token string, msgs []map[string]string) (int, bool) {
	body, err := json.Marshal(map[string]any{"messages": msgs})
	if err != nil {
		return 0, false
	}
	req, err := http.NewRequest(http.MethodPost, strings.TrimRight(base, "/")+"/v1/token/encode", bytes.NewReader(body))
	if err != nil {
		return 0, false
	}
	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := probeClient.Do(req)
	if err != nil {
		return 0, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, false
	}
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	var p struct {
		Tokens int `json:"tokens"`
		Count  int `json:"count"`
		Length int `json:"length"`
	}
	if json.Unmarshal(b, &p) == nil {
		if p.Tokens > 0 {
			return p.Tokens, true
		}
		if p.Count > 0 {
			return p.Count, true
		}
		if p.Length > 0 {
			return p.Length, true
		}
	}
	var ids []int
	if json.Unmarshal(b, &ids) == nil && len(ids) > 0 {
		return len(ids), true
	}
	return 0, false
}

func httpGet(url, token string) ([]byte, error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := probeClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GET %s: status %d", url, resp.StatusCode)
	}
	return io.ReadAll(io.LimitReader(resp.Body, 1<<20))
}

// probeClient bounds the /props and /token/encode probes so a stalled
// upstream never wedges a chat turn.
var probeClient = &http.Client{Timeout: 2 * time.Second}
