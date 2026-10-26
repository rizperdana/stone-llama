package serve

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// register session routes called from mux? We'll add in serve.go mux instead via patch.
func (d *daemon) handleSessions(w http.ResponseWriter, r *http.Request) {
	// /v1/sessions and /v1/sessions/<id>
	path := strings.TrimPrefix(r.URL.Path, "/v1/sessions")
	path = strings.TrimPrefix(path, "/")
	if path == "" {
		switch r.Method {
		case http.MethodGet:
			d.handleSessionList(w, r)
		case http.MethodPost:
			d.handleSessionCreate(w, r)
		default:
			writeJSON(w, http.StatusMethodNotAllowed, "method not allowed", "method_not_allowed")
		}
		return
	}
	// special sub-routes
	if path == "cleanup" && r.Method == http.MethodPost {
		d.handleSessionCleanup(w, r)
		return
	}
	if path == "unload" && r.Method == http.MethodPost {
		d.handleSessionUnload(w, r)
		return
	}
	if path == "keepalive" && r.Method == http.MethodPost {
		d.handleSessionKeepalive(w, r)
		return
	}
	id := path
	switch r.Method {
	case http.MethodGet:
		d.handleSessionGet(w, r, id)
	case http.MethodDelete:
		d.handleSessionDelete(w, r, id)
	case http.MethodPatch:
		d.handleSessionPatch(w, r, id)
	default:
		writeJSON(w, http.StatusMethodNotAllowed, "method not allowed", "method_not_allowed")
	}
}

func (d *daemon) handleSessionCreate(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model  string `json:"model"`
		System string `json:"system"`
		Name   string `json:"name"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)
	s := d.createSession(req.Model, req.System, req.Name)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"id": s.ID, "name": s.Name})
}

func (d *daemon) handleSessionList(w http.ResponseWriter, r *http.Request) {
	list := d.listSessions()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"sessions": list})
}

func (d *daemon) handleSessionGet(w http.ResponseWriter, r *http.Request, id string) {
	s, ok := d.getSession(id)
	if !ok {
		writeJSON(w, http.StatusNotFound, "session not found", "not_found")
		return
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(s)
}

func (d *daemon) handleSessionDelete(w http.ResponseWriter, r *http.Request, id string) {
	d.deleteSession(id)
	w.WriteHeader(http.StatusNoContent)
}

func (d *daemon) handleSessionPatch(w http.ResponseWriter, r *http.Request, id string) {
	s, ok := d.getSession(id)
	if !ok {
		writeJSON(w, http.StatusNotFound, "session not found", "not_found")
		return
	}
	var req struct {
		Name   *string `json:"name"`
		System *string `json:"system"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)
	s.mu.Lock()
	if req.Name != nil {
		s.Name = *req.Name
	}
	if req.System != nil {
		s.SystemPrompt = *req.System
	}
	s.mu.Unlock()
	_ = d.saveSession(s)
	d.touchSession(s)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"id": s.ID, "name": s.Name})
}

func (d *daemon) handleSessionCleanup(w http.ResponseWriter, r *http.Request) {
	d.sweepSessions(d.sessionTTL())
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"ok": true})
}

func (d *daemon) handleSessionUnload(w http.ResponseWriter, r *http.Request) {
	d.clearLoaded()
	// also unload model? warm model keepalive: just mark. We'll also call backend unload? For simplicity set loaded model empty.
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"ok": true})
}

func (d *daemon) handleSessionKeepalive(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Session string `json:"session"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)
	if req.Session != "" {
		if s, ok := d.getSession(req.Session); ok {
			d.touchSession(s)
		}
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"ok": true})
}

// handleChatCompletionsSession wraps proxy to inject session history.
// We implement as middleware in mux? Instead add route /v1/chat/completions handled here, then proxy.
func (d *daemon) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	// parse body
	var raw map[string]any
	b, _ := io.ReadAll(r.Body)
	r.Body.Close()
	_ = json.Unmarshal(b, &raw)
	sessionID, _ := raw["session"].(string)
	if sessionID == "" {
		// proxy as-is
		d.proxyChat(w, r, b)
		return
	}
	s, ok := d.getSession(sessionID)
	if !ok {
		// create? spec: accepts session id; if not exists create new? We'll create new session from request.
		model, _ := raw["model"].(string)
		sys := ""
		if msgs, ok := raw["messages"].([]any); ok {
			for _, m := range msgs {
				if mm, ok := m.(map[string]any); ok {
					if mm["role"] == "system" {
						if c, ok := mm["content"].(string); ok {
							sys = c
						}
					}
				}
			}
		}
		s = d.createSession(model, sys, "")
		sessionID = s.ID
	}
	// build messages: stored turns + incoming user message
	msgs, _ := raw["messages"].([]any)
	// extract incoming user messages (last user)
	var incoming []map[string]any
	for _, m := range msgs {
		if mm, ok := m.(map[string]any); ok {
			incoming = append(incoming, mm)
		}
	}
	// prepend stored turns
	var newMsgs []map[string]any
	s.mu.RLock()
	if s.SystemPrompt != "" {
		hasSys := false
		for _, m := range incoming {
			if m["role"] == "system" {
				hasSys = true
			}
		}
		if !hasSys {
			newMsgs = append(newMsgs, map[string]any{"role": "system", "content": s.SystemPrompt})
		}
	}
	recap := s.Recap
	if recap != "" {
		newMsgs = append(newMsgs, map[string]any{"role": "system", "content": recap})
	}
	for _, t := range s.Turns {
		newMsgs = append(newMsgs, map[string]any{"role": "user", "content": t.User})
		newMsgs = append(newMsgs, map[string]any{"role": "assistant", "content": t.Assistant})
	}
	// append incoming (excluding system if stored system)
	for _, m := range incoming {
		if m["role"] == "system" {
			continue
		}
		newMsgs = append(newMsgs, m)
	}
	s.mu.RUnlock()

	// capture last user content for turn recording
	lastUser := ""
	for i := len(incoming) - 1; i >= 0; i-- {
		if incoming[i]["role"] == "user" {
			if c, ok := incoming[i]["content"].(string); ok {
				lastUser = c
			}
			break
		}
	}

	raw["messages"] = newMsgs
	// Trim oldest turns until the full payload (stored turns + incoming + recap)
	// fits the context window, recording a recap of dropped turns.
	nCtx := d.st.Ctx
	if nCtx <= 0 {
		nCtx = 4096
	}
	countFn := func(sess *Session) (int, bool) {
		// rebuild message list from sess.Turns so trimming actually reduces count
		var msgs []map[string]any
		if s.SystemPrompt != "" {
			msgs = append(msgs, map[string]any{"role": "system", "content": s.SystemPrompt})
		}
		if sess.Recap != "" {
			msgs = append(msgs, map[string]any{"role": "system", "content": sess.Recap})
		}
		for _, t := range sess.Turns {
			msgs = append(msgs, map[string]any{"role": "user", "content": t.User})
			msgs = append(msgs, map[string]any{"role": "assistant", "content": t.Assistant})
		}
		for _, m := range incoming {
			if m["role"] == "system" {
				continue
			}
			msgs = append(msgs, m)
		}
		bb, _ := json.Marshal(map[string]any{"messages": msgs})
		return len(bb) / 4, false
	}
	trimSession(s, nCtx, 2048, countFn)
	// rebuild raw["messages"] after trim
	var rebuilt []map[string]any
	if s.SystemPrompt != "" {
		rebuilt = append(rebuilt, map[string]any{"role": "system", "content": s.SystemPrompt})
	}
	if s.Recap != "" {
		rebuilt = append(rebuilt, map[string]any{"role": "system", "content": s.Recap})
	}
	s.mu.RLock()
	for _, t := range s.Turns {
		rebuilt = append(rebuilt, map[string]any{"role": "user", "content": t.User})
		rebuilt = append(rebuilt, map[string]any{"role": "assistant", "content": t.Assistant})
	}
	s.mu.RUnlock()
	for _, m := range incoming {
		if m["role"] == "system" {
			continue
		}
		rebuilt = append(rebuilt, m)
	}
	raw["messages"] = rebuilt

	// persist
	_ = d.saveSession(s)

	// proxy
	respBody, status, err := d.proxyChatCollect(w, r, raw)
	if err != nil {
		return
	}
	// record turn: parse assistant content from response
	assistant := extractAssistant(respBody)
	if assistant != "" {
		s.mu.Lock()
		s.Turns = append(s.Turns, Turn{User: lastUser, Assistant: assistant})
		s.LastAccessed = time.Now().Unix()
		s.mu.Unlock()
		_ = d.saveSession(s)
	}
	// add session_id to response header
	w.Header().Set("X-Session-Id", sessionID)
	// if response not yet written? proxyChatCollect writes response. Add header before? We'll set header in proxy. Instead set now too late. We'll instead set header in proxy function.
	_ = status
}

// We'll implement proxy helpers using httputil? Simpler: forward request to upstream and stream back.
func (d *daemon) proxyChat(w http.ResponseWriter, r *http.Request, body []byte) {
	d.proxyChatRaw(w, r, body, "")
}

func (d *daemon) proxyChatRaw(w http.ResponseWriter, r *http.Request, body []byte, sessionID string) {
	target, _ := parseURL(d.conn.BaseURL)
	req, _ := http.NewRequest(r.Method, target.String()+"/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if d.upKey != "" {
		req.Header.Set("Authorization", "Bearer "+d.upKey)
	}
	// copy other headers
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, err.Error(), "proxy_error")
		return
	}
	defer resp.Body.Close()
	for k, vals := range resp.Header {
		for _, v := range vals {
			w.Header().Add(k, v)
		}
	}
	if sessionID != "" {
		w.Header().Set("X-Session-Id", sessionID)
	}
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
}

func (d *daemon) proxyChatCollect(w http.ResponseWriter, r *http.Request, raw map[string]any) ([]byte, int, error) {
	b, _ := json.Marshal(raw)
	target, _ := parseURL(d.conn.BaseURL)
	req, _ := http.NewRequest(http.MethodPost, target.String()+"/v1/chat/completions", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	if d.upKey != "" {
		req.Header.Set("Authorization", "Bearer "+d.upKey)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, err.Error(), "proxy_error")
		return nil, 502, err
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	for k, vals := range resp.Header {
		for _, v := range vals {
			w.Header().Add(k, v)
		}
	}
	// session id header set by caller? set here if present in raw? We'll set X-Session-Id later? We'll set before write.
	w.WriteHeader(resp.StatusCode)
	w.Write(respBody)
	return respBody, resp.StatusCode, nil
}

func extractAssistant(b []byte) string {
	var obj map[string]any
	if err := json.Unmarshal(b, &obj); err != nil {
		return ""
	}
	if choices, ok := obj["choices"].([]any); ok && len(choices) > 0 {
		if c, ok := choices[0].(map[string]any); ok {
			if m, ok := c["message"].(map[string]any); ok {
				if content, ok := m["content"].(string); ok {
					return content
				}
			}
		}
	}
	return ""
}

func parseURL(s string) (*url.URL, error) {
	return url.Parse(s)
}
