package hf

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

func TestValidRepoID(t *testing.T) {
	ok := []string{"org/repo", "Org-Name/repo_2.0", "a/b/c"}
	bad := []string{"", "/x", "x/", "a//b", "with space", "evil\n", "../etc", "q?x=1"}
	for _, s := range ok {
		if !ValidRepoID(s) {
			t.Errorf("ValidRepoID(%q) = false, want true", s)
		}
	}
	for _, s := range bad {
		if ValidRepoID(s) {
			t.Errorf("ValidRepoID(%q) = true, want false", s)
		}
	}
}

const metaJSON = `{
	"id": "org/repo",
	"sha": "abc123",
	"gated": false,
	"siblings": [
		{"rfilename": "config.json", "size": 100},
		{"rfilename": "model.safetensors", "lfs": {"oid": "DEADBEEF", "size": 4096}},
		{"rfilename": "sub/other.safetensors", "size": 2048}
	]
}`

func newTestServer(t *testing.T, token string, handler http.Handler) (*httptest.Server, *Client) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return srv, NewClient(srv.URL, token)
}

func TestRepoMetaParsesBlobsAndDetectsAuthHeader(t *testing.T) {
	var gotAuth string
	_, c := newTestServer(t, "hf_secret", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		if r.URL.Path != "/api/models/org/repo" || r.URL.Query().Get("blobs") != "true" {
			t.Errorf("unexpected request: %s?%s", r.URL.Path, r.URL.RawQuery)
		}
		w.Write([]byte(metaJSON))
	}))

	repo, err := c.RepoMeta("org/repo")
	if err != nil {
		t.Fatal(err)
	}
	if gotAuth != "Bearer hf_secret" {
		t.Errorf("Authorization = %q", gotAuth)
	}
	if repo.ID != "org/repo" || repo.SHA != "abc123" || repo.Gated {
		t.Errorf("repo = %+v", repo)
	}
	if len(repo.Files) != 3 {
		t.Fatalf("files = %d", len(repo.Files))
	}
	if repo.Files[0].Path != "config.json" || repo.Files[0].Size != 100 {
		t.Errorf("file0 = %+v", repo.Files[0])
	}
	w := repo.Files[1]
	if w.Size != 4096 || w.SHA256 != "deadbeef" {
		t.Errorf("weights file = %+v, want size 4096 oid lowercased", w)
	}
	if d := repo.DirFiles("sub"); len(d) != 1 || d[0].Path != "sub/other.safetensors" {
		t.Errorf("DirFiles(sub) = %+v", d)
	}
	if !repo.HasSuffix(".safetensors") || repo.HasSuffix(".gguf") {
		t.Error("HasSuffix wrong")
	}
}

func TestRepoMetaStatusMapping(t *testing.T) {
	for _, tc := range []struct {
		code int
		want error
	}{
		{404, ErrNotFound},
		{401, ErrGated},
		{403, ErrGated},
	} {
		_, c := newTestServer(t, "", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(tc.code)
		}))
		_, err := c.RepoMeta("org/repo")
		if !errors.Is(err, tc.want) {
			t.Errorf("HTTP %d → %v, want %v", tc.code, err, tc.want)
		}
	}
}

func TestRepoMetaGatedFlag(t *testing.T) {
	_, c := newTestServer(t, "", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"id":"org/g","sha":"s","gated":"auto","siblings":[]}`))
	}))
	repo, err := c.RepoMeta("org/g")
	if err != nil {
		t.Fatal(err)
	}
	if !repo.Gated {
		t.Error("Gated = false, want true for gated:auto")
	}
}

func TestRepoMetaRejectsBadIDBeforeNetwork(t *testing.T) {
	var hit bool
	_, c := newTestServer(t, "", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { hit = true }))
	if _, err := c.RepoMeta("../../etc/passwd"); err == nil {
		t.Fatal("want error")
	}
	if hit {
		t.Error("request must not leave the process for a bad id")
	}
}

func TestSearch(t *testing.T) {
	_, c := newTestServer(t, "", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("search") != "tiny" || r.URL.Query().Get("limit") != "20" {
			t.Errorf("query = %s", r.URL.RawQuery)
		}
		w.Write([]byte(`[{"id":"org/tiny-exl3"},{"id":"other/tinything"}]`))
	}))
	ids, err := c.Search("tiny")
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 2 || ids[0] != "org/tiny-exl3" {
		t.Errorf("ids = %v", ids)
	}
}

func TestFetchFile(t *testing.T) {
	_, c := newTestServer(t, "", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/config.json"):
			w.Write([]byte(`{"ok":true}`))
		case strings.HasSuffix(r.URL.Path, "/missing.json"):
			w.WriteHeader(404)
		case strings.HasSuffix(r.URL.Path, "/gated.json"):
			w.WriteHeader(401)
		}
	}))
	data, err := c.FetchFile("org/repo", "abc123", "config.json")
	if err != nil || string(data) != `{"ok":true}` {
		t.Fatalf("FetchFile = %q, %v", data, err)
	}
	if _, err := c.FetchFile("org/repo", "abc123", "missing.json"); !errors.Is(err, ErrNotFound) {
		t.Errorf("missing → %v, want ErrNotFound", err)
	}
	if _, err := c.FetchFile("org/repo", "abc123", "gated.json"); !errors.Is(err, ErrGated) {
		t.Errorf("gated → %v, want ErrGated", err)
	}
	if _, err := c.FetchFile("org/repo", "abc123", "bad path.json"); err == nil {
		t.Error("path with space must fail")
	}
}

func TestResolveURL(t *testing.T) {
	c := NewClient("", "")
	want := "https://huggingface.co/org/repo/resolve/main/config.json"
	if got := c.ResolveURL("org/repo", "", "config.json"); got != want {
		t.Errorf("ResolveURL = %q, want %q", got, want)
	}
}

func TestLoadTokenEnvWinsOverFile(t *testing.T) {
	file := t.TempDir() + "/tok"
	if got := LoadToken(" hf_env ", file); got != "hf_env" {
		t.Errorf("env token = %q", got)
	}
	if err := os.WriteFile(file, []byte("hf_from_file\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := LoadToken("", file); got != "hf_from_file" {
		t.Errorf("file token = %q", got)
	}
	if got := LoadToken("hf_env", file); got != "hf_env" {
		t.Errorf("env must win = %q", got)
	}
}

// noRetrySleep neutralizes backoff so retry tests run instantly.
func noRetrySleep(t *testing.T) {
	t.Helper()
	origSleep, origJitter := sleepFn, jitterFn
	sleepFn = func(time.Duration) {}
	jitterFn = func(d time.Duration) time.Duration { return d }
	t.Cleanup(func() { sleepFn, jitterFn = origSleep, origJitter })
}

func TestRetryFiveXXThenSucceedsInThreeAttempts(t *testing.T) {
	noRetrySleep(t)
	var hits int
	_, c := newTestServer(t, "", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		switch hits {
		case 1, 2:
			w.WriteHeader(http.StatusInternalServerError)
		default:
			w.Write([]byte(metaJSON))
		}
	}))
	repo, err := c.RepoMeta("org/repo")
	if err != nil {
		t.Fatal(err)
	}
	if hits != 3 {
		t.Errorf("attempts = %d, want exactly 3", hits)
	}
	if repo.SHA != "abc123" {
		t.Errorf("SHA = %q, want abc123", repo.SHA)
	}
}

func TestNoRetryOn404(t *testing.T) {
	var hits int
	_, c := newTestServer(t, "", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		w.WriteHeader(http.StatusNotFound)
	}))
	_, err := c.RepoMeta("org/repo")
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("err = %v, want ErrNotFound", err)
	}
	if hits != 1 {
		t.Errorf("attempts = %d, want 1 (no retry on 4xx)", hits)
	}
}

func TestAuthErrorsDistinctAndUnretried(t *testing.T) {
	for _, tc := range []struct {
		code int
		want []string
	}{
		{http.StatusUnauthorized, []string{
			"hf: 401 unauthorized for ",
			"token invalid; run 'stone-llama login'",
		}},
		{http.StatusForbidden, []string{
			"hf: 403 forbidden for ",
			"token lacks access or the repo is gated",
			"run 'stone-llama login' with a token that can read it",
		}},
	} {
		var hits int
		_, c := newTestServer(t, "", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			hits++
			w.WriteHeader(tc.code)
		}))
		_, err := c.RepoMeta("org/repo")
		if err == nil {
			t.Errorf("HTTP %d: want error", tc.code)
			continue
		}
		for _, sub := range tc.want {
			if !strings.Contains(err.Error(), sub) {
				t.Errorf("HTTP %d: err %q missing %q", tc.code, err, sub)
			}
		}
		if hits != 1 {
			t.Errorf("HTTP %d: attempts = %d, want 1 (4xx never retried)", tc.code, hits)
		}
	}
}

func TestTimeoutSurfacesTimeoutAfter(t *testing.T) {
	noRetrySleep(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(80 * time.Millisecond)
		w.Write([]byte(metaJSON))
	}))
	t.Cleanup(srv.Close)
	c := NewClient(srv.URL, "")
	c.HTTP.Timeout = 10 * time.Millisecond
	_, err := c.RepoMeta("org/repo")
	if err == nil {
		t.Fatal("want timeout error")
	}
	for _, sub := range []string{"timeout after", "hf: GET "} {
		if !strings.Contains(err.Error(), sub) {
			t.Errorf("err %q missing %q", err, sub)
		}
	}
}
