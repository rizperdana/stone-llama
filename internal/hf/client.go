// Package hf is a minimal HuggingFace Hub client: repo metadata, file
// fetch, search, and resolve URLs. Metadata calls are KB-scale; weight
// bytes flow only through pull's downloader.
package hf

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

const DefaultBaseURL = "https://huggingface.co"

// maxMetaFile caps FetchFile — config/tokenizer JSONs are KBs; anything
// megabyte-scale here means we're pointed at a weight file.
const maxMetaFile = 8 << 20

// requestTimeout bounds dial+response for every metadata call.
const requestTimeout = 60 * time.Second

// Bounded retry: maxAttempts total, only on 5xx/429/timeouts.
const (
	maxAttempts = 3
	backoffBase = 250 * time.Millisecond
)

// Injectable so retry tests run instantly.
var (
	sleepFn  = time.Sleep
	jitterFn = func(d time.Duration) time.Duration {
		return d/2 + time.Duration(rand.Int64N(int64(d/2)+1))
	}
)

var (
	ErrNotFound = errors.New("repository not found on HuggingFace")
	ErrGated    = errors.New("repository is gated")
)

type Client struct {
	BaseURL string
	Token   string
	HTTP    *http.Client
}

func NewClient(baseURL, token string) *Client {
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	return &Client{
		BaseURL: strings.TrimRight(baseURL, "/"),
		Token:   token,
		HTTP:    &http.Client{Timeout: requestTimeout},
	}
}

// LoadToken resolves the HF token: env value wins over the 0600 file
// written by `stone-llama login`. Never logs either value.
func LoadToken(envValue, tokenFile string) string {
	if v := strings.TrimSpace(envValue); v != "" {
		return v
	}
	data, err := os.ReadFile(tokenFile)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

// File is one sibling entry of a repo.
type File struct {
	Path   string
	Size   int64
	SHA256 string // LFS object id; "" for non-LFS files
}

type Repo struct {
	ID    string
	SHA   string // revision (commit sha) to download at
	Gated bool
	Files []File
}

// DirFiles returns files whose directory prefix equals dir ("" = repo root).
func (r Repo) DirFiles(dir string) []File {
	var out []File
	for _, f := range r.Files {
		if fileDir(f.Path) == dir {
			out = append(out, f)
		}
	}
	return out
}
func (r Repo) HasSuffix(suffix string) bool {
	suffix = strings.ToLower(suffix)
	for _, f := range r.Files {
		if strings.HasSuffix(strings.ToLower(f.Path), suffix) {
			return true
		}
	}
	return false
}

func fileDir(path string) string {
	if i := strings.LastIndex(path, "/"); i >= 0 {
		return path[:i]
	}
	return ""
}

// ValidRepoID guards URL construction against injection.
func ValidRepoID(id string) bool {
	if id == "" || strings.HasPrefix(id, "/") || strings.HasSuffix(id, "/") || strings.Contains(id, "//") {
		return false
	}
	for _, r := range id {
		ok := r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' ||
			r == '-' || r == '_' || r == '.' || r == '/'
		if !ok {
			return false
		}
	}
	for _, seg := range strings.Split(id, "/") {
		if seg == "." || seg == ".." || seg == "" {
			return false // path traversal
		}
	}
	return true
}

// isTimeout reports whether err is a network timeout (client timeout,
// dial timeout, deadline).
func isTimeout(err error) bool {
	var ne net.Error
	return errors.As(err, &ne) && ne.Timeout()
}

// wrapGet shapes a transport error as op+URL+timeout so callers can see
// which URL failed and how long they waited.
func wrapGet(rawurl string, err error) error {
	if isTimeout(err) {
		return fmt.Errorf("hf: GET %s: timeout after %s: %w", rawurl, requestTimeout, err)
	}
	return fmt.Errorf("hf: GET %s: %w", rawurl, err)
}

// get GETs rawurl with bounded retries: maxAttempts total on 5xx, 429
// and timeouts only — 401/403/404/other 4xx return immediately for the
// caller to judge.
func (c *Client) get(rawurl string) (*http.Response, error) {
	var err error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		if attempt > 1 {
			sleepFn(jitterFn(backoffBase << (attempt - 2)))
		}
		var resp *http.Response
		resp, err = c.do(rawurl)
		if err == nil {
			if resp.StatusCode < 500 && resp.StatusCode != http.StatusTooManyRequests {
				return resp, nil
			}
			resp.Body.Close()
			err = fmt.Errorf("hf: GET %s: HTTP %d", rawurl, resp.StatusCode)
			if attempt == maxAttempts {
				return nil, fmt.Errorf("%s after %d attempts", err, maxAttempts)
			}
			continue
		}
		if !isTimeout(err) || attempt == maxAttempts {
			return nil, err
		}
	}
	return nil, err
}

func (c *Client) do(rawurl string) (*http.Response, error) {
	req, err := http.NewRequest(http.MethodGet, rawurl, nil)
	if err != nil {
		return nil, fmt.Errorf("hf: GET %s: %w", rawurl, err)
	}
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, wrapGet(rawurl, err)
	}
	return resp, nil
}

// authKind classifies an HTTP 401 from the HF API. HF answers 401 to
// anonymous callers for BOTH gated and absent repos, so only the body
// tells them apart (gated repos carry "gated": true/auto or "access to
// model … is restricted" wording).
type authKind int

const (
	authAbsent authKind = iota // repo does not exist
	authGated                  // gated repo — needs login/HF_TOKEN
	authToken                  // caller presented a token HF rejects
)

func classify401(body []byte) authKind {
	b := strings.ToLower(string(body))
	// Any non-false "gated" field marks a gated repo; HF renders it
	// true/auto/manual with or without JSON whitespace.
	gatedField := strings.Contains(b, `"gated":`) &&
		!strings.Contains(b, `"gated": false`) && !strings.Contains(b, `"gated":false`)
	switch {
	case gatedField, strings.Contains(b, "is restricted"), strings.Contains(b, "gatedrepoerror"):
		return authGated
	case strings.Contains(b, "invalid_token"), strings.Contains(b, "invalid token"),
		strings.Contains(b, "token is invalid"):
		return authToken
	default:
		return authAbsent
	}
}

// statusErr maps a non-retryable HTTP status to an actionable error
// keyed by the request URL; auth failures say what to do next. body is
// the (bounded) response body used to classify 401.
func statusErr(rawurl string, code int, body []byte) error {
	switch code {
	case http.StatusUnauthorized:
		switch classify401(body) {
		case authGated:
			return fmt.Errorf("hf: 401 for %s — repository is gated; run 'stone-llama login' or set HF_TOKEN: %w", rawurl, ErrGated)
		case authToken:
			return fmt.Errorf("hf: 401 unauthorized for %s — token invalid; run 'stone-llama login'", rawurl)
		default:
			return fmt.Errorf("hf: 401 for %s — repository not found; check the id (owner/repo) for a typo: %w", rawurl, ErrNotFound)
		}
	case http.StatusForbidden:
		return fmt.Errorf("hf: 403 forbidden for %s — token lacks access or the repo is gated; run 'stone-llama login' with a token that can read it: %w", rawurl, ErrGated)
	case http.StatusNotFound:
		return fmt.Errorf("hf: 404 for %s: %w", rawurl, ErrNotFound)
	default:
		return fmt.Errorf("hf: GET %s: HTTP %d", rawurl, code)
	}
}

// RepoMeta fetches repo tree + sizes + LFS hashes (?blobs=true). KB-scale.
func (c *Client) RepoMeta(repoID string) (Repo, error) {
	return c.RepoMetaRev(repoID, "")
}

// RepoMetaRev is RepoMeta for an explicit branch or tag; revision ""
// means the repo's default ref.
func (c *Client) RepoMetaRev(repoID, revision string) (Repo, error) {
	if !ValidRepoID(repoID) {
		return Repo{}, fmt.Errorf("invalid repository id %q", repoID)
	}
	rawurl := c.BaseURL + "/api/models/" + repoID + "?blobs=true"
	if revision != "" {
		rawurl += "&revision=" + url.QueryEscape(revision)
	}
	resp, err := c.get(rawurl)
	if err != nil {
		return Repo{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		return Repo{}, statusErr(rawurl, resp.StatusCode, b)
	}

	var body struct {
		ID       string          `json:"id"`
		SHA      string          `json:"sha"`
		Gated    json.RawMessage `json:"gated"`
		Siblings []struct {
			RFilename string `json:"rfilename"`
			Size      *int64 `json:"size"`
			LFS       *struct {
				OID    string `json:"oid"`
				SHA256 string `json:"sha256"`
				Size   *int64 `json:"size"`
			} `json:"lfs"`
		} `json:"siblings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return Repo{}, fmt.Errorf("decode repo metadata: %w", err)
	}

	repo := Repo{ID: body.ID, SHA: body.SHA}
	if repo.ID == "" {
		repo.ID = repoID
	}
	gated := strings.TrimSpace(string(body.Gated))
	repo.Gated = gated != "" && gated != "false" && gated != "null"
	for _, s := range body.Siblings {
		f := File{Path: s.RFilename}
		if s.Size != nil {
			f.Size = *s.Size
		}
		if s.LFS != nil {
			oid := s.LFS.OID
			if oid == "" {
				oid = s.LFS.SHA256 // current hub API spells it sha256
			}
			f.SHA256 = strings.ToLower(oid)
			if s.LFS.Size != nil {
				f.Size = *s.LFS.Size
			}
		}
		repo.Files = append(repo.Files, f)
	}
	return repo, nil
}

// GetJSON GETs an API path against BaseURL and decodes it into dst;
// token, bounded retry and timeout surfacing come from the shared get().
func (c *Client) GetJSON(path string, dst any) error {
	rawurl := c.BaseURL + path
	resp, err := c.get(rawurl)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		return statusErr(rawurl, resp.StatusCode, b)
	}
	if err := json.NewDecoder(resp.Body).Decode(dst); err != nil {
		return fmt.Errorf("decode %s: %w", rawurl, err)
	}
	return nil
}

// Search returns candidate repo ids for a bare model name (KB-scale).
func (c *Client) Search(query string) ([]string, error) {
	rawurl := c.BaseURL + "/api/models?search=" + url.QueryEscape(query) + "&limit=20"
	resp, err := c.get(rawurl)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		return nil, statusErr(rawurl, resp.StatusCode, b)
	}
	var body []struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("decode search results: %w", err)
	}
	var ids []string
	for _, r := range body {
		if r.ID != "" {
			ids = append(ids, r.ID)
		}
	}
	return ids, nil
}

// ResolveURL is the byte URL for a file at a revision.
func (c *Client) ResolveURL(repoID, rev, file string) string {
	if rev == "" {
		rev = "main"
	}
	return c.BaseURL + "/" + repoID + "/resolve/" + rev + "/" + file
}

// FetchFile downloads one small metadata file (config.json etc.), capped
// at maxMetaFile. KB-scale by design (A5 gate).
func (c *Client) FetchFile(repoID, rev, file string) ([]byte, error) {
	if strings.ContainsAny(file, " \t") {
		return nil, fmt.Errorf("invalid file path %q", file)
	}
	rawurl := c.ResolveURL(repoID, rev, file)
	resp, err := c.get(rawurl)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		return nil, statusErr(rawurl, resp.StatusCode, b)
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxMetaFile+1))
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", file, err)
	}
	if len(data) > maxMetaFile {
		return nil, fmt.Errorf("%s exceeds %d bytes — not a metadata file", file, maxMetaFile)
	}
	return data, nil
}

// FetchRange issues a GET against a resolve URL, optionally requesting
// bytes from offset (resumable download). The caller reads the body.
func (c *Client) FetchRange(rawurl string, offset int64) (*http.Response, error) {
	req, err := http.NewRequest(http.MethodGet, rawurl, nil)
	if err != nil {
		return nil, fmt.Errorf("hf: GET %s: %w", rawurl, err)
	}
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
	if offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, wrapGet(rawurl, err)
	}
	return resp, nil
}
