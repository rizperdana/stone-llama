// Package hf is a minimal HuggingFace Hub client: repo metadata, file
// fetch, search, and resolve URLs. Metadata calls are KB-scale; weight
// bytes flow only through pull's downloader.
package hf

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
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
		HTTP:    &http.Client{Timeout: 60 * time.Second},
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

func (c *Client) get(rawurl string) (*http.Response, error) {
	req, err := http.NewRequest(http.MethodGet, rawurl, nil)
	if err != nil {
		return nil, err
	}
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
	return c.HTTP.Do(req)
}

func (c *Client) statusErr(repoID string, code int) error {
	switch code {
	case http.StatusUnauthorized, http.StatusForbidden:
		return fmt.Errorf("%s: %w", repoID, ErrGated)
	case http.StatusNotFound:
		return fmt.Errorf("%s: %w", repoID, ErrNotFound)
	default:
		return fmt.Errorf("huggingface: HTTP %d for %s", code, repoID)
	}
}

// RepoMeta fetches repo tree + sizes + LFS hashes (?blobs=true). KB-scale.
func (c *Client) RepoMeta(repoID string) (Repo, error) {
	if !ValidRepoID(repoID) {
		return Repo{}, fmt.Errorf("invalid repository id %q", repoID)
	}
	resp, err := c.get(c.BaseURL + "/api/models/" + repoID + "?blobs=true")
	if err != nil {
		return Repo{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return Repo{}, c.statusErr(repoID, resp.StatusCode)
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

// Search returns candidate repo ids for a bare model name (KB-scale).
func (c *Client) Search(query string) ([]string, error) {
	resp, err := c.get(c.BaseURL + "/api/models?search=" + url.QueryEscape(query) + "&limit=20")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("huggingface search: HTTP %d", resp.StatusCode)
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
	resp, err := c.get(c.ResolveURL(repoID, rev, file))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, c.statusErr(repoID+"/"+file, resp.StatusCode)
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
		return nil, err
	}
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
	if offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	}
	return c.HTTP.Do(req)
}
