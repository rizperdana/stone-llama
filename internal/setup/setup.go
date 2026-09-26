// Package setup provisions stone-llama's runtime: a pinned uv binary,
// CPython 3.12, a pinned TabbyAPI checkout, the hash-locked Python
// dependencies for the chosen extra (cu13/cu12), and a smoke test.
//
// Plan §6 deviation: the plan specified downloading a TabbyAPI release
// tarball pinned by SHA-256. We pin the full git commit instead — a
// content-addressed checkout (git verify-tag not needed at this layer).
// The zero-download rule for tests precludes precomputing release-tarball
// hashes, and a commit is the smallest content-addressed selector we can
// embed and assert against.
package setup

import (
	"archive/tar"
	"compress/gzip"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"syscall"
	"time"
)

//go:embed runtime.lock.json
var lockJSON []byte

//go:embed requirements-cu13.lock requirements-cu12.lock
var lockFS embed.FS

// TabbyPin returns the pinned TabbyAPI commit from the embedded lock
// (startup banners; "" when unreadable).
func TabbyPin() string {
	var l runtimeLock
	if err := json.Unmarshal(lockJSON, &l); err != nil || l.TabbyAPI.Commit == "" {
		return ""
	}
	return l.TabbyAPI.Commit
}

// Step is one stage of the runtime bootstrap.
type Step struct {
	ID, Desc string
	Bytes    int64 // bytes attributable to this step (0 if unknown)
}

// Download is a file the bootstrap will fetch.
type Download struct {
	URL   string
	Bytes int64
}

// Plan is the full, side-effect-free description of a Run.
type Plan struct {
	Steps     []Step
	Downloads []Download
	Dest      string // runtime dir
	FreeBytes int64
}

// Options configures Preflight and Run.
type Options struct {
	RuntimeDir string // e.g. <DataDir>/runtime
	Extra      string // "cu13" (default when empty) or "cu12"
	Yes        bool   // consent already given
	// Confirm is invoked exactly once over the printed plan when !Yes.
	// Returning false aborts the run.
	Confirm func(Plan) bool
	Stdout  io.Writer
	// HeadSize returns the Content-Length for url, via HTTP HEAD by default.
	HeadSize func(url string) (int64, error)
	// Exec runs a command for a step and returns its combined output.
	// Default prefixes RuntimeDir/bin on PATH and sets UV_REQUIRE_HASHES=1
	// for the venv-deps step.
	Exec func(stepID, name string, args ...string) ([]byte, error)
	// LogPath is where step output is appended; default
	// <RuntimeDir>/logs/setup.log.
	LogPath string

	// Fetch returns a body for url (default: http.Get, status-checked).
	// Exposed as a test seam so tests never touch the network.
	Fetch func(url string) (io.ReadCloser, error)
	// ExpectedSHA returns the expected base16 sha256 for url.
	// Default: the uv sha256 from runtime.lock.json.
	ExpectedSHA func(url string) string

	// --- adoption (internal/sl-adopt/adoption-design.md §5) ---
	// Adopt is an explicit pointer path for detection (--adopt <path>); when
	// set it is probe 1 and only that candidate is considered first.
	Adopt string
	// Provision forces the full provision path (skips detection entirely).
	Provision bool
	// PromptReuse is invoked for adoption consent (message (a)) when !Yes.
	// Returning false aborts; when nil and !Yes, adoption is refused.
	PromptReuse func(Report) bool
	// Detect collects candidates for adoption (default: Detect itself).
	// A test seam so tests never scan the real machine.
	Detect func(DetectInput) []Candidate
}

// runtimeLock mirrors runtime.lock.json.
type runtimeLock struct {
	UV           uvLock   `json:"uv"`
	Python       specLock `json:"python"`
	TabbyAPI     refLock  `json:"tabbyapi"`
	Requirements reqLock  `json:"requirements"`
}
type uvLock struct {
	Version, URL, SHA256 string
}
type specLock struct{ Spec string }
type refLock struct{ Repo, Commit string }
type reqLock struct{ Cu13, Cu12 string }

// requirementsFile resolves Extra to an embedded lock filename.
func (l runtimeLock) requirementsFile(extra string) (string, error) {
	switch extra {
	case "", "cu13":
		return l.Requirements.Cu13, nil
	case "cu12":
		return l.Requirements.Cu12, nil
	}
	return "", fmt.Errorf("setup: unknown extra %q (want cu13 or cu12)", extra)
}

var directURLRE = regexp.MustCompile(`^([A-Za-z0-9_.-]+) @ (https://\S+)`)

// wheelURLs pulls the torch and exllamav3 direct-URL lines out of a lock.
func wheelURLs(text string) (torchURL, exlURL string, err error) {
	for _, line := range strings.Split(text, "\n") {
		m := directURLRE.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		u := strings.TrimSuffix(strings.TrimSpace(m[2]), "\\")
		switch m[1] {
		case "torch":
			torchURL = u
		case "exllamav3":
			exlURL = u
		}
	}
	if torchURL == "" || exlURL == "" {
		err = errors.New("setup: lock is missing torch or exllamav3 direct URL")
	}
	return torchURL, exlURL, err
}

// loadLock parses the embedded runtime.lock.json.
func loadLock() (runtimeLock, error) {
	var l runtimeLock
	if err := json.Unmarshal(lockJSON, &l); err != nil {
		return l, fmt.Errorf("setup: runtime.lock.json: %w", err)
	}
	return l, nil
}

func withDefaults(o Options) Options {
	if o.Detect == nil {
		o.Detect = Detect
	}
	if o.Extra == "" {
		o.Extra = "cu13"
	}
	if o.Stdout == nil {
		o.Stdout = os.Stdout
	}
	if o.HeadSize == nil {
		o.HeadSize = defaultHeadSize
	}
	if o.LogPath == "" && o.RuntimeDir != "" {
		o.LogPath = filepath.Join(o.RuntimeDir, "logs", "setup.log")
	}
	return o
}

// defaultHeadSize performs a real HTTP HEAD and returns Content-Length.
func defaultHeadSize(u string) (int64, error) {
	req, err := http.NewRequest(http.MethodHead, u, nil)
	if err != nil {
		return 0, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode > 399 {
		return 0, fmt.Errorf("HEAD %s: %s", u, resp.Status)
	}
	if resp.ContentLength < 0 {
		return 0, fmt.Errorf("HEAD %s: no content-length", u)
	}
	return resp.ContentLength, nil
}

// defaultFreeBytes reports free bytes for the path, walking up to the
// nearest existing ancestor (the per-platform disk probe fails on a
// nonexistent dir).
func defaultFreeBytes(p string) (int64, error) {
	cur := p
	var lastErr error
	for {
		if n, err := diskFree(cur); err == nil {
			return n, nil
		} else {
			lastErr = err
		}
		next := filepath.Dir(cur)
		if next == cur {
			return 0, lastErr
		}
		cur = next
	}
}

// defaultFetch is the production Fetch (network). Tests always inject Fetch.
func defaultFetch(u string) (io.ReadCloser, error) {
	resp, err := http.Get(u)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode > 399 {
		resp.Body.Close()
		return nil, fmt.Errorf("GET %s: %s", u, resp.Status)
	}
	return resp.Body, nil
}

func defaultExec(runtimeDir string) func(string, string, ...string) ([]byte, error) {
	bin := filepath.Join(runtimeDir, "bin")
	return func(stepID, name string, args ...string) ([]byte, error) {
		env := setEnv(os.Environ(), "PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
		if stepID == "venv-deps" {
			env = setEnv(env, "UV_REQUIRE_HASHES", "1")
		}
		cmd := exec.Command(name, args...)
		cmd.Env = env
		return cmd.CombinedOutput()
	}
}

func setEnv(env []string, key, val string) []string {
	prefix := key + "="
	out := make([]string, 0, len(env)+1)
	for _, kv := range env {
		if !strings.HasPrefix(kv, prefix) {
			out = append(out, kv)
		}
	}
	return append(out, prefix+val)
}

// Preflight computes a Plan with no side effects: HEAD sizes, statfs free
// bytes, destination paths, and the step list for the chosen extra.
func Preflight(opts Options) (Plan, error) {
	opts = withDefaults(opts)
	if opts.RuntimeDir == "" {
		return Plan{}, errors.New("setup: RuntimeDir is required")
	}
	lock, err := loadLock()
	if err != nil {
		return Plan{}, err
	}
	lockName, err := lock.requirementsFile(opts.Extra)
	if err != nil {
		return Plan{}, err
	}
	lockText, err := lockFS.ReadFile(lockName)
	if err != nil {
		return Plan{}, fmt.Errorf("setup: read %s: %w", lockName, err)
	}
	torchURL, exlURL, err := wheelURLs(string(lockText))
	if err != nil {
		return Plan{}, err
	}

	dl := make([]Download, 0, 3)
	for _, u := range []string{lock.UV.URL, torchURL, exlURL} {
		n, err := opts.HeadSize(u)
		if err != nil {
			return Plan{}, fmt.Errorf("setup: HEAD %s: %w", u, err)
		}
		dl = append(dl, Download{URL: u, Bytes: n})
	}

	free, err := defaultFreeBytes(opts.RuntimeDir)
	if err != nil {
		return Plan{}, fmt.Errorf("setup: free space on %s: %w", opts.RuntimeDir, err)
	}

	steps := []Step{
		{ID: "uv", Desc: "download and verify uv " + lock.UV.Version, Bytes: dl[0].Bytes},
		{ID: "python", Desc: "install Python " + lock.Python.Spec + " via uv"},
		{ID: "tabby", Desc: "clone TabbyAPI @ " + lock.TabbyAPI.Commit[:12]},
		{ID: "venv-deps", Desc: "create venv and install " + lockName, Bytes: dl[1].Bytes + dl[2].Bytes},
		{ID: "smoke", Desc: "verify exllamav3 import"},
	}

	return Plan{Steps: steps, Downloads: dl, Dest: opts.RuntimeDir, FreeBytes: free}, nil
}

// printPlan writes the plan to the configured writer.
func printPlan(w io.Writer, p Plan) {
	fmt.Fprintf(w, "setup plan\n")
	fmt.Fprintf(w, "  dest:  %s\n", p.Dest)
	fmt.Fprintf(w, "  free:  %d bytes\n", p.FreeBytes)
	fmt.Fprintf(w, "  downloads (%d):\n", len(p.Downloads))
	for _, d := range p.Downloads {
		fmt.Fprintf(w, "    %s (%d bytes)\n", d.URL, d.Bytes)
	}
	fmt.Fprintf(w, "  steps (%d):\n", len(p.Steps))
	for i, s := range p.Steps {
		fmt.Fprintf(w, "    %d. %s — %s (%d bytes)\n", i+1, s.ID, s.Desc, s.Bytes)
	}
}

// Run is Preflight → print plan → consent → execute steps in order with
// journal resume. On failure the returned error carries the last 20 log
// lines and a human-cause mapping.
func Run(opts Options) error {
	opts = withDefaults(opts)
	if opts.RuntimeDir == "" {
		return errors.New("setup: RuntimeDir is required")
	}
	if !opts.Yes && opts.Confirm == nil {
		return errors.New("setup: refusing to download without consent (pass --yes or run interactively)")
	}

	// Adoption runs before Preflight (design §8 slice 1 step 2) unless
	// --provision forces a full provision.
	if !opts.Provision {
		decision, err := detectAdoption(opts)
		if err != nil {
			return err
		}
		switch decision.kind {
		case adoptFound:
			// message (a) already printed by detectAdoption
			if !opts.Yes {
				if opts.PromptReuse != nil {
					if !opts.PromptReuse(decision.report) {
						return errors.New("setup: aborted (adoption not confirmed)")
					}
				} else if !opts.Confirm(Plan{}) {
					return errors.New("setup: aborted (adoption not confirmed)")
				}
			}
			return runAdoption(opts, decision.report)
		case adoptReject:
			if decision.explicit {
				return decision.err
			}
			// non-explicit candidate rejected → fall through to provision
		case adoptNone:
			// no candidate → fall through to provision
		}
	}

	plan, err := Preflight(opts)
	if err != nil {
		return err
	}
	printPlan(opts.Stdout, plan)
	if !opts.Yes && !opts.Confirm(plan) {
		return errors.New("setup: aborted (plan not confirmed)")
	}

	lock, err := loadLock()
	if err != nil {
		return err
	}
	if opts.Fetch == nil {
		opts.Fetch = defaultFetch
	}
	if opts.Exec == nil {
		opts.Exec = defaultExec(opts.RuntimeDir)
	}
	if opts.ExpectedSHA == nil {
		want := lock.UV.SHA256
		opts.ExpectedSHA = func(string) string { return want }
	}

	r := &runner{
		opts:       opts,
		lock:       lock,
		runtimeDir: opts.RuntimeDir,
		stdout:     opts.Stdout,
		logPath:    opts.LogPath,
	}
	return r.run(plan)
}

type runner struct {
	opts       Options
	lock       runtimeLock
	runtimeDir string
	stdout     io.Writer
	logPath    string

	lastURL  string
	lastName string
}

func (r *runner) run(plan Plan) error {
	if err := os.MkdirAll(r.runtimeDir, 0o750); err != nil {
		return fmt.Errorf("setup: mkdir %s: %w", r.runtimeDir, err)
	}

	j, err := loadJournal(filepath.Join(r.runtimeDir, "setup-journal.json"))
	if err != nil {
		return err
	}

	for _, st := range plan.Steps {
		if ts, ok := j.Done[st.ID]; ok {
			fmt.Fprintf(r.stdout, "skipping %s (done %s)\n", st.ID, ts)
			continue
		}
		fmt.Fprintf(r.stdout, "running %s: %s\n", st.ID, st.Desc)
		out, stepErr := r.step(st)
		if lerr := r.logStep(st.ID, out); lerr != nil {
			return fmt.Errorf("setup: write %s: %w", r.logPath, lerr)
		}
		if stepErr != nil {
			return r.fail(st, plan, stepErr)
		}
		j.Done[st.ID] = time.Now().UTC().Format(time.RFC3339)
		if err := saveJournal(filepath.Join(r.runtimeDir, "setup-journal.json"), j); err != nil {
			return err
		}
	}
	return nil
}

func (r *runner) step(st Step) ([]byte, error) {
	switch st.ID {
	case "uv":
		return r.stepUV()
	case "python":
		return r.exec("python", "uv", "python", "install", r.lock.Python.Spec)
	case "tabby":
		return r.stepTabby()
	case "venv-deps":
		return r.stepVenvDeps()
	case "smoke":
		return r.exec("smoke",
			filepath.Join(r.runtimeDir, "venv", "bin", "python"),
			"-c", "import exllamav3")
	}
	return nil, fmt.Errorf("setup: unknown step %q", st.ID)
}

func (r *runner) exec(stepID, name string, args ...string) ([]byte, error) {
	r.lastName = name
	return r.opts.Exec(stepID, name, args...)
}

func (r *runner) stepUV() ([]byte, error) {
	url := r.lock.UV.URL
	r.lastURL = url
	r.lastName = "uv"

	rc, err := r.opts.Fetch(url)
	if err != nil {
		return nil, err
	}
	defer rc.Close()

	bin := filepath.Join(r.runtimeDir, "bin")
	if err := os.MkdirAll(bin, 0o750); err != nil {
		return nil, fmt.Errorf("setup: mkdir %s: %w", bin, err)
	}
	tmp, err := os.CreateTemp(bin, ".uv-*.tar.gz")
	if err != nil {
		return nil, err
	}
	tmpName := tmp.Name()
	hasher := sha256.New()
	n, copyErr := io.Copy(io.MultiWriter(tmp, hasher), rc)
	closeErr := tmp.Close()
	if copyErr != nil {
		os.Remove(tmpName)
		return nil, fmt.Errorf("setup: download %s: %w", url, copyErr)
	}
	if closeErr != nil {
		os.Remove(tmpName)
		return nil, fmt.Errorf("setup: download %s: %w", url, closeErr)
	}

	got := hex.EncodeToString(hasher.Sum(nil))
	want := r.opts.ExpectedSHA(url)
	if !strings.EqualFold(got, want) {
		os.Remove(tmpName)
		return nil, fmt.Errorf("setup: sha256 mismatch for %s: got %s want %s", url, got, want)
	}

	out := filepath.Join(bin, "uv")
	if err := extractUV(tmpName, out); err != nil {
		os.Remove(tmpName)
		return nil, err
	}
	os.Remove(tmpName)
	return []byte(fmt.Sprintf("downloaded %d bytes, sha256 ok, extracted %s\n", n, out)), nil
}

// extractUV pulls the `uv` binary out of an archive into dest.
func extractUV(archive, dest string) error {
	f, err := os.Open(archive)
	if err != nil {
		return fmt.Errorf("setup: open archive: %w", err)
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return fmt.Errorf("setup: decode archive: %w", err)
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("setup: read tar: %w", err)
		}
		if hdr.Typeflag != tar.TypeReg || path.Base(hdr.Name) != "uv" {
			continue
		}
		tmp, err := os.CreateTemp(filepath.Dir(dest), ".uv-bin-*")
		if err != nil {
			return err
		}
		tmpName := tmp.Name()
		if _, err := io.Copy(tmp, tr); err != nil {
			tmp.Close()
			os.Remove(tmpName)
			return fmt.Errorf("setup: extract uv: %w", err)
		}
		if err := tmp.Close(); err != nil {
			os.Remove(tmpName)
			return err
		}
		if err := os.Chmod(tmpName, 0o755); err != nil {
			os.Remove(tmpName)
			return err
		}
		if err := os.Rename(tmpName, dest); err != nil {
			os.Remove(tmpName)
			return err
		}
		return nil
	}
	return errors.New("setup: 'uv' binary not found in tarball")
}

func (r *runner) stepTabby() ([]byte, error) {
	tabbyDir := filepath.Join(r.runtimeDir, "tabbyAPI")
	r.lastURL = r.lock.TabbyAPI.Repo
	r.lastName = "git"

	out1, err := r.exec("tabby", "git", "clone", r.lock.TabbyAPI.Repo, tabbyDir)
	if err != nil {
		return out1, err
	}
	out2, err := r.exec("tabby", "git", "-C", tabbyDir, "checkout", r.lock.TabbyAPI.Commit)
	return append(append([]byte{}, out1...), out2...), err
}

func (r *runner) stepVenvDeps() ([]byte, error) {
	lockName, err := r.lock.requirementsFile(r.opts.Extra)
	if err != nil {
		return nil, err
	}
	lockBytes, err := lockFS.ReadFile(lockName)
	if err != nil {
		return nil, fmt.Errorf("setup: read %s: %w", lockName, err)
	}
	lockPath := filepath.Join(r.runtimeDir, lockName)
	if err := os.WriteFile(lockPath, lockBytes, 0o644); err != nil {
		return nil, fmt.Errorf("setup: write %s: %w", lockPath, err)
	}
	r.lastName = "uv"

	venv := filepath.Join(r.runtimeDir, "venv")
	out1, err := r.exec("venv-deps", "uv", "venv", venv, "--python", r.lock.Python.Spec)
	if err != nil {
		return out1, err
	}
	out2, err := r.exec("venv-deps", "uv", "pip", "install",
		"--python", filepath.Join(venv, "bin", "python"),
		"--require-hashes", "-r", lockPath)
	return append(append([]byte{}, out1...), out2...), err
}

func (r *runner) logStep(stepID string, out []byte) error {
	if r.logPath == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(r.logPath), 0o750); err != nil {
		return err
	}
	f, err := os.OpenFile(r.logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o640)
	if err != nil {
		return err
	}
	defer f.Close()
	fmt.Fprintf(f, "[%s] step %s\n", time.Now().UTC().Format(time.RFC3339), stepID)
	if len(out) > 0 {
		if _, err := f.Write(out); err != nil {
			return err
		}
		if out[len(out)-1] != '\n' {
			if _, err := f.Write([]byte("\n")); err != nil {
				return err
			}
		}
	}
	return nil
}

func (r *runner) tailLog(n int) string {
	b, err := os.ReadFile(r.logPath)
	if err != nil {
		return "(log unavailable)"
	}
	lines := strings.Split(strings.TrimRight(string(b), "\n"), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return strings.Join(lines, "\n")
}

func (r *runner) fail(st Step, plan Plan, err error) error {
	msg := fmt.Sprintf("setup: %s failed: %v", st.ID, err)
	msg += fmt.Sprintf("\nlast 20 lines of %s:\n%s", r.logPath, r.tailLog(20))
	if c := humanCause(err, r.lastName, r.lastURL, plan); c != "" {
		msg += "\n" + c
	}
	return errors.New(msg)
}

// humanCause maps a step error to a remediation string, or "" if none.
func humanCause(err error, name, src string, plan Plan) string {
	if errors.Is(err, syscall.ENOSPC) || strings.Contains(err.Error(), "no space left") {
		var need int64
		for _, d := range plan.Downloads {
			need += d.Bytes
		}
		return fmt.Sprintf("disk full: need %d bytes, %d bytes free on %s", need, plan.FreeBytes, plan.Dest)
	}

	var unk x509.UnknownAuthorityError
	var cert *tls.CertificateVerificationError
	if errors.As(err, &unk) || errors.As(err, &cert) ||
		strings.Contains(err.Error(), "x509:") || strings.HasPrefix(err.Error(), "tls:") {
		return fmt.Sprintf("TLS failure reaching %s — check proxy/CA certs", hostOr(src, "the server"))
	}

	if errors.Is(err, exec.ErrNotFound) {
		return fmt.Sprintf("missing %s — install it or check PATH", name)
	}
	var eerr *exec.Error
	if errors.As(err, &eerr) {
		return fmt.Sprintf("missing %s — install it or check PATH", eerr.Name)
	}
	return ""
}

func hostOr(raw, fallback string) string {
	u, err := url.Parse(raw)
	if err != nil || u.Host == "" {
		return fallback
	}
	return u.Hostname()
}

// journal is the resumable-done marker file.
type journal struct {
	Done map[string]string `json:"done"`
}

func loadJournal(p string) (journal, error) {
	j := journal{Done: map[string]string{}}
	b, err := os.ReadFile(p)
	if errors.Is(err, os.ErrNotExist) {
		return j, nil
	}
	if err != nil {
		return j, fmt.Errorf("setup: read journal %s: %w", p, err)
	}
	if err := json.Unmarshal(b, &j); err != nil {
		return j, fmt.Errorf("setup: parse journal %s: %w", p, err)
	}
	if j.Done == nil {
		j.Done = map[string]string{}
	}
	return j, nil
}

// saveJournal writes the journal atomically (temp + rename).
func saveJournal(p string, j journal) error {
	b, err := json.Marshal(j)
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(p), ".setup-journal-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(b); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return err
	}
	if err := os.Rename(tmpName, p); err != nil {
		os.Remove(tmpName)
		return err
	}
	return nil
}
