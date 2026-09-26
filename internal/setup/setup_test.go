package setup

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
)

type call struct {
	stepID, name string
	args         []string
}

// stubHead sizes by substring match against the URL.
func stubHead(u string) (int64, error) {
	switch {
	case strings.Contains(u, "astral-sh/uv"):
		return 1000, nil
	case strings.Contains(u, "download-r2"):
		return 2000, nil
	case strings.Contains(u, "exllamav3/releases"):
		return 3000, nil
	}
	return 0, fmt.Errorf("unexpected HEAD %s", u)
}

// uvURL returns the embedded uv download URL.
func uvURL(t *testing.T) string {
	t.Helper()
	lock, err := loadLock()
	if err != nil {
		t.Fatal(err)
	}
	return lock.UV.URL
}

// TestDefaultFreeBytes pins the platform disk probe: free bytes > 0 on
// an existing dir, and the walk-up finds an ancestor when the path
// doesn't exist yet. The Windows GetDiskFreeSpaceExW twin compiles via
// GOOS=windows but is proven only by that compile check (v1 does not
// claim Windows runtime support).
func TestDefaultFreeBytes(t *testing.T) {
	dir := t.TempDir()
	free, err := defaultFreeBytes(dir)
	if err != nil {
		t.Fatalf("free bytes on %s: %v", dir, err)
	}
	if free <= 0 {
		t.Errorf("free bytes = %d, want > 0", free)
	}
	deep := filepath.Join(dir, "not", "created", "yet")
	if _, err := defaultFreeBytes(deep); err != nil {
		t.Errorf("walk-up from %s: %v", deep, err)
	}
}

type recorder struct {
	calls    []call
	fetches  []string
	heads    []string
	confirms []Plan
}

// testTarball is a tiny tar.gz containing an executable `uv` member, plus
// its sha256 hex, computed once at package init.
var testTarball, testTarballSHA = func() ([]byte, string) {
	body := []byte("#!/bin/sh\ntrue\n")
	hdr := &tar.Header{
		Name:     "uv-x86_64-unknown-linux-gnu/uv",
		Mode:     0o755,
		Typeflag: tar.TypeReg,
		Size:     int64(len(body)),
	}
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	if err := tw.WriteHeader(hdr); err != nil {
		panic(err)
	}
	if _, err := tw.Write(body); err != nil {
		panic(err)
	}
	if err := tw.Close(); err != nil {
		panic(err)
	}
	if err := gz.Close(); err != nil {
		panic(err)
	}
	sha := sha256.Sum256(buf.Bytes())
	return buf.Bytes(), hex.EncodeToString(sha[:])
}()

// buildOpts returns fully-stubbed options so Run never touches the network
// or the real uv/git/python binaries.
func buildOpts(rt string, rec *recorder, stdout io.Writer, yes bool) Options {
	return Options{
		RuntimeDir: rt,
		Yes:        yes,
		Stdout:     stdout,
		HeadSize: func(u string) (int64, error) {
			rec.heads = append(rec.heads, u)
			return stubHead(u)
		},
		Fetch: func(u string) (io.ReadCloser, error) {
			rec.fetches = append(rec.fetches, u)
			if strings.Contains(u, "astral-sh/uv") {
				return io.NopCloser(bytes.NewReader(testTarball)), nil
			}
			return nil, fmt.Errorf("unexpected fetch %s", u)
		},
		ExpectedSHA: func(string) string { return testTarballSHA },
		Exec: func(stepID, name string, args ...string) ([]byte, error) {
			rec.calls = append(rec.calls, call{stepID, name, args})
			return []byte("ok\n"), nil
		},
		// never scan the real machine for adoption candidates
		Detect: func(DetectInput) []Candidate { return nil },
	}
}

func logPathFor(rt string) string {
	return filepath.Join(rt, "logs", "setup.log")
}

// TestPreflightPlan checks plan arithmetic, downloads, free bytes, and extra
// selection without executing anything.
func TestPreflightPlan(t *testing.T) {
	rt := filepath.Join(t.TempDir(), "runtime")
	plan, err := Preflight(Options{RuntimeDir: rt, HeadSize: stubHead})
	if err != nil {
		t.Fatalf("Preflight: %v", err)
	}

	wantIDs := []string{"uv", "python", "tabby", "venv-deps", "smoke"}
	if len(plan.Steps) != len(wantIDs) {
		t.Fatalf("step count %d", len(plan.Steps))
	}
	for i, id := range wantIDs {
		if plan.Steps[i].ID != id {
			t.Errorf("step %d: got %q want %q", i, plan.Steps[i].ID, id)
		}
	}

	if plan.Dest != rt {
		t.Errorf("Dest = %q want %q", plan.Dest, rt)
	}
	if plan.FreeBytes <= 0 {
		t.Errorf("FreeBytes = %d, want > 0", plan.FreeBytes)
	}
	if len(plan.Downloads) != 3 {
		t.Fatalf("downloads = %d, want 3", len(plan.Downloads))
	}
	wantSizes := map[string]int64{
		plan.Downloads[0].URL: 1000,
		plan.Downloads[1].URL: 2000,
		plan.Downloads[2].URL: 3000,
	}
	for _, d := range plan.Downloads {
		if d.Bytes != wantSizes[d.URL] {
			t.Errorf("download %q size = %d", d.URL, d.Bytes)
		}
	}
	if plan.Steps[0].Bytes != 1000 {
		t.Errorf("uv step bytes = %d, want 1000", plan.Steps[0].Bytes)
	}
	if plan.Steps[3].Bytes != 5000 {
		t.Errorf("venv-deps step bytes = %d, want 5000", plan.Steps[3].Bytes)
	}

	plan12, err := Preflight(Options{RuntimeDir: rt, Extra: "cu12", HeadSize: stubHead})
	if err != nil {
		t.Fatalf("Preflight cu12: %v", err)
	}
	if !strings.Contains(plan12.Downloads[1].URL, "cu128") {
		t.Errorf("cu12 torch URL = %q, want cu128", plan12.Downloads[1].URL)
	}
	if !strings.Contains(plan12.Steps[3].Desc, "requirements-cu12.lock") {
		t.Errorf("cu12 desc = %q", plan12.Steps[3].Desc)
	}

	// Invalid extra is rejected before any HEAD.
	called := 0
	_, err = Preflight(Options{
		RuntimeDir: rt,
		Extra:      "rocm",
		HeadSize:   func(string) (int64, error) { called++; return 0, nil },
	})
	if err == nil {
		t.Fatal("expected error for invalid extra")
	}
	if !strings.Contains(err.Error(), "unknown extra") {
		t.Errorf("err = %v, want unknown extra", err)
	}
	if called != 0 {
		t.Errorf("HEAD called %d times before validation", called)
	}
}

// TestRunRefusesWithoutConsent ensures Run aborts before any network use.
func TestRunRefusesWithoutConsent(t *testing.T) {
	rt := filepath.Join(t.TempDir(), "runtime")
	heads := 0
	err := Run(Options{
		RuntimeDir: rt,
		Yes:        false,
		Stdout:     io.Discard,
		HeadSize:   func(string) (int64, error) { heads++; return 0, nil },
	})
	if err == nil {
		t.Fatal("expected consent-refusal error")
	}
	if err.Error() != "setup: refusing to download without consent (pass --yes or run interactively)" {
		t.Errorf("err = %q", err.Error())
	}
	if heads != 0 {
		t.Errorf("HEAD called %d times, want 0", heads)
	}
}

// TestRunHappyPath runs the full flow with stubs and asserts exactly ordered
// invocations, the uv binary on disk, journal completion, and log contents.
func TestRunHappyPath(t *testing.T) {
	rt := filepath.Join(t.TempDir(), "runtime")
	var buf bytes.Buffer
	rec := &recorder{}
	opts := buildOpts(rt, rec, &buf, false)
	opts.Confirm = func(p Plan) bool {
		rec.confirms = append(rec.confirms, p)
		return true
	}

	if err := Run(opts); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if len(rec.confirms) != 1 {
		t.Fatalf("Confirm called %d times, want 1", len(rec.confirms))
	}
	cp := rec.confirms[0]
	if cp.FreeBytes <= 0 || len(cp.Downloads) != 3 {
		t.Errorf("confirm plan: free=%d downloads=%d", cp.FreeBytes, len(cp.Downloads))
	}
	if cp.Steps[0].Bytes != 1000 {
		t.Errorf("confirm plan uv bytes = %d, want 1000", cp.Steps[0].Bytes)
	}

	if len(rec.fetches) != 1 {
		t.Fatalf("Fetch called %d times, want 1", len(rec.fetches))
	}

	want := []call{
		{"python", "uv", []string{"python", "install", "3.12"}},
		{"tabby", "git", []string{"clone", "https://github.com/theroyallab/tabbyAPI", filepath.Join(rt, "tabbyAPI")}},
		{"tabby", "git", []string{"-C", filepath.Join(rt, "tabbyAPI"), "checkout", "f07131cd8fe34e449fe87cdd3a066b52b96d3cac"}},
		{"venv-deps", "uv", []string{"venv", filepath.Join(rt, "venv"), "--python", "3.12"}},
		{"venv-deps", "uv", []string{"pip", "install", "--python", filepath.Join(rt, "venv", "bin", "python"), "--require-hashes", "-r", filepath.Join(rt, "requirements-cu13.lock")}},
		{"smoke", filepath.Join(rt, "venv", "bin", "python"), []string{"-c", "import exllamav3"}},
	}
	if len(rec.calls) != len(want) {
		t.Fatalf("Exec calls = %d, want %d\n%v", len(rec.calls), len(want), rec.calls)
	}
	for i, w := range want {
		got := rec.calls[i]
		if got.stepID != w.stepID || got.name != w.name {
			t.Errorf("call %d: got (%s,%q) want (%s,%q)", i, got.stepID, got.name, w.stepID, w.name)
		}
		if len(got.args) != len(w.args) {
			t.Errorf("call %d args len %d want %d", i, len(got.args), len(w.args))
			continue
		}
		for j, a := range got.args {
			if a != w.args[j] {
				t.Errorf("call %d arg %d: got %q want %q", i, j, a, w.args[j])
			}
		}
	}

	uvBin := filepath.Join(rt, "bin", "uv")
	info, err := os.Stat(uvBin)
	if err != nil {
		t.Fatalf("uv binary missing: %v", err)
	}
	if info.Mode().Perm() != 0o755 {
		t.Errorf("uv mode = %o, want 755", info.Mode().Perm())
	}

	cj, err := os.ReadFile(filepath.Join(rt, "setup-journal.json"))
	if err != nil {
		t.Fatalf("journal missing: %v", err)
	}
	once := struct {
		Done map[string]string `json:"done"`
	}{}
	if err := json.Unmarshal(cj, &once); err != nil {
		t.Fatalf("journal json: %v", err)
	}
	wantSteps := []string{"uv", "python", "tabby", "venv-deps", "smoke"}
	if len(once.Done) != len(wantSteps) {
		t.Errorf("journal done steps = %d, want %d", len(once.Done), len(wantSteps))
	}
	for _, id := range wantSteps {
		if _, ok := once.Done[id]; !ok {
			t.Errorf("journal missing %s", id)
		}
	}

	logData, err := os.ReadFile(logPathFor(rt))
	if err != nil {
		t.Fatalf("log missing: %v", err)
	}
	log := string(logData)
	for _, id := range []string{"uv", "venv-deps", "smoke"} {
		if !strings.Contains(log, "step "+id) {
			t.Errorf("log missing step %s", id)
		}
	}
	if !strings.Contains(buf.String(), "running uv:") {
		t.Errorf("stdout missing 'running uv': %q", buf.String())
	}
}

// TestRunJournalResume asserts a second Run skips every step already marked
// done — no exec, no fetch — and re-announces them.
func TestRunJournalResume(t *testing.T) {
	rt := filepath.Join(t.TempDir(), "runtime")
	var buf bytes.Buffer
	rec := &recorder{}
	if err := Run(buildOpts(rt, rec, &buf, true)); err != nil {
		t.Fatalf("first Run: %v", err)
	}
	if len(rec.calls) == 0 || len(rec.fetches) == 0 {
		t.Fatalf("first run did nothing: exec=%d fetch=%d", len(rec.calls), len(rec.fetches))
	}

	var buf2 bytes.Buffer
	rec2 := &recorder{}
	if err := Run(buildOpts(rt, rec2, &buf2, true)); err != nil {
		t.Fatalf("second Run: %v", err)
	}
	if len(rec2.calls) != 0 {
		t.Errorf("second run Exec calls = %d, want 0 (%v)", len(rec2.calls), rec2.calls)
	}
	if len(rec2.fetches) != 0 {
		t.Errorf("second run Fetch calls = %d, want 0", len(rec2.fetches))
	}
	if !strings.Contains(buf2.String(), "skipping uv (done ") {
		t.Errorf("stdout missing skip announcement: %q", buf2.String())
	}
	if !strings.Contains(buf2.String(), "skipping smoke (done ") {
		t.Errorf("stdout missing skip smoke: %q", buf2.String())
	}
}

// TestRunFailureMappingENOSPC asserts the error from a failed step carries
// the last 20 log lines and a human cause with needed vs free bytes.
func TestRunFailureMappingENOSPC(t *testing.T) {
	rt := filepath.Join(t.TempDir(), "runtime")
	var buf bytes.Buffer
	rec := &recorder{}
	opts := buildOpts(rt, rec, &buf, true)
	opts.Exec = func(stepID, name string, args ...string) ([]byte, error) {
		rec.calls = append(rec.calls, call{stepID, name, args})
		if stepID == "venv-deps" {
			return []byte("uv: write error: no space left on device\n"), syscall.ENOSPC
		}
		return []byte("ok\n"), nil
	}

	err := Run(opts)
	if err == nil {
		t.Fatal("expected failure")
	}
	msg := err.Error()

	if !strings.Contains(msg, "setup: venv-deps failed:") {
		t.Errorf("err missing step id: %q", msg)
	}
	if !strings.Contains(msg, "no space left on device") {
		t.Errorf("err missing cause: %q", msg)
	}
	if !strings.Contains(msg, "last 20 lines of "+logPathFor(rt)) {
		t.Errorf("err missing last20 header: %q", msg)
	}
	if !strings.Contains(msg, "uv: write error: no space left on device") {
		t.Errorf("err tail missing failing output: %q", msg)
	}
	if !strings.Contains(msg, "disk full") {
		t.Errorf("err missing human cause: %q", msg)
	}

	plan, perr := Preflight(Options{RuntimeDir: rt, HeadSize: stubHead})
	if perr != nil {
		t.Fatal(perr)
	}
	var need int64
	for _, d := range plan.Downloads {
		need += d.Bytes
	}
	if !strings.Contains(msg, strconv.FormatInt(need, 10)) {
		t.Errorf("err missing needed bytes %d: %q", need, msg)
	}
	if !strings.Contains(msg, "bytes free on "+rt) {
		t.Errorf("err missing free bytes on %s: %q", rt, msg)
	}
}

// TestSmokeRunsAfterVenvDeps asserts ordering: smoke runs last, after
// venv-deps.
func TestSmokeRunsAfterVenvDeps(t *testing.T) {
	rt := filepath.Join(t.TempDir(), "runtime")
	var buf bytes.Buffer
	rec := &recorder{}
	if err := Run(buildOpts(rt, rec, &buf, true)); err != nil {
		t.Fatalf("Run: %v", err)
	}
	venvIdx, smokeIdx := -1, -1
	for i, c := range rec.calls {
		if c.stepID == "venv-deps" {
			venvIdx = i
		}
		if c.stepID == "smoke" {
			smokeIdx = i
		}
	}
	if venvIdx < 0 || smokeIdx < 0 {
		t.Fatalf("missing venv-deps or smoke; calls=%v", rec.calls)
	}
	if smokeIdx <= venvIdx {
		t.Errorf("smoke (%d) must run after venv-deps (%d)", smokeIdx, venvIdx)
	}
	if smokeIdx != len(rec.calls)-1 {
		t.Errorf("smoke is not the final step: %d/%d", smokeIdx, len(rec.calls)-1)
	}
}

// TestHumanCauseMapping covers the three human-cause branches directly.
func TestHumanCauseMapping(t *testing.T) {
	plan := Plan{
		Downloads: []Download{{URL: "https://dl.example/x", Bytes: 5}, {Bytes: 7}},
		FreeBytes: 9,
		Dest:      "/data",
	}

	cases := []struct {
		name string
		err  error
		src  string
		want string
	}{
		{
			name: "enospc",
			err:  syscall.ENOSPC,
			src:  "",
			want: "disk full: need 12 bytes, 9 bytes free on /data",
		},
		{
			name: "tls",
			err:  fmt.Errorf(`Get "https://dl.example/x": x509: certificate signed by unknown authority`),
			src:  "https://dl.example/x",
			want: "TLS failure reaching dl.example — check proxy/CA certs",
		},
		{
			name: "missing exec",
			err:  fmt.Errorf("exec: %q: %w", "git", exec.ErrNotFound),
			src:  "",
			want: "missing git — install it or check PATH",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := humanCause(tc.err, "git", tc.src, plan)
			if !strings.Contains(got, tc.want) {
				t.Errorf("got %q, want substring %q", got, tc.want)
			}
		})
	}
}
