package selfmgmt

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
)

// ---- hermetic fake GitHub ----------------------------------------------------

// relDef describes one simulated release.
type relDef struct {
	tag        string
	pre        bool
	draft      bool
	omitBundle bool // no asset for this platform
	omitSums   bool // no checksums.txt asset
	sumsEmpty  bool // checksums.txt without our line
	badSum     bool // checksums.txt with a wrong digest
	goos       string
	goarch     string
	member     []byte // binary content shipped in the bundle
}

type fakeGH struct {
	srv        *httptest.Server
	bundleHits atomic.Int32
	apiHits    atomic.Int32
}

// newFakeGH serves /api (release list) and /dl/<tag>/<name> (assets)
// from httptest — no real network, ever.
func newFakeGH(t *testing.T, defs ...relDef) *fakeGH {
	t.Helper()
	g := &fakeGH{}
	type files struct {
		bundle, sums []byte
		art          string
	}
	per := make(map[string]files, len(defs))
	mux := http.NewServeMux()
	g.srv = httptest.NewServer(mux)
	for _, d := range defs {
		goos, goarch := d.goos, d.goarch
		if goos == "" {
			goos, goarch = "linux", "amd64"
		}
		art := artifactName(goos, goarch)
		memberName := binaryName(goos)
		memberDir := strings.TrimSuffix(art, filepath.Ext(art))
		content := d.member
		if content == nil {
			content = []byte("NEW-BINARY-" + d.tag)
		}
		var bundle []byte
		if !d.omitBundle {
			if strings.HasSuffix(art, ".zip") {
				bundle = buildZip(t, memberDir+"/"+memberName, content)
			} else {
				bundle = buildTgz(t, memberDir+"/"+memberName, content)
			}
		}
		sum := sha256.Sum256(bundle)
		sums := ""
		switch {
		case d.sumsEmpty:
			sums = "0000000000000000000000000000000000000000000000000000000000000000  unrelated.tgz\n"
		case d.badSum:
			sums = strings.Repeat("f", 64) + "  " + art + "\n"
		default:
			sums = hex.EncodeToString(sum[:]) + "  " + art + "\n"
		}
		per[d.tag] = files{bundle: bundle, sums: []byte(sums), art: art}
	}
	mux.HandleFunc("/api", func(w http.ResponseWriter, r *http.Request) {
		g.apiHits.Add(1)
		out := make([]map[string]any, 0, len(defs))
		for _, d := range defs {
			assets := []map[string]any{}
			if !d.omitBundle {
				f := per[d.tag]
				assets = append(assets, map[string]any{
					"name":                 f.art,
					"size":                 len(f.bundle),
					"browser_download_url": g.srv.URL + "/dl/" + d.tag + "/" + f.art,
				})
			}
			if !d.omitSums {
				assets = append(assets, map[string]any{
					"name":                 "checksums.txt",
					"size":                 len(per[d.tag].sums),
					"browser_download_url": g.srv.URL + "/dl/" + d.tag + "/checksums.txt",
				})
			}
			out = append(out, map[string]any{
				"tag_name": d.tag, "prerelease": d.pre, "draft": d.draft,
				"assets": assets,
			})
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(out); err != nil {
			t.Errorf("encode releases: %v", err)
		}
	})
	mux.HandleFunc("/dl/", func(w http.ResponseWriter, r *http.Request) {
		parts := strings.SplitN(strings.TrimPrefix(r.URL.Path, "/dl/"), "/", 2)
		if len(parts) != 2 {
			http.NotFound(w, r)
			return
		}
		f, ok := per[parts[0]]
		if !ok {
			http.NotFound(w, r)
			return
		}
		if parts[1] == "checksums.txt" {
			_, _ = w.Write(f.sums)
			return
		}
		g.bundleHits.Add(1)
		_, _ = w.Write(f.bundle)
	})
	t.Cleanup(g.srv.Close)
	return g
}

func buildTgz(t *testing.T, memberPath string, content []byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	hdr := &tar.Header{
		Name: memberPath, Mode: 0o755, Size: int64(len(content)),
		Typeflag: tar.TypeReg,
	}
	if err := tw.WriteHeader(hdr); err != nil {
		t.Fatal(err)
	}
	if _, err := tw.Write(content); err != nil {
		t.Fatal(err)
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func buildZip(t *testing.T, memberPath string, content []byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	w, err := zw.CreateHeader(&zip.FileHeader{Name: memberPath, Method: zip.Deflate})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Write(content); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// fakeBinary drops a binary-like file under t.TempDir() and returns its
// path — tests never touch the real installed binary.
func fakeBinary(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "stone-llama")
	if err := os.WriteFile(p, []byte(content), 0o755); err != nil {
		t.Fatal(err)
	}
	mustUnderTemp(t, p)
	return p
}

// mustUnderTemp asserts hermeticity: every path a test mutates lives
// under t.TempDir() or the OS temp dir (for Update's staged files).
func mustUnderTemp(t *testing.T, paths ...string) {
	t.Helper()
	root := t.TempDir()
	for _, p := range paths {
		if !strings.HasPrefix(p, root) && !strings.HasPrefix(p, os.TempDir()) {
			t.Fatalf("test path %q escapes the temp roots (%q, %q) — hermeticity violated", p, root, os.TempDir())
		}
	}
}

// runUpdate wires a hermetic opts set around Update.
func runUpdate(t *testing.T, g *fakeGH, opts UpdateOpts) (UpdateResult, string, error) {
	t.Helper()
	if opts.Executable == "" {
		opts.Executable = fakeBinary(t, "OLD-BINARY")
	}
	mustUnderTemp(t, opts.Executable)
	var out bytes.Buffer
	opts.API = g.srv.URL + "/api"
	opts.Stdout = &out
	res, err := Update(context.Background(), opts)
	return res, out.String(), err
}

// ---- update tests -------------------------------------------------------------

func TestUpdateCheckPreReleaseOnly(t *testing.T) {
	g := newFakeGH(t, relDef{tag: "v0.1.0-rc1", pre: true, member: []byte("RC1")})
	res, out, err := runUpdate(t, g, UpdateOpts{Version: "v0.0.1", Check: true})
	if err != nil {
		t.Fatalf("Update --check: %v", err)
	}
	if res.Available != "v0.1.0-rc1" || !res.Prerelease || !res.Check {
		t.Errorf("result = %+v", res)
	}
	for _, want := range []string{
		"stone-llama update --check",
		"current:  v0.0.1",
		"latest:   v0.1.0-rc1 (pre-release — no stable release yet)",
		"update available",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("--check output missing %q:\n%s", want, out)
		}
	}
	if g.bundleHits.Load() != 0 {
		t.Error("--check must not download the bundle")
	}
}

func TestUpdatePrefersStableAndHintsNewerPre(t *testing.T) {
	g := newFakeGH(t,
		relDef{tag: "v0.2.0-rc1", pre: true},
		relDef{tag: "v0.1.0"},
	)
	res, out, err := runUpdate(t, g, UpdateOpts{Version: "v0.0.1", Check: true})
	if err != nil {
		t.Fatalf("Update --check: %v", err)
	}
	if res.Available != "v0.1.0" || res.Prerelease {
		t.Errorf("stable must win: %+v", res)
	}
	if !strings.Contains(out, "newer pre-release v0.2.0-rc1 — pin it with --version v0.2.0-rc1") {
		t.Errorf("missing pin hint:\n%s", out)
	}
}

func TestUpdateDraftExcluded(t *testing.T) {
	g := newFakeGH(t,
		relDef{tag: "v9.0.0", draft: true},
		relDef{tag: "v0.1.0"},
	)
	res, _, err := runUpdate(t, g, UpdateOpts{Version: "v0.0.1", Check: true})
	if err != nil {
		t.Fatalf("Update --check: %v", err)
	}
	if res.Available != "v0.1.0" {
		t.Errorf("draft must be excluded, got %q", res.Available)
	}
}

func TestUpdateVersionPinIncludingPreRelease(t *testing.T) {
	g := newFakeGH(t,
		relDef{tag: "v0.1.0"},
		relDef{tag: "v0.2.0-rc1", pre: true, member: []byte("RC-BINARY")},
	)
	bin := fakeBinary(t, "OLD-BINARY")
	// pin without the "v" prefix, like install.sh accepts
	res, out, err := runUpdate(t, g, UpdateOpts{Version: "v0.1.0", Want: "0.2.0-rc1", Yes: true, Executable: bin})
	if err != nil {
		t.Fatalf("Update --version: %v", err)
	}
	if res.Available != "v0.2.0-rc1" || !res.Changed {
		t.Errorf("result = %+v", res)
	}
	got, err := os.ReadFile(bin)
	if err != nil || string(got) != "RC-BINARY" {
		t.Errorf("binary = %q err=%v, want RC-BINARY", got, err)
	}
	if !strings.Contains(out, "v0.1.0 → v0.2.0-rc1") {
		t.Errorf("missing old → new line:\n%s", out)
	}
	if g.bundleHits.Load() != 1 {
		t.Errorf("bundle downloads = %d, want 1", g.bundleHits.Load())
	}
}

func TestUpdatePinNotFound(t *testing.T) {
	g := newFakeGH(t, relDef{tag: "v0.1.0"})
	_, _, err := runUpdate(t, g, UpdateOpts{Version: "v0.0.1", Want: "v9.9.9", Check: true})
	if err == nil || !strings.Contains(err.Error(), "not found (available: v0.1.0)") {
		t.Fatalf("err = %v", err)
	}
}

func TestUpdateChecksumMismatchRefused(t *testing.T) {
	g := newFakeGH(t, relDef{tag: "v0.1.0-rc1", pre: true, badSum: true, member: []byte("EVIL")})
	bin := fakeBinary(t, "OLD-BINARY")
	_, out, err := runUpdate(t, g, UpdateOpts{Version: "v0.0.1", Yes: true, Executable: bin})
	if err == nil || !strings.Contains(err.Error(), "checksum mismatch") {
		t.Fatalf("err = %v", err)
	}
	if !strings.Contains(err.Error(), "nothing was replaced") {
		t.Errorf("mismatch error must say nothing was replaced: %v", err)
	}
	got, rerr := os.ReadFile(bin)
	if rerr != nil || string(got) != "OLD-BINARY" {
		t.Errorf("target changed after failed verification: %q err=%v", got, rerr)
	}
	if !strings.Contains(out, "Downloading") {
		t.Errorf("download announcement missing:\n%s", out)
	}
	mustNoBundleTempLeft(t)
}

func TestUpdateChecksumsMissingLineRefused(t *testing.T) {
	g := newFakeGH(t, relDef{tag: "v0.1.0-rc1", pre: true, sumsEmpty: true})
	bin := fakeBinary(t, "OLD-BINARY")
	_, _, err := runUpdate(t, g, UpdateOpts{Version: "v0.0.1", Yes: true, Executable: bin})
	if err == nil || !strings.Contains(err.Error(), "no entry for") {
		t.Fatalf("err = %v", err)
	}
	got, _ := os.ReadFile(bin)
	if string(got) != "OLD-BINARY" {
		t.Error("target changed despite unverifiable bundle")
	}
}

func TestUpdateMissingAsset(t *testing.T) {
	g := newFakeGH(t, relDef{tag: "v0.1.0-rc1", pre: true, omitBundle: true})
	_, _, err := runUpdate(t, g, UpdateOpts{Version: "v0.0.1", Check: true})
	if err == nil {
		t.Fatal("missing asset must error")
	}
	want := "no asset for " + runtime.GOOS + "/" + runtime.GOARCH
	if !strings.Contains(err.Error(), want) {
		t.Errorf("err = %v, want substring %q", err, want)
	}
}

func TestUpdateAlreadyCurrent(t *testing.T) {
	g := newFakeGH(t, relDef{tag: "v0.1.0-rc1", pre: true})
	// --check reports no-op
	res, out, err := runUpdate(t, g, UpdateOpts{Version: "v0.1.0-rc1", Check: true})
	if err != nil {
		t.Fatalf("check: %v", err)
	}
	if res.Changed {
		t.Error("--check must not change anything")
	}
	if !strings.Contains(out, "up to date (v0.1.0-rc1)") {
		t.Errorf("check output:\n%s", out)
	}
	// plain invocation is a no-op without --force
	res, out, err = runUpdate(t, g, UpdateOpts{Version: "v0.1.0-rc1", Yes: true})
	if err != nil {
		t.Fatalf("plain: %v", err)
	}
	if res.Changed || g.bundleHits.Load() != 0 {
		t.Errorf("already-current must not download (hits=%d, changed=%v)", g.bundleHits.Load(), res.Changed)
	}
	if !strings.Contains(out, "already the latest release") || !strings.Contains(out, "--force") {
		t.Errorf("plain output:\n%s", out)
	}
}

func TestUpdateForceReinstallsSameVersion(t *testing.T) {
	g := newFakeGH(t, relDef{tag: "v0.1.0-rc1", pre: true, member: []byte("FRESH")})
	bin := fakeBinary(t, "OLD-BINARY")
	res, _, err := runUpdate(t, g, UpdateOpts{Version: "v0.1.0-rc1", Force: true, Yes: true, Executable: bin})
	if err != nil {
		t.Fatalf("force: %v", err)
	}
	if !res.Changed || g.bundleHits.Load() != 1 {
		t.Errorf("force must reinstall: changed=%v hits=%d", res.Changed, g.bundleHits.Load())
	}
	got, _ := os.ReadFile(bin)
	if string(got) != "FRESH" {
		t.Errorf("binary = %q, want FRESH", got)
	}
}

func TestUpdateNonInteractiveRequiresYes(t *testing.T) {
	g := newFakeGH(t, relDef{tag: "v0.1.0-rc1", pre: true})
	bin := fakeBinary(t, "OLD-BINARY")
	res, out, err := runUpdate(t, g, UpdateOpts{Version: "v0.0.1", Executable: bin})
	if err == nil || !strings.Contains(err.Error(), "pass --yes") {
		t.Fatalf("err = %v", err)
	}
	if g.bundleHits.Load() != 0 {
		t.Error("no download may happen without consent")
	}
	// the size is announced before the refusal
	if !strings.Contains(out, fmt.Sprintf("%d bytes will be downloaded", res.Size)) {
		t.Errorf("size not announced:\n%s", out)
	}
	if string(mustRead(t, bin)) != "OLD-BINARY" {
		t.Error("binary changed without consent")
	}
}

func TestUpdateConfirmSeesSizeAndDeclineChangesNothing(t *testing.T) {
	g := newFakeGH(t, relDef{tag: "v0.1.0-rc1", pre: true, member: []byte("NEW")})
	bin := fakeBinary(t, "OLD-BINARY")
	var outAtConfirm string
	var out bytes.Buffer
	opts := UpdateOpts{
		Version: "v0.0.1", Executable: bin,
		API: g.srv.URL + "/api", Stdout: &out,
		Confirm: func() bool {
			outAtConfirm = out.String()
			return false
		},
	}
	_, err := Update(context.Background(), opts)
	if err == nil || !strings.Contains(err.Error(), "aborted") {
		t.Fatalf("err = %v", err)
	}
	if !strings.Contains(outAtConfirm, "bytes will be downloaded") {
		t.Errorf("size must be announced BEFORE confirmation:\n%s", outAtConfirm)
	}
	if g.bundleHits.Load() != 0 {
		t.Error("declined confirmation must not download")
	}
	if string(mustRead(t, bin)) != "OLD-BINARY" {
		t.Error("binary changed after declined confirmation")
	}
}

func TestUpdateSuccessfulAtomicReplace(t *testing.T) {
	member := []byte("#!/bin/sh\necho new stone-llama\n" + strings.Repeat("x", 4096))
	g := newFakeGH(t, relDef{tag: "v0.1.0-rc1", pre: true, member: member})
	bin := fakeBinary(t, "OLD-BINARY")
	res, out, err := runUpdate(t, g, UpdateOpts{Version: "v0.0.1", Yes: true, Executable: bin})
	if err != nil {
		t.Fatalf("Update: %v", err)
	}
	if !res.Changed || res.Available != "v0.1.0-rc1" {
		t.Errorf("result = %+v", res)
	}
	got := mustRead(t, bin)
	if !bytes.Equal(got, member) {
		t.Errorf("installed bytes differ from the bundle member (%d vs %d bytes)", len(got), len(member))
	}
	st, err := os.Stat(bin)
	if err != nil || st.Mode().Perm() != 0o755 {
		t.Errorf("mode = %v err=%v, want 0755", st.Mode(), err)
	}
	for _, want := range []string{
		"stone-llama update: v0.0.1 → v0.1.0-rc1",
		"bytes will be downloaded",
		"Verified SHA-256 against checksums.txt",
		"stone-llama updated: v0.0.1 → v0.1.0-rc1",
		"installed: " + res.Path,
		"stone-llama stop && stone-llama serve",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q:\n%s", want, out)
		}
	}
	mustNoBundleTempLeft(t)
}

func TestUpdateSymlinkedInstallPath(t *testing.T) {
	g := newFakeGH(t, relDef{tag: "v0.1.0-rc1", pre: true, member: []byte("REAL-NEW")})
	root := t.TempDir()
	real := filepath.Join(root, "real", "stone-llama")
	if err := os.MkdirAll(filepath.Dir(real), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(real, []byte("REAL-OLD"), 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(root, "bin", "stone-llama")
	if err := os.MkdirAll(filepath.Dir(link), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(real, link); err != nil {
		t.Fatal(err)
	}
	mustUnderTemp(t, real, link)
	res, out, err := runUpdate(t, g, UpdateOpts{Version: "v0.0.1", Yes: true, Executable: link})
	if err != nil {
		t.Fatalf("Update: %v", err)
	}
	if res.Path != real {
		t.Errorf("Path = %q, want real file %q", res.Path, real)
	}
	if string(mustRead(t, real)) != "REAL-NEW" {
		t.Error("real file not replaced")
	}
	st, err := os.Lstat(link)
	if err != nil || st.Mode()&os.ModeSymlink == 0 {
		t.Fatalf("link is no longer a symlink: %v %v", st, err)
	}
	if !strings.Contains(out, "is a symlink") {
		t.Errorf("symlink note missing:\n%s", out)
	}
}

func TestUpdateTargetDirNotWritableKeepsVerifiedBinary(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("running as root — permission bits do not deny writes")
	}
	g := newFakeGH(t, relDef{tag: "v0.1.0-rc1", pre: true, member: []byte("NEW-BINARY")})
	bin := fakeBinary(t, "OLD-BINARY")
	dir := filepath.Dir(bin)
	if err := os.Chmod(dir, 0o555); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(dir, 0o755) }) // let t.TempDir clean up
	res, _, err := runUpdate(t, g, UpdateOpts{Version: "v0.0.1", Yes: true, Executable: bin})
	if err == nil {
		t.Fatal("unwritable target dir must fail")
	}
	if !strings.Contains(err.Error(), "sudo install -m 0755") {
		t.Errorf("manual command missing: %v", err)
	}
	// target untouched
	if string(mustRead(t, bin)) != "OLD-BINARY" {
		t.Error("target was modified despite unwritable dir")
	}
	// verified binary preserved at the printed path
	fields := strings.Fields(res.Manual)
	if len(fields) != 6 {
		t.Fatalf("Manual = %q", res.Manual)
	}
	kept := fields[4]
	if !strings.Contains(err.Error(), kept) {
		t.Errorf("error must contain the kept path %s: %v", kept, err)
	}
	kst, kerr := os.Stat(kept)
	if kerr != nil {
		t.Fatalf("kept binary missing: %v", kerr)
	}
	if kst.Mode().Perm() != 0o755 {
		t.Errorf("kept binary mode = %v, want 0755", kst.Mode().Perm())
	}
	if string(mustRead(t, kept)) != "NEW-BINARY" {
		t.Error("kept binary content wrong")
	}
	_ = os.Remove(kept)
}

func TestUpdateDevVersionWarns(t *testing.T) {
	g := newFakeGH(t, relDef{tag: "v0.1.0-rc1", pre: true, member: []byte("NEW")})
	bin := fakeBinary(t, "source-built")
	_, out, err := runUpdate(t, g, UpdateOpts{Version: "dev", Yes: true, Executable: bin})
	if err != nil {
		t.Fatalf("Update: %v", err)
	}
	if !strings.Contains(out, "source build") || !strings.Contains(out, "replaces the source-built binary") {
		t.Errorf("dev warning missing:\n%s", out)
	}
	if string(mustRead(t, bin)) != "NEW" {
		t.Error("dev binary not replaced")
	}
}

func TestUpdateZipBundleWindows(t *testing.T) {
	member := []byte("WINDOWS-NEW")
	g := newFakeGH(t, relDef{tag: "v0.1.0-rc1", pre: true, goos: "windows", goarch: "amd64", member: member})
	bin := fakeBinary(t, "OLD")
	res, _, err := runUpdate(t, g, UpdateOpts{
		Version: "v0.0.1", Yes: true, Executable: bin,
		Goos: "windows", Goarch: "amd64",
	})
	if err != nil {
		t.Fatalf("Update (windows zip): %v", err)
	}
	if res.Asset != "stone-llama-windows-amd64.zip" {
		t.Errorf("Asset = %q", res.Asset)
	}
	if !bytes.Equal(mustRead(t, bin), member) {
		t.Error("zip member not installed")
	}
}

func TestUpdateChecksumsMissingAssetRefused(t *testing.T) {
	g := newFakeGH(t, relDef{tag: "v0.1.0-rc1", pre: true, omitSums: true})
	bin := fakeBinary(t, "OLD-BINARY")
	_, _, err := runUpdate(t, g, UpdateOpts{Version: "v0.0.1", Yes: true, Executable: bin})
	if err == nil || !strings.Contains(err.Error(), "no checksums.txt asset") {
		t.Fatalf("err = %v", err)
	}
	if string(mustRead(t, bin)) != "OLD-BINARY" {
		t.Error("binary changed without verification material")
	}
}

// ---- small helpers -------------------------------------------------------------

func mustRead(t *testing.T, p string) []byte {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func mustNoBundleTempLeft(t *testing.T) {
	t.Helper()
	matches, err := filepath.Glob(filepath.Join(os.TempDir(), "stone-llama-bundle-*"))
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 0 {
		t.Errorf("bundle temp files left behind: %v", matches)
	}
}
