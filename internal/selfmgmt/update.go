package selfmgmt

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// UpdateOpts configures Update. Zero values select the production
// defaults; tests inject httptest URLs, temp paths, and fake binaries.
type UpdateOpts struct {
	// Version is the current version (main.version from ldflags; "" → "dev").
	Version string
	// Want pins an explicit release tag (--version <tag>); "" → resolve latest.
	Want string
	// Check reports current vs available and changes nothing (--check).
	Check bool
	// Force reinstalls when already on the target version (--force).
	Force bool
	// Yes skips the download confirmation (--yes).
	Yes bool
	// Confirm asks for consent after the download size is announced.
	// nil + !Yes → refusal (non-interactive callers must pass --yes).
	Confirm func() bool
	// Stdout receives all announcements; nil → io.Discard.
	Stdout io.Writer
	// Client is the HTTP client; nil → a 60s-timeout client.
	Client *http.Client
	// API is the GitHub releases API base; nil → defaultRepo's API.
	API string
	// Executable is the installed binary path; nil → os.Executable().
	Executable string
	// Goos/Goarch override the host platform (tests; "" → runtime).
	Goos, Goarch string
}

// UpdateResult reports what an Update did (or would do, with Check).
type UpdateResult struct {
	Current    string // version of the running binary
	Available  string // resolved release tag
	Prerelease bool   // Available is a pre-release
	Asset      string // platform asset name
	Size       int64  // announced download size (bytes)
	Check      bool   // --check mode: nothing was downloaded
	Changed    bool   // a new binary was installed
	Path       string // real install path (or kept temp path in manual mode)
	Manual     string // exact manual command when the dir is not writable ("" otherwise)
}

// ghRelease is the subset of the GitHub releases API we consume.
type ghRelease struct {
	Tag    string    `json:"tag_name"`
	Draft  bool      `json:"draft"`
	Pre    bool      `json:"prerelease"`
	Assets []ghAsset `json:"assets"`
}

type ghAsset struct {
	Name string `json:"name"`
	Size int64  `json:"size"`
	URL  string `json:"browser_download_url"`
}

// Update resolves the target release, announces the download size,
// confirms, downloads, verifies SHA-256 against the release's
// combined checksums.txt, and atomically renames the new binary over
// the old one. Nothing is replaced before verification passes.
func Update(ctx context.Context, opts UpdateOpts) (UpdateResult, error) {
	w := opts.Stdout
	if w == nil {
		w = io.Discard
	}
	client := opts.Client
	if client == nil {
		client = &http.Client{Timeout: 60 * time.Second}
	}
	api := opts.API
	if api == "" {
		api = defaultAPIURL
	}
	execPath := opts.Executable
	if execPath == "" {
		var err error
		execPath, err = os.Executable()
		if err != nil {
			return UpdateResult{}, fmt.Errorf("cannot locate the running binary: %w", err)
		}
	}
	execPath, err := filepath.Abs(execPath)
	if err != nil {
		return UpdateResult{}, err
	}
	goos, goarch := hostPlatform(opts.Goos, opts.Goarch)
	cur := opts.Version
	if cur == "" {
		cur = "dev"
	}
	res := UpdateResult{Current: cur, Check: opts.Check}

	// ---- resolve the release -------------------------------------------------
	releases, err := fetchReleases(ctx, client, api)
	if err != nil {
		return res, err
	}
	rel, preWarn, newerPre, err := pickRelease(releases, opts.Want)
	if err != nil {
		return res, err
	}
	res.Available = rel.Tag
	if sv, ok := parseSemver(rel.Tag); ok {
		res.Prerelease = rel.Pre || sv.pre != ""
	} else {
		res.Prerelease = rel.Pre
	}

	// ---- locate this platform's asset ---------------------------------------
	art := artifactName(goos, goarch)
	var bundle, sums ghAsset
	for _, a := range rel.Assets {
		switch a.Name {
		case art:
			bundle = a
		case "checksums.txt":
			sums = a
		}
	}
	if bundle.URL == "" {
		names := make([]string, 0, len(rel.Assets))
		for _, a := range rel.Assets {
			names = append(names, a.Name)
		}
		return res, fmt.Errorf("release %s has no asset for %s/%s (expected %s); assets: %s",
			rel.Tag, goos, goarch, art, strings.Join(names, ", "))
	}
	if sums.URL == "" {
		return res, fmt.Errorf("release %s has no checksums.txt asset — refusing to install an unverifiable bundle", rel.Tag)
	}
	res.Asset, res.Size = bundle.Name, bundle.Size

	// ---- --check: report and stop (the CI-safe path) ------------------------
	upToDate := normalizeTag(cur) == normalizeTag(rel.Tag)
	if opts.Check {
		fmt.Fprint(w, "stone-llama update --check\n")
		fmt.Fprintf(w, "  current:  %s%s\n", cur, curLabel(cur))
		fmt.Fprintf(w, "  latest:   %s%s\n", rel.Tag, preLabel(res.Prerelease, preWarn))
		fmt.Fprintf(w, "  asset:    %s (%d bytes)\n", res.Asset, res.Size)
		if upToDate {
			fmt.Fprintf(w, "  result:   up to date (%s)\n", rel.Tag)
		} else {
			fmt.Fprint(w, "  result:   update available — run 'stone-llama update'\n")
		}
		if newerPre != "" {
			fmt.Fprintf(w, "  note:     newer pre-release %s — pin it with --version %s\n", newerPre, newerPre)
		}
		return res, nil
	}
	if upToDate && !opts.Force {
		fmt.Fprintf(w, "stone-llama %s is already the latest release (%s) — nothing to do (use --force to reinstall)\n", cur, rel.Tag)
		return res, nil
	}

	// ---- announce, then confirm ---------------------------------------------
	target := execPath
	if real, err := filepath.EvalSymlinks(execPath); err == nil {
		target = real
	} else if !os.IsNotExist(err) {
		return res, fmt.Errorf("cannot resolve install path %s: %w", execPath, err)
	}
	res.Path = target
	if cur == "dev" {
		fmt.Fprintf(w, "warning: this binary reports version \"dev\" (source build) — updating replaces the source-built binary with release %s\n", rel.Tag)
	}
	fmt.Fprintf(w, "stone-llama update: %s → %s\n", cur, rel.Tag)
	fmt.Fprintf(w, "  asset:    %s — %d bytes will be downloaded\n", res.Asset, res.Size)
	fmt.Fprintf(w, "  target:   %s\n", target)
	if target != execPath {
		fmt.Fprintf(w, "  note:     %s is a symlink → %s (the real file is replaced; the link stays)\n", execPath, target)
	}
	if opts.Want == "" {
		if preWarn {
			fmt.Fprintf(w, "note: no stable release exists yet — installing pre-release %s (pin any release with --version <tag>)\n", rel.Tag)
		} else if newerPre != "" {
			fmt.Fprintf(w, "note: newer pre-release %s exists — pin it with --version %s\n", newerPre, newerPre)
		}
	}
	switch {
	case opts.Yes:
		// consent already given via --yes
	case opts.Confirm == nil:
		return res, fmt.Errorf("non-interactive — pass --yes to confirm the %d-byte download", res.Size)
	case !opts.Confirm():
		return res, fmt.Errorf("aborted (download not confirmed)")
	}

	// ---- download ------------------------------------------------------------
	fmt.Fprintf(w, "Downloading %s (%d bytes)...\n", res.Asset, res.Size)
	sumsBody, err := get(ctx, client, sums.URL, 1<<20)
	if err != nil {
		return res, fmt.Errorf("fetch checksums.txt: %w", err)
	}
	want, ok := lookupChecksum(string(sumsBody), res.Asset)
	if !ok {
		return res, fmt.Errorf("checksums.txt has no entry for %s — refusing to install an unverifiable bundle", res.Asset)
	}
	bundleTmp, err := os.CreateTemp("", "stone-llama-bundle-*")
	if err != nil {
		return res, err
	}
	bundlePath := bundleTmp.Name()
	defer func() {
		bundleTmp.Close()
		os.Remove(bundlePath) // no-op if already consumed
	}()
	h := sha256.New()
	if _, err := copyURL(ctx, client, bundle.URL, io.MultiWriter(bundleTmp, h)); err != nil {
		return res, fmt.Errorf("download %s: %w", res.Asset, err)
	}
	if err := bundleTmp.Close(); err != nil {
		return res, err
	}
	got := hex.EncodeToString(h.Sum(nil))
	if got != want {
		return res, fmt.Errorf("checksum mismatch for %s: got %s, want %s — download discarded, nothing was replaced", res.Asset, got, want)
	}
	fmt.Fprintf(w, "Verified SHA-256 against checksums.txt (%s…).\n", got[:12])

	// ---- stage next to the target, then rename (atomic on the same fs) ------
	swap, err := os.CreateTemp(filepath.Dir(target), ".stone-llama-new-*")
	manual := false
	if err != nil {
		// Directory not writable (e.g. root-owned /usr/local/bin): never sudo
		// ourselves — keep the verified binary and print the exact command.
		swap, err = os.CreateTemp("", ".stone-llama-new-*")
		if err != nil {
			return res, fmt.Errorf("cannot stage the new binary: %w", err)
		}
		manual = true
	}
	swapPath := swap.Name()
	delivered := false // true once swapPath must survive (manual mode's deliverable)
	renamed := false
	defer func() {
		if !delivered && !renamed {
			os.Remove(swapPath)
		}
	}()
	if err := extractBinary(bundlePath, res.Asset, binaryName(goos), swap); err != nil {
		swap.Close()
		return res, fmt.Errorf("extract %s: %w", res.Asset, err)
	}
	if err := swap.Chmod(0o755); err != nil {
		swap.Close()
		return res, err
	}
	if err := swap.Sync(); err != nil {
		swap.Close()
		return res, err
	}
	if err := swap.Close(); err != nil {
		return res, err
	}
	if manual {
		cmd := fmt.Sprintf("sudo install -m 0755 %s %s", swapPath, target)
		res.Manual, res.Path, delivered = cmd, swapPath, true
		return res, fmt.Errorf("%s is not writable by you — the verified new binary is kept at %s; run:\n  %s",
			filepath.Dir(target), swapPath, cmd)
	}
	if err := os.Rename(swapPath, target); err != nil {
		return res, fmt.Errorf("replace %s: %w", target, err)
	}
	renamed = true

	res.Changed = true
	fmt.Fprintf(w, "stone-llama updated: %s → %s\n", cur, rel.Tag)
	fmt.Fprintf(w, "  installed: %s (mode 0755)\n", target)
	fmt.Fprint(w, "  a running daemon keeps serving the OLD code until restarted — run:\n    stone-llama stop && stone-llama serve\n")
	return res, nil
}

func curLabel(cur string) string {
	if cur == "dev" {
		return " (source build)"
	}
	return ""
}

func preLabel(pre, warn bool) string {
	if pre {
		if warn {
			return " (pre-release — no stable release yet)"
		}
		return " (pre-release)"
	}
	return ""
}

// fetchReleases lists releases (drafts excluded). The list endpoint is
// used instead of /releases/latest because GitHub's "latest" EXCLUDES
// pre-releases — and this project's only release (v0.1.0-rc1) is one.
func fetchReleases(ctx context.Context, c *http.Client, api string) ([]ghRelease, error) {
	u := api + "?per_page=100"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	// GitHub's API rejects requests without a User-Agent.
	req.Header.Set("User-Agent", "stone-llama/"+defaultRepo)
	req.Header.Set("Accept", "application/vnd.github+json")
	resp, err := c.Do(req)
	if err != nil {
		return nil, fmt.Errorf("resolve latest release: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("resolve latest release: GET %s: HTTP %d", u, resp.StatusCode)
	}
	var rels []ghRelease
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&rels); err != nil {
		return nil, fmt.Errorf("parse releases: %w", err)
	}
	out := rels[:0]
	for _, r := range rels {
		if !r.Draft {
			out = append(out, r)
		}
	}
	return out, nil
}

// pickRelease chooses the target release.
//
// Rule: with an explicit --version pin, that exact tag wins (including
// pre-releases). Otherwise the NEWEST STABLE release by semver wins;
// only when no stable release exists does it fall back to the newest
// pre-release (labelled by the caller). Justification: /releases/latest
// 404s for this repo today (its only release is an rc), so resolution
// must list releases; preferring stable keeps `update` conservative
// once v0.1.0 exists (a newer rc will not hijack stable users), while
// the pre-release fallback keeps it functional right now. A newer-but-
// unselected pre-release is reported as a pin hint.
func pickRelease(rels []ghRelease, want string) (rel ghRelease, preWarn bool, newerPre string, err error) {
	if want != "" {
		want = normalizeTag(want)
		for _, r := range rels {
			if r.Tag == want {
				sv, ok := parseSemver(r.Tag)
				return r, r.Pre || (ok && sv.pre != ""), "", nil
			}
		}
		names := make([]string, 0, len(rels))
		for _, r := range rels {
			names = append(names, r.Tag)
		}
		return ghRelease{}, false, "", fmt.Errorf("release %s not found (available: %s)", want, strings.Join(names, ", "))
	}
	var bestStable, bestAny *ghRelease
	var bestStableSV, bestAnySV semver
	for i := range rels {
		sv, ok := parseSemver(rels[i].Tag)
		if !ok {
			continue // tags that are not semver cannot be ordered
		}
		isPre := rels[i].Pre || sv.pre != ""
		if bestAny == nil || sv.compare(bestAnySV) > 0 {
			bestAny, bestAnySV = &rels[i], sv
		}
		if !isPre && (bestStable == nil || sv.compare(bestStableSV) > 0) {
			bestStable, bestStableSV = &rels[i], sv
		}
	}
	switch {
	case bestStable != nil:
		if bestAny != nil && bestAny.Tag != bestStable.Tag && bestAnySV.compare(bestStableSV) > 0 {
			newerPre = bestAny.Tag
		}
		return *bestStable, false, newerPre, nil
	case bestAny != nil:
		return *bestAny, bestAny.Pre || bestAnySV.pre != "", "", nil
	default:
		return ghRelease{}, false, "", fmt.Errorf("no releases with parseable semver tags")
	}
}

// lookupChecksum finds name's line in the combined checksums.txt
// (sha256sum format: "<hex>  <name>", optional "*" binary marker).
func lookupChecksum(content, name string) (string, bool) {
	for _, line := range strings.Split(content, "\n") {
		f := strings.Fields(line)
		if len(f) == 2 && strings.TrimPrefix(f[1], "*") == name {
			sum := strings.ToLower(f[0])
			if len(sum) == 64 {
				if _, e := hex.DecodeString(sum); e == nil {
					return sum, true
				}
			}
		}
	}
	return "", false
}

// get fetches a small URL (checksums.txt) with a size cap.
func get(ctx context.Context, c *http.Client, url string, limit int64) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "stone-llama/"+defaultRepo)
	resp, err := c.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GET %s: HTTP %d", url, resp.StatusCode)
	}
	return io.ReadAll(io.LimitReader(resp.Body, limit))
}

// copyURL streams url into dst, returning the byte count.
func copyURL(ctx context.Context, c *http.Client, url string, dst io.Writer) (int64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return 0, err
	}
	req.Header.Set("User-Agent", "stone-llama/"+defaultRepo)
	resp, err := c.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("GET %s: HTTP %d", url, resp.StatusCode)
	}
	return io.Copy(dst, resp.Body)
}

// extractBinary streams the binary member out of a release bundle
// (.tgz for linux/darwin, .zip for windows) into dst. Bundles contain a
// top-level stone-llama-<os>-<arch>/ directory (RELEASE.md); the member
// is matched by base name, like install.sh's `find -name stone-llama`.
func extractBinary(bundlePath, asset, member string, dst io.Writer) error {
	if strings.HasSuffix(asset, ".zip") {
		return extractZip(bundlePath, member, dst)
	}
	return extractTgz(bundlePath, member, dst)
}

func extractTgz(bundlePath, member string, dst io.Writer) error {
	f, err := os.Open(bundlePath)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return fmt.Errorf("member %q not found in bundle", member)
		}
		if err != nil {
			return err
		}
		if hdr.Typeflag == tar.TypeReg && filepath.Base(hdr.Name) == member {
			_, err := io.Copy(dst, tr)
			return err
		}
	}
}

func extractZip(bundlePath, member string, dst io.Writer) error {
	zr, err := zip.OpenReader(bundlePath)
	if err != nil {
		return err
	}
	defer zr.Close()
	for _, zf := range zr.File {
		if !zf.FileInfo().IsDir() && filepath.Base(zf.Name) == member {
			rc, err := zf.Open()
			if err != nil {
				return err
			}
			defer rc.Close()
			_, err = io.Copy(dst, rc)
			return err
		}
	}
	return fmt.Errorf("member %q not found in bundle", member)
}
