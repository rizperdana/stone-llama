// Package selfmgmt implements stone-llama's self-management commands:
// `update` (self-update from a GitHub release) and `uninstall` (remove
// the binary and the tool's data completely).
//
// Conventions mirror scripts/install.sh exactly: ollama-style asset
// names (stone-llama-<os>-<arch>.tgz/.zip, no version in the filename),
// a single combined checksums.txt verified with SHA-256, and tags that
// start with "v". Stdlib only — no third-party dependencies.
package selfmgmt

import (
	"fmt"
	"runtime"
	"strings"
)

// repo/API defaults — same coordinates as install.sh (REPO=rizperdana/stone-llama).
const (
	defaultRepo   = "rizperdana/stone-llama"
	defaultAPIURL = "https://api.github.com/repos/rizperdana/stone-llama/releases"
)

// binaryName is the member name inside release bundles (install.sh
// locates the same name with `find -name stone-llama -type f`).
func binaryName(goos string) string {
	if goos == "windows" {
		return "stone-llama.exe"
	}
	return "stone-llama"
}

// artifactName is the ollama-style release asset for a platform
// (RELEASE.md "Artifact naming"; install.sh line ARTIFACT=...).
func artifactName(goos, goarch string) string {
	ext := "tgz"
	if goos == "windows" {
		ext = "zip"
	}
	return fmt.Sprintf("stone-llama-%s-%s.%s", goos, goarch, ext)
}

// ---- minimal semver (tags like v0.1.0-rc1); stdlib has no semver ----

type semver struct {
	major, minor, patch string // numeric strings, compared as numbers
	pre                 string // "" for a stable release
	raw                 string
}

// parseSemver accepts an optional leading "v" and requires a full
// major.minor.patch core. Build metadata ("+meta") is ignored for
// ordering, as semver requires.
func parseSemver(s string) (semver, bool) {
	raw := s
	s = strings.TrimPrefix(s, "v")
	if i := strings.IndexByte(s, '+'); i >= 0 {
		s = s[:i]
	}
	pre := ""
	if i := strings.IndexByte(s, '-'); i >= 0 {
		pre, s = s[i+1:], s[:i]
	}
	parts := strings.Split(s, ".")
	if len(parts) != 3 {
		return semver{}, false
	}
	for _, p := range parts {
		if p == "" || strings.TrimLeft(p, "0123456789") != "" {
			return semver{}, false
		}
	}
	// A stable core with an empty prerelease section is malformed.
	if pre == "" && strings.HasSuffix(raw, "-") {
		return semver{}, false
	}
	return semver{major: parts[0], minor: parts[1], patch: parts[2], pre: pre, raw: raw}, true
}

// fmtSize renders exact bytes: "1957008720 B".
func fmtSize(n int64) string {
	return fmt.Sprintf("%d B", n)
}

// fmtHuman renders IEC units: "1.8 GiB".
func fmtHuman(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for m := n / unit; m >= unit; m /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(n)/float64(div), "KMGTPE"[exp])
}

// compareNum compares digit strings as arbitrary-size numbers.
func compareNum(a, b string) int {
	a, b = strings.TrimLeft(a, "0"), strings.TrimLeft(b, "0")
	if len(a) != len(b) {
		if len(a) < len(b) {
			return -1
		}
		return 1
	}
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	}
	return 0
}

// compare orders two parsed semvers: core first; a stable release
// always outranks any prerelease of the same core (v0.1.0 > v0.1.0-rc1);
// prerelease identifiers compare per semver §11 (numeric < lexical).
func (a semver) compare(b semver) int {
	for _, p := range [][2]string{{a.major, b.major}, {a.minor, b.minor}, {a.patch, b.patch}} {
		if c := compareNum(p[0], p[1]); c != 0 {
			return c
		}
	}
	switch {
	case a.pre == b.pre:
		return 0
	case a.pre == "":
		return 1
	case b.pre == "":
		return -1
	}
	ai, bi := strings.Split(a.pre, "."), strings.Split(b.pre, ".")
	for i := 0; i < len(ai) && i < len(bi); i++ {
		x, y := ai[i], bi[i]
		xn := x != "" && strings.TrimLeft(x, "0123456789") == ""
		yn := y != "" && strings.TrimLeft(y, "0123456789") == ""
		switch {
		case xn && yn:
			if c := compareNum(x, y); c != 0 {
				return c
			}
		case xn != yn:
			if xn { // numeric identifiers are lower than alphanumeric
				return -1
			}
			return 1
		default:
			if x < y {
				return -1
			}
			if x > y {
				return 1
			}
		}
	}
	switch {
	case len(ai) < len(bi):
		return -1
	case len(ai) > len(bi):
		return 1
	}
	return 0
}

// normalizeTag mirrors install.sh: ensure the tag starts with "v".
func normalizeTag(t string) string {
	t = strings.TrimSpace(t)
	if t != "" && !strings.HasPrefix(t, "v") {
		return "v" + t
	}
	return t
}

// hostPlatform reports the platform of the running binary — the
// installed artifact's platform (overridable for tests).
func hostPlatform(goos, goarch string) (string, string) {
	if goos == "" {
		goos = runtime.GOOS
	}
	if goarch == "" {
		goarch = runtime.GOARCH
	}
	return goos, goarch
}
