package selfmgmt

import "testing"

func TestParseSemver(t *testing.T) {
	cases := []struct {
		in  string
		ok  bool
		pre string
	}{
		{"v0.1.0", true, ""},
		{"0.1.0", true, ""},
		{"v0.1.0-rc1", true, "rc1"},
		{"v1.2.3-beta.2", true, "beta.2"},
		{"v1.2.3+build5", true, ""},
		{"v1.2.3-rc1+build5", true, "rc1"},
		{"v1.2", false, ""},
		{"v1.2.x", false, ""},
		{"", false, ""},
		{"v0.1.0-", false, ""},
	}
	for _, c := range cases {
		sv, ok := parseSemver(c.in)
		if ok != c.ok {
			t.Errorf("parseSemver(%q) ok=%v, want %v", c.in, ok, c.ok)
			continue
		}
		if ok && sv.pre != c.pre {
			t.Errorf("parseSemver(%q) pre=%q, want %q", c.in, sv.pre, c.pre)
		}
	}
}

func TestSemverCompare(t *testing.T) {
	ordered := []string{
		"v0.1.0-rc1", "v0.1.0-rc2", "v0.1.0", "v0.2.0-rc1", "v0.2.0",
		"v0.9.0", "v0.10.0", "v1.0.0",
	}
	for i := 0; i < len(ordered); i++ {
		for j := 0; j < len(ordered); j++ {
			a, aok := parseSemver(ordered[i])
			b, bok := parseSemver(ordered[j])
			if !aok || !bok {
				t.Fatalf("parse failed: %s / %s", ordered[i], ordered[j])
			}
			got := a.compare(b)
			want := 0
			if i < j {
				want = -1
			} else if i > j {
				want = 1
			}
			if got != want {
				t.Errorf("%s compare %s = %d, want %d", ordered[i], ordered[j], got, want)
			}
		}
	}
	// semver §11: numeric identifiers sort below alphanumeric ones
	rc, _ := parseSemver("v1.0.0-rc.2")
	num, _ := parseSemver("v1.0.0-9")
	if rc.compare(num) <= 0 {
		t.Error("v1.0.0-rc.2 should outrank v1.0.0-9 (numeric identifiers are lower)")
	}
	// build metadata is ignored for ordering
	x, _ := parseSemver("v1.0.0+aaa")
	y, _ := parseSemver("v1.0.0+bbb")
	if x.compare(y) != 0 {
		t.Error("build metadata must not affect ordering")
	}
}

func TestNormalizeTag(t *testing.T) {
	if got := normalizeTag("0.1.0-rc1"); got != "v0.1.0-rc1" {
		t.Errorf("normalizeTag(0.1.0-rc1) = %q", got)
	}
	if got := normalizeTag("v0.1.0-rc1"); got != "v0.1.0-rc1" {
		t.Errorf("normalizeTag(v0.1.0-rc1) = %q", got)
	}
}

func TestArtifactName(t *testing.T) {
	if got := artifactName("linux", "amd64"); got != "stone-llama-linux-amd64.tgz" {
		t.Errorf("artifactName linux = %q", got)
	}
	if got := artifactName("darwin", "arm64"); got != "stone-llama-darwin-arm64.tgz" {
		t.Errorf("artifactName darwin = %q", got)
	}
	if got := artifactName("windows", "amd64"); got != "stone-llama-windows-amd64.zip" {
		t.Errorf("artifactName windows = %q", got)
	}
	if got := binaryName("linux"); got != "stone-llama" {
		t.Errorf("binaryName linux = %q", got)
	}
	if got := binaryName("windows"); got != "stone-llama.exe" {
		t.Errorf("binaryName windows = %q", got)
	}
}

func TestLookupChecksum(t *testing.T) {
	sum := "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
	content := sum + "  stone-llama-linux-amd64.tgz\n" +
		"0000000000000000000000000000000000000000000000000000000000000000  other.tgz\n"
	got, ok := lookupChecksum(content, "stone-llama-linux-amd64.tgz")
	if !ok || got != sum {
		t.Fatalf("lookupChecksum = %q, %v", got, ok)
	}
	if _, ok := lookupChecksum(content, "missing.tgz"); ok {
		t.Error("missing asset must not resolve")
	}
	// sha256sum -b binary marker form
	if _, ok := lookupChecksum(sum+" *stone-llama-linux-amd64.tgz\n", "stone-llama-linux-amd64.tgz"); !ok {
		t.Error("star-prefixed (binary mode) line must resolve")
	}
	// malformed hex is rejected
	if _, ok := lookupChecksum("zzzz  stone-llama-linux-amd64.tgz\n", "stone-llama-linux-amd64.tgz"); ok {
		t.Error("non-hex digest must be rejected")
	}
}

func TestPickReleaseRules(t *testing.T) {
	mk := func(tag string, pre bool) ghRelease {
		return ghRelease{Tag: tag, Pre: pre}
	}
	// pre-release-only repo: fall back to the newest pre-release, flagged
	rel, preWarn, newer, err := pickRelease([]ghRelease{mk("v0.1.0-rc1", true)}, "")
	if err != nil || rel.Tag != "v0.1.0-rc1" || !preWarn || newer != "" {
		t.Errorf("pre-only: got %q preWarn=%v newer=%q err=%v", rel.Tag, preWarn, newer, err)
	}
	// stable preferred over a newer pre-release; the newer pre is a pin hint
	rel, preWarn, newer, err = pickRelease([]ghRelease{mk("v0.2.0-rc1", true), mk("v0.1.0", false)}, "")
	if err != nil || rel.Tag != "v0.1.0" || preWarn || newer != "v0.2.0-rc1" {
		t.Errorf("prefer-stable: got %q preWarn=%v newer=%q err=%v", rel.Tag, preWarn, newer, err)
	}
	// newest stable by semver wins regardless of API order
	rel, _, _, err = pickRelease([]ghRelease{mk("v0.1.0", false), mk("v0.10.0", false)}, "")
	if err != nil || rel.Tag != "v0.10.0" {
		t.Errorf("newest stable: got %q err=%v", rel.Tag, err)
	}
	// explicit pin wins, including pre-releases, with v-normalisation
	rel, preWarn, _, err = pickRelease([]ghRelease{mk("v0.1.0", false), mk("v0.2.0-rc1", true)}, "0.2.0-rc1")
	if err != nil || rel.Tag != "v0.2.0-rc1" || !preWarn {
		t.Errorf("pin: got %q preWarn=%v err=%v", rel.Tag, preWarn, err)
	}
	// unknown pin errors and lists what exists
	if _, _, _, err = pickRelease([]ghRelease{mk("v0.1.0", false)}, "v9.9.9"); err == nil {
		t.Error("unknown pin must error")
	}
	// unparseable tags are skipped, not ordered
	rel, _, _, err = pickRelease([]ghRelease{mk("nightly", true), mk("v0.1.0", false)}, "")
	if err != nil || rel.Tag != "v0.1.0" {
		t.Errorf("garbage tag skipped: got %q err=%v", rel.Tag, err)
	}
	if _, _, _, err = pickRelease([]ghRelease{mk("nightly", true)}, ""); err == nil {
		t.Error("no parseable tags must error")
	}
	// no releases at all
	if _, _, _, err = pickRelease(nil, ""); err == nil {
		t.Error("empty release list must error")
	}
}
