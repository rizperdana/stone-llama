package cli

import (
	"os"
	"strings"
	"testing"
)

// --draft-mode accepts exactly ngram|off; "off" maps to "disabled" —
// explicit drafting-off, a value config.Autofit and TabbyAPI's
// draft_mode Literal both accept ("" would be indistinguishable from
// "never set" once the payload key reaches the backend).
func TestDraftModeValueMapsNgramAndOff(t *testing.T) {
	if m, ok := draftModeValue("ngram"); !ok || m != "ngram" {
		t.Errorf("ngram → %q/%v, want ngram/true", m, ok)
	}
	if m, ok := draftModeValue("off"); !ok || m != "disabled" {
		t.Errorf("off → %q/%v, want disabled/true", m, ok)
	}
	for _, bad := range []string{"", "bogus", "Ngram", "model", "mtp"} {
		if _, ok := draftModeValue(bad); ok {
			t.Errorf("%q must be rejected by the flag parser", bad)
		}
	}
}

// A bad value is a usage error (exit 2) on both commands, before any
// daemon or backend work — same gate other bad flag values use.
func TestDraftModeFlagRejectsBogus(t *testing.T) {
	isolateConfig(t)
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	code, _, errb := run("run", "--draft-mode", "bogus")
	if code != 2 || !strings.Contains(errb, "bad --draft-mode") {
		t.Errorf("run --draft-mode bogus: code/err = %d/%q, want 2/bad --draft-mode", code, errb)
	}
	code, _, errb = run("serve", "--draft-mode", "bogus")
	if code != 2 || !strings.Contains(errb, "invalid --draft-mode") {
		t.Errorf("serve --draft-mode bogus: code/err = %d/%q, want 2/invalid --draft-mode", code, errb)
	}
}

// --draft-mode ngram parses past the flag loop on both commands: run
// then stops at the suppressed auto-start (exit 1, never 2), serve at
// the missing runtime (exit 1, never 2). Neither reaches a backend.
func TestDraftModeFlagParsesNgram(t *testing.T) {
	if os.Getenv("SL_H4_CHILD") == "1" {
		t.Skip("recursive starter guard (base re-execs the test binary)")
	}
	t.Setenv("SL_H4_CHILD", "1")
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("STONE_LLAMA_NO_AUTOSTART", "1")
	t.Setenv("STONE_LLAMA_UPSTREAM", "http://127.0.0.1:1")
	t.Setenv("STONE_LLAMA_PORT", freeLocalPort(t))
	if code, _, errb := run("run", "somemodel", "--draft-mode", "ngram", "-p", "hi"); code != 1 {
		t.Errorf("run --draft-mode ngram: code/err = %d/%q, want 1 (flag parsed)", code, errb)
	}
	isolateConfig(t)
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	if code, _, errb := run("serve", "--draft-mode", "ngram", "--port", freeLocalPort(t)); code != 1 {
		t.Errorf("serve --draft-mode ngram: code/err = %d/%q, want 1 (flag parsed)", code, errb)
	}
}
