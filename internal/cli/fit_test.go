package cli

import (
	"strings"
	"testing"
)

// A stray flag must fail as usage (exit 2) before any GPU probe, HF
// metadata fetch, or "search this repo id" interpretation — `fit
// --bogus` used to treat the flag as a repo and go looking for it.
func TestFitRejectsUnknownFlag(t *testing.T) {
	for _, arg := range []string{"--bogus", "-x", "--ctx"} {
		var errOut strings.Builder
		code := runFit([]string{arg}, strings.NewReader(""), &strings.Builder{}, &errOut)
		if code != 2 {
			t.Errorf("fit %s: exit = %d, want 2 (usage)", arg, code)
		}
		if !strings.Contains(errOut.String(), "usage: stone-llama fit") {
			t.Errorf("fit %s: stderr = %q, want usage line", arg, errOut.String())
		}
	}
}
