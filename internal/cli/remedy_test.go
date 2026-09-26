package cli

import (
	"strings"
	"testing"
)

// The auto-start failure message embeds the child's own output, which
// already carries one remedy block (live demo caught the double
// print). backendStartHelp must add nothing in that case; a fresh
// runtime-incomplete error with no block still gets one. This file
// references backendStartHelp (fixed tree only) — excluded when copying
// tests onto the base checkout, like serve's diagnose_test.go.
func TestRemedyBlockNeverPrintsTwice(t *testing.T) {
	msg := "auto-start failed:\n" +
		"stone-llama serve: runtime incomplete — run 'stone-llama setup' first (x missing)\n" +
		"remedies, least friction first:\n  1. attach to a running OpenAI-compatible server\n"
	if got := backendStartHelp(msg); got != "" {
		t.Errorf("second remedy block appended: %q", got)
	}
	fresh := "stone-llama serve: runtime incomplete — run 'stone-llama setup' first (x missing)"
	got := backendStartHelp(fresh)
	if !strings.Contains(got, "remedies, least friction first") {
		t.Errorf("fresh runtime-incomplete error got no remedies: %q", got)
	}
}
