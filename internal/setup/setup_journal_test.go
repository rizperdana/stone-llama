package setup

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A corrupt journal (truncated write, SIGKILL, hand-edit) must not
// block setup forever: the journal is only resume state — report it,
// start the step list fresh, and let the steps re-run (mirrors the
// missing-file path).
func TestRunCorruptJournalResumesFromScratch(t *testing.T) {
	rt := filepath.Join(t.TempDir(), "runtime")
	var buf bytes.Buffer
	rec := &recorder{}
	if err := os.MkdirAll(rt, 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(rt, "setup-journal.json"), []byte("{oops"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := Run(buildOpts(rt, rec, &buf, true)); err != nil {
		t.Fatalf("corrupt journal must not abort setup: %v", err)
	}
	if len(rec.calls) == 0 {
		t.Error("steps did not re-run from scratch")
	}
	if !strings.Contains(buf.String(), "journal") {
		t.Errorf("stdout does not mention the corrupt journal: %q", buf.String())
	}
}
