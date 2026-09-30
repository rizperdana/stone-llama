package setup

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// setup's log follows the daemon-log discipline: 0700 dir, 0600 file.
func TestSetupLogModes(t *testing.T) {
	rt := filepath.Join(t.TempDir(), "runtime")
	var buf bytes.Buffer
	if err := Run(buildOpts(rt, &recorder{}, &buf, true)); err != nil {
		t.Fatalf("Run: %v", err)
	}
	lp := logPathFor(rt)
	fi, err := os.Stat(lp)
	if err != nil {
		t.Fatalf("log missing: %v", err)
	}
	if fi.Mode().Perm() != 0o600 {
		t.Errorf("log mode = %o, want 600", fi.Mode().Perm())
	}
	di, err := os.Stat(filepath.Dir(lp))
	if err != nil {
		t.Fatalf("log dir missing: %v", err)
	}
	if di.Mode().Perm() != 0o700 {
		t.Errorf("log dir mode = %o, want 700", di.Mode().Perm())
	}
}
