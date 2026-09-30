package setup

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
)

// Two concurrent setups must not race: while the first holds the
// setup lock, a second is refused instead of loading the same journal
// and clobbering bin/uv and the step markers.
func TestRunRefusesConcurrentSetup(t *testing.T) {
	rt := filepath.Join(t.TempDir(), "runtime")
	if err := os.MkdirAll(rt, 0o750); err != nil {
		t.Fatal(err)
	}
	lk, err := os.Create(filepath.Join(rt, "setup.lock"))
	if err != nil {
		t.Fatal(err)
	}
	defer lk.Close()
	if err := syscall.Flock(int(lk.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		t.Fatalf("hold lock: %v", err)
	}
	var buf bytes.Buffer
	if err := Run(buildOpts(rt, &recorder{}, &buf, true)); err == nil || !strings.Contains(err.Error(), "another stone-llama") {
		t.Fatalf("err = %v, want lock contention", err)
	}
}
