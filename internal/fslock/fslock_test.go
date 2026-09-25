package fslock

import (
	"errors"
	"path/filepath"
	"testing"
)

func TestTryAcquireExclusiveAndRelease(t *testing.T) {
	path := filepath.Join(t.TempDir(), ".pull.lock")

	l1, err := TryAcquire(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := TryAcquire(path); !errors.Is(err, ErrBusy) {
		t.Errorf("second TryAcquire = %v, want ErrBusy", err)
	}
	if err := l1.Release(); err != nil {
		t.Fatal(err)
	}

	l2, err := TryAcquire(path)
	if err != nil {
		t.Fatalf("acquire after release: %v", err)
	}
	// double release is harmless
	if err := l2.Release(); err != nil {
		t.Errorf("first release: %v", err)
	}
	if err := l2.Release(); err != nil {
		t.Errorf("double release: %v", err)
	}
}

func TestAcquireReacquiresAfterAllReleases(t *testing.T) {
	path := filepath.Join(t.TempDir(), "x.lock")
	for i := 0; i < 3; i++ {
		l, err := TryAcquire(path)
		if err != nil {
			t.Fatalf("cycle %d: %v", i, err)
		}
		l.Release()
	}
}
