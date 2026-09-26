// Package fslock provides a cross-process exclusive lock so two
// stone-llama processes can't race to spawn the daemon or corrupt a
// download (A7, ARCHITECTURE.md §3).
//
// The primitive is platform-specific — flock on Unix
// (fslock_unix.go), LockFileEx on Windows (fslock_windows.go); this
// file is the shared API.
package fslock

import (
	"errors"
	"os"
	"path/filepath"
)

// ErrBusy: the lock is held by another process.
var ErrBusy = errors.New("lock is held by another stone-llama process")

type Lock struct{ f *os.File }

// TryAcquire acquires the exclusive lock without blocking; returns
// ErrBusy when another process holds it.
func TryAcquire(path string) (*Lock, error) {
	f, err := open(path)
	if err != nil {
		return nil, err
	}
	if err := tryLock(f); err != nil {
		f.Close()
		return nil, err
	}
	return &Lock{f: f}, nil
}

func open(path string) (*os.File, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, err
	}
	return os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
}

// Release drops the lock and closes the file. Safe on a nil/already
// released lock.
func (l *Lock) Release() error {
	if l == nil || l.f == nil {
		return nil
	}
	unlock(l.f)
	err := l.f.Close()
	l.f = nil
	return err
}
