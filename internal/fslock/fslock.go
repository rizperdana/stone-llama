// Package fslock provides a cross-process exclusive lock (flock) so two
// stone-llama processes can't race to spawn the daemon or corrupt a
// download (A7, ARCHITECTURE.md §3).
package fslock

import (
	"errors"
	"os"
	"path/filepath"
	"syscall"
)

// ErrBusy: the lock is held by another process.
var ErrBusy = errors.New("lock is held by another stone-llama process")

type Lock struct{ f *os.File }

// TryAcquire acquires the exclusive lock without blocking; returns
// ErrBusy when another process holds it. (Separate open()s are separate
// file descriptions, so this detects a competing holder even within one
// test process.)
func TryAcquire(path string) (*Lock, error) {
	f, err := open(path)
	if err != nil {
		return nil, err
	}
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		f.Close()
		if errors.Is(err, syscall.EWOULDBLOCK) || errors.Is(err, syscall.EAGAIN) {
			return nil, ErrBusy
		}
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
	syscall.Flock(int(l.f.Fd()), syscall.LOCK_UN)
	err := l.f.Close()
	l.f = nil
	return err
}
