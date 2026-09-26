//go:build !windows

package fslock

import (
	"errors"
	"os"
	"syscall"
)

// tryLock mirrors flock(LOCK_EX|LOCK_NB). (Separate open()s are
// separate file descriptions, so this detects a competing holder even
// within one test process.)
func tryLock(f *os.File) error {
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		if errors.Is(err, syscall.EWOULDBLOCK) || errors.Is(err, syscall.EAGAIN) {
			return ErrBusy
		}
		return err
	}
	return nil
}

// unlock releases the flock; the close in Release drops it regardless.
func unlock(f *os.File) {
	syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
}
