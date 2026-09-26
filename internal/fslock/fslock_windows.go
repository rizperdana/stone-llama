//go:build windows

// Windows lock primitive: stdlib syscall only (no golang.org/x/sys).
// Byte-range locks via kernel32 LockFileEx/UnlockFileEx — the closest
// stdlib-only equivalent of flock(LOCK_EX|LOCK_NB). Locks are
// mandatory and die with the handle, like flock.
//
// Status: compiles, NOT tested at runtime — Windows is not a supported
// platform in v1 (ARCHITECTURE.md §8). If a future Windows runtime
// path needs more than this, prove it there.
package fslock

import (
	"errors"
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

var (
	kernel32         = syscall.NewLazyDLL("kernel32.dll")
	procLockFileEx   = kernel32.NewProc("LockFileEx")
	procUnlockFileEx = kernel32.NewProc("UnlockFileEx")
)

// LockFileEx flags (winbase.h).
const (
	lockfileFailImmediately = 0x00000001
	lockfileExclusiveLock   = 0x00000002
)

// tryLock takes a non-blocking exclusive lock on byte 0 of the lock
// file. When another process holds the range, LockFileEx fails
// immediately — mapped to ErrBusy, never silently ignored; any other
// failure is returned as an error so a lock that can't be taken is
// loud, not a no-op.
func tryLock(f *os.File) error {
	var ov syscall.Overlapped
	r1, _, e1 := procLockFileEx.Call(f.Fd(),
		lockfileFailImmediately|lockfileExclusiveLock,
		0 /* reserved */, 1 /* byte count: low */, 0, /* high */
		uintptr(unsafe.Pointer(&ov)))
	if r1 != 0 {
		return nil
	}
	if errno, ok := e1.(syscall.Errno); ok {
		switch errno {
		case 33, // ERROR_LOCK_VIOLATION: held by another process
			32,  // ERROR_SHARING_VIOLATION: file open without sharing
			997: // ERROR_IO_PENDING: lock would block
			return ErrBusy
		case 0:
			return errors.New("fslock: LockFileEx failed with no error code")
		}
	}
	return fmt.Errorf("fslock: LockFileEx: %w", e1)
}

// unlock releases the byte-range lock. Closing the handle would also
// drop it, so a failure here is not fatal — but try it explicitly to
// mirror the unix path.
func unlock(f *os.File) {
	var ov syscall.Overlapped
	procUnlockFileEx.Call(f.Fd(), 0, 1, 0, uintptr(unsafe.Pointer(&ov)))
}
