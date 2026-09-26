//go:build windows

// Windows disk probe: stdlib syscall only (no golang.org/x/sys).
// GetDiskFreeSpaceExW via kernel32 — the stdlib equivalent of statfs.
//
// Status: compiles, NOT tested at runtime — Windows is not a supported
// platform in v1 (ARCHITECTURE.md §8).
package setup

import (
	"fmt"
	"syscall"
	"unsafe"
)

var (
	kernel32             = syscall.NewLazyDLL("kernel32.dll")
	procDiskFreeSpaceExW = kernel32.NewProc("GetDiskFreeSpaceExW")
)

// diskFree reports the bytes available to this user on dir's
// filesystem: lpFreeBytesAvailable, the quota-aware figure and closest
// match to statfs Bavail. Errors surface as syscall.Errno, same
// formatting as the unix probe — callers stay platform-agnostic.
func diskFree(dir string) (int64, error) {
	p, err := syscall.UTF16PtrFromString(dir)
	if err != nil {
		return 0, fmt.Errorf("disk path %q: %w", dir, err)
	}
	var avail, total, free uint64
	r1, _, e1 := procDiskFreeSpaceExW.Call(
		uintptr(unsafe.Pointer(p)),
		uintptr(unsafe.Pointer(&avail)),
		uintptr(unsafe.Pointer(&total)),
		uintptr(unsafe.Pointer(&free)),
	)
	if r1 == 0 {
		if errno, ok := e1.(syscall.Errno); ok && errno == 0 {
			return 0, fmt.Errorf("GetDiskFreeSpaceExW(%q) failed with no error code", dir)
		}
		return 0, e1
	}
	return int64(avail), nil
}
