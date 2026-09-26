//go:build windows

// Windows disk probe: stdlib syscall only (no golang.org/x/sys).
// GetDiskFreeSpaceExW's lpFreeBytesAvailable is the quota-aware figure,
// the closest match to statfs Bavail.
//
// Status: compiles, NOT tested at runtime — Windows is not a supported
// platform in v1 (ARCHITECTURE.md §8).
package pull

import (
	"fmt"
	"syscall"
	"unsafe"
)

var (
	kernel32             = syscall.NewLazyDLL("kernel32.dll")
	procDiskFreeSpaceExW = kernel32.NewProc("GetDiskFreeSpaceExW")
)

// defaultFreeBytes reports filesystem free space for the models dir.
func defaultFreeBytes(path string) (int64, error) {
	p, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return 0, fmt.Errorf("disk path %q: %w", path, err)
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
			return 0, fmt.Errorf("GetDiskFreeSpaceExW(%q) failed with no error code", path)
		}
		return 0, e1
	}
	return int64(avail), nil
}
