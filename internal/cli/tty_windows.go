//go:build windows

package cli

import (
	"os"
	"syscall"
	"unsafe"
)

// isTTY reports whether f is a console via kernel32 GetConsoleMode —
// stdlib syscall only (same constraint as internal/fslock on Windows).
func isTTY(f *os.File) bool {
	var mode uint32
	r, _, _ := syscall.NewLazyDLL("kernel32.dll").
		NewProc("GetConsoleMode").
		Call(f.Fd(), uintptr(unsafe.Pointer(&mode)))
	return r != 0
}
