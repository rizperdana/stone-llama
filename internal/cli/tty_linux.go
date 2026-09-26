//go:build linux

package cli

import (
	"os"
	"syscall"
	"unsafe"
)

// isTTY reports whether f is a real terminal via TCGETS. os.ModeCharDevice
// is not enough: /dev/null is a character device but not a TTY, and
// prompting on it breaks piped/scripted runs.
func isTTY(f *os.File) bool {
	var t syscall.Termios
	_, _, errno := syscall.Syscall6(syscall.SYS_IOCTL, f.Fd(),
		uintptr(syscall.TCGETS), uintptr(unsafe.Pointer(&t)), 0, 0, 0)
	return errno == 0
}
