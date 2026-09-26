//go:build darwin

package cli

import (
	"os"
	"syscall"
	"unsafe"
)

// isTTY reports whether f is a real terminal via TIOCGETA (the BSD
// equivalent of TCGETS).
func isTTY(f *os.File) bool {
	var t syscall.Termios
	_, _, errno := syscall.Syscall6(syscall.SYS_IOCTL, f.Fd(),
		uintptr(syscall.TIOCGETA), uintptr(unsafe.Pointer(&t)), 0, 0, 0)
	return errno == 0
}
