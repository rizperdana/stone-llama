//go:build linux

package lineedit

import (
	"os"
	"syscall"
	"unsafe"
)

// Raw switches f into raw mode (no echo, no line buffering, no ISIG — the
// editor consumes ^C/^D as bytes) and returns a restore func for the saved
// termios. Pure syscall: TCGETS/TCSETS ioctls, no cgo.
func Raw(f *os.File) (func(), error) {
	var t syscall.Termios
	if _, _, errno := syscall.Syscall6(syscall.SYS_IOCTL, f.Fd(),
		uintptr(syscall.TCGETS), uintptr(unsafe.Pointer(&t)), 0, 0, 0); errno != 0 {
		return nil, errno
	}
	old := t
	t.Lflag &^= syscall.ECHO | syscall.ICANON | syscall.ISIG
	t.Iflag &^= syscall.IXON | syscall.ICRNL
	t.Cc[syscall.VMIN] = 1
	t.Cc[syscall.VTIME] = 0
	if _, _, errno := syscall.Syscall6(syscall.SYS_IOCTL, f.Fd(),
		uintptr(syscall.TCSETS), uintptr(unsafe.Pointer(&t)), 0, 0, 0); errno != 0 {
		return nil, errno
	}
	return func() {
		syscall.Syscall6(syscall.SYS_IOCTL, f.Fd(),
			uintptr(syscall.TCSETS), uintptr(unsafe.Pointer(&old)), 0, 0, 0)
	}, nil
}

// RawSupported reports whether raw-mode editing is available here.
func RawSupported() bool { return true }
