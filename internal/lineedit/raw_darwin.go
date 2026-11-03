//go:build darwin

package lineedit

import (
	"os"
	"syscall"
	"unsafe"
)

// Raw switches f into raw mode via TIOCGETA/TIOCSETA (the BSD equivalents
// of TCGETS/TCSETS) and returns a restore func for the saved termios.
func Raw(f *os.File) (func(), error) {
	var t syscall.Termios
	if _, _, errno := syscall.Syscall6(syscall.SYS_IOCTL, f.Fd(),
		uintptr(syscall.TIOCGETA), uintptr(unsafe.Pointer(&t)), 0, 0, 0); errno != 0 {
		return nil, errno
	}
	old := t
	t.Lflag &^= syscall.ECHO | syscall.ICANON | syscall.ISIG
	t.Iflag &^= syscall.IXON | syscall.ICRNL
	t.Cc[syscall.VMIN] = 1
	t.Cc[syscall.VTIME] = 0
	if _, _, errno := syscall.Syscall6(syscall.SYS_IOCTL, f.Fd(),
		uintptr(syscall.TIOCSETA), uintptr(unsafe.Pointer(&t)), 0, 0, 0); errno != 0 {
		return nil, errno
	}
	return func() {
		syscall.Syscall6(syscall.SYS_IOCTL, f.Fd(),
			uintptr(syscall.TIOCSETA), uintptr(unsafe.Pointer(&old)), 0, 0, 0)
	}, nil
}

// RawSupported reports whether raw-mode editing is available here.
func RawSupported() bool { return true }
