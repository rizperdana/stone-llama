//go:build !linux && !darwin

package lineedit

import (
	"errors"
	"os"
)

// ErrUnsupported marks raw-mode editing unavailable on this platform —
// callers keep the cooked-mode read path (windows keeps its console).
var ErrUnsupported = errors.New("lineedit: raw mode unsupported on this platform")

// Raw is unsupported off unix — callers keep the cooked-mode read path.
func Raw(f *os.File) (func(), error) { return nil, ErrUnsupported }

// RawSupported reports whether raw-mode editing is available.
func RawSupported() bool { return false }
