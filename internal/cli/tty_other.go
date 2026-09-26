//go:build !linux && !darwin && !windows

package cli

import "os"

// isTTY falls back to "not a terminal" on untested platforms — the
// conservative direction here (non-interactive path, no prompts).
func isTTY(f *os.File) bool { return false }
