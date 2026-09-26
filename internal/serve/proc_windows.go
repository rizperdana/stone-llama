//go:build windows

package serve

// procCmdline returns "" on Windows: Stop is refused earlier on this
// platform, and an empty cmdline must never be read as a match.
func procCmdline(pid int) string { return "" }
