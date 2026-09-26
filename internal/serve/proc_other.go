//go:build !linux && !darwin && !windows

package serve

// procCmdline is unknown on untested platforms — "" (Stop's guard then
// treats the pid as not ours, the conservative direction).
func procCmdline(pid int) string { return "" }
