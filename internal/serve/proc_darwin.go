//go:build darwin

package serve

import "strconv"

// procCmdline uses `ps -o args=` (no /proc on darwin); "" when the pid
// is gone. Used by Stop to refuse signalling a recycled pid.
func procCmdline(pid int) string {
	if pid <= 0 {
		return ""
	}
	return execOutput("ps", "-p", strconv.Itoa(pid), "-o", "args=")
}
