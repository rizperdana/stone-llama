//go:build linux

package serve

import (
	"os"
	"strconv"
	"strings"
)

// procCmdline reads /proc/<pid>/cmdline (NUL-separated argv → spaces),
// "" when the pid is gone. Used by Stop to refuse signalling a recycled
// pid that is no longer a stone-llama process.
func procCmdline(pid int) string {
	if pid <= 0 {
		return ""
	}
	b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/cmdline")
	if err != nil {
		return ""
	}
	return strings.ReplaceAll(string(b), "\x00", " ")
}
