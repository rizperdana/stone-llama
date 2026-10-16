//go:build linux

package doctor

import (
	"os"
	"strconv"
	"strings"
)

// cpuInfo returns the CPU model name and total system RAM in MiB, read from
// /proc. Unreadable or unparseable input yields the zero value for that
// field: the report is diagnostic output, so missing CPU facts degrade the
// message rather than failing the call.
func cpuInfo() (string, int) {
	return cpuName(), memTotalMiB()
}

func cpuName() string {
	b, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(b), "\n") {
		name, value, ok := strings.Cut(line, ":")
		if !ok || strings.TrimSpace(name) != "model name" {
			continue
		}
		return strings.TrimSpace(value)
	}
	return ""
}

func memTotalMiB() int {
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(b), "\n") {
		name, value, ok := strings.Cut(line, ":")
		if !ok || strings.TrimSpace(name) != "MemTotal" {
			continue
		}
		fields := strings.Fields(value) // "123456 kB"
		if len(fields) == 0 {
			return 0
		}
		kb, err := strconv.Atoi(fields[0])
		if err != nil {
			return 0
		}
		return kb / 1024
	}
	return 0
}
