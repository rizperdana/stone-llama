//go:build !linux

package doctor

// cpuInfo reports no CPU facts: /proc is not read on this platform.
func cpuInfo() (string, int) { return "", 0 }
