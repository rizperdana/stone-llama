//go:build !windows

package setup

import "syscall"

// diskFree reports the bytes available to this user on dir's
// filesystem: statfs Bavail (excludes root-reserved blocks), the
// figure a download preflight should promise.
func diskFree(dir string) (int64, error) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(dir, &st); err != nil {
		return 0, err
	}
	return int64(st.Bavail) * int64(st.Bsize), nil
}
