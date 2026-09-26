//go:build !windows

package pull

import "syscall"

// defaultFreeBytes reports filesystem free space for the models dir
// (statfs Bavail — excludes root-reserved blocks).
func defaultFreeBytes(path string) (int64, error) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(path, &st); err != nil {
		return 0, err
	}
	return int64(st.Bavail) * int64(st.Bsize), nil
}
