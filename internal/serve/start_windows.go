//go:build windows

package serve

import "errors"

// StartDetached is unsupported on Windows (v1 targets Linux/macOS;
// the windows build compiles but is untested at runtime) — start the
// daemon in a console instead.
func StartDetached(exe, dataDir string) (<-chan struct{}, error) {
	return nil, errors.New("auto-start is not supported on windows — run 'stone-llama serve' directly")
}
