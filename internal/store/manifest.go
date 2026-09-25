package store

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// FileEntry records one downloaded file for integrity checks.
type FileEntry struct {
	Path   string `json:"path"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256,omitempty"`
}

// Verdict is the A5 pre-download gate result, recorded at pull time (M2).
type Verdict struct {
	Status    string `json:"status"` // "fit" | "warn" | "refuse"
	MaxCtx    int    `json:"max_ctx"`
	CacheMode string `json:"cache_mode"`
	Note      string `json:"note,omitempty"`
}

// Verdict status values.
const (
	VerdictOK     = "fit"
	VerdictWarn   = "warn"
	VerdictRefuse = "refuse"
)

// Manifest is the pull provenance + integrity record stored as
// manifest.json inside a model directory. Directories without one are
// listed as imported/unmanaged.
type Manifest struct {
	RepoID   string      `json:"repo_id"`
	Revision string      `json:"revision"`
	Quant    string      `json:"quant"`
	PulledAt time.Time   `json:"pulled_at"`
	Files    []FileEntry `json:"files"`
	Verdict  *Verdict    `json:"verdict,omitempty"`
}

const manifestName = "manifest.json"

// LoadManifest reads dir/manifest.json. Missing → (nil, nil);
// corrupt → error (silent data loss is worse than a loud failure).
func LoadManifest(dir string) (*Manifest, error) {
	path := filepath.Join(dir, manifestName)
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("manifest %s: %w", path, err)
	}
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("manifest %s: %w", path, err)
	}
	return &m, nil
}

// SaveManifest writes dir/manifest.json atomically (temp + rename) so an
// interrupted pull never leaves a half-written manifest behind.
func SaveManifest(dir string, m *Manifest) error {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Errorf("encode manifest: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".manifest-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return err
	}
	if err := os.Chmod(tmpName, 0o644); err != nil {
		os.Remove(tmpName)
		return err
	}
	if err := os.Rename(tmpName, filepath.Join(dir, manifestName)); err != nil {
		os.Remove(tmpName)
		return err
	}
	return nil
}
