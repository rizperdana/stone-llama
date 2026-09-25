// Package store manages the models directory: scanning for `list`,
// symlinked imports, removal, and manifest read/write. A model is a
// directory shaped like a HF model dir (config.json + weights), which is
// exactly what TabbyAPI loads.
package store

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Model is one entry of `stone-llama list`.
type Model struct {
	Name      string
	Path      string // entry path inside models_dir (symlink for imports)
	SizeBytes int64  // sum of regular file sizes
	Source    string // manifest repo id | "imported" | "unmanaged"
	Quant     string // manifest quant, else "-"
	Verdict   *Verdict
	Manifest  *Manifest
}

// Scan lists model entries in modelsDir sorted by name. A missing
// modelsDir means "no models yet", not an error. A corrupt manifest is
// an error: surface it, don't silently misreport provenance.
func Scan(modelsDir string) ([]Model, error) {
	entries, err := os.ReadDir(modelsDir)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	var models []Model
	for _, e := range entries {
		name := e.Name()
		if strings.HasPrefix(name, ".") {
			continue // staging/hidden files are never models
		}
		entry := filepath.Join(modelsDir, name)

		// Resolve to the real directory: imports are symlinks.
		path := entry
		if e.Type()&os.ModeSymlink != 0 {
			target, err := os.Readlink(entry)
			if err != nil {
				continue
			}
			if !filepath.IsAbs(target) {
				target = filepath.Join(filepath.Dir(entry), target)
			}
			if _, err := os.Stat(target); err != nil {
				continue // broken link: not a model
			}
			path = target
		} else if !e.IsDir() {
			continue
		}

		m, err := LoadManifest(path)
		if err != nil {
			return nil, err
		}
		// Qualify: a model dir carries a manifest or a config.json.
		if m == nil && !fileExists(filepath.Join(path, "config.json")) {
			continue
		}

		source, quant := "unmanaged", "-"
		switch {
		case m != nil:
			source, quant = m.RepoID, m.Quant
		case e.Type()&os.ModeSymlink != 0:
			source = "imported"
		}

		size, err := dirSize(path)
		if err != nil {
			return nil, fmt.Errorf("size of %s: %w", name, err)
		}

		model := Model{
			Name:      name,
			Path:      entry,
			SizeBytes: size,
			Source:    source,
			Quant:     quant,
			Manifest:  m,
		}
		if m != nil {
			model.Verdict = m.Verdict
		}
		models = append(models, model)
	}

	sort.Slice(models, func(i, j int) bool { return models[i].Name < models[j].Name })
	return models, nil
}

// Import symlink-joins an existing model directory into modelsDir
// (zero-copy). The source must be a model dir (config.json present) and
// must not already live inside modelsDir.
func Import(modelsDir, src, name string) (string, error) {
	if err := validName(name); err != nil {
		return "", err
	}
	absSrc, err := filepath.Abs(src)
	if err != nil {
		return "", err
	}
	fi, err := os.Stat(absSrc)
	if err != nil {
		return "", fmt.Errorf("source %s: %w", absSrc, err)
	}
	if !fi.IsDir() {
		return "", fmt.Errorf("source %s: not a directory", absSrc)
	}
	if !fileExists(filepath.Join(absSrc, "config.json")) {
		return "", fmt.Errorf("source %s: no config.json — not a model directory", absSrc)
	}
	absModels, err := filepath.Abs(modelsDir)
	if err != nil {
		return "", err
	}
	if absSrc == absModels || strings.HasPrefix(absSrc, absModels+string(os.PathSeparator)) {
		return "", fmt.Errorf("source %s is already inside the models dir", absSrc)
	}

	if err := os.MkdirAll(modelsDir, 0o700); err != nil {
		return "", err
	}
	link := filepath.Join(modelsDir, name)
	if _, err := os.Lstat(link); err == nil {
		return "", fmt.Errorf("model %q already exists", name)
	}
	if err := os.Symlink(absSrc, link); err != nil {
		return "", err
	}
	return link, nil
}

// Remove deletes a model entry. A symlinked import is unlinked only —
// the user's original files are never touched.
func Remove(modelsDir, name string) error {
	if err := validName(name); err != nil {
		return err
	}
	path := filepath.Join(modelsDir, name)
	fi, err := os.Lstat(path)
	if os.IsNotExist(err) {
		return fmt.Errorf("model %q not found", name)
	}
	if err != nil {
		return err
	}
	switch {
	case fi.Mode()&os.ModeSymlink != 0:
		return os.Remove(path)
	case fi.IsDir():
		return os.RemoveAll(path)
	default:
		return fmt.Errorf("%q is not a model directory", name)
	}
}

func validName(name string) error {
	if name == "" || name == "." || name == ".." || strings.ContainsAny(name, `/\`) {
		return fmt.Errorf("invalid model name %q", name)
	}
	return nil
}

func fileExists(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && fi.Mode().IsRegular()
}

func dirSize(root string) (int64, error) {
	var total int64
	manifest := filepath.Join(root, manifestName)
	err := filepath.Walk(root, func(p string, fi os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		// our own metadata, not model payload
		if p == manifest {
			return nil
		}
		if fi.Mode().IsRegular() {
			total += fi.Size()
		}
		return nil
	})
	return total, err
}
