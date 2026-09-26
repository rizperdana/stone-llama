package store

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// makeModel creates a fake model dir (config.json + weights of set size).
func makeModel(t *testing.T, parent, name string, weightBytes int) string {
	t.Helper()
	dir := filepath.Join(parent, name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"architectures":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "weights.bin"), make([]byte, weightBytes), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestScanMissingDirIsEmptyNotError(t *testing.T) {
	models, err := Scan(filepath.Join(t.TempDir(), "nope"))
	if err != nil || models != nil {
		t.Errorf("Scan = %v, %v; want nil, nil", models, err)
	}
}

func TestScanClassifiesAndSorts(t *testing.T) {
	modelsDir := t.TempDir()
	extDir := makeModel(t, t.TempDir(), "external-model", 4096)

	// pulled model: manifest present
	pulled := makeModel(t, modelsDir, "zz-pulled", 2048)
	man := &Manifest{
		RepoID:   "turboderp/zz-pulled-exl3",
		Revision: "abc123",
		Quant:    "3.5bpw",
		PulledAt: time.Now(),
		Files:    []FileEntry{{Path: "weights.bin", Size: 2048, SHA256: "deadbeef"}},
		Verdict:  &Verdict{Status: "fit", MaxCtx: 65536, CacheMode: "Q4"},
	}
	if err := SaveManifest(pulled, man); err != nil {
		t.Fatal(err)
	}

	// plain dir with config.json only → unmanaged
	makeModel(t, modelsDir, "aa-unmanaged", 1024)

	// imported symlink
	if err := os.Symlink(extDir, filepath.Join(modelsDir, "mm-imported")); err != nil {
		t.Fatal(err)
	}

	// non-models: plain file, hidden dir, dir without config/manifest, broken symlink
	if err := os.WriteFile(filepath.Join(modelsDir, "notes.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(modelsDir, ".staging"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(modelsDir, "junk"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(t.TempDir(), "gone"), filepath.Join(modelsDir, "broken")); err != nil {
		t.Fatal(err)
	}

	models, err := Scan(modelsDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 3 {
		t.Fatalf("got %d models, want 3: %+v", len(models), models)
	}
	// sorted by name
	names := []string{models[0].Name, models[1].Name, models[2].Name}
	wantOrder := []string{"aa-unmanaged", "mm-imported", "zz-pulled"}
	for i, w := range wantOrder {
		if names[i] != w {
			t.Fatalf("order = %v, want %v", names, wantOrder)
		}
	}

	if models[0].Source != "unmanaged" || models[0].Quant != "-" {
		t.Errorf("aa-unmanaged = %+v", models[0])
	}
	if models[1].Source != "imported" {
		t.Errorf("mm-imported = %+v", models[1])
	}
	if models[1].SizeBytes != int64(4096+len(`{"architectures":[]}`)) {
		t.Errorf("imported size = %d", models[1].SizeBytes)
	}
	p := models[2]
	if p.Source != "turboderp/zz-pulled-exl3" || p.Quant != "3.5bpw" {
		t.Errorf("pulled provenance = %+v", p)
	}
	if p.Verdict == nil || p.Verdict.Status != "fit" || p.Verdict.MaxCtx != 65536 || p.Verdict.CacheMode != "Q4" {
		t.Errorf("pulled verdict = %+v", p.Verdict)
	}
	if p.SizeBytes != int64(2048+len(`{"architectures":[]}`)) {
		t.Errorf("pulled size = %d", p.SizeBytes)
	}
}

func TestScanCorruptManifestIsError(t *testing.T) {
	modelsDir := t.TempDir()
	dir := makeModel(t, modelsDir, "bad", 8)
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), []byte("{oops"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Scan(modelsDir); err == nil || !strings.Contains(err.Error(), "manifest") {
		t.Errorf("err = %v, want manifest parse error", err)
	}
}

func TestManifestRoundtrip(t *testing.T) {
	dir := t.TempDir()
	in := &Manifest{
		RepoID:   "org/model-exl3",
		Revision: "rev",
		Quant:    "4.0bpw",
		PulledAt: time.Date(2026, 9, 25, 12, 0, 0, 0, time.UTC),
		Files:    []FileEntry{{Path: "model.safetensors", Size: 100, SHA256: "aa"}},
		Verdict:  &Verdict{Status: "warn", MaxCtx: 32768, CacheMode: "Q4", Note: "tight"},
	}
	if err := SaveManifest(dir, in); err != nil {
		t.Fatal(err)
	}
	out, err := LoadManifest(dir)
	if err != nil {
		t.Fatal(err)
	}
	if out.RepoID != in.RepoID || out.Quant != in.Quant || !out.PulledAt.Equal(in.PulledAt) ||
		len(out.Files) != 1 || out.Files[0].SHA256 != "aa" ||
		out.Verdict == nil || out.Verdict.Status != "warn" || out.Verdict.MaxCtx != 32768 {
		t.Errorf("roundtrip mismatch: %+v", out)
	}
	// missing manifest → nil, nil
	if m, err := LoadManifest(t.TempDir()); m != nil || err != nil {
		t.Errorf("missing manifest = %v, %v; want nil, nil", m, err)
	}
}

func TestImport(t *testing.T) {
	modelsDir := filepath.Join(t.TempDir(), "models")
	src := makeModel(t, t.TempDir(), "src-model", 512)

	link, err := Import(modelsDir, src, "my-model")
	if err != nil {
		t.Fatal(err)
	}
	fi, err := os.Lstat(link)
	if err != nil || fi.Mode()&os.ModeSymlink == 0 {
		t.Fatalf("import did not create a symlink: %v %v", fi, err)
	}

	// conflict
	if _, err := Import(modelsDir, src, "my-model"); err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Errorf("conflict err = %v", err)
	}
	// source without config.json
	plain := t.TempDir()
	if _, err := Import(modelsDir, plain, "plain"); err == nil || !strings.Contains(err.Error(), "config.json") {
		t.Errorf("no-config err = %v", err)
	}
	// source already inside modelsDir
	if _, err := Import(modelsDir, src, "x"); err != nil {
		t.Fatalf("first import of second model: %v", err)
	}
	if _, err := Import(modelsDir, filepath.Join(modelsDir, "x"), "y"); err == nil || !strings.Contains(err.Error(), "already inside") {
		t.Errorf("inside-dir err = %v", err)
	}
	// invalid names
	for _, bad := range []string{"", ".", "..", "a/b"} {
		if _, err := Import(modelsDir, src, bad); err == nil || !strings.Contains(err.Error(), "invalid") {
			t.Errorf("name %q: err = %v", bad, err)
		}
	}
}

func TestRemove(t *testing.T) {
	modelsDir := t.TempDir()
	dir := makeModel(t, modelsDir, "victim", 64)
	if err := Remove(modelsDir, "victim"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Errorf("dir still exists: %v", err)
	}

	// symlink: link removed, target intact
	ext := makeModel(t, t.TempDir(), "keep", 64)
	if err := os.Symlink(ext, filepath.Join(modelsDir, "imp")); err != nil {
		t.Fatal(err)
	}
	if err := Remove(modelsDir, "imp"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Lstat(filepath.Join(modelsDir, "imp")); !os.IsNotExist(err) {
		t.Errorf("link still exists: %v", err)
	}
	if _, err := os.Stat(filepath.Join(ext, "config.json")); err != nil {
		t.Errorf("target was touched: %v", err)
	}

	// missing → named error
	if err := Remove(modelsDir, "ghost"); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Errorf("err = %v, want not found", err)
	}
	// plain file → refused
	if err := os.WriteFile(filepath.Join(modelsDir, "stray"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := Remove(modelsDir, "stray"); err == nil || !strings.Contains(err.Error(), "not a model") {
		t.Errorf("err = %v, want not a model", err)
	}
}

func TestSaveManifestAtomicNoTempResidue(t *testing.T) {
	dir := t.TempDir()
	if err := SaveManifest(dir, &Manifest{RepoID: "org/x", Revision: "abc"}); err != nil {
		t.Fatal(err)
	}
	got, err := LoadManifest(dir)
	if err != nil || got == nil || got.RepoID != "org/x" || got.Revision != "abc" {
		t.Fatalf("LoadManifest = %+v, %v", got, err)
	}
	if residue, _ := filepath.Glob(filepath.Join(dir, "*.tmp")); len(residue) != 0 {
		t.Errorf("temp residue: %v", residue)
	}
}

func TestSaveManifestFailureKeepsOldManifest(t *testing.T) {
	dir := t.TempDir()
	if err := SaveManifest(dir, &Manifest{RepoID: "org/old"}); err != nil {
		t.Fatal(err)
	}
	orig := renameFile
	renameFile = func(string, string) error { return errors.New("forced rename failure") }
	t.Cleanup(func() { renameFile = orig })

	if err := SaveManifest(dir, &Manifest{RepoID: "org/new"}); err == nil {
		t.Fatal("want forced failure")
	}
	got, err := LoadManifest(dir)
	if err != nil || got == nil || got.RepoID != "org/old" {
		t.Fatalf("old manifest lost: %+v, %v", got, err)
	}
	if residue, _ := filepath.Glob(filepath.Join(dir, "*.tmp")); len(residue) != 0 {
		t.Errorf("temp residue after failure: %v", residue)
	}
}
