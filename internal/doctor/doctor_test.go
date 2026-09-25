package doctor

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fakeSMI installs an nvidia-smi stub running the given shell lines and
// points PATH at it exclusively, so tests never touch the real driver.
func fakeSMI(t *testing.T, lines ...string) {
	t.Helper()
	dir := t.TempDir()
	script := "#!/bin/sh\n"
	for _, l := range lines {
		script += l + "\n"
	}
	if err := os.WriteFile(filepath.Join(dir, "nvidia-smi"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir)
}

func TestProbeCu13Ready(t *testing.T) {
	fakeSMI(t, `echo "NVIDIA GeForce RTX 3050 Laptop GPU, 4096, 580.178.04"`)
	r, err := Probe()
	if err != nil {
		t.Fatal(err)
	}
	if !r.Ready() || r.Extra != "cu13" {
		t.Fatalf("Ready/Extra = %v/%q, want true/cu13", r.Ready(), r.Extra)
	}
	if len(r.GPUs) != 1 {
		t.Fatalf("GPUs = %d, want 1", len(r.GPUs))
	}
	g := r.GPUs[0]
	if g.Name != "NVIDIA GeForce RTX 3050 Laptop GPU" || g.VRAMMiB != 4096 || g.Driver != "580.178.04" {
		t.Errorf("GPU = %+v", g)
	}
	out := r.Format()
	if !strings.Contains(out, "cu13") || !strings.Contains(out, "RTX 3050") {
		t.Errorf("Format missing fields:\n%s", out)
	}
}

func TestProbeCu12Fallback(t *testing.T) {
	fakeSMI(t, `echo "NVIDIA GeForce GTX 1660, 6144, 570.124.06"`)
	r, err := Probe()
	if err != nil {
		t.Fatal(err)
	}
	if !r.Ready() || r.Extra != "cu12" {
		t.Fatalf("Ready/Extra = %v/%q, want true/cu12", r.Ready(), r.Extra)
	}
	if !strings.Contains(r.Format(), "cu12") {
		t.Errorf("Format should mention cu12:\n%s", r.Format())
	}
}

func TestProbeDriverTooOld(t *testing.T) {
	fakeSMI(t, `echo "NVIDIA GeForce GTX 1050, 4096, 550.90.07"`)
	r, err := Probe()
	if err != nil {
		t.Fatal(err)
	}
	if r.Ready() {
		t.Fatal("driver 550 must not be ready")
	}
	if !strings.Contains(r.Format(), "too old") {
		t.Errorf("Format should say too old:\n%s", r.Format())
	}
}

func TestProbeNoNvidiaSmiGivesOllamaGuidance(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	r, err := Probe()
	if err != nil {
		t.Fatal(err)
	}
	if r.Ready() {
		t.Fatal("no nvidia-smi must not be ready")
	}
	out := r.Format()
	for _, want := range []string{"NVIDIA GPU (CUDA)", "CUDA-only", "ollama with GGUF"} {
		if !strings.Contains(out, want) {
			t.Errorf("Format missing %q:\n%s", want, out)
		}
	}
}

func TestProbeParsesCommaNamesFromTheRight(t *testing.T) {
	fakeSMI(t,
		`echo "NVIDIA, GeForce RTX 4090, 24564, 580.178.04"`,
		`echo "NVIDIA GeForce RTX 3050 Laptop GPU, 4096, 580.178.04"`,
	)
	r, err := Probe()
	if err != nil {
		t.Fatal(err)
	}
	if len(r.GPUs) != 2 {
		t.Fatalf("GPUs = %d, want 2", len(r.GPUs))
	}
	if r.GPUs[0].Name != "NVIDIA, GeForce RTX 4090" || r.GPUs[0].VRAMMiB != 24564 {
		t.Errorf("GPU0 = %+v", r.GPUs[0])
	}
	if !strings.Contains(r.Format(), "GPU#2") {
		t.Errorf("Format should list GPU#2:\n%s", r.Format())
	}
}

func TestProbeSkipsGarbageLines(t *testing.T) {
	fakeSMI(t,
		`echo "garbage"`,
		`echo "NVIDIA GeForce RTX 3050 Laptop GPU, 4096, 580.178.04"`,
	)
	r, err := Probe()
	if err != nil {
		t.Fatal(err)
	}
	if len(r.GPUs) != 1 || !r.Ready() {
		t.Fatalf("GPUs/Ready = %d/%v, want 1/true", len(r.GPUs), r.Ready())
	}
}
