package pull

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestLivePullGateZeroDownload runs the full pull pipeline against the
// real HuggingFace Hub and real gate arithmetic, stopping before any
// weight byte: once by gate refusal (8B@4.0bpw on 4096 MiB cannot fit)
// and once by the non-interactive consent requirement (absurd VRAM fits →
// --yes needed). A5 sanctions the KB-scale metadata fetches; the test
// asserts the models dir holds nothing but the lock file afterwards:
//
//	HF_LIVE=1 go test ./internal/pull -run Live -v
func TestLivePullGateZeroDownload(t *testing.T) {
	if os.Getenv("HF_LIVE") == "" {
		t.Skip("set HF_LIVE=1 for a live HuggingFace pull-gate smoke")
	}

	models := t.TempDir()
	base := Options{
		ModelsDir:  models,
		Ref:        "async0x42/Qwen3-8B-exl3_4.0bpw",
		GPUName:    "RTX 3050",
		FreeBytes:  func(string) (int64, error) { return 1 << 40, nil },
		RetryDelay: func(int) time.Duration { return 0 },
	}

	// 1) Real weights (~5 GB) on a 4096 MiB card → gate refusal.
	var out1 bytes.Buffer
	opts1 := base
	opts1.Out = &out1
	opts1.VRAMMiB = 4096
	_, err := Run(opts1)
	if err == nil || !strings.Contains(err.Error(), "refused by pre-download gate") {
		t.Fatalf("want gate refusal, got: %v", err)
	}
	if !strings.Contains(out1.String(), "gate: fit") {
		t.Errorf("fit report missing:\n%s", out1.String())
	}

	// 2) Absurd VRAM → gate passes → non-interactive consent stops it.
	var out2 bytes.Buffer
	opts2 := base
	opts2.Out = &out2
	opts2.VRAMMiB = 1 << 20
	_, err = Run(opts2)
	if err == nil || !strings.Contains(err.Error(), "--yes") {
		t.Fatalf("want --yes consent requirement, got: %v", err)
	}
	if !strings.Contains(out2.String(), "gate: fit") {
		t.Errorf("fit report missing:\n%s", out2.String())
	}

	// Zero download: only the flock file may exist in the models dir.
	entries, rerr := os.ReadDir(models)
	if rerr != nil {
		t.Fatal(rerr)
	}
	for _, e := range entries {
		if e.Name() != ".pull.lock" {
			t.Errorf("unexpected artifact %q — weight bytes must never move", e.Name())
		}
	}
	if _, serr := os.Stat(filepath.Join(models, ".Qwen3-8B-exl3_4.0bpw.staging")); serr == nil {
		t.Error("staging must not exist — consent gate ran before mkdir")
	}
}
