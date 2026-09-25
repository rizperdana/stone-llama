package preflight

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// excludedArchs are draft/assistant architectures stone-llama v1 does not
// serve (no speculative-draft support); they are deliberately absent from
// SupportedArchs.
var excludedArchs = map[string]bool{
	"DFlashDraftModel":          true,
	"DFlash2DraftModel":         true,
	"DFlashLagunaForCausalLM":   true,
	"MuseGlimmerAssistantModel": true,
}

// TestReconcileAgainstInstalledExllamav3 re-derives arch_string values
// from a real exllamav3 checkout and fails on any drift in either
// direction. Gated by EXLLAMAV3_ARCH_DIR (machine-specific path):
//
//	EXLLAMAV3_ARCH_DIR=.../exllamav3/architecture go test ./internal/preflight -run Reconcile
func TestReconcileAgainstInstalledExllamav3(t *testing.T) {
	dir := os.Getenv("EXLLAMAV3_ARCH_DIR")
	if dir == "" {
		t.Skip("set EXLLAMAV3_ARCH_DIR to reconcile against a real exllamav3 install")
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	re := regexp.MustCompile(`arch_string\s*=\s*["']([^"']+)["']`)
	installed := map[string]bool{}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".py") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range re.FindAllStringSubmatch(string(data), -1) {
			installed[m[1]] = true
		}
	}
	if len(installed) < 50 {
		t.Fatalf("only %d arch strings parsed from %s — extraction broken", len(installed), dir)
	}

	var missing, stale []string
	for s := range installed {
		if excludedArchs[s] {
			continue
		}
		if !SupportedArchs[s] {
			missing = append(missing, s)
		}
	}
	for s := range SupportedArchs {
		if !installed[s] {
			stale = append(stale, s)
		}
	}
	if len(missing) > 0 {
		t.Errorf("installed exllamav3 declares archs missing from SupportedArchs: %v", missing)
	}
	if len(stale) > 0 {
		t.Errorf("SupportedArchs entries not present in the install: %v", stale)
	}
	for s := range excludedArchs {
		if SupportedArchs[s] {
			t.Errorf("draft/assistant arch %q must stay excluded", s)
		}
	}
}
