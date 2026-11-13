package serve

import (
	"os"
	"strings"
	"testing"

	"github.com/rizperdana/stone-llama/internal/autofit"
	"github.com/rizperdana/stone-llama/internal/config"
)

// Drafting is boot-YAML-only: fitLoadArgs must never carry draft keys —
// the pinned ModelLoadRequest would drop them at validation, so emitting
// them would lie about what the engine does.
func TestFitLoadArgsHasNoDraftKeys(t *testing.T) {
	args := fitLoadArgs(&autofit.Result{ChunkSize: 2048, Warmup: true})
	for _, k := range []string{"draft_mode", "ngram_match_min", "draft_num_tokens", "draft_model_dir"} {
		if v, ok := args[k]; ok {
			t.Errorf("load payload must not carry %s, got %v", k, v)
		}
	}
	if args["chunk_size"] != 2048 || args["warmup"] != true {
		t.Errorf("chunk/warmup verdict lost: %#v", args)
	}
}

// bootTabbyConfig is the single place draft settings enter the child:
// the configured values reach the rendered boot YAML (RenderTabbyYAML
// owns the draft_model block rules), the default (unset) renders with no
// block at all, and draft_model_dir never renders.
func TestBootTabbyConfigCarriesDraftToYAML(t *testing.T) {
	tc := bootTabbyConfig(Options{
		ModelsDir: "m",
		Fit:       config.Autofit{DraftMode: "ngram", NgramMatchMin: 4, DraftNumTokens: 6},
	}, 5111)
	if tc.DraftMode != "ngram" || tc.NgramMatchMin != 4 || tc.DraftNumTokens != 6 {
		t.Fatalf("boot TabbyConfig draft = %q/%d/%d, want ngram/4/6",
			tc.DraftMode, tc.NgramMatchMin, tc.DraftNumTokens)
	}
	yaml := RenderTabbyYAML(tc)
	for _, want := range []string{
		"draft_model:\n  draft_mode: \"ngram\"\n",
		"  ngram_match_min: 4\n",
		"  draft_num_tokens: 6\n",
	} {
		if !strings.Contains(yaml, want) {
			t.Errorf("boot YAML missing %q:\n%s", want, yaml)
		}
	}
	if strings.Contains(yaml, "draft_model_dir") {
		t.Errorf("draft_model_dir must never render:\n%s", yaml)
	}

	// Default off: no draft_model block — TabbyAPI's own default is
	// draft_mode "model", which drafts nothing without a draft model.
	def := bootTabbyConfig(Options{ModelsDir: "m"}, 5111)
	if got := RenderTabbyYAML(def); strings.Contains(got, "draft_model:") || strings.Contains(got, "draft_mode") {
		t.Errorf("default boot YAML must carry no draft block:\n%s", got)
	}
}

// childEnv: engine_env extras reach the child, the parent environment
// stays (PATH/venv discovery), extras append after it sorted, and a
// config key lands after its parent namesake (exec keeps the last).
func TestChildEnvMergesParentAndExtras(t *testing.T) {
	t.Setenv("SL_CHILDENV_PARENT", "from-parent")
	env := childEnv(map[string]string{"Z_VAR": "z", "A_VAR": "a", "SL_CHILDENV_PARENT": "from-config"})
	base := len(os.Environ())

	idx := func(v string) int {
		for i, kv := range env {
			if kv == v {
				return i
			}
		}
		return -1
	}
	prefixIdx := func(p string) int {
		for i, kv := range env {
			if strings.HasPrefix(kv, p) {
				return i
			}
		}
		return -1
	}

	if iA, iZ := idx("A_VAR=a"), idx("Z_VAR=z"); iA < 0 || iZ < 0 {
		t.Fatalf("extras missing from child env: %v", env)
	} else if iA < base || iZ < base {
		t.Errorf("extras must append after the parent environment: A=%d Z=%d base=%d", iA, iZ, base)
	} else if iA > iZ {
		t.Errorf("extras must be sorted: A_VAR=%d after Z_VAR=%d", iA, iZ)
	}
	if prefixIdx("PATH=") < 0 {
		t.Errorf("parent environment dropped (PATH missing): %v", env)
	}
	iP, iC := idx("SL_CHILDENV_PARENT=from-parent"), idx("SL_CHILDENV_PARENT=from-config")
	if iP < 0 || iC < 0 || iC < iP {
		t.Errorf("config value must override the parent's (last wins): parent=%d config=%d", iP, iC)
	}
}
