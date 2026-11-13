package serve

import (
	"strings"
	"testing"
)

func TestNormalizeForTabby(t *testing.T) {
	cases := map[string]string{
		"FP16": "FP16", "q8": "Q8", "Q4": "Q4", "Q6": "Q6",
		"Q2": "2,2", "Q3": "3,3", "2,4": "2,4",
	}
	for in, want := range cases {
		if got := NormalizeForTabby(in); got != want {
			t.Errorf("NormalizeForTabby(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestRenderTabbyYAML(t *testing.T) {
	got := RenderTabbyYAML(TabbyConfig{
		Host: "127.0.0.1", Port: 41337,
		ModelDir: "/data/models", ModelName: "SmolLM3-3B-exl3",
		Ctx: 65536, CacheMode: "q4",
	})
	want := `network:
  host: "127.0.0.1"
  port: 41337
  api_servers: ["OAI"]

model:
  model_dir: "/data/models"
  model_name: "SmolLM3-3B-exl3"
  max_seq_len: 65536
  cache_size: 65536
  cache_mode: "Q4"
  tool_format: auto
  gpu_split_auto: true
  autosplit_reserve: [96]
`
	if got != want {
		t.Errorf("YAML mismatch:\ngot:\n%s\nwant:\n%s", got, want)
	}

	q2 := RenderTabbyYAML(TabbyConfig{Host: "h", Port: 1, ModelDir: "d", ModelName: "m", Ctx: 4096, CacheMode: "Q2"})
	if !strings.Contains(q2, `cache_mode: "2,2"`) {
		t.Errorf("Q2 must normalize to pair syntax:\n%s", q2)
	}

	// model-less child: ctx/cache keys blank, not zero — TabbyAPI takes
	// its defaults and per-load args override.
	blank := RenderTabbyYAML(TabbyConfig{Host: "h", Port: 1, ModelDir: "d"})
	if strings.Contains(blank, "max_seq_len: 0") || strings.Contains(blank, "cache_mode: \"\"") {
		t.Errorf("blank ctx must emit empty keys, not zeros:\n%s", blank)
	}
	if !strings.Contains(blank, "  max_seq_len:\n  cache_size:\n") ||
		!strings.Contains(blank, "  tool_format: auto\n") {
		t.Errorf("expected blank seq lines + explicit tool_format:\n%s", blank)
	}
}

// draft_model is emitted only when a mode is set, and each numeric key only
// above 0 — the same omit-when-unset rule chunk_size/warmup follow, so the
// model-less boot render stays byte-identical.
func TestRenderTabbyYAMLDraftModel(t *testing.T) {
	boot := RenderTabbyYAML(TabbyConfig{Host: "h", Port: 1, ModelDir: "d"})
	if strings.Contains(boot, "draft_model:") || strings.Contains(boot, "draft_mode") {
		t.Errorf("boot YAML gained a draft block without a mode:\n%s", boot)
	}

	ngram := RenderTabbyYAML(TabbyConfig{
		Host: "h", Port: 1, ModelDir: "d",
		DraftMode: "ngram", NgramMatchMin: 2, DraftNumTokens: 4,
	})
	want := "draft_model:\n  draft_mode: \"ngram\"\n  ngram_match_min: 2\n  draft_num_tokens: 4\n"
	if !strings.Contains(ngram, want) {
		t.Errorf("draft block mismatch, want:\n%s\ngot:\n%s", want, ngram)
	}
	if strings.Contains(ngram, "draft_model_dir") {
		t.Errorf("draft_model_dir must never render (no draft weights shipped):\n%s", ngram)
	}

	// "disabled" is a legitimate backend mode, not an omission.
	if got := RenderTabbyYAML(TabbyConfig{Host: "h", Port: 1, ModelDir: "d", DraftMode: "disabled"}); !strings.Contains(got, "draft_model:\n  draft_mode: \"disabled\"\n") {
		t.Errorf("draft_mode: disabled must still emit the block:\n%s", got)
	}

	// zero numbers keep the backend's own default; only draft_mode is set.
	bare := RenderTabbyYAML(TabbyConfig{Host: "h", Port: 1, ModelDir: "d", DraftMode: "model"})
	if !strings.Contains(bare, "draft_model:\n  draft_mode: \"model\"\n") ||
		strings.Contains(bare, "draft_num_tokens") || strings.Contains(bare, "ngram_match_min") {
		t.Errorf("zero draft numbers must omit their keys:\n%s", bare)
	}
}
