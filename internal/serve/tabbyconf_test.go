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
