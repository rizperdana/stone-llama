package serve

import (
	"strings"
	"testing"

	"github.com/rizperdana/stone-llama/internal/autofit"
)

// One autofit.Result must drive BOTH surfaces: the rendered tabby YAML
// and the /-/load payload keys. They can never disagree because neither
// computes anything — both copy the same Result fields.

func TestLoadArgsAndYAMLCarrySameChunkDecision(t *testing.T) {
	thin := autofit.Result{ChunkSize: 1024, Warmup: true}

	args := fitLoadArgs(&thin)
	if args["chunk_size"] != 1024 || args["warmup"] != true {
		t.Errorf("load payload = %#v, want chunk_size 1024 + warmup true", args)
	}

	yaml := RenderTabbyYAML(TabbyConfig{
		Host: "h", Port: 1, ModelDir: "d", ModelName: "m",
		ChunkSize: thin.ChunkSize, Warmup: thin.Warmup,
	})
	for _, want := range []string{"  chunk_size: 1024\n", "  warmup: true\n"} {
		if !strings.Contains(yaml, want) {
			t.Errorf("YAML missing %q:\n%s", want, yaml)
		}
	}
}

func TestLoadArgsAndYAMLComfortableKeepsDefaults(t *testing.T) {
	// A comfortable verdict keeps TabbyAPI's shipped defaults: chunk_size
	// 2048 (config_sample.yml), no warmup (default false) — explicit in
	// the payload, absent in the YAML (omit == backend default).
	comfort := autofit.Result{ChunkSize: 2048, Warmup: false}

	args := fitLoadArgs(&comfort)
	if args["chunk_size"] != 2048 {
		t.Errorf("load payload = %#v, want chunk_size 2048", args)
	}
	if _, ok := args["warmup"]; ok {
		t.Errorf("load payload must not force warmup on a comfortable fit: %#v", args)
	}

	yaml := RenderTabbyYAML(TabbyConfig{
		Host: "h", Port: 1, ModelDir: "d", ModelName: "m",
		ChunkSize: comfort.ChunkSize, Warmup: comfort.Warmup,
	})
	if !strings.Contains(yaml, "  chunk_size: 2048\n") {
		t.Errorf("YAML missing chunk_size: 2048:\n%s", yaml)
	}
	if strings.Contains(yaml, "warmup") {
		t.Errorf("YAML must not carry warmup on a comfortable fit:\n%s", yaml)
	}
}

// The boot render passes no verdict (model-less child config): it must
// stay byte-identical to the pre-wiring template — no new keys for
// TabbyAPI to reject.
func TestRenderTabbyYAMLBootUnchangedWithoutVerdict(t *testing.T) {
	got := RenderTabbyYAML(TabbyConfig{Host: "h", Port: 1, ModelDir: "d", ModelName: "m", Ctx: 4096, CacheMode: "Q4"})
	if strings.Contains(got, "chunk_size") || strings.Contains(got, "warmup") {
		t.Errorf("boot YAML gained keys without a verdict:\n%s", got)
	}
}
