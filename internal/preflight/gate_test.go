package preflight

import (
	"strings"
	"testing"

	"github.com/rizperdana/stone-llama/internal/autofit"
)

func smolSpec() autofit.Spec {
	return autofit.Spec{
		Architectures: []string{"SmolLM3ForCausalLM"},
		Layers:        36, KVHeads: 4, HeadDim: 128, MaxCtx: 65536,
	}
}

func baseInput() Input {
	return Input{
		Architectures: []string{"SmolLM3ForCausalLM"},
		QuantMethod:   "exl3",
		HasQuantCfg:   true,
		RepoFiles:     []string{"config.json", "quantization_config.json", "model.safetensors"},
		Spec:          smolSpec(),
		WeightsBytes:  1_957_008_720,
		VRAMMiB:       4096,
		GPUName:       "RTX 3050",
	}
}

func TestEvaluateCleanExl3Fits(t *testing.T) {
	r := Evaluate(baseInput())
	if r.Refused || r.Warned {
		t.Fatalf("checks = %+v", r.Checks)
	}
	if r.Verdict.Status != "fit" {
		t.Errorf("verdict = %q", r.Verdict.Status)
	}
	if r.Verdict.MaxCtx != 65536 || r.Verdict.CacheMode != "Q4" {
		t.Errorf("verdict ctx/mode = %d/%s", r.Verdict.MaxCtx, r.Verdict.CacheMode)
	}
	if r.Verdict.Note != "" {
		t.Errorf("note = %q", r.Verdict.Note)
	}
	lines := r.Format("RTX 3050")
	joined := strings.Join(lines, "\n")
	for _, want := range []string{"gate: arch", "✓", "quant ✓ exl3", "3146 MiB (budget 3584) ✓"} {
		if !strings.Contains(joined, want) {
			t.Errorf("Format missing %q:\n%s", want, joined)
		}
	}
}

func TestEvaluateUnknownArchWarnsNeverRefuses(t *testing.T) {
	in := baseInput()
	in.Architectures = []string{"TotallyUnknownForCausalLM"}
	r := Evaluate(in)
	if r.Refused || !r.Warned {
		t.Fatalf("Refused/Warned = %v/%v", r.Refused, r.Warned)
	}
	if r.Verdict.Status != "warn" {
		t.Errorf("verdict = %q, want warn", r.Verdict.Status)
	}
	if !strings.Contains(r.Verdict.Note, "TotallyUnknownForCausalLM") {
		t.Errorf("note = %q", r.Verdict.Note)
	}
	arch := r.Checks[0]
	if arch.Status != StatusWarn || !strings.Contains(arch.Detail, "not refusing") {
		t.Errorf("arch check = %+v", arch)
	}
}

func TestEvaluateNoArchitecturesWarns(t *testing.T) {
	in := baseInput()
	in.Architectures = nil
	r := Evaluate(in)
	if r.Warned && !r.Refused {
		return
	}
	t.Errorf("want warn-only, got %+v", r)
}

func TestEvaluateExl2Refuses(t *testing.T) {
	in := baseInput()
	in.QuantMethod = "exl2"
	r := Evaluate(in)
	if !r.Refused {
		t.Fatal("exl2 must refuse")
	}
	if r.Verdict.Status != "refuse" {
		t.Errorf("verdict = %q", r.Verdict.Status)
	}
	q := r.Checks[1]
	if q.Status != StatusRefuse || !strings.Contains(q.Detail, "ExLlamaV2") {
		t.Errorf("quant check = %+v", q)
	}
}

func TestEvaluateOtherQuantMethodsRefuse(t *testing.T) {
	in := baseInput()
	in.QuantMethod = "gptq"
	r := Evaluate(in)
	if !r.Refused || !strings.Contains(r.Checks[1].Detail, "not exl3") {
		t.Errorf("checks = %+v", r.Checks)
	}
}

func TestEvaluateGGUFOnlyRefusesWithOllamaHint(t *testing.T) {
	in := baseInput()
	in.QuantMethod = ""
	in.HasQuantCfg = false
	in.RepoFiles = []string{"config.json", "model.gguf"}
	in.WeightsBytes = 0
	r := Evaluate(in)
	if !r.Refused {
		t.Fatal("gguf-only must refuse")
	}
	if !strings.Contains(r.Checks[1].Detail, "ollama with GGUF") {
		t.Errorf("quant detail = %q", r.Checks[1].Detail)
	}
}

func TestEvaluateMissingQuantConfigWarns(t *testing.T) {
	cases := map[string]struct {
		hasCfg  bool
		wantSub string
	}{
		"absent":        {false, "no quantization_config.json"},
		"missing field": {true, "no quant_method"},
	}
	for name, tc := range cases {
		in := baseInput()
		in.QuantMethod = ""
		in.HasQuantCfg = tc.hasCfg
		r := Evaluate(in)
		if r.Refused {
			t.Errorf("%s: must warn, not refuse", name)
			continue
		}
		if !strings.Contains(r.Checks[1].Detail, tc.wantSub) {
			t.Errorf("%s: detail = %q", name, r.Checks[1].Detail)
		}
	}
}

func TestEvaluateNoWeightsRefuses(t *testing.T) {
	in := baseInput()
	in.QuantMethod = ""
	in.HasQuantCfg = false
	in.RepoFiles = []string{"config.json"}
	in.WeightsBytes = 0
	r := Evaluate(in)
	if !r.Refused || !strings.Contains(r.Checks[1].Detail, ".safetensors") {
		t.Errorf("checks = %+v", r.Checks)
	}
}

func TestEvaluateReducedFitWarns(t *testing.T) {
	in := baseInput()
	in.WeightsBytes = 2600 << 20
	r := Evaluate(in)
	if r.Refused || !r.Warned {
		t.Fatalf("Refused/Warned = %v/%v", r.Refused, r.Warned)
	}
	if r.Verdict.MaxCtx != 32768 || r.Verdict.CacheMode != "Q4" {
		t.Errorf("verdict = %d %s", r.Verdict.MaxCtx, r.Verdict.CacheMode)
	}
	if !strings.Contains(r.Verdict.Note, "reduced ctx: Q4 @ 32768") {
		t.Errorf("note = %q", r.Verdict.Note)
	}
}

func TestEvaluateNoFitRefusesWithArithmetic(t *testing.T) {
	in := baseInput()
	in.WeightsBytes = 3400 << 20
	r := Evaluate(in)
	if !r.Refused {
		t.Fatal("must refuse")
	}
	if !strings.Contains(r.Verdict.Note, "3600 MiB > 3584 MiB") {
		t.Errorf("note = %q", r.Verdict.Note)
	}
	if !strings.Contains(r.Verdict.Note, "largest ctx that would fit: 3072") {
		t.Errorf("note misses largest ctx: %q", r.Verdict.Note)
	}
}

func TestEvaluateFormatContinuationLines(t *testing.T) {
	in := baseInput()
	in.WeightsBytes = 3400 << 20
	r := Evaluate(in)
	lines := r.Format("")
	joined := strings.Join(lines, "\n")
	if !strings.Contains(joined, "gate: fit") || !strings.Contains(joined, "      largest ctx") {
		t.Errorf("Format = \n%s", joined)
	}
}
