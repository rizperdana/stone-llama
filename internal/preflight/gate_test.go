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
		Repo:          "org/model",
		Spec:          smolSpec(),
		WeightsBytes:  1_957_008_720,
		VRAMMiB:       4096,
		GPUName:       "RTX 3050",
	}
}

func TestEvaluateCleanExl3Fits(t *testing.T) {
	in := baseInput()
	// Small weights: the prefill-aware headroom still leaves a fat margin
	// at the trained max, so this stays a clean, note-less fit.
	in.WeightsBytes = 400 << 20
	r := Evaluate(in)
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
	for _, want := range []string{"gate: arch", "✓", "quant ✓ exl3", "1680 MiB (headroom 1536, budget 2560) ✓"} {
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
	if arch.Status != StatusWarn || !strings.Contains(arch.Detail, "unknown to the embedded support table") {
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

// Real ExLlamaV2 repos (Royallab-style) ship measurement.json and no
// quantization_config.json — the gate must refuse them as EXL2, never
// warn "may be unquantized FP16".
func TestEvaluateExl2MeasurementJSONRefuses(t *testing.T) {
	for _, files := range [][]string{
		{"config.json", "tokenizer.json", "model-00001-of-00002.safetensors",
			"model-00002-of-00002.safetensors", "measurement.json"},
		{"config.json", "tokenizer.json", "exl2-3.0bpw/model.safetensors",
			"exl2-3.0bpw/measurement.json"}, // scoped layout
	} {
		in := baseInput()
		in.QuantMethod = ""
		in.HasQuantCfg = false
		in.RepoFiles = files
		in.WeightsBytes = 4_000_000_000
		r := Evaluate(in)
		if !r.Refused {
			t.Errorf("%v: measurement.json repo must refuse, got %+v", files, r)
			continue
		}
		q := r.Checks[1]
		if q.Status != StatusRefuse || !strings.Contains(q.Detail, "ExLlamaV2") ||
			!strings.Contains(q.Detail, "measurement.json") {
			t.Errorf("quant check = %+v", q)
		}
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

func TestEvaluateMissingQuantConfigWarnsNotRefuses(t *testing.T) {
	cases := map[string]struct {
		hasCfg  bool
		wantSub string
	}{
		"absent":        {false, "no quantization_config.json"},
		"missing field": {true, "has no quant_method"},
	}
	for name, tc := range cases {
		in := baseInput()
		in.QuantMethod = ""
		in.HasQuantCfg = tc.hasCfg
		r := Evaluate(in)
		if r.Refused || !r.Warned {
			t.Errorf("%s: missing quant config must warn, not refuse (FP16 ladder stays open); refused=%v", name, r.Refused)
			continue
		}
		detail := r.Checks[1].Detail
		if !strings.Contains(detail, tc.wantSub) {
			t.Errorf("%s: detail = %q", name, detail)
		}
		if !strings.Contains(detail, "look for '-exl3'") {
			t.Errorf("%s: missing exl3 hint: %q", name, detail)
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
	want := "no model weights found in org/model (only 1 non-weight files) — pick a repo that publishes .safetensors"
	if !r.Refused || !strings.Contains(r.Checks[1].Detail, want) {
		t.Errorf("checks = %+v, want %q", r.Checks, want)
	}
}

func TestEvaluateReducedFitWarns(t *testing.T) {
	in := baseInput()
	in.WeightsBytes = 2600 << 20
	r := Evaluate(in)
	if r.Refused || !r.Warned {
		t.Fatalf("Refused/Warned = %v/%v", r.Refused, r.Warned)
	}
	if r.Verdict.MaxCtx != 8192 || r.Verdict.CacheMode != "Q4" {
		t.Errorf("verdict = %d %s", r.Verdict.MaxCtx, r.Verdict.CacheMode)
	}
	if !strings.Contains(r.Verdict.Note, "reduced ctx: Q4 @ 8192") {
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
	if !strings.Contains(r.Verdict.Note, "3600 MiB > 3040 MiB") {
		t.Errorf("note = %q", r.Verdict.Note)
	}
	if !strings.Contains(r.Verdict.Note, "largest ctx that would fit: none") {
		t.Errorf("note misses largest ctx: %q", r.Verdict.Note)
	}
	if !strings.Contains(r.Verdict.Note, "reduce --ctx or pick a smaller quant") {
		t.Errorf("refusal misses next action: %q", r.Verdict.Note)
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

func TestOOMAdviceReduceCtxAndCacheModeLines(t *testing.T) {
	res := autofit.Result{
		Ctx:         65536,
		Mode:        "FP16",
		Fits:        true,
		Warning:     "only 100 MiB margin above the headroom",
		ElemsPerTok: 36864,
	}
	got := OOMAdvice(res)
	for _, sub := range []string{
		"CUDA out of memory before the first token (prefill workspace)",
		"reduce --ctx",
		"--cache-mode Q4",
		"frees ~",
	} {
		if !strings.Contains(got, sub) {
			t.Errorf("OOMAdvice missing %q:\n%s", sub, got)
		}
	}
}

func TestOOMAdviceLargestCtxLine(t *testing.T) {
	res := autofit.Result{Fits: false, LargestCtx: 32768, ElemsPerTok: 36864}
	got := OOMAdvice(res)
	for _, sub := range []string{
		"largest ctx that would fit now: 32768",
		"retry with --ctx 32768",
	} {
		if !strings.Contains(got, sub) {
			t.Errorf("OOMAdvice missing %q:\n%s", sub, got)
		}
	}
}
