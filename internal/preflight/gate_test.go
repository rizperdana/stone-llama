package preflight

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"

	"github.com/rizperdana/stone-llama/internal/autofit"
	"github.com/rizperdana/stone-llama/internal/store"
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
		FetchHeader:   headerFetcher(exl3StorageTensors()),
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
	for _, want := range []string{"gate: arch", "✓", "quant ✓ exl3", "1824 MiB (headroom 1536, budget 2560) ✓"} {
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

// An unconfirmed quant format must refuse: `fit` exits 3 and `pull` stops
// before consent, so --yes can never turn "unknown format" into a download
// (README:197-198 promises exl2 → refused; an EXL2-shaped repo without the
// measurement.json marker used to slip through as a warn).
func TestEvaluateUnconfirmedQuantRefuses(t *testing.T) {
	cases := map[string]struct {
		hasCfg  bool
		wantSub string
	}{
		"absent":        {false, "no EXL3 tensor group"},
		"missing field": {true, "no EXL3 tensor group"},
	}
	for name, tc := range cases {
		in := baseInput()
		in.QuantMethod = ""
		in.HasQuantCfg = tc.hasCfg
		// Header reads cleanly and carries no EXL3 group → verified
		// "not EXL3 storage", refused before any weight byte moves.
		in.FetchHeader = headerFetcher(exl2Tensors())
		r := Evaluate(in)
		if !r.Refused {
			t.Errorf("%s: unconfirmed quant must refuse, got %+v", name, r.Checks)
			continue
		}
		detail := r.Checks[1].Detail
		for _, sub := range []string{tc.wantSub, "not EXL3 storage", "look for '-exl3'"} {
			if !strings.Contains(detail, sub) {
				t.Errorf("%s: detail missing %q: %q", name, sub, detail)
			}
		}
		if !strings.Contains(r.RefusalSummary(), "quant format") {
			t.Errorf("%s: RefusalSummary = %q", name, r.RefusalSummary())
		}
	}
}

// Live slip: LoneStriker/TinyLlama-1.1B-Chat-v0.3-3.0bpw-h6-exl2 ships
// config.json + output.safetensors and NEITHER marker (no measurement.json,
// no quantization_config.json). It must not pass as fitting.
func TestEvaluateExl2ShapedWithoutMarkerRefuses(t *testing.T) {
	in := baseInput()
	in.QuantMethod = ""
	in.HasQuantCfg = false
	in.FetchHeader = headerFetcher(exl2Tensors())
	in.RepoFiles = []string{"config.json", "output.safetensors", "tokenizer.json", "README.md"}
	in.WeightsBytes = 550_831_360
	r := Evaluate(in)
	if !r.Refused {
		t.Fatalf("EXL2-shaped repo must refuse, checks = %+v", r.Checks)
	}
	q := r.Checks[1]
	if q.Status != StatusRefuse || !strings.Contains(q.Detail, "not EXL3 storage") {
		t.Errorf("quant check = %+v", q)
	}
	if r.Verdict.Status != store.VerdictRefuse {
		t.Errorf("verdict = %+v, want refuse", r.Verdict)
	}
}

// Confirmation may also come from quantization_config embedded in
// config.json (exllamav3 conversion writes both) — a repo with only the
// embedded block is a confirmed EXL3, not an unconfirmed one.
func TestEvaluateExl3EmbeddedConfigConfirms(t *testing.T) {
	in := baseInput()
	in.QuantMethod = ""
	in.HasQuantCfg = false
	in.RepoFiles = []string{"config.json", "model.safetensors"}
	in.Spec.QuantMethod = "exl3"
	r := Evaluate(in)
	if r.Refused {
		t.Fatalf("embedded exl3 must not be refused, checks = %+v", r.Checks)
	}
	if r.Checks[1].Status != StatusOK || r.Checks[1].Detail != "exl3" {
		t.Errorf("quant check = %+v", r.Checks[1])
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
	if !strings.Contains(r.Verdict.Note, "3609 MiB > 3040 MiB") {
		t.Errorf("note = %q", r.Verdict.Note)
	}
	if !strings.Contains(r.Verdict.Note, "largest ctx that would fit: none") {
		t.Errorf("note misses largest ctx: %q", r.Verdict.Note)
	}
	if !strings.Contains(r.Verdict.Note, "reduce --ctx or pick a smaller quant") {
		t.Errorf("refusal misses next action: %q", r.Verdict.Note)
	}
}

// The top-level summary must name the FIRST failing gate: a quant-format
// refusal is never reported as a VRAM problem (dogfood bug: fit printed
// "no context fits this model in VRAM" for an EXL2 repo).
func TestRefusalSummaryNamesFirstFailingGate(t *testing.T) {
	quant := Evaluate(Input{
		Architectures: []string{"LlamaForCausalLM"},
		RepoFiles:     []string{"config.json", "model.safetensors", "measurement.json"},
		Repo:          "org/exl2-model",
		Spec:          smolSpec(),
		WeightsBytes:  4_000_000_000,
		VRAMMiB:       4096,
	})
	if !quant.Refused {
		t.Fatal("measurement.json fixture must refuse")
	}
	vram := Evaluate(func() Input {
		in := baseInput()
		in.WeightsBytes = 3400 << 20 // fits nowhere on 4096 MiB
		return in
	}())
	if !vram.Refused {
		t.Fatal("oversized fixture must refuse")
	}
	qs, vs := quant.RefusalSummary(), vram.RefusalSummary()
	if qs == vs {
		t.Fatalf("summaries must differ:\nquant: %s\nvram:  %s", qs, vs)
	}
	if !strings.Contains(qs, "quant format") || !strings.Contains(qs, "EXL2 quant") {
		t.Errorf("quant summary = %q, want the EXL2 cause", qs)
	}
	if strings.Contains(qs, "VRAM") {
		t.Errorf("quant summary blames VRAM: %q", qs)
	}
	if vs != "no context fits this model in VRAM" {
		t.Errorf("vram summary = %q", vs)
	}
	// Both gates failing → lead with the more fundamental one (quant).
	both := Evaluate(func() Input {
		in := baseInput()
		in.QuantMethod = "exl2"
		in.WeightsBytes = 3400 << 20
		return in
	}())
	if !both.Refused || !strings.Contains(both.RefusalSummary(), "quant format") {
		t.Errorf("both-fail summary = %q, want quant first", both.RefusalSummary())
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

// Terra3312/GLM-5.3-Flash-EXL3-4bpw-MUL1 shape: arch name carries no
// "moe" substring, expert fields are present. Warn-only — the gate
// cannot see tensor names, so it must not pretend to verdict on .mul1.
func TestEvaluateMoEWarnsAboutUnverifiableMul1(t *testing.T) {
	in := baseInput()
	spec := smolSpec()
	spec.Architectures = []string{"Glm5NextForConditionalGeneration"}
	spec.MoE = true
	in.Spec = spec
	in.Architectures = spec.Architectures
	r := Evaluate(in)
	if r.Refused {
		t.Fatal("MoE must stay warn-only — metadata cannot confirm or deny .mul1")
	}
	if !r.Warned {
		t.Fatal("MoE config must set Warned")
	}
	var moe *Check
	for i := range r.Checks {
		if r.Checks[i].Name == "moe" {
			moe = &r.Checks[i]
		}
	}
	if moe == nil {
		t.Fatalf("no moe check in %+v", r.Checks)
	}
	if moe.Status != StatusWarn {
		t.Errorf("moe status = %s, want warn", moe.Status)
	}
	for _, want := range []string{".mul1", "tensor names", "n_routed_experts", "inspect"} {
		if !strings.Contains(moe.Detail, want) {
			t.Errorf("moe detail missing %q:\n%s", want, moe.Detail)
		}
	}
}

func TestEvaluateDenseModelHasNoMoECheck(t *testing.T) {
	r := Evaluate(baseInput())
	for _, c := range r.Checks {
		if c.Name == "moe" {
			t.Errorf("dense model must not gain a moe check: %+v", c)
		}
	}
}

// ---------- safetensors header probe (engine parity) ----------

// headerFetcher serves one crafted safetensors header exactly as the
// wire carries it: 8-byte little-endian header_size + JSON.
func headerFetcher(tensors map[string]string) func(string) (io.ReadCloser, error) {
	return func(string) (io.ReadCloser, error) {
		type entry struct {
			Dtype string `json:"dtype"`
		}
		obj := make(map[string]entry, len(tensors))
		for k, v := range tensors {
			obj[k] = entry{v}
		}
		j, err := json.Marshal(obj)
		if err != nil {
			panic(err)
		}
		buf := make([]byte, 8+len(j))
		binary.LittleEndian.PutUint64(buf, uint64(len(j)))
		copy(buf[8:], j)
		return io.NopCloser(bytes.NewReader(buf)), nil
	}
}

// Live shapes captured from HF headers (evidence/*-hdr.json).
func exl3StorageTensors() map[string]string { // async0x42: trellis I16, suh/svh F16
	return map[string]string{
		"model.layers.0.self_attn.q_proj.trellis": "I16",
		"model.layers.0.self_attn.q_proj.suh":     "F16",
		"model.layers.0.self_attn.q_proj.svh":     "F16",
		"model.layers.0.self_attn.q_proj.weight":  "F16",
	}
}

func u16Tensors() map[string]string { // 0xzknw/LFM2.5-2.6B: trellis U16 (engine ValueError)
	return map[string]string{
		"model.layers.0.mlp.experts.trellis": "U16",
		"model.layers.0.mlp.experts.mcg":     "I32",
		"model.layers.0.mlp.experts.suh":     "F16",
		"model.layers.0.mlp.experts.svh":     "F16",
	}
}

func exl2Tensors() map[string]string { // LoneStriker EXL2: q_* only, no group
	return map[string]string{
		"model.layers.0.self_attn.q_proj.q_weight": "I32",
		"model.layers.0.self_attn.q_proj.q_groups": "I16",
		"model.layers.0.self_attn.q_proj.weight":   "F16",
	}
}

// Sidecar-less (embedded-only or fully bare) EXL3 must be ACCEPTED —
// quantization_config.json is optional for exllamav3, the tensor
// suffix group is what the engine itself tests.
func TestEvaluateEmbeddedOnlyExl3AcceptedViaHeader(t *testing.T) {
	for _, tc := range []struct {
		name    string
		embed   bool
		quantOK bool
	}{
		{"embedded block only, no sidecar", true, true},
		{"bare repo, no sidecar no embedded", false, true},
	} {
		in := baseInput()
		in.QuantMethod, in.HasQuantCfg = "", false
		in.Spec.QuantMethod = ""
		if tc.embed {
			in.Spec.QuantMethod = "exl3"
		}
		in.FetchHeader = headerFetcher(exl3StorageTensors())
		in.WeightsBytes = 400 << 20 // clean fit: Warned can only come from quant
		r := Evaluate(in)
		if r.Refused || r.Warned {
			t.Errorf("%s: checks = %+v, want clean accept", tc.name, r.Checks)
			continue
		}
		if r.Checks[1].Status != StatusOK || r.Checks[1].Detail != "exl3" {
			t.Errorf("%s: quant check = %+v", tc.name, r.Checks[1])
		}
	}
}

// The U16 trap (0xzknw/LFM2.5-2.6B-EXL3-4bpw): EXL3 storage group is
// present, but the engine's convert_dtype has no U16 — load_exl3 would
// raise ValueError AFTER gigabytes of download. Refuse before a byte
// moves, naming the dtype and the engine limitation.
func TestEvaluateU16QuantDtypeRefused(t *testing.T) {
	in := baseInput() // sidecar claims exl3 — corroboration must not save it
	in.FetchHeader = headerFetcher(u16Tensors())
	r := Evaluate(in)
	if !r.Refused {
		t.Fatalf("U16 quant tensors must refuse, checks = %+v", r.Checks)
	}
	q := r.Checks[1]
	for _, want := range []string{"U16", "exllamav3", "cannot load", ".trellis"} {
		if !strings.Contains(q.Detail, want) {
			t.Errorf("quant detail missing %q:\n%s", want, q.Detail)
		}
	}
}

// Header readable, no EXL3 tensor group → verified NOT EXL3 storage →
// refuse, regardless of whether a sidecar claims exl3.
func TestEvaluateNoExl3StorageGroupRefused(t *testing.T) {
	for _, tc := range []struct {
		name    string
		sidecar bool
	}{
		{"no sidecar", false},
		{"sidecar claims exl3", true},
	} {
		in := baseInput()
		in.QuantMethod, in.HasQuantCfg = "", false
		in.Spec.QuantMethod = ""
		if tc.sidecar {
			in.QuantMethod, in.HasQuantCfg = "exl3", true
		}
		in.FetchHeader = headerFetcher(exl2Tensors())
		r := Evaluate(in)
		if !r.Refused {
			t.Errorf("%s: must refuse, checks = %+v", tc.name, r.Checks)
			continue
		}
		d := r.Checks[1].Detail
		for _, want := range []string{".trellis", "not EXL3 storage", "look for '-exl3'"} {
			if !strings.Contains(d, want) {
				t.Errorf("%s: detail missing %q: %s", tc.name, want, d)
			}
		}
	}
}

// Could-not-look ≠ must-refuse: an unreadable header degrades to a
// WARN with the reason, never a crash and never a silent pass.
func TestEvaluateUnreadableHeaderWarnsUnverified(t *testing.T) {
	in := baseInput()
	in.FetchHeader = func(string) (io.ReadCloser, error) {
		return nil, errors.New("broken pipe")
	}
	r := Evaluate(in)
	if r.Refused {
		t.Fatalf("unreadable header must not refuse, checks = %+v", r.Checks)
	}
	if !r.Warned || r.Verdict.Status != store.VerdictWarn {
		t.Fatalf("verdict = %+v, want warn", r.Verdict)
	}
	d := r.Checks[1].Detail
	if !strings.Contains(d, "UNVERIFIED") || !strings.Contains(d, "broken pipe") {
		t.Errorf("quant detail = %q, want reason + unverified", d)
	}
}

func TestEvaluateNilHeaderFetcherWarnsUnverified(t *testing.T) {
	in := baseInput()
	in.FetchHeader = nil
	r := Evaluate(in)
	if r.Refused || !r.Warned {
		t.Fatalf("nil fetcher: Refused/Warned = %v/%v, checks = %+v", r.Refused, r.Warned, r.Checks)
	}
	if !strings.Contains(r.Checks[1].Detail, "UNVERIFIED") {
		t.Errorf("quant detail = %q", r.Checks[1].Detail)
	}
}

// Fit must stay responsive: a many-shard repo probes at most
// maxHeaderFiles headers, one bounded request each.
func TestEvaluateProbeCappedAtFourFiles(t *testing.T) {
	in := baseInput()
	var files []string
	for i := range 10 {
		files = append(files, fmt.Sprintf("shard-%05d-of-00010.safetensors", i))
	}
	in.RepoFiles = files
	calls := 0
	in.FetchHeader = func(string) (io.ReadCloser, error) {
		calls++
		return headerFetcher(exl3StorageTensors())("")
	}
	r := Evaluate(in)
	if r.Refused {
		t.Errorf("checks = %+v", r.Checks)
	}
	if calls > maxHeaderFiles {
		t.Errorf("probed %d headers, cap is %d", calls, maxHeaderFiles)
	}
}

// Inert buffers in exotic dtypes are out of scope — the engine
// structurally validates but never size-checks dtypes it does not read
// (loader/safetensors.py:52-54). Only quant-group tensors gate.
func TestEvaluateInertBufferDtypeIgnored(t *testing.T) {
	in := baseInput()
	tensors := exl3StorageTensors()
	tensors["model.layers.0.self_attn.inv_freq"] = "F64" // never loaded
	in.WeightsBytes = 400 << 20                          // clean fit: Warned can only come from quant
	in.FetchHeader = headerFetcher(tensors)
	r := Evaluate(in)
	if r.Refused || r.Warned {
		t.Errorf("inert buffer dtype must not affect the verdict: %+v", r.Checks)
	}
}
