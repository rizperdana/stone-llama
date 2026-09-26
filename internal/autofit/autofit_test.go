package autofit

import (
	"math"
	"os"
	"strings"
	"testing"
)

// Real weights file size of testdata/smollm3-config.json's model.
const smolWeights = 1_957_008_720 // 1866.15 MiB, 3.5bpw

func realSpec(t *testing.T) Spec {
	t.Helper()
	data, err := os.ReadFile("testdata/smollm3-config.json")
	if err != nil {
		t.Fatal(err)
	}
	spec, err := ParseSpec(data)
	if err != nil {
		t.Fatal(err)
	}
	return spec
}

func smolOpts() Options { return Options{} } // approved defaults

func TestParseRealFixture(t *testing.T) {
	spec := realSpec(t)
	if len(spec.Architectures) != 1 || spec.Architectures[0] != "SmolLM3ForCausalLM" {
		t.Errorf("Architectures = %v", spec.Architectures)
	}
	if spec.Layers != 36 || spec.KVHeads != 4 || spec.HeadDim != 128 || spec.MaxCtx != 65536 {
		t.Errorf("spec = %+v, want 36/4/128/65536", spec)
	}
}

func TestParseDerivesHeadDim(t *testing.T) {
	spec, err := ParseSpec([]byte(`{
		"num_hidden_layers": 4, "num_key_value_heads": 2,
		"hidden_size": 1024, "num_attention_heads": 32,
		"max_position_embeddings": 4096
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if spec.HeadDim != 32 {
		t.Errorf("HeadDim = %d, want 32", spec.HeadDim)
	}
}

func TestParseErrors(t *testing.T) {
	cases := []struct {
		name, body, want string
	}{
		{"indivisible", `{"num_hidden_layers":1,"num_key_value_heads":2,"hidden_size":1000,"num_attention_heads":32,"max_position_embeddings":4096}`, "not divisible"},
		{"missing kv heads", `{"num_hidden_layers":4,"hidden_size":1024,"num_attention_heads":32,"max_position_embeddings":4096}`, "incomplete"},
		{"bad json", `{oops`, "parse config.json"},
	}
	for _, c := range cases {
		if _, err := ParseSpec([]byte(c.body)); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err = %v, want %q", c.name, err, c.want)
		}
	}
}

func TestParseHybridLayerTypesExcludeStateLayers(t *testing.T) {
	spec, err := ParseSpec([]byte(`{
		"num_hidden_layers": 4, "num_key_value_heads": 8,
		"hidden_size": 2048, "num_attention_heads": 32,
		"max_position_embeddings": 131072,
		"layer_types": ["full_attention", "conv", "linear_attention", "full_attention"]
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if spec.Layers != 2 {
		t.Errorf("Layers = %d, want 2 (conv + linear_attention excluded)", spec.Layers)
	}
	if spec.HeadDim != 64 {
		t.Errorf("HeadDim = %d, want 64", spec.HeadDim)
	}
}

func TestParseTextConfigNested(t *testing.T) {
	spec, err := ParseSpec([]byte(`{
		"architectures": ["SomeVLForConditionalGeneration"],
		"num_hidden_layers": 0,
		"text_config": {"num_hidden_layers": 20, "num_key_value_heads": 4,
			"hidden_size": 2048, "num_attention_heads": 16, "max_position_embeddings": 32768}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if spec.Layers != 20 || spec.MaxCtx != 32768 || spec.Architectures[0] != "SomeVLForConditionalGeneration" {
		t.Errorf("spec = %+v", spec)
	}
}

func TestFitRealSmolLM3DefaultQ4At32768(t *testing.T) {
	res, err := Fit(realSpec(t), smolWeights, 4096, smolOpts())
	if err != nil {
		t.Fatal(err)
	}
	// Prefill-evidence rule: 65536/Q4 (3146 load + 1536 headroom = 4682 >
	// 4096) is refused; the ladder halves and lands Q4@32768 with room for
	// a cold prefill workspace — and the still-thin margin is called out.
	if !res.Fits || res.Ctx != 32768 || res.Mode != "Q4" || !res.Reduced || res.Clamped {
		t.Fatalf("result = %+v, want reduced Q4@32768 fit", res)
	}
	if res.WeightsMiB != 1866 || res.KVMiB != 648 || res.BudgetMiB != 2816 || res.HeadroomMiB != 1280 {
		t.Errorf("weights/kv/budget/headroom = %d/%d/%d/%d, want 1866/648/2816/1280",
			res.WeightsMiB, res.KVMiB, res.BudgetMiB, res.HeadroomMiB)
	}
	if !strings.Contains(res.Warning, "drop --ctx") {
		t.Errorf("Warning = %q, want drop --ctx guidance", res.Warning)
	}
	sum := res.Summary()
	if !strings.Contains(sum, "2642") || !strings.Contains(sum, "2816") {
		t.Errorf("Summary = %q, want 2642/2816", sum)
	}
}

func TestFitPrefersFP16WhenItFits(t *testing.T) {
	spec := Spec{Architectures: []string{"X"}, Layers: 36, KVHeads: 4, HeadDim: 128, MaxCtx: 8192}
	res, err := Fit(spec, 100<<20, 4096, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Fits || res.Mode != "FP16" || res.Ctx != 8192 {
		t.Errorf("result = %+v, want FP16@8192", res)
	}
	if res.KVMiB != 576 {
		t.Errorf("KVMiB = %d, want 576", res.KVMiB)
	}
}

func TestFitReducesCtxBeforeQuality(t *testing.T) {
	// 2600 MiB weights: every mode at 65536/32768/16384 busts the
	// headroom-aware budgets → Q4@8192 (2890 ≤ 3008) is the first fit.
	res, err := Fit(realSpec(t), 2600<<20, 4096, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Fits || res.Ctx != 8192 || res.Mode != "Q4" || !res.Reduced {
		t.Errorf("result = %+v, want reduced Q4@8192", res)
	}
}

func TestFitRefusalCarriesFullArithmetic(t *testing.T) {
	res, err := Fit(realSpec(t), 3400<<20, 4096, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Fits {
		t.Fatal("3400 MiB weights must not fit")
	}
	// The prefill workspace lives in the headroom, so even the 4096 floor
	// refuses — and the largest-fitting-ctx answer is honestly "none".
	if res.LargestCtx != 0 {
		t.Errorf("LargestCtx = %d, want 0", res.LargestCtx)
	}
	for _, want := range []string{"weights 3400", "= 3609 MiB > 3040 MiB", "prefill workspace [est]"} {
		if !strings.Contains(res.Reason, want) {
			t.Errorf("Reason missing %q:\n%s", want, res.Reason)
		}
	}
}

func TestFitUserCtxAboveTrainedMaxClamps(t *testing.T) {
	res, err := Fit(realSpec(t), smolWeights, 4096, Options{UserCtx: 999999})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Clamped || res.Ctx > 65536 {
		t.Errorf("Clamped/Ctx = %v/%d, want clamped and ≤ 65536", res.Clamped, res.Ctx)
	}
}
func TestFitUserCtxAlignedTo256(t *testing.T) {
	res, err := Fit(realSpec(t), smolWeights, 4096, Options{UserCtx: 50000})
	if err != nil {
		t.Fatal(err)
	}
	// 50000 → 49920 (256-aligned); at 49920 no mode fits under the prefill
	// headroom, so the ladder halves to the 256-aligned 24832 tier. With
	// the scale overhead counted, Q8 busts 2878 there and Q4 is the pick
	// (the old payload-only arithmetic wrongly landed Q8).
	if res.Ctx != 24832 || res.Mode != "Q4" {
		t.Errorf("Ctx/Mode = %d/%s, want 24832/Q4", res.Ctx, res.Mode)
	}
}

func TestFitForceModeBypassesLadder(t *testing.T) {
	res, err := Fit(realSpec(t), smolWeights, 4096, Options{ForceMode: "Q8", UserCtx: 8192})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Fits || res.Mode != "Q8" || res.Ctx != 8192 {
		t.Errorf("result = %+v, want forced Q8@8192", res)
	}

	// Forced mode that busts the budget still reports the projection.
	res, err = Fit(realSpec(t), smolWeights, 4096, Options{ForceMode: "FP16"})
	if err != nil {
		t.Fatal(err)
	}
	if res.Fits || res.Ctx != 0 || res.LargestCtx != 13568 {
		t.Errorf("result = %+v, want no-fit with LargestCtx 13568", res)
	}
	if !strings.Contains(res.Reason, "largest ctx that would fit: 13568 (FP16)") {
		t.Errorf("Reason = %s", res.Reason)
	}

	if _, err := Fit(realSpec(t), smolWeights, 4096, Options{ForceMode: "banana"}); err == nil {
		t.Error("invalid ForceMode must error")
	}
}

func TestFitTinyModelNeverExceedsTrainedMax(t *testing.T) {
	spec := Spec{Layers: 12, KVHeads: 4, HeadDim: 64, MaxCtx: 2048}
	res, err := Fit(spec, 50<<20, 4096, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Fits || res.Ctx != 2048 || res.Reduced {
		t.Errorf("result = %+v, want fit at trained max 2048", res)
	}
}

func TestBytesPerElement(t *testing.T) {
	cases := []struct {
		in   string
		want float64
	}{
		{"FP16", 2}, {"Q8", 1.0625}, {"Q6", 0.8125}, {"Q4", 0.5625}, {"Q2", 0.3125},
		{"2,4", 0.4375}, {"8,8", 1.0625}, {" q4 ", 0.5625},
	}
	for _, c := range cases {
		got, err := BytesPerElement(c.in)
		if err != nil || got != c.want {
			t.Errorf("BytesPerElement(%q) = %v, %v; want %v", c.in, got, err, c.want)
		}
	}
	for _, bad := range []string{"Q9", "Q1", "banana", "4", "1,8", "4,"} {
		if _, err := BytesPerElement(bad); err == nil {
			t.Errorf("BytesPerElement(%q) must fail", bad)
		}
	}
}

// Guard against arithmetic drift: 36,864 elems/token × 65536 — the
// payload plane (0.5 B, Q4) is exactly 1152 MiB, and the real Q4 cost
// incl. fp16 group scales (0.5625 B) is exactly 1296 MiB.
func TestKVConstantsExact(t *testing.T) {
	spec := realSpec(t)
	elems := int64(spec.Layers) * 2 * int64(spec.KVHeads) * int64(spec.HeadDim)
	if elems != 36864 {
		t.Fatalf("elems/token = %d, want 36864", elems)
	}
	kvQ4 := float64(elems) * 0.5 * 65536
	if kvQ4/(1<<20) != 1152 {
		t.Fatalf("Q4 payload @65536 = %f MiB, want 1152", kvQ4/(1<<20))
	}
	if math.Ceil(kvQ4/(1<<20)) != 1152 {
		t.Fatal("ceil mismatch")
	}
	bpe, err := BytesPerElement("Q4")
	if err != nil {
		t.Fatal(err)
	}
	if got := float64(elems) * bpe * 65536 / (1 << 20); got != 1296 {
		t.Fatalf("Q4 incl. scales @65536 = %f MiB, want 1296", got)
	}
}

// The ladder must reach the asymmetric "4,2" (K4V2) rung: weights that
// bust Q4 at the trained max but fit K4V2 stay at full ctx instead of
// halving the context and dropping to a lower-quality cache.
func TestFitLadderK4V2Rung(t *testing.T) {
	// 1300 MiB @65536: Q4 needs 1296 KV (2724 > 2560 budget) but "4,2"
	// needs 1008 (2436 ≤ 2560) — K4V2 buys the full trained context.
	res, err := Fit(realSpec(t), 1300<<20, 4096, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Fits || res.Ctx != 65536 || res.Mode != "4,2" {
		t.Fatalf("result = %+v, want fit at 65536 on \"4,2\"", res)
	}
	if res.Reduced {
		t.Error("full trained ctx must not be reported as reduced")
	}
}

// Q6 and Q2 are policy-excluded from the ladder (Q6 band is empty — Q4
// is ≈lossless and Q6 @65536 measured OOM; Q2 is manual-only, documented
// 2-bit cliff). Across a weights sweep the ladder may only ever emit the
// four intended rungs.
func TestLadderNeverPicksPolicyExcludedModes(t *testing.T) {
	spec := realSpec(t)
	allowed := map[string]bool{"FP16": true, "Q8": true, "Q4": true, "4,2": true}
	for w := 100; w <= 3600; w += 200 {
		res, err := Fit(spec, int64(w)<<20, 4096, Options{})
		if err != nil {
			t.Fatal(err)
		}
		if res.Fits && !allowed[res.Mode] {
			t.Errorf("weights %d MiB: ladder picked excluded mode %q", w, res.Mode)
		}
	}
}

// exllamav3 conversion writes quantization_config both as a separate file
// and embedded in config.json; ParseSpec captures the embedded block so the
// pre-download gate can confirm EXL3 either way.
func TestParseEmbeddedQuantMethod(t *testing.T) {
	const base = `"architectures":["LlamaForCausalLM"],"num_hidden_layers":4,` +
		`"num_key_value_heads":4,"hidden_size":512,"num_attention_heads":8,` +
		`"max_position_embeddings":2048`
	cases := []struct {
		name, cfg, want string
	}{
		{"top level", `{` + base + `,"quantization_config":{"quant_method":"exl3","bits":4}}`, "exl3"},
		{"absent", `{` + base + `}`, ""},
		{"text_config wrapper", `{"architectures":["GlmForCausalLM"],"text_config":{` + base + `,` +
			`"quantization_config":{"quant_method":"exl3"}}}`, "exl3"},
	}
	for _, c := range cases {
		spec, err := ParseSpec([]byte(c.cfg))
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		if spec.QuantMethod != c.want {
			t.Errorf("%s: QuantMethod = %q, want %q", c.name, spec.QuantMethod, c.want)
		}
	}
}

// Real KV cost includes the fp16 group scales exllamav3 stores per 32
// channels (engine cache/quant.py: storage_size = payload + 2 fp16 scale
// planes): Qn = n/8 + 0.5/8 bytes/element, a pair (k+v+1)/16, FP16
// unchanged (no scales). Under-counting by the scale plane hid ≈144 MiB
// at ctx 65536 on this card.
func TestKVCostIncludesScales(t *testing.T) {
	spec := realSpec(t)
	elems := int64(spec.Layers) * 2 * int64(spec.KVHeads) * int64(spec.HeadDim)
	cases := []struct {
		mode    string
		ctx     int
		wantMiB int64
	}{
		{"Q4", 65536, 1296}, // 0.5625 B/elem: 1152 payload + 144 scales
		{"Q4", 32768, 648},
		{"Q8", 65536, 2448},   // 1.0625 B/elem
		{"4,2", 65536, 1008},  // (4+2+1)/16 B/elem
		{"FP16", 65536, 4608}, // raw fp16: no scales
	}
	for _, c := range cases {
		bpe, err := BytesPerElement(c.mode)
		if err != nil {
			t.Fatalf("%s: %v", c.mode, err)
		}
		kv := math.Ceil(float64(c.ctx) * float64(elems) * bpe / (1 << 20))
		if int64(kv) != c.wantMiB {
			t.Errorf("%s @ %d: KV = %.0f MiB, want %d MiB (bpe=%g)", c.mode, c.ctx, kv, c.wantMiB, bpe)
		}
	}
}

// The load-tuning decision comes from the accepted fit's slack — one
// source for the rendered config and the load payload. Comfortable
// margins keep TabbyAPI's shipped defaults (2048, no warmup); thin
// margins cap the chunk and pre-warm on a throwaway prefill.
func TestFitChunkSizeFromSlack(t *testing.T) {
	spec := realSpec(t) // 65536 trained max, 36,864 elems/token
	cases := []struct {
		name      string
		weights   int // MiB → Q4@65536 slack = 1136 − weights
		wantChunk int
		wantWarm  bool
	}{
		{"comfortable keeps defaults", 400, 2048, false},      // slack 736
		{"thin margin smaller chunk+warmup", 800, 1024, true}, // slack 336
		{"very thin tightest chunk+warmup", 936, 512, true},   // slack 200
	}
	for _, c := range cases {
		res, err := Fit(spec, int64(c.weights)<<20, 4096, Options{})
		if err != nil {
			t.Fatal(err)
		}
		if !res.Fits || res.Mode != "Q4" {
			t.Fatalf("%s: result = %+v, want accepted Q4@65536", c.name, res)
		}
		if res.ChunkSize != c.wantChunk || res.Warmup != c.wantWarm {
			t.Errorf("%s: chunk/warmup = %d/%v, want %d/%v",
				c.name, res.ChunkSize, res.Warmup, c.wantChunk, c.wantWarm)
		}
	}
}

// An explicit config override wins over the auto-derived chunk; warmup
// stays margin-driven.
func TestFitChunkSizeOverride(t *testing.T) {
	res, err := Fit(realSpec(t), 400<<20, 4096, Options{ChunkSizeOverride: 4096})
	if err != nil {
		t.Fatal(err)
	}
	if res.ChunkSize != 4096 || res.Warmup {
		t.Errorf("chunk/warmup = %d/%v, want 4096/false", res.ChunkSize, res.Warmup)
	}
	// Refusals carry no load decision.
	big, err := Fit(realSpec(t), 3400<<20, 4096, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if big.Fits || big.ChunkSize != 0 || big.Warmup {
		t.Errorf("refused fit = %+v, want no chunk/warmup decision", big)
	}
}

// MoE detection is generic config probing — arch names lie (GLM-5.3-
// Flash is Glm5NextForConditionalGeneration with 288 routed experts).
func TestParseMoEFields(t *testing.T) {
	dense := `{"architectures":["SmolLM3ForCausalLM"],"num_hidden_layers":36,
		"hidden_size":2048,"num_attention_heads":16,"num_key_value_heads":4,
		"max_position_embeddings":65536}`
	moeTop := `{"architectures":["FooForCausalLM"],"num_hidden_layers":48,
		"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":8,
		"max_position_embeddings":65536,"n_routed_experts":288,
		"moe_intermediate_size":2048}`
	moeNested := `{"architectures":["Glm5NextForConditionalGeneration"],
		"text_config":{"num_hidden_layers":48,"hidden_size":4096,
		"num_attention_heads":32,"num_key_value_heads":8,
		"max_position_embeddings":65536,"num_local_experts":128,
		"moe_intermediate_size":2048}}`
	for _, tc := range []struct {
		name string
		body string
		want bool
	}{
		{"dense", dense, false},
		{"top-level routed experts", moeTop, true},
		{"text_config local experts", moeNested, true},
	} {
		spec, err := ParseSpec([]byte(tc.body))
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if spec.MoE != tc.want {
			t.Errorf("%s: MoE = %v, want %v", tc.name, spec.MoE, tc.want)
		}
	}
}
