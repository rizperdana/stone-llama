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
	if res.WeightsMiB != 1866 || res.KVMiB != 576 || res.BudgetMiB != 2816 || res.HeadroomMiB != 1280 {
		t.Errorf("weights/kv/budget/headroom = %d/%d/%d/%d, want 1866/576/2816/1280",
			res.WeightsMiB, res.KVMiB, res.BudgetMiB, res.HeadroomMiB)
	}
	if !strings.Contains(res.Warning, "drop --ctx") {
		t.Errorf("Warning = %q, want drop --ctx guidance", res.Warning)
	}
	sum := res.Summary()
	if !strings.Contains(sum, "2570") || !strings.Contains(sum, "2816") {
		t.Errorf("Summary = %q, want 2570/2816", sum)
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
	// headroom-aware budgets → Q4@8192 (2872 ≤ 3008) is the first fit.
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
	for _, want := range []string{"weights 3400", "= 3600 MiB > 3040 MiB", "prefill workspace [est]"} {
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
	// headroom, so the ladder halves to the 256-aligned 24832 tier.
	if res.Ctx != 24832 || res.Mode != "Q8" {
		t.Errorf("Ctx/Mode = %d/%s, want 24832/Q8", res.Ctx, res.Mode)
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
		{"FP16", 2}, {"Q8", 1}, {"Q6", 0.75}, {"Q4", 0.5}, {"Q2", 0.25},
		{"2,4", 0.375}, {"8,8", 1}, {" q4 ", 0.5},
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
}

// Guard against arithmetic drift: 36,864 elems/token × 0.5 B × 65536.
func TestKVConstantsExact(t *testing.T) {
	spec := realSpec(t)
	elems := int64(spec.Layers) * 2 * int64(spec.KVHeads) * int64(spec.HeadDim)
	if elems != 36864 {
		t.Fatalf("elems/token = %d, want 36864", elems)
	}
	kvQ4 := float64(elems) * 0.5 * 65536
	if kvQ4/(1<<20) != 1152 {
		t.Fatalf("Q4 KV @65536 = %f MiB, want 1152", kvQ4/(1<<20))
	}
	if math.Ceil(kvQ4/(1<<20)) != 1152 {
		t.Fatal("ceil mismatch")
	}
}
