// Package autofit derives a VRAM-fitting context/cache configuration from
// a model's config.json plus the GPU's total VRAM. This is the only place
// the KV-cache arithmetic exists (ARCHITECTURE.md §5):
//
//	elems/token   = layers × 2 × kv_heads × head_dim
//	load          = weights + ctx × elems/token × bytes/element + overhead
//	headroom(ctx) = base + prefill workspace [est] + ctx margin [est]
//	fits          ⟺ load ≤ vram_total − headroom(ctx)
//
// The two [est] terms come from live prefill evidence (2026-09-26, measured
// in attach mode against a running TabbyAPI on a used 4 GB card): CUDA OOM
// happens during PREFILL, in a cold hgemm/cublas workspace allocated before
// the first token is produced (max_tokens=1 does not avoid it), and a
// ~2.5k-token agent prompt OOMs on a used card under a config that reads as
// "fit" at idle — sustained to ~1,282 prompt tokens, intermittent around
// ~1,780, consistent OOM at ~2,500+, while a 60k-token prompt had worked
// earlier from a clean card (capacity + fragmentation, not generation).
// Both terms are tunable (autofit.workspace_mib / autofit.ctx_headroom_mib)
// and should be replaced by measurements when someone can produce them.
//
// Bytes per element: exllamav3 quantises KV per token in groups of 32
// channels after the H32 rotation, with an fp16 scale per group (engine
// cache/quant.py storage_size = payload + 2 fp16 scale planes), so Qn
// costs n/8 + 0.5/8 bytes/element and a "k,v" pair (k+v+1)/16 — FP16
// stores no scales and stays 2.
package autofit

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
)

const mib = 1 << 20

// thinMarginMiB: an accepted config with less than this above its full
// headroom is "thin" — it fits at idle but leaves the prefill workspace no
// slack on a used card, so it prints a warning (drop --ctx if prompts OOM
// before the first token).
const thinMarginMiB = 512

// Spec is everything the fit math needs from a HF config.json.
type Spec struct {
	Architectures []string
	Layers        int // layers contributing to the per-token KV cache
	KVHeads       int
	HeadDim       int
	MaxCtx        int // max_position_embeddings
	// QuantMethod is quantization_config.quant_method as embedded in
	// config.json (exllamav3 writes both a separate file and this block;
	// "" when absent). Not used by the fit math — it is the pre-download
	// gate's second, independent way to confirm a repo is EXL3.
	QuantMethod string
	// MoE is true when the config declares expert-parallel fields
	// (n_routed_experts / num_local_experts / num_experts /
	// moe_intermediate_size, top level or text_config). The gate uses it
	// to warn that .mul1 expert shards cannot be confirmed from metadata.
	MoE bool
}

type hfConfig struct {
	Architectures         []string `json:"architectures"`
	NumHiddenLayers       int      `json:"num_hidden_layers"`
	NumKeyValueHeads      int      `json:"num_key_value_heads"`
	HiddenSize            int      `json:"hidden_size"`
	NumAttentionHeads     int      `json:"num_attention_heads"`
	HeadDim               int      `json:"head_dim"`
	MaxPositionEmbeddings int      `json:"max_position_embeddings"`
	LayerTypes            []string `json:"layer_types"`
	QuantizationConfig    *struct {
		QuantMethod string `json:"quant_method"`
	} `json:"quantization_config"`
	// MoE expert markers — generic fields, not arch names (GLM-5.3-Flash
	// is Glm5NextForConditionalGeneration with 288 routed experts).
	NRoutedExperts      int              `json:"n_routed_experts"`
	NumLocalExperts     int              `json:"num_local_experts"`
	NumExperts          int              `json:"num_experts"`
	MoEIntermediateSize int              `json:"moe_intermediate_size"`
	TextConfig          *json.RawMessage `json:"text_config"`
}

// ParseSpec extracts the fit-relevant fields from a config.json.
// Hybrid/multimodal models keep text params under text_config (one level).
func ParseSpec(configJSON []byte) (Spec, error) {
	var raw hfConfig
	if err := json.Unmarshal(configJSON, &raw); err != nil {
		return Spec{}, fmt.Errorf("parse config.json: %w", err)
	}
	quantMethod := embeddedQuantMethod(raw)
	moe := hasMoEFields(raw)
	if raw.NumHiddenLayers == 0 && raw.TextConfig != nil {
		var inner hfConfig
		if err := json.Unmarshal(*raw.TextConfig, &inner); err == nil && inner.NumHiddenLayers > 0 {
			if len(inner.Architectures) == 0 {
				inner.Architectures = raw.Architectures
			}
			raw = inner
			if quantMethod == "" {
				quantMethod = embeddedQuantMethod(raw)
			}
			moe = moe || hasMoEFields(raw)
		}
	}

	headDim := raw.HeadDim
	if headDim <= 0 {
		if raw.NumAttentionHeads <= 0 || raw.HiddenSize <= 0 {
			return Spec{}, fmt.Errorf("config.json: cannot derive head_dim (hidden_size=%d, num_attention_heads=%d)",
				raw.HiddenSize, raw.NumAttentionHeads)
		}
		if raw.HiddenSize%raw.NumAttentionHeads != 0 {
			return Spec{}, fmt.Errorf("config.json: hidden_size %d not divisible by num_attention_heads %d",
				raw.HiddenSize, raw.NumAttentionHeads)
		}
		headDim = raw.HiddenSize / raw.NumAttentionHeads
	}

	spec := Spec{
		Architectures: raw.Architectures,
		Layers:        kvLayers(raw.NumHiddenLayers, raw.LayerTypes),
		KVHeads:       raw.NumKeyValueHeads,
		HeadDim:       headDim,
		MaxCtx:        raw.MaxPositionEmbeddings,
		QuantMethod:   quantMethod,
		MoE:           moe,
	}
	if spec.Layers <= 0 || spec.KVHeads <= 0 || spec.MaxCtx <= 0 {
		return Spec{}, fmt.Errorf("config.json: incomplete model description (layers=%d kv_heads=%d max_ctx=%d)",
			spec.Layers, spec.KVHeads, spec.MaxCtx)
	}
	return spec, nil
}

// embeddedQuantMethod reads quantization_config.quant_method from one
// config.json level (top level, or text_config for hybrid wrappers).
func embeddedQuantMethod(c hfConfig) string {
	if c.QuantizationConfig == nil {
		return ""
	}
	return c.QuantizationConfig.QuantMethod
}

// hasMoEFields reports whether one config level declares expert-parallel
// MoE fields. Probed generically because arch names don't reliably carry
// "moe" (Glm5NextForConditionalGeneration ships 288 routed experts).
func hasMoEFields(c hfConfig) bool {
	return c.NRoutedExperts > 0 || c.NumLocalExperts > 0 || c.NumExperts > 0 || c.MoEIntermediateSize > 0
}

// kvLayers counts layers that hold a per-token KV cache. Hybrid models
// interleave attention with state-based layers (linear attention / conv /
// recurrent) that carry O(1) state — those must not be charged per token.
// ponytail: substring rule over the known vocab; a family-specific parser
// is the upgrade path if a new layer type name appears.
func kvLayers(total int, layerTypes []string) int {
	if len(layerTypes) == 0 {
		return total
	}
	n := 0
	for _, t := range layerTypes {
		lt := strings.ToLower(t)
		if strings.Contains(lt, "linear_attention") || strings.Contains(lt, "conv") || strings.Contains(lt, "recurrent") {
			continue
		}
		n++
	}
	return n
}

// Options are the config knobs (ARCHITECTURE.md §5/§7). Zero values take
// the approved defaults.
type Options struct {
	HeadroomMiB       int    // base fragmentation headroom; default 512
	WorkspaceMiB      int    // cold prefill workspace reserve [est]; default 512
	CtxHeadroomMiB    int    // extra headroom at 65536 ctx, linear in ctx [est]; default 512
	OverheadMiB       int    // default 128 (measured fixed overhead)
	MinCtx            int    // default 4096 (ladder floor)
	UserCtx           int    // 0 → trained max; above trained max is clamped (Q3a)
	ForceMode         string // non-empty → bypass the ladder entirely (Q3b)
	ChunkSizeOverride int    // config autofit.chunk_size: 0 auto, else forced (512..4096)
}

// Result is the autofit decision plus the arithmetic behind it.
type Result struct {
	Ctx         int    // chosen (or attempted) context; 0 when !Fits
	Mode        string // FP16 | Q8 | Q4 | forced mode
	WeightsMiB  int
	KVMiB       int
	OverheadMiB int
	BudgetMiB   int // vram_total − headroom at the chosen ctx
	HeadroomMiB int // base + prefill workspace + ctx margin at the chosen ctx
	Fits        bool
	Clamped     bool   // UserCtx exceeded the trained max and was clamped (Q3a)
	Reduced     bool   // ladder lowered ctx below the target to make it fit
	Warning     string // thin headroom: prefill may OOM on a used card; drop --ctx
	LargestCtx  int    // largest ctx that would fit at the effective mode, 256-aligned
	Reason      string
	ElemsPerTok int64
	// Load tuning the verdict recommends for TabbyAPI's ModelLoadRequest;
	// 0/unset when the fit refused (no verdict to serve).
	ChunkSize int  // chunk_size in tokens: 2048 = comfortable (backend default)
	Warmup    bool // warmup: true only when slack is thin
}

func (o *Options) fillDefaults() {
	if o.HeadroomMiB <= 0 {
		o.HeadroomMiB = 512
	}
	if o.WorkspaceMiB <= 0 {
		o.WorkspaceMiB = 512
	}
	if o.CtxHeadroomMiB <= 0 {
		o.CtxHeadroomMiB = 512
	}
	if o.OverheadMiB <= 0 {
		o.OverheadMiB = 128
	}
	if o.MinCtx <= 0 {
		o.MinCtx = 4096
	}
}

// headroomFor is the space that must stay free above the load for a ctx:
// base headroom + cold prefill workspace [est] + a margin that scales with
// the on-card KV reservation [est] — a fuller card fragments worse, and an
// agent session grows its KV toward ctx mid-run. At 65536 ctx the margin is
// CtxHeadroomMiB; it halves with each ctx halving.
func (o Options) headroomFor(ctx int) int {
	return o.HeadroomMiB + o.WorkspaceMiB + o.CtxHeadroomMiB*ctx/65536
}

// Fit runs the ladder (or the forced override) against real numbers.
// The ladder maximizes ctx first (ctx tiers from target down to the
// floor), preferring cache quality within a tier: FP16 → Q8 → Q4 → "4,2".
func Fit(spec Spec, weightsBytes int64, vramTotalMiB int, opts Options) (Result, error) {
	if spec.Layers <= 0 || spec.KVHeads <= 0 || spec.HeadDim <= 0 || spec.MaxCtx <= 0 {
		return Result{}, fmt.Errorf("incomplete model spec: %+v", spec)
	}
	opts.fillDefaults()

	elems := int64(spec.Layers) * 2 * int64(spec.KVHeads) * int64(spec.HeadDim)
	weightsB := float64(weightsBytes)
	overheadB := float64(opts.OverheadMiB) * mib

	res := Result{
		WeightsMiB:  int(math.Round(weightsB / mib)),
		OverheadMiB: opts.OverheadMiB,
		ElemsPerTok: elems,
	}

	kvB := func(ctx int, bpe float64) float64 {
		return float64(ctx) * float64(elems) * bpe
	}
	fits := func(ctx int, bpe float64) bool {
		return weightsB+kvB(ctx, bpe)+overheadB <= float64(vramTotalMiB-opts.headroomFor(ctx))*mib
	}
	setKV := func(ctx int, bpe float64) {
		res.KVMiB = int(math.Ceil(kvB(ctx, bpe) / mib))
	}
	// accept records the budget/headroom for a config that passed fits()
	// and warns when the remaining margin is thin (see thinMarginMiB).
	accept := func(ctx int) {
		res.HeadroomMiB = opts.headroomFor(ctx)
		res.BudgetMiB = vramTotalMiB - res.HeadroomMiB
		load := res.WeightsMiB + res.KVMiB + res.OverheadMiB
		slack := vramTotalMiB - load - res.HeadroomMiB
		if slack < thinMarginMiB {
			res.Warning = fmt.Sprintf(
				"only %d MiB margin above the %d MiB headroom (prefill workspace [est] included): "+
					"multi-KB prompts can OOM during prefill on a used card — if you see CUDA OOM before the first token, drop --ctx",
				slack, res.HeadroomMiB)
		}
		// Load tuning from the SAME slack that gated the fit: thin slack
		// means the cold prefill workspace is the scarce resource, so cap
		// TabbyAPI's chunk and pre-warm on a throwaway prefill (transient
		// alloc stays small and failures show up before the real prompt).
		// Comfortable fits keep TabbyAPI's shipped defaults (2048, no
		// warmup). A config override forces the chunk size, never warmup.
		res.ChunkSize, res.Warmup = 2048, false
		switch {
		case slack < 256:
			res.ChunkSize, res.Warmup = 512, true
		case slack < thinMarginMiB:
			res.ChunkSize, res.Warmup = 1024, true
		}
		if opts.ChunkSizeOverride > 0 {
			res.ChunkSize = opts.ChunkSizeOverride
		}
	}

	// Target ctx — Q3a: never exceed the trained maximum.
	target := spec.MaxCtx
	if opts.UserCtx > 0 {
		if opts.UserCtx > spec.MaxCtx {
			res.Clamped = true
		} else {
			target = opts.UserCtx
		}
	}
	target = align256(target)

	// Q3b: an explicit mode bypasses the ladder; report the projection.
	if opts.ForceMode != "" {
		bpe, err := BytesPerElement(opts.ForceMode)
		if err != nil {
			return Result{}, err
		}
		res.Ctx = target
		res.Mode = strings.TrimSpace(opts.ForceMode)
		setKV(target, bpe)
		res.Fits = fits(target, bpe)
		if !res.Fits {
			res.Ctx = 0
			res.HeadroomMiB = opts.headroomFor(target)
			res.BudgetMiB = vramTotalMiB - res.HeadroomMiB
			res.LargestCtx = largestCtx(elems, bpe, vramTotalMiB, res.WeightsMiB, res.OverheadMiB, opts)
			res.Reason = refusalReason(res, opts, target, res.Mode)
		} else {
			accept(target)
		}
		return res, nil
	}

	// Cache-quality ladder within a ctx tier, best quality first:
	// FP16 → Q8 → Q4 → "4,2" (asymmetric K4V2, ~1008 MiB @65536 vs 1296
	// for Q4 incl. scales — pays for a full trained ctx when Q4 busts).
	// Q6 is skipped deliberately: Q4 is ~lossless for KV and Q6 @65536
	// has been measured to OOM on this 4 GB card, so the band is empty.
	// Q2 stays manual-only (--cache-mode Q2): documented 2-bit cliff
	// (KIVI AIME 51.88 → 64.79), never auto-picked.
	// Every bpe comes from BytesPerElement so the fit math and the
	// reported arithmetic can never drift apart.
	type rung struct {
		name string
		bpe  float64
	}
	var ladder []rung
	for _, name := range []string{"FP16", "Q8", "Q4", "4,2"} {
		bpe, err := BytesPerElement(name)
		if err != nil {
			return Result{}, err // ladder names are valid by construction
		}
		ladder = append(ladder, rung{name, bpe})
	}

	floor := opts.MinCtx
	if target < floor {
		floor = target // never push ctx above the trained max just to reach the floor
	}

	for ctx := target; ; {
		for _, m := range ladder {
			if fits(ctx, m.bpe) {
				res.Ctx, res.Mode = ctx, m.name
				setKV(ctx, m.bpe)
				res.Fits = true
				res.Reduced = ctx < target
				accept(ctx)
				return res, nil
			}
		}
		if ctx <= floor {
			break
		}
		next := align256(ctx / 2)
		if next >= ctx {
			break
		}
		if next < floor {
			next = floor
		}
		ctx = next
	}

	// Nothing fits — arithmetic for the refusal message.
	res.HeadroomMiB = opts.headroomFor(floor)
	res.BudgetMiB = vramTotalMiB - res.HeadroomMiB
	q4BPE, _ := BytesPerElement("Q4") // valid by construction (see ladder)
	res.LargestCtx = largestCtx(elems, q4BPE, vramTotalMiB, res.WeightsMiB, res.OverheadMiB, opts)
	setKV(floor, q4BPE)
	res.Reason = refusalReason(res, opts, floor, "Q4")
	return res, nil
}

// Summary is the always-printed projection line (Q3c), plus the thin-headroom
// warning when one applies.
func (r Result) Summary() string {
	s := fmt.Sprintf("weights %d + KV %d + overhead %d = %d MiB (headroom %d, budget %d)",
		r.WeightsMiB, r.KVMiB, r.OverheadMiB,
		r.WeightsMiB+r.KVMiB+r.OverheadMiB, r.HeadroomMiB, r.BudgetMiB)
	if r.Warning != "" {
		s += "\nwarning: " + r.Warning
	}
	return s
}

func refusalReason(res Result, opts Options, ctx int, modeLabel string) string {
	required := res.WeightsMiB + res.KVMiB + res.OverheadMiB
	largest := "none — no context fits; pull a smaller quant or use a bigger GPU"
	if res.LargestCtx > 0 {
		largest = strconv.Itoa(res.LargestCtx) + " (" + modeLabel + ")"
	}
	blanket := res.HeadroomMiB - opts.HeadroomMiB - opts.WorkspaceMiB
	if blanket < 0 {
		blanket = 0
	}
	return fmt.Sprintf(
		"no config fits: weights %d + KV (%s @ %d) %d + overhead %d = %d MiB > %d MiB budget (%d MiB VRAM − %d headroom: %d base + %d prefill workspace [est] + %d ctx margin [est])\n"+
			"largest ctx that would fit: %s",
		res.WeightsMiB, modeLabel, ctx, res.KVMiB, res.OverheadMiB,
		required, res.BudgetMiB, res.BudgetMiB+res.HeadroomMiB, res.HeadroomMiB,
		opts.HeadroomMiB, opts.WorkspaceMiB, blanket, largest)
}

// largestCtx solves load(ctx) + headroom(ctx) ≤ vram_total for ctx: the
// per-token cost is the KV bytes plus the ctx margin's per-token share, so
// the answer is smaller than the old headroom-free division (which ignored
// the margin's growth with ctx). Always 256-aligned downward.
func largestCtx(elems int64, bpe float64, vramTotalMiB, weightsMiB, overheadMiB int, opts Options) int {
	availMiB := float64(vramTotalMiB - opts.HeadroomMiB - opts.WorkspaceMiB - weightsMiB - overheadMiB)
	if availMiB <= 0 {
		return 0
	}
	perTokenMiB := float64(elems)*bpe/mib + float64(opts.CtxHeadroomMiB)/65536
	return align256(int(availMiB / perTokenMiB))
}

func align256(n int) int {
	if n <= 0 {
		return 0
	}
	return n - n%256 // TabbyAPI cache_size must be a multiple of 256
}

// scaleBPE is the fp16 group-scale overhead per KV element for quantised
// modes: one fp16 scale per group of 32 channels = 0.5 bit/element
// (engine cache/quant.py: storage_size = payload + 2 fp16 scale planes).
// FP16 stores no scales.
const scaleBPE = 0.5 / 8

// BytesPerElement converts a cache mode to the real bytes per KV element:
// quantised modes carry their group scales on top of the payload.
// Accepts legacy names (FP16, Q2..Q8) and TabbyAPI pair syntax (k,v), 2-8.
func BytesPerElement(mode string) (float64, error) {
	m := strings.ToUpper(strings.TrimSpace(mode))
	switch {
	case m == "FP16":
		return 2, nil
	case strings.HasPrefix(m, "Q"):
		n, err := strconv.Atoi(m[1:])
		if err != nil || n < 2 || n > 8 {
			return 0, invalidMode(mode)
		}
		return float64(n)/8 + scaleBPE, nil
	}
	if parts := strings.Split(m, ","); len(parts) == 2 {
		k, err1 := strconv.Atoi(strings.TrimSpace(parts[0]))
		v, err2 := strconv.Atoi(strings.TrimSpace(parts[1]))
		if err1 == nil && err2 == nil && k >= 2 && k <= 8 && v >= 2 && v <= 8 {
			return float64(k+v+1) / 16, nil // each side's scales: (k+v)/16 + 1/16
		}
	}
	return 0, invalidMode(mode)
}

func invalidMode(mode string) error {
	return fmt.Errorf("invalid cache mode %q (want FP16, Q2-Q8, or a k,v pair like \"2,4\")", mode)
}
