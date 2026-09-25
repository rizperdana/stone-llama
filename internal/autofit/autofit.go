// Package autofit derives a VRAM-fitting context/cache configuration from
// a model's config.json plus the GPU's total VRAM. This is the only place
// the KV-cache arithmetic exists (ARCHITECTURE.md §5):
//
//	elems/token = layers × 2 × kv_heads × head_dim
//	load        = weights + ctx × elems/token × bytes/element + overhead
//	fits        ⟺ load ≤ vram_total − headroom
//
// Bytes per element: FP16 = 2, Q8 = 1, Q6 = 0.75, Q4 = 0.5, Q2 = 0.25;
// a "k,v" pair averages to (k+v)/16.
package autofit

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
)

const mib = 1 << 20

// Spec is everything the fit math needs from a HF config.json.
type Spec struct {
	Architectures []string
	Layers        int // layers contributing to the per-token KV cache
	KVHeads       int
	HeadDim       int
	MaxCtx        int // max_position_embeddings
}

type hfConfig struct {
	Architectures         []string         `json:"architectures"`
	NumHiddenLayers       int              `json:"num_hidden_layers"`
	NumKeyValueHeads      int              `json:"num_key_value_heads"`
	HiddenSize            int              `json:"hidden_size"`
	NumAttentionHeads     int              `json:"num_attention_heads"`
	HeadDim               int              `json:"head_dim"`
	MaxPositionEmbeddings int              `json:"max_position_embeddings"`
	LayerTypes            []string         `json:"layer_types"`
	TextConfig            *json.RawMessage `json:"text_config"`
}

// ParseSpec extracts the fit-relevant fields from a config.json.
// Hybrid/multimodal models keep text params under text_config (one level).
func ParseSpec(configJSON []byte) (Spec, error) {
	var raw hfConfig
	if err := json.Unmarshal(configJSON, &raw); err != nil {
		return Spec{}, fmt.Errorf("parse config.json: %w", err)
	}
	if raw.NumHiddenLayers == 0 && raw.TextConfig != nil {
		var inner hfConfig
		if err := json.Unmarshal(*raw.TextConfig, &inner); err == nil && inner.NumHiddenLayers > 0 {
			if len(inner.Architectures) == 0 {
				inner.Architectures = raw.Architectures
			}
			raw = inner
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
	}
	if spec.Layers <= 0 || spec.KVHeads <= 0 || spec.MaxCtx <= 0 {
		return Spec{}, fmt.Errorf("config.json: incomplete model description (layers=%d kv_heads=%d max_ctx=%d)",
			spec.Layers, spec.KVHeads, spec.MaxCtx)
	}
	return spec, nil
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
	HeadroomMiB int    // default 512
	OverheadMiB int    // default 128 (measured fixed overhead)
	MinCtx      int    // default 4096 (ladder floor)
	UserCtx     int    // 0 → trained max; above trained max is clamped (Q3a)
	ForceMode   string // non-empty → bypass the ladder entirely (Q3b)
}

// Result is the autofit decision plus the arithmetic behind it.
type Result struct {
	Ctx         int    // chosen (or attempted) context; 0 when !Fits
	Mode        string // FP16 | Q8 | Q4 | forced mode
	WeightsMiB  int
	KVMiB       int
	OverheadMiB int
	BudgetMiB   int // vram_total − headroom
	Fits        bool
	Clamped     bool // UserCtx exceeded the trained max and was clamped (Q3a)
	Reduced     bool // ladder lowered ctx below the target to make it fit
	LargestCtx  int  // largest ctx that would fit at the effective mode, 256-aligned
	Reason      string
	ElemsPerTok int64
}

func (o *Options) fillDefaults() {
	if o.HeadroomMiB <= 0 {
		o.HeadroomMiB = 512
	}
	if o.OverheadMiB <= 0 {
		o.OverheadMiB = 128
	}
	if o.MinCtx <= 0 {
		o.MinCtx = 4096
	}
}

// Fit runs the ladder (or the forced override) against real numbers.
// The ladder maximizes ctx first (ctx tiers from target down to the
// floor), preferring cache quality within a tier: FP16 → Q8 → Q4.
func Fit(spec Spec, weightsBytes int64, vramTotalMiB int, opts Options) (Result, error) {
	if spec.Layers <= 0 || spec.KVHeads <= 0 || spec.HeadDim <= 0 || spec.MaxCtx <= 0 {
		return Result{}, fmt.Errorf("incomplete model spec: %+v", spec)
	}
	opts.fillDefaults()

	elems := int64(spec.Layers) * 2 * int64(spec.KVHeads) * int64(spec.HeadDim)
	weightsB := float64(weightsBytes)
	overheadB := float64(opts.OverheadMiB) * mib
	budgetMiB := vramTotalMiB - opts.HeadroomMiB
	budgetB := float64(budgetMiB) * mib

	res := Result{
		WeightsMiB:  int(math.Round(weightsB / mib)),
		OverheadMiB: opts.OverheadMiB,
		BudgetMiB:   budgetMiB,
		ElemsPerTok: elems,
	}

	kvB := func(ctx int, bpe float64) float64 {
		return float64(ctx) * float64(elems) * bpe
	}
	fits := func(ctx int, bpe float64) bool {
		return weightsB+kvB(ctx, bpe)+overheadB <= budgetB
	}
	setKV := func(ctx int, bpe float64) {
		res.KVMiB = int(math.Ceil(kvB(ctx, bpe) / mib))
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
			res.LargestCtx = largestCtx(elems, bpe, budgetB, weightsB, overheadB)
			res.Reason = refusalReason(res, opts, target, res.Mode)
		}
		return res, nil
	}

	ladderModes := []struct {
		name string
		bpe  float64
	}{{"FP16", 2}, {"Q8", 1}, {"Q4", 0.5}}

	floor := opts.MinCtx
	if target < floor {
		floor = target // never push ctx above the trained max just to reach the floor
	}

	for ctx := target; ; {
		for _, m := range ladderModes {
			if fits(ctx, m.bpe) {
				res.Ctx, res.Mode = ctx, m.name
				setKV(ctx, m.bpe)
				res.Fits = true
				res.Reduced = ctx < target
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
	res.LargestCtx = largestCtx(elems, 0.5, budgetB, weightsB, overheadB)
	setKV(floor, 0.5)
	res.Reason = refusalReason(res, opts, floor, "Q4")
	return res, nil
}

// Summary is the always-printed projection line (Q3c).
func (r Result) Summary() string {
	return fmt.Sprintf("weights %d + KV %d + overhead %d = %d MiB (budget %d)",
		r.WeightsMiB, r.KVMiB, r.OverheadMiB,
		r.WeightsMiB+r.KVMiB+r.OverheadMiB, r.BudgetMiB)
}

func refusalReason(res Result, opts Options, ctx int, modeLabel string) string {
	required := res.WeightsMiB + res.KVMiB + res.OverheadMiB
	largest := "none — no context fits; pull a smaller quant or use a bigger GPU"
	if res.LargestCtx > 0 {
		largest = strconv.Itoa(res.LargestCtx) + " (" + modeLabel + ")"
	}
	return fmt.Sprintf(
		"no config fits: weights %d + KV (%s @ %d) %d + overhead %d = %d MiB > %d MiB budget (%d MiB VRAM − %d headroom)\n"+
			"largest ctx that would fit: %s",
		res.WeightsMiB, modeLabel, ctx, res.KVMiB, res.OverheadMiB,
		required, res.BudgetMiB, res.BudgetMiB+opts.HeadroomMiB, opts.HeadroomMiB, largest)
}

func largestCtx(elems int64, bpe, budgetB, weightsB, overheadB float64) int {
	avail := budgetB - weightsB - overheadB
	if avail <= 0 {
		return 0
	}
	return align256(int(avail / (float64(elems) * bpe)))
}

func align256(n int) int {
	if n <= 0 {
		return 0
	}
	return n - n%256 // TabbyAPI cache_size must be a multiple of 256
}

// BytesPerElement converts a cache mode to bytes per KV element.
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
		return float64(n) / 8, nil
	}
	if parts := strings.Split(m, ","); len(parts) == 2 {
		k, err1 := strconv.Atoi(strings.TrimSpace(parts[0]))
		v, err2 := strconv.Atoi(strings.TrimSpace(parts[1]))
		if err1 == nil && err2 == nil && k >= 2 && k <= 8 && v >= 2 && v <= 8 {
			return float64(k+v) / 16, nil
		}
	}
	return 0, invalidMode(mode)
}

func invalidMode(mode string) error {
	return fmt.Errorf("invalid cache mode %q (want FP16, Q2-Q8, or a k,v pair like \"2,4\")", mode)
}
