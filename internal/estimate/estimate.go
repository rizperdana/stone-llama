// Package estimate predicts decode and prefill tokens-per-second for EXL3
// models on a given NVIDIA GPU from metadata alone (weights, quant, KV
// geometry, context) — no model on disk required.
//
// Calibration status: decode is fitted to the single measured anchor below;
// prefill is unfitted (the research note exists, but Predict cannot use its
// prompt-length curve — see prefillEfficiency). Source:
//
//	decode anchor: SmolLM3-3B-exl3 3.5bpw, 1,957,008,720 B weights,
//	               42.7 tok/s on an RTX 3050 Laptop (spec 224 GB/s)
//
// decodeEfficiency = 42.7 × 1,957,008,720 / 224e9 ≈ 0.373 — the note's
// §1 "Alt. (224 GB/s assumption)" row (K ≈ 83.6 GB/s either way).
// Prefill in the note is prompt-length-dependent (two measured anchors,
// 1,224 tok/s @ 15,797 prompt tokens and 691 tok/s @ 60,076, fitted by
// 1689 / (1 + L/41608)); Predict has no prompt-length input, so the
// prefillEfficiency constant cannot reproduce that curve — see its
// comment. Recalibration must stay a one-line change here. Everything
// this package produces is an estimate and must be labelled [est] by
// callers.
package estimate

import (
	"fmt"
	"strings"

	"github.com/rizperdana/stone-llama/internal/autofit"
)

const (
	// decodeEfficiency is the fraction of the card's spec memory bandwidth
	// actually achieved during autoregressive decode: each generated token
	// streams the full weights plus the live KV cache once.
	// [est] — derived from the research note (toks-estimator.md §1/§3):
	// K = 42.7 × 1,957,008,720 ≈ 83.57 GB/s ≈ 0.373 × 224 GB/s.
	decodeEfficiency = 0.373

	// prefillEfficiency is the fraction of spec'd FP16 tensor throughput
	// achieved during a dense prefill GEMM. [est] placeholder — NOT
	// calibrated. The research note (toks-estimator.md §3) has two measured
	// prefill anchors — 1,224 tok/s @ L=15,797 and 691 tok/s @ L=60,076,
	// reproduced by 1689 / (1 + L/41608) — but Predict is prompt-length-
	// independent (there is no L input; the value below is a constant), so
	// it cannot reproduce that curve: for SmolLM3-3B it emits ~407 tok/s at
	// any prompt length, against a measured 1,224 already at L=15,797.
	// Wiring a prompt-length term in would need per-arch C/L_char
	// validation against BOTH anchors plus a caller-chosen default L that
	// keeps every documented fit/list figure consistent — neither exists
	// today, so behaviour is deliberately left alone. Not a pending task;
	// an honest limitation until someone measures per-arch anchors.
	prefillEfficiency = 0.40

	// defaultBits is assumed when the quant label carries no bpw number.
	defaultBits = 4.0
)

// card holds the spec numbers an estimate is derived from.
// Spec-sheet values, [est] — see the package doc.
type card struct {
	bwGBs     float64 // spec memory bandwidth, GB/s
	prefillTF float64 // spec FP16 tensor TFLOPS (dense)
}

// cards is keyed by a substring matched case-insensitively against the
// nvidia-smi product name. Add a card here when calibrating on it.
var cards = map[string]card{
	"rtx 3050": {bwGBs: 224, prefillTF: 9.1},
	"rtx 4090": {bwGBs: 1008, prefillTF: 165},
}

// Input is everything the model needs. All of it comes from metadata or the
// autofit decision — never from running the model.
type Input struct {
	WeightsBytes  int64
	BitsPerWeight float64 // from the quant label ("3.5bpw"); <= 0 → defaultBits
	ElemsPerTok   int64   // KV elements per token (layers × 2 × kv_heads × head_dim)
	Ctx           int     // candidate context the estimate is for
	Mode          string  // cache mode: FP16 | Q8 | Q4 | …
	GPUName       string  // nvidia-smi product name
}

// Prediction is the estimated rates for one Input.
type Prediction struct {
	Known          bool    // false → the GPU has no calibration profile
	DecodeTokPerS  float64 // generated tokens/s (streaming chat)
	PrefillTokPerS float64 // prompt-processing tokens/s
	ParamsApprox   float64 // derived: weights×8/bpw (overcounts tied embeddings) [est]
	Note           string  // non-empty → append to output verbatim
}

// Predict estimates decode and prefill rates. Deterministic, allocation-light,
// safe to call per row in a table.
func Predict(in Input) Prediction {
	if in.WeightsBytes <= 0 {
		return Prediction{Known: false, Note: "no weight size known — estimates unavailable"}
	}
	c, ok := lookup(in.GPUName)
	if !ok {
		return Prediction{
			Known: false,
			Note: fmt.Sprintf("no calibration profile for GPU %q — estimates unavailable",
				strings.TrimSpace(in.GPUName)),
		}
	}
	bits := in.BitsPerWeight
	var note string
	if bits <= 0 {
		bits = defaultBits
		note = "bpw not read from the quant label; assuming 4.0 bpw [est]"
	}
	bpe, err := autofit.BytesPerElement(in.Mode)
	if err != nil {
		bpe = 0.5 // unknown mode → treat as Q4 KV cost; caller's gate already validated it
	}
	decodeBytes := float64(in.WeightsBytes) + float64(in.Ctx)*float64(in.ElemsPerTok)*bpe
	decode := decodeEfficiency * c.bwGBs * 1e9 / decodeBytes
	params := float64(in.WeightsBytes) * 8 / bits
	prefill := prefillEfficiency * c.prefillTF * 1e12 / (2 * params)
	return Prediction{
		Known:          true,
		DecodeTokPerS:  decode,
		PrefillTokPerS: prefill,
		ParamsApprox:   params,
		Note:           note,
	}
}

func lookup(gpuName string) (card, bool) {
	name := strings.ToLower(gpuName)
	for key, c := range cards {
		if strings.Contains(name, key) {
			return c, true
		}
	}
	return card{}, false
}
