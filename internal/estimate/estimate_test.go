package estimate

import "testing"

// The measured anchor this whole package is calibrated against:
// SmolLM3-3B-exl3 3.5bpw (1,957,008,720 B) → 42.7 decode tok/s on an
// RTX 3050 Laptop. If this test breaks, recalibration moved on purpose —
// update the package doc's citation and the research note together.
func TestAnchorDecode427(t *testing.T) {
	p := Predict(Input{
		WeightsBytes:  1_957_008_720,
		BitsPerWeight: 3.5,
		ElemsPerTok:   36_864,
		Ctx:           0,
		Mode:          "Q4",
		GPUName:       "NVIDIA GeForce RTX 3050 Laptop GPU",
	})
	if !p.Known {
		t.Fatalf("anchor card must be known: %s", p.Note)
	}
	if p.DecodeTokPerS < 42.6 || p.DecodeTokPerS > 42.8 {
		t.Errorf("decode = %.2f t/s, want 42.7 ± 0.1 (calibration anchor)", p.DecodeTokPerS)
	}
	if p.ParamsApprox < 4.4e9 || p.ParamsApprox > 4.55e9 {
		t.Errorf("params ≈ %.3e, want ~4.47e9 (weights×8/3.5)", p.ParamsApprox)
	}
	if p.Note != "" {
		t.Errorf("unexpected note: %q", p.Note)
	}
}

func TestPrefillEstimate(t *testing.T) {
	p := Predict(Input{
		WeightsBytes:  1_957_008_720,
		BitsPerWeight: 3.5,
		GPUName:       "RTX 3050 Laptop",
	})
	if !p.Known {
		t.Fatal("must be known")
	}
	// 0.40 × 9.1e12 / (2 × weights×8/3.5) ≈ 407 tok/s [est]
	if p.PrefillTokPerS < 395 || p.PrefillTokPerS > 420 {
		t.Errorf("prefill = %.0f t/s, want ~407 ± 12 [est]", p.PrefillTokPerS)
	}
}

func TestDecodeFallsWithCtxAndMode(t *testing.T) {
	base := Input{WeightsBytes: 1_957_008_720, BitsPerWeight: 3.5, ElemsPerTok: 36_864, GPUName: "RTX 3050"}
	atZero := Predict(base)
	base.Ctx = 65536
	q4 := Predict(base)
	base.Mode = "Q8"
	q8 := Predict(base)
	if q4.DecodeTokPerS >= atZero.DecodeTokPerS {
		t.Errorf("ctx 65536 Q4 (%.1f) must be slower than ctx 0 (%.1f)", q4.DecodeTokPerS, atZero.DecodeTokPerS)
	}
	if q8.DecodeTokPerS >= q4.DecodeTokPerS {
		t.Errorf("Q8 KV (%.1f) must cost more than Q4 (%.1f)", q8.DecodeTokPerS, q4.DecodeTokPerS)
	}
}

func TestUnknownGPU(t *testing.T) {
	p := Predict(Input{WeightsBytes: 1_957_008_720, GPUName: "AMD Radeon RX 6800"})
	if p.Known || p.Note == "" {
		t.Fatalf("unknown GPU must be unknown with a note: %+v", p)
	}
}

func TestMissingWeights(t *testing.T) {
	p := Predict(Input{GPUName: "RTX 3050"})
	if p.Known {
		t.Fatal("zero weights must be unknown")
	}
}

func TestMissingBPWNoted(t *testing.T) {
	p := Predict(Input{WeightsBytes: 1 << 30, GPUName: "RTX 3050"})
	if !p.Known || p.Note == "" {
		t.Fatalf("assumed-bits must be noted: %+v", p)
	}
}

// The estimator must reproduce its own calibration point with the real
// KV footprint in play (18,432 B/token = 36,864 elems × 0.5 B at Q4),
// not only at ctx 0: 42.7 tok/s ±15% = [36.3, 49.1].
func TestCalibrationWithKVWithin15Percent(t *testing.T) {
	const ctx = 8192
	p := Predict(Input{
		WeightsBytes:  1_957_008_720, // 1866 MiB, SmolLM3-3B 3.5bpw
		BitsPerWeight: 3.5,
		ElemsPerTok:   36_864, // = 18,432 B/token at Q4
		Ctx:           ctx,
		Mode:          "Q4",
		GPUName:       "RTX 3050 Laptop",
	})
	if !p.Known {
		t.Fatalf("calibration card must be known: %s", p.Note)
	}
	if p.DecodeTokPerS < 42.7*0.85 || p.DecodeTokPerS > 42.7*1.15 {
		t.Errorf("decode @ctx %d = %.1f t/s, want 42.7 ±15%% ([%.1f, %.1f])",
			ctx, p.DecodeTokPerS, 42.7*0.85, 42.7*1.15)
	}
}

// Property: identical architecture family (same KV bytes/token), same
// cache mode, same ctx — less weight-bytes must never predict fewer
// tok/s. This catches the defect class where weights are double-counted,
// a unit flips (MiB vs bytes), or the efficiency constant gets applied
// per-model: any of those mis-orders a 1.7B against its 3B sibling at a
// fixed context. Cross-arch/cross-ctx comparisons are NOT this property
// (KV bytes/token differ per family; decode reads weights + full cache).
func TestDecodeMonotoneInWeights(t *testing.T) {
	const ctx = 40960
	weights := []int64{
		1_563_189_248, // 1491 MiB — 4.0bpw 1.7B-class
		1_957_008_720, // 1866 MiB — the 3.5bpw anchor
		3_221_225_472, // 3 GiB
		6_442_450_944, // 6 GiB
	}
	prev := 1e9
	for i, w := range weights {
		p := Predict(Input{
			WeightsBytes:  w,
			BitsPerWeight: 3.5,
			ElemsPerTok:   36_864,
			Ctx:           ctx,
			Mode:          "Q4",
			GPUName:       "RTX 3050",
		})
		if !p.Known {
			t.Fatalf("weights[%d]=%d: %s", i, w, p.Note)
		}
		if p.DecodeTokPerS >= prev {
			t.Errorf("weights %d B: decode %.1f t/s must be < previous %.1f (monotone in weight bytes)",
				w, p.DecodeTokPerS, prev)
		}
		prev = p.DecodeTokPerS
	}
}
