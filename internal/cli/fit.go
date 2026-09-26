package cli

import (
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/rizperdana/stone-llama/internal/autofit"
	"github.com/rizperdana/stone-llama/internal/config"
	"github.com/rizperdana/stone-llama/internal/doctor"
	"github.com/rizperdana/stone-llama/internal/estimate"
	"github.com/rizperdana/stone-llama/internal/hf"
	"github.com/rizperdana/stone-llama/internal/pull"
	"github.com/rizperdana/stone-llama/internal/store"
)

// hfBaseURL points HF traffic at HF_ENDPOINT when set (mirrors, tests);
// empty → https://huggingface.co.
func hfBaseURL() string {
	if v := strings.TrimRight(os.Getenv("HF_ENDPOINT"), "/"); v != "" {
		return v
	}
	return hf.DefaultBaseURL
}

// tokenFilePath is where `stone-llama login` stores the HF token (0600).
func tokenFilePath() string {
	return filepath.Join(config.DataDir(), "hf_token")
}

// dryOpts builds a metadata-only pull option set for fit/rank: the A5
// gate runs, no weight byte is fetched, no lock or dir is touched.
func dryOpts(cfg *config.Config, rep doctor.Report) pull.Options {
	return pull.Options{
		ModelsDir: cfg.ModelsDir,
		DryRun:    true,
		Yes:       true,
		BaseURL:   hfBaseURL(),
		Token:     hf.LoadToken(os.Getenv("HF_TOKEN"), tokenFilePath()),
		VRAMMiB:   rep.GPUs[0].VRAMMiB,
		GPUName:   rep.GPUs[0].Name,
		Autofit: autofit.Options{
			HeadroomMiB:    cfg.Autofit.HeadroomMiB,
			WorkspaceMiB:   cfg.Autofit.WorkspaceMiB,
			CtxHeadroomMiB: cfg.Autofit.CtxHeadroomMiB,
			OverheadMiB:    cfg.Autofit.OverheadMiB,
			MinCtx:         cfg.Autofit.MinCtx,
		},
	}
}

// runFit answers "does this model fit here, at what context, how fast" —
// HF metadata only, nothing written (exit 3 when the gate refuses).
func runFit(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	if len(args) != 1 {
		fmt.Fprintln(stderr, "usage: stone-llama fit <repo[@branch][:quant]>   fit verdict + ctx/cache pick + tok/s estimate, no download (exit 3 = refused)")
		return 2
	}
	cfg, rep, ok := gpuPreamble("fit", stdout, stderr)
	if !ok {
		return 1
	}
	opts := dryOpts(cfg, rep)
	opts.Ref = args[0]
	opts.Out = stdout
	opts.Stdin = stdin
	opts.Interactive = isTerminal(stdin)
	opts.TTY = isTerminal(stdout)
	res, err := pull.Run(opts)
	if err != nil {
		fmt.Fprintf(stderr, "stone-llama fit: %v\n", err)
		return 1
	}
	if res.Refused {
		fmt.Fprintln(stdout, "fit: refused — no context fits this model in VRAM (see the gate report above)")
		return 3
	}
	fmt.Fprintln(stdout, estimateLine(res, rep.GPUs[0].Name))
	return 0
}

// estimateLine renders the single-line speed prediction for a fit result.
func estimateLine(res pull.Result, gpuName string) string {
	if res.Verdict.MaxCtx <= 0 {
		return "estimate: unavailable — the gate picked no viable context (see the report above)"
	}
	p := predict(res, gpuName)
	if !p.Known {
		return "estimate: unavailable — " + p.Note
	}
	line := fmt.Sprintf("estimate: ~%.0f tok/s decode, ~%.0f tok/s prefill [est] at %s ctx %d (%s) — low confidence, anchored to the measured 42.7 tok/s SmolLM3-3B 3.5bpw point (1866 MiB) on this GPU; calibration pending",
		p.DecodeTokPerS, p.PrefillTokPerS, res.Verdict.CacheMode, res.Verdict.MaxCtx, gpuName)
	if p.Note != "" {
		line += "; " + p.Note
	}
	return line
}

// predict runs the estimator for a dry-run pull result.
func predict(res pull.Result, gpuName string) estimate.Prediction {
	return estimate.Predict(estimate.Input{
		WeightsBytes:  res.WeightsBytes,
		BitsPerWeight: bitsFromLabel(res.QuantLabel),
		ElemsPerTok:   specElems(res.Spec),
		Ctx:           res.Verdict.MaxCtx,
		Mode:          res.Verdict.CacheMode,
		GPUName:       gpuName,
	})
}

// estCells turns a dry-run fit result into two table cells ("-","-"
// when the gate refused or the GPU has no calibration profile).
func estCells(res pull.Result, gpuName string) (string, string) {
	if res.Refused || res.Verdict.MaxCtx <= 0 {
		return "-", "-"
	}
	p := predict(res, gpuName)
	if !p.Known {
		return "-", "-"
	}
	return fmt.Sprintf("%.0f", p.DecodeTokPerS), fmt.Sprintf("%.0f", p.PrefillTokPerS)
}

// specElems is the per-token KV element count: layers × 2 (K+V) × heads × dim.
func specElems(s autofit.Spec) int64 {
	return int64(s.Layers) * 2 * int64(s.KVHeads) * int64(s.HeadDim)
}

// bitsFromLabel reads "3.5bpw"-style quant labels; 0 = not readable
// (the estimator then assumes its default and says so).
func bitsFromLabel(label string) float64 {
	lower := strings.ToLower(label)
	i := strings.Index(lower, "bpw")
	if i < 0 {
		return 0
	}
	start := i
	for start > 0 && (lower[start-1] == '.' || (lower[start-1] >= '0' && lower[start-1] <= '9')) {
		start--
	}
	f, err := strconv.ParseFloat(lower[start:i], 64)
	if err != nil || f <= 0 {
		return 0
	}
	return f
}

// localEstimate computes the two EST columns for an installed model from
// its on-disk config.json + weight bytes (verdict ctx when stored,
// otherwise a live autofit projection); "-","-" when inputs are missing.
func localEstimate(m store.Model, gpuName string, vramMiB int, cfg config.Config) (string, string) {
	if gpuName == "" {
		return "-", "-"
	}
	w := manifestWeights(m)
	if w <= 0 {
		return "-", "-"
	}
	spec, err := specFromDir(m.Path)
	if err != nil || spec.Layers == 0 {
		return "-", "-"
	}
	ctx, mode := 0, ""
	if m.Verdict != nil && m.Verdict.MaxCtx > 0 {
		ctx, mode = m.Verdict.MaxCtx, m.Verdict.CacheMode
	} else {
		res, err := autofit.Fit(spec, w, vramMiB, autofit.Options{
			HeadroomMiB:    cfg.Autofit.HeadroomMiB,
			WorkspaceMiB:   cfg.Autofit.WorkspaceMiB,
			CtxHeadroomMiB: cfg.Autofit.CtxHeadroomMiB,
			OverheadMiB:    cfg.Autofit.OverheadMiB,
			MinCtx:         cfg.Autofit.MinCtx,
		})
		if err != nil || !res.Fits {
			return "-", "-"
		}
		ctx, mode = res.Ctx, res.Mode
	}
	p := estimate.Predict(estimate.Input{
		WeightsBytes:  w,
		BitsPerWeight: bitsFromLabel(m.Quant),
		ElemsPerTok:   specElems(spec),
		Ctx:           ctx,
		Mode:          mode,
		GPUName:       gpuName,
	})
	if !p.Known {
		return "-", "-"
	}
	return fmt.Sprintf("%.0f", p.DecodeTokPerS), fmt.Sprintf("%.0f", p.PrefillTokPerS)
}

// manifestWeights sums .safetensors bytes from the manifest, falling
// back to a directory walk for imported/unmanaged models (no manifest).
func manifestWeights(m store.Model) int64 {
	if m.Manifest != nil {
		var sum int64
		for _, f := range m.Manifest.Files {
			if strings.HasSuffix(strings.ToLower(f.Path), ".safetensors") {
				sum += f.Size
			}
		}
		if sum > 0 {
			return sum
		}
	}
	return safetensorsBytes(m.Path)
}

// safetensorsBytes walks dir (following the top-level symlink) and sums
// .safetensors files, up to 2 directories deep.
func safetensorsBytes(dir string) int64 {
	real, err := filepath.EvalSymlinks(dir)
	if err != nil {
		return 0
	}
	var sum int64
	_ = filepath.WalkDir(real, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(real, path)
		depth := strings.Count(filepath.ToSlash(rel), "/")
		if d.IsDir() {
			if depth > 2 {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(strings.ToLower(d.Name()), ".safetensors") {
			if info, err := d.Info(); err == nil {
				sum += info.Size()
			}
		}
		return nil
	})
	return sum
}

// specFromDir finds config.json in a model dir (≤2 levels deep) and
// parses it into an autofit spec.
func specFromDir(dir string) (autofit.Spec, error) {
	real, err := filepath.EvalSymlinks(dir)
	if err != nil {
		return autofit.Spec{}, err
	}
	var spec autofit.Spec
	err = filepath.WalkDir(real, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(real, path)
		depth := strings.Count(filepath.ToSlash(rel), "/")
		if d.IsDir() {
			if depth > 2 {
				return filepath.SkipDir
			}
			return nil
		}
		if depth <= 2 && d.Name() == "config.json" {
			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			spec, err = autofit.ParseSpec(data)
			return err
		}
		return nil
	})
	return spec, err
}
