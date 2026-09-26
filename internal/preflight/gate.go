// Package preflight implements the A5 pre-download gate: architecture,
// quant, and fit checks run against metadata only — before any weight
// byte moves (ARCHITECTURE.md §4).
package preflight

import (
	"fmt"
	"strings"

	"github.com/rizperdana/stone-llama/internal/autofit"
	"github.com/rizperdana/stone-llama/internal/store"
)

const (
	StatusOK     = "ok"
	StatusWarn   = "warn"
	StatusRefuse = "refuse"
)

// unknownArchAdvice is the warn-only text for archs outside SupportedArchs.
const unknownArchAdvice = "unknown to the embedded support table; the model may still load — verify before pulling"

type Check struct {
	Name   string // "arch" | "quant" | "fit"
	Status string
	Detail string
}

type Report struct {
	Checks  []Check
	Verdict store.Verdict  // persisted to manifest.json
	Fit     autofit.Result // full projection (ctx, mode, arithmetic)
	Refused bool
	Warned  bool // warnings present (caller decides: TTY-confirm or --yes)
}

// Input is everything Evaluate needs; pull assembles it from the HF
// metadata fetch (KB-scale) and config parse — no weight bytes involved.
type Input struct {
	Architectures []string // config.architectures
	QuantMethod   string   // quantization_config.quant_method ("" when absent)
	HasQuantCfg   bool     // quantization_config.json exists
	RepoFiles     []string // all repo file paths
	Repo          string   // repo id shown in gate messages ("" → "this repo")
	Spec          autofit.Spec
	WeightsBytes  int64 // sum of .safetensors sizes in scope
	VRAMMiB       int
	GPUName       string
	Opts          autofit.Options
}

// Evaluate runs the three checks and computes the verdict. A refusal
// never short-circuits: the report always shows the full picture.
func Evaluate(in Input) Report {
	r := Report{Fit: autofit.Result{}}
	fitRes, fitErr := autofit.Fit(in.Spec, in.WeightsBytes, in.VRAMMiB, in.Opts)
	r.Fit = fitRes

	r.Checks = []Check{checkArch(in), checkQuant(in)}
	if fitErr != nil {
		r.Checks = append(r.Checks, Check{Name: "fit", Status: StatusRefuse,
			Detail: "fit projection failed: " + fitErr.Error()})
	} else {
		r.Checks = append(r.Checks, Check{Name: "fit", Status: fitStatus(fitRes), Detail: fitDetail(fitRes)})
	}

	var notes []string
	for _, c := range r.Checks {
		switch c.Status {
		case StatusRefuse:
			r.Refused = true
			notes = append(notes, strings.ReplaceAll(c.Detail, "\n", " "))
		case StatusWarn:
			r.Warned = true
			notes = append(notes, strings.ReplaceAll(c.Detail, "\n", " "))
		}
	}
	r.Verdict = store.Verdict{
		Status:    store.VerdictOK,
		Note:      strings.Join(notes, "; "),
		MaxCtx:    fitRes.Ctx,
		CacheMode: fitRes.Mode,
	}
	if r.Refused {
		r.Verdict.Status = store.VerdictRefuse
	} else if r.Warned {
		r.Verdict.Status = store.VerdictWarn
	}
	return r
}

func checkArch(in Input) Check {
	if len(in.Architectures) == 0 {
		return Check{Name: "arch", Status: StatusWarn,
			Detail: "config.json has no architectures[] — exllamav3 support cannot be checked"}
	}
	var unknown []string
	for _, a := range in.Architectures {
		if !SupportedArchs[a] {
			unknown = append(unknown, a)
		}
	}
	switch {
	case len(unknown) == len(in.Architectures):
		return Check{Name: "arch", Status: StatusWarn,
			Detail: fmt.Sprintf("unknown architecture %q — %s", strings.Join(unknown, ", "), unknownArchAdvice)}
	case len(unknown) > 0:
		return Check{Name: "arch", Status: StatusWarn,
			Detail: fmt.Sprintf("partially unknown architecture: %q — %s", strings.Join(unknown, ", "), unknownArchAdvice)}
	default:
		return Check{Name: "arch", Status: StatusOK, Detail: strings.Join(in.Architectures, ", ")}
	}
}

func checkQuant(in Input) Check {
	repo := in.Repo
	if repo == "" {
		repo = "this repo"
	}
	hint := "stone-llama only loads EXL3 quantizations — look for '-exl3' / exl3 quantization_config in " + repo
	method := strings.ToLower(strings.TrimSpace(in.QuantMethod))
	gguf := hasSuffix(in.RepoFiles, ".gguf")
	safetensors := hasSuffix(in.RepoFiles, ".safetensors")

	switch {
	// Real ExLlamaV2 repos ship their quant data in measurement.json and
	// omit quantization_config.json — without this check they slipped
	// through as "maybe FP16" and were only refused by VRAM later.
	case hasSuffix(in.RepoFiles, "measurement.json"):
		return Check{Name: "quant", Status: StatusRefuse,
			Detail: "EXL2 quant (ExLlamaV2 format, measurement.json) — exllamav3 cannot load it; " + hint}
	case !gguf && !safetensors:
		return Check{Name: "quant", Status: StatusRefuse,
			Detail: fmt.Sprintf("no model weights found in %s (only %d non-weight files) — pick a repo that publishes .safetensors",
				repo, len(in.RepoFiles))}
	case method == "exl3":
		return Check{Name: "quant", Status: StatusOK, Detail: "exl3"}
	case method == "exl2":
		return Check{Name: "quant", Status: StatusRefuse,
			Detail: "EXL2 quant (ExLlamaV2 format) — exllamav3 cannot load it; " + hint}
	case method != "":
		return Check{Name: "quant", Status: StatusRefuse,
			Detail: fmt.Sprintf("quant_method %q not exl3 — detected format is unsupported; %s", method, hint)}
	case gguf && !safetensors:
		return Check{Name: "quant", Status: StatusRefuse,
			Detail: "no EXL3 weights — this repo ships GGUF; use ollama with GGUF instead; " + hint}
	case in.HasQuantCfg:
		return Check{Name: "quant", Status: StatusWarn,
			Detail: "quantization_config.json has no quant_method — cannot confirm EXL3; " + hint}
	default:
		return Check{Name: "quant", Status: StatusWarn,
			Detail: "no quantization_config.json — cannot confirm EXL3 (may be unquantized FP16); " + hint}
	}
}

func fitStatus(res autofit.Result) string {
	switch {
	case !res.Fits:
		return StatusRefuse
	case res.Reduced || res.Warning != "":
		return StatusWarn
	default:
		return StatusOK
	}
}

func fitDetail(res autofit.Result) string {
	switch {
	case !res.Fits:
		return res.Reason + "\nreduce --ctx or pick a smaller quant"
	case res.Reduced:
		return fmt.Sprintf("fits only at reduced ctx: %s @ %d (trained max)\n%s",
			res.Mode, res.Ctx, res.Summary())
	default:
		return res.Summary()
	}
}

// OOMAdvice turns a stored autofit projection into a runtime-CUDA-OOM
// hint (load or prefill): what happened, the largest ctx that would fit
// (when known), and the cheapest KV-cost reduction. 2–4 lines.
// ponytail: largest ctx comes from Result only when autofit refused at
// fit time; for a Fits result it recomputes nothing — fall back to
// res.Ctx/res.Mode advice.
func OOMAdvice(res autofit.Result) string {
	mode := res.Mode
	if mode == "" {
		mode = "Q4" // ladder refusals project at Q4 (autofit.refusalReason)
	}
	var lines []string
	switch {
	case res.Warning != "":
		lines = append(lines, "CUDA out of memory before the first token (prefill workspace): config too tight for current card state.")
	case !res.Fits:
		lines = append(lines, fmt.Sprintf("CUDA out of memory: no config fits the current card state (projected at %s).", mode))
	default:
		lines = append(lines, fmt.Sprintf("CUDA out of memory at ctx %d (%s) — the card holds more than it did at fit time.", res.Ctx, mode))
	}
	if res.LargestCtx > 0 {
		suffix := ""
		if res.Ctx > 0 {
			suffix = fmt.Sprintf(" (current: %d %s)", res.Ctx, mode)
		}
		lines = append(lines, fmt.Sprintf("largest ctx that would fit now: %d at %s%s — retry with --ctx %d",
			res.LargestCtx, mode, suffix, res.LargestCtx))
	} else if res.Ctx > 0 {
		lines = append(lines, fmt.Sprintf("reduce --ctx: retry below %d (current: %d %s) — smaller ctx lowers load and KV.",
			res.Ctx, res.Ctx, mode))
	} else {
		lines = append(lines, "reduce --ctx or pick a smaller quant: no context fits the current card state.")
	}
	if bpe, err := autofit.BytesPerElement(res.Mode); err == nil && bpe > 0.5 && res.ElemsPerTok > 0 && res.Ctx > 0 {
		freed := float64(res.Ctx) * float64(res.ElemsPerTok) * (bpe - 0.5) / float64(1<<20)
		lines = append(lines, fmt.Sprintf("or lower KV cost with --cache-mode Q4 (%s → Q4 frees ~%.0f MiB at %d) — current: %s",
			res.Mode, freed, res.Ctx, res.Mode))
	}
	return strings.Join(lines, "\n")
}

// Format renders the terminal report lines (Q3c: projection always
// printed before consent).
func (r Report) Format(gpu string) []string {
	symbols := map[string]string{StatusOK: "✓", StatusWarn: "⚠", StatusRefuse: "✗"}
	var out []string
	for _, c := range r.Checks {
		detail := c.Detail
		if c.Name == "fit" && c.Status == StatusOK && gpu != "" {
			detail = "on " + gpu + ": " + detail + " ✓"
		}
		lines := strings.Split(detail, "\n")
		out = append(out, fmt.Sprintf("gate: %-5s %s %s", c.Name, symbols[c.Status], lines[0]))
		for _, l := range lines[1:] {
			out = append(out, "      "+l)
		}
	}
	return out
}

func hasSuffix(paths []string, suffix string) bool {
	suffix = strings.ToLower(suffix)
	for _, p := range paths {
		if len(p) >= len(suffix) && strings.EqualFold(p[len(p)-len(suffix):], suffix) {
			return true
		}
	}
	return false
}
