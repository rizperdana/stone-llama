// Package preflight implements the A5 pre-download gate: architecture,
// quant, and fit checks run against metadata only — before any weight
// byte moves (ARCHITECTURE.md §4).
package preflight

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"sort"
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
	RepoSHA       string   // resolved revision (header probe identity)
	// FetchHeader reads a bounded byte prefix of one repo file so the
	// gate can inspect its safetensors header; nil = offline/local path
	// → the quant check degrades to an "unverified" warning.
	FetchHeader  func(path string) (io.ReadCloser, error)
	Spec         autofit.Spec
	WeightsBytes int64 // sum of .safetensors sizes in scope
	VRAMMiB      int
	GPUName      string
	Opts         autofit.Options
}

// Evaluate runs the three checks and computes the verdict. A refusal
// never short-circuits: the report always shows the full picture.
func Evaluate(in Input) Report {
	r := Report{Fit: autofit.Result{}}
	fitRes, fitErr := autofit.Fit(in.Spec, in.WeightsBytes, in.VRAMMiB, in.Opts)
	r.Fit = fitRes

	r.Checks = []Check{checkArch(in), checkQuant(in)}
	if in.Spec.MoE {
		r.Checks = append(r.Checks, checkMoE())
	}
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
	if method == "" {
		// No (or unreadable) quantization_config.json: exllamav3 conversion
		// also embeds quantization_config in config.json — ParseSpec reads it.
		method = strings.ToLower(strings.TrimSpace(in.Spec.QuantMethod))
	}
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
	case method == "exl2":
		return Check{Name: "quant", Status: StatusRefuse,
			Detail: "EXL2 quant (ExLlamaV2 format) — exllamav3 cannot load it; " + hint}
	case method != "" && method != "exl3":
		return Check{Name: "quant", Status: StatusRefuse,
			Detail: fmt.Sprintf("quant_method %q not exl3 — detected format is unsupported; %s", method, hint)}
	case gguf && !safetensors:
		return Check{Name: "quant", Status: StatusRefuse,
			Detail: "no EXL3 weights — this repo ships GGUF; use ollama with GGUF instead; " + hint}
	}

	// Safetensors present, nothing excluded the format: the verdict comes
	// from the header itself. The tensor-suffix group is the engine's own
	// EXL3 test (linear.py is_exl3_storage), quant-group dtypes must be in
	// the engine's convert_dtype set, and the sidecar/embedded
	// quantization_config — optional for exllamav3 — is corroboration
	// only: it can neither rescue a repo without the group nor be
	// required when the group is there.
	return checkHeader(in, method, hint)
}

// checkHeader renders the quant verdict from the safetensors header
// probe (see probeHeaders for bounds). Could-not-look degrades to a
// WARN, verified-not-EXL3 refuses, loadable EXL3 passes.
func checkHeader(in Input, method, hint string) Check {
	p := probeHeaders(in)
	unverified := func(why string) Check {
		return Check{Name: "quant", Status: StatusWarn,
			Detail: "safetensors header not fully readable (" + why + ") — EXL3 storage and quant dtypes are UNVERIFIED; " + hint}
	}
	switch {
	case p.read == 0:
		return unverified(p.reason)
	case p.storage && p.badDtype != "":
		return Check{Name: "quant", Status: StatusRefuse,
			Detail: fmt.Sprintf("quant tensor %s has dtype %s, which the installed exllamav3 cannot load "+
				"(convert_dtype supports I32,I64,I8,F8_E8M0,I16,F16,BF16,F32,F8_E4M3,U8 — the engine raises \"Unknown dtype\"): "+
				"refusing before the download so the failure costs KB, not GB; %s",
				p.badTensor, p.badDtype, hint)}
	case p.storage:
		return Check{Name: "quant", Status: StatusOK, Detail: "exl3"}
	case p.reason == "":
		// Every probed header read cleanly and none carries the group —
		// this is verified "not EXL3 storage", not a failure to look.
		if method == "exl3" {
			return Check{Name: "quant", Status: StatusRefuse,
				Detail: "quantization_config claims exl3, but the safetensors header has no EXL3 tensor group " +
					"(.trellis + .su/.suh + .sv/.svh) — the engine identifies EXL3 by that group, so this is not EXL3 storage; " + hint}
		}
		return Check{Name: "quant", Status: StatusRefuse,
			Detail: "no EXL3 tensor group (.trellis + .su/.suh + .sv/.svh) in the safetensors header — not EXL3 storage " +
				"(unquantized FP16 or an EXL2 conversion); " + hint}
	default:
		return unverified(p.reason + "; no EXL3 tensor group confirmed in the headers that did read")
	}
}

const (
	// HeaderReadBytes bounds one probe request: the 8-byte size prefix
	// plus at most 4 MiB of header JSON. Exported so pull's Range header
	// matches exactly what the gate will consume.
	HeaderReadBytes = 8 + 4<<20
	// maxHeaderFiles caps probe requests per gate run — a hundred-shard
	// repo must not turn `fit` into a crawl (≤ 4 bounded reads).
	maxHeaderFiles = 4
)

// headerProbe aggregates the safetensors header reads for one gate run.
type headerProbe struct {
	read      int    // headers parsed
	reason    string // first unreadable cause ("" = none)
	storage   bool   // engine EXL3 tensor-suffix group found
	badTensor string // quant-group tensor with an unloadable dtype
	badDtype  string
}

// probeHeaders reads up to maxHeaderFiles safetensors headers via the
// injected FetchHeader (one bounded ranged read each, closed after the
// header). Metadata only: no tensor data is ever requested or read.
func probeHeaders(in Input) headerProbe {
	var p headerProbe
	if in.FetchHeader == nil {
		p.reason = "no header reader wired (offline or local path)"
		return p
	}
	probed := 0
	for _, path := range in.RepoFiles {
		if !strings.HasSuffix(path, ".safetensors") {
			continue
		}
		if probed >= maxHeaderFiles {
			break
		}
		probed++
		var err error
		rc, ferr := in.FetchHeader(path)
		if ferr == nil {
			var names map[string]string
			names, err = readHeader(rc)
			if err == nil {
				p.read++
				if hasExl3Storage(names) {
					p.storage = true
				}
				if p.badTensor == "" {
					p.badTensor, p.badDtype = firstBadQuantDtype(names)
				}
			}
		} else {
			err = ferr
		}
		if err != nil && p.reason == "" {
			p.reason = path + ": " + err.Error()
		}
	}
	return p
}

// readHeader consumes one safetensors header: the 8-byte little-endian
// header_size — validated BEFORE any allocation, remote input is never
// trusted — then exactly that many JSON bytes, then closes the reader.
func readHeader(rc io.ReadCloser) (map[string]string, error) {
	defer rc.Close()
	var prefix [8]byte
	if _, err := io.ReadFull(rc, prefix[:]); err != nil {
		return nil, fmt.Errorf("read size prefix: %w", err)
	}
	size := int64(binary.LittleEndian.Uint64(prefix[:]))
	if size <= 0 || size > HeaderReadBytes-8 {
		return nil, fmt.Errorf("header_size %d outside allowed range (0, %d]", size, HeaderReadBytes-8)
	}
	buf := make([]byte, size)
	if _, err := io.ReadFull(rc, buf); err != nil {
		return nil, fmt.Errorf("read header JSON: %w", err)
	}
	var raw map[string]struct {
		Dtype string `json:"dtype"`
	}
	if err := json.Unmarshal(buf, &raw); err != nil {
		return nil, fmt.Errorf("parse header JSON: %w", err)
	}
	names := make(map[string]string, len(raw))
	for k, v := range raw {
		names[k] = v.Dtype
	}
	return names, nil
}

// hasExl3Storage mirrors the engine's own test: is_exl3_storage() at
// modules/linear.py:385 = has_tensor_group(key, [["sv","svh"],
// ["su","suh"],"trellis"]) with the semantics at loader/safetensors.py:
// 472 — all groups required, list entries alternatives. Applied at
// repo level: any module prefix carrying .trellis plus an su- and an
// sv- variant.
func hasExl3Storage(names map[string]string) bool {
	for name := range names {
		if !strings.HasSuffix(name, ".trellis") {
			continue
		}
		p := strings.TrimSuffix(name, ".trellis")
		_, su := names[p+".su"]
		_, suh := names[p+".suh"]
		_, sv := names[p+".sv"]
		_, svh := names[p+".svh"]
		if (su || suh) && (sv || svh) {
			return true
		}
	}
	return false
}

// quantDtypeSuffixes are the EXL3 quant-storage tensors load_exl3
// actually reads (modules/linear.py:391-406). Inert buffers the loader
// never reads are deliberately out of scope — the engine itself only
// structurally validates them (loader/safetensors.py:52-54).
var quantDtypeSuffixes = []string{".trellis", ".su", ".suh", ".sv", ".svh", ".mcg", ".mul1"}

// engineLoadable is convert_dtype's supported set
// (loader/safetensors.py:32-43); any other tag raises
// ValueError("Unknown dtype ...") at load time.
var engineLoadable = map[string]bool{
	"I32": true, "I64": true, "I8": true, "F8_E8M0": true, "I16": true,
	"F16": true, "BF16": true, "F32": true, "F8_E4M3": true, "U8": true,
}

// firstBadQuantDtype returns the first quant-group tensor (sorted for a
// deterministic message) whose dtype the installed engine cannot load.
func firstBadQuantDtype(names map[string]string) (tensor, dtype string) {
	keys := make([]string, 0, len(names))
	for k := range names {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, name := range keys {
		for _, suf := range quantDtypeSuffixes {
			if !strings.HasSuffix(name, suf) {
				continue
			}
			if dt := names[name]; dt != "" && !engineLoadable[dt] {
				return name, dt
			}
			break // one suffix match per tensor is enough
		}
	}
	return "", ""
}

// checkMoE warns when the config declares expert-parallel MoE fields.
// Whether the conversion ships .mul1 (per-expert shards that enable CPU
// offload) cannot be confirmed from config.json: the gate's header
// probe lists tensor names but does not gate on .mul1 — a converter may
// omit it without any other metadata changing. Warn-only by design —
// refuse would block genuine .mul1 repos (e.g. Terra3312/GLM-5.3-Flash-
// EXL3-4bpw-MUL1) that parse identically here.
func checkMoE() Check {
	return Check{Name: "moe", Status: StatusWarn,
		Detail: "MoE config detected (n_routed_experts / num_local_experts / num_experts / moe_intermediate_size in config.json): " +
			"whether this conversion ships .mul1 expert shards is NOT gated — tensor names appear in the safetensors header, " +
			"but their presence is not checked; without .mul1, CPU expert offload is silently disabled and every expert must fit on the GPU, " +
			"so a quant-slim model can still OOM — inspect the repo's converter flags or README before pulling"}
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

// RefusalSummary names the FIRST failing gate so the top-level
// "refused" line never misattributes a format/architecture failure to
// VRAM. Checks are ordered arch → quant → fit (most fundamental first);
// detail lines stay untouched in the report above.
func (r Report) RefusalSummary() string {
	for _, c := range r.Checks {
		if c.Status != StatusRefuse {
			continue
		}
		switch c.Name {
		case "quant":
			return "unsupported quant format: " + firstClause(c.Detail)
		case "arch":
			return "unsupported architecture: " + firstClause(c.Detail)
		default: // fit
			return "no context fits this model in VRAM"
		}
	}
	return ""
}

// firstClause cuts a check detail at the first clause/line break: the
// cause, not the remedy ("… — exllamav3 cannot load it; look for -exl3…").
func firstClause(s string) string {
	if i := strings.IndexAny(s, "\n;"); i >= 0 {
		s = s[:i]
	}
	if i := strings.Index(s, " — "); i >= 0 {
		s = s[:i]
	}
	return strings.TrimSpace(s)
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
	if bpe, err := autofit.BytesPerElement(res.Mode); err == nil && res.ElemsPerTok > 0 && res.Ctx > 0 {
		if q4, qerr := autofit.BytesPerElement("Q4"); qerr == nil && bpe > q4 {
			freed := float64(res.Ctx) * float64(res.ElemsPerTok) * (bpe - q4) / float64(1<<20)
			lines = append(lines, fmt.Sprintf("or lower KV cost with --cache-mode Q4 (%s → Q4 frees ~%.0f MiB at %d) — current: %s",
				res.Mode, freed, res.Ctx, res.Mode))
		}
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
