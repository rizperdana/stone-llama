// Package pull implements `stone-llama pull`: HF metadata resolution,
// the A5 pre-download gate, consent, resumable download with sha256
// verification, and an atomic staging→final rename (ARCHITECTURE.md §4).
package pull

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/rizperdana/stone-llama/internal/autofit"
	"github.com/rizperdana/stone-llama/internal/fslock"
	"github.com/rizperdana/stone-llama/internal/hf"
	"github.com/rizperdana/stone-llama/internal/preflight"
	"github.com/rizperdana/stone-llama/internal/store"
)

type Options struct {
	ModelsDir   string
	Ref         string // "name", "owner/repo", or "owner/repo:quant"
	Force       bool
	Yes         bool // non-interactive: accept warnings + consent
	Quiet       bool // suppress progress and informational output
	Out         io.Writer
	Stdin       io.Reader
	Interactive bool // stdin is a TTY (prompts allowed)
	TTY         bool // Out supports carriage-return progress
	Token       string
	VRAMMiB     int
	GPUName     string
	Autofit     autofit.Options
	BaseURL     string // test seam; "" = https://huggingface.co
	Now         func() time.Time
	FreeBytes   func(path string) (int64, error)
	RetryDelay  func(attempt int) time.Duration
}

type Result struct {
	Name    string
	Verdict store.Verdict
	Files   int
	Bytes   int64
}

func (o *Options) defaults() {
	if o.Out == nil {
		o.Out = io.Discard
	}
	if o.Stdin == nil {
		o.Stdin = strings.NewReader("")
	}
	if o.Now == nil {
		o.Now = time.Now
	}
	if o.FreeBytes == nil {
		o.FreeBytes = defaultFreeBytes
	}
	if o.RetryDelay == nil {
		o.RetryDelay = func(int) time.Duration { return 2 * time.Second }
	}
}

// Run executes the full pull. It returns an error for every negative
// path; staging is left in place on interruption so the next run resumes.
func Run(opts Options) (Result, error) {
	opts.defaults()
	if opts.Ref == "" {
		return Result{}, errors.New("no model specified (usage: stone-llama pull <model>[:quant])")
	}
	_, explicitTag, _ := strings.Cut(opts.Ref, ":")
	in := bufio.NewReader(opts.Stdin)

	if err := os.MkdirAll(opts.ModelsDir, 0o700); err != nil {
		return Result{}, err
	}
	lock, err := fslock.TryAcquire(filepath.Join(opts.ModelsDir, ".pull.lock"))
	if err != nil {
		return Result{}, fmt.Errorf("another stone-llama process holds the models lock: %w", err)
	}
	defer lock.Release()

	client := hf.NewClient(opts.BaseURL, opts.Token)
	repo, err := resolveRepo(client, opts, in)
	if err != nil {
		return Result{}, err
	}

	scope, err := pickScope(repo, explicitTag, opts)
	if err != nil {
		return Result{}, err
	}
	if scope != "" {
		if err := validateScopeName(scope); err != nil {
			return Result{}, err
		}
	}

	configPath, err := pickConfig(repo, scope)
	if err != nil {
		return Result{}, err
	}
	cfgBytes, err := client.FetchFile(repo.ID, repo.SHA, configPath)
	if err != nil {
		return Result{}, fmt.Errorf("fetch %s: %w", configPath, err)
	}
	spec, err := autofit.ParseSpec(cfgBytes)
	if err != nil {
		return Result{}, err
	}

	quantMethod, quantLabel := "", ""
	quantPath, hasQuantCfg := findQuantCfg(repo, scope)
	if hasQuantCfg {
		qb, err := client.FetchFile(repo.ID, repo.SHA, quantPath)
		if err == nil {
			quantMethod, quantLabel = parseQuantCfg(qb)
		}
	}
	if quantLabel == "" {
		quantLabel = scope
	}

	var paths []string
	for _, f := range repo.Files {
		paths = append(paths, f.Path)
	}
	weights := scopeWeights(repo, scope)
	report := preflight.Evaluate(preflight.Input{
		Architectures: spec.Architectures,
		QuantMethod:   quantMethod,
		HasQuantCfg:   hasQuantCfg,
		RepoFiles:     paths,
		Spec:          spec,
		WeightsBytes:  weights,
		VRAMMiB:       opts.VRAMMiB,
		GPUName:       opts.GPUName,
		Opts:          opts.Autofit,
	})
	if !opts.Quiet {
		for _, l := range report.Format(opts.GPUName) {
			fmt.Fprintln(opts.Out, l)
		}
	}
	if report.Refused {
		return Result{}, errors.New("model refused by pre-download gate — see the report above")
	}

	files := includedFiles(repo, scope)
	var total int64
	for _, f := range files {
		total += f.Size
	}

	free, err := opts.FreeBytes(opts.ModelsDir)
	if err != nil {
		return Result{}, fmt.Errorf("check free disk: %w", err)
	}
	if need := total * 105 / 100; need > free {
		return Result{}, fmt.Errorf("not enough disk: need %s (total + 5%%), have %s free",
			HumanBytes(need), HumanBytes(free))
	}

	name := modelName(repo.ID, explicitTag, scope)
	if !opts.Yes {
		if !opts.Quiet {
			if report.Warned {
				fmt.Fprintln(opts.Out, "warnings above — review them before continuing")
			}
			fmt.Fprintf(opts.Out, "pulling %s → %s: %d files, %s (sha256-verified), %s free\n",
				repo.ID, name, len(files), HumanBytes(total), HumanBytes(free))
		}
		if !opts.Interactive {
			return Result{}, errors.New("non-interactive pull requires --yes to confirm the download")
		}
		fmt.Fprint(opts.Out, "Proceed? [y/N] ")
		if !confirm(in) {
			return Result{}, errors.New("aborted")
		}
	}

	final := filepath.Join(opts.ModelsDir, name)
	staging := filepath.Join(opts.ModelsDir, "."+name+".staging")
	if !opts.Force {
		if _, err := os.Stat(final); err == nil {
			return Result{}, fmt.Errorf("model %q already exists (use --force to re-pull)", name)
		}
	}
	if opts.Force {
		os.RemoveAll(staging)
	}
	if err := os.MkdirAll(staging, 0o755); err != nil {
		return Result{}, err
	}

	prog := &progress{out: opts.Out, enabled: opts.TTY && !opts.Quiet}
	var entries []store.FileEntry
	for i, f := range files {
		label := fmt.Sprintf("[%d/%d] %s", i+1, len(files), filepath.Base(f.Path))
		fileURL := client.ResolveURL(repo.ID, repo.SHA, f.Path)
		sum, err := fetchWithRetry(client, opts, staging, fileURL, f, label, prog)
		if err != nil {
			prog.finish()
			return Result{}, fmt.Errorf("%s: %w — staging kept, re-run pull to resume", f.Path, err)
		}
		size := f.Size
		if size == 0 {
			if st, serr := os.Stat(filepath.Join(staging, f.Path)); serr == nil {
				size = st.Size()
			}
		}
		entries = append(entries, store.FileEntry{Path: f.Path, Size: size, SHA256: sum})
	}
	prog.finish()

	manifest := &store.Manifest{
		RepoID:   repo.ID,
		Revision: repo.SHA,
		Quant:    quantLabel,
		PulledAt: opts.Now(),
		Files:    entries,
		Verdict:  &report.Verdict,
	}
	if err := store.SaveManifest(staging, manifest); err != nil {
		return Result{}, err
	}

	// Swap into place only after a complete, verified staging directory.
	if _, err := os.Stat(final); err == nil {
		if err := os.RemoveAll(final); err != nil {
			return Result{}, fmt.Errorf("replace existing model: %w", err)
		}
	}
	if err := os.Rename(staging, final); err != nil {
		return Result{}, fmt.Errorf("finalize model: %w", err)
	}

	if !opts.Quiet {
		fmt.Fprintf(opts.Out, "pulled %s (%s) — verdict %s:%d %s\n",
			name, HumanBytes(total), report.Verdict.Status, report.Verdict.MaxCtx, report.Verdict.CacheMode)
	}
	return Result{Name: name, Verdict: report.Verdict, Files: len(files), Bytes: total}, nil
}

// resolveRepo turns opts.Ref into repo metadata: direct owner/repo, or a
// bare-name search filtered to EXL3 candidates (interactive picker when
// several match).
func resolveRepo(client *hf.Client, opts Options, in *bufio.Reader) (hf.Repo, error) {
	id, _, _ := strings.Cut(opts.Ref, ":")
	if !strings.Contains(id, "/") {
		ids, err := client.Search(id)
		if err != nil {
			return hf.Repo{}, fmt.Errorf("search %q: %w", id, err)
		}
		var matches []string
		for _, cand := range ids {
			if strings.Contains(strings.ToLower(cand), "exl3") {
				matches = append(matches, cand)
			}
		}
		switch len(matches) {
		case 0:
			return hf.Repo{}, fmt.Errorf(
				"no EXL3 repo found for %q — use the exact id (owner/repo); if the model only ships GGUF, use ollama instead", id)
		case 1:
			id = matches[0]
		default:
			if !opts.Interactive || opts.Yes {
				return hf.Repo{}, fmt.Errorf(
					"several EXL3 repos match %q: %s — pick one with owner/repo",
					id, strings.Join(limit(matches, 5), ", "))
			}
			for i, m := range matches {
				fmt.Fprintf(opts.Out, "  %d) %s\n", i+1, m)
			}
			fmt.Fprintf(opts.Out, "Choose 1-%d: ", len(matches))
			choice, err := readLine(in)
			if err != nil {
				return hf.Repo{}, err
			}
			n, err := strconv.Atoi(strings.TrimSpace(choice))
			if err != nil || n < 1 || n > len(matches) {
				return hf.Repo{}, fmt.Errorf("invalid choice %q", choice)
			}
			id = matches[n-1]
		}
	}

	repo, err := client.RepoMeta(id)
	if err != nil {
		switch {
		case errors.Is(err, hf.ErrGated):
			return hf.Repo{}, fmt.Errorf("%s is gated — run 'stone-llama login' or set HF_TOKEN", id)
		case errors.Is(err, hf.ErrNotFound):
			return hf.Repo{}, fmt.Errorf("repo %q not found on HuggingFace — check the id (owner/repo)", id)
		default:
			return hf.Repo{}, err
		}
	}
	if repo.Gated && opts.Token == "" {
		return hf.Repo{}, fmt.Errorf("%s is gated — run 'stone-llama login' or set HF_TOKEN", id)
	}
	return repo, nil
}

// pickScope selects the weight directory: explicit tag must match one;
// otherwise a unique weight dir is taken implicitly, and multiple quants
// require the tag.
func pickScope(repo hf.Repo, tag string, opts Options) (string, error) {
	dirs := weightDirs(repo)
	contains := func(d string) bool {
		for _, x := range dirs {
			if x == d {
				return true
			}
		}
		return false
	}
	if tag != "" {
		if contains(tag) {
			return tag, nil
		}
		if len(dirs) == 0 {
			return tag, nil // no weights at all — gate refuses later
		}
		return "", fmt.Errorf("repo %s has no quant %q (available: %s)",
			repo.ID, tag, strings.Join(displayDirs(dirs), ", "))
	}
	switch len(dirs) {
	case 0:
		return "", nil
	case 1:
		return dirs[0], nil
	default:
		return "", fmt.Errorf("repo %s has multiple quants (%s) — pull %s:<quant>",
			repo.ID, strings.Join(displayDirs(dirs), ", "), repo.ID)
	}
}

func weightDirs(repo hf.Repo) []string {
	seen := map[string]bool{}
	var dirs []string
	for _, f := range repo.Files {
		if hasSuffix(f.Path, ".safetensors") {
			d := dirOf(f.Path)
			if !seen[d] {
				seen[d] = true
				dirs = append(dirs, d)
			}
		}
	}
	sort.Strings(dirs)
	return dirs
}

func displayDirs(dirs []string) []string {
	out := make([]string, len(dirs))
	for i, d := range dirs {
		if d == "" {
			out[i] = "root"
		} else {
			out[i] = d
		}
	}
	return out
}

func validateScopeName(scope string) error {
	for _, r := range scope {
		ok := r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' || r == '_' || r == '.'
		if !ok {
			return fmt.Errorf("quant dir %q contains unsupported characters", scope)
		}
	}
	return nil
}

func pickConfig(repo hf.Repo, scope string) (string, error) {
	var candidates []string
	if scope != "" {
		candidates = append(candidates, scope+"/config.json")
	}
	candidates = append(candidates, "config.json")
	for _, c := range candidates {
		for _, f := range repo.Files {
			if f.Path == c {
				return c, nil
			}
		}
	}
	return "", fmt.Errorf("repo %s has no config.json — not an EXL3 model layout", repo.ID)
}

func findQuantCfg(repo hf.Repo, scope string) (path string, found bool) {
	checked := []string{}
	if scope != "" {
		checked = append(checked, scope+"/quantization_config.json")
	}
	checked = append(checked, "quantization_config.json")
	for _, c := range checked {
		for _, f := range repo.Files {
			if f.Path == c {
				return c, true
			}
		}
	}
	return "", false
}

func parseQuantCfg(data []byte) (method, label string) {
	var qc struct {
		Method string  `json:"quant_method"`
		Bits   float64 `json:"bits"`
	}
	if err := json.Unmarshal(data, &qc); err != nil {
		return "", ""
	}
	method = qc.Method
	if qc.Bits > 0 {
		label = strconv.FormatFloat(qc.Bits, 'f', -1, 64) + "bpw"
	}
	return method, label
}

func scopeWeights(repo hf.Repo, scope string) int64 {
	var sum int64
	for _, f := range repo.Files {
		if hasSuffix(f.Path, ".safetensors") && dirOf(f.Path) == scope {
			sum += f.Size
		}
	}
	return sum
}

func includedFiles(repo hf.Repo, scope string) []hf.File {
	var out []hf.File
	for _, f := range repo.Files {
		p := f.Path
		base := filepath.Base(p)
		if base == "README.md" || base == ".gitattributes" {
			continue
		}
		d := dirOf(p)
		switch {
		case hasSuffix(p, ".safetensors"):
			if d == scope {
				out = append(out, f)
			}
		case scope == "":
			if d == "" {
				out = append(out, f)
			}
		default:
			if d == "" || d == scope {
				out = append(out, f)
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out
}

func modelName(repoID, tag, scope string) string {
	base := repoID[strings.LastIndex(repoID, "/")+1:]
	if tag != "" && scope != "" {
		return base + "-" + scope
	}
	return base
}

func confirm(in *bufio.Reader) bool {
	line, err := readLine(in)
	if err != nil {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(line)) {
	case "y", "yes":
		return true
	}
	return false
}

func readLine(in *bufio.Reader) (string, error) {
	line, err := in.ReadString('\n')
	if err != nil && line == "" {
		return "", errors.New("aborted: no input")
	}
	return line, nil
}

func limit(items []string, n int) []string {
	if len(items) <= n {
		return items
	}
	return items[:n]
}

func dirOf(path string) string {
	if i := strings.LastIndex(path, "/"); i >= 0 {
		return path[:i]
	}
	return ""
}

func hasSuffix(path, suffix string) bool {
	return len(path) >= len(suffix) && strings.EqualFold(path[len(path)-len(suffix):], suffix)
}

// HumanBytes formats a byte count for humans (two decimals above B).
func HumanBytes(n int64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.2f GiB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.2f MiB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.2f KiB", float64(n)/(1<<10))
	default:
		return fmt.Sprintf("%d B", n)
	}
}
