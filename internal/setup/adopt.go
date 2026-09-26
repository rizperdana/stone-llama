// Package setup's adoption layer reuses an already-present Python runtime
// (a user's TabbyAPI venv + checkout) instead of provisioning one.
//
// Design: internal/sl-adopt/adoption-design.md §4-§6.
//
// The validation gate (V1-V5) is the ONLY thing that makes a candidate trusted
// — a signal (flag/env/VIRTUAL_ENV/proc/record) is just a pointer. Every public
// entry point that trusts a runtime MUST run Validate first.
package setup

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Candidate points at an environment someone claims is reusable. Never a
// truth claim: Validate is the only trust anchor.
type Candidate struct {
	Venv     string // path to venv root (holds pyvenv.cfg, bin/python)
	Checkout string // TabbyAPI checkout ("" if unknown)
	Source   string // "flag", "env", "virtualenv", "proc", "journal", "provisioned"
	Running  bool   // running-process probe found a live server (attach hint)
	Port     uint16 // 0 if unknown
}

// Diff is one found-vs-needed package row in message (b).
type Diff struct {
	Name  string
	Found string // "(missing)" when absent
	Need  string
}

// Report is the validated result of a Candidate. OK is true only when every
// V1-V5 gate passed. Unverifiable is true when a check could not run at all
// (a clear "could not verify", never a pass and never a false refusal —
// behavioural requirement 4).
type Report struct {
	Candidate     Candidate
	OK            bool
	Unverifiable  bool
	Failed        string // "V1".."V5", "" when OK
	Reason        string // per-check phrasing (§5 message-(b) swap)
	Python        string // "3.12.13"
	PinsMatched   int
	PinsNeeded    int
	Diff          []Diff // first 10 shown + "+N more" in message (b)
	Extras        int    // installed-but-unpinned count (informational)
	Torch         string // "2.11.0+cu130"
	TorchCUDA     string // "13.0"
	Exllamav3     string // "1.5.1+cu132.torch2.11.0"
	ImportsOK     bool
	CUDAAvailable bool
	// V6 informational
	StartPy        bool
	CheckoutCommit string
	CheckoutMatch  bool
	CheckoutDirty  bool
}

// ErrNotAdoptable is returned by Validate when a V-check fails. The Report is
// still attached (for the diff table / message-(b) wording).
type ErrNotAdoptable struct{ Report Report }

func (e ErrNotAdoptable) Error() string { return e.Report.Reason }

// ErrUnverifiable is returned when a candidate could not be checked at all.
type ErrUnverifiable struct{ Report Report }

func (e ErrUnverifiable) Error() string { return e.Report.Reason }

// adoptionRecord is the persisted, re-validated adoption pointer (§4.3-§4.4).
type adoptionRecord struct {
	Venv        string        `json:"venv_path"`
	Checkout    string        `json:"checkout_path"`
	Extra       string        `json:"extra"`
	LockDigest  string        `json:"lock_digest"`
	ValidatedAt string        `json:"validated_at"`
	Python      string        `json:"python"`
	PinsMatched int           `json:"pins_matched"`
	PinsNeeded  int           `json:"pins_needed"`
	Torch       string        `json:"torch"`
	TorchCUDA   string        `json:"torch_cuda"`
	Exllamav3   string        `json:"exllamav3"`
	ImportsOK   bool          `json:"imports_ok"`
	CUDA        bool          `json:"cuda_available"`
	Invalid     *adoptInvalid `json:"invalid,omitempty"`
}

type adoptInvalid struct {
	Check  string `json:"check"`
	Reason string `json:"reason"`
	At     string `json:"at"`
}

// loadAdoption reads a stored record (os.ErrNotExist when nothing stored).
func loadAdoption(p string) (adoptionRecord, error) {
	var rec adoptionRecord
	b, err := os.ReadFile(p)
	if err != nil {
		return rec, err
	}
	if err := json.Unmarshal(b, &rec); err != nil {
		return rec, fmt.Errorf("setup: parse adoption %s: %w", p, err)
	}
	return rec, nil
}

// adoptPath is the recorded-adoption pointer (§4.3).
func adoptPath(runtimeDir string) string {
	return filepath.Join(runtimeDir, "adoption.json")
}

// saveAdoption writes the record atomically (temp + rename), mirroring
// saveJournal (§4.4).
func saveAdoption(p string, rec adoptionRecord) error {
	b, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		return err
	}
	return writeAtomic(p, b)
}

// invalidateAdoption rewrites a stale record atomically so it never looks
// green again ("a stale adoption.json invalidated atomically, never left green").
func invalidateAdoption(p string, check, reason string) error {
	var rec adoptionRecord
	if b, err := os.ReadFile(p); err == nil {
		_ = json.Unmarshal(b, &rec)
	}
	rec.Invalid = &adoptInvalid{Check: check, Reason: reason, At: time.Now().UTC().Format(time.RFC3339)}
	return saveAdoption(p, rec)
}

// writeAtomic writes data to p via temp+rename in p's directory, mirroring
// saveJournal (same filesystem guaranteed: temp created in p's directory).
func writeAtomic(p string, data []byte) error {
	tmp, err := os.CreateTemp(filepath.Dir(p), ".adoption-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return err
	}
	return os.Rename(tmpName, p)
}

// --- interpreter probes (read-only; PYTHONDONTWRITEBYTECODE keeps us from
// writing __pycache__ into the candidate — §6 "never write into a candidate") ---

// identity is the V2+V3 probe output: version, venv-ness, and the full dist
// closure via importlib.metadata (stdlib; names ≠ import names — paired with
// V4's real import below).
type identity struct {
	Python string     `json:"python"`
	Venv   bool       `json:"venv"`
	Dists  [][]string `json:"dists"`
}

// importResult is the V4+V5 probe: real exllamav3 + torch + CUDA state.
type importResult struct {
	Torch struct {
		V      string `json:"v"`
		Cuda   string `json:"cuda"`
		CudaOK bool   `json:"cuda_ok"`
	} `json:"torch"`
	TorchError     string `json:"torch_error,omitempty"`
	Exllamav3      string `json:"exllamav3,omitempty"`
	Exllamav3Error string `json:"exllamav3_error,omitempty"`
}

// identityCode is the V2+V3 probe (one read-only interpreter round-trip).
const identityCode = "import sys,json\nfrom importlib import metadata\nds=[]\nfor d in metadata.distributions():\n    md=d.metadata\n    if not md: continue\n    n=md['Name']; v=d.version\n    if n: ds.append([n,v])\nprint(json.dumps({'python':'%d.%d.%d'%sys.version_info[:3],'venv':sys.prefix!=sys.base_prefix,'dists':ds}))"

// importCode is the V4+V5 probe (one torch import; exllamav3 + CUDA too).
const importCode = "import json\ntry:\n    import torch;o={'torch':{'v':torch.__version__,'cuda':getattr(torch.version,'cuda',None),'cuda_ok':bool(torch.cuda.is_available())}}\nexcept Exception as e:o={'torch_error':repr(e)}\ntry:\n    import exllamav3;o['exllamav3']='ok'\nexcept Exception as e:o['exllamav3_error']=repr(e)\nprint(json.dumps(o))"

// probeRun runs a -c script against the candidate interpreter. The candidate
// is only ever read.
func probeRun(venv, code string, timeout time.Duration) ([]byte, error) {
	python := venvPython(venv)
	if python == "" {
		return nil, errors.New("venv: bin/python not found")
	}
	ctx := context.Background()
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}
	cmd := exec.CommandContext(ctx, python, "-c", code)
	cmd.Env = append(stripEnv(os.Environ(), "PYTHONPATH"), "PYTHONDONTWRITEBYTECODE=1")
	out, err := cmd.CombinedOutput()
	return bytes.TrimSpace(out), err
}

func venvPython(venv string) string {
	if venv == "" {
		return ""
	}
	return filepath.Join(venv, "bin", "python")
}

func stripEnv(env []string, key string) []string {
	pre := key + "="
	out := make([]string, 0, len(env))
	for _, kv := range env {
		if !strings.HasPrefix(kv, pre) {
			out = append(out, kv)
		}
	}
	return out
}

// Validate runs the read-only V1-V5 gate (V6 informational) on a candidate.
// It never writes inside the candidate. nil means adoptable; ErrNotAdoptable
// carries the failing check + diff for message (b); ErrUnverifiable means a
// check could not be run (degrade honestly — requirement 4).
func Validate(cand Candidate, lockText, extra string) (Report, error) {
	rep := Report{Candidate: cand}

	// V1: venv identity (venv files + interpreter resolves to a real binary).
	if err := v1(cand); err != nil {
		rep.Failed = "V1"
		if errors.Is(err, errUnverifiable) {
			rep.Unverifiable = true
			rep.Reason = "could not verify: " + err.Error()
			return rep, &ErrUnverifiable{Report: rep}
		}
		rep.Reason = "the venv or its base Python is gone"
		return rep, &ErrNotAdoptable{Report: rep}
	}

	// V2 + V3: one interpreter round-trip (≈0.3 s measured in §1.2).
	raw, err := probeRun(cand.Venv, identityCode, 15*time.Second)
	if err != nil {
		rep.Failed, rep.Unverifiable = "V2", true
		rep.Reason = "could not verify interpreter"
		return rep, &ErrUnverifiable{Report: rep}
	}
	var id identity
	if err := json.Unmarshal(raw, &id); err != nil {
		rep.Failed, rep.Unverifiable = "V2", true
		rep.Reason = fmt.Sprintf("could not verify: identity probe returned non-JSON (%s)", trunc(string(raw)))
		return rep, &ErrUnverifiable{Report: rep}
	}
	rep.Python = id.Python
	if !specSatisfied(rep.Python, pythonSpec()) {
		rep.Failed = "V2"
		rep.Reason = fmt.Sprintf("found Python %s, wheels are cp312", rep.Python)
		return rep, &ErrNotAdoptable{Report: rep}
	}
	if !id.Venv {
		rep.Failed = "V1"
		rep.Reason = "the venv or its base Python is gone (not a virtual environment)"
		return rep, &ErrNotAdoptable{Report: rep}
	}

	pins, err := parsePins(lockText)
	if err != nil {
		rep.Failed = "V3"
		rep.Unverifiable = true
		rep.Reason = "could not verify lock: " + err.Error()
		return rep, &ErrUnverifiable{Report: rep}
	}
	rep.PinsNeeded = len(pins)
	matched, diff, extras := comparePins(pins, id.Dists)
	rep.PinsMatched = matched
	rep.Diff = diff
	rep.Extras = extras
	if len(diff) > 0 {
		rep.Failed = "V3"
		rep.Reason = "locked-package equality (" + pinFailPhrase(diff, len(pins)) + ")"
		return rep, &ErrNotAdoptable{Report: rep}
	}

	// V4 + V5: one torch import (the expensive hop — combined to save time).
	raw, err = probeRun(cand.Venv, importCode, 120*time.Second)
	if err != nil {
		rep.Failed, rep.Unverifiable = "V4", true
		rep.Reason = fmt.Sprintf("could not verify: import probe failed to run (%s)", trunc(err.Error()))
		return rep, &ErrUnverifiable{Report: rep}
	}
	var ip importResult
	if err := json.Unmarshal(raw, &ip); err != nil {
		rep.Failed, rep.Unverifiable = "V4", true
		rep.Reason = fmt.Sprintf("could not verify: import probe returned non-JSON (%s)", trunc(string(raw)))
		return rep, &ErrUnverifiable{Report: rep}
	}
	if ip.Exllamav3Error != "" {
		rep.Failed = "V4"
		rep.Reason = "import exllamav3 — " + trunc(ip.Exllamav3Error)
		return rep, &ErrNotAdoptable{Report: rep}
	}
	if ip.TorchError != "" {
		rep.Failed = "V4"
		rep.Reason = "import torch — " + trunc(ip.TorchError)
		return rep, &ErrNotAdoptable{Report: rep}
	}
	rep.Torch = distVersion(id.Dists, "torch")
	rep.Exllamav3 = distVersion(id.Dists, "exllamav3")
	rep.ImportsOK = true
	rep.TorchCUDA = ip.Torch.Cuda
	if ip.Torch.Cuda == "" || ip.Torch.Cuda == "None" {
		rep.Failed = "V5"
		rep.Reason = fmt.Sprintf("CUDA generation — torch reports no CUDA (CPU-only wheel); the %s extra needs CUDA", extra)
		return rep, &ErrNotAdoptable{Report: rep}
	}
	wantMajor := cudaMajor(extra)
	gotMajor, _ := strconv.Atoi(strings.Split(ip.Torch.Cuda, ".")[0])
	if gotMajor != wantMajor {
		rep.Failed = "V5"
		rep.Reason = fmt.Sprintf("CUDA generation — found CUDA %s, this machine selects the %s extra (CUDA %d)", ip.Torch.Cuda, extra, wantMajor)
		return rep, &ErrNotAdoptable{Report: rep}
	}
	if !ip.Torch.CudaOK {
		rep.Failed = "V5"
		rep.Reason = "CUDA init — torch.cuda.is_available() is false (driver too old for these wheels)"
		return rep, &ErrNotAdoptable{Report: rep}
	}
	rep.CUDAAvailable = true

	// V6 informational.
	v6(&rep, cand, id.Dists)
	rep.OK = true
	return rep, nil
}

// distVersion returns the installed version for name from probe dists.
func distVersion(dists [][]string, name string) string {
	for _, d := range dists {
		if len(d) == 2 && pinName(d[0]) == pinName(name) {
			return d[1]
		}
	}
	return ""
}

func v6(rep *Report, cand Candidate, dists [][]string) {
	if cand.Checkout == "" {
		return
	}
	if fi, err := os.Stat(filepath.Join(cand.Checkout, "start.py")); err == nil && !fi.IsDir() {
		rep.StartPy = true
	}
	if _, err := exec.LookPath("git"); err != nil {
		return
	}
	if out, err := exec.Command("git", "--no-optional-locks", "-C", cand.Checkout, "rev-parse", "HEAD").CombinedOutput(); err == nil {
		rep.CheckoutCommit = strings.TrimSpace(string(out))
	}
	if out, err := exec.Command("git", "--no-optional-locks", "-C", cand.Checkout, "status", "--porcelain").CombinedOutput(); err == nil {
		rep.CheckoutDirty = strings.TrimSpace(string(out)) != ""
	}
	if rep.CheckoutCommit != "" {
		rep.CheckoutMatch = rep.CheckoutCommit == tabbyPin()
	}
}

// --- V1 ---

var pyvenvRE = regexp.MustCompile(`(?m)^(version_info|home|implementation)\s*=`)

// errUnverifiable marks a V1 failure that is "could not verify" vs "gone".
var errUnverifiable = errors.New("could not verify")

type unverifiableErr struct{ msg string }

func newUnverifiable(s string) *unverifiableErr { return &unverifiableErr{msg: s} }
func (e *unverifiableErr) Error() string        { return e.msg }

// v1 checks the venv files are present and the interpreter resolves to a real
// binary (symlink resolves, pyvenv.cfg readable and well-formed).
func v1(cand Candidate) error {
	py := venvPython(cand.Venv)
	if py == "" {
		return newUnverifiable("no venv path")
	}
	fi, err := os.Lstat(py)
	if err != nil {
		return errors.New("bin/python is missing (the venv or its base Python is gone)")
	}
	if fi.Mode().Perm()&0o111 == 0 {
		return newUnverifiable("bin/python is not executable")
	}
	if fi.Mode()&os.ModeSymlink != 0 {
		if _, err := os.Stat(py); err != nil {
			return newUnverifiable("interpreter symlink points at a missing target")
		}
	}
	real, err := filepath.EvalSymlinks(py)
	if err != nil || !isFile(real) {
		return newUnverifiable("interpreter symlink does not resolve to a real interpreter")
	}
	cfg := filepath.Join(cand.Venv, "pyvenv.cfg")
	b, err := os.ReadFile(cfg)
	if err != nil {
		return newUnverifiable("pyvenv.cfg unreadable")
	}
	if !pyvenvRE.Match(b) {
		return newUnverifiable("pyvenv.cfg present but malformed")
	}
	return nil
}

func isFile(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && !fi.IsDir()
}

// --- lock / pin helpers (read-only against the embedded lock) ---

// pythonSpec returns the pinned interpreter spec (e.g. "3.12").
func pythonSpec() string {
	l, err := loadLock()
	if err != nil {
		return "3.12"
	}
	return l.Python.Spec
}

// tabbyPin returns the pinned TabbyAPI commit ("" on error).
func tabbyPin() string {
	l, err := loadLock()
	if err != nil {
		return ""
	}
	return l.TabbyAPI.Commit
}

// requirementsLockText returns the hash lock for extra (embedded).
func requirementsLockText(extra string) (string, error) {
	l, err := loadLock()
	if err != nil {
		return "", err
	}
	name, err := l.requirementsFile(extra)
	if err != nil {
		return "", err
	}
	b, err := lockFS.ReadFile(name)
	if err != nil {
		return "", fmt.Errorf("setup: read %s: %w", name, err)
	}
	return string(b), nil
}

// lockDigest is sha256(lockText) — recorded so a lock change invalidates the
// saved adoption.
func lockDigest(text string) string {
	h := sha256.Sum256([]byte(text))
	return "sha256:" + hex.EncodeToString(h[:])
}

// specSatisfied: py "3.12.13" matches spec "3.12" when major.minor equal.
func specSatisfied(py, spec string) bool {
	p := strings.Split(py, ".")
	s := strings.Split(spec, ".")
	if len(p) < 2 || len(s) < 2 {
		return false
	}
	return p[0] == s[0] && p[1] == s[1]
}

// cudaMajor maps the runtime extra to the expected CUDA major generation.
func cudaMajor(extra string) int {
	switch extra {
	case "cu12":
		return 12
	default:
		return 13
	}
}

// parsePins parses pip-tools lock lines (name==ver, name @ url) into normalized
// pins (PEP 503 normalized names).
func parsePins(text string) (map[string]string, error) {
	pins := map[string]string{}
	for _, raw := range strings.Split(text, "\n") {
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "--hash") {
			continue
		}
		line = strings.TrimSuffix(line, "\\")
		if m := directURLRE.FindStringSubmatch(line); m != nil {
			name := pinName(m[1])
			if v, err := wheelVersion(m[2]); err == nil {
				pins[name] = v
			}
			continue
		}
		if i := strings.Index(line, "=="); i > 0 {
			name := pinName(line[:i])
			pins[name] = strings.TrimSpace(line[i+2:])
		}
	}
	return pins, nil
}

// wheelVersion extracts the version token from a direct-URL wheel path
// (torch-2.11.0+cu130-cp312-...whl -> 2.11.0+cu130).
func wheelVersion(rawURL string) (string, error) {
	u := strings.TrimSuffix(strings.TrimSpace(rawURL), "\\")
	u = strings.ReplaceAll(u, "%2B", "+")
	base := u
	if i := strings.LastIndex(u, "/"); i >= 0 {
		base = u[i+1:]
	}
	base = strings.TrimSuffix(base, ".whl")
	parts := strings.Split(base, "-")
	if len(parts) < 2 {
		return "", fmt.Errorf("wheel name %q", base)
	}
	return parts[1], nil
}

// pinName normalizes a distribution name (PEP 503).
func pinName(s string) string {
	return strings.ToLower(regexp.MustCompile(`[-_.]+`).ReplaceAllString(s, "-"))
}

// comparePins returns matched count, diff rows, and the installed-but-unpinned
// count. Only name+version equality counts; extras (e.g. tabbyapi) tolerated.
func comparePins(pins map[string]string, dists [][]string) (matched int, diff []Diff, extras int) {
	have := make(map[string]string, len(dists))
	for _, d := range dists {
		if len(d) == 2 && d[0] != "" {
			have[pinName(d[0])] = d[1]
		}
	}
	for name, want := range pins {
		if got, ok := have[name]; ok {
			if got == want {
				matched++
			} else {
				diff = append(diff, Diff{Name: name, Found: got, Need: want})
			}
		} else {
			diff = append(diff, Diff{Name: name, Found: "(missing)", Need: want})
		}
	}
	for name := range have {
		if _, ok := pins[name]; !ok {
			extras++
		}
	}
	sort.Slice(diff, func(i, j int) bool { return diff[i].Name < diff[j].Name })
	return matched, diff, extras
}

// pinFailPhrase builds the design §5 summary inside the V3 parentheses:
// "3 of 88 differ" (pure version drift), "2 missing, 1 differ" (mixed),
// "3 missing" (§4 V3 rule: 0 missing, 0 differ).
func pinFailPhrase(diff []Diff, need int) string {
	missing, differ := 0, 0
	for _, d := range diff {
		if d.Found == "(missing)" {
			missing++
		} else {
			differ++
		}
	}
	switch {
	case missing > 0 && differ > 0:
		return fmt.Sprintf("%d missing, %d differ", missing, differ)
	case missing > 0:
		return fmt.Sprintf("%d of %d missing", missing, need)
	default:
		return fmt.Sprintf("%d of %d differ", len(diff), need)
	}
}

// duBytes stat-walks a dir (read-only) — total regular-file size.
func duBytes(p string) (int64, error) {
	var tot int64
	err := filepath.WalkDir(p, func(_ string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // ignore unreadable entries
		}
		if d.IsDir() {
			return nil
		}
		fi, err := d.Info()
		if err == nil && fi.Mode().IsRegular() {
			tot += fi.Size()
		}
		return nil
	})
	return tot, err
}

func trunc(s string) string {
	const n = 300
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// --- detection (§4) ---

// DetectInput configures candidate collection (§4 probes 1-5). Env/ProcScan
// are injectable so tests never scan the real machine.
type DetectInput struct {
	Explicit   string                   // --adopt <path>
	RuntimeDir string                   // our provisioned runtime (probe 2)
	Env        func(string) string      // default os.Getenv
	ProcScan   func(string) []Candidate // default procScan
}

// Detect collects candidate pointers in §4 precedence order. It does NOT
// validate — Validate is the only trust anchor. Probe 6 (interpreter-only
// fallback, option O3) is out of slice 1.
func Detect(in DetectInput) []Candidate {
	getenv := in.Env
	if getenv == nil {
		getenv = os.Getenv
	}
	proc := in.ProcScan
	if proc == nil {
		proc = func(string) []Candidate { return procScan(filepath.Dir(in.RuntimeDir)) }
	}
	out := make([]Candidate, 0, 6)
	seen := map[string]bool{}
	add := func(c Candidate) {
		if c.Venv == "" || seen[c.Venv] {
			return
		}
		seen[c.Venv] = true
		out = append(out, c)
	}

	// 1. explicit pointer: flag first, then $STONE_LLAMA_RUNTIME.
	if in.Explicit != "" {
		if c, ok := normalizeCandidate(in.Explicit, "flag"); ok {
			add(c)
		}
	} else if v := getenv("STONE_LLAMA_RUNTIME"); v != "" {
		if c, ok := normalizeCandidate(v, "env"); ok {
			add(c)
		}
	}
	// 2. our own provisioned runtime.
	if c, ok := normalizeCandidate(filepath.Join(in.RuntimeDir, "venv"), "provisioned"); ok {
		add(c)
	}
	// 3. recorded adoption (re-validated by the caller).
	if rec, err := loadAdoption(adoptPath(in.RuntimeDir)); err == nil && rec.Venv != "" {
		add(Candidate{Venv: rec.Venv, Checkout: rec.Checkout, Source: "journal"})
	}
	// 4. activated shell venv.
	if v := getenv("VIRTUAL_ENV"); v != "" {
		if c, ok := normalizeCandidate(v, "virtualenv"); ok {
			add(c)
		}
	}
	// 5. running TabbyAPI process (read-only /proc scan).
	for _, c := range proc(filepath.Dir(in.RuntimeDir)) {
		add(c)
	}
	return out
}

// normalizeCandidate resolves a pointer path into a Venv (+ Checkout when the
// pointer is a checkout root). It only reads the filesystem.
func normalizeCandidate(path, source string) (Candidate, bool) {
	path = filepath.Clean(path)
	if fi, err := os.Stat(path); err == nil && !fi.IsDir() &&
		strings.HasSuffix(path, filepath.Join("bin", "python")) {
		venv := filepath.Dir(filepath.Dir(path))
		return finalizeVenv(venv, "", source)
	}
	// venv at <path> itself
	if _, err := os.Stat(filepath.Join(path, "pyvenv.cfg")); err == nil {
		return finalizeVenv(path, "", source)
	}
	// venv nested under <path>/venv, checkout = <path>
	if _, err := os.Stat(filepath.Join(path, "venv", "pyvenv.cfg")); err == nil {
		return finalizeVenv(filepath.Join(path, "venv"), path, source)
	}
	return Candidate{}, false
}

func finalizeVenv(venv, checkout, source string) (Candidate, bool) {
	if !isFile(venvPython(venv)) || !isFile(filepath.Join(venv, "pyvenv.cfg")) {
		return Candidate{}, false
	}
	if checkout != "" {
		if _, err := os.Stat(filepath.Join(checkout, "start.py")); err != nil {
			checkout = "" // unknown
		}
	}
	return Candidate{Venv: venv, Checkout: checkout, Source: source}, true
}

// procScan scans /proc/*/cmdline for a running TabbyAPI started with a venv
// interpreter. Read-only; never contacts a port. On non-Linux /proc is absent
// and it simply returns nil (no build-tag split needed).
func procScan(skipPrefix string) []Candidate {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return nil
	}
	var out []Candidate
	for _, e := range entries {
		name := e.Name()
		if _, err := strconv.Atoi(name); err != nil || name == "." || name == ".." {
			continue
		}
		cmdline, err := os.ReadFile("/proc/" + name + "/cmdline")
		if err != nil {
			continue
		}
		args := strings.Split(strings.TrimRight(string(cmdline), "\x00"), "\x00")
		var py string
		var start bool
		var port uint16
		for _, a := range args {
			if py == "" && strings.Contains(a, "venv/bin/python") {
				py = a
			}
			if py == "" && strings.HasSuffix(a, "/python") && strings.Contains(filepath.Dir(a), "venv") {
				py = a
			}
			if strings.HasSuffix(filepath.Base(a), "start.py") {
				start = true
			}
			port = parsePortArg(a, port)
		}
		if py == "" || !start {
			continue
		}
		cwd, err := os.Readlink("/proc/" + name + "/cwd")
		if err != nil {
			continue
		}
		// skip our own daemon child (cwd under its runtime tree)
		if skipPrefix != "" && strings.HasPrefix(cwd, skipPrefix) {
			continue
		}
		venv := filepath.Dir(filepath.Dir(py))
		checkout := ""
		if dirHasCheckout(cwd) {
			checkout = cwd
		}
		out = append(out, Candidate{Venv: venv, Checkout: checkout, Source: "proc", Running: true, Port: port})
	}
	return out
}

func parsePortArg(a string, cur uint16) uint16 {
	if strings.HasPrefix(a, "--port=") {
		if n, err := strconv.Atoi(strings.TrimPrefix(a, "--port=")); err == nil && n > 0 && n < 65536 {
			return uint16(n)
		}
	}
	return cur
}

func dirHasCheckout(dir string) bool {
	fi, err := os.Stat(filepath.Join(dir, "start.py"))
	return err == nil && !fi.IsDir()
}

// provisionAnnounce returns the announced download bytes for a fresh provision
// (HEAD-measured via the pinned, immutable URLs — content-pinned, so stable).
func provisionAnnounce(extra string, head func(string) (int64, error)) (download int64, err error) {
	text, err := requirementsLockText(extra)
	if err != nil {
		return 0, err
	}
	torchURL, exlURL, err := wheelURLs(text)
	if err != nil {
		return 0, err
	}
	l, err := loadLock()
	if err != nil {
		return 0, err
	}
	for _, u := range []string{l.UV.URL, torchURL, exlURL} {
		n, herr := head(u)
		if herr != nil {
			return download, herr
		}
		download += n
	}
	return download, nil
}

// diskLabel renders an approximate disk figure (design §5: "≈6.9 GB").
func diskLabel(bytes int64) string {
	return fmt.Sprintf("≈%.1f GB", float64(bytes)/1e9)
}

// adoptionSummary builds the record kept at adoption time (only what passed).
func adoptionSummary(rep Report, lockText, extra string) adoptionRecord {
	return adoptionRecord{
		Venv:        rep.Candidate.Venv,
		Checkout:    rep.Candidate.Checkout,
		Extra:       extra,
		LockDigest:  lockDigest(lockText),
		ValidatedAt: time.Now().UTC().Format(time.RFC3339),
		Python:      rep.Python,
		PinsMatched: rep.PinsMatched,
		PinsNeeded:  rep.PinsNeeded,
		Torch:       rep.Torch,
		TorchCUDA:   rep.TorchCUDA,
		Exllamav3:   rep.Exllamav3,
		ImportsOK:   rep.ImportsOK,
		CUDA:        rep.CUDAAvailable,
	}
}

// ReuseStatus re-validates the recorded adoption and renders doctor's reuse
// line. The record is only a pointer: V1-V5 re-run every time (never a record
// read), so a stale record never reads green — a failed re-run invalidates it
// atomically, a passing one clears a stale invalid marker. "" when nothing
// is adopted (doctor then prints nothing extra).
func ReuseStatus(runtimeDir string) string {
	rec, err := loadAdoption(adoptPath(runtimeDir))
	if err != nil || rec.Venv == "" {
		return ""
	}
	lockText, err := requirementsLockText(rec.Extra)
	if err != nil {
		return ""
	}
	rep, verr := Validate(Candidate{Venv: rec.Venv, Checkout: rec.Checkout, Source: "journal"}, lockText, rec.Extra)
	if verr != nil {
		_ = invalidateAdoption(adoptPath(runtimeDir), rep.Failed, rep.Reason)
		check, reason := rep.Failed, rep.Reason
		if check == "" {
			check = "?"
		}
		if reason == "" {
			reason = verr.Error()
		}
		return fmt.Sprintf("Reuse      ✗ adopted %s — failed %s: %s (record invalidated; run stone-llama setup)\n", rec.Venv, check, reason)
	}
	if rec.Invalid != nil {
		rec.Invalid = nil // fixed: the gate passes again — clear the marker
		_ = saveAdoption(adoptPath(runtimeDir), rec)
	}
	return fmt.Sprintf("Reuse      adopted %s — re-validated: python %s · %d/%d locked pins exact · torch %s (CUDA %s) · exllamav3 %s\n",
		rec.Venv, rep.Python, rep.PinsMatched, rep.PinsNeeded, rep.Torch, rep.TorchCUDA, rep.Exllamav3)
}

// FormatFound renders message (a): a verified, adoptable runtime.
// downloadBytes/disk describe the fresh-provision counterfactual (design §5
// verbatim; sizes omitted when a probe failed).
func FormatFound(rep Report, downloadBytes int64, disk string) string {
	var b strings.Builder
	b.WriteString("✓ Found a matching runtime: " + rep.Candidate.Venv + "\n")
	fmt.Fprintf(&b, "  validated: python %s · %d/%d locked pins exact · torch %s (CUDA %s)\n",
		rep.Python, rep.PinsMatched, rep.PinsNeeded, rep.Torch, rep.TorchCUDA)
	b.WriteString("             · exllamav3 " + rep.Exllamav3 + " · import exllamav3 OK · cuda available\n")
	b.WriteString("  Reusing it downloads 0 bytes and adds 0 bytes of disk.\n")
	if downloadBytes > 0 {
		if disk != "" {
			fmt.Fprintf(&b, "  A fresh provision would download %s B and write %s to disk.\n", commas(downloadBytes), disk)
		} else {
			fmt.Fprintf(&b, "  A fresh provision would download %s B.\n", commas(downloadBytes))
		}
	}
	b.WriteString("  Your environment stays untouched: stone-llama only reads it (your venv, a checkout\n")
	b.WriteString("  cloned for stone-llama, your config.yml and api_tokens.yml are never modified).\n")
	return b.String()
}

// FormatNotAdoptable renders message (b) for a candidate the gate rejected:
// the failed check, the found-vs-needed diff (first 10 + "+N more"), and the
// three options with real numbers (design §5 verbatim). attachPort is the
// running-server port found by probe 5 (0 → TabbyAPI's default 5002).
func FormatNotAdoptable(rep Report, downloadBytes int64, disk string, attachPort uint16) string {
	var b strings.Builder
	head := "✗ Cannot reuse"
	if rep.Unverifiable {
		head = "✗ Could not verify"
	}
	fmt.Fprintf(&b, "%s %s — failed: %s\n", head, rep.Candidate.Venv, rep.Reason)
	printDiff(&b, rep.Diff)
	b.WriteString("  stone-llama never installs into an environment it does not own — this venv will not be modified.\n")
	b.WriteString("  Your options:\n")
	opt1 := "    1. provision stone-llama's own pinned runtime"
	if downloadBytes > 0 {
		opt1 += "   downloads " + commas(downloadBytes) + " B"
		if disk != "" {
			opt1 += ", writes " + disk
		}
	}
	b.WriteString(opt1 + "   (stone-llama setup --provision)\n")
	if attachPort == 0 {
		attachPort = 5002 // TabbyAPI's default; probe 5 didn't find a live one
	}
	fmt.Fprintf(&b, "    %-84s(%s)\n", "2. reuse a running TabbyAPI, download nothing",
		fmt.Sprintf("stone-llama serve --attach 127.0.0.1:%d", attachPort))
	fmt.Fprintf(&b, "    %-84s(%s)\n", "3. point at a different runtime",
		"stone-llama setup --adopt <path>")
	return b.String()
}

func printDiff(b *strings.Builder, diff []Diff) {
	if len(diff) == 0 {
		return
	}
	for i := 0; i < len(diff) && i < 10; i++ {
		d := diff[i]
		fmt.Fprintf(b, "      %-14s found %-13s need %s\n", d.Name, d.Found, d.Need)
	}
	if len(diff) > 10 {
		fmt.Fprintf(b, "      + %d more\n", len(diff)-10)
	}
}

// commas groups a decimal integer: 974640904 → "974,640,904" (design §5 shows
// thousands-separated byte counts).
func commas(n int64) string {
	s := strconv.FormatInt(n, 10)
	if len(s) <= 3 {
		return s
	}
	var out []byte
	for i := range len(s) {
		if i > 0 && (len(s)-i)%3 == 0 {
			out = append(out, ',')
		}
		out = append(out, s[i])
	}
	return string(out)
}
