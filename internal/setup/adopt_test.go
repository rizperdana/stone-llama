package setup

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// --- fake-venv fixture -------------------------------------------------
//
// The gate probes a candidate by running its own bin/python. The fixture's
// bin/python is a shell script that answers the two probe shapes (identity,
// import) with canned JSON — no network, no real interpreter, no real venv.

// mkFakeVenv writes <tmp>/venv with a pyvenv.cfg and an answering bin/python.
func mkFakeVenv(t *testing.T, identityJSON, importJSON string) string {
	return buildFakeVenv(t, t.TempDir(), "venv", identityJSON, importJSON)
}

// mkFakeVenvAt builds a passing fake venv at <root>/<name> (probe 2 and the
// checkout fixtures need venvs at specific paths).
func mkFakeVenvAt(t *testing.T, root, name string) string {
	return buildFakeVenv(t, root, name, passIdentity(t), passImport(t))
}

func buildFakeVenv(t *testing.T, root, name, identityJSON, importJSON string) string {
	t.Helper()
	venv := filepath.Join(root, name)
	if err := os.MkdirAll(filepath.Join(venv, "bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	cfg := "home = /usr/bin\nimplementation = CPython\nversion_info = 3.12.13.final.0\n"
	if err := os.WriteFile(filepath.Join(venv, "pyvenv.cfg"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	script := "#!/bin/sh\ncase \"$2\" in\n" +
		"*distributions*) printf '%s' '" + identityJSON + "' ;;\n" +
		"*) printf '%s' '" + importJSON + "' ;;\n" +
		"esac\n"
	if err := os.WriteFile(filepath.Join(venv, "bin", "python"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return venv
}

// passIdentity builds the V2+V3 identity JSON whose dists satisfy the cu13
// lock exactly (V3 closure equality).
func passIdentity(t *testing.T) string {
	t.Helper()
	id := identity{Python: "3.12.13", Venv: true}
	for name, ver := range lockPins(t) {
		id.Dists = append(id.Dists, []string{name, ver})
	}
	return marshalJSON(t, id)
}

// passImport is the V4+V5 import JSON for a healthy cu13 environment.
func passImport(t *testing.T) string {
	t.Helper()
	var ir importResult
	ir.Torch.V = "2.11.0+cu130"
	ir.Torch.Cuda = "13.0"
	ir.Torch.CudaOK = true
	ir.Exllamav3 = "ok"
	return marshalJSON(t, ir)
}

func marshalJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func mustIdentity(t *testing.T, s string) identity {
	t.Helper()
	var id identity
	if err := json.Unmarshal([]byte(s), &id); err != nil {
		t.Fatal(err)
	}
	return id
}

// lockPins returns the parsed cu13 pins for assertions.
func lockPins(t *testing.T) map[string]string {
	t.Helper()
	lockText, err := requirementsLockText("cu13")
	if err != nil {
		t.Fatal(err)
	}
	pins, err := parsePins(lockText)
	if err != nil {
		t.Fatal(err)
	}
	return pins
}

// --- V1-V5 gate table --------------------------------------------------

// TestAdoptGateTable drives Validate over pass/fail fixtures: every V-check
// fails closed with its design-§4 phrasing; the pass fixture walks the whole
// gate and pins the report fields.
func TestAdoptGateTable(t *testing.T) {
	lockText, err := requirementsLockText("cu13")
	if err != nil {
		t.Fatal(err)
	}
	pins := lockPins(t)

	// V3 drift: torch/transformers/triton wrong (design §5 example shape).
	drifted := func(t *testing.T) string {
		t.Helper()
		id := mustIdentity(t, passIdentity(t))
		for _, d := range id.Dists {
			switch d[0] {
			case "torch":
				d[1] = "2.13.0"
			case "transformers", "triton":
				d[1] = "9.9.9"
			}
		}
		return marshalJSON(t, id)
	}
	// V3 drift: a rotating 12-pin miss (feeds the "+N more" truncation).
	missing12 := func(t *testing.T) string {
		t.Helper()
		id := mustIdentity(t, passIdentity(t))
		for i := range id.Dists {
			if i%7 == 0 {
				id.Dists[i][1] = "9.9.9"
			}
		}
		return marshalJSON(t, id)
	}

	cases := []struct {
		name      string
		ident     func(t *testing.T) string // nil → passIdentity
		imp       func(t *testing.T) string // nil → passImport
		noPython  bool                      // venv without bin/python (V1 gone)
		wantFail  string                    // "" → gate passes
		wantPlain bool                      // ErrNotAdoptable vs ErrUnverifiable
		wantSub   string
	}{
		{
			name: "pass",
		},
		{
			name:      "V1 gone (no bin/python)",
			noPython:  true,
			wantFail:  "V1",
			wantPlain: true,
			wantSub:   "the venv or its base Python is gone",
		},
		{
			name: "V2 wrong python",
			ident: func(t *testing.T) string {
				return `{"python":"3.13.1","venv":true,"dists":[]}`
			},
			wantFail:  "V2",
			wantPlain: true,
			wantSub:   "found Python 3.13.1, wheels are cp312",
		},
		{
			name:      "V3 drifted versions",
			ident:     drifted,
			wantFail:  "V3",
			wantPlain: true,
			wantSub:   "locked-package equality (3 of ",
		},
		{
			name:      "V3 missing pins",
			ident:     missing12,
			wantFail:  "V3",
			wantPlain: true,
			wantSub:   "locked-package equality (",
		},
		{
			name: "V4 exllamav3 import fails",
			imp: func(t *testing.T) string {
				return `{"exllamav3_error":"ModuleNotFoundError('exllamav3')"}`
			},
			wantFail:  "V4",
			wantPlain: true,
			wantSub:   "import exllamav3 — ModuleNotFoundError",
		},
		{
			name: "V4 torch import fails",
			imp: func(t *testing.T) string {
				return `{"torch_error":"ImportError('libcudart.so.12')","exllamav3":"ok"}`
			},
			wantFail:  "V4",
			wantPlain: true,
			wantSub:   "import torch — ImportError",
		},
		{
			name: "V5 cpu-only wheel",
			imp: func(t *testing.T) string {
				var ir importResult
				ir.Torch.V = "2.11.0+cpu"
				ir.Exllamav3 = "ok"
				return marshalJSON(t, ir)
			},
			wantFail:  "V5",
			wantPlain: true,
			wantSub:   "torch reports no CUDA (CPU-only wheel); the cu13 extra needs CUDA",
		},
		{
			name: "V5 wrong cuda generation",
			imp: func(t *testing.T) string {
				var ir importResult
				ir.Torch.V = "2.11.0+cu124"
				ir.Torch.Cuda = "12.4"
				ir.Torch.CudaOK = true
				ir.Exllamav3 = "ok"
				return marshalJSON(t, ir)
			},
			wantFail:  "V5",
			wantPlain: true,
			wantSub:   "found CUDA 12.4, this machine selects the cu13 extra (CUDA 13)",
		},
		{
			name: "V5 driver too old",
			imp: func(t *testing.T) string {
				var ir importResult
				ir.Torch.V = "2.11.0+cu130"
				ir.Torch.Cuda = "13.0"
				ir.Torch.CudaOK = false
				ir.Exllamav3 = "ok"
				return marshalJSON(t, ir)
			},
			wantFail:  "V5",
			wantPlain: true,
			wantSub:   "torch.cuda.is_available() is false (driver too old for these wheels)",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ident := passIdentity(t)
			if tc.ident != nil {
				ident = tc.ident(t)
			}
			imp := passImport(t)
			if tc.imp != nil {
				imp = tc.imp(t)
			}
			venv := mkFakeVenv(t, ident, imp)
			if tc.noPython {
				if err := os.Remove(filepath.Join(venv, "bin", "python")); err != nil {
					t.Fatal(err)
				}
			}

			rep, err := Validate(Candidate{Venv: venv, Source: "virtualenv"}, lockText, "cu13")

			if tc.wantFail == "" {
				if err != nil {
					t.Fatalf("Validate pass: %v", err)
				}
				if !rep.OK || rep.Failed != "" {
					t.Fatalf("pass report: OK=%v Failed=%q", rep.OK, rep.Failed)
				}
				if rep.PinsMatched != rep.PinsNeeded || rep.PinsNeeded != len(pins) {
					t.Fatalf("pins: %d/%d matched, want %d", rep.PinsMatched, rep.PinsNeeded, len(pins))
				}
				if rep.Torch == "" || rep.TorchCUDA != "13.0" || !rep.ImportsOK || !rep.CUDAAvailable {
					t.Fatalf("pass fields: torch=%q cuda=%q imports=%v avail=%v",
						rep.Torch, rep.TorchCUDA, rep.ImportsOK, rep.CUDAAvailable)
				}
				return
			}
			if err == nil {
				t.Fatalf("Validate should fail %s", tc.wantFail)
			}
			if rep.Failed != tc.wantFail {
				t.Fatalf("Failed=%q want %q (reason %q)", rep.Failed, tc.wantFail, rep.Reason)
			}
			var na *ErrNotAdoptable
			var un *ErrUnverifiable
			isPlain := errors.As(err, &na)
			isUn := errors.As(err, &un)
			if isPlain == isUn {
				t.Fatalf("error must be exactly one kind: plain=%v unverifiable=%v", isPlain, isUn)
			}
			if isPlain != tc.wantPlain {
				t.Fatalf("error kind: plain=%v, want plain=%v", isPlain, tc.wantPlain)
			}
			if !strings.Contains(rep.Reason, tc.wantSub) {
				t.Fatalf("reason %q does not contain %q", rep.Reason, tc.wantSub)
			}
		})
	}
}

// TestAdoptCandidateNotWritten asserts the gate never writes into the
// candidate (PYTHONDONTWRITEBYTECODE: no __pycache__ after V1-V5).
func TestAdoptCandidateNotWritten(t *testing.T) {
	lockText, err := requirementsLockText("cu13")
	if err != nil {
		t.Fatal(err)
	}
	venv := mkFakeVenv(t, passIdentity(t), passImport(t))
	if _, err := Validate(Candidate{Venv: venv, Source: "virtualenv"}, lockText, "cu13"); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	var seen []string
	if err := filepath.Walk(venv, func(p string, fi os.FileInfo, err error) error {
		if err == nil && strings.Contains(p, "__pycache__") {
			seen = append(seen, p)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(seen) > 0 {
		t.Fatalf("gate wrote into candidate: %v", seen)
	}
}

// --- message (a) -------------------------------------------------------

// TestFormatFoundWording pins message (a) to design §5's console block.
func TestFormatFoundWording(t *testing.T) {
	pins := lockPins(t)
	lockText, err := requirementsLockText("cu13")
	if err != nil {
		t.Fatal(err)
	}
	venv := mkFakeVenv(t, passIdentity(t), passImport(t))
	rep, err := Validate(Candidate{Venv: venv, Source: "virtualenv"}, lockText, "cu13")
	if err != nil {
		t.Fatal(err)
	}

	msg := FormatFound(rep, 6000, "≈6.9 GB")
	want := []string{
		"✓ Found a matching runtime: " + venv,
		"  validated: python 3.12.13 · " + itoa(len(pins)) + "/" + itoa(len(pins)) +
			" locked pins exact · torch " + pins["torch"] + " (CUDA 13.0)",
		"             · exllamav3 " + pins["exllamav3"] + " · import exllamav3 OK · cuda available",
		"  Reusing it downloads 0 bytes and adds 0 bytes of disk.",
		"  A fresh provision would download 6,000 B and write ≈6.9 GB to disk.",
		"  Your environment stays untouched: stone-llama only reads it (your venv, a checkout",
		"  cloned for stone-llama, your config.yml and api_tokens.yml are never modified).",
	}
	for _, w := range want {
		if !strings.Contains(msg, w) {
			t.Errorf("message (a) missing line:\n%s\n--- got:\n%s", w, msg)
		}
	}
}

// --- message (b) -------------------------------------------------------

// TestFormatNotAdoptableWording pins message (b) to design §5: header with
// the exact failed-check phrasing, the found-vs-needed rows, the ownership
// line, and the three options with thousands-separated bytes.
func TestFormatNotAdoptableWording(t *testing.T) {
	pins := lockPins(t)
	lockText, err := requirementsLockText("cu13")
	if err != nil {
		t.Fatal(err)
	}
	id := mustIdentity(t, passIdentity(t))
	for _, d := range id.Dists {
		switch d[0] {
		case "torch":
			d[1] = "2.13.0"
		case "transformers", "triton":
			d[1] = "9.9.9"
		}
	}
	venv := mkFakeVenv(t, marshalJSON(t, id), passImport(t))
	rep, err := Validate(Candidate{Venv: venv, Source: "virtualenv"}, lockText, "cu13")
	if err == nil {
		t.Fatal("expected V3 failure")
	}

	msg := FormatNotAdoptable(rep, 974640904, "≈6.9 GB", 0)
	want := []string{
		"✗ Cannot reuse " + venv + " — failed: locked-package equality (3 of " + itoa(len(pins)) + " differ)",
		"      torch          found 2.13.0        need " + pins["torch"],
		"      transformers   found 9.9.9         need " + pins["transformers"],
		"      triton         found 9.9.9         need " + pins["triton"],
		"  stone-llama never installs into an environment it does not own — this venv will not be modified.",
		"  Your options:",
		"    1. provision stone-llama's own pinned runtime   downloads 974,640,904 B, writes ≈6.9 GB   (stone-llama setup --provision)",
		"stone-llama serve --attach 127.0.0.1:5002",
		"stone-llama setup --adopt <path>",
	}
	for _, w := range want {
		if !strings.Contains(msg, w) {
			t.Errorf("message (b) missing line:\n%s\n--- got:\n%s", w, msg)
		}
	}
	if strings.Contains(msg, "more\n") {
		t.Errorf("3-row diff must not truncate:\n%s", msg)
	}
}

// TestFormatNotAdoptableTruncates asserts first-10 rows + "+N more" and
// that unknown sizes never render as fake zeros.
func TestFormatNotAdoptableTruncates(t *testing.T) {
	rep := Report{Reason: "locked-package equality (12 of 88 differ)"}
	for i := range 12 {
		rep.Diff = append(rep.Diff, Diff{Name: "pkg" + itoa(i), Found: "9.9.9", Need: "1.0.0"})
	}
	msg := FormatNotAdoptable(rep, 0, "", 0)
	if got := strings.Count(msg, "9.9.9"); got != 10 {
		t.Errorf("shown rows = %d, want 10:\n%s", got, msg)
	}
	if !strings.Contains(msg, "+ 2 more") {
		t.Errorf("missing truncation marker:\n%s", msg)
	}
	if strings.Contains(msg, "downloads 0 B") {
		t.Errorf("unknown download size must not render as 0 B:\n%s", msg)
	}
}

// itoa keeps assertion helpers dependency-light.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// --- adoption flow (Run / detectAdoption) ------------------------------

// TestRunAdoptsAndResumes pins the slice-1 happy path: message (a), the
// local clone, the venv symlink, the adoption record, adopted journal
// markers — and a second Run resumes without re-cloning.
func TestRunAdoptsAndResumes(t *testing.T) {
	pins := lockPins(t)
	venv := mkFakeVenv(t, passIdentity(t), passImport(t))
	checkout := t.TempDir() // their TabbyAPI checkout (start.py absent → still cloned by path)
	stdout := &strings.Builder{}
	rec := &recorder{}
	rt := filepath.Join(t.TempDir(), "runtime")
	opts := buildOpts(rt, rec, stdout, true)
	opts.Detect = func(DetectInput) []Candidate {
		return []Candidate{{Venv: venv, Checkout: checkout, Source: "virtualenv"}}
	}

	if err := Run(opts); err != nil {
		t.Fatalf("Run: %v", err)
	}
	out := stdout.String()
	for _, want := range []string{
		"✓ Found a matching runtime: " + venv,
		" would download 6,000 B",
		"Your environment stays untouched",
		"adopted runtime recorded (" + adoptPath(rt) + ")",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("stdout missing %q:\n%s", want, out)
		}
	}

	// provision path must NOT have run: only clone steps were executed
	for _, c := range rec.calls {
		if c.stepID != "tabby-local-clone" {
			t.Errorf("adoption ran provision step %q", c.stepID)
		}
	}
	if len(rec.fetches) != 0 {
		t.Errorf("adoption must fetch nothing, got %v", rec.fetches)
	}
	var cloned, checkedOut bool
	for _, c := range rec.calls {
		if c.stepID == "tabby-local-clone" && c.name == "git" && len(c.args) >= 3 && c.args[0] == "clone" {
			cloned = c.args[1] == checkout && c.args[2] == filepath.Join(rt, "tabbyAPI")
		}
		if c.stepID == "tabby-local-clone" && len(c.args) >= 4 && c.args[0] == "-C" && c.args[2] == "checkout" {
			checkedOut = c.args[3] == tabbyPin()
		}
	}
	if !cloned || !checkedOut {
		t.Errorf("clone=%v checkout=%v calls=%+v", cloned, checkedOut, rec.calls)
	}

	// symlink → candidate venv
	if got, err := os.Readlink(filepath.Join(rt, "venv")); err != nil || got != venv {
		t.Errorf("symlink = %q, %v; want %q", got, err, venv)
	}

	// adoption record: only what passed
	ad, err := loadAdoption(adoptPath(rt))
	if err != nil {
		t.Fatalf("adoption record: %v", err)
	}
	if ad.Venv != venv || ad.PinsMatched != len(pins) || ad.PinsNeeded != len(pins) {
		t.Errorf("record: venv=%q pins=%d/%d want %d", ad.Venv, ad.PinsMatched, ad.PinsNeeded, len(pins))
	}
	if !strings.HasPrefix(ad.LockDigest, "sha256:") || ad.Invalid != nil || !ad.ImportsOK || !ad.CUDA {
		t.Errorf("record fields: digest=%q invalid=%+v imports=%v cuda=%v",
			ad.LockDigest, ad.Invalid, ad.ImportsOK, ad.CUDA)
	}

	// journal adoption markers
	j, err := loadJournal(filepath.Join(rt, "setup-journal.json"))
	if err != nil {
		t.Fatalf("journal: %v", err)
	}
	for _, step := range []string{"uv", "python", "venv-deps"} {
		if j.Done[step] != "adopted" {
			t.Errorf("journal[%q] = %q, want adopted", step, j.Done[step])
		}
	}
	if j.Done["tabby-local-clone"] != "done" {
		t.Errorf("journal clone = %q", j.Done["tabby-local-clone"])
	}

	// resume: second Run must not clone again
	stdout.Reset()
	rec.calls = nil
	if err := Run(opts); err != nil {
		t.Fatalf("second Run: %v", err)
	}
	for _, c := range rec.calls {
		t.Errorf("resume executed a step: %+v", c)
	}
}

// TestRunAdoptExplicitReject: a failed explicit candidate stops with
// message (b) and an error — no provision fallback, no exec, no fetch.
func TestRunAdoptExplicitReject(t *testing.T) {
	pins := lockPins(t)
	id := mustIdentity(t, passIdentity(t))
	for _, d := range id.Dists {
		if d[0] == "torch" {
			d[1] = "2.13.0"
		}
	}
	venv := mkFakeVenv(t, marshalJSON(t, id), passImport(t))
	stdout := &strings.Builder{}
	rec := &recorder{}
	rt := filepath.Join(t.TempDir(), "runtime")
	opts := buildOpts(rt, rec, stdout, true)
	opts.Adopt = venv
	opts.Detect = func(DetectInput) []Candidate {
		return []Candidate{{Venv: venv, Source: "flag"}}
	}

	err := Run(opts)
	if err == nil {
		t.Fatal("explicit reject must stop the run")
	}
	var na *ErrNotAdoptable
	if !errors.As(err, &na) || na.Report.Failed != "V3" {
		t.Fatalf("error = %v, want ErrNotAdoptable V3", err)
	}
	out := stdout.String()
	for _, want := range []string{
		"✗ Cannot reuse " + venv + " — failed: locked-package equality (1 of " + itoa(len(pins)) + " differ)",
		"      torch          found 2.13.0        need " + pins["torch"],
		"  Your options:",
		"downloads 6,000 B",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("stdout missing %q:\n%s", want, out)
		}
	}
	if len(rec.calls) != 0 || len(rec.fetches) != 0 {
		t.Errorf("reject must not provision: calls=%v fetches=%v", rec.calls, rec.fetches)
	}
	if _, err := os.Stat(adoptPath(rt)); !os.IsNotExist(err) {
		t.Errorf("adoption record must not exist after reject")
	}
}

// TestRunAdoptUnresolvableExplicit: --adopt pointing at a path that is not a
// venv at all still stops with (b) (design §4 probe 1: never silently fall
// back to another candidate).
func TestRunAdoptUnresolvableExplicit(t *testing.T) {
	stdout := &strings.Builder{}
	rec := &recorder{}
	rt := filepath.Join(t.TempDir(), "runtime")
	missing := filepath.Join(t.TempDir(), "nope")
	opts := buildOpts(rt, rec, stdout, true)
	opts.Adopt = missing
	opts.Detect = func(DetectInput) []Candidate { return nil } // normalization failed

	err := Run(opts)
	var na *ErrNotAdoptable
	if !errors.As(err, &na) || na.Report.Failed != "V1" {
		t.Fatalf("error = %v, want ErrNotAdoptable V1", err)
	}
	if !strings.Contains(stdout.String(), "✗ Cannot reuse "+missing) ||
		!strings.Contains(stdout.String(), "the venv or its base Python is gone") {
		t.Errorf("stdout missing message (b):\n%s", stdout.String())
	}
	if len(rec.calls) != 0 {
		t.Errorf("must not provision: %v", rec.calls)
	}
}

// TestRunProvisionSkipsDetection: --provision forces the classic path; the
// detection seam must not even be consulted.
func TestRunProvisionSkipsDetection(t *testing.T) {
	stdout := &strings.Builder{}
	rec := &recorder{}
	rt := filepath.Join(t.TempDir(), "runtime")
	opts := buildOpts(rt, rec, stdout, true)
	opts.Provision = true
	opts.Detect = func(DetectInput) []Candidate {
		t.Fatal("detection ran despite --provision")
		return nil
	}
	if err := Run(opts); err != nil {
		t.Fatalf("Run: %v", err)
	}
	found := false
	for _, c := range rec.calls {
		if c.stepID == "venv-deps" {
			found = true
		}
	}
	if !found {
		t.Errorf("provision did not run: %+v", rec.calls)
	}
}

// TestAdoptJournalInvalidatedAndFallback: a stale adoption record fails the
// gate → message (b) + atomic invalidation, and does NOT mask the healthy
// candidate probed after it (design §4 probe 3).
func TestAdoptJournalInvalidatedAndFallback(t *testing.T) {
	id := mustIdentity(t, passIdentity(t))
	for _, d := range id.Dists {
		if d[0] == "torch" {
			d[1] = "2.13.0"
		}
	}
	stale := mkFakeVenv(t, marshalJSON(t, id), passImport(t))
	healthy := mkFakeVenv(t, passIdentity(t), passImport(t))

	stdout := &strings.Builder{}
	rec := &recorder{}
	rt := filepath.Join(t.TempDir(), "runtime")
	if err := os.MkdirAll(rt, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := saveAdoption(adoptPath(rt), adoptionRecord{Venv: stale, ValidatedAt: "2020-01-01T00:00:00Z"}); err != nil {
		t.Fatal(err)
	}

	opts := buildOpts(rt, rec, stdout, false)
	opts.Detect = func(DetectInput) []Candidate {
		return []Candidate{
			{Venv: stale, Source: "journal"},
			{Venv: healthy, Source: "virtualenv"},
		}
	}
	d, err := detectAdoption(opts)
	if err != nil {
		t.Fatalf("detectAdoption: %v", err)
	}
	if d.kind != adoptFound || d.report.Candidate.Venv != healthy {
		t.Fatalf("decision = %+v, want adoptFound(healthy)", d)
	}
	out := stdout.String()
	if !strings.Contains(out, "✗ Cannot reuse "+stale) || !strings.Contains(out, "✓ Found a matching runtime: "+healthy) {
		t.Errorf("stdout missing (b)/(a) pair:\n%s", out)
	}

	// record invalidated atomically with the exact failed check
	ad, err := loadAdoption(adoptPath(rt))
	if err != nil {
		t.Fatal(err)
	}
	if ad.Invalid == nil || ad.Invalid.Check != "V3" || ad.Invalid.Reason == "" {
		t.Errorf("record not invalidated: %+v", ad.Invalid)
	}
}

// --- detection probes (design §4) --------------------------------------

// TestDetectPrecedenceAndSources pins probe order flag > provisioned >
// journal > virtualenv > proc, the env fallback when no explicit flag, and
// pointer normalization (checkout root, bin/python path) with de-duplication.
func TestDetectPrecedenceAndSources(t *testing.T) {
	// shared venvs so probes can point at each other
	flagVenv := mkFakeVenv(t, passIdentity(t), passImport(t))
	rt := filepath.Join(t.TempDir(), "runtime")
	if err := os.MkdirAll(rt, 0o755); err != nil {
		t.Fatal(err)
	}
	// probe 2: provisioned venv lives at <rt>/venv (build it in place)
	provVenv := mkFakeVenvAt(t, rt, "venv")
	// probe 3: recorded adoption
	journalVenv := mkFakeVenv(t, passIdentity(t), passImport(t))
	if err := saveAdoption(adoptPath(rt), adoptionRecord{Venv: journalVenv}); err != nil {
		t.Fatal(err)
	}
	// probe 4: activated shell venv
	venvEnv := mkFakeVenv(t, passIdentity(t), passImport(t))
	// probe 5: running process
	procVenv := mkFakeVenv(t, passIdentity(t), passImport(t))
	env := map[string]string{"VIRTUAL_ENV": venvEnv}
	proc := []Candidate{{Venv: procVenv, Source: "proc", Running: true, Port: 5002}}

	got := Detect(DetectInput{
		Explicit:   flagVenv,
		RuntimeDir: rt,
		Env:        func(k string) string { return env[k] },
		ProcScan:   func(string) []Candidate { return proc },
	})
	wantOrder := []string{flagVenv, provVenv, journalVenv, venvEnv, procVenv}
	if len(got) != len(wantOrder) {
		t.Fatalf("candidates = %+v", got)
	}
	for i, w := range wantOrder {
		if got[i].Venv != w {
			t.Errorf("candidate[%d] = %q, want %q", i, got[i].Venv, w)
		}
	}
	if got[0].Source != "flag" || got[1].Source != "provisioned" ||
		got[2].Source != "journal" || got[3].Source != "virtualenv" || got[4].Source != "proc" {
		t.Errorf("sources = %q %q %q %q %q", got[0].Source, got[1].Source, got[2].Source, got[3].Source, got[4].Source)
	}
	if !got[4].Running || got[4].Port != 5002 {
		t.Errorf("proc candidate lost running hint: %+v", got[4])
	}

	// no explicit flag → env probe wins first (STONE_LLAMA_RUNTIME)
	env["STONE_LLAMA_RUNTIME"] = flagVenv
	got = Detect(DetectInput{
		RuntimeDir: rt,
		Env:        func(k string) string { return env[k] },
		ProcScan:   func(string) []Candidate { return proc },
	})
	if got[0].Venv != flagVenv || got[0].Source != "env" {
		t.Errorf("env probe not first: %+v", got[0])
	}

	// dedupe: VIRTUAL_ENV pointing at the provisioned venv is added once
	env["VIRTUAL_ENV"] = provVenv
	got = Detect(DetectInput{
		RuntimeDir: rt,
		Env:        func(k string) string { return env[k] },
		ProcScan:   func(string) []Candidate { return nil },
	})
	seen := map[string]int{}
	for _, c := range got {
		seen[c.Venv]++
	}
	if seen[provVenv] != 1 {
		t.Errorf("provisioned venv counted %d times: %+v", seen[provVenv], got)
	}

	// pointer normalization: checkout root and bin/python path both resolve
	co := t.TempDir()
	if err := os.WriteFile(filepath.Join(co, "start.py"), []byte("#"), 0o644); err != nil {
		t.Fatal(err)
	}
	mkFakeVenvAt(t, co, "venv")
	c, ok := normalizeCandidate(co, "flag")
	if !ok || c.Venv != filepath.Join(co, "venv") || c.Checkout != co {
		t.Errorf("checkout root: %+v ok=%v", c, ok)
	}
	c, ok = normalizeCandidate(filepath.Join(flagVenv, "bin", "python"), "flag")
	if !ok || c.Venv != flagVenv {
		t.Errorf("bin/python pointer: %+v ok=%v", c, ok)
	}
	if _, ok := normalizeCandidate(filepath.Join(t.TempDir(), "nothing"), "flag"); ok {
		t.Error("nonexistent path normalized")
	}
}

// TestReuseStatusReValidates: doctor's reuse line re-runs the gate against
// the recorded candidate — a healthy record reports reuse with fresh numbers;
// a broken candidate yields the failed check and the record is invalidated
// atomically (never left green).
func TestReuseStatusReValidates(t *testing.T) {
	rt := filepath.Join(t.TempDir(), "runtime")
	if line := ReuseStatus(rt); line != "" {
		t.Errorf("no record → line %q, want empty", line)
	}
	if err := os.MkdirAll(rt, 0o755); err != nil {
		t.Fatal(err)
	}
	venv := mkFakeVenv(t, passIdentity(t), passImport(t))
	if err := saveAdoption(adoptPath(rt), adoptionRecord{Venv: venv, Extra: "cu13"}); err != nil {
		t.Fatal(err)
	}
	pins := len(lockPins(t))
	line := ReuseStatus(rt)
	for _, want := range []string{"Reuse      adopted " + venv, "re-validated: python 3.12.13", itoa(pins) + "/" + itoa(pins) + " locked pins exact"} {
		if !strings.Contains(line, want) {
			t.Errorf("reuse line missing %q: %q", want, line)
		}
	}

	// break the candidate → re-validation fails and the record goes stale-red
	if err := os.RemoveAll(filepath.Join(venv, "bin")); err != nil {
		t.Fatal(err)
	}
	line = ReuseStatus(rt)
	if !strings.Contains(line, "failed V1") || !strings.Contains(line, "record invalidated") {
		t.Errorf("stale line: %q", line)
	}
	ad, err := loadAdoption(adoptPath(rt))
	if err != nil || ad.Invalid == nil || ad.Invalid.Check != "V1" {
		t.Errorf("record not invalidated: %+v err=%v", ad, err)
	}
}
