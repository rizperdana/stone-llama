// Package setup — adoption flow: detection → gate → consent → journal.
//
// Design: internal/sl-adopt/adoption-design.md §4-§6 (slice 1).
// Keeps setup.go free of the detection logic while staying one package.
package setup

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

// adoptKind classifies the detection outcome.
type adoptKind int

const (
	adoptNone adoptKind = iota
	adoptFound
	adoptReject // explicit candidate failed the gate (message (b) printed)
)

type adoptDecision struct {
	kind     adoptKind
	report   Report
	explicit bool
	err      error
}

// detectAdoption runs probes 1-5, applies the gate, and prints the message for
// the first found/rejected candidate (design §4 precedence). It does NOT
// install — Run decides whether to adopt or fall back to provision.
// The announced provision sizes are computed lazily, only when a message is
// rendered, so a machine with no candidate never issues a HEAD.
func detectAdoption(opts Options) (adoptDecision, error) {
	lockText, err := requirementsLockText(opts.Extra)
	if err != nil {
		return adoptDecision{}, err
	}

	candidates := opts.Detect(DetectInput{
		Explicit:   opts.Adopt,
		RuntimeDir: opts.RuntimeDir,
		Env:        os.Getenv,
	})
	sizes := func(venv string) (download int64, disk string) {
		if opts.HeadSize != nil {
			if d, err := provisionAnnounce(opts.Extra, opts.HeadSize); err == nil {
				download = d
			}
		}
		if du, derr := duBytes(venv); derr == nil && du > 0 {
			disk = diskLabel(du)
		}
		return download, disk
	}

	// design §4 probe 1: an explicit pointer (--adopt / $STONE_LLAMA_RUNTIME)
	// never falls back silently — if it could not even be resolved, report V1
	// and stop ("the user asked for this path, they get the truth").
	explicitPath, wantSource := opts.Adopt, "flag"
	if explicitPath == "" {
		explicitPath, wantSource = os.Getenv("STONE_LLAMA_RUNTIME"), "env"
	}
	if explicitPath != "" && !hasSource(candidates, wantSource) {
		rep := Report{
			Candidate: Candidate{Venv: explicitPath, Source: wantSource},
			Failed:    "V1",
			Reason:    "the venv or its base Python is gone",
		}
		download, disk := sizes("")
		fmt.Fprint(opts.Stdout, FormatNotAdoptable(rep, download, disk, 0))
		return adoptDecision{kind: adoptReject, report: rep, explicit: true,
			err: &ErrNotAdoptable{Report: rep}}, nil
	}

	attachPort := uint16(0)
	for i, cand := range candidates {
		if cand.Running && cand.Port != 0 {
			attachPort = cand.Port
		}
		rep, verr := Validate(cand, lockText, opts.Extra)
		explicit := cand.Source == "flag" || cand.Source == "env"
		switch {
		case verr == nil:
			download, disk := sizes(cand.Venv)
			fmt.Fprint(opts.Stdout, FormatFound(rep, download, disk))
			// design §4: note other full matches informationally (lower
			// precedence still wins nothing, but the user sees them)
			for _, other := range candidates[i+1:] {
				if orep, oerr := Validate(other, lockText, opts.Extra); oerr == nil {
					fmt.Fprintf(opts.Stdout, "another matching runtime: %s\n", orep.Candidate.Venv)
				}
			}
			return adoptDecision{kind: adoptFound, report: rep}, nil
		case errors.As(verr, new(*ErrNotAdoptable)), errors.As(verr, new(*ErrUnverifiable)):
			download, disk := sizes(cand.Venv)
			fmt.Fprint(opts.Stdout, FormatNotAdoptable(rep, download, disk, attachPort))
			if cand.Source == "journal" {
				// stale record invalidated atomically, never left green
				_ = invalidateAdoption(adoptPath(opts.RuntimeDir), rep.Failed, rep.Reason)
			}
			if explicit {
				return adoptDecision{kind: adoptReject, report: rep, explicit: true, err: verr}, nil
			}
			// non-explicit rejected candidate → continue probing
		default:
			return adoptDecision{}, verr
		}
	}
	return adoptDecision{kind: adoptNone}, nil
}

func hasSource(cs []Candidate, src string) bool {
	for _, c := range cs {
		if c.Source == src {
			return true
		}
	}
	return false
}

// runAdoption executes the recorded adoption: clone their checkout, symlink
// their venv, record the passed gate. Never mutates the candidate.
func runAdoption(opts Options, rep Report) error {
	cand := rep.Candidate
	if opts.Exec == nil {
		opts.Exec = defaultExec(opts.RuntimeDir)
	}
	if err := os.MkdirAll(opts.RuntimeDir, 0o755); err != nil {
		return fmt.Errorf("setup: mkdir %s: %w", opts.RuntimeDir, err)
	}

	// journal (done-as-adopted markers)
	jpath := filepath.Join(opts.RuntimeDir, "setup-journal.json")
	j, _ := loadJournal(jpath)
	if j.Done == nil {
		j.Done = map[string]string{}
	}
	mark := func(step, note string) { j.Done[step] = note }
	mark("uv", "adopted")
	mark("python", "adopted")
	mark("venv-deps", "adopted")

	// tabby-local-clone: clone their checkout into runtime/tabbyAPI at the pin
	// (0 network when the pin object is present — design §4 probe 2/O1).
	cloneStep := "tabby-local-clone"
	cloneDir := filepath.Join(opts.RuntimeDir, "tabbyAPI")
	if _, ok := j.Done[cloneStep]; !ok && cand.Checkout != "" {
		if _, err := exec.LookPath("git"); err == nil {
			if out, err := opts.Exec(cloneStep, "git", "clone", cand.Checkout, cloneDir); err != nil {
				return fmt.Errorf("setup: %s: %s (%w)", cloneStep, trunc(string(out)), err)
			}
			if pin := tabbyPin(); pin != "" {
				if out, err := opts.Exec(cloneStep, "git", "-C", cloneDir, "checkout", pin); err != nil {
					return fmt.Errorf("setup: %s checkout: %s (%w)", cloneStep, trunc(string(out)), err)
				}
			}
			mark(cloneStep, "done")
		}
	}

	// symlink runtime/venv -> their venv (spawnChild's path check passes unchanged)
	venvLink := filepath.Join(opts.RuntimeDir, "venv")
	if err := os.Symlink(cand.Venv, venvLink); err != nil {
		if !errors.Is(err, os.ErrExist) {
			return fmt.Errorf("setup: symlink venv: %w", err)
		}
		// resume: an existing link must already point at this candidate
		if got, lerr := os.Readlink(venvLink); lerr != nil || got != cand.Venv {
			return fmt.Errorf("setup: %s exists but does not point at %s", venvLink, cand.Venv)
		}
	}

	// adoption record (only what passed)
	lockText, err := requirementsLockText(opts.Extra)
	if err != nil {
		return err
	}
	if err := saveAdoption(adoptPath(opts.RuntimeDir), adoptionSummary(rep, lockText, opts.Extra)); err != nil {
		return err
	}
	if err := saveJournal(jpath, j); err != nil {
		return err
	}

	// final smoke through the symlink (existing step 5)
	if _, err := probeRun(venvLink, importCode, 120*time.Second); err != nil {
		return fmt.Errorf("setup: smoke: %w", err)
	}
	fmt.Fprintf(opts.Stdout, "adopted runtime recorded (%s)\n", adoptPath(opts.RuntimeDir))
	return nil
}
