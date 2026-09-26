package selfmgmt

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/rizperdana/stone-llama/internal/config"
	"github.com/rizperdana/stone-llama/internal/serve"
)

// UninstallOpts configures Uninstall/PlanUninstall. Zero values select
// the production defaults (real data/config dirs, real binary).
type UninstallOpts struct {
	// Yes skips the confirmation prompt (--yes). Without Yes and without
	// Confirm, Uninstall refuses: destructive work needs consent.
	Yes bool
	// DryRun prints the plan and removes nothing (--dry-run).
	DryRun bool
	// KeepModels preserves the model store, including symlinked imports
	// and real weight directories (--keep-models).
	KeepModels bool
	// Confirm receives the plan for interactive consent (CLI wires the
	// TTY prompt here, like setup.Run). nil + !Yes → refusal.
	Confirm func(Report) bool
	// Stdout receives the plan and summary; nil → io.Discard.
	Stdout io.Writer
	// Executable is the installed binary path; nil → os.Executable().
	Executable string
	// DataDir overrides the tool data dir; "" → config.DataDir().
	DataDir string
	// ConfigPath overrides the config file path; "" → the config
	// package's resolution (STONE_LLAMA_CONFIG / XDG / ~/.config).
	ConfigPath string
	// StopDaemon stops a running daemon before removal; nil →
	// serve.Stop(dataDir, 10s). Injected by tests.
	StopDaemon func(dataDir string) error
}

// Entry is one path in the plan/report.
type Entry struct {
	Path string
	Kind string // "binary" | "symlink" | "dir" | "file"
	Size int64  // bytes removal frees (symlink: link size only)
	Note string // human detail, incl. symlink targets and warnings
	Keep bool   // deliberately kept (not removed)
	deep bool   // contents NOT itemised → delete recursively (Lstat-guarded)
}

// Failure records a path that could not be removed, with the exact
// manual command (never executed — no sudo from this tool).
type Failure struct {
	Path   string
	Err    error
	Manual string
}

// Report is the uninstall plan and, after Uninstall, its outcome.
type Report struct {
	DataDir       string
	ConfigDir     string
	ConfigPath    string
	Binary        string // real binary path ("" when absent)
	BinaryLink    string // argv symlink when the install is symlinked
	DaemonPID     int
	DaemonAlive   bool
	Plan          []Entry // removal rows, in execution order
	Kept          []Entry // deliberately kept rows
	Removed       []Entry // successfully removed rows
	Failed        []Failure
	StoppedDaemon bool
	DryRun        bool
	WeightsSize   int64 // real model-weight dirs slated for deletion (bytes)
	modelsSeen    bool  // the store was itemised (weights line renders even at 0 B)
}

// PlanUninstall builds the full removal plan without changing
// anything. Every row is Lstat-classified; symlink rows name their
// target, which is never touched. Paths outside the tool's own
// data/config dirs (and the binary itself) are never planned for
// deletion — external dirs are reported under Kept with a manual hint.
func PlanUninstall(opts UninstallOpts) (Report, error) {
	dataDir := opts.DataDir
	if dataDir == "" {
		dataDir = config.DataDir()
	}
	cfgPath := opts.ConfigPath
	if cfgPath == "" {
		cfgPath = defaultConfigPath()
	}
	cfgDir := filepath.Dir(cfgPath)
	// An explicit STONE_LLAMA_CONFIG names a single file: its directory
	// (which may be a repo or any user dir) is never enumerated.
	explicitConfig := opts.ConfigPath == "" && os.Getenv("STONE_LLAMA_CONFIG") != ""
	rep := Report{DataDir: dataDir, ConfigDir: cfgDir, ConfigPath: cfgPath, DryRun: opts.DryRun}

	// ---- binary ---------------------------------------------------------------
	execPath := opts.Executable
	if execPath == "" {
		var err error
		if execPath, err = os.Executable(); err != nil {
			return rep, fmt.Errorf("cannot locate the running binary: %w", err)
		}
	}
	if abs, err := filepath.Abs(execPath); err == nil {
		execPath = abs
	}
	realBin := execPath
	if r, err := filepath.EvalSymlinks(execPath); err == nil {
		realBin = r
	}
	rep.Binary = realBin
	linkPlanned := false
	if st, err := os.Lstat(execPath); err == nil && st.Mode()&os.ModeSymlink != 0 {
		rep.BinaryLink = execPath
		rep.Plan = append(rep.Plan, Entry{
			Path: execPath, Kind: "symlink", Size: st.Size(),
			Note: "tool binary link → " + realBin + " (removed with its target)",
		})
		linkPlanned = true
	}
	// The resolved target gets its own row — unless it is the same path
	// as an already-planned (dangling) link.
	if st, err := os.Lstat(realBin); err == nil && !(linkPlanned && realBin == execPath) {
		if !st.Mode().IsRegular() {
			rep.Kept = append(rep.Kept, Entry{
				Path: realBin, Kind: "file",
				Note: "not a regular file — refused, remove manually if intended",
			})
		} else {
			rep.Plan = append(rep.Plan, Entry{
				Path: realBin, Kind: "binary", Size: st.Size(),
				Note: "the tool binary",
			})
		}
	}

	// ---- data dir -------------------------------------------------------------
	modelsDir := resolveModelsDir(dataDir, cfgPath)
	runtimeDir := resolveRuntimeDir(dataDir, cfgPath)
	if err := rep.planDirTree(dataDir, "data", modelsDir, runtimeDir, opts.KeepModels); err != nil {
		return rep, err
	}

	// ---- config ---------------------------------------------------------------
	cfgOwned := !explicitConfig && filepath.Base(cfgDir) == "stone-llama"
	if cfgOwned {
		if err := rep.planDirTree(cfgDir, "config", "", "", false); err != nil {
			return rep, err
		}
	} else if st, err := os.Lstat(cfgPath); err == nil {
		if !st.Mode().IsRegular() {
			rep.Kept = append(rep.Kept, Entry{
				Path: cfgPath, Kind: "file",
				Note: "not a regular file — refused, remove manually if intended",
			})
		} else {
			rep.Plan = append(rep.Plan, Entry{
				Path: cfgPath, Kind: "file", Size: st.Size(),
				Note: "config file (its directory is not stone-llama's — left alone)",
			})
		}
	}

	// ---- external dirs: never deleted, reported with a manual hint -----------
	if modelsDir != "" && !within(dataDir, modelsDir) {
		if st, err := os.Lstat(modelsDir); err == nil {
			rep.Kept = append(rep.Kept, Entry{
				Path: modelsDir, Kind: "dir", Size: dirSize(modelsDir, st),
				Note: "models_dir outside the tool's data dir — never deleted by uninstall; manual: sudo rm -rf " + modelsDir,
			})
		}
	}
	if runtimeDir != "" && runtimeDir != filepath.Join(dataDir, "runtime") && !within(dataDir, runtimeDir) {
		if st, err := os.Lstat(runtimeDir); err == nil {
			rep.Kept = append(rep.Kept, Entry{
				Path: runtimeDir, Kind: "dir", Size: dirSize(runtimeDir, st),
				Note: "runtime_dir outside the tool's data dir — never deleted by uninstall; manual: sudo rm -rf " + runtimeDir,
			})
		}
	}

	rep.probeDaemon()
	return rep, nil
}

// planDirTree itemises dir (one level, plus a second level under
// models/ and runtime/ so symlinked user files are visible). A symlink
// dir is planned link-only; a missing dir is not an error.
func (rep *Report) planDirTree(dir, kind, modelsDir, runtimeDir string, keepModels bool) error {
	st, err := os.Lstat(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	note := "config dir — removed after its contents (empty dir only)"
	if kind == "data" {
		note = "data dir — removed after its contents (empty dir only)"
	}
	if st.Mode()&os.ModeSymlink != 0 {
		target, _ := os.Readlink(dir)
		rep.Plan = append(rep.Plan, Entry{
			Path: dir, Kind: "symlink", Size: st.Size(),
			Note: note + ": → " + target + " (link removed; target NOT touched)",
		})
		rep.Kept = append(rep.Kept, Entry{
			Path: target, Kind: "dir", Size: 0,
			Note: "symlink target — never deleted; remove manually if intended: sudo rm -rf " + target,
		})
		return nil
	}
	if !st.IsDir() {
		return nil // a non-dir data/config root: leave it alone
	}
	children, err := os.ReadDir(dir)
	if err != nil {
		return fmt.Errorf("read %s: %w", dir, err)
	}
	for _, c := range children {
		p := filepath.Join(dir, c.Name())
		note := "tool data"
		switch kind {
		case "data":
			note = dataNote(c.Name())
		case "config":
			note = "config"
			if c.Name() == "config.json" {
				note = "user config"
			}
		}
		e, err := entryFor(p, note)
		if err != nil {
			return err
		}
		if e.Path == "" {
			continue // vanished between ReadDir and Lstat
		}
		if keepModels && modelsDir != "" && (p == modelsDir || strings.HasPrefix(p, modelsDir+string(filepath.Separator))) {
			e.Keep = true
			e.Note = appendNote(e.Note, "kept (--keep-models)")
			rep.Kept = append(rep.Kept, e)
			continue
		}
		if e.Kind == "dir" && (p == modelsDir || p == runtimeDir) && (modelsDir != "" || runtimeDir != "") {
			// itemise the level that can hold symlinks to user files:
			// rows below remove first, the (now empty) dir is rmdir'ed
			e.deep = false
			e.Note = note + " — contents itemised below"
			rep.Plan = append(rep.Plan, e)
			nested, err := readEntries(p, c.Name())
			if err != nil {
				return err
			}
			if p == modelsDir {
				rep.modelsSeen = true
				for _, n := range nested {
					if n.Kind == "dir" {
						rep.WeightsSize += n.Size
					}
				}
			}
			rep.Plan = append(rep.Plan, nested...)
			continue
		}
		rep.Plan = append(rep.Plan, e)
	}
	rep.Plan = append(rep.Plan, Entry{Path: dir, Kind: "dir", Note: note})
	return nil
}

// probeDaemon records whether a live daemon answers on loopback.
func (rep *Report) probeDaemon() {
	st, err := serve.ReadState(rep.DataDir)
	if err != nil {
		return // no daemon.json
	}
	rep.DaemonPID = st.PID
	if _, qerr := serve.Query(rep.DataDir); qerr == nil {
		rep.DaemonAlive = true
	}
}

// Uninstall prints the plan, requires consent, stops a running daemon,
// and removes the binary plus the tool's data and config. DryRun stops
// after printing. Nothing outside the tool's dirs is ever touched.
func Uninstall(ctx context.Context, opts UninstallOpts) (Report, error) {
	_ = ctx // removal is local; the daemon probe has its own timeouts
	w := opts.Stdout
	if w == nil {
		w = io.Discard
	}
	rep, err := PlanUninstall(opts)
	if err != nil {
		return rep, err
	}
	rep.DryRun = opts.DryRun
	RenderPlan(w, rep)

	if len(rep.Plan) == 0 {
		fmt.Fprintf(w, "nothing to remove — stone-llama does not appear to be installed (no %s, no %s)\n",
			rep.DataDir, rep.ConfigPath)
		return rep, nil
	}
	if opts.DryRun {
		fmt.Fprintln(w, "dry run — nothing was removed")
		return rep, nil
	}
	switch {
	case opts.Yes:
	case opts.Confirm == nil:
		return rep, errors.New("non-interactive — pass --yes to confirm the uninstall plan")
	case !opts.Confirm(rep):
		return rep, errors.New("aborted (plan not confirmed)")
	}

	// Stop a live daemon first: it may hold the runtime venv and GPU
	// memory open — never leave an orphan process on deleted files.
	if rep.DaemonAlive {
		stop := opts.StopDaemon
		if stop == nil {
			stop = func(dataDir string) error { return serve.Stop(dataDir, 10*time.Second) }
		}
		if err := stop(rep.DataDir); err != nil {
			return rep, fmt.Errorf("daemon (pid %d) could not be stopped: %w — nothing was removed; run 'stone-llama stop' and retry", rep.DaemonPID, err)
		}
		rep.StoppedDaemon = true
		fmt.Fprintf(w, "stopped daemon (pid %d)\n", rep.DaemonPID)
	}

	binaries := make([]string, 0, 2)
	for _, b := range []string{rep.Binary, rep.BinaryLink} {
		if b != "" {
			binaries = append(binaries, b)
		}
	}
	removeOne := func(e Entry) {
		if e.Keep {
			rep.Kept = append(rep.Kept, e)
			return
		}
		if err := removeEntry(rep.DataDir, rep.ConfigDir, rep.ConfigPath, binaries, e); err != nil {
			if errors.Is(err, errEscape) {
				// containment refusal: never delete, report honestly
				rep.Failed = append(rep.Failed, Failure{Path: e.Path, Err: err})
				return
			}
			if isNotEmpty(err) && keepsUnder(rep, e.Path) {
				// expected: kept rows (e.g. --keep-models) fill the dir
				e.Keep = true
				e.Note = appendNote(e.Note, "kept (not empty after removal)")
				rep.Kept = append(rep.Kept, e)
				return
			}
			rep.Failed = append(rep.Failed, Failure{
				Path: e.Path, Err: err,
				Manual: manualRemove(e),
			})
			return
		}
		rep.Removed = append(rep.Removed, e)
	}
	// Pass 1 removes the leaves; pass 2 rmdir's the itemised containers
	// (models/, runtime/, the data and config dirs) only after their
	// contents are gone — never a container before its own rows.
	for _, e := range rep.Plan {
		if e.Kind == "dir" && !e.deep {
			continue
		}
		removeOne(e)
	}
	for _, e := range rep.Plan {
		if e.Kind == "dir" && !e.deep {
			removeOne(e)
		}
	}

	RenderSummary(w, rep)
	if len(rep.Failed) > 0 {
		return rep, fmt.Errorf("%d path(s) could not be removed — see the manual commands above", len(rep.Failed))
	}
	return rep, nil
}

func appendNote(note, add string) string {
	if note == "" {
		return add
	}
	return note + "; " + add
}

func keepsUnder(rep Report, dir string) bool {
	for _, k := range rep.Kept {
		if within(dir, k.Path) {
			return true
		}
	}
	return false
}

func manualRemove(e Entry) string {
	kind := "-f"
	if e.Kind == "dir" || e.deep {
		kind = "-rf"
	}
	// sudo: a failure here usually means a root-owned path; we never
	// run sudo ourselves — the user does, explicitly.
	return "sudo rm " + kind + " " + e.Path
}

// ---- rendering ---------------------------------------------------------------

// RenderPlan prints every path (with size and symlink targets), the
// model-weights total, and the daemon state — the picture the user
// consents to.
func RenderPlan(w io.Writer, rep Report) {
	fmt.Fprintln(w, "stone-llama uninstall plan")
	switch {
	case rep.DaemonAlive:
		fmt.Fprintf(w, "  daemon:  running (pid %d) — will be stopped first\n", rep.DaemonPID)
	case rep.DaemonPID != 0:
		fmt.Fprintln(w, "  daemon:  not running (stale daemon.json will be removed)")
	}
	if len(rep.Plan) == 0 {
		fmt.Fprintln(w, "  (nothing to remove)")
	}
	for _, e := range rep.Plan {
		fmt.Fprintf(w, "  %-8s %14s  %s", e.Kind, fmtSize(e.Size), e.Path)
		if e.Note != "" {
			fmt.Fprintf(w, "  (%s)", e.Note)
		}
		fmt.Fprintln(w)
	}
	if rep.modelsSeen {
		fmt.Fprintf(w, "  model weights to delete: %s (%s)", fmtSize(rep.WeightsSize), fmtHuman(rep.WeightsSize))
		if rep.WeightsSize == 0 {
			fmt.Fprint(w, " — every model is a symlink; their targets are never touched")
		} else {
			fmt.Fprint(w, " — real directories inside the store")
		}
		fmt.Fprintln(w)
	}
	for _, e := range rep.Kept {
		fmt.Fprintf(w, "  KEEP    %14s  %s  (%s)\n", fmtSize(e.Size), e.Path, e.Note)
	}
	fmt.Fprintln(w)
}

// RenderSummary prints the honest post-mortem: counts, bytes, keeps,
// and every failure with its exact manual command.
func RenderSummary(w io.Writer, rep Report) {
	if rep.DryRun {
		return
	}
	var bytes int64
	for _, e := range rep.Removed {
		bytes += e.Size
	}
	fmt.Fprintf(w, "removed %d path(s), %s (%s); kept %d; failed %d\n",
		len(rep.Removed), fmtSize(bytes), fmtHuman(bytes), len(rep.Kept), len(rep.Failed))
	for _, f := range rep.Failed {
		fmt.Fprintf(w, "  FAILED %s: %v\n", f.Path, f.Err)
		if f.Manual != "" {
			fmt.Fprintf(w, "    manual: %s\n", f.Manual)
		}
	}
}

// ---- path safety -------------------------------------------------------------

var errEscape = errors.New("outside the tool's data/config dirs")

// removeEntry deletes one planned path with containment enforcement:
// the cleaned path must live under dataDir or under a config dir named
// stone-llama, or be exactly the tool binary (or its argv symlink) or
// the config file (an explicit STONE_LLAMA_CONFIG may live anywhere);
// the resolved parent must stay inside the resolved root (defeats
// symlinked parents); and the removal itself is Lstat-based, so a
// symlink is unlinked — never followed.
func removeEntry(dataDir, cfgDir, cfgPath string, binaries []string, e Entry) error {
	abs := cleanAbs(e.Path)
	isBinary := false
	for _, b := range binaries {
		if b != "" && abs == cleanAbs(b) {
			isBinary = true
			break
		}
	}
	isCfgFile := cfgPath != "" && abs == cleanAbs(cfgPath)
	cfgRoot := ""
	if filepath.Base(cleanAbs(cfgDir)) == "stone-llama" {
		cfgRoot = cfgDir // only a dir named stone-llama is the tool's
	}
	var root string
	switch {
	case isBinary || isCfgFile:
		// exact-match allowances: no tree containment to enforce
	case within(dataDir, abs):
		root = resolved(dataDir)
	case cfgRoot != "" && within(cfgRoot, abs):
		root = resolved(cfgRoot)
	default:
		return fmt.Errorf("refusing to delete %s: %w", abs, errEscape)
	}
	if root != "" {
		// The root itself is allowed (its parent lies outside by
		// definition); anything BELOW it must not resolve outside.
		isRoot := abs == cleanAbs(dataDir) || abs == cleanAbs(cfgDir)
		parent := resolved(filepath.Dir(abs))
		if !isRoot && parent != "" && !within(root, parent) {
			return fmt.Errorf("refusing to delete %s: %w (parent resolves to %s)", abs, errEscape, parent)
		}
	}
	st, err := os.Lstat(abs)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // already gone
		}
		return err
	}
	switch {
	case st.Mode()&os.ModeSymlink != 0:
		return os.Remove(abs) // unlink the link; target untouched
	case st.IsDir() && e.deep:
		return os.RemoveAll(abs) // contents not itemised; RemoveAll never follows symlinks
	case st.IsDir():
		return os.Remove(abs) // itemised container: empty dir only
	default:
		return os.Remove(abs)
	}
}

// within reports whether path is inside root (both made absolute and
// cleaned). Uses filepath.Rel — no string-prefix tricks.
func within(root, path string) bool {
	rel, err := filepath.Rel(cleanAbs(root), cleanAbs(path))
	if err != nil {
		return false
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && !filepath.IsAbs(rel)
}

func cleanAbs(p string) string {
	if a, err := filepath.Abs(p); err == nil {
		p = a
	}
	return filepath.Clean(p)
}

// resolved returns the symlink-resolved form of p ("" when p cannot
// be resolved, e.g. it does not exist).
func resolved(p string) string {
	if r, err := filepath.EvalSymlinks(p); err == nil {
		return cleanAbs(r)
	}
	return ""
}

func isNotEmpty(err error) bool {
	return errors.Is(err, syscall.ENOTEMPTY) || errors.Is(err, syscall.EEXIST)
}

// ---- entry construction ------------------------------------------------------

func dataNote(name string) string {
	switch name {
	case "models":
		return "model weights"
	case "runtime":
		return "tool runtime (python, venv, TabbyAPI checkout)"
	case "logs":
		return "logs"
	case "downloads":
		return "download staging"
	case "hf_token":
		return "credential — removed; run 'stone-llama login' again after reinstall"
	case "daemon.json":
		return "daemon state (holds a bearer token)"
	case "stone-llama.lock":
		return "lock file"
	default:
		return "tool data"
	}
}

// entryFor Lstat-classifies one path. Symlinks are never followed and
// never descended into: only the link itself is planned for removal.
func entryFor(path, note string) (Entry, error) {
	st, err := os.Lstat(path)
	if os.IsNotExist(err) {
		return Entry{}, nil
	}
	if err != nil {
		return Entry{}, fmt.Errorf("stat %s: %w", path, err)
	}
	if st.Mode()&os.ModeSymlink != 0 {
		target, _ := os.Readlink(path)
		return Entry{
			Path: path, Kind: "symlink", Size: st.Size(),
			Note: note + ": → " + target + " (link removed; target NOT touched)",
		}, nil
	}
	if st.IsDir() {
		// deep: contents not itemised → recursive removal (planDirTree
		// downgrades models/runtime to itemised rmdir when it enumerates).
		return Entry{Path: path, Kind: "dir", Size: dirSize(path, st), Note: note, deep: true}, nil
	}
	return Entry{Path: path, Kind: "file", Size: st.Size(), Note: note}, nil
}

// readEntries itemises one level under models/ or runtime/ so that
// symlinked user files (imported models, an adopted venv) are visible
// and unlink-only.
func readEntries(dir, parent string) ([]Entry, error) {
	children, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", dir, err)
	}
	var out []Entry
	for _, c := range children {
		p := filepath.Join(dir, c.Name())
		note := "tool data"
		switch parent {
		case "models":
			note = "model"
		case "runtime":
			note = "runtime"
		}
		e, err := entryFor(p, note)
		if err != nil {
			return nil, err
		}
		if e.Path == "" {
			continue
		}
		if e.Kind == "dir" {
			what := "contents will be deleted"
			if parent == "models" {
				what = "contents (incl. weights) will be deleted"
			}
			e.Note = note + " — real directory: " + what
			e.deep = true
		}
		out = append(out, e)
	}
	return out, nil
}

// dirSize sums bytes under root WITHOUT following any symlink (Lstat
// walk): a symlinked model contributes its link size only.
func dirSize(root string, st os.FileInfo) int64 {
	if st.Mode()&os.ModeSymlink != 0 || !st.IsDir() {
		return st.Size()
	}
	var total int64
	var walk func(dir string)
	walk = func(dir string) {
		entries, err := os.ReadDir(dir)
		if err != nil {
			return
		}
		for _, c := range entries {
			p := filepath.Join(dir, c.Name())
			info, err := os.Lstat(p)
			if err != nil {
				continue
			}
			if info.Mode()&os.ModeSymlink != 0 {
				total += info.Size()
				continue
			}
			if info.IsDir() {
				walk(p)
				continue
			}
			total += info.Size()
		}
	}
	walk(root)
	return total
}

// ---- config resolution (mirrors internal/config/config.go) -------------------

// defaultConfigPath mirrors config.configPath (unexported there):
// STONE_LLAMA_CONFIG wins, then XDG_CONFIG_HOME/stone-llama/config.json,
// then ~/.config/stone-llama/config.json.
func defaultConfigPath() string {
	if p := os.Getenv("STONE_LLAMA_CONFIG"); p != "" {
		return p
	}
	if x := os.Getenv("XDG_CONFIG_HOME"); x != "" {
		return filepath.Join(x, "stone-llama", "config.json")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "stone-llama", "config.json")
}

// resolveModelsDir mirrors config.Load's precedence for models_dir:
// STONE_LLAMA_MODELS_DIR env > config file > dataDir/models.
func resolveModelsDir(dataDir, cfgPath string) string {
	if x := os.Getenv("STONE_LLAMA_MODELS_DIR"); x != "" {
		return expandHome(x)
	}
	if p, ok := readCfgField(cfgPath, "models_dir"); ok {
		return expandHome(p)
	}
	return filepath.Join(dataDir, "models")
}

// resolveRuntimeDir mirrors config.RuntimeDir plus serve's
// normalization: "" → dataDir/runtime (the default setup target).
func resolveRuntimeDir(dataDir, cfgPath string) string {
	if p, ok := readCfgField(cfgPath, "runtime_dir"); ok {
		return expandHome(p)
	}
	return filepath.Join(dataDir, "runtime")
}

func readCfgField(cfgPath, field string) (string, bool) {
	b, err := os.ReadFile(cfgPath)
	if err != nil {
		return "", false
	}
	var m map[string]any
	if json.Unmarshal(b, &m) != nil {
		return "", false
	}
	s, ok := m[field].(string)
	return s, ok && s != ""
}

func expandHome(p string) string {
	if p == "~" || strings.HasPrefix(p, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, strings.TrimPrefix(p[1:], "/"))
		}
	}
	return p
}
