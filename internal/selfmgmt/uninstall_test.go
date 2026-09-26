package selfmgmt

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/rizperdana/stone-llama/internal/serve"
)

// fixture builds a realistic install layout entirely under t.TempDir():
// binary, data dir (symlinked model import + real model dir, symlinked
// runtime/venv, logs, credential, lock), config dir, the user's
// EXTERNAL weights dir, the user's external venv, and a fake source
// repo — all outside the tool's dirs must survive every test.
type fixture struct {
	root       string
	bin        string
	dataDir    string
	cfgDir     string
	cfgPath    string
	extModels  string // user's own weights (import target)
	extVenv    string // user's own virtualenv (adopted target)
	repo       string // fake source checkout
	modelLink  string // models/SmolLM3-3B-exl3 → extModels
	weightsDir string // models/real-model (real dir)
	weightSum  int64  // bytes inside weightsDir
	venvLink   string // runtime/venv → extVenv
	extFile    []byte // extModels payload, asserted byte-identical
	sentinel   []byte // extVenv sentinel payload
}

func buildFixture(t *testing.T) *fixture {
	t.Helper()
	f := &fixture{root: t.TempDir()}
	write := func(p string, content []byte, mode os.FileMode) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, content, mode); err != nil {
			t.Fatal(err)
		}
		mustUnderTemp(t, p)
	}

	// binary
	f.bin = filepath.Join(f.root, "bin", "stone-llama")
	write(f.bin, []byte("BINARY"), 0o755)

	// data dir
	f.dataDir = filepath.Join(f.root, "share", "stone-llama")
	f.modelLink = filepath.Join(f.dataDir, "models", "SmolLM3-3B-exl3")
	f.weightsDir = filepath.Join(f.dataDir, "models", "real-model")
	f.venvLink = filepath.Join(f.dataDir, "runtime", "venv")

	// user's external weights (import target)
	f.extModels = filepath.Join(f.root, "ai", "models", "SmolLM3-3B-exl3")
	f.extFile = []byte(strings.Repeat("W", 4096))
	write(filepath.Join(f.extModels, "model.safetensors"), f.extFile, 0o644)
	write(filepath.Join(f.extModels, "config.json"), []byte(`{"architectures":["LlamaForCausalLM"]}`), 0o644)

	// user's external venv (adoption target)
	f.extVenv = filepath.Join(f.root, "venvs", "tabby")
	f.sentinel = []byte("venv-sentinel-do-not-delete")
	write(filepath.Join(f.extVenv, "sentinel.txt"), f.sentinel, 0o644)
	write(filepath.Join(f.extVenv, "pyvenv.cfg"), []byte("home = /usr\n"), 0o644)

	// store entries
	if err := os.MkdirAll(filepath.Dir(f.modelLink), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(f.extModels, f.modelLink); err != nil {
		t.Fatal(err)
	}
	// real model dir (weights that WILL be deleted)
	cfg := bytes.Repeat([]byte("c"), 64)
	weights := bytes.Repeat([]byte("w"), 2048)
	f.weightSum = int64(len(cfg) + len(weights))
	write(filepath.Join(f.weightsDir, "config.json"), cfg, 0o644)
	write(filepath.Join(f.weightsDir, "weights.bin"), weights, 0o644)

	// runtime: symlinked venv + a real tool checkout
	if err := os.MkdirAll(filepath.Dir(f.venvLink), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(f.extVenv, f.venvLink); err != nil {
		t.Fatal(err)
	}
	write(filepath.Join(f.dataDir, "runtime", "tabbyAPI", "start.py"), []byte("# tabby"), 0o644)

	// misc data
	write(filepath.Join(f.dataDir, "logs", "daemon.log"), []byte("log\n"), 0o644)
	write(filepath.Join(f.dataDir, "hf_token"), []byte("hf_dummy_credential_for_tests\n"), 0o600)
	write(filepath.Join(f.dataDir, "stone-llama.lock"), nil, 0o600)

	// config dir
	f.cfgDir = filepath.Join(f.root, "cfg", "stone-llama")
	f.cfgPath = filepath.Join(f.cfgDir, "config.json")
	write(f.cfgPath, []byte(`{"port":5111}`), 0o644)

	// source repo (must never be touched)
	f.repo = filepath.Join(f.root, "src", "stone-llama")
	write(filepath.Join(f.repo, "main.go"), []byte("package main\n"), 0o644)
	return f
}

func (f *fixture) opts(stdout *bytes.Buffer) UninstallOpts {
	return UninstallOpts{
		Executable: f.bin,
		DataDir:    f.dataDir,
		ConfigPath: f.cfgPath,
		Stdout:     stdout,
	}
}

func (f *fixture) mustExist(t *testing.T, paths ...string) {
	t.Helper()
	for _, p := range paths {
		if _, err := os.Lstat(p); err != nil {
			t.Errorf("expected to exist: %s (%v)", p, err)
		}
	}
}

func (f *fixture) mustGone(t *testing.T, paths ...string) {
	t.Helper()
	for _, p := range paths {
		if _, err := os.Lstat(p); !os.IsNotExist(err) {
			t.Errorf("expected to be gone: %s (err=%v)", p, err)
		}
	}
}

func (f *fixture) assertExternalIntact(t *testing.T) {
	t.Helper()
	got, err := os.ReadFile(filepath.Join(f.extModels, "model.safetensors"))
	if err != nil {
		t.Fatalf("external weights gone: %v", err)
	}
	if !bytes.Equal(got, f.extFile) {
		t.Error("external weights bytes changed")
	}
	f.mustExist(t, f.extModels, f.repo, filepath.Join(f.repo, "main.go"))
	sent, err := os.ReadFile(filepath.Join(f.extVenv, "sentinel.txt"))
	if err != nil || !bytes.Equal(sent, f.sentinel) {
		t.Errorf("external venv sentinel lost: %q err=%v", sent, err)
	}
}

// ---- plan ---------------------------------------------------------------------

func TestUninstallPlanShowsSizesSymlinksAndWeights(t *testing.T) {
	f := buildFixture(t)
	var out bytes.Buffer
	rep, err := PlanUninstall(f.opts(&out))
	if err != nil {
		t.Fatalf("PlanUninstall: %v", err)
	}
	RenderPlan(&out, rep)
	o := out.String()

	byPath := map[string]Entry{}
	for _, e := range append(append([]Entry{}, rep.Plan...), rep.Kept...) {
		byPath[e.Path] = e
	}
	// symlinked import: link row with its target, target never deleted
	linkRow, ok := byPath[f.modelLink]
	if !ok || linkRow.Kind != "symlink" {
		t.Fatalf("model symlink row missing: %+v", linkRow)
	}
	if !strings.Contains(linkRow.Note, f.extModels) || !strings.Contains(linkRow.Note, "target NOT touched") {
		t.Errorf("model symlink note = %q", linkRow.Note)
	}
	// symlinked adopted venv
	venvRow, ok := byPath[f.venvLink]
	if !ok || venvRow.Kind != "symlink" || !strings.Contains(venvRow.Note, f.extVenv) {
		t.Fatalf("venv symlink row: %+v", venvRow)
	}
	// real model dir is visibly deletable
	realRow, ok := byPath[f.weightsDir]
	if !ok || realRow.Kind != "dir" || realRow.Size != f.weightSum {
		t.Fatalf("real model row = %+v, want dir with %d bytes", realRow, f.weightSum)
	}
	if !strings.Contains(realRow.Note, "real directory") {
		t.Errorf("real model note must say weights are deleted: %q", realRow.Note)
	}
	// weights total is unmistakable
	if !strings.Contains(o, "model weights to delete: "+fmtSize(f.weightSum)) {
		t.Errorf("weights total missing:\n%s", o)
	}
	// credential is called out
	if !strings.Contains(o, "credential") || !strings.Contains(o, "stone-llama login") {
		t.Errorf("hf_token credential note missing:\n%s", o)
	}
	// binary + config rows
	if _, ok := byPath[f.bin]; !ok {
		t.Error("binary row missing")
	}
	if _, ok := byPath[f.cfgPath]; !ok {
		t.Error("config.json row missing")
	}
	// sizes render for every plan row
	for _, e := range rep.Plan {
		if !strings.Contains(o, fmtSize(e.Size)) {
			t.Errorf("size %s for %s not rendered", fmtSize(e.Size), e.Path)
		}
	}
	// planning mutates nothing
	f.mustExist(t, f.bin, f.dataDir, f.modelLink, f.venvLink, f.cfgPath)
	f.assertExternalIntact(t)
}

func TestUninstallDryRunRemovesNothing(t *testing.T) {
	f := buildFixture(t)
	var out bytes.Buffer
	opts := f.opts(&out)
	opts.DryRun = true // no Yes, no Confirm: dry run never asks
	rep, err := Uninstall(context.Background(), opts)
	if err != nil {
		t.Fatalf("Uninstall --dry-run: %v", err)
	}
	if !rep.DryRun || len(rep.Removed) != 0 {
		t.Errorf("dry run mutated state: %+v", rep)
	}
	if !strings.Contains(out.String(), "dry run — nothing was removed") {
		t.Errorf("missing dry-run line:\n%s", out.String())
	}
	f.mustExist(t, f.bin, f.dataDir, f.modelLink, f.venvLink, f.cfgPath, f.cfgDir)
	f.assertExternalIntact(t)
}

// ---- confirmation -------------------------------------------------------------

func TestUninstallRequiresYesWithoutTTY(t *testing.T) {
	f := buildFixture(t)
	var out bytes.Buffer
	_, err := Uninstall(context.Background(), f.opts(&out)) // no Yes, no Confirm
	if err == nil || !strings.Contains(err.Error(), "pass --yes") {
		t.Fatalf("err = %v", err)
	}
	f.mustExist(t, f.bin, f.dataDir, f.cfgPath)
	f.mustExist(t, filepath.Join(f.dataDir, "hf_token"))
}

func TestUninstallDeclinedConfirmationChangesNothing(t *testing.T) {
	f := buildFixture(t)
	var out bytes.Buffer
	opts := f.opts(&out)
	var seen Report
	opts.Confirm = func(r Report) bool {
		seen = r
		return false
	}
	_, err := Uninstall(context.Background(), opts)
	if err == nil || !strings.Contains(err.Error(), "aborted") {
		t.Fatalf("err = %v", err)
	}
	if len(seen.Plan) == 0 {
		t.Error("Confirm must receive a populated plan")
	}
	f.mustExist(t, f.bin, f.dataDir, f.cfgPath)
	f.assertExternalIntact(t)
}

// ---- full removal -------------------------------------------------------------

func TestUninstallFullRemovesDataBinaryConfigButNeverTargets(t *testing.T) {
	f := buildFixture(t)
	var out bytes.Buffer
	opts := f.opts(&out)
	opts.Yes = true
	rep, err := Uninstall(context.Background(), opts)
	if err != nil {
		t.Fatalf("Uninstall: %v", err)
	}
	if len(rep.Failed) != 0 {
		t.Fatalf("unexpected failures: %+v", rep.Failed)
	}
	f.mustGone(t,
		f.bin,
		f.dataDir,   // whole data dir, incl. models/ runtime/ logs/ locks
		f.modelLink, // the store symlink itself
		f.venvLink,
		filepath.Join(f.dataDir, "hf_token"),
		filepath.Join(f.dataDir, "daemon.log"),
		f.cfgPath,
		f.cfgDir,
	)
	// external targets byte-identical and present
	f.assertExternalIntact(t)
	// honest summary: count + bytes
	var sum int64
	for _, e := range rep.Removed {
		sum += e.Size
	}
	o := out.String()
	wantSummary := "removed " + strconv.Itoa(len(rep.Removed)) + " path(s), " + fmtSize(sum)
	if !strings.Contains(o, wantSummary) {
		t.Errorf("summary %q missing from:\n%s", wantSummary, o)
	}
	if !strings.Contains(o, "failed 0") {
		t.Errorf("failed-0 line missing:\n%s", o)
	}
	for _, want := range []string{
		"target NOT touched",
		"stone-llama uninstall plan",
		"model weights to delete: " + fmtSize(f.weightSum),
		"the tool binary",
	} {
		if !strings.Contains(o, want) {
			t.Errorf("output missing %q:\n%s", want, o)
		}
	}
}

func TestUninstallKeepModels(t *testing.T) {
	f := buildFixture(t)
	var out bytes.Buffer
	opts := f.opts(&out)
	opts.Yes, opts.KeepModels = true, true
	rep, err := Uninstall(context.Background(), opts)
	if err != nil {
		t.Fatalf("Uninstall --keep-models: %v", err)
	}
	// store kept, including the symlinked import and the real model
	f.mustExist(t,
		filepath.Join(f.dataDir, "models"),
		f.modelLink,
		filepath.Join(f.weightsDir, "weights.bin"),
	)
	// everything else gone
	f.mustGone(t,
		f.bin,
		filepath.Join(f.dataDir, "runtime"),
		f.venvLink,
		filepath.Join(f.dataDir, "logs"),
		filepath.Join(f.dataDir, "hf_token"),
		f.cfgPath,
		f.cfgDir,
	)
	// data dir itself survives (models fill it) and is reported kept
	f.mustExist(t, f.dataDir)
	keptPaths := map[string]bool{}
	for _, k := range rep.Kept {
		keptPaths[k.Path] = true
	}
	if !keptPaths[filepath.Join(f.dataDir, "models")] {
		t.Errorf("models not in Kept: %+v", rep.Kept)
	}
	if !strings.Contains(out.String(), "KEEP") || !strings.Contains(out.String(), "--keep-models") {
		t.Errorf("keep-models visibility missing:\n%s", out.String())
	}
	f.assertExternalIntact(t)
}

// ---- symlink + containment crux ----------------------------------------------

func TestUninstallSymlinkedModelTargetSurvives(t *testing.T) {
	f := buildFixture(t)
	var out bytes.Buffer
	opts := f.opts(&out)
	opts.Yes = true
	if _, err := Uninstall(context.Background(), opts); err != nil {
		t.Fatalf("Uninstall: %v", err)
	}
	// the link is gone from the store …
	if _, err := os.Lstat(f.modelLink); !os.IsNotExist(err) {
		t.Errorf("store symlink still present: %v", err)
	}
	// … and the user's weights are byte-identical on disk
	f.assertExternalIntact(t)
	got, err := os.ReadFile(filepath.Join(f.extModels, "config.json"))
	if err != nil || !strings.Contains(string(got), "LlamaForCausalLM") {
		t.Errorf("external config.json lost: %q err=%v", got, err)
	}
	if !strings.Contains(out.String(), "target NOT touched") {
		t.Errorf("plan never said targets survive:\n%s", out.String())
	}
}

func TestUninstallSymlinkedVenvTargetSurvives(t *testing.T) {
	f := buildFixture(t)
	var out bytes.Buffer
	opts := f.opts(&out)
	opts.Yes = true
	if _, err := Uninstall(context.Background(), opts); err != nil {
		t.Fatalf("Uninstall: %v", err)
	}
	if _, err := os.Lstat(f.venvLink); !os.IsNotExist(err) {
		t.Errorf("runtime/venv symlink still present: %v", err)
	}
	// the adopted venv and the sentinel inside it survive intact
	f.assertExternalIntact(t)
	st, err := os.Stat(f.extVenv)
	if err != nil || !st.IsDir() {
		t.Fatalf("external venv dir lost: %v", err)
	}
	if !strings.Contains(out.String(), f.extVenv) {
		t.Errorf("plan never showed the venv target:\n%s", out.String())
	}
}

func TestRemoveEntryRefusesEscapingPath(t *testing.T) {
	f := buildFixture(t)
	// crafted: sibling of the data dir via ".."
	victim := filepath.Join(f.root, "share", "escape-file")
	if err := os.WriteFile(victim, []byte("must live"), 0o644); err != nil {
		t.Fatal(err)
	}
	err := removeEntry(f.dataDir, f.cfgDir, f.cfgPath, nil,
		Entry{Path: filepath.Join(f.dataDir, "..", "escape-file"), Kind: "file"})
	if !errors.Is(err, errEscape) {
		t.Fatalf("escape not refused: %v", err)
	}
	if _, rerr := os.ReadFile(victim); rerr != nil {
		t.Errorf("victim deleted: %v", rerr)
	}
}

func TestRemoveEntryRefusesSymlinkedParentEscape(t *testing.T) {
	f := buildFixture(t)
	outside := filepath.Join(f.root, "outside")
	victim := filepath.Join(outside, "victim.txt")
	if err := os.MkdirAll(outside, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(victim, []byte("must live"), 0o644); err != nil {
		t.Fatal(err)
	}
	sub := filepath.Join(f.dataDir, "sub")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(sub, "evil")); err != nil {
		t.Fatal(err)
	}
	// string-wise inside the data dir, but the resolved parent escapes
	err := removeEntry(f.dataDir, f.cfgDir, f.cfgPath, nil,
		Entry{Path: filepath.Join(sub, "evil", "victim.txt"), Kind: "file"})
	if !errors.Is(err, errEscape) {
		t.Fatalf("symlinked-parent escape not refused: %v", err)
	}
	if _, rerr := os.ReadFile(victim); rerr != nil {
		t.Errorf("victim deleted: %v", rerr)
	}
}

func TestUninstallExternalModelsDirKept(t *testing.T) {
	f := buildFixture(t)
	// point models_dir at the user's own directory (config override)
	if err := os.WriteFile(f.cfgPath, []byte(`{"models_dir":"`+f.extModels+`"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	opts := f.opts(&out)
	opts.Yes = true
	rep, err := Uninstall(context.Background(), opts)
	if err != nil {
		t.Fatalf("Uninstall: %v", err)
	}
	kept := false
	for _, k := range rep.Kept {
		if k.Path == f.extModels {
			kept = true
			if !strings.Contains(k.Note, "never deleted by uninstall") {
				t.Errorf("external models note: %q", k.Note)
			}
		}
	}
	if !kept {
		t.Errorf("external models_dir not kept: %+v", rep.Kept)
	}
	if !strings.Contains(out.String(), "KEEP") || !strings.Contains(out.String(), f.extModels) {
		t.Errorf("kept external dir not visible:\n%s", out.String())
	}
	f.assertExternalIntact(t)
	// the default (leftover) store inside the data dir is still removed
	f.mustGone(t, filepath.Join(f.dataDir, "models"))
}

func TestUninstallExplicitConfigTouchesOnlyItsFile(t *testing.T) {
	f := buildFixture(t)
	// STONE_LLAMA_CONFIG names a file inside a dir that ALSO holds a
	// source file: only the config file may go.
	sentinel := filepath.Join(f.cfgDir, "main.go")
	if err := os.WriteFile(sentinel, []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("STONE_LLAMA_CONFIG", f.cfgPath)
	var out bytes.Buffer
	opts := UninstallOpts{
		Executable: f.bin,
		DataDir:    f.dataDir,
		// ConfigPath deliberately empty: resolve via env
		Stdout: &out,
		Yes:    true,
	}
	rep, err := Uninstall(context.Background(), opts)
	if err != nil {
		t.Fatalf("Uninstall: %v", err)
	}
	f.mustGone(t, f.cfgPath)
	f.mustExist(t, sentinel, f.cfgDir)
	if !strings.Contains(out.String(), "its directory is not stone-llama's") {
		t.Errorf("explicit-config note missing:\n%s", out.String())
	}
	_ = rep
}

// ---- permissions & daemon -----------------------------------------------------

func TestUninstallPermissionDeniedReportsManualCommand(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("running as root — permission bits do not deny deletion")
	}
	f := buildFixture(t)
	blocked := filepath.Join(f.dataDir, "blocked")
	if err := os.MkdirAll(blocked, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(blocked, "file"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(blocked, 0o555); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(blocked, 0o755) }) // let TempDir clean up

	var out bytes.Buffer
	opts := f.opts(&out)
	opts.Yes = true
	rep, err := Uninstall(context.Background(), opts)
	if err == nil {
		t.Fatal("undeletable dir must surface an error")
	}
	if len(rep.Failed) == 0 {
		t.Fatalf("no failures reported: %+v", rep)
	}
	o := out.String()
	foundManual := false
	for _, fl := range rep.Failed {
		if fl.Path == blocked && strings.HasPrefix(fl.Manual, "sudo rm -rf ") {
			foundManual = true
		}
	}
	if !foundManual {
		t.Errorf("manual command for %s missing: %+v", blocked, rep.Failed)
	}
	if !strings.Contains(o, "manual: sudo rm -rf "+blocked) {
		t.Errorf("manual line missing from output:\n%s", o)
	}
	// partial success is reported honestly; no crash
	if _, err := os.Stat(f.bin); !os.IsNotExist(err) {
		t.Error("binary should still have been removed")
	}
	if _, err := os.Stat(blocked); err != nil {
		t.Errorf("blocked dir should remain: %v", err)
	}
	f.assertExternalIntact(t)
}

// fakeDaemon stands up a loopback /-/status endpoint and writes a
// daemon.json pointing at it — no real daemon, no port 20128/5002/5001.
func fakeDaemon(t *testing.T, dataDir string, pid int) *httptest.Server {
	t.Helper()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"mode":"supervised","ready":true}`))
	}))
	t.Cleanup(ts.Close)
	u, err := url.Parse(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatal(err)
	}
	if err := serve.WriteState(dataDir, serve.State{
		PID: pid, Host: "127.0.0.1", Port: port,
		StartedAt: 1,
	}); err != nil {
		t.Fatal(err)
	}
	return ts
}

func TestUninstallStopsRunningDaemonFirst(t *testing.T) {
	f := buildFixture(t)
	fakeDaemon(t, f.dataDir, 424242)
	var out bytes.Buffer
	opts := f.opts(&out)
	opts.Yes = true
	stopped := ""
	opts.StopDaemon = func(dataDir string) error {
		stopped = dataDir
		return nil
	}
	rep, err := Uninstall(context.Background(), opts)
	if err != nil {
		t.Fatalf("Uninstall: %v", err)
	}
	if stopped != f.dataDir {
		t.Errorf("StopDaemon called with %q, want %q", stopped, f.dataDir)
	}
	if !rep.StoppedDaemon || !rep.DaemonAlive {
		t.Errorf("daemon flags: %+v", rep)
	}
	o := out.String()
	if !strings.Contains(o, "daemon:  running (pid 424242) — will be stopped first") {
		t.Errorf("plan must announce the daemon:\n%s", o)
	}
	if !strings.Contains(o, "stopped daemon (pid 424242)") {
		t.Errorf("stop report missing:\n%s", o)
	}
	// removal proceeded after the stop
	f.mustGone(t, f.bin, f.dataDir, filepath.Join(f.dataDir, "daemon.json"))
	f.assertExternalIntact(t)
}

func TestUninstallDaemonStopFailureRefusesEverything(t *testing.T) {
	f := buildFixture(t)
	fakeDaemon(t, f.dataDir, 424242)
	var out bytes.Buffer
	opts := f.opts(&out)
	opts.Yes = true
	opts.StopDaemon = func(string) error { return errors.New("stub stop failed") }
	rep, err := Uninstall(context.Background(), opts)
	if err == nil || !strings.Contains(err.Error(), "could not be stopped") {
		t.Fatalf("err = %v", err)
	}
	if !strings.Contains(err.Error(), "nothing was removed") {
		t.Errorf("error must promise nothing was removed: %v", err)
	}
	if rep.StoppedDaemon {
		t.Error("StoppedDaemon must stay false on failure")
	}
	// nothing removed: the orphan-process safety invariant
	f.mustExist(t, f.bin, f.dataDir, filepath.Join(f.dataDir, "hf_token"), f.cfgPath)
	f.assertExternalIntact(t)
}

func TestUninstallStaleDaemonStateIsJustAFile(t *testing.T) {
	f := buildFixture(t)
	// daemon.json pointing at a dead port: not alive, no stop needed
	if err := serve.WriteState(f.dataDir, serve.State{
		PID: 424243, Host: "127.0.0.1", Port: 1, StartedAt: 1,
	}); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	opts := f.opts(&out)
	opts.Yes = true
	opts.StopDaemon = func(string) error {
		t.Fatal("stale state must not trigger a stop")
		return nil
	}
	rep, err := Uninstall(context.Background(), opts)
	if err != nil {
		t.Fatalf("Uninstall: %v", err)
	}
	if rep.DaemonAlive || rep.StoppedDaemon {
		t.Errorf("stale daemon flags: %+v", rep)
	}
	if !strings.Contains(out.String(), "stale daemon.json will be removed") {
		t.Errorf("stale line missing:\n%s", out.String())
	}
	f.mustGone(t, f.dataDir)
}

// ---- graceful empty -----------------------------------------------------------

func TestUninstallNothingToRemove(t *testing.T) {
	root := t.TempDir()
	var out bytes.Buffer
	opts := UninstallOpts{
		Executable: filepath.Join(root, "absent", "stone-llama"),
		DataDir:    filepath.Join(root, "absent", "share", "stone-llama"),
		ConfigPath: filepath.Join(root, "absent", "cfg", "stone-llama", "config.json"),
		Stdout:     &out,
	}
	rep, err := Uninstall(context.Background(), opts) // no Yes: empty plan never asks
	if err != nil {
		t.Fatalf("Uninstall on empty layout: %v", err)
	}
	if len(rep.Removed) != 0 {
		t.Errorf("removed %+v", rep.Removed)
	}
	if !strings.Contains(out.String(), "nothing to remove") {
		t.Errorf("output:\n%s", out.String())
	}
}
