// Package cli dispatches stone-llama subcommands.
package cli

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"text/tabwriter"

	"github.com/rizperdana/stone-llama/internal/autofit"
	"github.com/rizperdana/stone-llama/internal/config"
	"github.com/rizperdana/stone-llama/internal/doctor"
	"github.com/rizperdana/stone-llama/internal/hf"
	"github.com/rizperdana/stone-llama/internal/pull"
	"github.com/rizperdana/stone-llama/internal/store"
)

// planned lists commands that exist in the product surface but are not
// implemented yet, mapped to the milestone that ships them (ARCHITECTURE.md §11).
var planned = map[string]string{
	"setup": "M4",
	"serve": "M5", "ps": "M5", "stop": "M5",
	"run": "M6",
}

// Run executes one CLI invocation and returns the process exit code.
func Run(args []string, version string, stdin io.Reader, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprint(stdout, usage())
		return 0
	}
	cmd, rest := args[0], args[1:]
	switch cmd {
	case "help", "-h", "--help":
		fmt.Fprint(stdout, usage())
		return 0
	case "version", "-v", "--version":
		fmt.Fprintf(stdout, "stone-llama %s (%s/%s)\n", version, runtime.GOOS, runtime.GOARCH)
		return 0
	case "doctor":
		return runDoctor(rest, stdout, stderr)
	case "list":
		return runList(stdout, stderr)
	case "rm":
		return runRm(rest, stdout, stderr)
	case "import":
		return runImport(rest, stdout, stderr)
	case "pull":
		return runPull(rest, stdin, stdout, stderr)
	case "login":
		return runLogin(rest, stdin, stdout, stderr)
	}
	if ms, ok := planned[cmd]; ok {
		fmt.Fprintf(stderr, "stone-llama %s: not implemented yet (ships in %s)\n", cmd, ms)
		return 2
	}
	fmt.Fprintf(stderr, "stone-llama: unknown command %q\n\n", cmd)
	fmt.Fprint(stderr, usage())
	return 2
}

func runDoctor(args []string, stdout, stderr io.Writer) int {
	if len(args) > 0 {
		fmt.Fprintln(stderr, "stone-llama doctor: takes no arguments")
		return 2
	}
	if _, err := config.Load(); err != nil {
		fmt.Fprintf(stderr, "stone-llama doctor: %v\n", err)
		return 1
	}
	rep, err := doctor.Probe()
	if err != nil {
		fmt.Fprintf(stderr, "stone-llama doctor: %v\n", err)
		return 1
	}
	fmt.Fprint(stdout, rep.Format())
	if !rep.Ready() {
		return 1
	}
	return 0
}

func runList(stdout, stderr io.Writer) int {
	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(stderr, "stone-llama list: %v\n", err)
		return 1
	}
	models, err := store.Scan(cfg.ModelsDir)
	if err != nil {
		fmt.Fprintf(stderr, "stone-llama list: %v\n", err)
		return 1
	}
	if len(models) == 0 {
		fmt.Fprintln(stdout, "no models — pull one with 'stone-llama pull <model>'")
		return 0
	}
	w := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "NAME\tQUANT\tSIZE\tVERDICT\tSOURCE")
	for _, m := range models {
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n", m.Name, m.Quant, pull.HumanBytes(m.SizeBytes), formatVerdict(m.Verdict), m.Source)
	}
	w.Flush()
	return 0
}

func formatVerdict(v *store.Verdict) string {
	if v == nil {
		return "-"
	}
	s := v.Status
	if v.MaxCtx > 0 {
		s += fmt.Sprintf(":%d", v.MaxCtx)
	}
	if v.CacheMode != "" {
		s += " " + v.CacheMode
	}
	return s
}

func runRm(args []string, stdout, stderr io.Writer) int {
	if len(args) != 1 {
		fmt.Fprintln(stderr, "usage: stone-llama rm <model>")
		return 2
	}
	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(stderr, "stone-llama rm: %v\n", err)
		return 1
	}
	if err := store.Remove(cfg.ModelsDir, args[0]); err != nil {
		fmt.Fprintf(stderr, "stone-llama rm: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "removed %s\n", args[0])
	return 0
}

func runImport(args []string, stdout, stderr io.Writer) int {
	// --name may appear before or after the dir; stdlib flag stops at
	// the first positional argument, so scan manually.
	name := ""
	var pos []string
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case a == "--name" || a == "-name":
			if i+1 >= len(args) {
				fmt.Fprintln(stderr, "stone-llama import: --name needs a value")
				return 2
			}
			name = args[i+1]
			i++
		case strings.HasPrefix(a, "--name="):
			name = strings.TrimPrefix(a, "--name=")
		default:
			pos = append(pos, a)
		}
	}
	if len(pos) != 1 || name == "" {
		fmt.Fprintln(stderr, "usage: stone-llama import <dir> --name <model>")
		return 2
	}
	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(stderr, "stone-llama import: %v\n", err)
		return 1
	}
	link, err := store.Import(cfg.ModelsDir, pos[0], name)
	if err != nil {
		fmt.Fprintf(stderr, "stone-llama import: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "imported %s → %s\n", name, link)
	return 0
}

func runPull(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	force, quiet, yes := false, false, false
	var pos []string
	for _, a := range args {
		switch a {
		case "--force", "-force":
			force = true
		case "--quiet", "-quiet":
			quiet = true
		case "--yes", "-y", "-yes":
			yes = true
		default:
			pos = append(pos, a)
		}
	}
	if len(pos) != 1 {
		fmt.Fprintln(stderr, "usage: stone-llama pull <model>[:quant] [--force] [--quiet] [--yes]")
		return 2
	}
	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(stderr, "stone-llama pull: %v\n", err)
		return 1
	}
	rep, err := doctor.Probe()
	if err != nil {
		fmt.Fprintf(stderr, "stone-llama pull: %v\n", err)
		return 1
	}
	if !rep.Ready() {
		fmt.Fprint(stdout, rep.Format())
		fmt.Fprintln(stderr, "stone-llama pull: needs a working NVIDIA GPU — run 'stone-llama doctor'")
		return 1
	}
	token := hf.LoadToken(os.Getenv("HF_TOKEN"), filepath.Join(config.DataDir(), "hf_token"))
	_, err = pull.Run(pull.Options{
		ModelsDir:   cfg.ModelsDir,
		Ref:         pos[0],
		Force:       force,
		Quiet:       quiet,
		Yes:         yes,
		Out:         stdout,
		Stdin:       stdin,
		Interactive: isTerminal(stdin),
		TTY:         isTerminal(stdout),
		Token:       token,
		VRAMMiB:     rep.GPUs[0].VRAMMiB,
		GPUName:     rep.GPUs[0].Name,
		Autofit: autofit.Options{
			HeadroomMiB:    cfg.Autofit.HeadroomMiB,
			WorkspaceMiB:   cfg.Autofit.WorkspaceMiB,
			CtxHeadroomMiB: cfg.Autofit.CtxHeadroomMiB,
			OverheadMiB:    cfg.Autofit.OverheadMiB,
			MinCtx:         cfg.Autofit.MinCtx,
		},
	})
	if err != nil {
		fmt.Fprintf(stderr, "stone-llama pull: %v\n", err)
		return 1
	}
	return 0
}

func runLogin(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	if len(args) > 0 {
		fmt.Fprintln(stderr, "stone-llama login: takes no arguments (the token is read from stdin, never argv)")
		return 2
	}
	if f, ok := stdin.(*os.File); ok {
		// hide paste echo when stdin is a real terminal
		cmd := exec.Command("stty", "-echo")
		cmd.Stdin = f
		_ = cmd.Run()
		defer func() {
			c := exec.Command("stty", "echo")
			c.Stdin = f
			_ = c.Run()
		}()
	}
	fmt.Fprint(stdout, "Paste a HuggingFace token (input hidden): ")
	line, err := bufio.NewReader(stdin).ReadString('\n')
	if err != nil && line == "" {
		fmt.Fprintln(stderr, "stone-llama login: no token entered")
		return 1
	}
	tok := strings.TrimSpace(line)
	if !strings.HasPrefix(tok, "hf_") || len(tok) < 8 {
		fmt.Fprintln(stderr, "stone-llama login: that doesn't look like a HuggingFace token (expected hf_… prefix)")
		return 1
	}
	dir := config.DataDir()
	if err := os.MkdirAll(dir, 0o700); err != nil {
		fmt.Fprintf(stderr, "stone-llama login: %v\n", err)
		return 1
	}
	path := filepath.Join(dir, "hf_token")
	if err := os.WriteFile(path, []byte(tok+"\n"), 0o600); err != nil {
		fmt.Fprintf(stderr, "stone-llama login: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "token saved to %s (0600 — never printed, never in argv)\n", path)
	return 0
}

// isTerminal reports whether x is a character device (interactive TTY).
func isTerminal(x any) bool {
	f, ok := x.(*os.File)
	if !ok {
		return false
	}
	st, err := f.Stat()
	return err == nil && st.Mode()&os.ModeCharDevice != 0
}

func usage() string {
	return `stone-llama — ollama-like frontend for ExLlamaV3 + TabbyAPI

Usage:
  stone-llama <command>

Commands:
  version     print version
  doctor      check GPU, driver, and runtime readiness
  list        list installed models
  rm          remove a model
  import      symlink an existing model dir into the store
  pull        download a model
  login       set a HuggingFace token
  setup       provision the Python runtime     (M4)
  serve       start the OpenAI-compatible API  (M5)
  ps          show the loaded model            (M5)
  stop        stop the daemon                  (M5)
  run         chat with a model in the CLI     (M6)
`
}
