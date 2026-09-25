// Package cli dispatches stone-llama subcommands.
package cli

import (
	"fmt"
	"io"
	"runtime"
	"strings"
	"text/tabwriter"

	"github.com/rizperdana/stone-llama/internal/config"
	"github.com/rizperdana/stone-llama/internal/doctor"
	"github.com/rizperdana/stone-llama/internal/store"
)

// planned lists commands that exist in the product surface but are not
// implemented yet, mapped to the milestone that ships them (ARCHITECTURE.md §11).
var planned = map[string]string{
	"pull": "M2", "login": "M2",
	"setup": "M4",
	"serve": "M5", "ps": "M5", "stop": "M5",
	"run": "M6",
}

// Run executes one CLI invocation and returns the process exit code.
func Run(args []string, version string, stdout, stderr io.Writer) int {
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
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n", m.Name, m.Quant, humanBytes(m.SizeBytes), formatVerdict(m.Verdict), m.Source)
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

func humanBytes(n int64) string {
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
  pull        download a model                 (M2)
  login       set a HuggingFace token          (M2)
  setup       provision the Python runtime     (M4)
  serve       start the OpenAI-compatible API  (M5)
  ps          show the loaded model            (M5)
  stop        stop the daemon                  (M5)
  run         chat with a model in the CLI     (M6)
`
}
