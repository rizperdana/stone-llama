// Package cli dispatches stone-llama subcommands.
package cli

import (
	"fmt"
	"io"
	"runtime"

	"github.com/rizperdana/stone-llama/internal/config"
	"github.com/rizperdana/stone-llama/internal/doctor"
)

// planned lists commands that exist in the product surface but are not
// implemented yet, mapped to the milestone that ships them (ARCHITECTURE.md §11).
var planned = map[string]string{
	"list": "M1", "rm": "M1", "import": "M1",
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

func usage() string {
	return `stone-llama — ollama-like frontend for ExLlamaV3 + TabbyAPI

Usage:
  stone-llama <command>

Commands:
  version     print version
  doctor      check GPU, driver, and runtime readiness
  list        list installed models            (M1)
  rm          remove a model                   (M1)
  import      import an existing model dir     (M1)
  pull        download a model                 (M2)
  login       set a HuggingFace token          (M2)
  setup       provision the Python runtime     (M4)
  serve       start the OpenAI-compatible API  (M5)
  ps          show the loaded model            (M5)
  stop        stop the daemon                  (M5)
  run         chat with a model in the CLI     (M6)
`
}
