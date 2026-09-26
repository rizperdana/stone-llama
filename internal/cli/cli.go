// Package cli dispatches stone-llama subcommands.
package cli

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"text/tabwriter"
	"time"

	"github.com/rizperdana/stone-llama/internal/autofit"
	"github.com/rizperdana/stone-llama/internal/brand"
	"github.com/rizperdana/stone-llama/internal/config"
	"github.com/rizperdana/stone-llama/internal/doctor"
	"github.com/rizperdana/stone-llama/internal/hf"
	"github.com/rizperdana/stone-llama/internal/pull"
	"github.com/rizperdana/stone-llama/internal/selfmgmt"
	"github.com/rizperdana/stone-llama/internal/serve"
	"github.com/rizperdana/stone-llama/internal/setup"
	"github.com/rizperdana/stone-llama/internal/store"
)

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
		fmt.Fprintln(stdout, brand.Banner(version, runtime.GOOS+"/"+runtime.GOARCH))
		return 0
	case "doctor":
		return runDoctor(rest, stdout, stderr)
	case "list":
		return runList(rest, stdout, stderr)
	case "rm":
		return runRm(rest, stdout, stderr)
	case "import":
		return runImport(rest, stdout, stderr)
	case "pull":
		return runPull(rest, stdin, stdout, stderr)
	case "fit":
		return runFit(rest, stdin, stdout, stderr)
	case "rank":
		return runRank(rest, stdout, stderr)
	case "setup":
		return runSetup(rest, stdin, stdout, stderr)
	case "login":
		return runLogin(rest, stdin, stdout, stderr)
	case "serve":
		return runServe(rest, stdout, stderr)
	case "ps":
		return runPs(rest, stdout, stderr)
	case "stop":
		return runStop(rest, stdout, stderr)
	case "run":
		return runRun(rest, stdin, stdout, stderr)
	case "update":
		return runUpdate(rest, version, stdin, stdout, stderr)
	case "uninstall":
		return runUninstall(rest, stdin, stdout, stderr)
	}
	fmt.Fprintf(stderr, "stone-llama: unknown command %q\n\n", cmd)
	fmt.Fprint(stderr, usage())
	return 2
}

// runRun runs the interactive/one-shot REPL (M6): ensure the daemon,
// load the model through /-/load (autofit prints), stream completions.
// /bye unloads — supervised mode only; attach leaves the upstream's
// model alone (its lifecycle is not ours).
func runRun(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	usageLine := `usage: stone-llama run <model> [--ctx N] [--cache-mode M] [--no-autofit] [-p "prompt"]`
	model, cacheMode, oneShot := "", "", ""
	ctxN, noAutofit := 0, false
	// the loop reassigns i when consuming flag values (explicit classic loop)
	need := func(i *int, flag string) (string, bool) {
		if *i+1 >= len(args) {
			fmt.Fprintf(stderr, "stone-llama run: %s needs a value\n", flag)
			return "", false
		}
		*i++
		return args[*i], true
	}
	for i := 0; i < len(args); i++ {
		a := args[i]
		var val string
		var ok bool
		switch {
		case a == "--ctx" || strings.HasPrefix(a, "--ctx="):
			if strings.HasPrefix(a, "--ctx=") {
				val = strings.TrimPrefix(a, "--ctx=")
			} else if val, ok = need(&i, "--ctx"); !ok {
				return 2
			}
			n, aerr := strconv.Atoi(val)
			if aerr != nil || n < 0 {
				fmt.Fprintf(stderr, "stone-llama run: invalid --ctx %q\n", val)
				return 2
			}
			ctxN = n
		case a == "--cache-mode" || strings.HasPrefix(a, "--cache-mode="):
			if strings.HasPrefix(a, "--cache-mode=") {
				val = strings.TrimPrefix(a, "--cache-mode=")
			} else if val, ok = need(&i, "--cache-mode"); !ok {
				return 2
			}
			cacheMode = val
		case a == "--no-autofit":
			noAutofit = true
		case a == "-p" || a == "--prompt" || strings.HasPrefix(a, "--prompt="):
			if strings.HasPrefix(a, "--prompt=") {
				val = strings.TrimPrefix(a, "--prompt=")
			} else if val, ok = need(&i, "-p"); !ok {
				return 2
			}
			oneShot = val
		case a == "-h" || a == "--help":
			fmt.Fprintln(stdout, usageLine)
			return 0
		case strings.HasPrefix(a, "-"):
			fmt.Fprintf(stderr, "stone-llama run: unknown flag %q\n%s\n", a, usageLine)
			return 2
		default:
			if model != "" {
				fmt.Fprintln(stderr, usageLine)
				return 2
			}
			model = a
		}
	}
	if model == "" {
		fmt.Fprintln(stderr, usageLine)
		return 2
	}

	dataDir := config.DataDir()
	_, prevErr := serve.ReadState(dataDir)
	started := errors.Is(prevErr, serve.ErrNoDaemon)
	if _, err := serve.EnsureDaemon(dataDir, !noAutostart(), 30*time.Second); err != nil {
		fmt.Fprintf(stderr, "stone-llama run: %v%s\n", err, backendStartHelp(err.Error()))
		return 1
	}
	if started {
		fmt.Fprintln(stdout, "starting runtime…")
	}
	ctx := context.Background()
	st, err := serve.Query(dataDir)
	if err != nil {
		fmt.Fprintf(stderr, "stone-llama run: %v\n", err)
		return 1
	}
	token := ""
	if state, serr := serve.ReadState(dataDir); serr == nil ||
		(errors.Is(serr, serve.ErrCorruptState) && state.Token != "") {
		token = state.Token
	}
	if st.Model != model {
		st, err = serve.Load(ctx, dataDir, serve.LoadRequest{
			Model:     model,
			Ctx:       ctxN,
			CacheMode: cacheMode,
			NoAutofit: noAutofit,
		})
		if err != nil {
			fmt.Fprintf(stderr, "stone-llama run: %v\n", err)
			return 1
		}
		if !noAutofit && st.CacheMode != "" {
			fmt.Fprintf(stdout, "autofit: cache %s, max_seq_len %d\n", st.CacheMode, st.Ctx)
		}
		if s := st.FitSummary; s != "" {
			if strings.Contains(s, "warning") {
				fmt.Fprintf(stdout, "  %s\n", s)
			} else {
				fmt.Fprintf(stdout, "  %s ✓\n", s)
			}
		}
	}
	if st.Model == "" {
		fmt.Fprintln(stderr, "stone-llama run: no model is loaded")
		return 1
	}

	// one streamed completion; tokens are printed as they arrive.
	ask := func(prompt string) error {
		// max_tokens is required by the pinned backend: omitting it
		// aborts every completion ("Completion aborted", live-verified
		// 2026-09-26). 0 = generate to EOS, no cap.
		body, _ := json.Marshal(map[string]any{"model": st.Model, "prompt": prompt, "stream": true, "max_tokens": 0})
		req, rerr := http.NewRequest(http.MethodPost, "http://"+st.Addr()+"/v1/completions", bytes.NewReader(body))
		if rerr != nil {
			return rerr
		}
		req.Header.Set("Content-Type", "application/json")
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		resp, derr := http.DefaultClient.Do(req)
		if derr != nil {
			return derr
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			b, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
			return errors.New(serveEnvelope(b, resp.StatusCode))
		}
		br := bufio.NewReader(resp.Body)
		for {
			line, rerr := br.ReadString('\n')
			trimmed := strings.TrimSpace(line)
			if strings.HasPrefix(trimmed, "data:") {
				payload := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
				if payload == "[DONE]" {
					fmt.Fprintln(stdout)
					return nil
				}
				var chunk struct {
					Choices []struct {
						Text string `json:"text"`
					} `json:"choices"`
					Error *struct {
						Message string `json:"message"`
					} `json:"error"`
				}
				if json.Unmarshal([]byte(payload), &chunk) == nil {
					if chunk.Error != nil && chunk.Error.Message != "" {
						return errors.New(chunk.Error.Message)
					}
					if len(chunk.Choices) > 0 {
						io.WriteString(stdout, chunk.Choices[0].Text)
					}
				}
			}
			if rerr != nil {
				if rerr == io.EOF {
					return nil
				}
				return rerr
			}
		}
	}

	if oneShot != "" {
		if err := ask(oneShot); err != nil {
			fmt.Fprintf(stderr, "stone-llama run: %v\n", err)
			return 1
		}
	} else {
		in := bufio.NewReader(stdin)
		for {
			fmt.Fprint(stdout, ">>> ")
			line, rerr := in.ReadString('\n')
			text := strings.TrimSpace(line)
			if text == "" {
				if rerr != nil {
					break
				}
				continue
			}
			if text == "/bye" {
				break
			}
			if strings.HasPrefix(text, "/") {
				fmt.Fprintln(stdout, "unknown command (try /bye to quit)")
				continue
			}
			if err := ask(text); err != nil {
				fmt.Fprintf(stderr, "stone-llama run: %v\n", err)
			}
			if rerr != nil {
				break
			}
		}
	}
	if st.Mode != "attach" {
		if err := serve.Unload(ctx, dataDir); err == nil {
			fmt.Fprintln(stdout, "unloaded.")
		}
	}
	return 0
}

// serveEnvelope pulls the {"error":{"message"}} envelope out of a
// non-200 body.
func serveEnvelope(b []byte, code int) string {
	var env struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(b, &env) == nil && env.Error.Message != "" {
		return env.Error.Message
	}
	return fmt.Sprintf("HTTP %d: %s", code, strings.TrimSpace(string(b)))
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

func runList(args []string, stdout, stderr io.Writer) int {
	wantEst := false
	for _, a := range args {
		switch a {
		case "--estimate", "-estimate":
			wantEst = true
		default:
			fmt.Fprintf(stderr, "stone-llama list: unknown argument %q (only --estimate)\n", a)
			return 2
		}
	}
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
	estGPU, estVRAM := "", 0
	if wantEst {
		rep, err := doctor.Probe()
		switch {
		case err != nil:
			fmt.Fprintf(stderr, "stone-llama list: estimates unavailable: %v\n", err)
		case !rep.Ready():
			fmt.Fprintln(stderr, "stone-llama list: estimates unavailable: no ready GPU — run 'stone-llama doctor'")
		default:
			estGPU, estVRAM = rep.GPUs[0].Name, rep.GPUs[0].VRAMMiB
		}
	}
	w := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	if wantEst {
		fmt.Fprintln(w, "NAME\tQUANT\tSIZE\tVERDICT\tEST T/S\tEST PRE T/S\tSOURCE")
		for _, m := range models {
			dec, pre := localEstimate(m, estGPU, estVRAM, cfg)
			fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
				m.Name, m.Quant, pull.HumanBytes(m.SizeBytes), formatVerdict(m.Verdict), dec, pre, m.Source)
		}
		w.Flush()
		if estGPU != "" {
			fmt.Fprintln(stdout, "~ values are estimates [est] from metadata — not measured (calibration pending)")
		}
		return 0
	}
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
	if len(pos) != 1 {
		fmt.Fprintln(stderr, "usage: stone-llama import <dir> [--name <model>]")
		return 2
	}
	if name == "" {
		name = filepath.Base(filepath.Clean(pos[0]))
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

// gpuPreamble loads config and probes a ready GPU for model commands.
// ok=false means a cmd-scoped error was already printed to stderr.
func gpuPreamble(cmd string, stdout, stderr io.Writer) (*config.Config, doctor.Report, bool) {
	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(stderr, "stone-llama %s: %v\n", cmd, err)
		return nil, doctor.Report{}, false
	}
	rep, err := doctor.Probe()
	if err != nil {
		fmt.Fprintf(stderr, "stone-llama %s: %v\n", cmd, err)
		return nil, doctor.Report{}, false
	}
	if !rep.Ready() {
		fmt.Fprint(stdout, rep.Format())
		fmt.Fprintf(stderr, "stone-llama %s: needs a working NVIDIA GPU — run 'stone-llama doctor'\n", cmd)
		return nil, doctor.Report{}, false
	}
	return &cfg, rep, true
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
		fmt.Fprintln(stderr, "usage: stone-llama pull <repo[@branch][:quant]> [--force] [--quiet] [--yes]")
		return 2
	}
	cfg, rep, ok := gpuPreamble("pull", stdout, stderr)
	if !ok {
		return 1
	}
	token := hf.LoadToken(os.Getenv("HF_TOKEN"), tokenFilePath())
	_, err := pull.Run(pull.Options{
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
		BaseURL:     hfBaseURL(),
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

// isTerminal reports whether x is a real interactive TTY. A plain
// ModeCharDevice check is wrong: /dev/null is a character device but
// not a terminal, so `setup </dev/null` used to prompt. isTTY is the
// platform ioctl (TCGETS/TIOCGETA/GetConsoleMode).
func isTerminal(x any) bool {
	f, ok := x.(*os.File)
	if !ok {
		return false
	}
	return isTTY(f)
}

// runSetup provisions the pinned Python runtime. Consent-gated (A7):
// the plan prints first, then an explicit confirmation; --yes for
// scripts. The plan covers multi-GB downloads — run it deliberately.
func runSetup(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	yes, extra := false, ""
	for _, a := range args {
		switch a {
		case "--yes", "-y", "-yes":
			yes = true
		case "--cu12":
			extra = "cu12"
		case "--cu13":
			extra = "cu13"
		default:
			fmt.Fprintln(stderr, "usage: stone-llama setup [--yes] [--cu12|--cu13]")
			return 2
		}
	}
	opts := setup.Options{
		RuntimeDir: filepath.Join(config.DataDir(), "runtime"),
		Extra:      extra,
		Yes:        yes,
		Stdout:     stdout,
		Confirm: func(setup.Plan) bool {
			if !isTerminal(stdin) {
				fmt.Fprintln(stderr, "stone-llama setup: non-interactive — pass --yes to confirm the plan")
				return false
			}
			fmt.Fprint(stdout, "Proceed? [y/N] ")
			line, err := bufio.NewReader(stdin).ReadString('\n')
			if err != nil && line == "" {
				return false
			}
			return strings.HasPrefix(strings.ToLower(strings.TrimSpace(line)), "y")
		},
	}
	if err := setup.Run(opts); err != nil {
		fmt.Fprintf(stderr, "stone-llama setup: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "runtime ready under %s\n", opts.RuntimeDir)
	return 0
}

// noAutostart is the STONE_LLAMA_NO_AUTOSTART contract (ARCHITECTURE
// §2): exactly "1" opts out. Shared by every auto-starting command so
// the semantics cannot drift (run silently ignored it — H4).
func noAutostart() bool {
	return os.Getenv("STONE_LLAMA_NO_AUTOSTART") == "1"
}

// backendStartHelp is the single, once-per-failure remedy block for
// "the supervised backend could not start": least friction first —
// attach an already-running server, provision with setup, adoption
// (not yet supported). Only flags that actually exist are named, and
// it prints at most once per error (never as a repeating wall).
func backendStartHelp(msg string) string {
	// the captured auto-start tail can already carry the child's own
	// remedy block — never print a second copy (live demo caught this).
	if strings.Contains(msg, "remedies, least friction first") {
		return ""
	}
	if !strings.Contains(msg, "runtime incomplete") &&
		!strings.Contains(msg, "auto-start failed") &&
		!strings.Contains(msg, "still starting after") {
		return ""
	}
	return "\nremedies, least friction first:\n" +
		"  1. attach to a running OpenAI-compatible server: stone-llama serve --attach host:port --key-file <file> ('stone-llama run' works against an attached daemon)\n" +
		"  2. provision the pinned runtime: stone-llama setup\n" +
		"  3. adopting an existing local install (a working tabbyAPI checkout/venv) is not supported yet"
}

// runServe runs the daemon in the foreground (M5): a supervised child
// by default, or --attach against an existing OpenAI-compatible server
// (schemeless host:port accepted). Ctrl-C/SIGTERM shuts down cleanly.
func runServe(args []string, stdout, stderr io.Writer) int {
	attach, keyFile := "", ""
	port := 0
	usageLine := "usage: stone-llama serve [--attach <host:port|url>] [--port <n>] [--key-file <path>]"
	// the loop reassigns i when consuming flag values (explicit classic loop)
	need := func(i *int, flag string) (string, bool) {
		if *i+1 >= len(args) {
			fmt.Fprintf(stderr, "stone-llama serve: %s needs a value\n", flag)
			return "", false
		}
		*i++
		return args[*i], true
	}
	setPort := func(v string) bool {
		p, err := strconv.Atoi(v)
		if err != nil || p < 1 || p > 65535 {
			fmt.Fprintf(stderr, "stone-llama serve: invalid --port %q\n", v)
			return false
		}
		port = p
		return true
	}
	for i := 0; i < len(args); i++ {
		var ok bool
		switch a := args[i]; {
		case a == "--attach" || a == "-attach":
			if attach, ok = need(&i, "--attach"); !ok {
				return 2
			}
		case strings.HasPrefix(a, "--attach="):
			attach = strings.TrimPrefix(a, "--attach=")
		case a == "--key-file" || a == "-key-file":
			if keyFile, ok = need(&i, "--key-file"); !ok {
				return 2
			}
		case strings.HasPrefix(a, "--key-file="):
			keyFile = strings.TrimPrefix(a, "--key-file=")
		case a == "--port" || a == "-port":
			v, ok2 := need(&i, "--port")
			if !ok2 || !setPort(v) {
				return 2
			}
		case strings.HasPrefix(a, "--port="):
			if !setPort(strings.TrimPrefix(a, "--port=")) {
				return 2
			}
		default:
			fmt.Fprintln(stderr, usageLine)
			return 2
		}
	}
	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(stderr, "stone-llama serve: %v\n", err)
		return 1
	}
	if port == 0 {
		port = cfg.Port
	}
	if keyFile == "" {
		keyFile = cfg.UpstreamKeyFile
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	err = serve.Serve(ctx, serve.Options{
		Host:            cfg.Host,
		Port:            port,
		Attach:          attach,
		RuntimeDir:      cfg.RuntimeDir,
		ModelsDir:       cfg.ModelsDir,
		DataDir:         config.DataDir(),
		UpstreamKeyFile: keyFile,
		PinnedCommit:    setup.TabbyPin(),
		Fit:             cfg.Autofit,
		Stdout:          stdout,
		Stderr:          stderr,
	})
	if err != nil && !errors.Is(err, context.Canceled) {
		fmt.Fprintf(stderr, "stone-llama serve: %v%s\n", err, backendStartHelp(err.Error()))
		return 1
	}
	return 0
}

// runPs shows the running daemon (M5). Auto-starts one detached unless
// STONE_LLAMA_NO_AUTOSTART=1; the status round trip carries readiness,
// model, and the VRAM peak sample.
func runPs(args []string, stdout, stderr io.Writer) int {
	if len(args) > 0 {
		fmt.Fprintln(stderr, "stone-llama ps: takes no arguments")
		return 2
	}
	dataDir := config.DataDir()
	if _, err := serve.EnsureDaemon(dataDir, !noAutostart(), 30*time.Second); err != nil {
		fmt.Fprintf(stderr, "stone-llama ps: %v%s\n", err, backendStartHelp(err.Error()))
		return 1
	}
	// corrupt state that still reached a live daemon (H3): show the
	// anomaly alongside the status instead of hiding the daemon.
	if _, serr := serve.ReadState(dataDir); errors.Is(serr, serve.ErrCorruptState) {
		fmt.Fprintf(stderr, "stone-llama ps: warning: %v — pid/port salvaged; restart the daemon (stop, then start) to rewrite the file\n", serr)
	}
	st, err := serve.Query(dataDir)
	if err != nil {
		fmt.Fprintf(stderr, "stone-llama ps: daemon is starting (state written, status not up yet): %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "stone-llama: running (pid %d)\n", st.PID)
	fmt.Fprintf(stdout, "  address:  %s\n", net.JoinHostPort(st.Host, strconv.Itoa(st.Port)))
	if st.Mode == "attach" {
		fmt.Fprintf(stdout, "  mode:     attach (%s)\n", st.Attach)
		fmt.Fprintln(stdout, "            stone-llama does not own this process — it proxies only; load/unload is the upstream's")
	} else {
		fmt.Fprintf(stdout, "  mode:     supervised (child pid %d)\n", st.ChildPID)
	}
	fmt.Fprintf(stdout, "  ready:    %s\n", map[bool]string{true: "yes", false: "no"}[st.Ready])
	model := st.Model
	if model == "" {
		model = "(none loaded)"
	}
	fmt.Fprintf(stdout, "  model:    %s\n", model)
	if st.Ctx > 0 || st.CacheMode != "" {
		fmt.Fprintf(stdout, "  ctx:      %d  cache: %s\n", st.Ctx, st.CacheMode)
	}
	if st.VRAMPeak > 0 {
		fmt.Fprintf(stdout, "  vram peak: %d MiB\n", st.VRAMPeak)
	}
	fmt.Fprintf(stdout, "  uptime:   %s\n", time.Duration(st.UptimeS)*time.Second)
	return 0
}

// runStop terminates the daemon (M5); absent state is a friendly no-op.
func runStop(args []string, stdout, stderr io.Writer) int {
	if len(args) > 0 {
		fmt.Fprintln(stderr, "stone-llama stop: takes no arguments")
		return 2
	}
	dataDir := config.DataDir()
	if _, err := serve.ReadState(dataDir); errors.Is(err, serve.ErrNoDaemon) {
		fmt.Fprintln(stdout, "stone-llama: not running")
		return 0
	} else if err != nil && !errors.Is(err, serve.ErrCorruptState) {
		fmt.Fprintf(stderr, "stone-llama stop: %v\n", err)
		return 1
	}
	// corrupt state still reaches Stop (H3): it salvages pid/port and
	// the H2 gates decide whether signalling is safe; unusable corrupt
	// state errors there with the file kept and explained.
	if err := serve.Stop(dataDir, 8*time.Second); err != nil {
		fmt.Fprintf(stderr, "stone-llama stop: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, "stone-llama: stopped")
	return 0
}

// runUpdate self-updates from the latest GitHub release (or a pinned
// --version tag). --check reports without downloading; every download
// is size-announced and consent-gated (A7), --yes for scripts, and the
// SHA-256 in checksums.txt must match before anything is replaced.
func runUpdate(args []string, version string, stdin io.Reader, stdout, stderr io.Writer) int {
	usageLine := "usage: stone-llama update [--check] [--version <tag>] [--force] [--yes]"
	opts := selfmgmt.UpdateOpts{Version: version, Stdout: stdout}
	need := func(i *int, flag string) (string, bool) {
		if *i+1 >= len(args) {
			fmt.Fprintf(stderr, "stone-llama update: %s needs a value\n", flag)
			return "", false
		}
		*i++
		return args[*i], true
	}
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case a == "--check":
			opts.Check = true
		case a == "--force":
			opts.Force = true
		case a == "--yes" || a == "-y":
			opts.Yes = true
		case a == "--version" || strings.HasPrefix(a, "--version="):
			val, ok := "", false
			if strings.HasPrefix(a, "--version=") {
				val = strings.TrimPrefix(a, "--version=")
			} else if val, ok = need(&i, "--version"); !ok {
				return 2
			}
			opts.Want = val
		case a == "-h" || a == "--help":
			fmt.Fprintln(stdout, usageLine)
			return 0
		default:
			fmt.Fprintf(stderr, "stone-llama update: unknown flag %q\n%s\n", a, usageLine)
			return 2
		}
	}
	opts.Confirm = func() bool {
		if !isTerminal(stdin) {
			fmt.Fprintln(stderr, "stone-llama update: non-interactive — pass --yes to confirm the download")
			return false
		}
		fmt.Fprint(stdout, "Proceed? [y/N] ")
		line, err := bufio.NewReader(stdin).ReadString('\n')
		if err != nil && line == "" {
			return false
		}
		return strings.HasPrefix(strings.ToLower(strings.TrimSpace(line)), "y")
	}
	if _, err := selfmgmt.Update(context.Background(), opts); err != nil {
		fmt.Fprintf(stderr, "stone-llama update: %v\n", err)
		return 1
	}
	return 0
}

// runUninstall removes the binary and the tool's data completely (A7):
// the plan always prints first and consent is required (--yes for
// scripts); --dry-run stops after the plan, --keep-models preserves
// downloaded weights.
func runUninstall(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	usageLine := "usage: stone-llama uninstall [--dry-run] [--keep-models] [--yes]"
	opts := selfmgmt.UninstallOpts{Stdout: stdout}
	for _, a := range args {
		switch a {
		case "--dry-run":
			opts.DryRun = true
		case "--keep-models":
			opts.KeepModels = true
		case "--yes", "-y":
			opts.Yes = true
		case "-h", "--help":
			fmt.Fprintln(stdout, usageLine)
			return 0
		default:
			fmt.Fprintf(stderr, "stone-llama uninstall: unknown flag %q\n%s\n", a, usageLine)
			return 2
		}
	}
	opts.Confirm = func(selfmgmt.Report) bool {
		if !isTerminal(stdin) {
			fmt.Fprintln(stderr, "stone-llama uninstall: non-interactive — pass --yes to confirm the plan")
			return false
		}
		fmt.Fprint(stdout, "Proceed? [y/N] ")
		line, err := bufio.NewReader(stdin).ReadString('\n')
		if err != nil && line == "" {
			return false
		}
		return strings.HasPrefix(strings.ToLower(strings.TrimSpace(line)), "y")
	}
	if _, err := selfmgmt.Uninstall(context.Background(), opts); err != nil {
		fmt.Fprintf(stderr, "stone-llama uninstall: %v\n", err)
		return 1
	}
	return 0
}

func usage() string {
	return `stone-llama — local model server + CLI (ExLlamaV3 + TabbyAPI), OpenAI-compatible API

Usage:
  stone-llama <command>

Commands:
  version     print version
  doctor      check GPU, driver, and runtime readiness
  list        list installed models (--estimate: fit + tok/s columns)
  rm          remove a model
  import      symlink an existing model dir into the store
  pull        download a model <repo[@branch][:quant]>
  fit         fit verdict + ctx/cache pick + tok/s estimate (no download)
  rank        rank candidates by fit/speed (--collection|--file [--ratings])
  login       set a HuggingFace token
  setup       provision the pinned Python runtime (consent-gated)
  serve       start the OpenAI-compatible API [--attach host:port] [--port n]
  ps          show the running daemon + loaded model
  stop        stop the daemon
  run <model> [--ctx N] [--cache-mode M] [--no-autofit] [-p prompt]   chat (streams)
  update      self-update to the latest release [--check] [--version tag] [--force] [--yes]
  uninstall   remove the binary + its data completely [--dry-run] [--keep-models] [--yes]
`
}
