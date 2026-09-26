// Package cli's zero-config backend resolution for `run` (and `serve`).
//
// `stone-llama run <model>` resolves its backend silently, cheapest first:
//
//  1. our own daemon is already alive (state file + health marker) -> use it.
//  2. a compatible upstream is already serving on the configured/default
//     TabbyAPI address (127.0.0.1:5002) -> auto-attach (no daemon spawned,
//     nothing downloaded). This wins before any supervised spawn because a
//     live upstream already holds the GPU on a 4 GB card and double-loading
//     would OOM (project decision: rule 2 wins).
//  3. a provisioned or adopted runtime is present -> start the supervised
//     daemon transparently (today's behaviour).
//  4. nothing -> one actionable remedy message.
//
// Precedence for every knob: flag > env > file > default.
package cli

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/rizperdana/stone-llama/internal/config"
	"github.com/rizperdana/stone-llama/internal/serve"
	"github.com/rizperdana/stone-llama/internal/setup"
)

// Decision is the backend a run/serve command resolved.
type Decision struct {
	// Kind is "daemon" (our stone-llama daemon is live, possibly proxying) or
	// "direct" (stream straight to a live upstream, no daemon spawned).
	Kind string
	// Status is the live daemon status (Kind=="daemon" only).
	Status serve.Status
	// Base is the upstream base URL (Kind=="direct" only), e.g. http://127.0.0.1:5002.
	Base string
	// Token is the upstream bearer for direct mode (NEVER printed).
	Token string
	// Model is the model actually in use after name resolution.
	Model string
}

// Test seams — every branch of ResolveBackend is exercisable with fakes;
// production wiring points at the real implementations below.
var (
	queryStatusFn     = serve.Query
	probeUpstreamFn   = serve.ProbeAttach
	upstreamModelFn   = serve.UpstreamModel
	startSupervisedFn = func(dataDir string, timeout time.Duration) (serve.State, error) {
		return serve.EnsureDaemon(dataDir, true, timeout)
	}
	discoverCheckoutFn = discoverTabbyCheckout
)

// runReadyTimeout is the readiness bound for the auto-attach probe (matches
// the seam contract: any answer from /v1/models within the bound = serving).
const runReadyTimeout = 30 * time.Second

func runtimeDirOf(cfg config.Config, dataDir string) string {
	if cfg.RuntimeDir != "" {
		return cfg.RuntimeDir
	}
	return filepath.Join(dataDir, "runtime")
}

// discoverTabbyCheckout reuses the adoption layer's detection/precedence
// function (§4 probes 1-5) — it is NOT re-implemented here — and returns the
// first candidate carrying a TabbyAPI checkout path ("" when none).
func discoverTabbyCheckout(runtimeDir string) string {
	for _, c := range setup.Detect(setup.DetectInput{RuntimeDir: runtimeDir}) {
		if c.Checkout != "" {
			return c.Checkout
		}
	}
	return ""
}

// upstreamBase resolves the auto-attach probe address: an explicit --attach
// flag first, then the configured/default upstream (default 127.0.0.1:5002).
func upstreamBase(cfg config.Config, attachFlag string) string {
	if attachFlag != "" {
		return withScheme(attachFlag)
	}
	return withScheme(cfg.Upstream)
}

// withScheme ensures a schemeless host:port gets an http:// default (the
// documented attach form).
func withScheme(s string) string {
	if s == "" {
		return "http://127.0.0.1:5002"
	}
	if !strings.Contains(s, "://") {
		return "http://" + s
	}
	return s
}

// baseHostPort renders an upstream base URL as host:port for human messages.
func baseHostPort(base string) string {
	if u, err := url.Parse(base); err == nil && u.Host != "" {
		return u.Host
	}
	return strings.TrimPrefix(strings.TrimPrefix(base, "http://"), "https://")
}

// runtimeUsable reports whether a provisioned or adopted runtime is present
// (spawnChild's python check; adoption symlinks runtime/venv onto a candidate
// venv, so the same stat covers both).
func runtimeUsable(runtimeDir string) bool {
	if fi, err := os.Stat(filepath.Join(runtimeDir, "venv", "bin", "python")); err == nil {
		return !fi.IsDir()
	}
	return false
}

// resolveUpstreamTokens returns candidate bearer VALUES in precedence order
// (flag > config > env > runtime key file > the discovered TabbyAPI
// checkout's api_tokens.yml). Every source is collected, not just the
// first: a stale <DataDir>/runtime/upstream_key must not shadow a live
// checkout key — ResolveBackend walks the candidates and only stops when
// the upstream stops answering 401/403 (the same order as before, tried
// through). A path that cannot be read is skipped, never fatal.
//
// Values are never printed; diagnostics redact them as <key>.
func resolveUpstreamTokens(dataDir string, cfg config.Config, keyFileFlag string) []string {
	type src struct{ path, value string }
	var srcs []src
	if keyFileFlag != "" {
		srcs = append(srcs, src{path: keyFileFlag})
	}
	if cfg.UpstreamKeyFile != "" {
		srcs = append(srcs, src{path: cfg.UpstreamKeyFile})
	}
	if v := os.Getenv("STONE_LLAMA_UPSTREAM_KEY_FILE"); v != "" {
		srcs = append(srcs, src{path: v})
	}
	if v := os.Getenv("STONE_LLAMA_UPSTREAM_KEY"); v != "" {
		srcs = append(srcs, src{value: v})
	}
	srcs = append(srcs, src{path: filepath.Join(dataDir, "runtime", "upstream_key")})
	if co := discoverCheckoutFn(runtimeDirOf(cfg, dataDir)); co != "" {
		if p := serve.TabbyAPITokensFile(co); fileExists(p) {
			if k, err := serve.ParseTabbyAPIToken(p); err == nil && k != "" {
				srcs = append(srcs, src{value: k})
			}
		}
	}
	var out []string
	for _, s := range srcs {
		v := s.value
		if v == "" {
			if b, err := os.ReadFile(s.path); err == nil {
				v = strings.TrimSpace(string(b))
			}
		}
		if v != "" {
			out = append(out, v)
		}
	}
	return out
}

func fileExists(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && !fi.IsDir()
}

// ResolveBackend implements the zero-config resolution order (see file doc).
// noAttach forces the supervised path (--local); noAutostart() blocks the
// supervised spawn but NOT direct attach (attaching starts nothing).
func ResolveBackend(ctx context.Context, stdout, stderr io.Writer, dataDir string, cfg config.Config, attachFlag, keyFileFlag string, noAttach bool) (Decision, error) {
	// 1) our daemon already alive — use it (existing behaviour).
	if st, err := queryStatusFn(dataDir); err == nil {
		return Decision{Kind: "daemon", Status: st, Model: st.Model}, nil
	}

	// 2) a compatible upstream already serving on the configured/default
	//    address — auto-attach before any supervised spawn (rule 2).
	upstream := upstreamBase(cfg, attachFlag)
	if !noAttach {
		pr := probeUpstreamFn(ctx, upstream, runReadyTimeout)
		if pr.Ok && pr.Err == nil {
			// The probe counts any HTTP answer as serving — a 401 there
			// means "serving, wants a key", not "no model". Resolve the
			// loaded model with the discovered bearer, walking candidates
			// while the upstream keeps answering 401/403 (a stale runtime
			// key must not shadow the checkout's live key).
			tokens := resolveUpstreamTokens(dataDir, cfg, keyFileFlag)
			if len(tokens) == 0 {
				tokens = []string{""} // nothing discovered: ask unauthenticated
			}
			var model, token string
			var merr error
			for _, tok := range tokens {
				token = tok
				model, merr = upstreamModelFn(ctx, upstream, tok)
				if !errors.Is(merr, serve.ErrUpstreamAuth) {
					break // accepted, or a failure no other key can fix
				}
			}
			if errors.Is(merr, serve.ErrUpstreamAuth) {
				return Decision{}, fmt.Errorf(
					"attached upstream %s refused the model query (401/403 — bearer key required or wrong); no model claim made — pass --key-file <path> or set STONE_LLAMA_UPSTREAM_KEY",
					baseHostPort(upstream))
			}
			if merr == nil && model == "" {
				return Decision{}, fmt.Errorf(
					"attached upstream %s answered the model query with no model id — load one at the upstream server, or run `stone-llama run <model> --local` once a runtime is available",
					baseHostPort(upstream))
			}
			// Unreadable (merr != nil): model stays "" — run keeps the name
			// the user typed and lets the upstream answer it; its error is
			// more honest than a guess about what is loaded.
			fmt.Fprintf(stdout, "using the TabbyAPI already running on %s — nothing downloaded\n", baseHostPort(upstream))
			return Decision{Kind: "direct", Base: upstream, Token: token, Model: model}, nil
		}
	}

	// 3) a provisioned or adopted runtime is present — start the supervised
	//    daemon transparently (today's behaviour).
	if !noAutostart() {
		rt := runtimeDirOf(cfg, dataDir)
		if runtimeUsable(rt) {
			if _, err := startSupervisedFn(dataDir, runReadyTimeout); err != nil {
				return Decision{}, fmt.Errorf("auto-start failed: %v%s", err, backendStartRemedy(err.Error(), upstream))
			}
			if st, qerr := queryStatusFn(dataDir); qerr == nil {
				return Decision{Kind: "daemon", Status: st, Model: st.Model}, nil
			}
			return Decision{}, fmt.Errorf("daemon started but is not responding yet — check %s", filepath.Join(dataDir, "logs", "daemon.log"))
		}
	}

	// 4) nothing available — one actionable message, remedies least-friction first.
	return Decision{}, nothingAvailableErr(upstream, dataDir)
}

// backendStartRemedy is the single, once-per-error remedy block for a
// supervised start that failed: least friction first — attach a live
// upstream (auto-discoverable), adopt an existing checkout, provision the
// pinned runtime. It returns "" when errMsg already carries a remedy block
// (the child's own log tail), so the block prints at most once.
func backendStartRemedy(errMsg, upstream string) string {
	if strings.Contains(errMsg, "remedies, least friction first") {
		return ""
	}
	return "\nremedies, least friction first:\n" +
		"  1. attach to the running server on " + baseHostPort(upstream) + ": stone-llama run <model> --attach " + baseHostPort(upstream) + " --key-file <path>\n" +
		"  2. adopt an existing local TabbyAPI: stone-llama setup --adopt <checkout>\n" +
		"  3. provision the pinned runtime: stone-llama setup\n"
}

// nothingAvailableErr is the single message for "nothing is available" (step 4).
// It reuses the adoption detection/precedence function to mention an adoptable
// checkout if one is detectable, never re-implementing detection.
func nothingAvailableErr(upstream, dataDir string) error {
	parts := []string{
		"nothing is available — no daemon is running, " +
			baseHostPort(upstream) + " is not serving, and no runtime is provisioned or adopted",
		"remedies, least friction first:",
	}
	if co := discoverCheckoutFn(filepath.Join(dataDir, "runtime")); co != "" {
		parts = append(parts,
			"  1. adopt the detected TabbyAPI checkout: stone-llama setup --adopt "+co,
			"  2. provision the pinned runtime: stone-llama setup",
		)
	} else {
		parts = append(parts,
			"  1. provision the pinned runtime: stone-llama setup",
			"  2. adopt an existing TabbyAPI checkout: stone-llama setup --adopt <path>",
		)
	}
	parts = append(parts,
		"  3. attach to a running server: stone-llama run <model> --attach "+baseHostPort(upstream)+" --key-file <path>",
	)
	return errors.New("stone-llama run: " + strings.Join(parts, "\n"))
}
