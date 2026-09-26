# AGENTS.md

Read this first. Facts below are true at the current `main`; verify, don't assume.

## What this is

- One static Go binary: an ollama-style CLI + loopback OpenAI-compatible server for
  ExLlamaV3, which runs as **TabbyAPI — a separate process we spawn and talk to over HTTP**.
- The low-spec goal is the product constraint: reference card RTX 3050 Laptop,
  4096 MiB VRAM. Most trade-offs answer to 4 GB-class cards.
- Its only interface with the world is "serves an OpenAI-compatible API"; it is
  deliberately self-contained and integrates with nothing else.

## Hard rules — each one already caused harm

| Rule | Why |
|---|---|
| **Never run state-wide git commands** in a shared tree: `git stash`, `git stash -u`, `git clean`, `git checkout -- .`, `git reset --hard`, `git restore .`. Stage explicit paths; one writer per file. | a `git stash -u` here destroyed files once |
| **No large downloads without explicit user authorization**: model weights, the `setup` runtime (≈ 950 MB announced on this machine: torch 531 MB + exllamav3 419 MB + uv 24 MB, plus CPython + the TabbyAPI checkout; unpacked venv is larger — `docs/INSTALLATION.md`). Announce real sizes; get consent; never pass `--yes` on the user's behalf. | multi-GB bytes moved on someone's behalf once |
| **Never restart, stop, kill, or reconfigure live services**: 9Router on port **20128** and the TabbyAPI on **5002** serve real work. Never occupy port **5001**. Attach/read-only only. | they serve real work |
| **Secrets never in argv, logs, or stdout.** Keys go via `--key-file` (or stdin for `login`). `hf_token`, `daemon.json`, `api_tokens.yml` are `0600`; their dirs `0700`. | tokens were printed to logs once |
| **Honesty**: estimates carry `[est]`; never fabricate terminal output or benchmarks; never claim platform support the engine lacks. | the docs quote real captures |

## Platform truth (keep it exact)

| Platform | Truth |
|---|---|
| NVIDIA + Linux/amd64, driver ≥ 570 | **the only supported platform** |
| Windows | binary **builds** (CI cross-compiles) but is **untested at runtime** |
| macOS | binary runs `doctor`/`list`/`fit` but can **never serve** (no CUDA) |
| linux/arm64, darwin/* | cross-compiled by `make dist`; untested — no hardware (v1: unsupported) |
| CPU / AMD / Apple | exllamav3 has no path — advise ollama + GGUF, never a workaround |

## The architectural seam

- The proxy depends on exactly three things from the backend (the `Conn` contract,
  `internal/serve/serve.go`):
  1. an `http.Handler` at the base URL speaking the OpenAI-compatible surface;
  2. `GET {base}/v1/models` answers within the readiness timeout (default 120 s) —
     any status counts as ready;
  3. SSE passthrough with `FlushInterval: -1`.
- **Everything TabbyAPI-specific lives only in `internal/serve/tabby_backend.go` +
  `internal/serve/tabbyconf.go`**: YAML schema, `start.py`/venv/argv/cwd, the
  `api_tokens.yml` name, the readiness probe, the product name in user-facing strings
  (routed via `Conn.Name`). A TabbyAPI detail anywhere else is a bug.
- **Licence**: TabbyAPI is AGPL-3.0. We neither vendor, patch, nor redistribute it —
  `setup` fetches upstream at the user's request and we talk to it over HTTP as a
  separate process. Vendoring would change that analysis; it is a deliberate non-goal
  (`README.md`, `THIRD-PARTY.md`).

## Build, test, gates

```sh
gofmt -l internal cmd              # must be empty
go vet ./...
go test ./... -count=1
CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o stone-llama ./cmd/stone-llama   # static, stripped
make dist                          # 5 bundles + dist/checksums.txt
```

- The `Makefile` release flags add `-X main.version=$(VERSION_TAG)` on top of the
  flags above; `make test` = vet + test.
- `make dist` builds `linux/amd64 linux/arm64 windows/amd64 darwin/amd64 darwin/arm64`
  (the cross-build targets; CI runs the same five as `-o /dev/null` compile checks in
  `.github/workflows/ci.yml`, plus version/--help/doctor smoke tests).
- **Build-tag split** keeps Unix-only syscalls from breaking the Windows cross-build:
  `internal/{serve,fslock,pull,setup}/…_{unix,windows}.go`, `internal/cli/tty_*.go`.
  New syscalls go behind a `//go:build` pair, stdlib `syscall` only — **no
  `golang.org/x/sys`, no new dependencies** (`go list -m all` prints just this module).

## Verifying a change

- The standard here: a change is proven by **running the thing**, not by a green unit
  test alone.
- **Never run the CLI against the live state dir or the live backend.** Isolation
  (read `internal/config/config.go`):
  - data dir = `$XDG_DATA_HOME/stone-llama`, else `~/.local/share/stone-llama`
  - config = `$STONE_LLAMA_CONFIG`, else `$XDG_CONFIG_HOME/stone-llama/config.json`,
    else `~/.config/stone-llama/config.json`

  ```sh
  export XDG_DATA_HOME=$(mktemp -d) XDG_CONFIG_HOME=$(mktemp -d)
  ./stone-llama list      # verified: prints "no models" while the live dir has models;
                          # live dir mtimes unchanged, temp dir gains nothing but a lock
  ./stone-llama doctor    # verified: read-only GPU verdict, exit 0
  ```
- **Untestable without authorization** (announce, ask, wait): `setup`'s real
  provisioning (~950 MB), any `pull` of weights, and non-attach `serve` / `run` —
  they need that runtime and would contend for the GPU the live server holds.
- **Escape hatch that needs nothing**: `stone-llama serve --attach 127.0.0.1:5002
  --key-file <file> --port <non-default>` proxies to the already-running server and
  spawns nothing (verified: banner + `/healthz` + proxied `/v1/models`). Always pass a
  non-default `--port`.

## Behavioural invariants

- **Zero-buffer SSE**: the proxy sets `FlushInterval: -1`; streaming first bytes must
  arrive while generation is running, not at completion. `internal/serve/serve_test.go`
  asserts chunks pass through in real time.
- **Pre-download gate**: architecture / quant format / fit are decided from KB of
  metadata **before any weight byte moves**, and refusals print the arithmetic, never a
  bare error (`internal/preflight/gate.go`).
- **Fit arithmetic**: `weights + ctx × KV-per-token + overhead + ctx-scaled headroom ≤
  VRAM`; KV cost per element: **FP16 = 2 B, Q8 = 1 B, Q4 = 0.5 B**. The growing terms
  are `[est]` and tunable (`autofit.workspace_mib`, `autofit.ctx_headroom_mib`).
- **Exit codes** (verified at HEAD): `0` ok · `1` runtime error (missing GPU, bad ref,
  no `config.json`) · `2` usage · `3` fit refusal.
- **User-facing error strings are quoted verbatim in the docs** (`README.md`,
  `docs/QUICKSTART.md`, `docs/screenshots/`, `site/`). Changing a string means updating
  those in the same change.

## Commits and branches

- Log convention: `feat(scope):`, `fix(scope):`, `chore:`, `docs:` (with scope when
  touched). The release workflow builds notes from commit messages
  (`--generate-notes`, `.github/workflows/release.yml`), so the prefix is load-bearing.
- Branch naming: `feature/<assignee>/<TICKET-ID>`. No `CONTRIBUTING.md` at this HEAD —
  this section is the convention.

## Repo map

| Path | Responsibility |
|---|---|
| `cmd/stone-llama/` | `main` → `cli.Run`, process exit code |
| `internal/cli/` | subcommand dispatch, flags, exit codes |
| `internal/config/` | config load + XDG/env path resolution |
| `internal/serve/` | daemon, reverse proxy, state; `tabby_backend.go`/`tabbyconf.go` = the seam |
| `internal/autofit/` | VRAM-fit ctx/cache decision from `config.json` + GPU |
| `internal/preflight/` | pre-download gate (arch / quant / fit) |
| `internal/pull/` | gate → consent → resumable sha256 download |
| `internal/hf/` | minimal HuggingFace client (KB-scale metadata) |
| `internal/store/` | models dir scan / import / rm, manifests |
| `internal/setup/` | pinned runtime provisioning (uv, CPython, wheels, TabbyAPI checkout) |
| `internal/doctor/` | GPU / driver / runtime-extra verdict, contention probe |
| `internal/estimate/` | tok/s prediction from metadata `[est]` |
| `internal/fslock/` | cross-process flock (daemon spawn, downloads) |
| `internal/brand/` | embedded icon for banners |
| `docs/` | ARCHITECTURE, QUICKSTART, INSTALLATION, RELEASE, AGENT-SETUP |
| `site/`, `scripts/`, `.github/workflows/` | homepage, `install.sh`, CI + release |

## Durable references

| Doc | What's in it |
|---|---|
| `docs/ARCHITECTURE.md` | design decisions, decision log, milestones |
| `docs/QUICKSTART.md` | real captured terminal output |
| `docs/INSTALLATION.md` | install paths, `setup` provisioning, sizes |
| `docs/RELEASE.md` | release process |
| `docs/AGENT-SETUP.md` | copy-paste prompt for standing a machine up |
| `THIRD-PARTY.md` | per-package licence evidence |
| `README.md` | platform matrix, command table, autofit + gate explanations |

## Tooling (optional): structural code queries

If your environment exposes the codebase-memory MCP (`search_graph`, `trace_path`,
`get_code_snippet`, `query_graph`), prefer it for structural questions — "who calls X",
"what depends on Y", snippet retrieval by symbol — and fall back to `grep`/`read` for
literal text, prose, and anything the graph reports as uncovered. It is optional
convenience only: the repo must build and be understandable without it, and the index
is not a source of truth — the source is.
