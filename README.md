# stone-llama

**Ollama-style CLI + local model server for ExLlamaV3 + TabbyAPI** — one small Go
binary that pulls EXL3 models, auto-fits context to your VRAM, and serves an
OpenAI-compatible API.

**[Website](https://rizperdana.github.io/stone-llama/) · [Install](#install) · [Quickstart](docs/QUICKSTART.md)**

## Install

NVIDIA GPU · Linux x86_64 · driver ≥ 570 · Go ≥ 1.24 to build. The binary is
static — Go is only needed to build.

```bash
git clone https://github.com/rizperdana/stone-llama
cd stone-llama
go build -o stone-llama ./cmd/stone-llama
install -Dm755 stone-llama ~/.local/bin/stone-llama
```

Prerequisites, the one-line installer, verification, uninstall, and the
consent-gated `setup` step (≈ 1 GB announced before you confirm):
[docs/INSTALLATION.md](docs/INSTALLATION.md).

## Commands

| Command | Purpose |
|---|---|
| `doctor` | GPU/driver/runtime verdict, no changes made |
| `fit <repo>` | gate + predicted tok/s `[est]` from metadata — **no model download** |
| `list [--estimate]` | models with size, quant, gate verdict, source |
| `rank --collection <owner/name>` | rank a HF collection by fit + predicted tok/s `[est]` |
| `pull <repo[@branch][:quant>] [--force] [--quiet] [--yes]` | gate → consent → resumable download, sha256-verified |
| `import <dir>` | symlink an existing model dir in (zero copy) |
| `rm <model>` | remove a model (import symlinks: link only, target untouched) |
| `login` | HuggingFace token for gated repos (stored 0600) |
| `setup [--yes] [--cu12\|--cu13] [--adopt <path>] [--provision]` | provision the pinned Python runtime (consent-gated, resumable; no flag → `cu13`, so a `cu12` machine must pass `--cu12` — the driver→extra mapping lives in `doctor`, which prints it: ≥580 `cu13`, ≥570 `cu12`) — or **reuse** an already-present TabbyAPI venv when it passes the validation gate (0 bytes downloaded; `--provision` forces the classic path) |
| `serve [<model>] [--attach host:port] [--port n] [--key-file path]` | daemon: OpenAI-compatible API (attach = existing TabbyAPI upstream; the model positional is ignored in attach mode) |
| `ps` / `stop` | loaded model + sampled VRAM peak of the supervised child (line omitted in attach mode — no child to sample) / stop the daemon |
| `run <model> [--continue] [--history] [-p "prompt"]` | streaming CLI chat with persistent history — `--continue` resumes the newest saved session (by file mtime), `--history` prints that session's turns as JSON and exits without touching the backend |
| `version` | version |
| `update [--check] [--version <tag>] [--force] [--yes]` | self-update the binary — SHA-256 in `checksums.txt` must match before anything is replaced |
| `uninstall [--dry-run] [--keep-models] [--yes]` | remove the install (weights optional to keep) |

## Key flags & env

Flags that always win over autofit: `--ctx N`, `--cache-mode Q8|Q4|FP16|"2,2"`,
`--no-autofit`. `serve --draft-mode ngram|off` selects speculative drafting (default
off — no `draft_model` block is rendered; A/B run 2026-10-07 (n=1164/arm,
3 reps) measured **INCONCLUSIVE — within noise (0.00% delta)**, so it stays off and
no speedup is claimed;
`run --draft-mode` parses and validates the value only — it cannot reach an already-
running daemon and has no effect. Config `engine_env` adds extra environment for the
supervised backend child (parent env preserved; config keys win). Env: `STONE_LLAMA_HOST`,
`STONE_LLAMA_PORT`, `STONE_LLAMA_MODELS_DIR`, `STONE_LLAMA_CONFIG`,
`STONE_LLAMA_NO_AUTOSTART`, `HF_TOKEN`. `serve --backend tabby|llama` selects the
supervised engine (default `tabby` = TabbyAPI + ExLlamaV3; `llama` = `llama-server`
over GGUF weights — status in `docs/ARCHITECTURE.md`); an unknown value is a usage
error, never a silent fallback. Env `STONE_LLAMA_BACKEND` and config `backend` do the
same. State (`models/`, `runtime/`, `daemon.json`, logs, `chats/`) is
`$XDG_DATA_HOME/stone-llama`, config `$XDG_CONFIG_HOME/stone-llama/config.json`.
`chats/<model>/chat-<id>.json` holds `run` history (files 0600, dirs 0700); when a
session would overflow the context window it is trimmed to fit — **destructively:
dropped turns leave the file too** (deliberate — they could not be replayed anyway).

## Platform support

exllamav3 has **no CPU path, no AMD/ROCm path, no Apple/Metal path**. That is not
configurable:

| Machine | Works? |
|---|---|
| NVIDIA GPU, Linux x86_64, driver ≥ 570 | ✅ (runtime `cu12`) |
| NVIDIA GPU, Linux x86_64, driver ≥ 580 | ✅ (runtime `cu13`) |
| NVIDIA GPU, driver < 570 | ❌ upgrade the driver |
| AMD / Intel GPU | ❌ |
| CPU-only machine | ❌ |
| Apple Silicon / macOS | ❌ `doctor`/`list`/`fit` run, can **never serve** |
| Windows | ⚠️ binary published, **untested at runtime** |
| Linux arm64 | ❌ not v1 |

**Linux/amd64 is the only supported platform in v1.** `stone-llama doctor` prints this
verdict — with your actual GPU and driver numbers — before anything gets installed.

## Documentation

| Doc | What's in it |
|---|---|
| [Website](https://rizperdana.github.io/stone-llama/) | homepage — model compatibility table (fits + predicted tok/s for a 4 GB card) |
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | prerequisites, install paths, `setup` provisioning, uninstall |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | first-run walkthrough with real output |
| [docs/AGENT-SETUP.md](docs/AGENT-SETUP.md) | copy-paste prompt for a coding agent |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | full plan, milestones, open gaps |
| [docs/RELEASE.md](docs/RELEASE.md) | release process, provenance, artifact shape |

## License

MIT — see [LICENSE](LICENSE). Third-party runtime components: see
[THIRD-PARTY.md](THIRD-PARTY.md).

**AGPL note:** we neither distribute nor modify TabbyAPI — `setup` downloads upstream
sources to your disk on request, and stone-llama talks to it over HTTP as a separate
process.