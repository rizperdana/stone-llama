<div align="center">
  <img src="assets/stone-llama.png" alt="stone-llama icon" width="160">
</div>

# stone-llama

**Ollama-style CLI + local model server for ExLlamaV3 + TabbyAPI** — one small Go
binary that pulls EXL3 models, auto-fits context to your VRAM, and serves an
OpenAI-compatible API.

**[Website](https://rizperdana.github.io/stone-llama/) · [Install](#install) · [Quickstart](docs/QUICKSTART.md)**

Three things it does that a stock config cannot:

- **VRAM-aware autofit** — computes a safe context window and cache mode from your GPU
  and the model's own `config.json`, prints the arithmetic, warns when the margin is thin.
- **Pre-download fit gate** — `fit` and `pull` check architecture, EXL3 format and VRAM fit
  from metadata (KB of config, plus at most ~16 MiB of safetensors headers) *before any
  gigabyte moves*, and refuse with the numbers.
- **Auto-tuned, terse chat** — `run` builds a template-correct, bounded profile from the
  model's own metadata (sampling, thinking, one-line default system message). Every
  default is a flag: `--max-tokens` (0 = unbounded), `--temperature`, `--top-p`,
  `--system` / `--no-system`, `--thinking` / `--no-thinking`.

**Not a better ollama.** exllamav3 — the engine stone-llama wraps — has no CPU, AMD/ROCm
or Apple path. On CPU/AMD/Apple, use ollama with GGUF; see [Platform support](#platform-support).

## Install

NVIDIA GPU · Linux x86_64 · driver ≥ 570 · Go ≥ 1.24 to build.

```bash
git clone https://github.com/rizperdana/stone-llama
cd stone-llama
go build -o stone-llama ./cmd/stone-llama
install -Dm755 stone-llama ~/.local/bin/stone-llama
```

The binary is static — Go is only needed to build. Prerequisites, the one-line
installer, verification, uninstall, and the consent-gated `setup` step (≈ 1 GB
announced before you confirm): [docs/INSTALLATION.md](docs/INSTALLATION.md).

## Quickstart

```bash
stone-llama doctor                        # GPU + driver check, picks cu12/cu13 — no downloads
stone-llama setup                         # reuse a verified runtime if present (0 B), else provision Python/torch; sizes announced first
stone-llama fit async0x42/Qwen3-1.7B-exl3_4.0bpw    # gate + tok/s estimate from HF metadata — no download
stone-llama pull async0x42/Qwen3-1.7B-exl3_4.0bpw   # gate runs first, then resumable download
stone-llama run Qwen3-1.7B-exl3_4.0bpw    # CLI chat (prints the autofit decision)
stone-llama serve                         # OpenAI-compatible API on 127.0.0.1:5111
```

Step-by-step with expected output and a working `curl` request:
[docs/QUICKSTART.md](docs/QUICKSTART.md). Handing this to a coding agent? Copy
[docs/AGENT-SETUP.md](docs/AGENT-SETUP.md).

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

## Commands

| Command | Purpose |
|---|---|
| `doctor` | GPU/driver/runtime verdict, no changes made |
| `fit <repo>` | gate + predicted tok/s `[est]` from metadata — **no model download** |
| `list [--estimate]` | models with size, quant, gate verdict, source |
| `rank --collection <owner/name>` | rank a HF collection by fit + predicted tok/s `[est]` |
| `pull <model>[:tag]` | gate → consent → resumable download, sha256-verified |
| `import <dir>` | symlink an existing model dir in (zero copy) |
| `rm <model>` | remove a model (import symlinks: link only, target untouched) |
| `login` | HuggingFace token for gated repos (stored 0600) |
| `setup [--yes] [--cu12\|--cu13] [--adopt <path>] [--provision]` | provision the pinned Python runtime (consent-gated, resumable; extra picked from the driver unless overridden) — or **reuse** an already-present TabbyAPI venv when it passes the validation gate (0 bytes downloaded; `--provision` forces the classic path) |
| `serve [--attach host:port] [--port n] [--key-file path]` | daemon: OpenAI-compatible API (attach = existing TabbyAPI upstream) |
| `ps` / `stop` | loaded model + live VRAM / stop the daemon |
| `run <model>` | streaming CLI chat |
| `version` | version |
| `update [--check] [--version <tag>] [--force] [--yes]` | self-update the binary — SHA-256 in `checksums.txt` must match before anything is replaced |
| `uninstall [--dry-run] [--keep-models] [--yes]` | remove the install (weights optional to keep) |

Flags that always win over autofit: `--ctx N`, `--cache-mode Q8|Q4|FP16|"2,2"`,
`--no-autofit`. Env: `STONE_LLAMA_HOST`, `STONE_LLAMA_PORT`, `STONE_LLAMA_MODELS_DIR`,
`STONE_LLAMA_CONFIG`, `STONE_LLAMA_NO_AUTOSTART`, `HF_TOKEN`.
State (`models/`, `runtime/`, `daemon.json`, logs) is `$XDG_DATA_HOME/stone-llama`,
config `$XDG_CONFIG_HOME/stone-llama/config.json`. **Testing? Point `XDG_DATA_HOME` at
an empty directory** or `ps`/`run`/`serve` will find the live daemon.

`[est]` = computed from metadata + your VRAM, not measured. The one measured anchor:
42.7 tok/s (SmolLM3-3B, reference card).

## Real terminal output

Verbatim, 2026-09-26 (RTX 3050 Laptop, 4096&nbsp;MiB, driver 580.178.04, Linux/amd64)
— the gate refusing before any weight byte moves:

```console
$ stone-llama pull ggml-org/SmolLM3-3B-GGUF
stone-llama pull: repo ggml-org/SmolLM3-3B-GGUF has no config.json — not an EXL3 model layout
```

Every capture — `doctor`, `fit` (one that fits, one that does not), `rank`,
`list --estimate`, live [`serve`](docs/screenshots/serve.txt) and
[`run`](docs/screenshots/run.txt), [`version`](docs/screenshots/version.txt) — is in
[docs/screenshots/](docs/screenshots/), rendered in [cli.svg](docs/screenshots/cli.svg).

## How autofit picks your context

Everything comes from the model's own `config.json` plus your VRAM: the ladder tries
`ctx` = trained max, halves it down to 4096, and prefers better cache quality at each
tier (FP16 → Q8 → Q4). It never exceeds `max_position_embeddings`, the arithmetic is
always printed, and a thin margin warns. If prefill OOMs (CUDA OOM *before* the first
token), drop `--ctx` one tier; if nothing fits you get the full breakdown plus the
largest ctx that *would* fit. Derivation, headroom terms and a worked example:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## The pre-download fit gate

`fit` (and the gate inside `pull`) reads `config.json`, the file tree and at most
~16 MiB of safetensors headers — never weights — then checks **architecture** (known to
the installed exllamav3), **quant format** (`exl3` only; `exl2` and GGUF-only repos are
refused) and **fit** (the arithmetic above, against *your* GPU): proceed, warn, or refuse
with the numbers. An unreadable header warns, never silently passes. Caught mistakes
cost KB instead of gigabytes; the verdict is saved to `manifest.json` and shown by
`stone-llama list`. Detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Third-party components

The binary has **zero third-party Go dependencies** — standard library only (`go.mod`
has no `require` block, and there is no `go.sum`). Every component below is a *runtime*
one: stone-llama **redistributes none** of them; `setup` fetches each from upstream on
your machine, sizes announced, confirmation required. Per-package evidence in
[THIRD-PARTY.md](THIRD-PARTY.md):

| Component | License |
|---|---|
| uv | Apache-2.0 **OR** MIT (dual) |
| CPython 3.12 | PSF-2.0 |
| PyTorch (+cu130) | BSD-3-Clause |
| exllamav3 | MIT |
| Triton | MIT |
| Flash-linear-attention | MIT |
| NVIDIA CUDA/cuDNN (18 wheels) | Proprietary — NVIDIA EULA |
| TabbyAPI (`f07131cd`) | **AGPL-3.0** |

**AGPL note:** we neither distribute nor modify TabbyAPI — `setup` downloads upstream
sources to your disk on request, and stone-llama talks to it over HTTP as a separate
process.

## Out of scope for v1

- CPU/AMD/Apple inference; Windows/macOS/arm64 claims → [Platform support](#platform-support)
- Multi-GPU `gpu_split` autofit (single GPU; TabbyAPI autosplit passes through)
- Concurrent multi-model serving
- GGUF, EXL2, draft/speculative models
- Forking or vendoring TabbyAPI

## Documentation

| Doc | What's in it |
|---|---|
| [Website](https://rizperdana.github.io/stone-llama/) | homepage — model compatibility table (fits + predicted tok/s for a 4 GB card) |
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | prerequisites, install paths, `setup` provisioning, uninstall |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | first-run walkthrough with real output |
| [docs/AGENT-SETUP.md](docs/AGENT-SETUP.md) | copy-paste prompt for a coding agent |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | full plan, milestones, open gaps |

## License

MIT — see [LICENSE](LICENSE). Third-party runtime components: see
[THIRD-PARTY.md](THIRD-PARTY.md).
