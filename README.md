<div align="center">
  <img src="assets/stone-llama.png" alt="stone-llama icon" width="160">
</div>

# stone-llama

**Ollama-style CLI + local model server for ExLlamaV3 + TabbyAPI** — one small Go
binary that pulls EXL3 models, auto-fits context to your VRAM, and serves an
OpenAI-compatible API that any client can point at.

**[Website](https://rizperdana.github.io/stone-llama/) · [Install](#install) · [Quickstart](docs/QUICKSTART.md)**

Two things it does that a stock config cannot:

- **VRAM-aware autofit** — computes a safe context window and cache mode from your GPU and
  the model's own `config.json`, prints the arithmetic, and warns when the margin is thin.
- **Pre-download fit gate** — `fit` and `pull` check architecture, EXL3 format and VRAM fit
  from metadata — KB of config, plus at most ~16 MiB of safetensors headers — *before any
  gigabyte moves*, and refuse with the numbers when the model will not run.

**Not a better ollama.** exllamav3 — the engine stone-llama wraps — has no CPU, AMD/ROCm or
Apple path. On CPU/AMD/Apple, use ollama with GGUF; see [Platform support](#platform-support).

## Install

Prerequisites: NVIDIA GPU, Linux x86_64, driver ≥ 570, Go ≥ 1.24 to build.

```bash
git clone https://github.com/rizperdana/stone-llama
cd stone-llama
go build -o stone-llama ./cmd/stone-llama
install -Dm755 stone-llama ~/.local/bin/stone-llama
```

The binary is static — Go is only needed to build. A pre-release (`v0.1.0-rc1`) is
published, and `scripts/install.sh` verifies the SHA-256 against the release's combined
`checksums.txt` before installing — see [docs/INSTALLATION.md](docs/INSTALLATION.md) for
the one-line installer or build-from-source, plus prerequisites, the consent-gated
`setup` step (≈ 1 GB announced before you confirm), verification and uninstall.

## Quickstart

```bash
stone-llama doctor                        # GPU + driver check, picks cu12/cu13 — no downloads
stone-llama setup                         # Python/torch runtime; announces sizes, asks before any byte
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
| NVIDIA GPU, Linux x86_64, driver ≥ 570 | ✅ (runtime extra `cu12`) |
| NVIDIA GPU, Linux x86_64, driver ≥ 580 | ✅ (runtime extra `cu13`) |
| NVIDIA GPU, driver < 570 | ❌ upgrade the driver |
| AMD / Intel GPU | ❌ |
| CPU-only machine | ❌ |
| Apple Silicon / macOS | ❌ can run `doctor`/`list`/`fit`, can **never serve** (no CUDA) |
| Windows | ⚠️ binary published but **untested at runtime** (daemonization differs) |
| Linux arm64 | ❌ not v1 |

**Linux/amd64 is the only supported platform in v1.** On CPU/AMD/Apple, use ollama with
GGUF instead. `stone-llama doctor` prints this verdict — with your actual GPU and driver
numbers — before anything gets installed.

## Commands

| Command | Purpose | Status |
|---|---|---|
| `stone-llama doctor` | GPU/driver/runtime verdict, no changes made | ✅ shipped |
| `stone-llama fit <repo>` | gate + predicted tok/s from HF metadata — **no model download** | ✅ shipped |
| `stone-llama list [--estimate]` | models with size, quant, gate verdict, source (+ tok/s estimates) | ✅ shipped |
| `stone-llama rank --collection <owner/name>` | rank a HF collection by fit + predicted tok/s (also `--file`, `--ratings`) | ✅ shipped |
| `stone-llama import <dir> [--name <n>]` | symlink an existing model dir in (zero copy; name defaults to the dir name) | ✅ shipped |
| `stone-llama rm <model>` | remove a model (import symlinks: link only, target untouched) | ✅ shipped |
| `stone-llama pull <model>[:tag]` | pre-download gate → consent → resumable download + sha256 verify | ✅ shipped |
| `stone-llama login` | HuggingFace token for gated repos (stored 0600) | ✅ shipped |
| `stone-llama setup [--yes] [--cu12\|--cu13]` | provision the pinned Python runtime (consent-gated, resumable; extra picked from driver unless overridden) | ✅ shipped |
| `stone-llama serve [--attach host:port] [--port n] [--key-file path]` | daemon: OpenAI-compatible API (attach = existing TabbyAPI upstream) | ✅ shipped |
| `stone-llama ps` | loaded model + live VRAM | ✅ shipped |
| `stone-llama stop` | stop the daemon | ✅ shipped |
| `stone-llama run <model>` | streaming CLI chat | ✅ shipped |
| `stone-llama version` | version | ✅ shipped |

`rank --file <refs.txt>` takes one model ref per line — the same `owner/name`
(or `owner/name:tag`) form `pull` accepts; blank lines and `#`-prefixed comment
lines are skipped, and a file with no refs is an error.

Flags that always win over autofit: `--ctx N`, `--cache-mode Q8|Q4|FP16|"2,2"`,
`--no-autofit`. Env: `STONE_LLAMA_HOST`, `STONE_LLAMA_PORT`, `STONE_LLAMA_MODELS_DIR`,
`STONE_LLAMA_CONFIG`, `STONE_LLAMA_NO_AUTOSTART`, `HF_TOKEN`, plus the XDG pair below.

State and config locations: state (`models/`, `runtime/`, `daemon.json`, logs) is
`$XDG_DATA_HOME/stone-llama`, falling back to `~/.local/share/stone-llama`; config is
`$XDG_CONFIG_HOME/stone-llama/config.json`, falling back to
`~/.config/stone-llama/config.json`, and `STONE_LLAMA_CONFIG` names an explicit config
file instead. **Testing? Point `XDG_DATA_HOME` at an empty directory** — otherwise
`ps`, `run` and `serve` find (and can start or stop) the daemon living in the default
state dir.

## Real terminal output

Captured verbatim from a live run on 2026-09-26 (RTX 3050 Laptop, 4096 MiB, driver
580.178.04, Linux/amd64). The `serve` and `run` transcripts are attach mode against a
live TabbyAPI. Raw files for every capture: [docs/screenshots/](docs/screenshots/).

```console
$ stone-llama doctor
GPU        NVIDIA GeForce RTX 3050 Laptop GPU
VRAM       4096 MiB
Driver     580.178.04
Runtime    cu13 extra

$ stone-llama list
NAME                    QUANT  SIZE      VERDICT  SOURCE
SmolLM3-3B-exl3_4.0bpw  -      1.84 GiB  -        imported
```

The same gate runs inside `pull`, before any weight byte moves:

```console
$ stone-llama pull ggml-org/SmolLM3-3B-GGUF
stone-llama pull: repo ggml-org/SmolLM3-3B-GGUF has no config.json — not an EXL3 model layout

$ printf 'n\n' | stone-llama pull async0x42/Qwen3-1.7B-exl3_4.0bpw   # declined: non-interactive, --yes required, 'n' never read
gate: arch  ✓ Qwen3ForCausalLM
gate: quant ✓ exl3
gate: fit   ⚠ weights 1491 + KV 1120 + overhead 128 = 2739 MiB (headroom 1344, budget 2752)
      warning: only 13 MiB margin above the 1344 MiB headroom (prefill workspace [est] included): multi-KB prompts can OOM during prefill on a used card — if you see CUDA OOM before the first token, drop --ctx
warnings above — review them before continuing
pulling async0x42/Qwen3-1.7B-exl3_4.0bpw → Qwen3-1.7B-exl3_4.0bpw: 11 files, 1.47 GiB (sha256-verified), 83.05 GiB free
stone-llama pull: non-interactive pull requires --yes to confirm the download
```

The second run was refused (non-interactive stdin without `--yes`), so nothing was fetched
beyond gate metadata — KB of config plus, for the format verdict, the safetensors headers
(see the gate section below); `list` above shows a model linked in with `import`, also no
download.
Model-download steps are deliberately omitted from the captures. The rest — `fit` on a model
that fits and one that does not, `rank`, `list --estimate`, `setup` preflight, plus the live
[`serve`](docs/screenshots/serve.txt) and [`run`](docs/screenshots/run.txt) transcripts and
the [`version`](docs/screenshots/version.txt) banner — is in [docs/screenshots/](docs/screenshots/),
rendered together in [cli.svg](docs/screenshots/cli.svg).

## How autofit picks your context

Everything comes from the model's own `config.json` plus your VRAM:

```
head_dim  = hidden_size / num_attention_heads                    # 2048/16 = 128
elems/tok = num_hidden_layers × 2 × num_key_value_heads × head_dim

load_mib    = weights_mib + ctx × elems/tok × bytes_per_element/2²⁰ + 128   # fixed overhead
headroom_mib(ctx) = 512 base + 512 prefill workspace [est] + 512 × ctx/65536 [est]
fits        ⟺ load_mib ≤ vram_total_mib − headroom_mib(ctx)
```

Cache mode cost per token: **FP16 = 2 B, Q8 = 1 B, Q4 = 0.5 B per element.**

Headroom grows with ctx because prefill allocates a cublas/hgemm workspace **before the
first token exists** — measured on a used 4 GB card: prompts OOM around ~2,500 tokens with
~20 MiB free while the same prompt succeeds from a clean card. Both growing terms are
`[est]` until measured and tunable (`autofit.workspace_mib`, `autofit.ctx_headroom_mib`).

Worked example, measured on an RTX 3050 Laptop (4096 MiB) with `SmolLM3-3B-exl3`
(36 layers × 2 × 4 KV heads × 128 head_dim = **36,864 elems/token**):

| Config | KV @ ctx | headroom(ctx) | Total (1866 weights + 128) | Fits 4096? |
|---|---|---|---|---|
| FP16 @ 65536 | 4608 MiB | 1536 | 6602 MiB | ❌ |
| Q8 @ 65536 | 2304 MiB | 1536 | 4298 MiB | ❌ |
| Q4 @ 65536 | 1152 MiB | 1536 | 3146 MiB | ❌ (4682 > 4096) |
| **Q4 @ 32768** | **576 MiB** | **1280** | **2570 MiB** | ✅ → picks **Q4 @ 32768** |

(Q4 @ 65536 was the old default and measured a real peak of 3105 MiB — it serves, but
prefill of multi-KB prompts on a used card is where it OOMs. That is the capacity trade.)

The ladder tries `ctx = trained max`, halves it down to 4096, and prefers better cache
quality at each tier (FP16 → Q8 → Q4). Guard rails: it never exceeds
`max_position_embeddings`; `--ctx` / `--cache-mode` bypass the ladder entirely; the
decision and its arithmetic are always printed. A thin remaining margin prints a
warning — that is the **safe vs max context** trade: the ladder default is the safe
context, while an explicit `--ctx 65536` buys the max context and prints what it costs.
If prefill OOMs (CUDA OOM *before* the first token — generating fewer tokens will not
help), drop `--ctx` one tier (32768 → 16384). If even the smallest config does not fit,
you get the full breakdown plus the largest ctx that *would* fit — before anything is
downloaded.

Long version: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## The pre-download fit gate

`fit` (and the same gate inside `pull`) reads only small metadata first — `config.json`,
`quantization_config.json`, the file tree; KB, not weights — and, for the format verdict,
the **safetensors headers**: at most 4 files, one ranged `GET` each, `header_size`
validated and capped at 4 MiB before anything is allocated (≤ ~16 MiB worst case per repo),
bodies closed as soon as the header is parsed. The header is the engine's own test: the
per-module tensor-suffix group (`.trellis`, plus `.su`/`.suh` and `.sv`/`.svh`) for EXL3
storage, and the quant-group dtypes checked against the installed engine's supported set.
A header that cannot be read (gated, missing, oversize, transport error) yields
**warn / "unverified"** — never a silent pass, and never a refusal because we could not
look. It then checks three things **before a single weight byte moves**:

1. **Architecture** — is `config.architectures[]` one of the architectures the installed
   exllamav3 declares (reconciled from `architecture/*.py`)? Unknown → warned with the
   exact string, never silently accepted.
2. **Quant format** — `quant_method` must be `exl3`. `exl2` (ExLlamaV2 format, guaranteed
   load failure) → refused. GGUF-only repo → refused with "use ollama with GGUF".
3. **Fit** — the autofit arithmetic above, run against *your* GPU: proceed,
   proceed-with-warning (smaller ctx), or refuse — with numbers either way.

`fit` additionally prints a predicted decode/prefill tok/s `[est]` from metadata, so
`rank` can compare candidates before you commit bandwidth. A prediction is relative to
the chosen context — larger ctx streams more KV bytes per token and lowers decode speed,
so two configs quoted at different contexts are not comparable. The decode constant is
≈83.6 GB/s effective bandwidth on the reference GPU (decodeEfficiency 0.373 × 224 GB/s
spec), calibrated to one measured point (42.7 tok/s, SmolLM3-3B 3.5bpw); every other
number is `[est]` until calibration lands.

The verdict is saved to the model's `manifest.json` and shown by `stone-llama list`.
Cost of a caught mistake: KB of config plus, worst case, ~16 MiB of safetensors headers
(4 files × the 4 MiB cap) — three orders of magnitude below the smallest candidates, whose
weights are gigabytes — instead of a wasted multi-GB download. Concretely: a live 1.68 GB
repo whose quant tensors are a dtype the installed engine cannot load (unsigned int16
`.trellis`) is refused before the consent prompt — `fit` reports that refusal as `exit 3` —
naming the dtype and the engine limitation, where the previous gate accepted it as
`gate: quant ✓ exl3` and would have fetched the whole thing.

Model repos on the Hub churn — quant converters routinely delete or move weights (the
original example here, `turboderp/SmolLM3-3B-exl3`, now serves only a README). That is
why the gate runs before any byte moves, and why `import` can link a model dir you
already own instead of trusting the Hub to stay up.

## Third-party components

The binary itself has **zero third-party Go dependencies** — standard library only
(`go.mod` carries no `require` block, and there is no `go.sum`). Everything in this
section is a *runtime* component: stone-llama **redistributes none** of them; `setup`
fetches each from its upstream source on your machine, sizes announced and confirmation
required. Verified licenses (per-package evidence in [THIRD-PARTY.md](THIRD-PARTY.md)):

| Component | How it arrives | License |
|---|---|---|
| uv | upstream release binary, SHA-256 verified | Apache-2.0 **OR** MIT (dual) |
| CPython 3.12 | `uv python install` (python-build-standalone) | PSF-2.0 |
| PyTorch (+cu130) | pinned wheel from lock file | BSD-3-Clause |
| exllamav3 | pinned wheel from lock file | MIT |
| Triton | pinned wheel from lock file | MIT |
| Flash-linear-attention | pinned wheel from lock file | MIT |
| NVIDIA CUDA/cuDNN (18 wheels) | transitive deps of the `+cu130` torch wheel, pinned in lock file | Proprietary — NVIDIA EULA |
| TabbyAPI (`f07131cd`) | `git clone` + pinned checkout | **AGPL-3.0** |

The NVIDIA CUDA/cuDNN wheels are **not** open source: most ship NVIDIA's proprietary
End User License Agreement, which you accept directly at `setup` time (sizes announced,
one consent prompt). Exceptions: `nvidia-nvtx`, `cuda-bindings` and `cuda-pathfinder` are
Apache-2.0, and `cuda-toolkit` is a meta-package with no licence of its own — full
per-package breakdown in [THIRD-PARTY.md](THIRD-PARTY.md).

**AGPL note:** we neither distribute nor modify TabbyAPI — `setup` downloads upstream
sources to your disk at your request, and stone-llama talks to it as a separate process
over HTTP. Vendoring or shipping a patched TabbyAPI fork would change this analysis; it
is a deliberate non-goal of this project.

## Out of scope for v1

- CPU/AMD/Apple inference, and Windows/macOS/arm64 support claims — see
  [Platform support](#platform-support)
- Multi-GPU `gpu_split` autofit (single GPU; TabbyAPI autosplit passes through)
- Concurrent multi-model serving (one loaded model at a time, like ollama in practice)
- GGUF, EXL2, draft/speculative models
- Forking or vendoring TabbyAPI (AGPL — see above)

## Documentation

| Doc | What's in it |
|---|---|
| [Website](https://rizperdana.github.io/stone-llama/) | project homepage — rendered site with a model compatibility table (fits + predicted tok/s per model for a 4 GB card) |
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | prerequisites, install paths, `setup` provisioning, uninstall |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | first-run walkthrough with real output |
| [docs/AGENT-SETUP.md](docs/AGENT-SETUP.md) | copy-paste prompt for a coding agent |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | full plan, milestones, open gaps |

## License

MIT — see [LICENSE](LICENSE). Third-party runtime components: see
[THIRD-PARTY.md](THIRD-PARTY.md).
