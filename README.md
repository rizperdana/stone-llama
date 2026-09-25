# stone-llama

Ollama-style CLI for **ExLlamaV3 + TabbyAPI**: `pull`, `list`, `rm`, `run`, `serve`, `ps`
for local EXL3 models — as a single small Go binary with a supervised Python inference
sidecar.

**Positioning, honestly:** EXL3 inference speed and VRAM efficiency on NVIDIA GPUs, plus
**VRAM-aware auto-fit context** — stone-llama computes a safe context window and cache mode
from your GPU and the model's own `config.json`, instead of shipping conservative defaults
(it may also warn when your chosen config is too close to the edge for reliable prefill).
It is *not* a better ollama: if you want CPU/AMD/Apple support or a huge model catalog, use
ollama with GGUF.

## NVIDIA/CUDA only — read this first

exllamav3 — the inference engine this project wraps — has **no CPU path, no AMD/ROCm path,
no Apple/Metal path**. That is not configurable:

| Machine | Works? |
|---|---|
| NVIDIA GPU, Linux x86_64, driver ≥ 570 | ✅ (runtime extra `cu12`) |
| NVIDIA GPU, Linux x86_64, driver ≥ 580 | ✅ (runtime extra `cu13`) |
| NVIDIA GPU, driver < 570 | ❌ upgrade the driver |
| AMD / Intel GPU | ❌ |
| CPU-only machine | ❌ |
| Apple Silicon / macOS | ❌ |
| Windows / Linux arm64 | ❌ (not v1; may come later) |

**If you're on CPU/AMD/Apple, use ollama with GGUF instead.** `stone-llama doctor` prints
this verdict — with your actual GPU and driver numbers — before anything gets installed.

## Install

v1 ships as a single static binary (no runtime needed to run `doctor`/`list`):

```bash
git clone https://github.com/rizperdana/stone-llama
cd stone-llama
go build -o stone-llama ./cmd/stone-llama   # Go ≥ 1.24
sudo mv stone-llama ~/.local/bin/           # or anywhere on PATH
```

Release tarballs + `install.sh` (with sha256) land at M7 — see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the milestone status.

## Quickstart

```bash
stone-llama doctor                       # GPU + driver check, picks cu12/cu13 — no downloads
stone-llama setup                        # Python/torch runtime; announces sizes, asks before any byte
stone-llama pull async0x42/Qwen3-1.7B-exl3_4.0bpw  # gate runs first, then resumable download
stone-llama run Qwen3-1.7B-exl3_4.0bpw  # CLI chat (prints the autofit decision)
stone-llama serve                        # OpenAI-compatible API on 127.0.0.1:5111
```

```bash
curl http://127.0.0.1:5111/v1/chat/completions -d '{
  "model": "Qwen3-1.7B-exl3_4.0bpw",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

Model repos on the Hub churn: quant converters routinely delete or move
weights — the original example here (`turboderp/SmolLM3-3B-exl3`) now serves
only a README. That is why `pull` runs the A5 gate (arch / quant / fit, from
metadata) **before** any byte moves, and why `import` can link a model dir you
already own instead of trusting the Hub to stay up.

## Commands

| Command | Description | Status |
|---|---|---|
| `stone-llama doctor` | GPU/driver/runtime verdict, no changes made | ✅ shipped |
| `stone-llama list` | models with size, quant, gate verdict, source | ✅ shipped |
| `stone-llama import <dir> --name <n>` | symlink an existing model dir in (zero copy) | ✅ shipped |
| `stone-llama rm <model>` | remove a model (import symlinks: link only, target untouched) | ✅ shipped |
| `stone-llama pull <model>[:tag]` | pre-download gate → consent → resumable download + sha256 verify | ✅ shipped |
| `stone-llama login` | HuggingFace token for gated repos (stored 0600) | ✅ shipped |
| `stone-llama setup` | provision the pinned Python runtime (consent-gated, resumable) | 🚧 M4 |
| `stone-llama serve` | daemon: OpenAI-compatible proxy over supervised TabbyAPI | 🚧 M5 |
| `stone-llama ps` | loaded model + live VRAM | 🚧 M5 |
| `stone-llama stop` | stop the daemon | 🚧 M5 |
| `stone-llama run <model>` | streaming CLI chat | 🚧 M6 |
| `stone-llama version` | version | ✅ shipped |

Flags that always win over autofit: `--ctx N`, `--cache-mode Q8|Q4|FP16|"2,2"`,
`--no-autofit`. Env: `STONE_LLAMA_HOST`, `STONE_LLAMA_PORT`, `STONE_LLAMA_MODELS_DIR`,
`STONE_LLAMA_CONFIG`, `STONE_LLAMA_NO_AUTOSTART`, `HF_TOKEN`.

## How autofit picks your context

Everything is derived from the model's own `config.json` plus your VRAM:

```
head_dim  = hidden_size / num_attention_heads                    # 2048/16 = 128
elems/tok = num_hidden_layers × 2 × num_key_value_heads × head_dim

load_mib    = weights_mib + ctx × elems/tok × bytes_per_element/2²⁰ + 128   # fixed overhead
headroom_mib(ctx) = 512 base + 512 prefill workspace [est] + 512 × ctx/65536 [est]
fits        ⟺ load_mib ≤ vram_total_mib − headroom_mib(ctx)
```

Cache mode cost per token: **FP16 = 2 B, Q8 = 1 B, Q4 = 0.5 B per element.**

**Why headroom grows with ctx — [est], from live measurements.** The OOM you actually hit on a
4 GB card is in *prefill*: a cold cublas/hgemm workspace is allocated **before the first token is
generated**, so raising `max_tokens` does not avoid it. Measured live (2026-09-26): a *used* card
sustains ~1,300 prompt tokens and OOMs consistently around ~2,500, while the same prompt succeeds
from a clean card — capacity and fragmentation, not generation. So a config reserving more on-card
KV keeps a proportionally larger reserve. Both new terms are estimates until measured and are
tunable in config (`autofit.workspace_mib`, `autofit.ctx_headroom_mib`).

Worked example, measured on an RTX 3050 Laptop (4096 MiB) with `SmolLM3-3B-exl3`
(36 layers × 2 × 4 KV heads × 128 head_dim = **36,864 elems/token**):

| Config | KV @ ctx | headroom(ctx) | Total (1866 weights + 128) | Fits 4096? |
|---|---|---|---|---|
| FP16 @ 65536 | 4608 MiB | 1536 | 6602 MiB | ❌ |
| Q8 @ 65536 | 2304 MiB | 1536 | 4298 MiB | ❌ |
| Q4 @ 65536 | 1152 MiB | 1536 | 3146 MiB | ❌ (4682 > 4096) |
| **Q4 @ 32768** | **576 MiB** | **1280** | **2570 MiB** | ✅ → picks **Q4 @ 32768** |

(Calibration: Q4 @ 65536 was the old default and measured real peak 3105 MiB — it *serves*, but
prefill of multi-KB prompts on a used card is where it OOMs. That is the capacity trade.)

The ladder tries `ctx = trained max`, then halves it down to 4096, preferring better cache
quality at each tier (FP16 → Q8 → Q4). Guard rails:

- **never exceeds `max_position_embeddings`** (untrained extrapolation is not a feature),
- `--ctx` / `--cache-mode` bypass the ladder entirely,
- the decision and its arithmetic are **printed**, always — you can see *why* you got Q4,
- a config with a thin remaining margin prints a **warning**: multi-KB prompts can OOM during
  prefill on a used card — **drop `--ctx` if you see CUDA OOM before the first token**.

**Large `max_seq_len` on a 4 GB card is a capacity trade, not free.** 65536 context at Q4 KV
reserves ~1.15 GiB of a 4 GiB card before a single token exists. The ladder default is the
"safe context"; explicitly passing `--ctx 65536` buys the "max context" and prints exactly what
it costs. If prompts OOM during prefill (CUDA OOM *before* the first token — generating fewer
tokens will not help), lower `--ctx` one tier (32768 → 16384) or restart the model.

If even the smallest config doesn't fit, you get the full breakdown plus the largest ctx
that *would* fit — before anything is downloaded.

## The pre-download gate

`pull` reads only small metadata first (`config.json`, `quantization_config.json`, file
tree — KB, not weights) and checks three things **before a single weight byte moves**:

1. **Architecture** — is `config.architectures[]` one of the 63 architectures the installed
   exllamav3 declares (reconciled from `architecture/*.py`)? Unknown → warned with the
   exact string, never silently accepted.
2. **Quant format** — `quant_method` must be `exl3`. `exl2` (ExLlamaV2 format, guaranteed
   load failure) → refused. GGUF-only repo → refused with "use ollama with GGUF".
3. **Fit** — the autofit math above, run against *your* GPU: proceed, proceed-with-warning
   (smaller ctx), or refuse — with numbers either way.

The verdict is saved to the model's `manifest.json` and shown by `stone-llama list`.
Cost of a caught mistake: ~2 KB of metadata instead of a wasted multi-GB download.

## Storage

```
~/.config/stone-llama/config.json       # config (no secrets)
~/.local/share/stone-llama/
├── models/<model>/                     # TabbyAPI-shaped dirs + manifest.json
├── runtime/                            # uv + Python + venv + pinned TabbyAPI (M4)
├── logs/                               # daemon.log, tabby.log, setup.log
└── hf_token                            # 0600, only after `stone-llama login`
```

## Third-party components

stone-llama **redistributes none** of them; `setup` fetches each from its upstream source
on your machine, with sizes announced and your confirmation required. Verified licenses
(detail + evidence in [THIRD-PARTY.md](THIRD-PARTY.md)):

| Component | License |
|---|---|
| TabbyAPI (pinned commit) | **AGPL-3.0** |
| exllamav3 | MIT |
| PyTorch | BSD-3-Clause |

**AGPL note:** we neither distribute nor modify TabbyAPI — `setup` downloads upstream
sources to your disk at your request, and stone-llama talks to it as a separate process
over HTTP. **Vendoring or shipping a patched TabbyAPI fork would change this analysis —
it is a deliberate non-goal of this project.**

## Out of scope for v1

- Windows, macOS, arm64 (Linux x86_64 only)
- CPU/AMD/Apple inference (impossible — see above)
- Multi-GPU `gpu_split` autofit (single GPU; TabbyAPI autosplit passes through)
- Concurrent multi-model serving (one loaded model at a time, like ollama in practice)
- GGUF, EXL2, draft/speculative models

## Development

Zero-download test suite: autofit math is table-tested on fixture `config.json` files, the
gate and downloader run against an in-process fake HuggingFace server, the proxy/daemon
against a fake TabbyAPI — CI needs no GPU, no Python, no model weights.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full plan, milestones, and the
open-gap list.

## License

MIT — see [LICENSE](LICENSE). Third-party runtime components: see
[THIRD-PARTY.md](THIRD-PARTY.md).
