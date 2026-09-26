<div align="center">
  <img src="assets/stone-llama.png" alt="stone-llama icon" width="160">
</div>

# stone-llama

**Ollama-style CLI + local model server for ExLlamaV3 + TabbyAPI** — one small Go
binary that pulls EXL3 models, auto-fits context to your VRAM, and serves an
OpenAI-compatible API that any client or gateway can point at.

**Positioning, honestly:** EXL3 inference speed and VRAM efficiency on NVIDIA GPUs, plus
**VRAM-aware auto-fit context** — stone-llama computes a safe context window and cache mode
from your GPU and the model's own `config.json`, instead of shipping conservative defaults
(it may also warn when your chosen config is too close to the edge for reliable prefill).
It is *not* a better ollama: if you want CPU/AMD/Apple support or a huge model catalog, use
ollama with GGUF.

## Platform reality — NVIDIA/CUDA only, read this first

exllamav3 — the inference engine this project wraps — has **no CPU path, no AMD/ROCm path,
no Apple/Metal path**. That is not configurable:

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

**Linux/amd64 is the only supported platform in v1.** **If you're on CPU/AMD/Apple, use
ollama with GGUF instead.** `stone-llama doctor` prints this verdict — with your actual
GPU and driver numbers — before anything gets installed.

## Install

```bash
git clone https://github.com/rizperdana/stone-llama
cd stone-llama
go build -o stone-llama ./cmd/stone-llama   # Go ≥ 1.24
install -Dm755 stone-llama ~/.local/bin/stone-llama
```

or, once published, install a release binary (sha256-verified):

```bash
curl -fsSL https://raw.githubusercontent.com/rizperdana/stone-llama/main/scripts/install.sh | sh
```

Full prerequisites, the consent-gated `setup` provisioning step (≈ 1 GB announced
before you confirm), verification and uninstall:
[docs/INSTALLATION.md](docs/INSTALLATION.md).

## Quickstart

```bash
stone-llama doctor                       # GPU + driver check, picks cu12/cu13 — no downloads
stone-llama setup                        # Python/torch runtime; announces sizes, asks before any byte
stone-llama fit async0x42/Qwen3-1.7B-exl3_4.0bpw   # gate + tok/s estimate from HF metadata — no download
stone-llama pull async0x42/Qwen3-1.7B-exl3_4.0bpw  # gate runs first, then resumable download
stone-llama run Qwen3-1.7B-exl3_4.0bpw   # CLI chat (prints the autofit decision)
stone-llama serve                        # OpenAI-compatible API on 127.0.0.1:5111
```

```bash
curl http://127.0.0.1:5111/v1/chat/completions -d '{
  "model": "Qwen3-1.7B-exl3_4.0bpw",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

Step-by-step with expected output: [docs/QUICKSTART.md](docs/QUICKSTART.md). Handing this
to a coding agent? Copy [docs/AGENT-SETUP.md](docs/AGENT-SETUP.md).

Model repos on the Hub churn: quant converters routinely delete or move
weights — the original example here (`turboderp/SmolLM3-3B-exl3`) now serves
only a README. That is why `pull` runs the pre-download gate (arch / quant / fit, from
metadata) **before** any byte moves, and why `import` can link a model dir you
already own instead of trusting the Hub to stay up.

## Commands

| Command | Description | Status |
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
| `stone-llama serve [--attach host:port]` | daemon: OpenAI-compatible API (attach = existing TabbyAPI upstream) | 🚧 M5 |
| `stone-llama ps` | loaded model + live VRAM | 🚧 M5 |
| `stone-llama stop` | stop the daemon | 🚧 M5 |
| `stone-llama run <model>` | streaming CLI chat | 🚧 M6 |
| `stone-llama version` | version | ✅ shipped |

Flags that always win over autofit: `--ctx N`, `--cache-mode Q8|Q4|FP16|"2,2"`,
`--no-autofit`. Env: `STONE_LLAMA_HOST`, `STONE_LLAMA_PORT`, `STONE_LLAMA_MODELS_DIR`,
`STONE_LLAMA_CONFIG`, `STONE_LLAMA_NO_AUTOSTART`, `HF_TOKEN`.

## Real terminal output

Captured verbatim from a live run on 2026-09-26 (RTX 3050 Laptop, 4096 MiB, driver
580.178.04, Linux/amd64). Raw captures: [docs/screenshots/](docs/screenshots/).

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

```console
$ printf 'n\n' | stone-llama pull async0x42/Qwen3-1.7B-exl3_4.0bpw   # download declined
gate: arch  ✓ Qwen3ForCausalLM
gate: quant ✓ exl3
gate: fit   ⚠ weights 1491 + KV 1120 + overhead 128 = 2739 MiB (headroom 1344, budget 2752)
      warning: only 13 MiB margin above the 1344 MiB headroom (prefill workspace [est] included): multi-KB prompts can OOM during prefill on a used card — if you see CUDA OOM before the first token, drop --ctx
warnings above — review them before continuing
pulling async0x42/Qwen3-1.7B-exl3_4.0bpw → Qwen3-1.7B-exl3_4.0bpw: 11 files, 1.47 GiB (sha256-verified), 83.06 GiB free
stone-llama pull: non-interactive pull requires --yes to confirm the download
```

```console
$ stone-llama fit async0x42/Qwen3-1.7B-exl3_4.0bpw      # metadata only — no download
gate: arch  ✓ Qwen3ForCausalLM
gate: quant ✓ exl3
gate: fit   ⚠ weights 1491 + KV 1120 + overhead 128 = 2739 MiB (headroom 1344, budget 2752)
      warning: only 13 MiB margin above the 1344 MiB headroom (prefill workspace [est] included): multi-KB prompts can OOM during prefill on a used card — if you see CUDA OOM before the first token, drop --ctx
estimate: ~31 tok/s decode, ~582 tok/s prefill [est] at Q4 ctx 40960 (NVIDIA GeForce RTX 3050 Laptop GPU) — low confidence, anchored to the measured 42.7 tok/s SmolLM3-3B 3.5bpw point (1866 MiB) on this GPU; calibration pending

$ stone-llama fit async0x42/Qwen3-8B-exl3_4.0bpw        # too big for 4 GB
gate: arch  ✓ Qwen3ForCausalLM
gate: quant ✓ exl3
gate: fit   ✗ no config fits: weights 4950 + KV (Q4 @ 4096) 144 + overhead 128 = 5222 MiB > 3040 MiB budget (4096 MiB VRAM − 1056 headroom: 512 base + 512 prefill workspace [est] + 32 ctx margin [est])
      largest ctx that would fit: none — no context fits; pull a smaller quant or use a bigger GPU
      reduce --ctx or pick a smaller quant
fit: refused — no context fits this model in VRAM (see the gate report above)
# fit exits 3 when refused

$ stone-llama rank --file refs.txt                          # two HF refs, metadata only
REPO                              QUANT  VERDICT        EST T/S  EST PRE T/S  RATING
async0x42/Qwen3-1.7B-exl3_4.0bpw  4bpw   warn:40960 Q4  31       582          -
async0x42/Qwen3-8B-exl3_4.0bpw    4bpw   refuse         -        -            -
~ values are estimates [est] from metadata + GPU spec — not measured (calibration pending)
```

The gate runs **before** any weight byte moves — the `pull` run above stopped at
the consent prompt, so nothing was fetched beyond KB of metadata. `list` shows a
model linked in with `import` from a directory already on this machine — also no
download. A rendering of these captures: [docs/screenshots/cli.svg](docs/screenshots/cli.svg).
Model-download steps are deliberately omitted from these captures.

`serve` + `curl` is not captured yet: at the time of writing `serve` ships with
milestone M5 and the build reports `not implemented yet (ships in M5)` (raw
capture: [docs/screenshots/serve.txt](docs/screenshots/serve.txt)) — no server was
started and no upstream was disturbed. The `serve`/`curl` transcript lands here
when M5 does.

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
sustains ~1,300 prompt tokens and OOMs consistently around ~2,500 with ~20 MiB free, while the
same prompt succeeds from a clean card — capacity and fragmentation, not generation. So a config
reserving more on-card KV keeps a proportionally larger reserve. Both new terms are estimates
until measured and are tunable in config (`autofit.workspace_mib`, `autofit.ctx_headroom_mib`).

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

**Safe context vs max context — the live caveat.** On a 4 GB card, a maxed-out
`max_seq_len` can leave so little VRAM that **prefill** OOMs on agent-sized prompts
(measured: failure at ~2.5k tokens with ~20 MiB free). 65536 context at Q4 KV reserves
~1.15 GiB of a 4 GiB card before a single token exists. The ladder default is the
**safe context**; explicitly passing `--ctx 65536` buys the **max context** and prints
exactly what it costs. If prompts OOM during prefill (CUDA OOM *before* the first token —
generating fewer tokens will not help), lower `--ctx` one tier (32768 → 16384) or restart
the model.

If even the smallest config doesn't fit, you get the full breakdown plus the largest ctx
that *would* fit — before anything is downloaded.

## The pre-download fit gate

`fit` (and the same gate inside `pull`) reads only small metadata first (`config.json`,
`quantization_config.json`, file tree — KB, not weights) and checks three things
**before a single weight byte moves**:

1. **Architecture** — is `config.architectures[]` one of the architectures the installed
   exllamav3 declares (reconciled from `architecture/*.py`)? Unknown → warned with the
   exact string, never silently accepted.
2. **Quant format** — `quant_method` must be `exl3`. `exl2` (ExLlamaV2 format, guaranteed
   load failure) → refused. GGUF-only repo → refused with "use ollama with GGUF".
3. **Fit** — the autofit math above, run against *your* GPU: proceed, proceed-with-warning
   (smaller ctx), or refuse — with numbers either way.

`fit` additionally prints a predicted decode/prefill tok/s **[est]** from HF metadata —
no weights on disk needed. That's why you can rank candidates (`rank`) and decide before
committing bandwidth.

A predicted tok/s is **relative to the chosen context** — larger ctx streams more
KV bytes per token, which lowers decode speed; two configs or models quoted at
different contexts are not directly comparable.

The estimator's decode constant is ≈83.6 GB/s effective bandwidth on this GPU —
decodeEfficiency (0.373) × the RTX 3050 Laptop's 224 GB/s spec — calibrated to the
measured 42.7 tok/s SmolLM3-3B 3.5bpw anchor; that figure was derived two independent
ways, so it's the defensible baseline until calibration data lands.

The verdict is saved to the model's `manifest.json` and shown by `stone-llama list`.
Cost of a caught mistake: ~2 KB of metadata instead of a wasted multi-GB download —
which is exactly why a 2 GB pull never starts for a model that doesn't fit.

## Storage

```
~/.config/stone-llama/config.json       # config (no secrets)
~/.local/share/stone-llama/
├── models/<model>/                     # TabbyAPI-shaped dirs + manifest.json
├── runtime/                            # uv + Python + venv + pinned TabbyAPI
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

- Windows, macOS, arm64 support claims (Linux x86_64 is the only supported platform;
  a Windows binary may be published but is untested at runtime; macOS can never serve)
- CPU/AMD/Apple inference (impossible — see above)
- Multi-GPU `gpu_split` autofit (single GPU; TabbyAPI autosplit passes through)
- Concurrent multi-model serving (one loaded model at a time, like ollama in practice)
- GGUF, EXL2, draft/speculative models
- Forking or vendoring TabbyAPI (AGPL — see above)

## Documentation

| Doc | What's in it |
|---|---|
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | prerequisites, install paths, `setup` provisioning, uninstall |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | first-run walkthrough with real output |
| [docs/AGENT-SETUP.md](docs/AGENT-SETUP.md) | copy-paste prompt for a coding agent |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | full plan, milestones, open gaps |

## Development

Zero-download test suite: autofit math is table-tested on fixture `config.json` files, the
gate and downloader run against an in-process fake HuggingFace server, the proxy/daemon
against a fake TabbyAPI — CI needs no GPU, no Python, no model weights.

## License

MIT — see [LICENSE](LICENSE). Third-party runtime components: see
[THIRD-PARTY.md](THIRD-PARTY.md).
