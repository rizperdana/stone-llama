# stone-llama — Architecture & Delivery Plan (v2)

> **Status:** v1 approved with amendments; **v2 is executable**. Amendments A5–A8 folded in. This document is authoritative; `PLAN.md` was renamed here per director instruction (A8).
> **Author:** architect · **Date:** 2026-09-25 · **Revision:** v2 (approved)

**Goal:** an Ollama-like experience (`pull` / `list` / `rm` / `ps` / `serve` / `run`) for ExLlamaV3 + TabbyAPI, shipped as one small binary that any NVIDIA-CUDA user can run on a low-spec machine, with VRAM-aware automatic context/cache configuration and a **pre-download compatibility/fit gate** as the differentiators.

---

## 0. Facts this plan is built on (verified on this machine)

| Fact | Value |
|---|---|
| TabbyAPI checkout | `/path/to/tabbyAPI`, commit `f07131c`, **license AGPL-3.0** (verified: `LICENSE`) |
| TabbyAPI OpenAI routes | `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/apply-template` (Bearer-key gated, SSE) |
| TabbyAPI config override | `start.py --config <path>` (`args.config`) → we never touch a user's `config.yml` |
| TabbyAPI extras | `cu12` = torch 2.9.0+cu128 + exllamav3 1.5.1+cu128; `cu13` = torch 2.11.0+cu130 + exllamav3 1.5.1+cu132. **Both exist.** |
| exllamav3 wheel source | direct GitHub-release URLs in `pyproject.toml` → lockfile pins exact URLs + hashes |
| exllamav3 license | **MIT** (verified: dist-info `METADATA` `License-Expression: MIT`) |
| torch license | **BSD-3-Clause** (verified: dist-info `METADATA`) |
| Architecture support matrix | the embedded arch list (`internal/preflight/archlist.go`) — built from installed `architecture/*.py` + upstream README + TabbyAPI templates (A5 source) |
| GPU | RTX 3050 Laptop, **4096 MiB**, driver 580.178.04 |
| Measured model | `SmolLM3-3B-exl3`: `model.safetensors` = 1,957,008,720 B (**1866 MiB**) |
| Measured load | historical live run at Q4 @ 65536 peaked at **3105 MiB** (the *old* default, pre-A10); autofit's *reservation* for that config is 1866 weights + 1296 KV + 128 overhead = **3290 MiB**, which the headroom rule refuses (budget 2560 @ 65536) → the ladder lands Q4@32768 |
| KV arithmetic | `layers × 2 × kv_heads × head_dim` = 36 × 2 × 4 × 128 = **36,864 elems/token** → per token FP16 72 KiB, Q8 38.25 KiB, Q4 20.25 KiB, `"4,2"` 15.75 KiB (0.0625 B/elem of scales included); @65536: 4608 / 2448 / 1296 / 1008 MiB, @32768: 2304 / 1224 / 648 / 504 MiB. **Verified against `config.json` and live `fit`.** |
| Go | go1.24.4 linux/amd64 installed |
| `gh` | **active account `rizperdana`** (verified this session; two other logged-in accounts exist but are inactive — this check gated repo creation per A8) |
| Bootstrap pain (known) | missing `aiofiles`, no `pip` in uv venv, `--gpu-lib` = install-time extra selection, API key auto-generated to `api_tokens.yml` |

**Correction to the brief (kept from v1):** the "2909 MiB used" figure must not calibrate the heuristic. Autofit models **peak**: weights + full-cache KV + 128 MiB measured overhead. The historical observed peak at Q4@65536 was 3105 MiB, but full Q4 KV there is 1296 MiB (§5 — scales priced in), so the model's reservation for that config is 3290 MiB; an observed peak below it only means that run never filled the cache [est]. The earlier decomposition here (1152 KV + ≈87 ≈ 3105) priced Q4 KV at 0.5 B/elem without scales and is superseded by §5.

---

## Decision log (director answers to v1 §13)

| Q | Decision |
|---|---|
| Q1 Go + supervise-TabbyAPI + proxy | **APPROVED** |
| Q2 port 5111 public / ephemeral internal | **APPROVED** |
| Q3 ctx-maximizing ladder | **APPROVED with guard rails** → folded into §5: (a) never exceed `max_position_embeddings`; (b) `--ctx`/`--cache-mode` always win over autofit; (c) chosen config + VRAM projection always printed |
| Q4 512/128 constants, M5 re-measure | **APPROVED** |
| Q5 dual cu12/cu13 with driver preflight | **APPROVED** |
| Q6 real provisioning run | **REJECTED** — zero-download rule absolute. M4 proven against stubs + throwaway `runtime_dir`; multi-GB provisioning is a **blocked step** requiring separate user authorization, designed as one explicit, resumable, size-announced action (§6) |
| Q7 `import` + `stop` in v1 | **APPROVED** |
| Q8 Linux/amd64 only | **APPROVED** |
| Q9 single-model serving non-goal | **APPROVED** |
| Q10 honest positioning incl. "use ollama if …" | **APPROVED** |
| Q11 track pinned upstream, never fork | **APPROVED** |
| Q12 auto-start daemon + `STONE_LLAMA_NO_AUTOSTART=1` | **APPROVED** |

New amendments A5–A8, A10: §4 (A5 gate), §6 (A6 licensing + A7 setup preflight), §9 (A8 repo/delivery), §5 (A10 prefill workspace headroom), §12 (G13–G16 with residual risk). **No further review round is required before M0/M1** (director instruction); remaining open points are listed in §13 with defaults. (A9 = attach seam, lands with M5.)

---

## 1. Language: **Go** (approved Q1)

**Go 1.24, single static binary, Linux/amd64 v1.**

| Criterion | Go | Rust | Python control plane |
|---|---|---|---|
| Install | one static binary, measured 7,770,296 B ≈ 7.41 MiB (`-s -w -trimpath` — release builds are already stripped; `-buildvcs=false` measured 0 bytes saved) | same | needs the multi-GB venv before the UI works |
| Cold start | `version`/`list` ~4–6 ms, `doctor` ~24 ms median because it spawns `nvidia-smi` and probes the GPU (hardware-probe latency, not binary init); measured warm-cache on a quiet host — a loaded host taxes every process equally | < 10 ms [est] (no Rust build exists in this project — unverified) | 0.5–2 s (torch import alone is seconds) |
| Idle daemon RSS | ~15–30 MB [est] | ~5–15 MB [est] | 100–300 MB [est] |
| Stdlib fit | `net/http`, `httputil.ReverseProxy`, `encoding/json`, `os/exec` — all of it | no HTTP in stdlib | native, but competes with inference venv for deps |

The "low potato" constraint targets **memory and startup**; inference speed is Python/GPU regardless. A Python control plane adds ~150 MB RSS [est] and a startup tax to every command — on an 8 GB laptop that steals from the KV cache. Rust saves ~10 MB over Go [est] and costs weeks plus a hand-rolled HTTP stack. **Tradeoff accepted:** Go's GC (irrelevant here); `nvidia-smi` shell-out instead of NVML cgo.

**Dependencies:** zero third-party Go modules — standard library only (`go.mod` is just the
`module` line plus `go 1.24`, with no `require` block, and there is no `go.sum`). HTTP, SSE,
JSON, progress rendering: stdlib; the TabbyAPI config is emitted by hand —
`internal/serve/tabbyconf.go` records "deliberately no YAML dependency for a handful of
scalars", so no YAML library is involved either.

## 2. Architecture (approved Q1/Q2/Q11/Q12)

```
 user ── CLI (stone-llama) ──HTTP──▶ stone-llama daemon (Go)
                                        │  • storage, manifests, pull (+ A5 gate)
                                        │  • VRAM autofit → generates tabby-config.yml
                                        │  • reverse proxy, auth, health, crash-backoff
                                        ▼  child process (supervised)
                                     TabbyAPI (pinned commit f07131c, uv venv)
                                        ▼
                                     exllamav3 → CUDA GPU
```

**Decision: (a) supervise TabbyAPI as a child and proxy — not (b) a from-scratch exllamav3 server.** TabbyAPI owns every hard server problem (SSE, chat-template application, sampling, quant loading, gpu_split/autosplit, MoE offload keys, token auth); option (b) re-implements all of it against an exllamav3 API at 1.5.1 and moving. Upstream churn lands on TabbyAPI; we pin a commit. Cost of (a): one supervisor + ~150 lines of reverse proxy — which buys stable public port, independent downstream auth, SSE passthrough, and a runtime-swap seam.

**Ports:** public **5111** (avoids well-known local-AI ports: 11434/Ollama, 5000–5002/TabbyAPI). Occupied → health-probe: stone-llama answers → reuse (idempotent auto-start); foreign process → fatal with `--port` hint. TabbyAPI's internal port: **always bind `:0`**, write into generated config — never conflicts. Proxy hides it.

**Upstream changes:** pin `f07131c` in embedded `runtime.lock.json`; §10.5's contract test against the pin is **planned, not in the tree** (no `go:build integration` file), so a pin bump is reviewed by hand and rides a release. Never fork (Q11).

**Lifecycle:** `serve` = foreground daemon (it *is* the daemon). `run` auto-starts a detached daemon if none is reachable (re-exec self, `setsid`, log → `logs/daemon.log`, state → `daemon.json` 0600), unless `STONE_LLAMA_NO_AUTOSTART=1` (then it reports + remedies instead). `ps` observes only — read-only, never spawns or deletes state; `stop` controls only — clean shutdown. Singleton guard: **flock** on `data_dir/stone-llama.lock` around daemon spawn and downloads (A7; primitive lands with its first consumer in M2, daemon-spawn usage in M5).

## 3. Storage layout (approved; A6/A7 hygiene folded in)

**Decision: plain per-model directories + provenance manifest. No content-addressed blobs.** EXL3 weights are unique per quant (no cross-model dedupe available worth having), and TabbyAPI requires an HF-shaped directory — blobs would force a materialization layer that can half-fail. `list`/`rm` stay directory scans.

```
~/.config/stone-llama/config.json          # user config (JSON, stdlib; no secrets → 0644)
~/.local/share/stone-llama/                # data dir → mode 0700 on creation
├── models/
│   └── SmolLM3-3B-exl3/                   # one dir per model = TabbyAPI model dir
│       ├── config.json, tokenizer*, *.safetensors, …   (verbatim from HF)
│       └── manifest.json                  # provenance + integrity + A5 verdict (0644)
├── runtime/                               # uv binary, Python, venv, TabbyAPI checkout (§6)
├── downloads/<model>/*.part               # resumable pull staging
├── logs/                                  # daemon.log, tabby.log, setup.log
├── stone-llama.lock                       # A7 flock singleton (created 0600)
├── hf_token                               # 0600, only after `login`
└── daemon.json                            # pid, port, token → 0600 (A7)
```

- **Weights** in one configurable `models_dir` (default above; env/config override). `list` scans subdirs; `rm` deletes (symlink → unlink link only, never descend; directory → `RemoveAll`).
- **`manifest.json`** written by `pull`: `repo_id`, `revision`, `quant`, `pulled_at`, per-file `{path, size, sha256}` (HF LFS sha), and **`verdict`** (A5 pre-download gate result — shown by `list`). Dirs without manifest: symlink → `imported`, plain dir → `unmanaged`; both listed.
- **Integrity:** SHA-256 verified as each file completes, before `.part` → final rename (atomic; a killed pull never leaves a half-file posing as a model). Re-pull re-verifies and repairs.
- **Disk preflight** before first byte: sum HF tree sizes vs `statfs` free − 5 %.
- **Secret hygiene (A7):** data dir 0700; `hf_token`, `daemon.json`, lockfile 0600; tokens never logged, never placed in argv (read from file, injected via proxy header); config file carries no secrets (0644).

## 4. Download manager + **A5 pre-download gate**

### 4.1 A5 — compatibility + fit gate (runs BEFORE any weight byte; the budget guard)

`pull` phase 0 fetches **metadata only**: repo tree, `config.json`, `quantization_config.json` (KBs), plus the safetensors headers behind the format check — at most 4 files, one ranged `GET` each, `header_size` validated and capped at 4 MiB before anything is allocated (≤ ~16 MiB worst case per repo), bodies closed as soon as the header is parsed; a header that cannot be read (gated, missing, oversize, transport error) is **warn / "unverified"**, never a silent pass and never a refusal for failing to look. Then three checks, in order:

1. **Architecture support.** `config.architectures[]` checked against a list **embedded in the binary**, sourced from `internal/preflight/archlist.go` (verified against installed `architecture/*.py` + upstream README). Verified strings include: `Qwen2ForCausalLM`, `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`, `Qwen3VLForConditionalGeneration`, `Qwen3VLMoeForConditionalGeneration`, `Qwen3NextForCausalLM`, `Qwen3_5ForCausalLM`, `Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForConditionalGeneration`, `Qwen4ExpForCausalLM`, `LlamaForCausalLM`, `Gemma2ForCausalLM`, `Gemma3ForCausalLM`, `Gemma3ForConditionalGeneration`, `Gemma4ForConditionalGeneration` (E2B/E4B variants unsupported), `Phi3ForCausalLM`, `MistralForCausalLM`, `Mistral3ForConditionalGeneration`, `MixtralForCausalLM`, `DeepseekV3ForCausalLM`, `Glm4ForCausalLM`, `Glm4MoeForCausalLM`, `Glm4MoeLiteForCausalLM`, `GlmMoeDsaForCausalLM`, `GptOssForCausalLM`, `CohereForCausalLM`, `Cohere2ForCausalLM`, `Olmo3ForCausalLM`, `OlmoHybridForCausalLM`, `SmolLM3ForCausalLM`, `Lfm2ForCausalLM`, `Lfm2MoeForCausalLM`, `HYV3ForCausalLM`, `Step3p5ForCausalLM`, `SeedOssForCausalLM`, `SolarOpenForCausalLM`, `IQuestCoderForCausalLM`, `KimiLinearForCausalLM`, `HyperCLOVAXForCausalLM`, `LagunaForCausalLM`, `MiniMaxM2ForCausalLM`, `MuseGlimmerForCausalLM`, `ArceeForCausalLM`, `ApertusForCausalLM`, `Exaone4ForCausalLM`, `Dots1ForCausalLM`, `Ernie4_5_MoeForCausalLM`, `ArceeForCausalLM`. **Unknown arch → warn with the exact string + confirm on TTY (`--yes` to proceed), never silent.** The list is a snapshot that drifts with exllamav3 releases; warn-not-refuse means an incomplete list degrades to warnings, not false refusals (G13).
2. **Quant format.** `quantization_config.quant_method` must be **`exl3`**, and the safetensors header must agree: the per-module tensor-suffix group (`.trellis`, plus `.su`/`.suh` and `.sv`/`.svh`) for EXL3 storage, with the quant-group tensors' dtypes validated against the installed engine's supported set — a dtype the engine cannot load → **refuse**, naming the dtype and the limitation. `exl2` → **refuse**: "EXL2 quant (ExLlamaV2 format) — exllamav3 cannot load it." GGUF-only repo (`.gguf` files, no exl3) → **refuse**: "no EXL3 weights — this repo ships GGUF; the default backend (TabbyAPI + ExLlamaV3) loads EXL3 only, so use ollama with GGUF for now; a llama.cpp 'llama' backend exists for GGUF but is experimental — it has never served a token in this release." Missing/ambiguous metadata → warn + confirm; a header that cannot be read → warn / "unverified" (never a refusal for failing to look). Converts a 2 GB mistake into a KB-plus-headers check: KB of config, ≤ ~16 MiB of headers worst case per repo.
3. **Fit projection.** Run §5 autofit against the fetching machine's GPU (VRAM from `nvidia-smi`): print `weights + KV(ctx) + overhead + headroom vs available`, then **proceed** / **proceed-with-warning** / **refuse** (with the numbers and *the largest ctx that would fit*). `min_vram` rule (§8) wired into `pull`, not just `doctor`.

Verdict recorded in `manifest.json`; `list` shows it. **Residual risk:** G13.

### 4.2 Resolution, download, resume

**HF REST source** (`/api/models/<repo>` → siblings/sizes; `/tree/<rev>` → layout; `resolve/<rev>/<file>` → bytes).

1. `pull <name>`: exact repo id first; 404 → search (`?search=`) + client-side filter (`exl3` in name or `quantization_config.json` in tree). Zero → honest error (§8 wording). Multiple → interactive picker with sizes; non-TTY/`--yes` → error listing candidates (never guess). Multiple quant subdirs → require `:tag`.
2. **Consent** (A7 preflight before the prompt): disk space (`statfs`), driver→extra, destination paths, and **exact download size via HTTP HEAD on each pinned file URL** (Content-Length, no body) → one confirmation. No download without explicit consent.
3. **Resumable, single-stream.** `Range` against `resolve`; staged `downloads/<model>/<name>.part` + byte counter; retry ×5 with backoff. **Concurrent chunking: NO** — code tripled for rare consumer-link gains; resume proof stays simple. Tradeoff noted: gigabit users slower (G10).
4. **Integrity:** SHA-256 vs HF LFS oid on completion, before rename; mismatch → delete `.part`, retry from zero once, then fail naming the file.
5. **Gated repos:** `login` → `hf_token` 0600; `HF_TOKEN` env wins. 401/403 → "repo is gated. Run 'stone-llama login' or set HF_TOKEN."
6. **Interrupted pulls:** `.part` survives → next `pull` resumes; `pull --force` restarts; `rm` cleans staging.

**Progress:** single-line TTY writer (stdlib, ~60 lines), `--quiet` for machines, newline-per-10 % non-TTY.

## 5. VRAM autofit (Q3 guard rails folded in)

```
head_dim  = hidden_size / num_attention_heads                    # 2048/16 = 128; else head_dim key
elems/tok = num_hidden_layers × 2 × num_key_value_heads × head_dim  # 36×2×4×128 = 36,864
kv_bytes(ctx, mode) = ctx × elems/tok × {FP16: 2.0, Q8: 1.0625, Q4: 0.5625, "4,2": 0.4375}   # + 0.0625 B/elem of scales
weights_mib = Σ safetensors sizes / 2²⁰                          # 1866 for SmolLM3-3B-3.5bpw
load_mib(mode, ctx) = weights_mib + kv/2²⁰ + 128                 # 128 = measured fixed overhead
headroom_mib(ctx) = 512 base + 512 prefill workspace [est] + 512 × ctx/65536 [est]   # A10
fit ⟺ load_mib ≤ vram_total_mib − headroom_mib(ctx)
```

Calibration (live `fit`, 2026-09-29): `1866 + 648 + 128 = 2642 ≤ 4096 − 1280 = 2816` ✓ → **Q4@32768**
(174 MiB slack → thin-margin warning); at 32768 FP16 → 4298 ✗, Q8 → 3218 ✗; 65536/Q4 refused
(`1866 + 1296 + 128 = 3290 > 4096 − 1536 = 2560`), 65536/`"4,2"` too (3002 > 2560).
> **Rule labelling:** this gate uses the **headroom rule** above — `load ≤ vram_total − headroom(ctx)`,
> headroom = 512 base + 512 prefill workspace [est] + 512×ctx/65536 [est] (1536 @ 65536, 1280 @ 32768).
> The website's model table uses a different, static **budget rule** — `weights + KV ≤ 3508 MiB`
> (= 4096 − 88 overhead − 500 allowance, the research note's config rule), where SmolLM3 Q4 @65536
> (1866 + 1296 = 3162) fits. Both figures are true of their own rule; each is labelled where shown.

**A10 — prefill workspace (live evidence, 2026-09-26).** Every failure at ≥~1.8k prompt tokens on
the live 4 GB stack was **CUDA OOM during prefill-workspace allocation** (`reconstruct_hgemm`,
`cublasCreate`, `device_copy`) — the workspace is allocated *before* the first token exists, so
`max_tokens=1` does not help. Sustained to ~1,282 prompt tokens, intermittent ~1,780, consistent
OOM ≥~2,500 at ~20 MiB mean free VRAM; a 60,076-token prompt had succeeded earlier **from a clean
card** → capacity + fragmentation that degrades after traffic, not a generation limit. Hence the
ctx-scaled headroom: a fuller card fragments worse, and a live session grows its KV toward ctx.
Both new terms are **[est]** until measured and tunable (`autofit.workspace_mib` = 512,
`autofit.ctx_headroom_mib` = 512 @ 65536). Consequences: the SmolLM3 default moves 65536/Q4 →
32768/Q4; configs with <512 MiB margin above the headroom print a warning (drop `--ctx` on
prefill OOM); README frames large `max_seq_len` on 4 GB as a capacity trade — "safe context" =
the ladder's default, "max context" = explicit `--ctx`, printed with the warning. Q3 ladder order
and guard rails unchanged. Measured **in attach mode against the live TabbyAPI** (zero downloads,
stack untouched) — which validates the A9 attach seam as the intended live-test seam.

**Ladder (approved Q3 + guard rails):**

```
target = min(model.max_position_embeddings, user --ctx)     # (a) HARD CLAMP: never exceed trained ctx,
                                                            #     even when the user asks for more (warn)
for ctx in [target, target/2, target/4, … down to 4096]:
    for mode in [FP16, Q8, Q4, "4,2"]:                     # quality preferred within a ctx tier
        if fit(ctx, mode): pick
# (b) explicit --ctx / --cache-mode BYPASS the ladder entirely and are used as-is
#     (--cache-mode accepts any TabbyAPI value incl. Q6 and pair formats like "2,2";
#      the ladder runs FP16/Q8/Q4/"4,2" only — Q6 is skipped: Q4 is ~lossless for KV
#      and Q6 @65536 measured OOM on this 4 GB card; Q2/"2,2" is override-only)
# (c) the pick + projection is ALWAYS printed, never silent:
```

```
autofit: cache Q4, max_seq_len 32768
  weights 1866 + KV 648 + overhead 128 = 2642 MiB (headroom 1280, budget 2816)
warning: only 174 MiB margin above the 1280 MiB headroom (prefill workspace [est] included): multi-KB prompts can OOM during prefill on a used card — if you see CUDA OOM before the first token, drop --ctx
  (the ladder's rejected alternatives, from the arithmetic above: 32768 FP16 4298, Q8 3218; 65536/Q4 3290 > 2560)
```

Failure path prints the full breakdown (weights / per-token / ctx / required vs free) + the largest ctx that *would* fit. VRAM from `nvidia-smi` shell-out. `gpu_split`: not computed (single-GPU v1; `gpu_split_auto` passthrough).

Emitted `runtime/tabby-config.yml` sets `model.max_seq_len`, `model.cache_size` (same value, multiple of 256), `model.cache_mode`, `model.model_dir`, `network.host/port` (internal), `autosplit_reserve: [96]`.

## 6. Runtime bootstrap (Q6 rejection + A6 + A7 folded in)

**Decision: on-demand provisioning into `runtime/` via pinned `uv`, explicit consent first. Not bundled, not system Python.** (Bundling kills the single-binary story; system Python kills reproducibility.)

**Mechanism:** `setup` → (1) pinned `uv` binary (URL+SHA-256) → `runtime/bin/`; (2) `uv python install 3.12` (pinned CPython); (3) pinned TabbyAPI tarball (`f07131c` + SHA-256) → `runtime/tabbyAPI/`; (4) `uv venv` + `uv pip install -r runtime/requirements.lock` — lock generated by us at release time with hashes; extra (`cu12`/`cu13`) chosen by `doctor`; exllamav3 wheels are already exact URL pins; (5) smoke `python -c "import exllamav3"`.

**Q6 rejection — zero-download rule.** M4 is proven **without** the multi-GB run:
- **Unit/stub proof:** fake `uv`, fake `python`, fake `nvidia-smi` on PATH; consent-refusal paths (every branch that would fetch bytes is asserted to stop at the gate); lock parsing, step journal, error mapping, log surfacing — all against `httptest`-served fixture files.
- **Throwaway proof (still zero-weight):** a `runtime_dir` in a temp path exercised against stub executables end-to-end.
- **The real provisioning run is a BLOCKED STEP**, requiring separate director/user authorization with measured sizes. Design makes it exactly one action: one explicit confirmation with HEAD-measured sizes (A7), **journal-based step records** so an interrupted run resumes where it stopped (uv itself is idempotent; the journal skips completed steps and re-announces remaining sizes), full log at `logs/setup.log`.

**A7 — setup preflight (before consent):** exact download sizes via HEAD on pinned URLs; `statfs` disk check; driver→extra decision; destination paths listed; then **one** confirmation. Failure output: last 20 lines of `setup.log` + path, mapped to human causes (no `nvidia-smi` → §8; ENOSPC → numbers; TLS → mirror hint).

**A6 — licensing (verified, not assumed):**

- Our license: **MIT** (chosen: zero-distribution glue tool; maximum compatibility; Apache-2.0's patent grant buys nothing at this surface, MIT avoids NOTICE bookkeeping).
- `THIRD-PARTY.md`: components `setup` downloads at runtime — **all licences verified from real sources (A6 gate CLEARED, commit `1aaa785`)**: `uv` Apache-2.0 OR MIT, CPython PSF-2.0, TabbyAPI **AGPL-3.0** (`LICENSE`), PyTorch **BSD-3-Clause** (METADATA), exllamav3 **MIT** (METADATA), Triton MIT, FLA MIT, and the NVIDIA CUDA/cuDNN set as proprietary EULA (with the `nvidia-nvtx` Apache-2.0 exception). The former UNVERIFIED section is emptied — its "Still open" list reads "Nothing".
- README line: *we redistribute none of these; stone-llama's `setup` fetches them from their upstream sources on your machine, on your confirmation.*
- **AGPL reasoning (recorded):** we neither distribute nor modify TabbyAPI; `setup` fetches upstream sources to the user's disk at the user's request and talks to it via subprocess + HTTP. Vendoring or patching TabbyAPI would change this analysis — that is why forking is off the table in v1 (Q11).

**`doctor`:** GPU present?, driver → extra (≥580 cu13, ≥570 cu12, else refuse), runtime installed?, models dir writable?, ports free?, disk space. Every command runs the relevant precondition subset.

**Escape hatch:** `runtime_dir` config points at an existing checkout/venv (documented advanced path; used by this machine's live testing).

## 7. CLI surface

```
stone-llama doctor                       # environment verdict, no changes made
stone-llama setup [--yes] [--cu12|--cu13]  # consent-gated pinned runtime bootstrap (multi-GB)
stone-llama pull <repo[@branch][:quant]> [--force] [--quiet] [--yes]   # A5 gate runs first
stone-llama list [--estimate]            # --estimate adds fit verdict + tok/s [est] columns
stone-llama fit <repo[@branch][:quant]>  # gate + verdict + tok/s [est] — HF metadata only, no download
stone-llama rank --collection <id> | --file <path> [--ratings <file>]  # fit-ordered candidate table
stone-llama rm <model>
stone-llama import <path> [--name N]     # symlink (zero-copy); name defaults to base dir
stone-llama run <model> [--ctx N] [--cache-mode M] [--no-autofit] [-p "prompt"]   # shipped (M6)
stone-llama serve [<model>] [--attach <host:port|url>] [--port <n>] [--key-file <path>]   # OpenAI-compatible API (shipped M5)
stone-llama ps
stone-llama stop
stone-llama login
stone-llama version
```

Stdlib `flag` dispatch (no cobra). Sample output:

```
$ stone-llama pull turboderp/SmolLM3-3B-exl3@3.5bpw
gate: arch  ✓ SmolLM3ForCausalLM
gate: quant ✓ exl3
gate: fit   ⚠ fits only at reduced ctx: Q4 @ 32768 (VRAM-limited; trained max 65536)
      weights 1866 + KV 648 + overhead 128 = 2642 MiB (headroom 1280, budget 2816)
      warning: only 174 MiB margin … drop --ctx if prompts OOM during prefill
warnings above — review them before continuing
pulling turboderp/SmolLM3-3B-exl3 → SmolLM3-3B-exl3: 8 files, 1.84 GiB (sha256-verified), 76.07 GiB free
Proceed? [y/N] y
pulled SmolLM3-3B-exl3 (1.84 GiB) — verdict warn:32768 Q4
```

Between `Proceed?` and `pulled`, each file prints its own progress line
`  [1/8] model.safetensors   100.0%  <rate>/s` (sha256 verified per file, silently) — verified
against `internal/pull` and a live declined run (the gate, warnings, sizes and free-space
lines above are verbatim output; nothing was downloaded).
```
$ stone-llama list
NAME                  QUANT    SIZE      VERDICT       SOURCE
SmolLM3-3B-exl3       3.5bpw   1.84 GiB  warn:32768 Q4 turboderp/SmolLM3-3B-exl3
mistral-7B-exl3       4.0bpw   4.21 GiB  -             imported

$ stone-llama run SmolLM3-3B-exl3
autofit: cache Q4, max_seq_len 32768
  weights 1866 + KV 648 + overhead 128 = 2642 MiB (headroom 1280, budget 2816)
warning: only 174 MiB margin … drop --ctx if prompts OOM during prefill
>>> hello
Hi! …                                          (streamed)
>>> /bye
unloaded.

$ stone-llama ps
stone-llama: running (pid 419204)
  address:  127.0.0.1:5111
  mode:     supervised (child pid 419210)
  ready:    yes
  model:    SmolLM3-3B-exl3
  ctx:      32768  cache: Q4
  vram peak: 2529 MiB
  uptime:   4m12s

$ stone-llama serve
stone-llama listening on 127.0.0.1:5111 (OpenAI-compatible)
  runtime: TabbyAPI f07131c (internal port hidden)
```

**Sample provenance.** The `pull` gate/consent lines are verbatim live output (2026-09-29,
stopped at the consent prompt — nothing downloaded); the `fit` arithmetic matches the
verbatim captures in `docs/screenshots/fit.txt`. The `list` rows and the supervised
`run`/`ps`/`serve` lines are illustrative, rendered from the print paths in `internal/cli`
and `internal/serve` — this machine never ran `setup`, so supervised mode was never
exercised here (see `docs/screenshots/serve.txt` for attach-mode captures that *are* live).

**Config** `~/.config/stone-llama/config.json` (stdlib JSON; no comments; no secrets):

```json
{
  "models_dir": "~/.local/share/stone-llama/models",
  "host": "127.0.0.1",
  "port": 5111,
  "autofit": { "enabled": true, "headroom_mib": 512, "overhead_mib": 128, "min_ctx": 4096 },
  "runtime_dir": "",
  "engine_env": { "CUDA_MODULE_LOADING": "LAZY" },
  "backend": "tabby",
  "backend_device_budget_mib": 0
}
```

**Backend selection.** `Backend` (`internal/serve/backend.go`) is the seam between
the supervision loop and the engine it supervises: `Prepare` owns everything that
must exist before the child starts (validation, 0600 config/key files, the `Conn`
the proxy probes and injects the upstream Bearer from) and `Spawn` is one
incarnation. `BackendFor` resolves the name — `""`/`tabby` → `tabbyBackend`
(TabbyAPI + ExLlamaV3, the default and unchanged behaviour), `llama` →
`llamaBackend` (llama-server over GGUF); anything else is refused, never coerced.
`serve --backend` > `STONE_LLAMA_BACKEND` > config `backend` > default. Readiness
probing, the child lifecycle, supervision and shutdown stay engine-agnostic, and
both `Spawn`s delegate to one `startProcess`.

**llama backend status: UNVERIFIED.** No GGUF weights and no `llama-server` binary
exist on this machine, so the path is exercised only up to `Prepare` (GGUF
resolution, the weights budget, the missing-binary refusal) and its unit tests.
Nothing here has served a token. Two structural limits are implemented, not
guessed: the model is fixed at spawn (`-m`), so `/-/load` refuses **409**
`backend_fixed_model` on this backend instead of falling back to TabbyAPI; and
`backend_device_budget_mib` is our own arithmetic over the GGUF file size —
weights only, KV cache and runtime overhead excluded, so it is a floor on the
weights, not the engine's memory use.

**Precedence:** flag > env > file > default. Env: `STONE_LLAMA_HOST`, `STONE_LLAMA_PORT`, `STONE_LLAMA_MODELS_DIR`, `STONE_LLAMA_CONFIG`, `STONE_LLAMA_NO_AUTOSTART`, `STONE_LLAMA_BACKEND`, `HF_TOKEN`.

**Auth:** proxy injects TabbyAPI key upstream (read from file, never argv/log). Downstream, two header gates run before any handler: `Host` must name loopback or this daemon's own bind host (else **421** `invalid_request`, message `loopback only` — DNS-rebinding defence) and `Origin`, when present, must name that same accept-set (else **403** `browser_origin_forbidden`, message `cross-origin request refused`); an absent `Origin` (curl/CLI) passes. `/healthz` is bearer-exempt but not gate-exempt. The control routes `/-/status`, `/-/load`, `/-/unload` **always** require the daemon state token (`Authorization: Bearer`, constant-time compare → else **401** `unauthorized`, message `bearer token required`), including on a loopback bind. Every other route — the `/v1/*` proxy and the `/api/*` Ollama shim below — requires the token only when the configured bind host is not loopback; `/healthz` never does.

**Tool calling:** a server concern. stone-llama sets the format automatically and reports it; the OpenAI-compatible surface exposes whatever the loaded model advertises, unchanged, to any client or gateway.

**Ollama compatibility shim** (`internal/serve/ollama.go`): four routes translate ollama's API onto the OpenAI surface — `POST /api/chat` → `/v1/chat/completions`, `GET /api/tags` (model-store scan), `POST /api/show` (answerable subset only), `POST /api/generate` → `/v1/completions` (verbatim prompt, no fallback to chat). They sit behind the same Host/Origin gates and the same bearer rule as the chat surface (above). Mapping as implemented: `options.num_predict` 0/absent omits `max_tokens` entirely; `think` true/false → `enable_thinking`, an effort string → `enable_thinking` plus `reasoning_effort` passed through verbatim, absent/null → neither key, anything else → 400; `options.seed` is decoded but never forwarded (the pinned request types have no seed field). Unsupported fields are a clear 400 naming the field, never a silent drop: `/api/chat` rejects `tools`, `format`, `template`, `raw`, `context`; `/api/generate` rejects `format`, `template`, `context`, `suffix`, `system` and accepts `raw` (completions are verbatim by design). `keep_alive: 0` (or `"0s"`) POSTs our own `/-/unload` with the state token after the reply; any other `keep_alive` value is accepted and ignored — the daemon is resident by design.

**What the shim refuses to invent.** Durations not observable through the OpenAI surface (`load_duration`, `prompt_eval_duration`, `eval_duration`) are sent as `0`; `total_duration` is the request time we actually measured. `prompt_eval_count`/`eval_count` appear only when the upstream sends usage. Fields without a real on-disk source are omitted, never fabricated: `/api/tags` carries no `digest`, and `details` reports only what the model directory shows (`family` from `config.json`'s `model_type`, `format` `safetensors`, `quantization_level` from the manifest) — `parameter_size` stays out. `/api/show` answers `details`, a limited verbatim `model_info` subset, and `template` only when a template file exists on disk; `modelfile`, `parameters`, `license` and friends are omitted. `done_reason` is `length` or `stop` — nothing else is claimed. **Unverified:** the `/v1/completions` call on the generate route and the `enable_thinking`/`reasoning_effort` fields were never exercised against a live backend (TabbyAPI 5002 is down and stays down); their routes/types are verified against the pinned source tree and unit tests only, not proven live.

**Drafting + engine env.** Speculative drafting is configured via the boot `config.yml` `draft_model` block (`autofit.draft_mode` / `ngram_match_min` / `draft_num_tokens` in config.json; `serve --draft-mode ngram|off` overrides the mode). Default is **off**: no block is rendered at all, and `draft_model_dir` never renders (no draft weights ship). The block is daemon-level — it rides child start, not `/-/load`. Its effect is **unmeasured**: no A/B has been run, so no speedup is claimed. `run --draft-mode` parses and validates the value only — it cannot reach an already-running daemon and therefore has no effect. Config `engine_env` (a `map[string]string`) is extra environment for the supervised child: the parent environment is preserved (`PATH`/venv discovery unaffected) and config entries append last, so they win over a same-named parent variable.

## 8. Platform reality + hard limits (approved Q5/Q8)

**NVIDIA GPU with CUDA required. No CPU path, no ROCm, no Metal. Not configurable.**

- **Works:** Linux x86_64, NVIDIA driver **≥ 570** (→ `cu12` extra) or **≥ 580** (→ `cu13` extra). `doctor` picks the extra by driver; older → refusal with exact upgrade hint.
- **Cannot work on:** any machine without an NVIDIA GPU (AMD/Intel/Apple/CPU-only) — permanently, unless exllamav3 grows a backend; Windows/macOS/ARM in v1 (wheels exist for Windows → G9).
- **Minimum viable GPU:** `min_vram ≈ weights_mib + kv(4096, Q4) + 128 + 512` → SmolLM3-3B @3.5bpw ≈ **2.6 GiB** → any 4 GB card runs 3B-class; 8 GB handles 7–8B @4bpw. Doctor prints model-specific math, not a generic refusal.

```
$ stone-llama setup
✗ stone-llama needs an NVIDIA GPU (CUDA). None detected on this machine.
  Detected: no nvidia-smi executable.
  stone-llama runs ExLlamaV3, which is CUDA-only — no CPU, AMD, or Apple support.
  For CPU-only or AMD machines, use ollama with GGUF models instead.
```

README carries this in the first screen (Q10).

## 9. Packaging + **A8 repo/delivery shape**

- **Artifact:** `stone-llama-linux-amd64.tgz` (~5.5 MB bundle, 5,719,364 B; binary 8,933,560 B ≈ 8.5 MiB): binary + README.md + LICENSE + stone-llama.png; `-trimpath -ldflags "-s -w"`; sha256 published in the combined `checksums.txt`. Five targets built (linux amd64/arm64, windows amd64, darwin amd64/arm64) (G9: windows/macOS/arm64 runtime untested).
- **Install:** GitHub Releases + `install.sh` (download, verify sha256, `~/.local/bin`, PATH hint). No package managers in v1 (G11).
- **Release:** tag → build + sha256 → release (the §10.5 contract test is still a planned gate, not a step that runs today). No Docker, no telemetry.
- **A8 — repository (now):** `git init` at project start; **one commit per milestone**, conventional format (`feat(m1): …`); `.gitignore` excludes built binaries (`/stone-llama`, `/bin/`, `/dist/`), `/models/`, `/downloads/`, `/runtime/`, `*.part`, `/logs/`, `*.log`. **Public repo, owner `rizperdana`** — `gh auth status` verified this session: active account is `rizperdana` ✓ (gate satisfied; the repo **exists** at github.com/rizperdana/stone-llama). Plan doc committed as `docs/ARCHITECTURE.md`.
- **Residual risk:** G16.

## 10. Testing strategy — zero weight downloads (A5 gate testable from fixtures)

1. **Autofit math:** table tests over fixture `config.json` in `testdata/` — real SmolLM3, synthetic 1-layer, high-kv-head, tied, >128k-ctx, missing `head_dim`. Assert exact `(ctx, mode)`, hard-clamp behavior (Q3a: user ctx > trained → clamped), failure breakdown text.
2. **A5 gate:** fake-HF `httptest` fixtures serving tiny `config.json`/`quantization_config.json`/trees → arch warn/confirm, `exl2` refusal, GGUF-only refusal, unknown-arch warning text, fit verdicts (proceed/warn/refuse + largest-fitting-ctx).
3. **Pull against fake HF:** Range-resume after simulated interrupt, checksum mismatch, gated 403, disk-preflight refusal. KB fixtures; byte-identical code path.
4. **Proxy/daemon vs fake TabbyAPI:** ~100-line Go stub with SSE canned tokens → passthrough, key injection, crash-backoff, port-occupied handling. No Python.
5. **Contract test vs pinned TabbyAPI:** `go:build integration`, local only, no model load (config schema + routes). M4/M5 gate. **Status: planned — no integration-tagged file exists yet**, so §2/§9 must not cite it as an enforced gate.
6. **GPU E2E:** on-disk SmolLM3 only (zero download). M5 serve + curl; M6 interactive.

CI runs 1–4: no GPU, no Python, no external network.

## 11. Milestones (dependency order; one commit each per A8)

| # | Milestone | Size | Depends | Gate |
|---|---|---|---|---|
| M0 | Repo skeleton: `git init` + `.gitignore`, subcommand dispatch, config+env precedence, `version`, `doctor` (nvidia-smi probe, driver→extra) | **S** | — | unit: config precedence; doctor with fake `nvidia-smi` on PATH |
| M1 | Store: models-dir scan, `list`/`rm`/`import`, manifest read/write (incl. `verdict` field) | **S** | M0 | tempdir fixtures, fake model dirs, symlink safety |
| M2 | Pull: **A5 pre-download gate** (arch/quant/fit, embedded arch list), HF resolve/search/picker, resumable downloader, HEAD-size consent preflight, sha256 verify, progress, gated token; flock lockfile primitive (first A7 consumer) | **L** | M1 | fake-HF suite: gate + interrupt-resume + checksum paths |
| M3 | Autofit: config parse, ladder + Q3 guard rails (clamp/overrides/always-print), failure breakdown, YAML emission | **M** | M0 | fixture table tests (§10.1); M2∥M3 |
| M4 | Runtime bootstrap: lock file, consent gate, step journal (resume), **proven entirely on stubs + throwaway `runtime_dir`**; `doctor` deep; repair; error mapping. **Real multi-GB provisioning = BLOCKED step (Q6), separately authorized** | **L** | M0 | stubbed-binary tests; no network fetches in tests |
| M5 | Serve: daemon + auto-start/stop, TabbyAPI supervision, config generation, proxy, health/backoff, `ps`, daemon-spawn flock usage, `daemon.json` 0600 | **L** | M2+M3+M4 | fake-tabby suite + real E2E with on-disk SmolLM3 |
| M6 | Run: streaming REPL, flag passthrough, load progress, `/bye`, error polish | **M** | M5 | interactive + SSE tests |
| M7 | Packaging: tarball, sha256, install.sh, honest README, `THIRD-PARTY.md` final — **licences all verified (A6 gate cleared in `1aaa785`)** | **S** | M5 | install.sh dry-run in clean dir |

## 12. GAP ANALYSIS

**Strategic/inherited from v1 (approved posture):**

- **G1 · CUDA/driver skew (HIGH).** Dual extras widen the floor (570/580) but old drivers, old glibc (<2.28 manylinux), musl, and OEM driver branches still refuse. *Mitigation:* doctor preflights free and instantly, before any download. *Residual:* a fine GPU on a 4-year driver gets a refusal, not a workaround.
- **G2 · Multi-GB bootstrap UX (HIGH).** Sizes unmeasured until M4 (HEAD-measured at consent time mitigates); uv large-download resume imperfect — mitigated by step journal + per-step resume, not by pretending uv streams perfectly; setup may run 10+ min. *Residual:* torch is the funnel; no design makes it small. Q6 means we ship M4 unproven-on-real-hardware until the user authorizes the blocked run.
- **G3 · GPU-less/AMD users (HIGH, unfixable).** Hard no, stated plainly (§8). README shows "use ollama if …" (Q10).
- **G4 · TabbyAPI coupling (MED-HIGH).** Pinned config schema + OAI routes + auth-file behavior relied upon; contract test is the early-warning; pin bumps are deliberate releases. *Residual:* if upstream goes dormant we inherit a Python server; we do not fork (Q11).
- **G5 · EXL3 supply ceiling (MED-HIGH).** Few hundred EXL3 repos; some models have none at all (research doc: Llama-3.2-3B, Phi-3.5-mini — architecture supported, **no EXL3 published**). Zero-candidate messages redirect to Ollama+GGUF; conversion impossible on 4 GB (research doc §6: conversion needs ≥2× FP16 size in VRAM). *Residual:* product ceiling = upstream quant supply.
- **G6 · Concurrency (MED).** Single-model serving is a non-goal (Q9); second load while loaded = restart child (M5 decision, default restart). Requests during load queue/error per TabbyAPI behavior.
- **G7 · vs Ollama+GGUF (STRATEGIC).** For ≤3B on any hardware, Ollama wins on convenience. Our honest value: EXL3 speed/VRAM-frugal on NVIDIA, 65k ctx autofit on 4 GB cards, front-end for TabbyAPI users. README says exactly this + "use ollama if …" (Q10).
- **G8 · Multi-GPU (LOW-MED).** `gpu_split` passthrough only; no per-GPU budget math in v1 (no test hardware).
- **G9 · Windows (MED).** Upstream wheels exist; provisioning/paths/daemonization differ. Explicitly out of v1; claiming untested support is worse than none.
- **G10 · Beyond-pull integrity + chunked downloads (LOW).** No background re-verify (found on re-pull); no concurrent-chunk downloader (gigabit users slower). Both deliberate (§4.2), upgrade paths named.

**New — amendments:**

- **G13 · A5 gate correctness (MED).** *Risk:* bundled arch list drifts with exllamav3 releases → stale warnings (or stale confirms); `quant_method` absent in odd repos → heuristic fallback must not false-refuse (design: warn+confirm, only `exl2` and GGUF-only refuse — refusals are format-certain, warnings cover uncertainty); some repos put `config.json` only in quant subdirs → resolution fetches the subdir config (M2). *Residual:* a newly-supported arch not in our list ships as a warning until next release; a newly-**broken** arch still in our list passes the gate — the gate catches format/scale errors, not upstream regressions.
- **G14 · A6 licensing (CLEARED).** TabbyAPI **AGPL-3.0** verified — we neither distribute nor modify (reasoning recorded in §6); forking would change that analysis (reinforces Q11). `uv`, Python, NVIDIA wheels were the last unverified entries; commit `1aaa785` verified **every** licence from real sources and emptied `THIRD-PARTY.md`'s UNVERIFIED section ("Still open: Nothing"). Publishing with guessed licenses was the failure mode this gap existed to prevent — it can no longer happen. Remaining open project items are tracked elsewhere: M5/M6 delivery, the deferred provisioning run (Q6), Windows runtime untested (G9).
- **G15 · A7 singleton + secret hygiene (LOW-MED).** flock correct on local FS; pathological cases (NFS lock weirdness, SIGKILL mid-download leaving `.part` — acceptable, resume handles it, and `.part` is never loadable as a model). Token exposure: injected in proxy headers only, never argv/log/env — audited at M5 (grep-level test: logs contain no token substring).
- **G16 · A8 delivery shape (LOW).** Public repo from day one → no secrets/weights may ever enter git (`.gitignore` + review); one-commit-per-milestone = coarse history (accepted: matches user's "regular commits" ask with reviewable per-milestone gates); `gh` gate verified (`rizperdana` active) and the public repo exists.

## 13. Remaining open points (defaults stand unless director objects)

All v1 §13 questions are answered (decision log, top). Residual points with standing defaults:

1. **Arch-list entries where the research doc elided the exact class string** (e.g. DeepSeek V4, GLM5Next, Step 3.7) → default: omit from the embedded list; those land on the warn path (safe by design). Reconcile against `architecture/*.py` at M2.
2. **`list` output gains a `VERDICT` column** (A5) → default: always shown, `-` when absent (shown in §7 sample).
3. **Ladder modes are FP16/Q8/Q4/`"4,2"`**; Q6 is skipped deliberately (Q4 ≈ lossless for KV; Q6 @65536 measured OOM on this 4 GB card) and Q2/`"2,2"` pairs stay override-only → default: as stated.
4. **GitHub repo creation** → **done**: the public repo exists under `rizperdana` (the A8 gate — verified `gh` account — was satisfied first).
5. **Real provisioning run (blocked step, Q6)** → default: when authorized, run to a throwaway `runtime_dir` first (prove installer), then real paths only on second authorization.
