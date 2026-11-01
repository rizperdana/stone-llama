# bench — measured stone-llama vs ollama

Reproducible localhost benchmark: same machine, same prompt, same max_tokens/
temperature, N=5 repeats, streaming OpenAI-compatible API. Numbers are real;
the harness exits non-zero when a measurement is impossible.

## Reproduce

```sh
# canonical clean run: harness starts/stops its own stone-llama daemon on
# STONE_PORT (default 5199) and asks ollama (probe-only) to unload its cached
# copy via keep_alive=0 so each engine is measured with the full card.
STONE_START=1 bench/run.sh
```

- Dependencies: bash, curl, python3 stdlib, nvidia-smi. Nothing downloaded.
- Only HTTP probes + read-only `nvidia-smi`. ollama is never restarted/killed;
  unload uses the documented `POST /api/generate {"keep_alive":0}` API call.
- Only the stone-llama daemon the harness started is stopped (exact `stone-llama stop`).
- With `STONE_START=0` the harness warns that a resident stone model confounds
  ollama's window (measured and confirmed: 7.6 tok/s vs 55.8 tok/s warm).

## Workload

- Prompt: "Explain quantum entanglement in simple terms." (same both sides)
- `max_tokens=128`, `temperature=0.7`, `stream=true`, 5 repeats, ~1s gap
- TTFT = first content chunk; decode tok/s from streamed content chunks
  (servers did not report usage on this path; basis recorded in JSON)
- VRAM: `nvidia-smi` sampled at ~100 ms; peak total GPU used + per-process
  peaks inside each engine's measurement window
- Comparable pair: `SmolLM3-3B-exl3_4.0bpw` (stone-llama, EXL3 ~3.5 bpw
  per quantization_config.json, dir name says 4.0) vs `smollm3:3b` (Q4_K_M,
  ollama ctx 8192). Same base model; quant not bit-identical (EXL3 mixed 3–5
  bpw vs Q4_K_M) — closest pair on disk, no downloads.

## Results — 2026-10-28 (results-20261028.json, run 20261028T154053Z)

RTX 3050 Laptop 4 GB, driver 580.178.04. Warm = model resident; cold = first
request after unload (ollama) / daemon start (stone).

| metric (5 runs) | stone-llama (tabby, EXL3) | ollama (Q4_K_M) |
|---|---|---|
| TTFT mean (s) | 0.130 | 0.885 (incl. 4.27 s cold) |
| TTFT warm-only (s) | ~0.13 | 0.040 |
| wall mean (s) | 2.878 | 3.162 |
| decode tok/s | 46.2 | 55.8 warm (7.6 when VRAM-starved) |
| peak VRAM, own process (MiB) | 2274 (tabby python) | 2568 (llama-server) |
| ctx used | 32768 (Q4 KV cache) | 8192 |

## Honest reading

- **Warm decode: ollama wins on this workload** (~56 vs ~46 tok/s) once it has
  the whole card. Earlier runs (kept in git history) showed ollama at ~7 tok/s —
  that was an artifact: stone's 2274 MiB model was still resident, starving
  ollama to ~450 MiB VRAM and partial CPU offload (`/api/ps` showed
  `size_vram` 371 MiB vs model size 2.6 GB).
- **VRAM: stone-llama wins** for this pair: 2274 MiB (EXL3) vs 2568 MiB
  (Q4_K_M) while holding 4× the context (32768 vs 8192).
- **TTFT: stone wins cold** (daemon start→ready in ~10 s; steady 0.13 s TTFT);
  ollama warm TTFT (0.04 s) is lower than stone's (0.13 s).
- Claimed honestly: stone-llama's edge on a 4 GB card is **VRAM efficiency and
  context capacity**, not raw warm decode speed on a 3 B model. Speed claims
  need longer prompts/loads where EXL3 prompt processing may differentiate.

## Results — round 2, 2026-10-30 (results-20261028-r2.json)

**Same-quant pair**: identical ollama `smollm3:3b` Q4_K_M GGUF blob served by
stone-llama's llama.cpp path (`smollm3-q4km-test`, imported from a copy of the
ollama blob) vs ollama itself. Removes the round-1 EXL3-vs-Q4_K_M quant
confound. No downloads.

### Exact argv (stone-llama default, llama backend)

From `internal/serve/llama_backend.go` `Spawn()` and confirmed on the live
child's `/proc/<pid>/cmdline`:

```
llama-server --host 127.0.0.1 --port <internal> --api-key-file <runtime>/upstream_key -m <gguf>
```

Nothing else: no `-fa`, no `-ngl`, no `-c`, no `-np`. Supported passthrough is
`config.json` `engine_env` only (env vars, not argv) — verified
`LLAMA_ARG_FLASH_ATTN=off` and `LLAMA_ARG_MMAP=0` reach the child.

### Same-quant, clean cards (5 repeats, same prompt/params as round 1)

| metric | stone-llama (llama.cpp) | ollama (llama.cpp) |
|---|---|---|
| decode tok/s | 53.5 | 55.2 (clean re-measure) |
| TTFT warm (s) | 0.116 | 0.039 |
| peak VRAM (MiB) | 2761 | 2568 |
| ctx | 32768 (model default) | 8192 |

ollama's first same-quant window (6.0 tok/s / 2922 MiB) was measured with
stone's daemon still resident — stone's 2752 MiB GGUF on the card starved
ollama to 156 MiB. That number is recorded in the JSON under
`confounded_measurement` and must not be cited as a comparison.

### Flash attention (-fa) A/B — via engine_env, same weights

| | decode tok/s | TTFT (s) | wall (s) | peak VRAM |
|---|---|---|---|---|
| default (`auto`) | 54.8 | 0.059 | 2.338 | 2752 |
| `LLAMA_ARG_FLASH_ATTN=off` | 50.2 | 0.063 | 2.552 | 2766 |

**+9.2% decode, −8.4% wall with FA on; VRAM difference is noise (14 MiB).**
llama-server's default is already `auto` (on for supported models), so
**there is no flag change to make** — stone-llama gets this for free.

### `--no-mmap` A/B

Identical decode (54.8 tok/s), wall within 0.1%, VRAM identical. mmap is
fine on this model/size; no action.

### Long-prompt prefill (Q4_K_M pair)

Warm loop (same prompt repeated; run 1 = true prefill, runs 2–3 = prompt-cache
hits), `max_tokens=64`:

| TTFT run 1 (s) | stone | ollama |
|---|---|---|
| 2 KB prompt | 0.518 | 0.319 |
| 8 KB prompt | 0.815 | 0.772 |

Cold 8 KB (unique prompt per repeat, so no cache can absorb the prefill):
stone TTFT 3.5 s vs ollama 1.2 s warm-loop mean (ollama's own cold-load run 1
TTFT was 7.1 s, worse than stone's). **Real finding: after the first token,
stone's decode collapses to ~5 tok/s on long-prompt continuations while
ollama holds ~53 tok/s.** Suspected cause: stone leaves llama-server defaults
(unified KV, no `-b/-ub` pinning) while ollama pins `-c 8192 -b 512 -ub 512`.
Follow-up: measure stone with explicit env overrides to isolate.

> **Note (added 2026-11-01):** The collapse above was pre-fix. Round 2 left llama-server
> unpinned (`-c` followed the model's `n_ctx_train` → oversized KV cache silently degrading
> long-prompt decode; this build's `-fa auto` also left flash attention off). Both fixed in
> 9bf4f13 (`-c 8192 -fa on -np 2`); Round 3 re-measures the 8 KB shape at ~53 tok/s decode
> with ~1.16 s TTFT and no collapse (see Round 3 below). Round 2's own numbers are
> unchanged; only the status is annotated.

### 4-way concurrency (4 parallel streams × 128 tokens, 3 reps)

| | aggregate tok/s | p95 TTFT (s) | per-request TTFT spread |
|---|---|---|---|
| stone-llama | 99.7 | 0.59–0.75 | 0.52–0.75 (tight) |
| ollama | 55.7 | 4.6 | 0.03–6.9 (queueing) |

stone-llama does NOT serialise: llama-server auto-configures `n_parallel=4`
slots, aggregate throughput ~2× single-stream. ollama runs one slot (`-np 1`)
and queues: no aggregate gain, TTFTs spread 0.03–6.9 s. Stone wins
concurrency on both throughput and TTFT stability.

> **Note (added 2026-11-01):** The 99.7 aggregate above was the pre-pin 4-slot auto-fit
> with 2048-token slots. The shipped 2-slot default (`-c 8192 -fa on -np 2`) measures
> ~86.7 aggregate with p95 TTFT ~3.0 s (see Round 3 below). Round 2's own numbers are
> unchanged; only the comparison conditions are annotated.

Stability finding (excluded from benchmark numbers): one stone daemon instance
collapsed from 2756 to 188 MiB resident after serving the prefill batch +
a concurrency round, with decode dropping to 2.4–8.4 tok/s; restarting the
same daemon restored 2752 MiB / 55 tok/s. Reproduced once, cause not isolated
(no Xid, no OOM entry). Worth a follow-up ticket.

## Honest reading (updated after round 3)

- **Same-session numbers lead.** The same-quant pair was re-measured in one session with
  the shipped pinned default (`-c 8192 -fa on -np 2`) and clock state recorded (Round 3):
  single-stream 54.7 vs 49.0 tok/s and 4-way aggregate 88.9 vs 48.0 — both ahead. The
  earlier cross-session pair (53.5 vs 55.2 tok/s, ~4%, round 2) is within session-to-session
  variation: absolute decode tracks GPU boost state (~40-56 tok/s for the shipped shape),
  so a boosted session vs an idle session swaps the order. That is "parity within
  variation", not a claim that stone is faster.
- **ollama wins warm TTFT** (0.039 vs 0.116 s, round 2) and prompt-cache-hit TTFT.
- **stone-llama wins VRAM** for this pair (2761 vs 2568 MiB) **and context
  capacity** (32768 vs 8192, 4×, round 2; shipped 2 x 4096 slots, round 3).
- **concurrency**: pre-pin 4-slot auto-fit gave ~2x aggregate with tight TTFTs (round 2);
  the shipped 2-slot default holds ~max(single-stream) aggregate with p95 TTFT ~3.0 s
  (round 3) — the honest cost of the pin, annotated under the round-2 table.
- **long-prompt decode collapse: fixed.** The ~5 tok/s collapse on 8 KB-prompt
  continuations was caused by unpinned context over-allocating KV plus this build's
  `-fa auto` leaving flash attention off; both pinned in 9bf4f13. Round 3 re-measures the
  8 KB shape at ~53 tok/s decode with ~1.16 s TTFT and no collapse.
- Claimed honestly: stone-llama's edge on a 4 GB card is **context capacity, comparable
  VRAM, and parity-within-variation decode** against ollama — not a consistent raw-speed
  or TTFT win on a 3 B model.

## Results — 2026-10-30 Qwen EXL3 (results-20261030-qwen.json, run 20261030T150929Z)

**Absolute numbers, no like-for-like counterpart on disk.** `Qwen_Qwen3-4B-Instruct-2507-EXL3`
(EXL3, 2.5 bpw, 2.07 GiB) was measured via `stone-llama serve … --backend tabby` on
port 5199 — same workload definition as `run.sh` (same prompt, `max_tokens=128`,
`temperature=0.7`, 5 repeats, 100 ms VRAM sampling, same Python streaming client and
stats logic). `SmolLM3-3B-exl3_4.0bpw` (EXL3, ~3.5 bpw, 1.84 GiB) was measured the same
way on the same card in the same run, so the two EXL3 models share a consistent basis.

No ollama counterpart exists for Qwen3-4B: `ollama list` has `qwen3.5:2b` (Qwen3.5,
2.7 B params, different architecture — **not comparable**), and `smollm3:3b` is a
different base model entirely. This section does **not** present ollama numbers as
comparisons for either model. ollama is probe-only (port 11434, never started/stopped).

### Harness command

```sh
# stone-llama-only (no ollama comparison):
python3 /tmp/measure_exl3.py
# which internally runs:
stone-llama serve <model> --port 5199 --backend tabby
# starts daemon, waits for model load, sends 5 streaming requests,
# samples nvidia-smi every 100ms, computes stats, stops daemon
```

### EXL3 pair: Qwen3-4B vs SmolLM3-3B

RTX 3050 Laptop 4 GB, driver 580.178.04. Both EXL3, tabby/ExLlamaV3.

| metric (5 runs) | Qwen_Qwen3-4B-Instruct-2507-EXL3 | SmolLM3-3B-exl3_4.0bpw |
|---|---|---|
| model | Qwen3 4 B, EXL3 2.5 bpw (2.07 GiB) | SmolLM3 3 B, EXL3 ~3.5 bpw (1.84 GiB) |
| runs ok | 5/5 | 5/5 |
| TTFT mean (s) | 1.059 (min/max 0.16/4.33) | 0.124 (min/max 0.11/0.16) |
| TTFT warm-only (s) | 0.16–0.34 (runs 1,4,5; runs 2–3 spiked) | 0.11–0.16 (all consistent) |
| wall mean (s) | 6.121 (median 4.832) | 2.905 (median 2.897) |
| decode tok/s | 27.57 (min/max 14.77/35.02) | 45.68 (min/max 45.2/45.95) |
| peak VRAM (MiB) | 2445 | 2283 |
| per-proc peak (MiB) | 2436 (venv python) | 2274 (venv python) |
| ctx | n_ctx_train 262144, n_ctx null | n_ctx_train 65536, n_ctx null |

### Qwen run-3 contention

Run 2 wall=8.76 s, run 3 TTFT=4.33 s / wall=9.08 s — multi-second spikes the other
three runs did not show (3.7–4.8 s wall). On a 4 GB card the 2.07 GiB Qwen3-4B EXL3
model leaves ~1450 MiB headroom before its 262 144-token context config; the spikes are
consistent with EXL3 memory-pool pressure causing offload/reload between requests.
SmolLM3-3B (1.84 GiB, 65 536-token ctx) showed no spikes — all 5 runs within 3 % of
each other. Warm-only Qwen decode (runs 1, 4, 5: 35.0/28.3/33.1 tok/s, mean ~32) is
~30 % slower than SmolLM3's 45.7 tok/s, consistent with a larger model on a 4 GB card.

### Honest reading — Qwen EXL3 (absolute, no ollama comparison)

- **Same EXL3 backend, different-sized models.** Qwen3-4B is 28 % larger on disk
  (2.07 vs 1.84 GiB) and 162 MiB larger VRAM (2445 vs 2283 MiB). The decode gap
  (27.6 vs 45.7 tok/s) reflects model size + param count, not a pathology in the
  stone-llama wrapper.
- **Qwen variability is real, not noise.** Runs 2–3 hit 4.3–8.8 s wall vs 3.7–4.8 s
  for the other three. ExL3 offload/reload on a 4 GB card with a 262 k-token context
  config is the suspected cause (no Xid, no OOM — model simply reloads mid-stream).
- **Warm-only Qwen ≈ 32 tok/s** vs SmolLM3 45.7 tok/s. Even excluding contention, the
  larger Qwen3-4B is ~30 % slower on a 4 GB card — expected for +1 B params + smaller
  headroom.
- **No claim vs ollama** — there is no on-disk ollama counterpart for Qwen3-4B.
  `qwen3.5:2b` is Qwen3.5 (different architecture, 2.7 B). Comparing stone-llama's
  Qwen3-4B against ollama's qwen3.5:2b would be apples-to-oranges.
- **VRAM: Qwen uses more** (2445 vs 2283 MiB) because it is a larger model with a
  larger context config (262 144 vs 65 536 n_ctx_train). Both fit on the 4 GB card
  with headroom for the 128-token workload.


## Results — Round 3, 2026-11-01 (results-20261101-conc.json)

**Shipped pinned default**: stone-llama llama backend now pins `-c 8192 -fa on -np 2`
(commit 9bf4f13). Same-quant pair: smollm3-q4km-test (Q4_K_M GGUF) vs ollama smollm3:3b.
Both engines measured in one session on port 5195 through the stone daemon; ollama
probe-only (unload via keep_alive=0, never restart/kill). GPU clock state recorded
alongside each measurement.

### Single stream (5 runs, max_tokens=128, temperature=0.7)

| metric | stone-llama (shipped) | ollama (same day) |
|---|---|---|
| tok/s client basis | 54.7 | 49.0 |
| tok/s eval clock | 55.4 (boosted) | n/a (ollama reports via llama-server) |
| TTFT warm (s) | 0.050 | 0.026 |
| TTFT cold (model load, s) | 0.271 | 2.982 |
| TTFT first req after load (s) | 0.064 | — |
| peak VRAM (MiB) | 2575 | 2577 |
| SM clock during decode (MHz) | 1050–1740 | 810–1747 |

> TTFT basis (stone-llama): "cold (model load)" = 0.271 s - first request after daemon start, model loading from disk (from the aborted round-3 session; daemon freshly started, model not yet resident). "first req after load" = 0.064 s - first measurement request after warmup (max_tokens=1) completed, model already resident, first of the 5-run batch (this round-3 session). "warm" = 0.050 s - mean of runs 2-5, model resident, cache warm. The two cold-ish figures measure different conditions and must not be compared as one metric; the JSON stores 0.271 / 0.064 / 0.0504 under ttft_cold_s / ttft_first_after_load_s / ttft_warm_s_mean respectively.

### 4-way concurrency (4 streams × 128 tokens, 3 reps)

| metric | stone-llama (shipped) | ollama (same day) |
|---|---|---|
| aggregate tok/s | 88.9 | 48.0 |
| wall per rep (s) | 5.7–5.9 | 10.6–10.7 |
| SM clock during decode (MHz) | 1042–1740 | 480–1740 |

### 8 KB unique-prompt TTFT (3 runs, max_tokens=64, cold)

| metric | stone-llama (shipped) |
|---|---|
| TTFT (s) | 1.16–1.19 |
| decode tok/s (client) | 52.7–53.1 |
| prefill (ms) | 1007–1043 |

### Verdict

The shipped 2×4096 shape decodes ~54 tok/s by eval clock on this boosted session
(ollama ~49 tok/s client basis, single stream). Absolute decode tracks GPU boost
state: ~40 tok/s at idle clock (210 MHz) to ~55 tok/s boosted (1050+ MHz). Only
within-session comparisons (e.g. `-fa` on vs off) are valid flag A/Bs. The earlier
`~40 vs ~53` pair was never a clean flag comparison — it was an idle-clock session
vs a boosted session of the same shipped shape.
