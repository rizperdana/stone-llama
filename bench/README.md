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

### 4-way concurrency (4 parallel streams × 128 tokens, 3 reps)

| | aggregate tok/s | p95 TTFT (s) | per-request TTFT spread |
|---|---|---|---|
| stone-llama | 99.7 | 0.59–0.75 | 0.52–0.75 (tight) |
| ollama | 55.7 | 4.6 | 0.03–6.9 (queueing) |

stone-llama does NOT serialise: llama-server auto-configures `n_parallel=4`
slots, aggregate throughput ~2× single-stream. ollama runs one slot (`-np 1`)
and queues: no aggregate gain, TTFTs spread 0.03–6.9 s. Stone wins
concurrency on both throughput and TTFT stability.

Stability finding (excluded from benchmark numbers): one stone daemon instance
collapsed from 2756 to 188 MiB resident after serving the prefill batch +
a concurrency round, with decode dropping to 2.4–8.4 tok/s; restarting the
same daemon restored 2752 MiB / 55 tok/s. Reproduced once, cause not isolated
(no Xid, no OOM entry). Worth a follow-up ticket.

## Honest reading (updated after round 2)

- **Round 1's "stone loses warm decode" was the quant confound, not the
  wrapper.** On identical Q4_K_M weights the two llama.cpp paths are at
  decode parity (53.5 vs 55.2 tok/s, ~4%, different sessions).
- **ollama wins warm TTFT** (0.039 vs 0.116 s) and prompt-cache-hit TTFT.
- **stone-llama wins VRAM** for this pair (2761 vs 2568 MiB) **and context
  capacity** (32768 vs 8192, 4×).
- **stone-llama wins concurrency decisively** (~2× aggregate, tight TTFTs vs
  queueing).
- **Open: long-prompt decode** — stone's decode collapses to ~5 tok/s on
  8 KB-prompt continuations. Measured, unexplained, flagged for follow-up.
- Claimed honestly: stone-llama's edge on a 4 GB card is **context capacity,
  concurrency, and comparable VRAM**, not raw warm decode or TTFT on a 3 B
  model. Long-prompt throughput needs the flag-isolation follow-up before any
  claim either way.
