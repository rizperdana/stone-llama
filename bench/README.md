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
