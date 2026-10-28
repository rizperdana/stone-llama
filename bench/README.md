# Benchmark: stone-llama vs ollama (real measurements)

Compares **stone-llama** (tabby/exllamav3, EXL3) against **ollama** on localhost on the same machine. All numbers are real (timed via OpenAI-compatible streaming API, VRAM sampled with `nvidia-smi`). No fabricated/estimated values.

## Inventory (at time of run)
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB), driver 580.178.04
- stone-llama models: `SmolLM3-3B-exl3_4.0bpw` (EXL3, on disk), `Qwen_Qwen3-4B-Instruct-2507-EXL3`
- ollama models: `smollm3:3b` (Q4_K_M), `smollm3-withsystem:latest`, `smollm3-nosystem:latest`, `qwen3.5:2b`, etc.
- Comparable pair: `SmolLM3-3B-exl3_4.0bpw` (stone-llama) vs `smollm3:3b` (ollama) — same base model family

## Workload
- Prompt: `"Explain quantum entanglement in simple terms."`
- `max_tokens=128`, `temperature=0.7`, stream=true
- `N=5` repeats per server, gap 1s between runs, VRAM sampled at ~100ms intervals
- TTFT = time to first token (streaming). decode tok/s computed from completion tokens when reported else stream chunks as approximation.

## Reproduce
```bash
./bench/run.sh                # measure against already-running servers on 5199/11434
STONE_START=1 ./bench/run.sh  # also start/stop a stone-llama daemon on STONE_PORT
```
Script uses `bash + curl + python3 stdlib` only. Only probes/read-only queries `nvidia-smi`; never downloads, never restarts/stops/kills ollama, never signals other processes.

## Results (2026-10-28)
See `bench/results-20261028.json` for full raw measurements and per-process VRAM peaks.

### Summary
| Metric | stone-llama (tabby/exl3, SmolLM3-3B-exl3_4.0bpw) | ollama (smollm3:3b Q4_K_M) |
|--------|---------------------------------------------------|----------------------------|
| Runs OK | 5/5 | 5/5 |
| TTFT mean (s) | 0.168 | 3.786 |
| TTFT min/max (s) | 0.127 / 0.233 | 0.106 / 18.248 |
| Wall mean (s) | 2.968 | 21.625 |
| Decode tok/s mean | 45.4 | 7.2 |
| Peak VRAM used (MiB, sampled in windows) | 2283 | 2724 |

**Stone-llama won** on TTFT, wall time, and decode tok/s by a large margin. Ollama showed a large cold-start penalty on first run in this measurement; even with subsequent runs much faster, stone-llama remained substantially faster overall. Bottleneck observed: ollama had much higher TTFT/cold start and lower decode throughput on this setup; stone-llama (exl3/tabby) loaded faster and sustained higher tokens/s with lower peak VRAM in the measured windows.
