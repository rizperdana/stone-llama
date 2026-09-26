# Quickstart

First run on a supported machine (Linux/amd64, NVIDIA GPU, driver ≥ 570).
Outputs below are **real captures** from this machine (RTX 3050 Laptop, 4096 MiB,
driver 580.178.04) — raw files in [screenshots/](screenshots/).

## 1. `doctor` — environment check, no downloads, no changes

```console
$ stone-llama doctor
GPU        NVIDIA GeForce RTX 3050 Laptop GPU
VRAM       4096 MiB
Driver     580.178.04
Runtime    cu13 extra
```

Driver decides the runtime extra: ≥ 580 → `cu13`, ≥ 570 → `cu12`, older → refusal
with the upgrade hint. No NVIDIA GPU → hard refusal pointing you at ollama + GGUF.

## 2. `setup` — provision the inference runtime (multi-GB, consent-gated)

```bash
stone-llama setup
```

Downloads several GB (PyTorch + CUDA runtime wheels dominate) into
`~/.local/share/stone-llama/runtime/`: pinned `uv` → CPython 3.12 → TabbyAPI at a
pinned commit → hash-locked wheels. Sizes are HEAD-measured and printed, free disk
is checked, then **one** confirmation. Resumable via a step journal; log at
`logs/setup.log`. Details: [INSTALLATION.md](INSTALLATION.md#the-setup-provisioning-step-multi-gb-separate-from-install).

## 3. `fit <repo>` — will it run, how fast? **No download.**

Metadata only (KB from HuggingFace): architecture / quant-format gate plus the
autofit arithmetic against *your* VRAM, then a predicted tok/s **[est]**:

```console
$ stone-llama fit async0x42/Qwen3-1.7B-exl3_4.0bpw
```

(exit 0 = fits / fits-with-warning, exit 3 = refused with the full breakdown.)

What that gate block looks like today, captured live from `pull` (which runs the
same checks before any byte moves — this run declined the download, so nothing was
fetched beyond metadata):

```console
$ printf 'n\n' | stone-llama pull async0x42/Qwen3-1.7B-exl3_4.0bpw
gate: arch  ✓ Qwen3ForCausalLM
gate: quant ✓ exl3
gate: fit   ⚠ weights 1491 + KV 1120 + overhead 128 = 2739 MiB (headroom 1344, budget 2752)
      warning: only 13 MiB margin above the 1344 MiB headroom (prefill workspace [est] included): multi-KB prompts can OOM during prefill on a used card — if you see CUDA OOM before the first token, drop --ctx
warnings above — review them before continuing
pulling async0x42/Qwen3-1.7B-exl3_4.0bpw → Qwen3-1.7B-exl3_4.0bpw: 11 files, 1.47 GiB (sha256-verified), 83.06 GiB free
stone-llama pull: non-interactive pull requires --yes to confirm the download
```

An EXL3-shaped layout is required — GGUF repos are refused outright:

```console
$ stone-llama pull ggml-org/SmolLM3-3B-GGUF
stone-llama pull: repo ggml-org/SmolLM3-3B-GGUF has no config.json — not an EXL3 model layout
```

## 4. `pull` — download (consent-gated, resumable)

```bash
stone-llama pull async0x42/Qwen3-1.7B-exl3_4.0bpw
```

Gate first (above), then exact HEAD-measured size + free space, then one `y/N`.
Single-stream resumable download, sha256 verified before the file is renamed.
Already own the weights? Skip the download: `stone-llama import <dir> --name <n>`
symlinks them in, zero copy.

> Model downloads are omitted from this walkthrough's captures — bandwidth is
> constrained here and pulls are not authorized in this session.

## 5. `list` / `run`

```console
$ stone-llama list
NAME                    QUANT  SIZE      VERDICT  SOURCE
SmolLM3-3B-exl3_4.0bpw  -      1.84 GiB  -        imported
```

```bash
stone-llama run Qwen3-1.7B-exl3_4.0bpw        # streaming CLI chat
stone-llama run Qwen3-1.7B-exl3_4.0bpw --ctx 16384   # override autofit
```

`run` prints the autofit decision (ctx, cache mode, the arithmetic) before the
first token. `list --estimate` adds predicted decode/prefill tok/s columns.

## 6. `serve` + `curl` — OpenAI-compatible API

```bash
stone-llama serve                                  # loads the model, listens on 127.0.0.1:5111
stone-llama serve --attach 127.0.0.1:5002          # attach mode: proxy to a TabbyAPI you already run
```

Attach mode never restarts or stops the upstream server — it points stone-llama's
OpenAI-compatible endpoint at an existing TabbyAPI.

```bash
curl http://127.0.0.1:5111/v1/models

curl http://127.0.0.1:5111/v1/chat/completions -d '{
  "model": "Qwen3-1.7B-exl3_4.0bpw",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

Then `stone-llama ps` (loaded model + live VRAM) and `stone-llama stop`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| CUDA OOM **before** the first token | lower `--ctx` one tier (32768 → 16384) — prefill workspace, not generation |
| `gate: fit` refuses | GPU too small for this model; the printout includes the largest ctx that *would* fit |
| `exl2` / GGUF-only repo refused | wrong format for exllamav3; GGUF means use ollama with GGUF |
| pull 401/403 | gated repo → `stone-llama login` (or `HF_TOKEN`) |
| `setup` ENOSPC | free up disk; the error prints required vs available |
