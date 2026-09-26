# Quickstart

First run on a supported machine (Linux/amd64, NVIDIA GPU, driver ≥ 570).
Every capture here is real output from this machine (RTX 3050 Laptop, 4096 MiB,
driver 580.178.04) — raw files in [screenshots/](screenshots/).

## 1. `doctor` — environment check, no downloads, no changes

Expected output on this machine (RTX 3050 Laptop, driver 580):
[screenshots/doctor-list.txt](screenshots/doctor-list.txt).

Driver decides the runtime extra: ≥ 580 → `cu13`, ≥ 570 → `cu12`, older → refusal
with the upgrade hint. No NVIDIA GPU → hard refusal pointing you at ollama + GGUF.

## 2. `setup` — provision the inference runtime (≈ 1 GB download, consent-gated)

```bash
stone-llama setup
```

Downloads into `~/.local/share/stone-llama/runtime/`: pinned `uv` → CPython 3.12 →
TabbyAPI at a pinned commit → hash-locked wheels. Measured on this machine: torch
+cu130 531 MB + exllamav3 419 MB + uv 24 MB = 950 MB announced (HTTP HEAD), free
disk checked, then **one** confirmation. Resumable via a step journal; log at
`logs/setup.log`. Details: [INSTALLATION.md](INSTALLATION.md#the-setup-provisioning-step-separate-from-install).

## 3. `fit <repo>` — will it run, how fast? **No download.**

Metadata only (KB from HuggingFace): architecture / quant-format gate plus the
autofit arithmetic against *your* VRAM, then a predicted tok/s **[est]**:

```console
$ stone-llama fit async0x42/Qwen3-1.7B-exl3_4.0bpw
gate: arch  ✓ Qwen3ForCausalLM
gate: quant ✓ exl3
gate: fit   ⚠ weights 1491 + KV 1120 + overhead 128 = 2739 MiB (headroom 1344, budget 2752)
      warning: only 13 MiB margin above the 1344 MiB headroom (prefill workspace [est] included): multi-KB prompts can OOM during prefill on a used card — if you see CUDA OOM before the first token, drop --ctx
estimate: ~31 tok/s decode, ~582 tok/s prefill [est] at Q4 ctx 40960 (NVIDIA GeForce RTX 3050 Laptop GPU) — low confidence, anchored to the measured 42.7 tok/s SmolLM3-3B 3.5bpw point (1866 MiB) on this GPU; calibration pending

$ stone-llama fit async0x42/Qwen3-8B-exl3_4.0bpw
gate: arch  ✓ Qwen3ForCausalLM
gate: quant ✓ exl3
gate: fit   ✗ no config fits: weights 4950 + KV (Q4 @ 4096) 144 + overhead 128 = 5222 MiB > 3040 MiB budget (4096 MiB VRAM − 1056 headroom: 512 base + 512 prefill workspace [est] + 32 ctx margin [est])
      largest ctx that would fit: none — no context fits; pull a smaller quant or use a bigger GPU
      reduce --ctx or pick a smaller quant
fit: refused — no context fits this model in VRAM (see the gate report above)
EXIT=3
```

exit 0 = fits / fits-with-warning, exit 3 = refused with the full breakdown.

The same gate runs inside `pull` before any byte moves — this run declined the
download, so nothing was fetched beyond gate metadata (KB of config plus the
safetensors-header format check, ≤ ~16 MiB worst case) — and an EXL3-shaped
layout is required: a GGUF repo is refused with "not an EXL3 model layout".
Both runs: [screenshots/gate.txt](screenshots/gate.txt).

## 4. `pull` — download (consent-gated, resumable)

```bash
stone-llama pull async0x42/Qwen3-1.7B-exl3_4.0bpw
```

Gate first (above), then exact HEAD-measured size + free space, then one `y/N`.
Single-stream resumable download, sha256 verified before the file is renamed.
Already own the weights? Skip the download: `stone-llama import <dir> --name <n>`
symlinks them in, zero copy.

> Model downloads are omitted from this walkthrough's captures — no weights were
> pulled while recording it.

## 5. `list` / `run`

`stone-llama list` shows the imported model with size, quant, verdict and source
([screenshots/doctor-list.txt](screenshots/doctor-list.txt)); `list --estimate`
adds predicted decode/prefill tok/s columns
([screenshots/list-estimate.txt](screenshots/list-estimate.txt)).

```bash
stone-llama run Qwen3-1.7B-exl3_4.0bpw        # streaming CLI chat
stone-llama run Qwen3-1.7B-exl3_4.0bpw --ctx 16384   # override autofit
```

`run` prints the autofit decision (ctx, cache mode, the arithmetic) before the
first token.

In attach mode there is nothing for `run` to load, so no autofit line is
printed. Live capture (source build of `main` at 4f020dc, 2026-09-26, attach
mode against a live TabbyAPI; full transcript and caveats:
[screenshots/run.txt](screenshots/run.txt)):

```console
$ stone-llama ps
stone-llama: running (pid 458259)
  address:  127.0.0.1:5111
  mode:     attach (http://127.0.0.1:5002)
            stone-llama does not own this process — it proxies only; load/unload is the upstream's
  ready:    yes
  model:    SmolLM3-3B-exl3
  uptime:   22s
exit=0

$ printf '/bye\n' | stone-llama run SmolLM3-3B-exl3
>>> exit=0

$ ls ~/.local/share/stone-llama/   # state intact after run
daemon.json
logs
models
runtime
stone-llama.lock
```

`run` auto-tunes one profile line in the REPL (nothing extra in `-p` output):

```console
$ printf '/bye\n' | stone-llama run SmolLM3-3B-exl3
profile: max_tokens 2048, temperature 0.6 (generation_config.json), top_p 0.95 (generation_config.json), thinking model default, system "Answer directly and concisely."
>>>
```

What is auto-tuned, and how to turn each knob:

- **Chat dialect** — `run` posts to `/v1/chat/completions`, so the model's own chat
  template and generation prompt apply, and replies stop at EOS on their own.
- **Bound** — `--max-tokens` (default 2048); hitting it prints
  `stone-llama: output truncated at --max-tokens N (use --max-tokens 0 for unbounded)`.
  `--max-tokens 0` opts back into generate-to-EOS with no cap.
- **Sampling** — `temperature`/`top_p` come from the model's own
  `generation_config.json` in the models dir (pulled models ship it); when it is
  absent, the backend's fallback defaults apply. `--temperature F` / `--top-p F`
  always win.
- **System message** — default `Answer directly and concisely.`. A default system
  prompt does influence model behaviour — that is the intent, so it is documented
  here, replaced with `--system TEXT`, and removed with `--no-system`.
- **Thinking** — follows the model template's own default (SmolLM3's TabbyAPI
  template defaults to off); `--thinking` / `--no-thinking` override it per run.

In attach mode the profile line reports `sampling backend defaults` unless the model
also sits in your own models dir — the upstream's model directory is not exposed
over the API.

Two caveats:

- Loading is the upstream's job in attach mode: `/-/load` answers 409
  `attach_mode`, and a mismatched model name exits 1 with the same message.

## 6. `serve` + `curl` — OpenAI-compatible API

```bash
stone-llama serve                                  # loads the model, listens on 127.0.0.1:5111
stone-llama serve --attach 127.0.0.1:5002          # attach mode: proxy to a TabbyAPI you already run
```

Attach mode never restarts or stops the upstream server — it points stone-llama's
OpenAI-compatible endpoint at an existing TabbyAPI.

`serve`, `ps`, `stop` and `run` are all implemented in the current build. Live
attach-mode capture against a running TabbyAPI (source build of `main` at
4f020dc, 2026-09-26; the API key was passed with `--key-file`, never argv;
full transcript, headers and evidence:
[screenshots/serve.txt](screenshots/serve.txt)):

```console
$ stone-llama serve --attach 127.0.0.1:5002 --key-file <key-file>
stone-llama listening on 127.0.0.1:5111 (OpenAI-compatible)
  upstream: http://127.0.0.1:5002 (attach)

$ curl -sS -D raw/healthz-headers.txt -o raw/healthz-body.txt http://127.0.0.1:5111/healthz
HTTP=200

HTTP/1.1 200 OK
X-Stone-Llama: 1
Date: Sat, 26 Sep 2026 05:16:22 GMT
Content-Length: 12
Content-Type: text/plain; charset=utf-8

stone-llama

$ stone-llama ps
stone-llama: running (pid 419204)
  address:  127.0.0.1:5111
  mode:     attach (http://127.0.0.1:5002)
            stone-llama does not own this process — it proxies only; load/unload is the upstream's
  ready:    yes
  model:    SmolLM3-3B-exl3
  uptime:   3m45s

$ stone-llama stop
stone-llama: stopped
exit=0
```

`X-Stone-Llama: 1` is the daemon's own marker on **`/healthz`**; the literal
`/health` path is an upstream passthrough (`Server: uvicorn`) and carries no
marker header. Measured on that same daemon: a non-streaming completion
returned 200 in 0.344 s, and a streaming request had TTFB 0.0105 s against a
total of 1.4166 s across 56 SSE events — per-chunk passthrough, not buffering.
(The measured autofit numbers in this doc were obtained in attach mode against
a live TabbyAPI — zero downloads.)

```bash
curl http://127.0.0.1:5111/v1/models

curl http://127.0.0.1:5111/v1/chat/completions -d '{
  "model": "Qwen3-1.7B-exl3_4.0bpw",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

Then `stone-llama ps` (loaded model + live VRAM) and `stone-llama stop`.

With no daemon the two commands differ in exit code — scripts will care. Both
lines below are observed at revision 4f020dc, where `ps` finds the daemon only
through `daemon.json` and auto-starts one by default (that behaviour is being
reworked upstream; do not read the auto-start as the intended long-term
design):

```console
$ STONE_LLAMA_NO_AUTOSTART=1 stone-llama ps
stone-llama ps: no stone-llama daemon is running (start one with 'stone-llama serve')
exit=1
$ stone-llama stop
stone-llama: not running
exit=0
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| CUDA OOM **before** the first token | lower `--ctx` one tier (32768 → 16384) — prefill workspace, not generation |
| `gate: fit` refuses | GPU too small for this model; the printout includes the largest ctx that *would* fit |
| `exl2` / GGUF-only repo refused | wrong format for exllamav3; GGUF means use ollama with GGUF |
| pull 401/403 | gated repo → `stone-llama login` (or `HF_TOKEN`) |
| `setup` ENOSPC | free up disk; the error prints required vs available |
| Testing without touching the live daemon | point `XDG_DATA_HOME` at an empty directory (separate state dir) and `XDG_CONFIG_HOME` / `STONE_LLAMA_CONFIG` at a scratch config — see [README](../README.md#commands) |
