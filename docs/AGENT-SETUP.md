# Agent setup — copy-paste prompt

Give the block below verbatim to a coding agent (Claude Code, Codex, etc.) running
on the target machine. It is self-contained: no other context from this repo is
needed.

---

```text
Set up the stone-llama CLI on this machine. stone-llama is a standalone,
ollama-like CLI + local model server for ExLlamaV3 + TabbyAPI. Its only external
interface is an OpenAI-compatible HTTP endpoint; it does not integrate with any
other tool.

Repo: https://github.com/rizperdana/stone-llama (Go 1.24, MIT)

## Step 0 — environment checks (run these first, in order)

1. GPU: run `nvidia-smi`.
   - No nvidia-smi / no NVIDIA GPU → STOP. stone-llama cannot work here:
     exllamav3 is NVIDIA-CUDA-only (no CPU, no AMD/ROCm, no Apple/Metal).
     Report that the user should use ollama + GGUF instead. Do not try to
     work around it.
2. Driver: `nvidia-smi --query-gpu=driver_version --format=csv,noheader`.
   - >= 580  → runtime extra is `cu13`
   - >= 570  → runtime extra is `cu12`
   - <  570  → STOP: report "upgrade NVIDIA driver to >= 570 (cu12) or >= 580 (cu13)".
3. Platform: `uname -m` must be `x86_64`. v1 supports Linux/amd64 only.
   (A Windows binary may exist but is untested at runtime; macOS can run
   doctor/list/fit but can never serve — no CUDA.)
4. Disk: enough for ≈ 1 GB of runtime downloads plus your models (`df -h ~`).
   `setup` measures free space itself and refuses if short, so this check is
   advisory — but a nearly-full disk will fail Step 4.
5. Go toolchain: `go version` must be >= 1.24 (needed only to build from source).

## Step 1 — build the binary

    git clone https://github.com/rizperdana/stone-llama
    cd stone-llama
    go build -o stone-llama ./cmd/stone-llama
    install -Dm755 stone-llama ~/.local/bin/stone-llama

(If a release tarball + install.sh exist on the GitHub Releases page, that path
is fine too — verify the published sha256 before running anything.)

## Step 2 — doctor

    stone-llama doctor

Expected shape of output (numbers are this machine's, yours will differ):

    GPU        <name>
    VRAM       <MiB>
    Driver     <version>
    Runtime    cu13 extra

- doctor must NOT report a missing GPU or an old driver. If it does, STOP and
  report the exact message.
- It prints no downloads and makes no changes.

## Step 3 — fit check (NO download, metadata only)

    stone-llama fit async0x42/Qwen3-1.7B-exl3_4.0bpw

This fetches only HuggingFace metadata (KB) and prints whether the model fits
this GPU, the chosen (ctx, cache_mode), and an estimated tok/s.
- exit 0 = fits or fits-with-warning; exit 3 = refused (GPU too small) — report
  the printed breakdown, do not force it.

## Step 4 — provision the runtime (MULTI-GB, requires user consent)

    stone-llama setup

CRITICAL: this step downloads about a gigabyte (measured here: PyTorch+cu130
531 MB, exllamav3 419 MB, uv 24 MB; CPython + the TabbyAPI checkout are extra).
It must announce sizes and ask for confirmation before any byte moves. You MUST
surface the size announcement to the user and get explicit approval before
answering yes. Never pass --yes on the user's behalf without telling them the
size. It is resumable (step journal) if interrupted; full log at
~/.local/share/stone-llama/logs/setup.log.

Do NOT use pip/uv/ollama directly to install anything — `setup` is the only
sanctioned provisioning path.

## Step 5 — pull a model (download, get consent)

    stone-llama pull async0x42/Qwen3-1.7B-exl3_4.0bpw

The pre-download gate runs first (architecture / quant format / fit from
metadata) and then shows an exact HEAD-measured download size + free disk space
before asking. Again: show the user the size and get consent before confirming.
If you already have an EXL3 model directory on disk, `stone-llama import <dir>
--name <name>` links it in with zero copy — prefer this over downloading.

## Step 6 — serve + verify

    stone-llama serve            # OpenAI-compatible API on 127.0.0.1:5111
    # (attach mode, if a TabbyAPI already runs on this machine — never restart
    #  or stop an existing TabbyAPI, it may serve live work):
    stone-llama serve --attach 127.0.0.1:5002

Verify with:

    curl -s http://127.0.0.1:5111/v1/models
    curl -s http://127.0.0.1:5111/v1/chat/completions -d '{
      "model": "<model-name>",
      "messages": [{"role": "user", "content": "hello"}]
    }'

Then `stone-llama ps` (loaded model + live VRAM) and `stone-llama stop`.

If `serve`/`ps`/`stop`/`run` print "not implemented yet (ships in M4/M5/M6)",
they are not in the build you have — report that honestly as an unfinished
step; do not improvise an alternative server. `doctor`, `fit`, `list`,
`list --estimate`, `rank`, `import`, `rm`, `pull`, `login` are implemented.

## Decision points (recap)

- driver >= 580 → cu13 extra; 570–579 → cu12 extra; < 570 → refuse, upgrade.
- gate verdict "refuse" → do not pull; report numbers to the user.
- gate verdict "warn" (thin margin) → proceed but tell the user: if prompts OOM
  during prefill (CUDA OOM before the first token), lower --ctx one tier
  (32768 → 16384). On a 4 GB card, maxed-out max_seq_len can OOM prefill at
  ~2.5k prompt tokens with ~20 MiB free VRAM — safe context is the ladder
  default; explicit --ctx buys max context and prints the cost.

## Failure modes to watch for

- `no nvidia-smi` / driver < 570 → hard stop, see Step 0.
- `fit` exit 3 (refuse) → model too big, report the printed breakdown.
- Gate refusing `exl2` quant or a GGUF-only repo → expected; GGUF means
  "use ollama with GGUF".
- `setup` ENOSPC → report the printed numbers, do not retry blindly.
- Prefill CUDA OOM (before the first token) → lower --ctx, not max_tokens.
- 401/403 on pull → gated HuggingFace repo; `stone-llama login`, never put the
  token on argv or in a log.
- A model repo that 404s on weights → Hub churn; try `fit` on another repo or
  `import` a local dir.

## Verify-it-worked checklist

- [ ] `stone-llama version` prints a version and `linux/amd64`.
- [ ] `stone-llama doctor` shows GPU, VRAM, driver, and a runtime extra (no refusal).
- [ ] `stone-llama fit <repo>` exits 0 and prints the autofit decision + [est] tok/s.
- [ ] `stone-llama setup` completed all steps (log shows no failure).
- [ ] `stone-llama list` shows the pulled/imported model with a verdict.
- [ ] `stone-llama serve` is up; `curl /v1/models` returns the model list.
- [ ] `curl /v1/chat/completions` returns a completion (or streams tokens).
- [ ] `stone-llama ps` shows the loaded model + live VRAM; `stone-llama stop` stops it.

## Hard constraints while doing this

- Never restart/stop/kill an already-running TabbyAPI or any unrelated service
  on this machine.
- No downloads beyond: stone-llama's own build, `setup` runtime (with consent),
  and model pulls (with consent). No package installs via apt/pip/uv outside
  what `setup` does.
- Only run commands from this list; do not experiment on the user's GPU with
  other inference stacks.

```

---

## What each step checks

| Step | Proves | Cost |
|---|---|---|
| 0 | hardware/platform reality | none |
| 1 | binary builds and installs | none (Go build only) |
| 2 | GPU/driver verdict | none, no downloads |
| 3 | model fits this GPU + speed estimate | KB of HF metadata |
| 4 | Python inference runtime | ≈ 1 GB, consent-gated |
| 5 | model on disk | HEAD-measured size, consent-gated |
| 6 | OpenAI endpoint answers | none |
