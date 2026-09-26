# Installation

v1 supports **Linux/amd64 only**. Read the platform table in the
[README](../README.md#platform-reality--nvidiacuda-only-read-this-first) first —
exllamav3 is NVIDIA-CUDA-only, so on CPU/AMD/Apple machines nothing below will
ever serve.

## Prerequisites

| Requirement | Detail |
|---|---|
| GPU | NVIDIA with CUDA. No CPU path, no AMD/ROCm, no Apple/Metal. |
| Driver | ≥ **570** (installs the `cu12` runtime extra) or ≥ **580** (`cu13`). Older → `doctor` refuses with the upgrade hint. |
| OS/arch | Linux x86_64 (amd64). |
| Disk (binary alone) | ~10–15 MB. |
| Disk (`setup` runtime) | **≈ 1 GB of downloads** — measured here: torch+cu130 531 MB, exllamav3 419 MB, uv 24 MB (950 MB total, HEAD-announced per URL before consent), plus a pinned CPython and the TabbyAPI checkout; the unpacked venv takes more than the downloads. Free space is checked with `statfs` and the run refuses if short. |
| Disk (models) | Whatever the weights are — e.g. `SmolLM3-3B-exl3` is 1.84 GiB. Checked per-pull before any byte moves. |
| Network | Only for `setup` (runtime) and `pull` (weights). `doctor`, `list`, `fit`, `import` need none (aside from small HuggingFace metadata for `fit`). |

`stone-llama doctor` prints all of this for *your* machine — GPU name, VRAM,
driver, chosen runtime extra — and makes no changes.

## Option A — release binary / install.sh

Releases are tagged `vX.Y.Z` and publish, per platform:

| Artifact | Note |
|---|---|
| `stone-llama_X.Y.Z_linux_amd64.tar.gz` | **the supported target** |
| `stone-llama_X.Y.Z_windows_amd64.zip` | published, **untested at runtime** (daemonization differs) |
| `stone-llama_X.Y.Z_darwin_amd64.tar.gz` / `_darwin_arm64.tar.gz` | builds; can run `doctor`/`list`/`fit`, **can never serve** (no CUDA) |

Each has a `.sha256` sidecar (plus a combined `checksums.txt`).

One-line install (resolves the latest release, verifies sha256, installs to
`~/.local/bin`):

```bash
curl -fsSL https://raw.githubusercontent.com/rizperdana/stone-llama/main/scripts/install.sh | sh
```

Or from a checkout: `scripts/install.sh --version 0.1.0` pins a tag.

## Option B — build from source (always works)

```bash
git clone https://github.com/rizperdana/stone-llama
cd stone-llama
go build -o stone-llama ./cmd/stone-llama   # Go ≥ 1.24
install -Dm755 stone-llama ~/.local/bin/stone-llama
```

The binary is static; `doctor`, `list`, `import`, `rm`, and `fit` work with no
runtime installed. Go is only needed to build, not to run.

Verify:

```bash
stone-llama version
stone-llama doctor
```

Expected `doctor` output on a supported machine (RTX 3050 Laptop, driver 580):

```
GPU        NVIDIA GeForce RTX 3050 Laptop GPU
VRAM       4096 MiB
Driver     580.178.04
Runtime    cu13 extra
```

## The `setup` provisioning step (separate from install)

The binary does not bundle Python. `stone-llama setup` provisions the inference
runtime into `~/.local/share/stone-llama/runtime/`:

1. **uv** — pinned version, URL + SHA-256 verified.
2. **CPython 3.12** — installed via uv (not system Python; reproducibility).
3. **TabbyAPI** — upstream git checkout at a pinned commit (`f07131c`),
   content-addressed, never forked.
4. **venv + pinned wheels** — PyTorch (`cu12`/`cu13` extra chosen from your
   driver by `doctor`), exllamav3, CUDA runtime wheels, from hash-locked
   requirements files.

Why it's large: PyTorch + exllamav3 are the bulk of it (950 MB announced on this
machine) — that is the inference stack, not bloat. There is no small variant.

Safety properties:

- **Nothing downloads without consent.** Preflight prints per-URL sizes
  (HTTP HEAD), free disk space, destination paths, then asks once.
- **Resumable.** A step journal skips completed steps after an interruption and
  re-announces only the remaining sizes.
- **Logged.** Full output in `~/.local/share/stone-llama/logs/setup.log`.
- **Licenses.** We redistribute none of it — `setup` fetches each component from
  its upstream source on your disk, on your confirmation. TabbyAPI is
  **AGPL-3.0**; see [THIRD-PARTY.md](../THIRD-PARTY.md).

```bash
stone-llama setup          # shows sizes + free space, then one confirmation
stone-llama setup --yes    # non-interactive consent
```

Real preflight on this machine (sizes are HTTP HEAD requests; without consent
nothing is downloaded — the run aborts at the prompt):

```console
$ stone-llama setup
setup plan
  dest:  /home/anon/.local/share/stone-llama/runtime
  free:  88926429184 bytes
  downloads (3):
    https://github.com/astral-sh/uv/releases/download/0.11.6/uv-x86_64-unknown-linux-gnu.tar.gz (24284812 bytes)
    https://download-r2.pytorch.org/whl/cu130/torch-2.11.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl (531146695 bytes)
    https://github.com/turboderp-org/exllamav3/releases/download/v1.5.1/exllamav3-1.5.1%2Bcu132.torch2.11.0-cp312-cp312-linux_x86_64.whl (419209397 bytes)
  steps (5):
    1. uv — download and verify uv 0.11.6 (24284812 bytes)
    2. python — install Python 3.12 via uv (0 bytes)
    3. tabby — clone TabbyAPI @ f07131cd8fe3 (0 bytes)
    4. venv-deps — create venv and install requirements-cu13.lock (950356092 bytes)
    5. smoke — verify exllamav3 import (0 bytes)
Proceed? [y/N] stone-llama setup: setup: aborted (plan not confirmed)
```

Raw capture: [screenshots/setup-preflight.txt](screenshots/setup-preflight.txt).

## Post-install verification

```bash
stone-llama version        # version + GOOS/GOARCH
stone-llama doctor         # GPU, driver, runtime-extra verdict
stone-llama fit async0x42/Qwen3-1.7B-exl3_4.0bpw   # metadata only — no download
stone-llama list           # should show installed models (or the empty hint)
```

If `doctor` reports the runtime as missing, run `setup` before `serve`/`run`.

## Uninstall

```bash
rm -f  ~/.local/bin/stone-llama            # the binary
rm -rf ~/.local/share/stone-llama          # models, runtime, logs, manifests
rm -rf ~/.config/stone-llama               # config (only if you want it gone)
```

`stone-llama rm <model>` removes a single model without touching the rest.
Imported models are symlinks — `rm` deletes the link, never the target files.
