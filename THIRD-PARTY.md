# Third-party components

stone-llama **redistributes none of the components below**. Its `setup`
command fetches them from their upstream sources onto the user's machine,
only after explicit confirmation with announced sizes. Licenses listed here
were verified from local artifacts (installed package metadata / license
files), not from memory.

## Verified

| Component | Fetched by | License | Evidence |
|---|---|---|---|
| TabbyAPI (turboderp) | `setup` (git tarball, pinned commit `f07131c`) | **AGPL-3.0** | `LICENSE` in upstream checkout |
| exllamav3 1.5.1 | `setup` (GitHub release wheel URL) | **MIT** | installed `exllamav3-*.dist-info/METADATA`, `License-Expression: MIT` |
| PyTorch 2.11.0+cu130 | `setup` (PyTorch wheel URL) | **BSD-3-Clause** | installed `torch-*.dist-info/METADATA` |

### AGPL note (TabbyAPI)

We neither distribute nor modify TabbyAPI. `setup` downloads upstream
sources to the user's disk at the user's request; stone-llama communicates
with it as a separate process over HTTP. Vendoring or patching TabbyAPI
would change this analysis — that is a deliberate non-goal of this project.

## UNVERIFIED — must be confirmed before any public release (A6/M7 gate)

Do not publish license claims for these until verified from upstream:

- `uv` (fetched by `setup`; license file not present locally — confirm from
  upstream repository when pinning version + URL + SHA-256).
- CPython 3.12 via `uv python install` (expect PSF-2.0 — confirm from the
  installed LICENSE at provision time).
- NVIDIA CUDA runtime wheels / cuDNN wheels pulled by the `cu12`/`cu13`
  extra (NVIDIA EULA — enumerate the exact package set from the generated
  `requirements.lock`).
