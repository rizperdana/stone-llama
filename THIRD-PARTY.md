# Third-party components

stone-llama **redistributes none of the components below**. Its `setup`
command downloads upstream components onto the user's machine at the user's
request — only after announcing file sizes and requiring explicit
confirmation. Downloads are verified in transit: SHA-256 for the uv binary,
`--require-hashes` for every Python package from the pinned lock file.

Every license here has been verified from a real source (local license file or
package metadata, or authoritative upstream repository). The A6/M7 gate is
**cleared**.

## How `setup` fetches things

`internal/setup/runtime.lock.json` pins the exact versions and SHA-256s:

1. **uv** — `setup` downloads a release-binary tarball from
   `https://github.com/astral-sh/uv/releases/download/0.11.6/` and verifies
   the SHA-256 against `runtime.lock.json`; only the `uv` binary is extracted.
2. **CPython 3.12** — `uv python install 3.12` fetches a python-build-standalone
   interpreter. The spec (`"3.12"`) is pinned in `runtime.lock.json`; uv
   selects the concrete patch version.
3. **TabbyAPI** — `git clone` + `git checkout` of pinned commit `f07131cd`
   from [`theroyallab/tabbyAPI`](https://github.com/theroyallab/tabbyAPI).
4. **Python dependencies** — `uv pip install --require-hashes -r
   requirements-cu{12,13}.lock` into a fresh venv.

## AGPL note (TabbyAPI)

We neither distribute nor modify TabbyAPI. `setup` downloads upstream sources
to the user's disk at the user's request; stone-llama communicates with it as
a separate process over HTTP (no shared memory, no vendored code). **Vendoring
or patching TabbyAPI is a deliberate non-goal of this project — it would
change the license analysis and could trigger AGPL-3.0 source-disclosure
obligations.**

## Verified

| Component | How it arrives | License | Evidence |
|---|---|---|---|
| uv | `setup` downloads release binary tarball, SHA-256 verified | **Apache-2.0 OR MIT** (dual-licensed) | Pinned as v0.11.6 in `internal/setup/runtime.lock.json` (URL `https://github.com/astral-sh/uv/releases/download/0.11.6/uv-x86_64-unknown-linux-gnu.tar.gz`). Upstream `LICENSE-APACHE`: `https://raw.githubusercontent.com/astral-sh/uv/main/LICENSE-APACHE` — full Apache-2.0 text. Upstream `LICENSE-MIT`: `https://raw.githubusercontent.com/astral-sh/uv/main/LICENSE-MIT` — full MIT text. |
| CPython 3.12 | `uv python install` (python-build-standalone) | **PSF-2.0** | Locally-provisioned interpreter at `/home/anon/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/LICENSE.txt` — states "Python software and documentation are licensed under the Python Software Foundation License Version 2." Matches upstream `https://raw.githubusercontent.com/python/cpython/v3.12.0/LICENSE`. |
| TabbyAPI | `setup`: `git clone` + pinned checkout | **AGPL-3.0** | Local TabbyAPI checkout license file `/home/anon/ai/tabbyapi/tabbyAPI/LICENSE` — full text of "GNU Affero General Public License Version 3, 19 November 2007." Pinned commit `f07131cd8fe34e449fe87cdd3a066b52b96d3cac` in `runtime.lock.json`. |
| PyTorch 2.11.0+cu130 | `uv pip install -r requirements-cu13.lock` | **BSD-3-Clause** | Installed `torch-2.11.0+cu130.dist-info/METADATA`, `License: BSD-3-Clause`. Also ships `LICENSE` and `NOTICE` files per `License-File:` metadata field. |
| exllamav3 1.5.1+cu132 | `uv pip install -r requirements-cu13.lock` | **MIT** | Installed `exllamav3-1.5.1+cu132.torch2.11.0.dist-info/METADATA`, `License-Expression: MIT`, `License-File: LICENSE`. |
| Triton 3.6.0 | `uv pip install -r requirements-cu13.lock` | **MIT** | Installed `triton-3.6.0.dist-info/METADATA`, `Classifier: License :: OSI Approved :: MIT License`, `License-File: LICENSE`. |
| Flash-linear-attention 0.5.2 | `uv pip install -r requirements-cu13.lock` | **MIT** | Installed `flash_linear_attention-0.5.2.dist-info/METADATA`, `License: MIT License`. (Backend package `fla-core 0.5.2` also MIT.) |

### CUDA / cuDNN runtime packages (transitive deps of PyTorch +cu130)

These **are not** listed directly in the `cu13` extra — they are pulled in
transitively by the `torch` 2.11.0+cu130 wheel (via the `cuda-toolkit[...]`
extra and direct `Requires-Dist` lines in torch's METADATA). stone-llama
fetches them at the user's request through `uv pip install --require-hashes -r
requirements-cu13.lock`. Versions and SHA-256 hashes are pinned in
`internal/setup/requirements-cu13.lock`.

All NVIDIA packages below ship the **NVIDIA Software License Agreement / End
User License Agreement** (proprietary), confirmed from both the `License:`
field in each package's `METADATA` and the bundled `License.txt` file where
present.

> **Caveat:** NVIDIA's CUDA runtime and cuDNN are proprietary; their terms are
> governed by NVIDIA's EULA, not an open-source license. stone-llama does not
> redistribute, modify, or relicense them. The user must accept NVIDIA's terms
> directly. Downloading requires explicit consent at `setup` time with
> announced sizes. If NVIDIA's redistribution policy changes, this section must
> be re-reviewed.

| Package | Version | License (METADATA) | Evidence (local files in venv) |
|---|---|---|---|
| nvidia-cublas | 13.1.0.3 | `LicenseRef-NVIDIA-Proprietary` | `nvidia_cublas-13.1.0.3.dist-info/METADATA` `License:` + `licenses/License.txt` (NVIDIA EULA) |
| nvidia-cuda-cupti | 13.0.85 | `LicenseRef-NVIDIA-Proprietary` | `nvidia_cuda_cupti-13.0.85.dist-info/METADATA` `License:` |
| nvidia-cuda-nvrtc | 13.0.88 | `LicenseRef-NVIDIA-Proprietary` | `nvidia_cuda_nvrtc-13.0.88.dist-info/METADATA` `License:` + `licenses/License.txt` (NVIDIA EULA) |
| nvidia-cuda-runtime | 13.0.96 | `LicenseRef-NVIDIA-Proprietary` | `nvidia_cuda_runtime-13.0.96.dist-info/METADATA` `License:` + `licenses/License.txt` (NVIDIA EULA) |
| nvidia-cudnn-cu13 | 9.19.0.56 | `LicenseRef-NVIDIA-Proprietary` | `nvidia_cudnn_cu13-9.19.0.56.dist-info/METADATA` `License:` + `licenses/License.txt` (NVIDIA Software License Agreement + cuDNN Supplement) |
| nvidia-cufft | 12.0.0.61 | `LicenseRef-NVIDIA-Proprietary` | `nvidia_cufft-12.0.0.61.dist-info/METADATA` `License:` |
| nvidia-cufile | 1.15.1.6 | `LicenseRef-NVIDIA-Proprietary` | `nvidia_cufile-1.15.1.6.dist-info/METADATA` `License:` |
| nvidia-curand | 10.4.0.35 | `LicenseRef-NVIDIA-Proprietary` | `nvidia_curand-10.4.0.35.dist-info/METADATA` `License:` |
| nvidia-cusolver | 12.0.4.66 | `LicenseRef-NVIDIA-Proprietary` | `nvidia_cusolver-12.0.4.66.dist-info/METADATA` `License:` |
| nvidia-cusparse | 12.6.3.3 | `LicenseRef-NVIDIA-Proprietary` | `nvidia_cusparse-12.6.3.3.dist-info/METADATA` `License:` |
| nvidia-cusparselt-cu13 | 0.8.0 | `NVIDIA Proprietary Software` | `nvidia_cusparselt_cu13-0.8.0.dist-info/METADATA` `License:` (no bundled license file) |
| nvidia-nccl-cu13 | 2.28.9 | `LicenseRef-NVIDIA-Proprietary` | `nvidia_nccl_cu13-2.28.9.dist-info/METADATA` `License:` + `licenses/` |
| nvidia-nvjitlink | 13.0.88 | `LicenseRef-NVIDIA-Proprietary` | `nvidia_nvjitlink-13.0.88.dist-info/METADATA` `License:` |
| nvidia-nvshmem-cu13 | 3.4.5 | `LicenseRef-NVIDIA-Proprietary` | `nvidia_nvshmem_cu13-3.4.5.dist-info/METADATA` `License:` + `licenses/License.txt` (NVIDIA EULA) |
| nvidia-nvtx | 13.0.85 | **Apache-2.0** | `nvidia_nvtx-13.0.85.dist-info/METADATA` `License: Apache 2.0` + `licenses/License.txt` (Apache 2.0 with LLVM exceptions) |

Plus CUDA toolkit meta/helper packages (also transitive deps of torch):

| Package | Version | License | Evidence |
|---|---|---|---|
| cuda-toolkit | 13.0.2 | (no explicit license — meta-package) | `cuda_toolkit-13.0.2.dist-info/METADATA` has no `License:` field; bundles the `nvidia-*` packages whose licenses are listed above |
| cuda-bindings | 13.4.3 | **Apache-2.0** | `cuda_bindings-13.4.3.dist-info/METADATA` `License-Expression: Apache-2.0` + `licenses/LICENSE` |
| cuda-pathfinder | 1.8.2 | **Apache-2.0** | `cuda_pathfinder-1.8.2.dist-info/METADATA` `License-Expression: Apache-2.0` + `licenses/LICENSE` |

**cu12 variant:** the `cu12` extra pins `torch 2.9.0+cu128` and `triton 3.5.0`,
and pulls an analogous set of NVIDIA packages with a `cu12` suffix
(e.g. `nvidia-cudnn-cu12`, `nvidia-cublas-cu12`, `nvidia-cufft-cu12`, etc. —
same proprietary EULA terms). Pinned versions and hashes are in
`internal/setup/requirements-cu12.lock`.

## Still open

Nothing. All three previously-unverified items — **uv** (Apache-2.0 OR MIT),
**CPython 3.12** (PSF-2.0), and **NVIDIA CUDA/cuDNN wheels** (proprietary NVIDIA
EULA, exact package set enumerated above) — have been verified from real
sources. The A6/M7 gate is cleared.

## README alignment

The README's third-party table (README.md lines 279-288) was expanded in
`bbfbc16` to list all eight setup-fetched component families, matching the
enumeration in this document.
