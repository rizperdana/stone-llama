# stone-llama __VERSION__

__PRERELEASE_NOTICE__

A small static Go CLI + local OpenAI-compatible model server for ExLlamaV3
(wrapped via TabbyAPI), built for very low-spec NVIDIA PCs.

## Install

### Linux / macOS

```sh
curl -fsSL https://raw.githubusercontent.com/rizperdana/stone-llama/__VERSION__/scripts/install.sh | sh -s -- --version __VERSION__
```

Pin a different tag with `--version <tag>` (pre-releases such as `v0.1.0-rc1`
work when pinned). Choose the install directory with `--prefix <dir>` (or
`PREFIX=<dir>`); default is `~/.local/bin`. The installer verifies the
published SHA-256 checksum **before** extracting, and refuses to install an
unverified binary. Uninstall:

```sh
curl -fsSL https://raw.githubusercontent.com/rizperdana/stone-llama/__VERSION__/scripts/install.sh | sh -s -- --uninstall
```

### Windows

Download `stone-llama-windows-amd64.zip` from the assets below and extract it
manually — `install.sh` supports linux/darwin only. The Windows binary builds
but is **untested at runtime**.

## What's Changed

__CHANGELOG__

__FULL_CHANGELOG__

## Checksums (SHA-256)

Every published asset is listed in the combined `checksums.txt`; verify
manually with `sha256sum -c checksums.txt` (macOS: `shasum -a 256 -c checksums.txt`).

```
__CHECKSUMS__
```

## Platform support (honest matrix)

| Platform | Artifact | Status |
|----------|----------|--------|
| linux/amd64 | `stone-llama-linux-amd64.tgz` | ✅ Supported — smoke-tested in CI |
| linux/arm64 | `stone-llama-linux-arm64.tgz` | ⚠️ Builds, untested (no arm64 machine) |
| darwin/amd64 | `stone-llama-darwin-amd64.tgz` | ⚠️ Builds, untested — **can never serve** (no CUDA) |
| darwin/arm64 | `stone-llama-darwin-arm64.tgz` | ⚠️ Builds, untested — **can never serve** (no CUDA) |
| windows/amd64 | `stone-llama-windows-amd64.zip` | ⚠️ Builds, **untested at runtime** |

## Known limitations

- **NVIDIA-CUDA only.** ExLlamaV3 has no CPU, AMD/ROCm, or Apple Metal backend.
- **linux/amd64 is the only supported platform.** The other four builds are
  published for convenience and testing; do not rely on them. Per
  docs/RELEASE.md only `version`, `--help`, and `doctor` are known to work on
  any platform — `serve`/`run` require NVIDIA CUDA.
- macOS binaries can never serve — no CUDA.
- The Windows binary compiles and packages cleanly, but has never been executed
  on a Windows machine — treat it as untested.
- Verification of the release pipeline itself (install.sh end-to-end,
  checksums) is what pre-release tags like `v0.1.0-rc1` exist to prove.

See [docs/RELEASE.md](https://github.com/rizperdana/stone-llama/blob/main/docs/RELEASE.md)
for the release process and [docs/INSTALLATION.md](https://github.com/rizperdana/stone-llama/blob/main/docs/INSTALLATION.md)
for full install/uninstall instructions.
