# Release Runbook

## Overview

`stone-llama` publishes static Go binaries via GitHub Actions on every version
tag (`v*`).  The release pipeline lives in `.github/workflows/release.yml` and
the CI gate lives in `.github/workflows/ci.yml`.

## Platform Matrix — Read This Before Claiming Anything Works

| Platform     | Status                                           |
|--------------|--------------------------------------------------|
| linux/amd64  | **Supported** — requires NVIDIA GPU + CUDA at runtime (driver ≥ 570 for cu12, ≥ 580 for cu13). |
| windows/amd64| Published but **UNTESTED at runtime** — the daemonization path differs from Linux and we have no Windows test machine. Use at your own risk. |
| darwin/amd64 | Published for `doctor`/`list`/`fit` only. ExLlamaV3 has no Metal/CPU backend, so stone-llama **cannot serve** models on macOS. Compilation is a build-check, not support. |
| darwin/arm64 | Same as darwin/amd64 — CLI utilities only, no serving. |

> **Never claim a platform works because it compiles.** Compilation verifies the code builds for that target. Runtime support is a separate question documented above.

## How to Cut a Tag

```bash
# 1. Ensure main is clean and CI passes
git checkout main
git pull
make test   # go vet + go test ./...

# 2. Create an annotated tag matching the v* pattern
git tag -a v0.1.0 -m "v0.1.0"

# 3. Push — this triggers .github/workflows/release.yml
git push origin v0.1.0
```

The workflow triggers on `push: tags: 'v*'`.  The tag name becomes:
- The **GitHub Release tag** (`v0.1.0`)
- The **artifact version** in filenames (`stone-llama_0.1.0_linux_amd64.tar.gz`)
- The **injected version string** (`go build -ldflags "-X main.version=v0.1.0"`)

## What Gets Built

Each of the four targets is built in parallel by a matrix strategy:

| Target          | GOOS     | GOARCH   | CGO | Archive format |
|-----------------|----------|----------|-----|----------------|
| linux/amd64     | linux    | amd64    | 0   | `.tar.gz`      |
| windows/amd64   | windows  | amd64    | 0   | `.zip`         |
| darwin/amd64    | darwin   | amd64    | 0   | `.tar.gz`      |
| darwin/arm64    | darwin   | arm64    | 0   | `.tar.gz`      |

### Artifact naming

```
stone-llama_<version>_<os>_<arch>.tar.gz   # linux / darwin
stone-llama_<version>_windows_amd64.zip     # windows
stone-llama_<version>_<os>_<arch>.<ext>.sha256  # per-artifact checksum
checksums.txt                            # combined checksums
```

`<version>` is the tag name **without** the leading `v` (e.g. `0.1.0`).

### Archive contents

Each archive contains:

| File               | linux/darwin       | windows            |
|--------------------|:------------------:|:------------------:|
| `stone-llama`      | ✅                 | ✅ (`.exe`)        |
| `README.md`        | ✅ if present      | ✅ if present      |
| `LICENSE`          | ✅ if present      | ✅ if present      |
| `stone-llama.png`  | ✅ if present      | ✅ if present      |

Inclusion is guarded with `if [ -f ... ]` — a missing asset will **not** fail
the build.

## Local Verification

### Cross-compilation

```bash
# From the repo root, verify every target builds:
CGO_ENABLED=0 GOOS=linux   GOARCH=amd64   go build -trimpath -ldflags "-s -w" -o /tmp/sl-linux-amd64   ./cmd/stone-llama
CGO_ENABLED=0 GOOS=windows GOARCH=amd64   go build -trimpath -ldflags "-s -w" -o /tmp/sl-windows-amd64.exe ./cmd/stone-llama
CGO_ENABLED=0 GOOS=darwin  GOARCH=amd64   go build -trimpath -ldflags "-s -w" -o /tmp/sl-darwin-amd64  ./cmd/stone-llama
CGO_ENABLED=0 GOOS=darwin  GOARCH=arm64   go build -trimpath -ldflags "-s -w" -o /tmp/sl-darwin-arm64  ./cmd/stone-llama

# Verify file types:
file /tmp/sl-linux-amd64       # ELF 64-bit LSB executable
file /tmp/sl-windows-amd64.exe # PE32+ executable
file /tmp/sl-darwin-amd64      # Mach-O 64-bit x86_64
file /tmp/sl-darwin-arm64      # Mach-O 64-bit arm64
```

### Local dist build

```bash
make dist    # builds all four targets into dist/
ls -lh dist/
cat dist/checksums.txt

# Verify checksums independently:
cd dist
sha256sum -c stone-llama_*.tar.gz.sha256
sha256sum -c stone-llama_*.zip.sha256
```

### Verify the Linux binary runs

The binary does **not** require a GPU to print help or version:

```bash
/tmp/sl-linux-amd64 version    # stone-llama v0.1.0 (linux/amd64)
/tmp/sl-linux-amd64 --help     # prints usage
/tmp/sl-linux-amd64 doctor     # checks GPU/driver/runtime readiness
```

### Test install.sh locally

```bash
# Syntax check (no network):
sh -n scripts/install.sh

# Help (no network):
sh scripts/install.sh --help

# Against a real release (requires network):
sh scripts/install.sh --version v0.1.0 --prefix /tmp/sl-install-test
/tmp/sl-install-test/stone-llama version
```

## How to Verify Release Artifacts

After a tag push succeeds, check the GitHub Release page
(`https://github.com/rizperdana/stone-llama/releases/tag/v*`).

1. **Four archives** are present: one per target in the matrix.
2. **Four `.sha256` sidecars** — one per archive.
3. **One `checksums.txt`** — combined checksums for all four archives.

Verify any downloaded archive:

```bash
# Download manually, then:
sha256sum -c stone-llama_0.1.0_linux_amd64.tar.gz.sha256
tar xzf stone-llama_0.1.0_linux_amd64.tar.gz
./stone-llama_0.1.0_linux_amd64/stone-llama version
```

## Rollback Story

Releases are identified by **immutable** Git tags and GitHub Release assets.
Rolling back is always:

1. **Reinstall an older version:**

   ```bash
   sh scripts/install.sh --version v0.0.9
   ```

2. **Or pin a specific tag** in your shell profile:

   ```bash
   export STONE_LLAMA_VERSION=v0.0.9
   ```

3. **Rollback a bad tag** (only if the tag was pushed by mistake and **nothing**
   downstream has consumed it):

   ```bash
   git tag -d v0.1.0
   git push origin :refs/tags/v0.1.0   # delete remote tag
   # Fix, re-tag, re-push
   ```

   > **Never** move or delete a tag that anyone may have pulled.  Instead, cut a
   > new patch release (`v0.1.1`) with the fix.

4. **GitHub Release deletion** (same rule — only if no downstream consumer
   pulled the assets):

   ```bash
   gh release delete v0.1.0 --yes
   ```

## Makefile Targets

| Target | What it does                                  |
|--------|-----------------------------------------------|
| `build`| `CGO_ENABLED=0 go build` for the current platform |
| `test` | `go vet ./...` + `go test ./...`              |
| `fmt`  | `gofmt -s -w .`                               |
| `clean`| `rm -rf dist/` + local binary                 |
| `dist` | Cross-compile all four targets into `dist/`  |
