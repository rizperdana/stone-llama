# Release Engineering

This document is the runbook for cutting a `stone-llama` release. It covers
artifact naming, platform support, the CI/release workflow pipeline, local
verification, and the pre-release checklist.

---

## Artifact naming

Releases use **ollama-style** naming: a plain per-platform name with no version
embedded in the filename, plus a single combined `checksums.txt`.

```
stone-llama-linux-amd64.tgz
stone-llama-linux-arm64.tgz
stone-llama-darwin-amd64.tgz
stone-llama-darwin-arm64.tgz
stone-llama-windows-amd64.zip
checksums.txt
```

**Rationale:** ollama publishes `ollama-linux-amd64.tar.zst`,
`ollama-darwin.tgz`, `ollama-windows-amd64.zip`, and `sha256sum.txt` — no
per-file sidecars, no version in the filename. We mirror that pattern so users
and CI use consistent, predictable names regardless of the release tag. The
binary itself carries the version via `main.version` ldflags
(`-X main.version=vX.Y.Z`), so `stone-llama version` always reports the tag.

**Why not per-file `.sha256` sidecars?** They duplicate `checksums.txt` and
force `install.sh` to know the exact filename. With a single combined file,
`install.sh` greps for its artifact line:

```sh
grep -F "stone-llama-linux-amd64.tgz" checksums.txt | sha256sum -c -
```

## Platform support matrix

| Platform          | Artifact                      | Status                         |
| ----------------- | ----------------------------- | ------------------------------ |
| linux/amd64       | `stone-llama-linux-amd64.tgz` | ✅ Builds + runs (CI smoke)    |
| linux/arm64       | `stone-llama-linux-arm64.tgz` | ⚠️ Builds only (untested)      |
| darwin/amd64      | `stone-llama-darwin-amd64.tgz`| ⚠️ Builds only (untested)      |
| darwin/arm64      | `stone-llama-darwin-arm64.tgz`| ⚠️ Builds only (untested)      |
| windows/amd64     | `stone-llama-windows-amd64.zip`| ⚠️ Builds only (untested)     |

> **stone-llama is CUDA-only (ExLlamaV3).** A CPU-only fallback is not available —
> use [ollama with GGUF models](https://ollama.com) on non-NVIDIA hardware.
> `serve`/`ps`/`stop`/`run` (M5/M6) are still landing — do not cut `v0.1.0`
> until those are implemented. The `version`, `--help`, and `doctor` commands
> work on any platform; `serve`/`ps`/`stop`/`run` will fail at runtime.

**Builds** (cross-compiled) but **UNTESTED at runtime** for all platforms
except linux/amd64. linux/arm64 is compiled here but there is no arm64 machine
in the development environment — it is expected to work but has not been
executed.

## CI pipeline

**`.github/workflows/ci.yml`** triggers on `push` (main), `pull_request`, and
`workflow_dispatch`. It runs:

1. `gofmt -l .` — formatting check (fails on unformatted files)
2. `go vet ./...` — static analysis
3. `go test ./...` — unit tests
4. Compile-check for all 5 release targets (`-o /dev/null`, no workspace pollution)
5. **Run-smoke** (linux/amd64 only — no GPU in CI):
   - `./stone-llama-smoke version` → exit 0, prints `stone-llama`
   - `./stone-llama-smoke --help` → exit 0
   - `./stone-llama-smoke doctor` → no segfault; prints "needs an NVIDIA GPU" refusal

The compile-check matrix proves **compilation**, not that the artifact runs. The
platform table above is the honest record of what has actually been executed.

## Release pipeline

**`.github/workflows/release.yml`** triggers on:
- Tag push matching `v*`
- `workflow_dispatch` with a `version` input

The workflow has two jobs:

1. **`build`** — matrix over 5 targets, builds + packages each artifact,
   uploads to `actions/upload-artifact@v4`.
2. **`release`** — downloads all artifacts via `actions/download-artifact@v4`
   (note: singular, NOT `download-artifacts`), verifies checksums, creates the
   GitHub release (or updates if it already exists — idempotent), uploads all
   artifacts, and generates release notes with a platform table + install
   command + checksum table.

Pre-release is auto-detected: tags containing `-rc`, `-beta`, `-alpha`, or
`-dev` are marked as pre-releases.

## Local verification

After any Makefile or workflow change, run:

```sh
# 1. Build all targets
make dist

# 2. Verify checksums
cd dist && sha256sum -c checksums.txt

# 3. Check archive contents
tar tzf stone-llama-linux-amd64.tgz      # should list stone-llama-linux-amd64/{stone-llama,README.md,LICENSE,stone-llama.png}
unzip -l stone-llama-windows-amd64.zip   # should list stone-llama-windows-amd64/{stone-llama.exe,...}

# 4. Validate YAML
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); yaml.safe_load(open('.github/workflows/release.yml'))"

# 5. Test install.sh syntax and basic behavior
sh -n scripts/install.sh
sh scripts/install.sh --help
sh scripts/install.sh --bad-option   # should exit 1

# 6. Smoke-test the linux binary
cd dist && tar xzf stone-llama-linux-amd64.tgz && \
  ./stone-llama-linux-amd64/stone-llama version && \
  ./stone-llama-linux-amd64/stone-llama --help > /dev/null && \
  ./stone-llama-linux-amd64/stone-llama doctor 2>&1 | grep "NVIDIA GPU"
```

## install.sh end-to-end test

To test the full install path against a real release:

```sh
# Cut the pre-release tag first (see "Cutting a release" below), then:
PREFIX=/tmp/sl-install-test sh scripts/install.sh --version v0.1.0-rc1
/tmp/sl-install-test/stone-llama version
```

`install.sh` is POSIX `sh`, requires no root, and supports:
- `--version <tag>` — install a specific version (defaults to latest)
- `--prefix <path>` — install directory (default: `~/.local/bin`)
- `--help` / `-h` — usage

It detects the host OS/arch (linux/amd64, linux/arm64, darwin/amd64, darwin/arm64),
downloads the matching archive + `checksums.txt`, verifies the SHA-256, extracts,
and copies the binary to the prefix.

## Cutting a release

### Pre-release checklist (v0.1.0-rc1 is OK — v0.1.0 is NOT)

- [ ] **Serve/ps/stop (M5) implemented** — required before v0.1.0
- [ ] **Run (M6) implemented** — required before v0.1.0
- [ ] **CI green** on all 5 targets (compile + smoke test)
- [ ] **install.sh proven** against a staging release (rc1 exercises this)

> v0.1.0 must NOT be cut until serve/ps/stop (M5) and run (M6) are implemented.
> The rc pre-release exists to test the release machinery (naming, checksums,
> install.sh end-to-end). Cut rc1 first, verify install.sh, then cut v0.1.0
> after M5/M6 land.

### Tag-cutting steps

```sh
# 1. Commit release engineering changes with explicit paths only
git add Makefile .github/ scripts/ docs/RELEASE.md
git commit -m "feat(release): ollama-style artifacts, linux/arm64, workflow_dispatch

- Rename artifacts: stone-llama-<os>-<arch>.tgz/.zip (no version in name)
- Add linux/arm64 to build matrix (untested)
- Add workflow_dispatch with version input
- Make release job idempotent (gh release view check)
- Auto-detect pre-release from rc/beta/alpha/dev tags
- Single checksums.txt (no per-file .sha256 sidecars)
- Add run-smoke CI step: version, --help, doctor (GPU-less refusal)
- Verify download-artifact@v4 (not download-artifacts) action name"

git push origin main

# 2. Cut pre-release tag (annotated)
git tag -a v0.1.0-rc1 -m "Pre-release: test ollama-style artifact pipeline"
git push origin v0.1.0-rc1

# 3. Verify the release workflow ran and all artifacts uploaded
gh release view v0.1.0-rc1 --json assets --jq '.assets[].name'

# 4. Test install.sh against the release
PREFIX=/tmp/sl-install-test sh scripts/install.sh --version v0.1.0-rc1
/tmp/sl-install-test/stone-llama version
```

### Rollback

If a release is broken:

```sh
gh release delete v0.1.0-rc1 --yes     # removes GitHub release (keeps tag)
git tag -d v0.1.0-rc1                  # removes local tag
git push origin --delete v0.1.0-rc1    # removes remote tag
# Fix, re-tag, re-push
```

### Action name audit

| Action                      | Status | Notes                          |
| --------------------------- | ------ | ------------------------------ |
| `actions/checkout@v4`       | ✅     | Official                       |
| `actions/setup-go@v5`       | ✅     | Official                       |
| `actions/upload-artifact@v4`| ✅     | Official (singular)            |
| `actions/download-artifact@v4` | ✅  | Official (singular — NOT `download-artifacts`) |
| `gh CLI`                    | ✅     | Pre-installed on ubuntu-latest |
