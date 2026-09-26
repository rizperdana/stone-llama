# Release Engineering

This document is the runbook for cutting a `stone-llama` release. It covers
artifact naming, platform support, the CI/release workflow pipeline, the
release-notes template, local verification, and the release checklist.

It is written to match `.github/workflows/ci.yml` and
`.github/workflows/release.yml` **as they exist in the same commit** — if you
change a workflow, update this file in that commit.

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
`ollama-darwin.tgz`, `ollama-windows-amd64.zip`, their installer scripts, and a
single `sha256sum.txt` — no version in the filenames, one combined checksum
file. We mirror that pattern so users and CI use consistent, predictable names
regardless of the release tag. The binary itself carries the version via
`main.version` ldflags (`-X main.version=vX.Y.Z`), so `stone-llama version`
always reports the tag.

**Why not per-file `.sha256` sidecars?** They duplicate `checksums.txt` and
force `install.sh` to know the exact filename. With a single combined file,
`install.sh` captures its artifact's line and verifies it before extraction.
A bare `grep | sha256sum -c -` pipeline is deliberately *not* used: it loses
grep's exit status and leaves a missing entry to whatever the checksum tool
does with empty input (GNU `sha256sum` and `shasum` were both observed to
fail on empty input — the capture does not depend on that) and can only
report a generic error. Capturing the line refuses by name instead.

## Platform support matrix

| Platform          | Artifact                       | Status                         |
| ----------------- | ------------------------------ | ------------------------------ |
| linux/amd64       | `stone-llama-linux-amd64.tgz` | ✅ Builds + runs (CI smoke)    |
| linux/arm64       | `stone-llama-linux-arm64.tgz` | ⚠️ Builds only (untested)      |
| darwin/amd64      | `stone-llama-darwin-amd64.tgz`| ⚠️ Builds only (untested)      |
| darwin/arm64      | `stone-llama-darwin-arm64.tgz`| ⚠️ Builds only (untested)      |
| windows/amd64     | `stone-llama-windows-amd64.zip`| ⚠️ Builds only (untested)     |

> **linux/amd64 is the only supported platform.** stone-llama is CUDA-only
> (ExLlamaV3) — no CPU/AMD/Metal backend exists. CPU or Apple users should use
> [ollama with GGUF models](https://ollama.com). The Windows binary builds but
> has never been executed on Windows; macOS binaries can never serve.

**Builds** (cross-compiled) but **UNTESTED at runtime** for all platforms
except linux/amd64. linux/arm64 is compiled here but there is no arm64 machine
in the development environment — it is expected to work but has not been
executed.

Per `git log`, M5 (`feat(serve)`, `925e1f8`) and M6 (`feat(run)`, `2028710`)
are implemented — the pre-`v0.1.0` hold note has been removed; the gate is
CLEARED (see the checklist below). `version`, `--help`, and `doctor` are the
commands known to work on any platform.

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

**Tag gap (why release.yml has its own gate):** `ci.yml` triggers on
pushes/PRs to `main` only — its `on.push.branches: [main]` filter does not
match tag refs, so a tag pushed at an unvetted commit would bypass CI
entirely. The `verify` job in `release.yml` (next section) re-runs the same
format/vet/test commands against the tagged commit before anything is built.

## Release pipeline

**`.github/workflows/release.yml`** triggers on:
- Tag push matching `v*`
- `workflow_dispatch` with a `version` input (dispatch input wins over ref name)

The workflow has three jobs, chained `verify` → `build` → `release`; each
depends on the previous one, so a gate failure stops the release instead of
being reported after the fact.

1. **`verify`** — runs first; `build` has `needs: verify`:
   1. `actions/checkout@v4` + `actions/setup-go@v5` (Go 1.24)
   2. **Tagged-commit check** (tag pushes only): `git rev-parse HEAD` must
      equal `refs/tags/<tag>^{}` — a stale or wrong checkout fails the gate
   3. `gofmt -l .` (fail if non-empty), `go vet ./...`,
      `go test ./... -count=1` — the same commands `ci.yml` runs on `main`
      (cheap: one platform, no cross-compile matrix, no smoke build)
2. **`build`** — matrix over the 5 targets above (`fail-fast: false` so one
   failure doesn't mask others), builds + packages each artifact
   (`stone-llama-<os>-<arch>.tgz` / `.zip`), **attests build provenance**
   (`actions/attest-build-provenance@v4`), uploads to
   `actions/upload-artifact@v4`.
3. **`release`** — runs these steps in order:
   1. `actions/checkout@v4` with `fetch-depth: 0` (full history + tags — the
      release notes group `git log` by commit type)
   2. **Determine version** — validate `^v?[0-9]+\.[0-9]+\.[0-9]+(-…)?$`, then
      set `IS_PRERELEASE`: **any tag with a `-` suffix** (`v0.1.0-rc1`,
      `-beta`, `-alpha`, anything semver-pre) → pre-release; a **plain
      `vX.Y.Z`** → latest
   3. Download all artifacts (`actions/download-artifact@v4` with
      `merge-multiple: true` — singular, NOT `download-artifacts`)
   4. **Flatten + verify** — every artifact must exist and be > 1 MB; generate
      `checksums.txt` with `sha256sum`; then a **coverage check**: every
      artifact must have a line in `checksums.txt`, and the line count must
      equal the artifact count (nothing missing, nothing extra)
   5. **Attest `checksums.txt`** — signed provenance for the checksum file;
      the 5 archives are attested in their build legs (see Permissions)
   6. **Generate release notes** — render
      `.github/release-notes-template.md` (see next section)
   7. **Create or update release** — idempotent: if the release exists, re-run
      uploads assets with `--clobber` and refreshes the notes with
      `gh release edit --notes-file`; otherwise `gh release create` with
      `--notes-file` and `--prerelease` (pre-release tags) or `--latest`
      (plain tags)

There is **no `continue-on-error` anywhere** in this workflow, by design: an
unrendered placeholder in the notes, a missing checksum line, or an oversized
failure must fail the release loudly.

**Permissions (least privilege).** The workflow default is `contents: read`;
each job declares only what it needs:

| Job       | Permissions                                                              | Why                          |
| --------- | ------------------------------------------------------------------------ | ---------------------------- |
| `verify`  | `contents: read`                                                        | checkout                     |
| `build`   | `contents: read`, `id-token: write`, `attestations: write`, `artifact-metadata: write` | build + attestation |
| `release` | `contents: write`, `id-token: write`, `attestations: write`, `artifact-metadata: write` | create release, upload assets, attest `checksums.txt` |

The three attestation permissions are **required** by
`actions/attest-build-provenance@v4` per the permissions block documented by
its backing action, [`actions/attest`](https://github.com/actions/attest)
(`id-token` mints the OIDC token for the Sigstore signing certificate,
`attestations` persists the statement, `artifact-metadata` creates the
artifact storage record). Sources: the `actions/attest-build-provenance`
README (v4 is a wrapper over `actions/attest`; latest tag `v4.2.2`), the
`actions/attest` README, and GitHub's *Using artifact attestations to
establish provenance for builds* documentation.

**Provenance.** Every published asset — the 5 archives (attested in their
build legs) and `checksums.txt` (attested in the release job) — carries a
signed in-toto/SLSA statement binding it to this workflow run and commit.
Verify any download with:

```sh
gh attestation verify stone-llama-linux-amd64.tgz -o rizperdana/stone-llama
```

The gate re-runs on an idempotent re-run of an existing tag (cheap: one
platform, vet+test only) and does not change the `--clobber`/`edit` path.

## Release-notes template

`.github/release-notes-template.md` is the body of every release. Structure
(mirrors how ollama/llama.cpp releases read: headline, install, what's
changed, full-changelog link):

1. Headline `# stone-llama __VERSION__` + one-line project summary
2. `__PRERELEASE_NOTICE__` — rendered only for pre-release tags
3. **Install** — one-liners per platform (linux/macOS pinned `curl | sh`,
   Windows manual `.zip`), `--version`/`--prefix`/`--uninstall` usage
4. **What's Changed** — `__CHANGELOG__`, grouped by commit type
5. `__FULL_CHANGELOG__` — compare link to the previous tag (only when a
   previous tag exists)
6. **Checksums** — `__CHECKSUMS__` (the generated `checksums.txt`, verbatim)
7. **Platform support (honest matrix)** and **Known limitations** — CUDA-only,
   linux/amd64 only, Windows builds-but-untested, macOS can never serve,
   CPU/AMD/Apple → ollama + GGUF

Placeholders are substituted with `awk` in the release job; any placeholder
still present after rendering fails the run. `__CHANGELOG__` is built from
`git log --no-merges --pretty=%s <prev-tag>..<tag>` grouped by the repository's
Conventional Commits convention (`feat` → Features, `fix` → Bug fixes, `docs`
→ Documentation, `chore|ci|test|build|refactor|perf|style` → Maintenance,
everything else → Other changes). **Subjects are printed verbatim — notes are
never reworded or invented.** Edit the one-line summary in the template before
tagging if you want a per-release summary; it renders unchanged otherwise.

There is deliberately **no root `CHANGELOG.md`**: per-tag release notes are
generated from the real history, and a hand-maintained changelog drifts.

## install.sh

`scripts/install.sh` is POSIX `sh`, requires no root, and is safe to pipe
(`curl … | sh -s -- <flags>`). Flags and env:

- `--version <tag>` — pin a specific version; **pre-release tags work when
  pinned** (`--version v0.1.0-rc1`). Default: latest **non**-pre-release
  (GitHub's `/releases/latest` endpoint excludes pre-releases).
- `--prefix <path>` / `PREFIX=<path>` — install directory (flag wins),
  default `~/.local/bin`
- `--uninstall` — remove `${PREFIX}/stone-llama`; an uninstall hint is also
  printed after every install
- `--help` / `-h` — usage (works even without sha256 tools present)

Behavior guarantees:

- Robust OS/arch detection with explicit refusal text for unsupported
  platforms (Windows gets "download the .zip manually — builds but untested";
  any other OS/arch names what *is* published).
- The SHA-256 line for the artifact is **captured and checked non-empty
  before** extraction; a missing line or a mismatch aborts with the expected
  value printed. Verification happens before `tar` ever runs.
- Every `curl` uses `-fsSL` (HTTP errors exit non-zero, nothing hides a
  failure) and each download failure dies with the exact URL plus a hint.
- Tag input is validated (allowed charset) before it goes into a URL.

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

# 4. Validate YAML + workflows
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); yaml.safe_load(open('.github/workflows/release.yml'))"
actionlint .github/workflows/*.yml       # best offline signal; skip if not installed

# 4b. The release verify gate (same commands — must all pass)
gofmt -l .                      # must print nothing
go vet ./...
go test ./... -count=1

# 5. Test install.sh syntax and basic behavior
sh -n scripts/install.sh
sh scripts/install.sh --help
sh scripts/install.sh --bad-option   # should exit 1 with usage hint

# 6. Smoke-test the linux binary
cd dist && tar xzf stone-llama-linux-amd64.tgz && \
  ./stone-llama-linux-amd64/stone-llama version && \
  ./stone-llama-linux-amd64/stone-llama --help > /dev/null && \
  ./stone-llama-linux-amd64/stone-llama doctor 2>&1 | grep "NVIDIA GPU"
```

Render-test the release notes locally (same awk logic as the workflow, no
network, no GitHub write):

```sh
printf -- '- feat(release): example\n- fix(cli): example fix\n' > /tmp/subjects.txt
# …run the two awk blocks from release.yml "Generate release notes" with
# VERSION=v0.0.0-test IS_PRERELEASE=1 GITHUB_REPOSITORY=rizperdana/stone-llama
# then: grep -n '__[A-Z][A-Z_]*__' release/release_notes.md  # must print nothing
```

## install.sh end-to-end test

To test the full install path against a real release (downloads one ~few-MB
archive; no model weights):

```sh
PREFIX=/tmp/sl-install-test sh scripts/install.sh --version v0.1.0-rc1
/tmp/sl-install-test/stone-llama version
sh scripts/install.sh --uninstall --prefix /tmp/sl-install-test   # removes it
```

## Cutting a release

### Pre-release checklist

- [x] **Serve/ps/stop (M5) implemented** — required before v0.1.0 (shipped, `925e1f8`)
- [x] **Run (M6) implemented** — required before v0.1.0 (shipped, `2028710`)
- [x] **install.sh proven** against a staging release — `v0.1.0-rc1` verified
      end-to-end (published asset size matched, sha256 matched, binary ran)
- [ ] **CI green** on all 5 targets (compile + smoke test) — verify at tag time
- [ ] **Release-notes template summary** reviewed for this tag
- [ ] **README/docs install claims match the published artifacts** (see
      "Known doc drift" below)
- [ ] **Provenance verified** — `gh attestation verify <asset> -o rizperdana/stone-llama`
      passes for this tag's assets (first provable on the first tag pushed
      after this workflow change lands)

**`v0.1.0` gate: CLEARED.** M5/M6 are implemented and the `v0.1.0-rc1`
pre-release exercised the release machinery (naming, checksums, install.sh).
Cutting `v0.1.0` is the owner's decision — this document does not tag anything.

### Tag-cutting steps

```sh
# 1. Commit release engineering changes with explicit paths only
git add Makefile .github/ scripts/ docs/RELEASE.md
git commit -m "feat(release): ollama-style artifacts, linux/arm64, workflow_dispatch

- Rename artifacts: stone-llama-<os>-<arch>.tgz/.zip (no version in name)
- Add linux/arm64 to build matrix (untested)
- Add workflow_dispatch with version input
- Make release job idempotent (gh release view check)
- Auto-detect pre-release from any semver -suffix tag
- Single checksums.txt with full-coverage verification (no .sha256 sidecars)
- Add run-smoke CI step: version, --help, doctor (GPU-less refusal)
- Verify download-artifact@v4 (not download-artifacts) action name"

git push origin main

# 2. Cut pre-release tag (annotated)
git tag -a v0.1.0-rc1 -m "Pre-release: test ollama-style artifact pipeline"
git push origin v0.1.0-rc1

# 3. Verify the release workflow ran, artifacts uploaded, and the release
#    is marked pre-release (rc suffix) — plain vX.Y.Z would be "latest"
gh release view v0.1.0-rc1 --json assets,isPrerelease --jq '{assets:[.assets[].name],pre:.isPrerelease}'
gh attestation verify stone-llama-linux-amd64.tgz -o rizperdana/stone-llama

# 4. Test install.sh against the release
PREFIX=/tmp/sl-install-test sh scripts/install.sh --version v0.1.0-rc1
/tmp/sl-install-test/stone-llama version
```

For a stable release: tag `vX.Y.Z` (no suffix) — the workflow publishes it as
the **latest** release with `--latest` and renders the notes without the
pre-release notice.

### Rollback

If a release is broken:

```sh
gh release delete v0.1.0-rc1 --yes     # removes GitHub release (keeps tag)
git tag -d v0.1.0-rc1                  # removes local tag
git push origin --delete v0.1.0-rc1    # removes remote tag
# Fix, re-tag, re-push
```

## Rejected, with reasons

For a project of this size — **one small static Go binary, no vendoring, no
redistribution of TabbyAPI (AGPL)** — the following were considered and
rejected:

| Idea | Why rejected |
| ---- | ------------ |
| Docker image bundling torch/TabbyAPI | Multi-GB image, and TabbyAPI is AGPL — bundling it in an image we publish is a redistribution we deliberately avoid (`setup` fetches it consent-gated at runtime instead). |
| goreleaser | The 40-line workflow already builds all 5 targets with exact names and an idempotent `gh release` step. A release framework + config adds a dependency and a translation layer for zero extra targets. |
| Homebrew tap / winget / scoop / APT repo | Publishing a package for a platform whose runtime is untested (Windows) or can never serve (macOS) would overclaim support. `install.sh` + manual `.zip` cover the one supported platform. Hosted package repos also need ongoing sync maintenance. |
| `install.ps1` one-liner | Same reason: the Windows binary builds but is untested at runtime — shipping an installer for it would imply a support level that does not exist. |
| Multi-arch CI emulation (QEMU) | Cross-compilation already produces arm64; emulation adds minutes of flaky CI and still doesn't *prove* runtime behavior — arm64 verification needs real hardware. |
| Code signing / notarization (Apple, sigstore/cosign) | No Apple developer certificate; macOS can never serve. `checksums.txt` over GitHub's TLS is proportionate now; cosign keyless is the upgrade path if the threat model grows. |
| Per-file `.sha256` sidecars | Duplicate `checksums.txt` (see artifact naming). |
| Root `CHANGELOG.md` | The history contains merge commits and duplicate subjects from merged branches; curating it by hand invites invented entries. Per-tag notes are generated verbatim from `git log` instead. |

## Known doc drift (found while doing release work, NOT fixed here)

These files are owned by other writers — reported, not edited:

- `README.md` — "No release is published yet … a sha256-verifying install.sh
  lands with the first release" is stale: `v0.1.0-rc1` is published and
  `scripts/install.sh` exists. Also claims `run`/`serve` print "ships in M5/M6".
- `docs/INSTALLATION.md` — Option A shows old artifact names
  (`stone-llama_X.Y.Z_linux_amd64.tar.gz`), `.sha256` sidecars, and "No
  release has been published yet"; actual names are
  `stone-llama-<os>-<arch>.tgz` with a single combined `checksums.txt`, and
  rc1 is published.
- `docs/ARCHITECTURE.md` — artifact example `stone-llama-vX.Y.Z-linux-amd64.tar.gz`
  does not match the published naming scheme.
- `site/index.html` — "Release pending" install note is stale now that rc1
  exists.

## Action name audit

| Action                      | Status | Notes                          |
| --------------------------- | ------ | ------------------------------ |
| `actions/checkout@v4`       | ✅     | Official (`fetch-depth: 0` in the release job) |
| `actions/setup-go@v5`       | ✅     | Official                       |
| `actions/upload-artifact@v4`| ✅     | Official (singular)            |
| `actions/download-artifact@v4` | ✅  | Official (singular — NOT `download-artifacts`) |
| `actions/attest-build-provenance@v4` | ✅ | Official GitHub action (latest `v4.2.2`) — signed SLSA provenance; needs `id-token`/`attestations`/`artifact-metadata` write |
| `gh CLI`                    | ✅     | Pre-installed on ubuntu-latest |
