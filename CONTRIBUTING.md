# Contributing to stone-llama

Thanks for helping. stone-llama is deliberately small — one static binary, one
maintainer. Contributions that keep it small are the most welcome.

## Ground rules

- Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
- Vulnerabilities: follow [SECURITY.md](SECURITY.md) — never open a public issue.
- Use the bug / feature issue templates; the bug template asks for
  `stone-llama doctor` output because most "bugs" are environment problems
  (GPU, driver, CUDA, disk, ports) that `doctor` already sees.

## Setup

- Go 1.24+ (see `go.mod`).
- A GPU is **not** needed to build or run the unit tests — only `setup`,
  `serve` and `run` touch CUDA.

```sh
git clone https://github.com/<your-handle>/stone-llama.git
cd stone-llama
go build ./...
```

## Fork → branch → PR

```sh
# 1. Fork on GitHub, then:
git clone https://github.com/<your-handle>/stone-llama.git
cd stone-llama
git remote add upstream https://github.com/rizperdana/stone-llama.git
git fetch upstream

# 2. Branch from main — convention: feature/<your-handle>/<short-description>
git switch -c feature/<your-handle>/<short-description> upstream/main

# 3. Commit (see convention below), verify, push:
make check
git push -u origin feature/<your-handle>/<short-description>

# 4. Open a PR against rizperdana/stone-llama:main
```

`main` is protected: direct pushes are blocked for everyone except the
maintainer's admin bypass — everyone else goes through a PR.

## Commit-message convention

The release-notes generator (`gh release create --generate-notes`) groups
commits by these prefixes, so any convention you follow must be one of them:

| prefix          | for                                              |
| --------------- | ------------------------------------------------ |
| `feat(scope):`  | new user-visible behaviour                       |
| `fix(scope):`   | bug fixes                                        |
| `perf(scope):`  | performance improvements                         |
| `docs:`         | documentation only                               |
| `chore(scope):` | tooling, CI, housekeeping                        |

- Scope: lowercase, hyphenated (`serve`, `release`, `autofit`). Recommended for
  `feat`/`fix`/`perf`, optional for `docs`/`chore`.
- Real examples: `feat(run): M6 streaming REPL — load via /-/load…`,
  `fix(gate): refusal summary names first failing gate, not VRAM`,
  `chore: ignore *.exe cross-build artifacts`.
- The **PR title** must follow the same convention — the `PR title` check
  enforces it, and under squash-merge the PR title becomes the commit on `main`.

## Checks — run locally before pushing

```sh
make check
```

This is the fast set CI also runs:

1. `gofmt -l .` — must print nothing (CI: `.github/workflows/ci.yml`)
2. `go vet ./...`
3. `go test ./...`
4. `actionlint` over `.github/workflows/` (skipped with a note if you don't
   have it installed; CI always runs it: `.github/workflows/actionlint.yml`)

CI additionally cross-compiles all five release targets and smoke-tests
`version`, `--help` and `doctor`, plus `govulncheck`, CodeQL, dependency review
and workflow linting — see `.github/workflows/`.

## What a PR must include

- `make check` green.
- New or changed behaviour covered by tests; bug fixes get a regression test
  where practical.
- Docs updated when behaviour changes (README, `docs/`).
- Honest numbers: measured values come from a real run; estimates keep their
  `[est]` label — do not silently promote an estimate to a measurement.
- One concern per PR; no drive-by refactors.

## Review expectation

`CODEOWNERS` requests the maintainer's review automatically. Expect feedback on
behaviour, platform-matrix honesty (what is *supported* vs *builds but
untested*) and scope. Small focused PRs get small round-trips; releases are
maintainer-only (see `docs/RELEASE.md`).
