## What and why

<!-- The problem being solved, and why this approach. -->

## How tested

<!-- Commands run, output observed. "N/A — docs only" is fine for docs PRs. -->

## Checklist

- [ ] `make check` passes (`gofmt -l .` clean, `go vet ./...`, `go test ./...`, `actionlint`)
- [ ] New or changed behaviour is covered by tests; bug fixes include a regression test where practical
- [ ] Docs updated if behaviour changed (README, `docs/`)
- [ ] Numbers are honest: measured values from a real run; estimates keep their `[est]` label
- [ ] Commit messages follow the `type(scope):` convention — `feat` / `fix` / `perf` / `docs` / `chore` (release notes group on these)
- [ ] PR title follows the same convention (CI enforces it)
- [ ] No unrelated refactors or drive-by changes

Fixes #
