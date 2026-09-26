# Security Policy

## Supported versions

| Version            | Supported                                     |
| ------------------ | --------------------------------------------- |
| `main`             | ✅ fixes land here                            |
| latest release     | ✅ back-port only if the release is affected  |
| older releases     | ❌ update to the latest release               |

stone-llama is pre-1.0 (`v0.1.0-rc1`): there is no long-term-support line —
report against `main` or the latest release.

## Threat surface — what is in scope

stone-llama runs an HTTP server on **loopback** and shells out to a model
runtime (TabbyAPI / ExLlamaV3). "Local only" is not "safe by default", so the
following are explicitly in scope:

- **Request handling** — the OpenAI-compatible server (`serve`): route
  parsing, JSON/request body handling, streaming responses, and anything that
  turns a request into filesystem or process activity.
- **Token handling** — API tokens/keys for the local server and any token
  written to config files, manifests, logs or child-process environments.
- **File paths** — model directories, `manifests`/state files, archive
  extraction in `setup` and `scripts/install.sh`: path traversal, symlink
  handling, overwrite of unexpected files.
- **Child-process argv** — `pull`, `setup`, `serve` construct command lines
  for the model runtime from model ids, paths and flags; crafted values must
  never become extra arguments or shell syntax.
- **Supply chain in the install path** — downloads fetched by `install.sh` /
  `setup` (checksum verification, pinned versions).

Out of scope: prompt injection into model outputs, malicious model *weights*
(loaded only after explicit consent — report if the consent/guard rails can be
bypassed), and denial of service by exhausting GPU/disk yourself.

## Reporting a vulnerability

**Please do not open a public issue for a security report.**

1. **Preferred:** GitHub's private advisory — if you can see a
   *Report a vulnerability* button under the Security tab of this repository,
   use it (private vulnerability reporting; enable with
   `gh api -X PUT repos/rizperdana/stone-llama/private-vulnerability-reporting/enabled`).
2. **Fallback (always works):** email **perdana.rizki16@gmail.com** — the
   address published on the maintainer's GitHub profile
   ([rizperdana](https://github.com/rizperdana)) — subject
   `[SECURITY] stone-llama: <short summary>`.

Include: description and impact, reproduction steps, `stone-llama version` and
`stone-llama doctor` output, affected commit/tag if known.

**Disclosure policy:** you report privately → maintainer acknowledges as soon
as practical (target: within 7 days) → coordinated fix and credit → public
advisory/CVE when a fix ships. Reports that turn out non-issues are still
answered.
