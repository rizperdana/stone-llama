#!/bin/sh
# =============================================================================
# stone-llama installer
#
# Downloads a release from GitHub, verifies its SHA-256 checksum against the
# published checksums.txt BEFORE extracting anything, and installs the binary
# to ~/.local/bin (or $PREFIX / --prefix of your choice).
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/rizperdana/stone-llama/main/scripts/install.sh | sh
#   sh scripts/install.sh [--version v0.1.0-rc1] [--prefix ~/go/bin]
#   sh scripts/install.sh --uninstall [--prefix ~/go/bin]
#   sh scripts/install.sh --help
#
# Version resolution (ollama-style pinning):
#   --version <tag>  install exactly that tag — pre-releases included
#                    (e.g. v0.1.0-rc1)
#   (default)        latest NON-pre-release (GitHub's /releases/latest
#                    endpoint excludes pre-releases — pin one explicitly)
#
# Platform truth: install.sh supports linux/darwin, but only linux/amd64 is a
# supported runtime platform (ExLlamaV3 is NVIDIA-CUDA only). The Windows
# binary builds but is UNTESTED at runtime — download the .zip manually.
# =============================================================================
set -eu

BINARY="stone-llama"
REPO="rizperdana/stone-llama"
RELEASE_BASE="https://github.com/${REPO}/releases/download"
API_BASE="https://api.github.com/repos/${REPO}/releases"
PREFIX="${PREFIX:-${HOME}/.local/bin}"
VERSION=""
UNINSTALL=0

# ---- helpers ---------------------------------------------------------------

die() {
	printf 'install: %s\n' "$*" >&2
	exit 1
}

usage() {
	cat <<'USAGE'
stone-llama installer

Usage:
  sh scripts/install.sh [options]

Options:
  --version <tag>   Install a specific version (e.g. v0.1.0-rc1).
                    Pre-release tags work when pinned explicitly.
                    Defaults to the latest non-pre-release.
  --prefix <path>   Install directory (default: $PREFIX or ~/.local/bin)
  --uninstall       Remove the installed binary from the prefix
  --help, -h        Show this help message

Environment:
  PREFIX            Same as --prefix (flag wins if both are given)

Examples:
  sh scripts/install.sh
  sh scripts/install.sh --version v0.1.0-rc1 --prefix /usr/local/bin
  sh scripts/install.sh --uninstall
  curl -fsSL https://raw.githubusercontent.com/rizperdana/stone-llama/main/scripts/install.sh | sh -s -- --version v0.1.0-rc1
USAGE
}

# ---- argument parsing ------------------------------------------------------

while [ $# -gt 0 ]; do
	case "$1" in
		--version)
			[ $# -ge 2 ] || die "--version requires a value"
			VERSION="$2"
			shift 2
			;;
		--prefix)
			[ $# -ge 2 ] || die "--prefix requires a value"
			PREFIX="$2"
			shift 2
			;;
		--uninstall)
			UNINSTALL=1
			shift
			;;
		--help|-h)
			usage
			exit 0
			;;
		*)
			die "unknown option: $1 (use --help for usage)"
			;;
	esac
done

# ---- uninstall (runs on any platform, no download) -------------------------

if [ "$UNINSTALL" -eq 1 ]; then
	target="${PREFIX}/${BINARY}"
	if [ -e "$target" ]; then
		rm -f "$target" || die "could not remove $target (permissions? try sudo)"
		printf 'Removed %s\n' "$target"
	else
		printf 'Nothing to remove: %s does not exist\n' "$target"
	fi
	exit 0
fi

# ---- sha256 tool detection --------------------------------------------------
# After argument handling so --help/--uninstall work on any machine.
# Linux: sha256sum; macOS: shasum -a 256

if command -v sha256sum >/dev/null 2>&1; then
	sha256() { sha256sum "$@"; }
elif command -v shasum >/dev/null 2>&1; then
	sha256() { shasum -a 256 "$@"; }
else
	die "no SHA-256 tool found (need sha256sum or shasum) — refusing to install an unverified binary"
fi

# ---- OS / arch detection ---------------------------------------------------

# detect_platform prints "os:arch" for the current host.
# Only linux and darwin binaries are installable; everything else gets a
# clear unsupported-platform message.
detect_platform() {
	os=$(uname -s | tr '[:upper:]' '[:lower:]')
	arch=$(uname -m)
	case "$arch" in
		x86_64|amd64)  arch="amd64" ;;
		arm64|aarch64) arch="arm64" ;;
		*)
			die "unsupported architecture '$arch' — published builds: linux/amd64, linux/arm64, darwin/amd64, darwin/arm64 (no ${os}/${arch} build exists)"
			;;
	esac
	case "$os" in
		linux|darwin) ;;
		mingw*|msys*|cygwin*|windows)
			die "Windows is not supported by this installer — download stone-llama-windows-amd64.zip from https://github.com/${REPO}/releases manually (the binary builds but is UNTESTED at runtime)"
			;;
		*)
			die "unsupported OS '$os' — this installer supports linux and darwin only (no ${os} build is published)"
			;;
	esac
	printf '%s:%s\n' "$os" "$arch"
}

PLATFORM=$(detect_platform)
GOOS=$(printf '%s' "$PLATFORM" | cut -d: -f1)
GOARCH=$(printf '%s' "$PLATFORM" | cut -d: -f2)

# Artifact name: stone-llama-<os>-<arch>.tgz  (ollama-style, no version in name)
ARTIFACT="stone-llama-${GOOS}-${GOARCH}"
ext=tgz # linux/darwin only — Windows would be .zip but has no sh installer

# ---- resolve version -------------------------------------------------------

if [ -z "$VERSION" ]; then
	# Latest NON-pre-release release. Pre-releases are excluded by GitHub's
	# /releases/latest endpoint — pin one with --version vX.Y.Z-rc1.
	VERSION=$(curl -fsSL "${API_BASE}/latest" | \
		grep -m1 '"tag_name"' | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')
	[ -n "$VERSION" ] || die "could not determine the latest release from ${API_BASE}/latest — no non-pre-release published yet? Pin one explicitly: --version vX.Y.Z-rc1"
fi

# Normalise: ensure the tag starts with "v" for the download URL.
case "$VERSION" in
	v*) ;;
	*)  VERSION="v${VERSION}" ;;
esac

# The tag goes into a URL — reject anything that isn't tag-shaped.
case "$VERSION" in
	*[!A-Za-z0-9._-]*) die "invalid version tag '$VERSION' (allowed: letters, digits, '.', '_', '-')" ;;
esac

ARCHIVE_URL="${RELEASE_BASE}/${VERSION}/${ARTIFACT}.${ext}"
CHECKSUMS_URL="${RELEASE_BASE}/${VERSION}/checksums.txt"

printf 'Installing %s %s (%s/%s)\n' "$BINARY" "$VERSION" "$GOOS" "$GOARCH"
printf '  archive:   %s\n' "$ARCHIVE_URL"
printf '  checksums: %s\n' "$CHECKSUMS_URL"
printf '  prefix:    %s\n' "$PREFIX"

# ---- download --------------------------------------------------------------
# curl -f: HTTP errors (404 etc.) exit non-zero and print the status line, so
# no failure is ever swallowed. Each download gets an explicit die() with a
# hint, because set -e alone would exit with no context.

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT INT TERM

printf 'Downloading %s...\n' "${ARTIFACT}.${ext}"
curl -fsSL -o "$TMPDIR/${ARTIFACT}.${ext}" "$ARCHIVE_URL" \
	|| die "archive download failed: ${ARCHIVE_URL} — does release ${VERSION} exist? (gh release view ${VERSION})"
curl -fsSL -o "$TMPDIR/checksums.txt" "$CHECKSUMS_URL" \
	|| die "checksum download failed: ${CHECKSUMS_URL} — release ${VERSION} must publish checksums.txt (see docs/RELEASE.md)"

# ---- verify BEFORE extraction ----------------------------------------------

printf 'Verifying SHA-256 checksum...\n'
# Capture OUR artifact's checksum line first, and refuse explicitly if it is
# absent. A bare `grep … | sha256sum -c -` loses grep's exit status (the
# pipeline reports only the checksum tool's) and leaves a missing entry to
# each tool's empty-input behavior — GNU sha256sum and shasum were both
# observed to fail on empty input here, but the capture makes the refusal
# independent of that and names the artifact instead.
line=$(grep -E "(^|[[:space:]])${ARTIFACT}[.]${ext}\$" "$TMPDIR/checksums.txt" || true)
if [ -z "$line" ]; then
	die "no entry for ${ARTIFACT}.${ext} in checksums.txt — refusing to install an unverified binary"
fi
printf '%s\n' "$line" | ( cd "$TMPDIR" && sha256 -c - ) \
	|| die "checksum verification FAILED — the downloaded file is corrupt or tampered with (expected: ${line})"

printf 'Checksum OK.\n'

# ---- extract ---------------------------------------------------------------

printf 'Extracting...\n'
tar xzf "$TMPDIR/${ARTIFACT}.${ext}" -C "$TMPDIR" \
	|| die "failed to extract ${ARTIFACT}.${ext} — the archive is corrupt (re-run, or verify checksums.txt)"

# Locate the binary inside the extracted archive.
# Archives contain a top-level directory named stone-llama-<os>-<arch>/.
BIN_PATH=$(find "$TMPDIR" -name "$BINARY" -type f | head -1)
[ -n "$BIN_PATH" ] || die "binary '$BINARY' not found inside archive ${ARTIFACT}.${ext}"

# ---- install ---------------------------------------------------------------

mkdir -p "$PREFIX" || die "cannot create prefix '$PREFIX' (permissions?)"
cp "$BIN_PATH" "${PREFIX}/${BINARY}" || die "cannot write ${PREFIX}/${BINARY} (permissions?)"
chmod +x "${PREFIX}/${BINARY}"

printf '\n%s %s installed to %s\n' "$BINARY" "$VERSION" "${PREFIX}/${BINARY}"

# ---- PATH hint -------------------------------------------------------------

case ":${PATH}:" in
	*:${PREFIX}:*)
		printf '  Your PATH already includes %s\n' "$PREFIX"
		;;
	*)
		printf '  Add %s to your PATH, e.g.:\n' "$PREFIX"
		printf '    export PATH="%s:$PATH"\n' "$PREFIX"
		;;
esac

# ---- uninstall hint --------------------------------------------------------

printf 'Uninstall: sh scripts/install.sh --uninstall --prefix %s\n' "$PREFIX"
printf '  (piped: curl -fsSL https://raw.githubusercontent.com/%s/%s/scripts/install.sh | sh -s -- --uninstall --prefix %s)\n' \
	"${REPO%%/*}" "${REPO#*/}" "$PREFIX"

printf 'Done.\n'
