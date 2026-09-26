#!/bin/sh
# =============================================================================
# stone-llama installer
#
# Downloads a release from GitHub, verifies its SHA-256 checksum against the
# bundled checksums.txt, and installs the binary to ~/.local/bin (or a --prefix
# of your choice).
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/rizperdana/stone-llama/main/scripts/install.sh | sh
#   sh scripts/install.sh [--version v0.1.0-rc1] [--prefix ~/go/bin]
#   sh scripts/install.sh --help
# =============================================================================
set -eu

BINARY="stone-llama"
REPO="rizperdana/stone-llama"
RELEASE_BASE="https://github.com/${REPO}/releases/download"
API_BASE="https://api.github.com/repos/${REPO}/releases"
PREFIX="${HOME}/.local/bin"
VERSION=""

# ---- sha256 tool detection --------------------------------------------------

# Linux: sha256sum; macOS: shasum -a 256
if command -v sha256sum >/dev/null 2>&1; then
	SHA256="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
	SHA256="shasum -a 256"
else
	die "no SHA-256 tool found (need sha256sum or shasum)" 2>/dev/null || \
		{ echo "install: no SHA-256 tool found (need sha256sum or shasum)" >&2; exit 1; }
fi

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
  --version <tag>   Install a specific version (e.g. v0.1.0-rc1)
                    Defaults to the latest release.
  --prefix <path>   Install directory (default: ~/.local/bin)
  --help, -h        Show this help message

Examples:
  sh scripts/install.sh
  sh scripts/install.sh --version v0.1.0-rc1 --prefix /usr/local/bin
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
		--help|-h)
			usage
			exit 0
			;;
		*)
			die "unknown option: $1 (use --help for usage)"
			;;
	esac
done

# ---- OS / arch detection ---------------------------------------------------

# detect_platform prints "os:arch" for the current host.
# Only linux and darwin are supported; everything else is rejected.
detect_platform() {
	os=$(uname -s | tr '[:upper:]' '[:lower:]')
	arch=$(uname -m)
	case "$arch" in
		x86_64|amd64) arch="amd64" ;;
		arm64|aarch64) arch="arm64" ;;
		*)             die "unsupported architecture: $arch (linux/amd64, linux/arm64, darwin/amd64, darwin/arm64 supported)" ;;
	esac
	case "$os" in
		linux|darwin) ;;
		*)              die "unsupported OS: $os (linux and darwin only — download manually for Windows)" ;;
	esac
	printf '%s:%s\n' "$os" "$arch"
}

PLATFORM=$(detect_platform)
GOOS=$(printf '%s' "$PLATFORM" | cut -d: -f1)
GOARCH=$(printf '%s' "$PLATFORM" | cut -d: -f2)

# Determine archive extension based on OS.
case "$GOOS" in
	linux|darwin) ext=tgz ;;
	*)            die "unsupported OS: $GOOS" ;;
esac

# Artifact name: stone-llama-<os>-<arch>.<ext>  (ollama-style, no version in name)
ARTIFACT="stone-llama-${GOOS}-${GOARCH}"

# ---- resolve version -------------------------------------------------------

if [ -z "$VERSION" ]; then
	# Fetch latest release tag from the GitHub API
	VERSION=$(curl -fsSL "${API_BASE}/latest" | \
		grep -m1 '"tag_name"' | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')
	[ -n "$VERSION" ] || die "could not determine latest release version — are releases published?"
fi

# Normalise: ensure the tag starts with "v" for the download URL.
case "$VERSION" in
	v*) ;;
	*)  VERSION="v${VERSION}" ;;
esac

ARCHIVE_URL="${RELEASE_BASE}/${VERSION}/${ARTIFACT}.${ext}"
CHECKSUMS_URL="${RELEASE_BASE}/${VERSION}/checksums.txt"

printf 'Installing %s %s (%s/%s)\n' "$BINARY" "$VERSION" "$GOOS" "$GOARCH"
printf '  archive:   %s\n' "$ARCHIVE_URL"
printf '  checksums: %s\n' "$CHECKSUMS_URL"
printf '  prefix:    %s\n' "$PREFIX"

# ---- download & verify -----------------------------------------------------

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT INT TERM

printf 'Downloading %s...\n' "${ARTIFACT}.${ext}"
curl -fsSL "$ARCHIVE_URL"   -o "$TMPDIR/${ARTIFACT}.${ext}"
curl -fsSL "$CHECKSUMS_URL" -o "$TMPDIR/checksums.txt"

printf 'Verifying SHA-256 checksum...\n'
# Extract the line for our artifact from the combined checksums.txt,
# then verify with sha256sum -c (POSIX: shasum -a 256 -c on macOS).
(
	cd "$TMPDIR"
	grep -F "${ARTIFACT}.${ext}" checksums.txt \
		| $SHA256 -c -
) || die "checksum verification FAILED — the downloaded file is corrupt or tampered with"

printf 'Checksum OK.\n'

# ---- extract ---------------------------------------------------------------

printf 'Extracting...\n'
case "$ext" in
	tgz)
		tar xzf "$TMPDIR/${ARTIFACT}.${ext}" -C "$TMPDIR"
		;;
	zip)
		( cd "$TMPDIR" && unzip -q "${ARTIFACT}.zip" )
		;;
esac

# Locate the binary inside the extracted archive.
# Archives contain a top-level directory named stone-llama-<os>-<arch>/.
BIN_PATH=$(find "$TMPDIR" -name "$BINARY" -type f | head -1)
[ -n "$BIN_PATH" ] || die "binary '$BINARY' not found inside archive"

# ---- install ---------------------------------------------------------------

mkdir -p "$PREFIX"
cp "$BIN_PATH" "${PREFIX}/${BINARY}"
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

printf 'Done.\n'
