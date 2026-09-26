#!/bin/sh
# =============================================================================
# stone-llama installer
#
# Downloads a release from GitHub, verifies its SHA-256 checksum, and
# installs the binary to ~/.local/bin (or a --prefix of your choice).
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/rizperdana/stone-llama/main/scripts/install.sh | sh
#   sh scripts/install.sh [--version v1.2.3] [--prefix ~/go/bin]
#   sh scripts/install.sh --help
# =============================================================================
set -eu

BINARY="stone-llama"
REPO="rizperdana/stone-llama"
RELEASE_BASE="https://github.com/${REPO}/releases/download"
API_BASE="https://api.github.com/repos/${REPO}/releases"
PREFIX="${HOME}/.local/bin"
VERSION=""

# ---- helpers ---------------------------------------------------------------

die() {
    printf 'install: error: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Download and install stone-llama from GitHub Releases.

Options:
  --version <tag>   Install a specific release (e.g. v0.1.0).
                    Default: latest stable release.
  --prefix <dir>    Install directory.
                    Default: ~/.local/bin
  --help, -h        Show this help message

Examples:
  $(basename "$0")
  $(basename "$0") --version v0.2.0
  $(basename "$0") --prefix /usr/local/bin    # does NOT require root
EOF
}

# ---- argument parsing ------------------------------------------------------

while [ $# -gt 0 ]; do
    case "$1" in
        --version)
            [ $# -ge 2 ] || die "--version requires an argument"
            VERSION="$2"
            shift 2
            ;;
        --prefix)
            [ $# -ge 2 ] || die "--prefix requires an argument"
            PREFIX="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1 (try --help)"
            ;;
    esac
done

# ---- OS / arch detection ---------------------------------------------------

detect_platform() {
    os=$(uname -s)
    arch=$(uname -m)

    case "$os" in
        Linux)  goos="linux" ;;
        Darwin) goos="darwin" ;;
        *)
            die "unsupported OS: $os — only Linux and macOS are supported by this installer"
            ;;
    esac

    case "$arch" in
        x86_64|amd64) goarch="amd64" ;;
        aarch64|arm64) goarch="arm64" ;;
        *)
            die "unsupported architecture: $arch"
            ;;
    esac

    # Guard rails: only published combinations are valid.
    # linux only ships amd64; darwin ships amd64 + arm64.
    case "$goos:$goarch" in
        linux:arm64)
            die "linux/arm64 is not published — only linux/amd64 is supported at runtime (requires NVIDIA GPU + CUDA)"
            ;;
        darwin:*) : ;;        # darwin binaries support doctor/list/fit only
        linux:amd64) : ;;     # fully supported
    esac

    echo "${goos}:${goarch}"
}

PLATFORM=$(detect_platform)
GOOS=$(echo "$PLATFORM" | cut -d: -f1)
GOARCH=$(echo "$PLATFORM" | cut -d: -f2)

# Determine archive extension and binary name.
case "$GOOS" in
    windows)
        # Windows is not in the installer's supported set (only Linux + macOS),
        # but detect_platform would have died above for Linux/arm64.
        die "Windows is not supported by this installer — use the .zip from GitHub Releases manually"
        ;;
    *)
        ext="tar.gz"
        binname="$BINARY"
        ;;
esac

# ---- resolve version -------------------------------------------------------

if [ -z "$VERSION" ]; then
    printf 'Fetching latest release...\n'
    VERSION=$(curl -fsSL "${API_BASE}/latest" \
        | sed -n 's/.*"tag_name": "\([^"]*\)".*/\1/p')
    [ -n "$VERSION" ] || die "could not determine latest release version"
fi

# Normalise: ensure the tag starts with "v" for the download URL,
# and strip it for the artifact name.
case "$VERSION" in
    v*) VERSION_TAG="$VERSION" ;;
    *)  VERSION_TAG="v${VERSION}" ;;
esac
VERSION_NUM=$(echo "$VERSION_TAG" | sed 's/^v//')

ARTIFACT="stone-llama_${VERSION_NUM}_${GOOS}_${GOARCH}"
ARCHIVE_URL="${RELEASE_BASE}/${VERSION_TAG}/${ARTIFACT}.${ext}"
CHECKSUM_URL="${RELEASE_BASE}/${VERSION_TAG}/${ARTIFACT}.${ext}.sha256"

printf 'Installing %s %s (%s/%s)\n' "$BINARY" "$VERSION_TAG" "$GOOS" "$GOARCH"
printf '  archive:  %s\n' "$ARCHIVE_URL"
printf '  prefix:   %s\n' "$PREFIX"

# ---- download & verify -----------------------------------------------------

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT INT TERM

curl -fsSL "$ARCHIVE_URL"   -o "$TMPDIR/${ARTIFACT}.${ext}"
curl -fsSL "$CHECKSUM_URL"  -o "$TMPDIR/${ARTIFACT}.${ext}.sha256"

printf 'Verifying SHA-256 checksum...\n'
(
    cd "$TMPDIR"
    # The checksum file references the remote filename; our local file
    # has the same name, so sha256sum -c should resolve it directly.
    sha256sum -c "${ARTIFACT}.${ext}.sha256"
) || die "checksum verification FAILED — the downloaded file is corrupt or tampered with"

printf 'Checksum OK.\n'

# ---- extract ---------------------------------------------------------------

printf 'Extracting...\n'
case "$ext" in
    tar.gz)
        tar xzf "$TMPDIR/${ARTIFACT}.${ext}" -C "$TMPDIR"
        ;;
    *)
        unzip -q "$TMPDIR/${ARTIFACT}.${ext}" -d "$TMPDIR"
        ;;
esac

# Locate the binary inside the extracted archive.
# The archive contains a top-level directory named stone-llama_<ver>_<os>_<arch>.
BIN_PATH=$(find "$TMPDIR" -name "$binname" -type f | head -1)
[ -n "$BIN_PATH" ] || die "binary '$binname' not found inside archive"

# ---- install ---------------------------------------------------------------

mkdir -p "$PREFIX"
cp "$BIN_PATH" "${PREFIX}/${binname}"
chmod +x "${PREFIX}/${binname}"

printf '\n%s %s installed to %s\n' "$BINARY" "$VERSION_TAG" "${PREFIX}/${binname}"

# ---- PATH hint -------------------------------------------------------------

case ":${PATH}:" in
    *":${PREFIX}:"*)
        printf 'You already have %s in your PATH. Run: %s --help\n' "$PREFIX" "$BINARY"
        ;;
    *)
        printf '\n%s is not in your PATH.\n' "$PREFIX"
        printf 'Add it to your shell profile (~/.bashrc, ~/.zshrc, etc.):\n'
        printf '  export PATH="%s:$PATH"\n' "$PREFIX"
        ;;
esac
