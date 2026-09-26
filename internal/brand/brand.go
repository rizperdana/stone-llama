// Package brand carries the embedded app icon for version and serve
// banners. The terminal never renders the PNG — only its presence.
package brand

import (
	_ "embed"
	"fmt"
)

// icon.png is a byte-identical copy of assets/stone-llama-256.png
// (sha256 prefix b7deb60f16640c52; the docs worker owns assets/, so
// the embed target lives here). The banner never rasterises the PNG —
// only its length is printed — so the 256×256 asset is embedded, not
// the 2048×2048 one: 91,972 B instead of 1,254,528 B in the binary
// (−1,162,556 B, measured 2026-09-26).
//
//go:embed icon.png
var iconPNG []byte

// Banner is the one-line branding prefix for version/serve output.
// The PNG is treated as an opaque asset — no terminal rendering.
func Banner(version, platform string) string {
	return fmt.Sprintf("stone-llama %s (%s) — app icon stone-llama.png embedded (%d bytes)",
		version, platform, len(iconPNG))
}
