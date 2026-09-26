// Package brand carries the embedded app icon for version and serve
// banners. The terminal never renders the PNG — only its presence.
package brand

import (
	_ "embed"
	"fmt"
)

// icon.png is a copy of assets/stone-llama.png (the docs worker owns
// assets/, so the embed target lives here). Re-copy when the source
// icon changes.
//
//go:embed icon.png
var iconPNG []byte

// IconBytes reports the size of the embedded app icon.
func IconBytes() int { return len(iconPNG) }

// Banner is the one-line branding prefix for version/serve output.
// The PNG is treated as an opaque asset — no terminal rendering.
func Banner(version, platform string) string {
	return fmt.Sprintf("stone-llama %s (%s) — app icon stone-llama.png embedded (%d bytes)",
		version, platform, len(iconPNG))
}
