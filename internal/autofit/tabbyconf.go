package autofit

import (
	"fmt"
	"strconv"
	"strings"
)

// NormalizeForTabby converts a cache mode to a value TabbyAPI's config
// accepts: the legacy names FP16/Q8/Q6/Q4, otherwise the "k_bits,v_bits"
// pair syntax (Q2 → "2,2" — TabbyAPI has no legacy Q2).
func NormalizeForTabby(mode string) string {
	m := strings.ToUpper(strings.TrimSpace(mode))
	switch m {
	case "FP16", "Q8", "Q6", "Q4":
		return m
	}
	if strings.HasPrefix(m, "Q") {
		if n, err := strconv.Atoi(m[1:]); err == nil && n >= 2 && n <= 8 {
			return fmt.Sprintf("%d,%d", n, n)
		}
	}
	return m // already pair syntax like "2,4" (validated by BytesPerElement upstream)
}

// TabbyConfig is the slice of TabbyAPI's config.yml stone-llama owns.
type TabbyConfig struct {
	Host      string
	Port      int
	ModelDir  string
	ModelName string
	Ctx       int    // max_seq_len = cache_size (both set, 256-aligned)
	CacheMode string // raw mode; normalized to TabbyAPI syntax here
}

// RenderTabbyYAML emits the generated config passed to
// `start.py --config`. Fixed template, strconv.Quote for strings (valid
// YAML double-quoted scalars) — deliberately no YAML dependency for 8 scalars.
// ponytail: hand-rolled for exactly this shape; revisit if the config grows.
func RenderTabbyYAML(c TabbyConfig) string {
	q := strconv.Quote
	return fmt.Sprintf(`network:
  host: %s
  port: %d
  api_servers: ["OAI"]

model:
  model_dir: %s
  model_name: %s
  max_seq_len: %d
  cache_size: %d
  cache_mode: %s
  gpu_split_auto: true
  autosplit_reserve: [96]
`, q(c.Host), c.Port,
		q(c.ModelDir), q(c.ModelName),
		c.Ctx, c.Ctx,
		q(NormalizeForTabby(c.CacheMode)))
}
