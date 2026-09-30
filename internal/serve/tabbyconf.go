package serve

import (
	"fmt"
	"os"
	"path/filepath"
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
	// Load tuning from an autofit verdict (Result.ChunkSize/Warmup).
	// Zero/unset omits the keys entirely — TabbyAPI defaults apply, so
	// the boot (model-less) render stays byte-identical.
	ChunkSize int
	Warmup    bool
}

// RenderTabbyYAML emits the generated config passed to
// `start.py --config`. Fixed template, strconv.Quote for strings (valid
// YAML double-quoted scalars) — deliberately no YAML dependency for a
// handful of scalars. Ctx=0 / CacheMode="" leave those keys blank: a
// model-less child takes TabbyAPI defaults and per-load args win
// (load requests are ephemeral). tool_format is always set explicitly
// so tool_calls work without any client-side prompt dialect.
// ponytail: hand-rolled for exactly this shape; revisit if the config grows.
func RenderTabbyYAML(c TabbyConfig) string {
	q := strconv.Quote
	ctxLines := "  max_seq_len:\n  cache_size:\n"
	if c.Ctx > 0 {
		ctxLines = fmt.Sprintf("  max_seq_len: %d\n  cache_size: %d\n", c.Ctx, c.Ctx)
	}
	modeLine := "  cache_mode:\n"
	if c.CacheMode != "" {
		modeLine = "  cache_mode: " + q(NormalizeForTabby(c.CacheMode)) + "\n"
	}
	// Load tuning: omitted when the render carries no verdict (boot
	// model-less config) — omit == TabbyAPI default, so a key TabbyAPI
	// hasn't been shown to accept never reaches its YAML parser.
	loadLines := ""
	if c.ChunkSize > 0 {
		loadLines += fmt.Sprintf("  chunk_size: %d\n", c.ChunkSize)
	}
	if c.Warmup {
		loadLines += "  warmup: true\n"
	}
	return fmt.Sprintf(`network:
  host: %s
  port: %d
  api_servers: ["OAI"]

model:
  model_dir: %s
  model_name: %s
%s%s%s  tool_format: auto
  gpu_split_auto: true
  autosplit_reserve: [96]
`, q(c.Host), c.Port,
		q(c.ModelDir), q(c.ModelName),
		ctxLines, modeLine, loadLines)
}

// --- attach convenience (TabbyAPI-specific seam lives here, per the seam
// rule: no TabbyAPI knowledge past tabby_backend.go/tabbyconf.go) ---

// TabbyAPITokensFile returns the api_tokens.yml path inside a checkout.
// TabyAPI (the pinned f07131cd checkout) reads it from its start.py CWD, so
// it is the repo root in an adopted/provisioned checkout.
func TabbyAPITokensFile(checkout string) string {
	return filepath.Join(checkout, "api_tokens.yml")
}

// ParseTabbyAPIToken reads an api_tokens.yml and returns the bearer key,
// preferring api_key (the OpenAI surface) and falling back to admin_key.
// It is read-only and never copies the credential into our state — the
// discovered path is handed straight to --key-file by the caller.
func ParseTabbyAPIToken(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	// api_tokens.yml is two `key: value` lines — parse by hand, no YAML dep.
	api, admin := "", ""
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "#") || line == "" {
			continue
		}
		k, v, _ := strings.Cut(line, ":")
		v = strings.TrimSpace(v)
		switch strings.TrimSpace(k) {
		case "api_key":
			api = v
		case "admin_key":
			if admin == "" {
				admin = v
			}
		}
	}
	if api != "" {
		return api, nil
	}
	if admin != "" {
		return admin, nil
	}
	return "", fmt.Errorf("no api_key or admin_key in %s", path)
}
