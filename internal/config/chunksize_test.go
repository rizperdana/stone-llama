package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func loadWithConfig(t *testing.T, body string) (Config, error) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("STONE_LLAMA_CONFIG", path)
	return Load()
}

// autofit.chunk_size: 0 (auto, the default) or 512-4096 (forced override).
// Anything else is a config error at load time, not a surprise at serve
// time — TabbyAPI's ModelLoadRequest rejects out-of-band values as 422.
func TestLoadChunkSizeValidation(t *testing.T) {
	for _, tc := range []struct {
		body    string
		want    int
		wantErr string
	}{
		{`{}`, 0, ""},                                 // absent → auto
		{`{"autofit":{"chunk_size":0}}`, 0, ""},       // explicit auto
		{`{"autofit":{"chunk_size":512}}`, 512, ""},   // floor
		{`{"autofit":{"chunk_size":4096}}`, 4096, ""}, // ceiling
		{`{"autofit":{"chunk_size":256}}`, 0, "autofit.chunk_size"},
		{`{"autofit":{"chunk_size":4097}}`, 0, "autofit.chunk_size"},
		{`{"autofit":{"chunk_size":-1}}`, 0, "autofit.chunk_size"},
	} {
		cfg, err := loadWithConfig(t, tc.body)
		if tc.wantErr != "" {
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("%s: err = %v, want mention of %s", tc.body, err, tc.wantErr)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: unexpected error %v", tc.body, err)
			continue
		}
		if cfg.Autofit.ChunkSize != tc.want {
			t.Errorf("%s: chunk_size = %d, want %d", tc.body, cfg.Autofit.ChunkSize, tc.want)
		}
	}
}
