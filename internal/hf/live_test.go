package hf

import (
	"os"
	"strings"
	"testing"
)

// TestLiveRepoMetaSmoke hits the real HuggingFace Hub with metadata-only
// calls (KB-scale, A5-sanctioned; never weight bytes):
//
//	HF_LIVE=1 go test ./internal/hf -run Live -v
func TestLiveRepoMetaSmoke(t *testing.T) {
	if os.Getenv("HF_LIVE") == "" {
		t.Skip("set HF_LIVE=1 for a live HuggingFace metadata smoke")
	}
	c := NewClient("", os.Getenv("HF_TOKEN"))

	repo, err := c.RepoMeta("async0x42/Qwen3-8B-exl3_4.0bpw")
	if err != nil {
		t.Fatalf("RepoMeta: %v", err)
	}
	if repo.SHA == "" || len(repo.Files) == 0 {
		t.Fatalf("repo = %+v, want sha + files", repo)
	}
	var configFound, lfsOID bool
	for _, f := range repo.Files {
		if f.Path == "config.json" && f.Size > 0 {
			configFound = true
		}
		if len(f.SHA256) == 64 {
			lfsOID = true
		}
	}
	if !configFound {
		t.Error("config.json missing from repo tree")
	}
	if !lfsOID {
		t.Error("no LFS sha256 oid in repo tree — resume verification needs them")
	}

	cfg, err := c.FetchFile(repo.ID, repo.SHA, "config.json")
	if err != nil {
		t.Fatalf("FetchFile: %v", err)
	}
	if !strings.Contains(string(cfg), "architectures") {
		t.Errorf("config.json unexpected: %.80s", cfg)
	}

	ids, err := c.Search("exl3")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(ids) == 0 {
		t.Error("search returned nothing for exl3")
	}
	t.Logf("repo %s @ %.10s, %d files, %d search hits", repo.ID, repo.SHA, len(repo.Files), len(ids))
}
