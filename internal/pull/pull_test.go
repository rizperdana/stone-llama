package pull

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"github.com/rizperdana/stone-llama/internal/store"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

// fakeHF is a zero-download stand-in for the HuggingFace Hub: metadata
// endpoints plus /resolve/ that can truncate bodies mid-stream to prove
// resume works.
type fakeHF struct {
	mu            sync.Mutex
	files         map[string][]byte // "org/repo/path" → content
	lfs           map[string]string // path → sha256 hex (published oid)
	gated         bool
	truncateAll   map[string]bool // every response for path is cut in half
	ignoreRange   bool
	searchIDs     []string
	resolveHits   map[string]int // path → hits
	truncatedMade int
	ranges        []string
}

func newFakeHF(t *testing.T) (*fakeHF, *httptest.Server) {
	t.Helper()
	f := &fakeHF{
		files:       map[string][]byte{},
		lfs:         map[string]string{},
		truncateAll: map[string]bool{},
		resolveHits: map[string]int{},
	}
	srv := httptest.NewServer(f)
	t.Cleanup(srv.Close)
	return f, srv
}

func (f *fakeHF) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	q := r.URL.Query()

	switch {
	case r.URL.Path == "/api/models" && q.Get("search") != "":
		if f.gated {
			w.WriteHeader(403)
			return
		}
		var b strings.Builder
		b.WriteString("[")
		for i, id := range f.searchIDs {
			if i > 0 {
				b.WriteString(",")
			}
			fmt.Fprintf(&b, `{"id":%q}`, id)
		}
		b.WriteString("]")
		w.Write([]byte(b.String()))

	case strings.HasPrefix(r.URL.Path, "/api/models/"):
		id := strings.TrimPrefix(r.URL.Path, "/api/models/")
		if f.gated {
			w.WriteHeader(403)
			return
		}
		has := false
		for p := range f.files {
			if strings.HasPrefix(p, id+"/") {
				has = true
				break
			}
		}
		if !has {
			w.WriteHeader(404)
			return
		}
		var b strings.Builder
		fmt.Fprintf(&b, `{"id":%q,"sha":"rev123","gated":false,"siblings":[`, id)
		first := true
		for p, content := range f.files {
			if !strings.HasPrefix(p, id+"/") {
				continue
			}
			if !first {
				b.WriteString(",")
			}
			first = false
			rel := strings.TrimPrefix(p, id+"/")
			if oid, ok := f.lfs[p]; ok {
				fmt.Fprintf(&b, `{"rfilename":%q,"lfs":{"oid":%q,"size":%d}}`, rel, oid, len(content))
			} else {
				fmt.Fprintf(&b, `{"rfilename":%q,"size":%d}`, rel, len(content))
			}
		}
		b.WriteString("]}")
		w.Write([]byte(b.String()))

	case strings.Contains(r.URL.Path, "/resolve/"):
		// /<repo-id>/resolve/<rev>/<path...>
		p := strings.TrimPrefix(r.URL.Path, "/")
		i := strings.Index(p, "/resolve/")
		rest := p[i+len("/resolve/"):]
		slash := strings.Index(rest, "/")
		if slash < 0 {
			w.WriteHeader(404)
			return
		}
		key := p[:i] + "/" + rest[slash+1:]
		content, ok := f.files[key]
		if !ok || f.gated {
			w.WriteHeader(404)
			return
		}
		f.resolveHits[key]++

		rangeHdr := r.Header.Get("Range")
		if rangeHdr != "" {
			f.ranges = append(f.ranges, rangeHdr+" "+key)
		}
		offset := int64(0)
		if rangeHdr != "" && !f.ignoreRange {
			if _, err := fmt.Sscanf(rangeHdr, "bytes=%d-", &offset); err == nil && offset > 0 && offset <= int64(len(content)) {
				w.Header().Set("Content-Range",
					fmt.Sprintf("bytes %d-%d/%d", offset, len(content)-1, len(content)))
				w.Header().Set("Content-Length", strconv.Itoa(len(content)-int(offset)))
				w.WriteHeader(206)
				body := content[offset:]
				if f.truncateAll[key] {
					f.truncatedMade++
					w.Write(body[:len(body)/2]) // cut mid-stream; server closes
					return
				}
				w.Write(body)
				return
			}
		}
		// Full-body response.
		w.Header().Set("Content-Length", strconv.Itoa(len(content)))
		w.WriteHeader(200)
		if f.truncateAll[key] {
			f.truncatedMade++
			w.Write(content[:len(content)/2])
			return
		}
		w.Write(content)

	default:
		w.WriteHeader(404)
	}
}

func shaHex(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

const testConfigJSON = `{
	"architectures": ["SmolLM3ForCausalLM"],
	"num_hidden_layers": 36,
	"num_key_value_heads": 4,
	"hidden_size": 2048,
	"num_attention_heads": 16,
	"max_position_embeddings": 65536
}`

func seedTiny(f *fakeHF) {
	cfg := []byte(testConfigJSON)
	weights := bytes.Repeat([]byte("EXL3WEIGHTS!"), 4096) // 48 KiB
	f.files["org/tiny-exl3/config.json"] = cfg
	f.files["org/tiny-exl3/quantization_config.json"] = []byte(`{"quant_method":"exl3","bits":3.5}`)
	f.files["org/tiny-exl3/model.safetensors"] = weights
	f.files["org/tiny-exl3/tokenizer.json"] = []byte(`{"tok":1}`)
	f.files["org/tiny-exl3/README.md"] = []byte("# readme — must never download")
	f.lfs["org/tiny-exl3/model.safetensors"] = shaHex(weights)
	f.searchIDs = []string{"org/tiny-exl3"}
}

func baseOpts(srv *httptest.Server, modelsDir string) Options {
	return Options{
		ModelsDir:  modelsDir,
		Ref:        "org/tiny-exl3",
		Yes:        true,
		Out:        &bytes.Buffer{},
		VRAMMiB:    4096,
		GPUName:    "RTX 3050",
		BaseURL:    srv.URL,
		FreeBytes:  func(string) (int64, error) { return 1 << 40, nil },
		RetryDelay: func(int) time.Duration { return 0 },
	}
}

func out(opts Options) string { return opts.Out.(*bytes.Buffer).String() }

func TestPullHappyPath(t *testing.T) {
	f, srv := newFakeHF(t)
	seedTiny(f)
	models := t.TempDir()
	opts := baseOpts(srv, models)
	// Roomy VRAM so the gate's verdict is a clean OK at the trained max —
	// on a tight card the prefill margin correctly downgrades it to warn.
	opts.VRAMMiB = 8192

	res, err := Run(opts)
	if err != nil {
		t.Fatal(err)
	}
	if res.Name != "tiny-exl3" || res.Files != 4 {
		t.Errorf("result = %+v, want name tiny-exl3 / 4 files", res)
	}
	if res.Verdict.Status != store.VerdictOK || res.Verdict.MaxCtx != 65536 {
		t.Errorf("verdict = %+v", res.Verdict)
	}

	final := filepath.Join(models, "tiny-exl3")
	for _, p := range []string{"config.json", "quantization_config.json", "model.safetensors", "tokenizer.json"} {
		if _, err := os.Stat(filepath.Join(final, p)); err != nil {
			t.Errorf("missing %s: %v", p, err)
		}
	}
	if f.resolveHits["org/tiny-exl3/README.md"] != 0 {
		t.Error("README.md must not be downloaded")
	}

	m, err := store.LoadManifest(final)
	if err != nil || m == nil {
		t.Fatalf("manifest: %v %v", m, err)
	}
	if m.Quant != "3.5bpw" || m.RepoID != "org/tiny-exl3" || m.Revision != "rev123" {
		t.Errorf("manifest = %+v", m)
	}
	if len(m.Files) != 4 || m.Files[2].SHA256 == "" {
		t.Errorf("manifest files = %+v", m.Files)
	}

	text := out(opts)
	if !strings.Contains(text, "gate: arch") || !strings.Contains(text, "pulled tiny-exl3") {
		t.Errorf("output missing gate/pull lines:\n%s", text)
	}
}

func TestPullConsentDeclinedDownloadsNothing(t *testing.T) {
	f, srv := newFakeHF(t)
	seedTiny(f)
	opts := baseOpts(srv, t.TempDir())
	opts.Yes = false
	opts.Interactive = true
	opts.Stdin = strings.NewReader("n\n")

	_, err := Run(opts)
	if err == nil || !strings.Contains(err.Error(), "aborted") {
		t.Fatalf("err = %v, want aborted", err)
	}
	if !strings.Contains(out(opts), "Proceed? [y/N]") {
		t.Errorf("no consent prompt:\n%s", out(opts))
	}
	if got := f.resolveHits["org/tiny-exl3/model.safetensors"]; got != 0 {
		t.Errorf("weight bytes downloaded before consent: %d", got)
	}
}

func TestPullNonInteractiveRequiresYes(t *testing.T) {
	f, srv := newFakeHF(t)
	seedTiny(f)
	opts := baseOpts(srv, t.TempDir())
	opts.Yes = false
	opts.Interactive = false

	_, err := Run(opts)
	if err == nil || !strings.Contains(err.Error(), "--yes") {
		t.Fatalf("err = %v, want --yes requirement", err)
	}
	if f.resolveHits["org/tiny-exl3/model.safetensors"] != 0 {
		t.Error("weights must not download without consent")
	}
}

func TestPullWarnBlocksNonInteractiveWithoutYes(t *testing.T) {
	f, srv := newFakeHF(t)
	seedTiny(f)
	f.files["org/tiny-exl3/config.json"] = []byte(`{
		"architectures": ["UnknownForCausalLM"],
		"num_hidden_layers": 36, "num_key_value_heads": 4,
		"hidden_size": 2048, "num_attention_heads": 16,
		"max_position_embeddings": 65536}`)
	opts := baseOpts(srv, t.TempDir())
	opts.Yes = false

	_, err := Run(opts)
	if err == nil || !strings.Contains(err.Error(), "--yes") {
		t.Fatalf("err = %v, want --yes for warnings", err)
	}
	if !strings.Contains(out(opts), "⚠") {
		t.Errorf("warning not printed:\n%s", out(opts))
	}
	if f.resolveHits["org/tiny-exl3/model.safetensors"] != 0 {
		t.Error("weights must not download")
	}
}

func TestPullResumesAcrossRuns(t *testing.T) {
	f, srv := newFakeHF(t)
	seedTiny(f)
	key := "org/tiny-exl3/model.safetensors"
	f.truncateAll[key] = true

	models := t.TempDir()
	opts := baseOpts(srv, models)

	_, err := Run(opts)
	if err == nil {
		t.Fatal("run1 should fail while server truncates")
	}
	staging := filepath.Join(models, ".tiny-exl3.staging")
	partial := filepath.Join(staging, "model.safetensors")
	st, serr := os.Stat(partial)
	if serr != nil {
		t.Fatalf("staging partial missing: %v", serr)
	}
	if st.Size() == 0 || st.Size() >= int64(len(f.files[key])) {
		t.Fatalf("partial size = %d, want 0 < size < full", st.Size())
	}

	// Heal the server; run2 must resume from the on-disk offset via Range.
	f.mu.Lock()
	f.truncateAll[key] = false
	f.mu.Unlock()
	res, err := Run(opts)
	if err != nil {
		t.Fatalf("run2: %v", err)
	}
	if res.Files != 4 {
		t.Errorf("res.Files = %d", res.Files)
	}
	got, rerr := os.ReadFile(filepath.Join(models, "tiny-exl3", "model.safetensors"))
	if rerr != nil {
		t.Fatalf("final weights: %v", rerr)
	}
	if shaHex(got) != f.lfs[key] {
		t.Error("resumed file failed checksum")
	}
	wantRange := fmt.Sprintf("bytes=%d-", st.Size())
	found := false
	for _, r := range f.ranges {
		if strings.HasPrefix(r, wantRange) && strings.HasSuffix(r, key) {
			found = true
		}
	}
	if !found {
		t.Errorf("no Range %q in %v", wantRange, f.ranges)
	}
}

func TestPullChecksumMismatchFailsLoudly(t *testing.T) {
	f, srv := newFakeHF(t)
	seedTiny(f)
	f.lfs["org/tiny-exl3/model.safetensors"] = strings.Repeat("0", 64)

	models := t.TempDir()
	opts := baseOpts(srv, models)
	opts.RetryDelay = func(int) time.Duration { return 0 }

	_, err := Run(opts)
	if err == nil || !strings.Contains(err.Error(), "sha256") {
		t.Fatalf("err = %v, want sha256 failure", err)
	}
	if _, err := os.Stat(filepath.Join(models, "tiny-exl3")); err == nil {
		t.Error("corrupt download must never become a model")
	}
}

func TestPullGatedRepoRequiresLogin(t *testing.T) {
	f, srv := newFakeHF(t)
	seedTiny(f)
	f.gated = true

	opts := baseOpts(srv, t.TempDir())
	_, err := Run(opts)
	if err == nil || !strings.Contains(err.Error(), "login") {
		t.Fatalf("err = %v, want login hint", err)
	}
}

func TestPullExistingModelAndForce(t *testing.T) {
	f, srv := newFakeHF(t)
	seedTiny(f)
	models := t.TempDir()
	final := filepath.Join(models, "tiny-exl3")
	if err := os.MkdirAll(final, 0o755); err != nil {
		t.Fatal(err)
	}
	oldMarker := filepath.Join(final, "old.txt")
	os.WriteFile(oldMarker, []byte("old"), 0o644)

	opts := baseOpts(srv, models)
	if _, err := Run(opts); err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("err = %v, want already-exists", err)
	}
	if f.resolveHits["org/tiny-exl3/model.safetensors"] != 0 {
		t.Error("must fail before download")
	}

	opts.Force = true
	if _, err := Run(opts); err != nil {
		t.Fatalf("force: %v", err)
	}
	if _, err := os.Stat(oldMarker); err == nil {
		t.Error("--force must replace the old model dir")
	}
	if _, err := os.Stat(filepath.Join(final, "model.safetensors")); err != nil {
		t.Errorf("new content missing: %v", err)
	}
}

func TestPullMultipleQuantsRequireTag(t *testing.T) {
	f, srv := newFakeHF(t)
	cfg := []byte(testConfigJSON)
	w1 := bytes.Repeat([]byte("Q35"), 4096)
	w2 := bytes.Repeat([]byte("Q40"), 4096)
	f.files["org/multi-exl3/3.5bpw/config.json"] = cfg
	f.files["org/multi-exl3/3.5bpw/model.safetensors"] = w1
	f.files["org/multi-exl3/4bpw/config.json"] = cfg
	f.files["org/multi-exl3/4bpw/model.safetensors"] = w2
	f.lfs["org/multi-exl3/3.5bpw/model.safetensors"] = shaHex(w1)
	f.lfs["org/multi-exl3/4bpw/model.safetensors"] = shaHex(w2)

	models := t.TempDir()
	opts := baseOpts(srv, models)
	opts.Ref = "org/multi-exl3"
	_, err := Run(opts)
	if err == nil || !strings.Contains(err.Error(), "multiple quants") {
		t.Fatalf("err = %v, want multiple-quants error", err)
	}
	if f.resolveHits["org/multi-exl3/3.5bpw/model.safetensors"] != 0 {
		t.Error("no weights before scope is unambiguous")
	}

	opts.Ref = "org/multi-exl3:3.5bpw"
	res, err := Run(opts)
	if err != nil {
		t.Fatal(err)
	}
	if res.Name != "multi-exl3-3.5bpw" {
		t.Errorf("name = %s", res.Name)
	}
	if f.resolveHits["org/multi-exl3/4bpw/model.safetensors"] != 0 {
		t.Error("untagged quant dir must not download")
	}
	m, _ := store.LoadManifest(filepath.Join(models, res.Name))
	if m == nil || m.Quant != "3.5bpw" {
		t.Errorf("manifest = %+v", m)
	}
}

func TestPullGGUFRefusedBeforeWeightBytes(t *testing.T) {
	f, srv := newFakeHF(t)
	f.files["org/gguf-only/config.json"] = []byte(testConfigJSON)
	f.files["org/gguf-only/model.gguf"] = bytes.Repeat([]byte("gguf"), 1024)

	opts := baseOpts(srv, t.TempDir())
	opts.Ref = "org/gguf-only"
	_, err := Run(opts)
	if err == nil || !strings.Contains(err.Error(), "gate") {
		t.Fatalf("err = %v, want gate refusal", err)
	}
	if !strings.Contains(out(opts), "ollama with GGUF") {
		t.Errorf("no GGUF hint:\n%s", out(opts))
	}
	if f.resolveHits["org/gguf-only/model.gguf"] != 0 {
		t.Error("GGUF bytes must never download")
	}
}

func TestPullExl2RefusedBeforeWeightBytes(t *testing.T) {
	f, srv := newFakeHF(t)
	f.files["org/exl2-model/config.json"] = []byte(testConfigJSON)
	f.files["org/exl2-model/quantization_config.json"] = []byte(`{"quant_method":"exl2"}`)
	w := bytes.Repeat([]byte("w"), 4096)
	f.files["org/exl2-model/model.safetensors"] = w

	opts := baseOpts(srv, t.TempDir())
	opts.Ref = "org/exl2-model"
	_, err := Run(opts)
	if err == nil || !strings.Contains(err.Error(), "gate") {
		t.Fatalf("err = %v", err)
	}
	if !strings.Contains(out(opts), "ExLlamaV2") {
		t.Errorf("no exl2 detail:\n%s", out(opts))
	}
	if f.resolveHits["org/exl2-model/model.safetensors"] != 0 {
		t.Error("EXL2 bytes must never download")
	}
}

func TestPullDiskPreflightBlocksDownload(t *testing.T) {
	f, srv := newFakeHF(t)
	seedTiny(f)
	opts := baseOpts(srv, t.TempDir())
	opts.FreeBytes = func(string) (int64, error) { return 10, nil }

	_, err := Run(opts)
	if err == nil || !strings.Contains(err.Error(), "not enough disk") {
		t.Fatalf("err = %v", err)
	}
	if f.resolveHits["org/tiny-exl3/model.safetensors"] != 0 {
		t.Error("weights must not download when disk is short")
	}
}

func TestPullBareNameSearchResolution(t *testing.T) {
	f, srv := newFakeHF(t)
	seedTiny(f)
	opts := baseOpts(srv, t.TempDir())
	opts.Ref = "tiny"
	res, err := Run(opts)
	if err != nil {
		t.Fatal(err)
	}
	if res.Name != "tiny-exl3" {
		t.Errorf("name = %s", res.Name)
	}

	// No match: clear search and assert the GGUF-aware error.
	f.mu.Lock()
	f.searchIDs = nil
	f.mu.Unlock()
	opts2 := baseOpts(srv, t.TempDir())
	opts2.Ref = "tiny"
	_, err = Run(opts2)
	if err == nil || !strings.Contains(err.Error(), "no EXL3 repo found") {
		t.Fatalf("err = %v", err)
	}
}

func TestPullSearchMultipleNonInteractiveLists(t *testing.T) {
	f, srv := newFakeHF(t)
	seedTiny(f)
	f.files["org2/tiny-exl3-2/config.json"] = []byte(testConfigJSON)
	f.searchIDs = []string{"org/tiny-exl3", "org2/tiny-exl3-2"}

	opts := baseOpts(srv, t.TempDir())
	opts.Ref = "tiny"
	opts.Interactive = false
	_, err := Run(opts)
	if err == nil || !strings.Contains(err.Error(), "owner/repo") {
		t.Fatalf("err = %v, want disambiguation error", err)
	}
}

func TestPullLockContention(t *testing.T) {
	f, srv := newFakeHF(t)
	seedTiny(f)
	models := t.TempDir()

	lockPath := filepath.Join(models, ".pull.lock")
	if err := os.MkdirAll(models, 0o700); err != nil {
		t.Fatal(err)
	}
	lk, err := os.Create(lockPath)
	if err != nil {
		t.Fatal(err)
	}
	defer lk.Close()
	// Hold the flock the way another process would.
	if err := flock(lk); err != nil {
		t.Fatal(err)
	}

	opts := baseOpts(srv, models)
	_, err = Run(opts)
	if err == nil || !strings.Contains(err.Error(), "lock") {
		t.Fatalf("err = %v, want lock contention", err)
	}
	f.mu.Lock()
	hits := f.resolveHits["org/tiny-exl3/model.safetensors"]
	f.mu.Unlock()
	if hits != 0 {
		t.Error("no downloads under lock contention")
	}
}

func flock(f *os.File) error {
	return syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
}
