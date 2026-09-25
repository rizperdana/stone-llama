package pull

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"syscall"
	"time"

	"github.com/rizperdana/stone-llama/internal/hf"
)

// defaultFreeBytes reports filesystem free space for the models dir.
func defaultFreeBytes(path string) (int64, error) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(path, &st); err != nil {
		return 0, err
	}
	return int64(st.Bavail) * int64(st.Bsize), nil
}

var errChecksum = errors.New("sha256 does not match the published LFS hash")

const maxAttempts = 5

// fetchWithRetry downloads one file into staging, resuming across
// attempts and across process runs (Range from the on-disk size), and
// verifying the LFS sha256 before accepting the file. Checksum failures
// restart from scratch, at most twice — bad bytes should not loop forever.
func fetchWithRetry(client *hf.Client, opts Options, staging, fileURL string, f hf.File, label string, prog *progress) (string, error) {
	dest := filepath.Join(staging, f.Path)
	var lastErr error
	checksumFails := 0
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		if attempt > 1 {
			if d := opts.RetryDelay(attempt); d > 0 {
				time.Sleep(d)
			}
		}
		sum, err := fetchOnce(client, dest, fileURL, f, label, prog)
		if err == nil {
			return sum, nil
		}
		lastErr = err
		if errors.Is(err, errChecksum) {
			checksumFails++
			os.Remove(dest)
			if checksumFails >= 2 {
				return "", err
			}
		}
	}
	return "", lastErr
}

// fetchOnce performs one resumable download attempt. Every path either
// returns a verified sha256 or an error; partial bytes stay on disk for
// the next attempt (progress, never loss).
func fetchOnce(client *hf.Client, dest, fileURL string, f hf.File, label string, prog *progress) (string, error) {
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return "", err
	}

	offset := int64(0)
	if st, err := os.Stat(dest); err == nil && st.Size() > 0 {
		switch {
		case f.Size > 0 && st.Size() > f.Size:
			os.Truncate(dest, 0) // stale bytes beyond the known size
		case f.Size > 0 && st.Size() == f.Size:
			sum, herr := hashFile(dest)
			if herr == nil && (f.SHA256 == "" || sum == f.SHA256) {
				return sum, nil // already complete (prior run finished this file)
			}
			os.Truncate(dest, 0)
		default:
			offset = st.Size()
		}
	}

	resp, err := client.FetchRange(fileURL, offset)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusPartialContent: // 206: server honored the Range
	case http.StatusOK: // 200: full body — drop any resume offset
		if offset > 0 {
			os.Truncate(dest, 0)
			offset = 0
		}
	case http.StatusRequestedRangeNotSatisfiable:
		if offset > 0 && f.Size > 0 && offset == f.Size {
			if sum, herr := hashFile(dest); herr == nil && (f.SHA256 == "" || sum == f.SHA256) {
				return sum, nil
			}
		}
		os.Truncate(dest, 0)
		return "", fmt.Errorf("server could not resume at byte %d — restarting", offset)
	default:
		return "", fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	// Hash the already-present prefix, then stream the rest through
	// (file writer + hasher) so the final sha256 covers the whole file.
	h := sha256.New()
	if offset > 0 {
		rf, err := os.Open(dest)
		if err != nil {
			return "", err
		}
		if _, err := io.CopyN(h, rf, offset); err != nil {
			rf.Close()
			os.Truncate(dest, 0)
			return "", fmt.Errorf("read local prefix: %w", err)
		}
		rf.Close()
	}
	fh, err := os.OpenFile(dest, os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return "", err
	}
	defer fh.Close()
	if _, err := fh.Seek(offset, io.SeekStart); err != nil {
		return "", err
	}

	prog.begin(label, f.Size, offset)
	stream := &readerCB{r: resp.Body, cb: prog.add}
	n, err := io.Copy(io.MultiWriter(fh, h), stream)
	total := offset + n
	if err != nil {
		prog.finish()
		return "", fmt.Errorf("interrupted after %s of %s: %w",
			HumanBytes(total), HumanBytes(f.Size), err)
	}
	if f.Size > 0 && total != f.Size {
		return "", fmt.Errorf("truncated response: got %d bytes, want %d", total, f.Size)
	}
	sum := hex.EncodeToString(h.Sum(nil))
	if f.SHA256 != "" && sum != f.SHA256 {
		prog.finish()
		return "", errChecksum
	}
	prog.finish()
	return sum, nil
}

// hashFile computes the sha256 of an existing file.
func hashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// readerCB counts bytes flowing through Read for progress updates.
type readerCB struct {
	r  io.Reader
	cb func(int)
}

func (x *readerCB) Read(p []byte) (int, error) {
	n, err := x.r.Read(p)
	if n > 0 && x.cb != nil {
		x.cb(n)
	}
	return n, err
}

// progress renders a throttled single-line carriage-return status when
// Out is a TTY; silent otherwise (Quiet or non-TTY).
type progress struct {
	out      io.Writer
	enabled  bool
	label    string
	have     int64
	total    int64
	got      int64
	start    time.Time
	lastAt   time.Time
	rendered bool
}

func (p *progress) begin(label string, total, have int64) {
	if !p.enabled {
		return
	}
	p.label, p.total, p.have, p.got = label, total, have, 0
	p.start = time.Now()
	p.lastAt = time.Time{}
	p.rendered = false
}

func (p *progress) add(n int) {
	if !p.enabled {
		return
	}
	p.have += int64(n)
	p.got += int64(n)
	now := time.Now()
	if now.Sub(p.lastAt) < 100*time.Millisecond {
		return
	}
	p.lastAt = now
	p.render(now)
}

func (p *progress) render(now time.Time) {
	pct := 100.0
	if p.total > 0 {
		pct = float64(p.have) * 100 / float64(p.total)
		if pct > 100 {
			pct = 100
		}
	}
	elapsed := now.Sub(p.start).Seconds()
	if elapsed <= 0 {
		elapsed = 0.001
	}
	rate := int64(float64(p.got) / elapsed)
	fmt.Fprintf(p.out, "\r\x1b[K  %-32.32s %5.1f%%  %s/s", p.label, pct, HumanBytes(rate))
	p.rendered = true
}

func (p *progress) finish() {
	if p.enabled && p.rendered {
		fmt.Fprint(p.out, "\r\x1b[K")
		p.rendered = false
	}
}
