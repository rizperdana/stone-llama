// Package render is a small streaming markdown renderer for stone-llama's
// interactive chat output, matching ollama's simple style: bold, italic,
// code spans, fenced blocks, headers, bullets, and links — no syntax
// highlighting, no HTML, no nesting inside a span (a `**bold _x_**`
// keeps the inner markers literal; YAGNI). It is applied ONLY when stdout
// is a TTY; piped output bypasses it entirely (byte-identical
// passthrough).
//
// Streaming-safe by construction: lines are the flush points — a line is
// buffered until its newline, and a construct left open there (or at a
// stream-ending Flush) is emitted as the literal source text, so half a
// `**` never reaches the terminal.
package render

import (
	"bytes"
	"io"
	"strings"
)

const (
	reset     = "\x1b[0m"
	bold      = "\x1b[1m"
	italic    = "\x1b[3m"
	dim       = "\x1b[2m"
	underline = "\x1b[4m"
)

// Renderer turns a markdown-ish byte stream into styled terminal output.
// It implements io.Writer; Write always consumes the full slice. Not safe
// for concurrent use (one stream, one goroutine — the SSE reader).
type Renderer struct {
	w       io.Writer
	buf     []byte // bytes of the current line not yet decided
	inFence bool   // inside a ``` block: verbatim until the closing fence
}

// New returns a Renderer writing styled output to w.
func New(w io.Writer) *Renderer { return &Renderer{w: w} }

// Write buffers p and emits every line that is complete (has its newline).
func (r *Renderer) Write(p []byte) (int, error) {
	r.buf = append(r.buf, p...)
	for {
		i := bytes.IndexByte(r.buf, '\n')
		if i < 0 {
			break
		}
		r.emitLine(string(r.buf[:i+1]))
		r.buf = r.buf[i+1:]
	}
	return len(p), nil
}

// Flush emits the buffered remainder (stream end): a construct still
// open there is literal text.
func (r *Renderer) Flush() {
	if len(r.buf) > 0 {
		r.emitLine(string(r.buf))
		r.buf = r.buf[:0]
	}
}

// emitLine styles one line; the trailing newline (if any) is preserved,
// never invented.
func (r *Renderer) emitLine(line string) {
	nl := ""
	if strings.HasSuffix(line, "\n") {
		nl = "\n"
		line = line[:len(line)-1]
	}
	if r.inFence {
		if strings.HasPrefix(line, "```") {
			r.inFence = false
		}
		io.WriteString(r.w, line+nl) // verbatim
		return
	}
	if strings.HasPrefix(line, "```") {
		r.inFence = true
		io.WriteString(r.w, "```"+dim+strings.TrimPrefix(line, "```")+reset+nl)
		return
	}
	io.WriteString(r.w, styleLine(line)+nl)
}

// styleLine applies the line-level constructs (bullet, header), then the
// inline ones.
func styleLine(s string) string {
	if strings.HasPrefix(s, "- ") || strings.HasPrefix(s, "* ") {
		return "• " + inline(s[2:])
	}
	for n := 3; n >= 1; n-- { // ### before # so "### x" is H3, not H1+"##"
		if strings.HasPrefix(s, strings.Repeat("#", n)+" ") {
			open := bold
			if n == 1 {
				open = bold + underline
			}
			return open + strings.Repeat("#", n) + " " + inline(s[n+1:]) + reset
		}
	}
	return inline(s)
}

// openers in match priority; at the same index "**" beats "*".
var openers = []string{"**", "`", "[", "*", "_"}

// inline rewrites the span constructs left to right. Anything that has
// no closing delimiter before the line's end stays literal — the line
// was the flush point.
func inline(s string) string {
	var b strings.Builder
	for s != "" {
		idx, kind := -1, ""
		for _, k := range openers {
			if i := strings.Index(s, k); i >= 0 && (idx < 0 || i < idx) {
				idx, kind = i, k
			}
		}
		if idx < 0 {
			b.WriteString(s)
			break
		}
		b.WriteString(s[:idx])
		rest := s[idx:]
		s = rest[len(kind):]
		switch kind {
		case "**":
			if end := strings.Index(rest[2:], "**"); end >= 0 {
				b.WriteString(bold + rest[2:2+end] + reset)
				s = rest[2+end+2:]
			} else {
				b.WriteString("**") // unclosed: literal
			}
		case "`":
			if end := strings.Index(rest[1:], "`"); end >= 0 {
				b.WriteString(dim + rest[1:1+end] + reset)
				s = rest[1+end+1:]
			} else {
				b.WriteString("`")
			}
		case "[":
			if end := strings.Index(rest, "]("); end >= 0 {
				if close := strings.Index(rest[end:], ")"); close >= 0 {
					text := rest[1:end]
					url := rest[end+2 : end+close]
					b.WriteString(inline(text) + dim + "(" + url + ")" + reset)
					s = rest[end+close+1:]
					continue
				}
			}
			b.WriteString("[") // unclosed: literal
		case "*":
			if end := strings.Index(rest[1:], "*"); end >= 0 {
				b.WriteString(italic + rest[1:1+end] + reset)
				s = rest[1+end+1:]
			} else {
				b.WriteString("*")
			}
		case "_":
			if end := strings.Index(rest[1:], "_"); end >= 0 {
				b.WriteString(italic + rest[1:1+end] + reset)
				s = rest[1+end+1:]
			} else {
				b.WriteString("_")
			}
		}
	}
	return b.String()
}
