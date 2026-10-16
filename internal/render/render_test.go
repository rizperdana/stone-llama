package render

import (
	"bytes"
	"strings"
	"testing"
)

// feed streams text through a Renderer in the given chunk split and
// returns what reached the terminal after a Flush.
func feed(t *testing.T, chunks ...string) string {
	t.Helper()
	var out bytes.Buffer
	r := New(&out)
	for _, c := range chunks {
		if _, err := r.Write([]byte(c)); err != nil {
			t.Fatal(err)
		}
	}
	r.Flush()
	return out.String()
}

// Bold spanning several writes must never leak half a delimiter: the ANSI
// bold appears only once ** closes, and the raw ** never shows.
func TestBoldMidStream(t *testing.T) {
	got := feed(t, "**he", "llo** world\n")
	if got != "\x1b[1mhello\x1b[0m world\n" {
		t.Errorf("got %q", got)
	}
}

// A stream that ENDS inside an unclosed ** emits the literal text.
func TestUnclosedBoldAtEOF(t *testing.T) {
	got := feed(t, "a **b")
	if got != "a **b" {
		t.Errorf("got %q, want the literal text", got)
	}
}

// A newline is a flush point: ** that never closes before it stays literal.
func TestUnclosedBoldAtNewline(t *testing.T) {
	got := feed(t, "a **b\nnext\n")
	if got != "a **b\nnext\n" {
		t.Errorf("got %q, want literal lines", got)
	}
}

// Fenced code blocks render verbatim; only the language name on the
// opening line is dimmed. No syntax highlighting (YAGNI).
func TestFencedCodeBlock(t *testing.T) {
	got := feed(t, "```go\nfmt.Println(\"**not bold**\")\n```\n")
	want := "```" + "\x1b[2mgo\x1b[0m\nfmt.Println(\"**not bold**\")\n```\n"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

// A fence opened but never closed still passes its body through verbatim
// (stream ended inside the fence).
func TestUnclosedFenceAtEOF(t *testing.T) {
	got := feed(t, "```\nliteral **line**\n")
	if !strings.Contains(got, "literal **line**") {
		t.Errorf("fence body mangled: %q", got)
	}
}

// Headers: H1 bold+underline, H2/H3 bold; the hash marks stay visible.
func TestHeaders(t *testing.T) {
	got := feed(t, "# Title\n## Sub\n### Small\n")
	want := "\x1b[1m\x1b[4m# Title\x1b[0m\n\x1b[1m## Sub\x1b[0m\n\x1b[1m### Small\x1b[0m\n"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

// Inline styles: *italic*, _italic_, `code`, and bullets — numbered lists
// stay untouched.
func TestInlineAndBullets(t *testing.T) {
	got := feed(t, "- item\n1. numbered\n*em* _it_ `c`\n")
	want := "• item\n1. numbered\n\x1b[3mem\x1b[0m \x1b[3mit\x1b[0m \x1b[2mc\x1b[0m\n"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

// Links: [text](url) renders the text; the url stays, dimmed in parens.
func TestLink(t *testing.T) {
	got := feed(t, "see [docs](https://x.io) now\n")
	want := "see docs\x1b[2m(https://x.io)\x1b[0m now\n"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

// An unmatched link opener is literal, never a dropped bracket.
func TestUnclosedLinkLiteral(t *testing.T) {
	got := feed(t, "see [docs](\n")
	if got != "see [docs](\n" {
		t.Errorf("got %q", got)
	}
}

// Streaming flush point: a complete line renders as soon as its newline
// arrives — before the stream ends.
func TestLineRendersBeforeStreamEnd(t *testing.T) {
	var out bytes.Buffer
	r := New(&out)
	r.Write([]byte("**bold** line\nsecond"))
	if !strings.Contains(out.String(), "\x1b[1mbold\x1b[0m") {
		t.Errorf("first line not flushed at its newline: %q", out.String())
	}
	r.Flush()
	if !strings.Contains(out.String(), "second") {
		t.Errorf("flush lost the tail: %q", out.String())
	}
}

// Write must always accept the full byte slice (io.Writer contract).
func TestWriteFullLen(t *testing.T) {
	var out bytes.Buffer
	r := New(&out)
	n, err := r.Write([]byte("**x**\n"))
	if err != nil || n != 6 {
		t.Errorf("Write = %d, %v; want 6, nil", n, err)
	}
}
