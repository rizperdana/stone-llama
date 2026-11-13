// Package lineedit is a minimal raw-mode line editor for the stone-llama
// REPL: cursor movement, kill keys, and per-model history recall.
//
// Cursor math is rune-based (width-1 runes only — CJK width-2 is out of
// scope; a wide rune shifts the visual column but never corrupts the
// buffer). Non-TTY callers never enter raw mode; this package only reads
// bytes from an io.Reader and draws to an io.Writer, so tests feed byte
// sequences through a strings.Reader with no real terminal involved.
package lineedit

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"strings"
	"unicode/utf8"
)

// ErrInterrupted marks a line abandoned with Ctrl-C.
var ErrInterrupted = errors.New("lineedit: interrupted")

// maxHold bounds how many bytes an unterminated ESC sequence may swallow
// before it is dropped (runaway paste; ponytail ceiling — no legitimate
// sequence comes close).
const maxHold = 16

// Editor edits one prompt line at a time. Callers draw the initial prompt
// themselves (or accept the redraw's rendering); ReadLine rewrites the
// whole line on every keystroke.
type Editor struct {
	out    io.Writer
	prompt string
	buf    []rune
	pos    int
	hist   []string // most recent first
	hidx   int      // len(hist) = draft position
	draft  string
}

// New returns an Editor drawing redraws to out under prompt.
func New(out io.Writer, prompt string) *Editor {
	return &Editor{out: out, prompt: prompt}
}

// SetHistory sets the recall history for the Up/Down keys, most recent
// first. A recalled line never leaves the draft: Down past the newest
// entry restores whatever the user had typed.
func (e *Editor) SetHistory(h []string) {
	// stored oldest-first: Up decrements toward the newest entry.
	e.hist = make([]string, 0, len(h))
	for i := len(h) - 1; i >= 0; i-- {
		e.hist = append(e.hist, h[i])
	}
	e.hidx = len(e.hist)
}

// ReadLine reads one edited line from in. Enter returns the buffer;
// Ctrl-C returns ErrInterrupted (the ^C is already drawn); Ctrl-D on an
// empty line returns io.EOF. Raw-mode termios handling is the caller's
// job (see Raw) so that tests can drive an ordinary io.Reader.
func (e *Editor) ReadLine(in io.Reader) (string, error) {
	br, ok := in.(*bufio.Reader)
	if !ok {
		br = bufio.NewReader(in)
	}
	e.redraw()
	for {
		b, err := br.ReadByte()
		if err != nil {
			return "", err
		}
		again := true // reprocess a byte returned by escape (bare ESC + printable)
		for again {
			again = false
			switch b {
			case '\r', '\n':
				io.WriteString(e.out, "\r\n")
				line := string(e.buf)
				e.buf, e.pos = nil, 0 // fresh line next call (editors are reused)
				e.hidx = len(e.hist)
				return line, nil
			case 0x7f, 0x08: // Backspace / Ctrl-H
				if e.pos > 0 {
					e.buf = append(e.buf[:e.pos-1], e.buf[e.pos:]...)
					e.pos--
				}
			case 0x01: // Ctrl+A: start of line
				e.pos = 0
			case 0x05: // Ctrl+E: end of line
				e.pos = len(e.buf)
			case 0x0b: // Ctrl+K: kill to end of line
				e.buf = e.buf[:e.pos]
			case 0x15: // Ctrl+U: kill the whole line
				e.buf, e.pos = nil, 0
			case 0x03: // Ctrl+C: abandon the line, stay in the REPL
				io.WriteString(e.out, "^C\r\n")
				e.buf, e.pos = nil, 0
				e.hidx = len(e.hist)
				return "", ErrInterrupted
			case 0x04: // Ctrl+D: exit only on an empty line
				if len(e.buf) == 0 {
					return "", io.EOF
				}
			case 0x09: // Tab: swallowed (prompts are single-line text)
			case 0x1b: // ESC sequences: arrows, Delete, Home/End
				if rb := e.escape(br); rb != 0 {
					b = rb // bare ESC before a printable: reprocess it
					again = true
				}
			default:
				if b < 0x20 { // other control bytes: swallowed
					break
				}
				if r := readRune(br, b); r != utf8.RuneError {
					e.insertRune(r)
				}
			}
		}
		e.redraw()
	}
}

// insertRune splices r in at the cursor.
func (e *Editor) insertRune(r rune) {
	e.buf = append(e.buf[:e.pos], append([]rune{r}, e.buf[e.pos:]...)...)
	e.pos++
}

// readRune assembles one UTF-8 rune from lead plus continuation bytes.
// Invalid sequences return utf8.RuneError and are swallowed by the caller
// — a half-decoded rune must never land in the buffer.
func readRune(br *bufio.Reader, lead byte) rune {
	var n int
	switch {
	case lead < 0x80:
		return rune(lead)
	case lead&0xE0 == 0xC0:
		n = 1
	case lead&0xF0 == 0xE0:
		n = 2
	case lead&0xF8 == 0xF0:
		n = 3
	default:
		return utf8.RuneError
	}
	b := make([]byte, 1, 4)
	b[0] = lead
	for range n {
		c, err := br.ReadByte()
		if err != nil {
			return utf8.RuneError
		}
		b = append(b, c)
	}
	r, _ := utf8.DecodeRune(b)
	return r
}

// escape parses one ESC sequence. Unknown or malformed sequences swallow
// their remainder — bytes never leak into the buffer. A bare ESC before
// a printable returns that byte for reprocessing (Alt+key arrives as
// ESC+byte and degrades to the plain key; the ESC itself is dropped).
func (e *Editor) escape(br *bufio.Reader) byte {
	b, err := br.ReadByte()
	if err != nil {
		return 0
	}
	if b != '[' && b != 'O' {
		if b >= 0x20 {
			return b // bare ESC + printable: reprocess the printable
		}
		return 0
	}
	var seq []byte
	for {
		c, err := br.ReadByte()
		if err != nil {
			return 0
		}
		seq = append(seq, c)
		if c >= 0x40 && c <= 0x7e {
			break // final byte
		}
		if len(seq) > maxHold {
			return 0 // runaway: drop, never insert
		}
	}
	final := seq[len(seq)-1]
	param := strings.TrimSuffix(string(seq[:len(seq)-1]), ";")
	switch final {
	case 'D': // Left
		if e.pos > 0 {
			e.pos--
		}
	case 'C': // Right
		if e.pos < len(e.buf) {
			e.pos++
		}
	case 'H': // Home (CSI H)
		e.pos = 0
	case 'F': // End (CSI F)
		e.pos = len(e.buf)
	case 'A': // Up: older history
		e.recall(1)
	case 'B': // Down: newer history, back to the draft
		e.recall(-1)
	case '~': // Application-mode Home/End and Delete
		switch param {
		case "1", "7":
			e.pos = 0
		case "3":
			if e.pos < len(e.buf) {
				e.buf = append(e.buf[:e.pos], e.buf[e.pos+1:]...)
			}
		case "4", "8":
			e.pos = len(e.buf)
		}
	}
	return 0
}

// recall steps through history: dir > 0 older (Up), dir < 0 newer (Down).
func (e *Editor) recall(dir int) {
	if len(e.hist) == 0 {
		return
	}
	if dir > 0 {
		if e.hidx == len(e.hist) {
			e.draft = string(e.buf) // stash the draft once
		}
		if e.hidx > 0 {
			e.hidx--
			e.setBuf(e.hist[e.hidx])
		}
		return
	}
	if e.hidx < len(e.hist) {
		e.hidx++
		if e.hidx == len(e.hist) {
			e.setBuf(e.draft)
		} else {
			e.setBuf(e.hist[e.hidx])
		}
	}
}

func (e *Editor) setBuf(s string) {
	e.buf = []rune(s)
	e.pos = len(e.buf)
}

// redraw rewrites the prompt line in place: clear to end of line, print
// prompt + buffer, park the cursor.
func (e *Editor) redraw() {
	fmt.Fprintf(e.out, "\r\x1b[K%s%s", e.prompt, string(e.buf))
	if e.pos < len(e.buf) {
		fmt.Fprintf(e.out, "\r\x1b[%dC", utf8.RuneCountInString(e.prompt)+e.pos)
	}
}
