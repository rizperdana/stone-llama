package lineedit

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"
)

// feed runs keys through an Editor against a bytes.Reader stand-in for the
// terminal and returns the accepted line (or error) plus everything the
// editor drew.
func feed(t *testing.T, keys string, hist []string) (string, string, error) {
	t.Helper()
	var out bytes.Buffer
	e := New(&out, ">>> ")
	e.SetHistory(hist)
	line, err := e.ReadLine(strings.NewReader(keys))
	return line, out.String(), err
}

// Arrow-left then insert: one Left moves between b and c, the X lands
// there — rune-cursor movement, not byte appends.
func TestArrowLeftInsertAtCursor(t *testing.T) {
	line, out, err := feed(t, "abc\x1b[D"+"X\r", nil)
	if err != nil {
		t.Fatal(err)
	}
	if line != "abXc" {
		t.Errorf("line = %q, want %q", line, "abXc")
	}
	if !strings.Contains(out, "\x1b[6C") {
		t.Errorf("redraw never repositioned the cursor (CSI C): %q", out)
	}
}

// Backspace deletes the rune before the cursor, mid-line.
func TestBackspaceMidLine(t *testing.T) {
	line, _, err := feed(t, "hello\x7f\x7fx\r", nil)
	if err != nil {
		t.Fatal(err)
	}
	if line != "helx" {
		t.Errorf("line = %q, want %q", line, "helx")
	}
}

// Backspace after multi-byte UTF-8 must delete the whole rune — a byte-wise
// delete would leave a stray continuation byte.
func TestUTF8BackspaceDeletesWholeRune(t *testing.T) {
	// "aé" then backspace: é is U+00E9 (2 bytes).
	line, _, err := feed(t, "a\xc3\xa9\x7f\r", nil)
	if err != nil {
		t.Fatal(err)
	}
	if line != "a" {
		t.Errorf("line = %q, want %q (full é deleted)", line, "a")
	}
}

// Up-arrow walks this model's history most-recent-first; Down returns to
// the just-edited draft line.
func TestUpArrowHistoryRecallAndDraftReturn(t *testing.T) {
	line, _, err := feed(t, "draft\x1b[A\x1b[A\x1b[B\x1b[B\r", []string{"second", "first"})
	if err != nil {
		t.Fatal(err)
	}
	if line != "draft" {
		t.Errorf("line = %q, want the untouched draft %q", line, "draft")
	}
	line, _, err = feed(t, "x\x1b[A\r", []string{"second", "first"})
	if err != nil {
		t.Fatal(err)
	}
	if line != "second" {
		t.Errorf("line = %q, want most recent history entry %q", line, "second")
	}
}

// Unknown/malformed CSI sequences are swallowed — never inserted.
func TestEscapeGarbageSwallowed(t *testing.T) {
	for _, keys := range []string{
		"ab\x1b[15;3ac\r", // unknown final byte 'a'
		"ab\x1b[99~c\r",   // unknown ~ command, then a printable
	} {
		line, _, err := feed(t, keys, nil)
		if err != nil {
			t.Fatalf("%q: %v", keys, err)
		}
		if line != "abc" {
			t.Errorf("%q: line = %q, want %q", keys, line, "abc")
		}
	}
}

// A bare ESC before a printable degrades to the plain key (Alt+key):
// the ESC itself never lands in the buffer, the key does.
func TestBareEscDegradesToPlainKey(t *testing.T) {
	line, _, err := feed(t, "ab\x1bZc\r", nil)
	if err != nil {
		t.Fatal(err)
	}
	if line != "abZc" {
		t.Errorf("line = %q, want %q", line, "abZc")
	}
}

// Delete/Home/End and the Ctrl kill/home/end keys.
func TestDeleteHomeEndAndKillKeys(t *testing.T) {
	line, _, err := feed(t, "abc\x1b[H\x1b[3~\r", nil) // Home, Delete → drop 'a'
	if err != nil {
		t.Fatal(err)
	}
	if line != "bc" {
		t.Errorf("delete-at-home: line = %q, want %q", line, "bc")
	}
	line, _, err = feed(t, "abc\x1b[F\x01\x05Z\r", nil) // Ctrl-A, Ctrl-E, insert
	if err != nil {
		t.Fatal(err)
	}
	if line != "abcZ" {
		t.Errorf("ctrl-a/e: line = %q, want %q", line, "abcZ")
	}
	line, _, err = feed(t, "abcd\x1b[D\x1b[D\x0b\r", nil) // Ctrl-K: kill to end
	if err != nil {
		t.Fatal(err)
	}
	if line != "ab" {
		t.Errorf("ctrl-k: line = %q, want %q", line, "ab")
	}
	line, _, err = feed(t, "abcd\x01\x15X\r", nil) // Ctrl-U: kill line
	if err != nil {
		t.Fatal(err)
	}
	if line != "X" {
		t.Errorf("ctrl-u: line = %q, want %q", line, "X")
	}
}

// Ctrl-C abandons the line with ErrInterrupted and a visible ^C.
func TestCtrlCAbandonsLine(t *testing.T) {
	line, out, err := feed(t, "abc\x03", nil)
	if !errors.Is(err, ErrInterrupted) {
		t.Fatalf("err = %v, want ErrInterrupted", err)
	}
	if line != "" {
		t.Errorf("line = %q, want empty", line)
	}
	if !strings.Contains(out, "^C") {
		t.Errorf("no visible ^C: %q", out)
	}
}

// Ctrl-D on an empty line exits; on a non-empty line it is inert.
func TestCtrlDEmptyLineExits(t *testing.T) {
	_, _, err := feed(t, "\x04", nil)
	if !errors.Is(err, io.EOF) {
		t.Fatalf("err = %v, want io.EOF", err)
	}
	line, _, err := feed(t, "ab\x04\r", nil)
	if err != nil || line != "ab" {
		t.Errorf("ctrl-d mid-line: line=%q err=%v, want %q/nil", line, err, "ab")
	}
}

// Plain enter on an empty line returns an empty line, no error (the REPL
// keeps its "empty enter skips" behaviour).
func TestEmptyEnter(t *testing.T) {
	line, _, err := feed(t, "\r", nil)
	if err != nil || line != "" {
		t.Errorf("line=%q err=%v, want empty/nil", line, err)
	}
}
