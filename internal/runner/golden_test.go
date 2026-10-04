package runner

// Oracle-derived differential tests: every expected string in
// testdata/templates.json was produced by transformers 5.13.1's
// apply_chat_template (see testdata/gen_golden.py for the exact command).
// These tests run without Python — the fixture is the artifact.

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

type goldenFixture struct {
	Meta struct {
		FixtureVersion int    `json:"fixture_version"`
		Command        string `json:"command"`
		Transformers   string `json:"transformers"`
		Jinja2         string `json:"jinja2"`
		FixedClock     string `json:"fixed_clock"`
	} `json:"meta"`
	Entries []goldenEntry `json:"entries"`
}

type goldenEntry struct {
	Name     string       `json:"name"`
	Kind     string       `json:"kind"`
	Template string       `json:"template"`
	Cases    []goldenCase `json:"cases"`
	Skipped  []struct {
		Case  string `json:"case"`
		Error string `json:"error"`
	} `json:"skipped"`
}

type goldenCase struct {
	Name     string          `json:"name"`
	Vars     json.RawMessage `json:"vars"`
	Expected string          `json:"expected"`
}

func loadFixture(t *testing.T) goldenFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/templates.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx goldenFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	return fx
}

// fixedClock is the fixture's strftime_now override (2026-01-01 12:00 naive),
// mirroring gen_golden.py's FIXED datetime so results are date-independent.
var fixedClock = time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)

// testNow formats fixedClock the way Python's datetime.strftime would for the
// directive set this package supports. Hand-derived, not oracle-derived.
// Naive datetime: %z is the empty string, as in Python.
func testNow(format string) (string, error) {
	var b strings.Builder
	for i := 0; i < len(format); i++ {
		if format[i] != '%' {
			b.WriteByte(format[i])
			continue
		}
		i++
		if i >= len(format) {
			b.WriteByte('%')
			break
		}
		switch format[i] {
		case '%':
			b.WriteByte('%')
		case 'd':
			b.WriteString(fixedClock.Format("02"))
		case 'e':
			b.WriteString(fixedClock.Format("_2"))
		case 'm':
			b.WriteString(fixedClock.Format("01"))
		case 'B':
			b.WriteString(fixedClock.Format("January"))
		case 'b':
			b.WriteString(fixedClock.Format("Jan"))
		case 'A':
			b.WriteString(fixedClock.Format("Monday"))
		case 'a':
			b.WriteString(fixedClock.Format("Mon"))
		case 'Y':
			b.WriteString(fixedClock.Format("2006"))
		case 'y':
			b.WriteString(fixedClock.Format("06"))
		case 'H':
			b.WriteString(fixedClock.Format("15"))
		case 'I':
			b.WriteString(fixedClock.Format("03"))
		case 'M':
			b.WriteString(fixedClock.Format("04"))
		case 'S':
			b.WriteString(fixedClock.Format("05"))
		case 'p':
			b.WriteString(fixedClock.Format("PM"))
		case 'z':
			// naive datetime: Python's %z is empty
		default:
			return "", fmt.Errorf("testNow: unsupported directive %%%c", format[i])
		}
	}
	return b.String(), nil
}

// varsFrom decodes one case's vars with ParseJSON so nested objects keep
// Python's key order (repr/tojson depend on it), then exposes the top level
// as the map Render expects.
func varsFrom(t *testing.T, raw json.RawMessage) map[string]any {
	t.Helper()
	v, err := ParseJSON(raw)
	if err != nil {
		t.Fatalf("parse vars: %v", err)
	}
	switch d := v.(type) {
	case map[string]any:
		return d
	case Dict:
		out := make(map[string]any, len(d.Keys))
		for _, k := range d.Keys {
			out[k] = d.Vals[k]
		}
		return out
	default:
		t.Fatalf("vars top level is %T, want object", v)
		return nil
	}
}

func firstDiff(a, b string) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := range n {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

// TestGoldenAgainstTransformers renders every fixture case and demands
// byte-equality with transformers 5.13.1's apply_chat_template output. A
// divergence is a finding: fix the engine or record it — never relax this.
func TestGoldenAgainstTransformers(t *testing.T) {
	fx := loadFixture(t)
	if fx.Meta.Transformers != "5.13.1" {
		t.Fatalf("fixture was generated against transformers %s, want 5.13.1 — "+
			"regenerate with %s and re-verify", fx.Meta.Transformers, fx.Meta.Command)
	}
	if fx.Meta.FixedClock == "" {
		t.Fatal("fixture does not record a fixed clock; renders would be date-dependent")
	}
	for _, e := range fx.Entries {
		t.Run(e.Name, func(t *testing.T) {
			tpl, err := ParseTemplate(e.Template)
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			for _, c := range e.Cases {
				t.Run(c.Name, func(t *testing.T) {
					got, err := tpl.Render(varsFrom(t, c.Vars), testNow)
					if err != nil {
						t.Fatalf("render: %v", err)
					}
					if got != c.Expected {
						t.Errorf("divergence from transformers at byte %d\n got: %q\nwant: %q",
							firstDiff(got, c.Expected), got, c.Expected)
					}
				})
			}
			for _, s := range e.Skipped {
				t.Logf("oracle rejected this input (not compared): %s: %s", s.Case, s.Error)
			}
		})
	}
}
