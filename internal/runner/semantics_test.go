package runner

// Hand-derived unit tests for the repaired semantics: and/or operand values,
// ==/!= via eq, ordered comparisons, in/not in, ~ concat, +/- preserving
// int64, the setNode local/namespace-attribute behaviour, and tojson with
// ensure_ascii=False + insertion order. Expected strings were computed from
// Python/jinja semantics BY HAND — oracle-derived cases live in
// golden_test.go; nothing here claims transformers verification.

import (
	"encoding/json"
	"strings"
	"testing"
)

func renderNow(t *testing.T, src string, vars map[string]any) (string, error) {
	t.Helper()
	tpl, err := ParseTemplate(src)
	if err != nil {
		return "", err
	}
	return tpl.Render(vars, testNow)
}

func expectRender(t *testing.T, name, src, want string, vars map[string]any) {
	t.Helper()
	got, err := renderNow(t, src, vars)
	if err != nil {
		t.Fatalf("%s: render %q: %v", name, src, err)
	}
	if got != want {
		t.Errorf("%s: %q\n got %q\nwant %q", name, src, got, want)
	}
}

func expectRenderErr(t *testing.T, name, src, wantSub string, vars map[string]any) {
	t.Helper()
	got, err := renderNow(t, src, vars)
	if err == nil {
		t.Errorf("%s: %q rendered %q, want error containing %q", name, src, got, wantSub)
		return
	}
	if !strings.Contains(err.Error(), wantSub) {
		t.Errorf("%s: error %q does not contain %q", name, err, wantSub)
	}
}

// and/or must return the operand value, not a bool: a bool-returning
// implementation would render "True"/"False" where Python renders "2"/"0"/"".
func TestAndOrReturnOperandValues(t *testing.T) {
	cases := []struct{ src, want string }{
		{`{{ 1 and 2 }}`, "2"},
		{`{{ 3 and 4 }}`, "4"},
		{`{{ 0 and 4 }}`, "0"},
		{`{{ "" and "x" }}`, ""},
		{`{{ "a" or 2 }}`, "a"},
		{`{{ 0 or "z" }}`, "z"},
	}
	for _, c := range cases {
		expectRender(t, "and_or", c.src, c.want, nil)
	}
}

func TestEqualitySemantics(t *testing.T) {
	cases := []struct{ src, want string }{
		{`{{ 1 == 1 }}`, "True"},
		{`{{ 1 == 1.0 }}`, "True"},  // int/float cross-type, as in Python
		{`{{ true == 1 }}`, "True"}, // bool participates as 1
		{`{{ "1" == 1 }}`, "False"}, // a string never equals a number
		{`{{ none == 0 }}`, "False"},
		{`{{ 1 != 2 }}`, "True"},
		{`{{ "a" != "a" }}`, "False"},
	}
	for _, c := range cases {
		expectRender(t, "eq", c.src, c.want, nil)
	}
}

func TestOrderedComparisons(t *testing.T) {
	cases := []struct{ src, want string }{
		{`{{ 1 < 2 }}`, "True"},
		{`{{ 2 <= 2 }}`, "True"},
		{`{{ "b" > "a" }}`, "True"},
		{`{{ 3 >= 4 }}`, "False"},
		{`{{ 1.5 < 2 }}`, "True"},
	}
	for _, c := range cases {
		expectRender(t, "ord", c.src, c.want, nil)
	}
	// Mixed-type ordering is an error in Python; we must not guess.
	expectRenderErr(t, "ord_mixed", `{{ 1 < "a" }}`, "unsupported operand types", nil)
}

func TestMembership(t *testing.T) {
	vars := varsFrom(t, json.RawMessage(`{"lst": [1, 2, 3], "obj": {"k": 1}}`))
	cases := []struct{ src, want string }{
		{`{{ "b" in "abc" }}`, "True"},
		{`{{ 2 in lst }}`, "True"},
		{`{{ "k" in obj }}`, "True"},
		{`{{ "x" not in "abc" }}`, "True"},
		{`{{ 4 in lst }}`, "False"},
	}
	for _, c := range cases {
		expectRender(t, "in", c.src, c.want, vars)
	}
	expectRenderErr(t, "in_non_container", `{{ 2 in 3 }}`, "not a container", nil)
}

func TestConcatTilde(t *testing.T) {
	cases := []struct{ src, want string }{
		{`{{ "x" ~ 1 ~ true ~ none }}`, "x1TrueNone"},
		{`{{ "v=" ~ 1.5 }}`, "v=1.5"},
	}
	for _, c := range cases {
		expectRender(t, "concat", c.src, c.want, nil)
	}
}

// int64 + int64 must stay int64: a float leak renders "3.0"/"-3.0".
func TestArithPreservesInt64(t *testing.T) {
	cases := []struct{ src, want string }{
		{`{{ 1 + 2 }}`, "3"},
		{`{{ 2 - 5 }}`, "-3"},
		{`{{ 1 + 1.5 }}`, "2.5"},
		{`{{ 2 - 0.5 }}`, "1.5"},
	}
	for _, c := range cases {
		expectRender(t, "arith", c.src, c.want, nil)
	}
	expectRenderErr(t, "arith_string", `{{ "a" + 1 }}`, "unsupported operand types", nil)
}

// TestSetLocalAndNamespace covers the setNode repair: plain names assign in
// the frame, dotted targets assign the leaf onto a namespace container.
func TestSetLocalAndNamespace(t *testing.T) {
	expectRender(t, "set_local", `{% set x = 1 %}{% set x = x + 1 %}{{ x }}`, "2", nil)
	expectRender(t, "set_ns_attr",
		`{% set ns = namespace(v=1) %}{% set ns.v = ns.v + 1 %}{{ ns.v }}`, "2", nil)
	expectRender(t, "set_ns_deep",
		`{% set ns = namespace(inner=namespace(w=0)) %}{% set ns.inner.w = 5 %}{{ ns.inner.w }}`,
		"5", nil)
	// The dotted form must not also create a local variable named after the leaf.
	expectRender(t, "set_ns_no_local_leak",
		`{% set ns = namespace(v=1) %}{% set ns.v = 7 %}{{ v is defined }}`, "False", nil)
	expectRenderErr(t, "set_attr_non_namespace",
		`{% set s = "txt" %}{% set s.a = 1 %}`, "cannot set", nil)
}

// tojson mirrors json.dumps(ensure_ascii=False, sort_keys=False, default
// separators): insertion order, raw UTF-8, ", " / ": ".
func TestTojsonOrderAndUTF8(t *testing.T) {
	vars := varsFrom(t,
		json.RawMessage(`{"obj": {"b": 1, "a": "ü", "n": [1, 2], "s": "a'b"}}`))
	expectRender(t, "tojson", `{{ obj | tojson }}`,
		`{"b": 1, "a": "ü", "n": [1, 2], "s": "a'b"}`, vars)
}
