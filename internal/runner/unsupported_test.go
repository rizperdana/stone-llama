package runner

// Anything outside the documented subset (see the package doc in template.go)
// must fail LOUDLY — a parse or runtime error naming the construct — never
// render an approximation. Enumerated here: unknown tags, arithmetic
// operators * / % //, tuple unpacking in for/set targets, literal shapes the
// parser does not cover (list/dict/triple-quoted), unknown filters and tests,
// and runtime errors for undefined attributes, mixed comparisons,
// non-container membership, raise_exception, and attribute-set on
// non-namespaces.

import (
	"strings"
	"testing"
)

func TestUnsupportedFeaturesFailLoudly(t *testing.T) {
	cases := []struct{ name, src, wantSub string }{
		// Unknown tags, nested in {% if %} so the parser reports the
		// construct instead of stopping at end-of-template.
		{"macro tag", "{% if true %}{% macro m() %}{% endmacro %}{% endif %}", `unsupported tag "macro"`},
		{"block tag", "{% if true %}{% block x %}{% endblock %}{% endif %}", `unsupported tag "block"`},
		{"filter tag", "{% if true %}{% filter upper %}{% endfilter %}{% endif %}", `unsupported tag "filter"`},
		{"call tag", "{% if true %}{% call f() %}{% endcall %}{% endif %}", `unsupported tag "call"`},
		{"with tag", "{% if true %}{% with x=1 %}{% endwith %}{% endif %}", `unsupported tag "with"`},
		{"include tag", `{% if true %}{% include "x" %}{% endif %}`, `unsupported tag "include"`},
		// Arithmetic operators outside the subset.
		{"multiply", "{{ 1 * 2 }}", "unsupported operator"},
		{"divide", "{{ 7 / 2 }}", "unsupported operator"},
		{"modulo", "{{ 7 % 2 }}", "unsupported operator"},
		{"floordiv", "{{ 7 // 2 }}", "unsupported operator"},
		// Unknown filter / test.
		{"unknown filter", "{{ 1 | default(2) }}", `unsupported filter "default"`},
		{"unknown test", "{{ 1 is even }}", `unsupported test "even"`},
		// Tuple unpacking is silently wrong unless rejected.
		{"tuple for-target", "{% for a, b in pairs %}{% endfor %}", "unsupported for-target"},
		{"tuple set-target", "{% set a, b = 1, 2 %}", "unsupported set-target"},
		// Literal shapes outside the subset.
		{"triple-quoted string", `{{ """x""" }}`, "triple-quoted strings are unsupported"},
		{"list literal", "{{ [1, 2] }}", `unexpected "["`},
		{"dict literal", "{{ {\"k\": 1} }}", `unexpected "{"`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := renderNow(t, c.src, nil)
			if err == nil {
				t.Fatalf("%q: want error containing %q, got none", c.src, c.wantSub)
			}
			if !strings.Contains(err.Error(), c.wantSub) {
				t.Fatalf("%q: error %q does not contain %q", c.src, err, c.wantSub)
			}
		})
	}
}

func TestRuntimeErrorsAreLoud(t *testing.T) {
	cases := []struct{ name, src, wantSub string }{
		{"raise_exception", `{{ raise_exception("boom") }}`, "boom"},
		{"mixed ordered compare", `{{ 1 < "a" }}`, "unsupported operand types"},
		{"add string and int", `{{ "a" + 1 }}`, "unsupported operand types"},
		{"membership on non-container", `{{ 2 in 3 }}`, "not a container"},
		{"attribute of undefined", `{{ missing.deep }}`, "undefined"},
		{"set attr on non-namespace", `{% set s = "txt" %}{% set s.a = 1 %}`, "cannot set"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := renderNow(t, c.src, nil)
			if err == nil {
				t.Fatalf("%q: want error containing %q, got render %q", c.src, c.wantSub, got)
			}
			if !strings.Contains(err.Error(), c.wantSub) {
				t.Fatalf("%q: error %q does not contain %q", c.src, err, c.wantSub)
			}
		})
	}
}
