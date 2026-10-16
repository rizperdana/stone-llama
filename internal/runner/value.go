package runner

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"reflect"
	"strconv"
	"strings"
)

// --- value model ------------------------------------------------------------
//
// Template values mirror what Python/jinja2 sees: nil is None, int64/float64
// keep Python's int/float distinction, undefined is the jinja Undefined, and
// Dict preserves insertion key order because tojson is json.dumps with
// sort_keys=False — bytes must match what today's serving produces.

// undefined is jinja's Undefined: falsy, renders as "", errors when an
// attribute or call demands a real value. name feeds error messages.
type undefined struct{ name string }

// Dict is an object with insertion key order preserved (json.loads order).
// Order-sensitive operations (tojson, repr, iteration) reject plain
// map[string]any instead of silently sorting: unsorted output would not match
// the prompt bytes today's serving produces.
type Dict struct {
	Keys []string
	Vals map[string]any
}

// Namespace implements jinja's namespace(): a mutable attribute object.
type Namespace struct {
	Vals map[string]any
}

type loopInfo struct {
	index0, length int64
}

type boundMethod struct {
	recv any
	name string
}

type builtinFunc struct{ name string }

func isUndefined(v any) bool {
	_, ok := v.(undefined)
	return ok
}

// --- ordered JSON -----------------------------------------------------------

// ParseJSON decodes a JSON value, preserving object key order in Dict. This is
// how runner feeds OpenAI request bodies into templates: python's json.loads
// preserves order, so we must too for byte-identical prompts.
func ParseJSON(data []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	v, err := parseJSONValue(dec)
	if err != nil {
		return nil, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, fmt.Errorf("trailing data after JSON value")
	}
	return v, nil
}

func parseJSONValue(dec *json.Decoder) (any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	return parseJSONToken(dec, tok)
}

func parseJSONToken(dec *json.Decoder, tok json.Token) (any, error) {
	switch t := tok.(type) {
	case json.Delim:
		switch t {
		case '{':
			d := Dict{Vals: map[string]any{}}
			for dec.More() {
				kt, err := dec.Token()
				if err != nil {
					return nil, err
				}
				key, ok := kt.(string)
				if !ok {
					return nil, fmt.Errorf("non-string object key in JSON")
				}
				v, err := parseJSONValue(dec)
				if err != nil {
					return nil, err
				}
				// python keeps the first insertion position for duplicate keys.
				if _, exists := d.Vals[key]; !exists {
					d.Keys = append(d.Keys, key)
				}
				d.Vals[key] = v
			}
			if _, err := dec.Token(); err != nil {
				return nil, err
			}
			return d, nil
		case '[':
			var out []any
			for dec.More() {
				v, err := parseJSONValue(dec)
				if err != nil {
					return nil, err
				}
				out = append(out, v)
			}
			if _, err := dec.Token(); err != nil {
				return nil, err
			}
			return out, nil
		default:
			return nil, fmt.Errorf("unexpected JSON delimiter %v", t)
		}
	case string, bool, nil:
		return t, nil
	case json.Number:
		if n, err := t.Int64(); err == nil {
			return n, nil
		}
		f, err := t.Float64()
		if err != nil {
			return nil, fmt.Errorf("invalid JSON number %s", t)
		}
		return f, nil
	default:
		return nil, fmt.Errorf("unexpected JSON token %v", t)
	}
}

// normalize converts caller-supplied context values to the template value
// model. Plain map[string]any is accepted but untagged: order-sensitive
// operations reject it (see Dict).
func normalize(v any) (any, error) {
	switch t := v.(type) {
	case nil:
		return nil, nil
	case bool, string, float64, int64, undefined, *Namespace, loopInfo, boundMethod, builtinFunc:
		return t, nil
	case int:
		return int64(t), nil
	case int32:
		return int64(t), nil
	case float32:
		return float64(t), nil
	case []any:
		out := make([]any, len(t))
		for i := range t {
			n, err := normalize(t[i])
			if err != nil {
				return nil, err
			}
			out[i] = n
		}
		return out, nil
	case map[string]any:
		out := make(map[string]any, len(t))
		for k, val := range t {
			n, err := normalize(val)
			if err != nil {
				return nil, err
			}
			out[k] = n
		}
		return out, nil
	case Dict:
		out := Dict{Keys: append([]string(nil), t.Keys...), Vals: make(map[string]any, len(t.Vals))}
		for k, val := range t.Vals {
			n, err := normalize(val)
			if err != nil {
				return nil, err
			}
			out.Vals[k] = n
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported context value of type %T", v)
	}
}

// --- truthiness, equality, membership ---------------------------------------

func truthy(v any) bool {
	switch t := v.(type) {
	case undefined:
		return false
	case nil:
		return false
	case bool:
		return t
	case int64:
		return t != 0
	case float64:
		return t != 0
	case string:
		return t != ""
	case []any:
		return len(t) > 0
	case Dict:
		return len(t.Keys) > 0
	case map[string]any:
		return len(t) > 0
	default:
		return true
	}
}

func asNumber(v any) (float64, bool) {
	switch t := v.(type) {
	case bool:
		if t {
			return 1, true
		}
		return 0, true
	case int64:
		return float64(t), true
	case float64:
		return t, true
	}
	return 0, false
}

// eq mirrors Python == for the values templates compare. Any comparison with
// Undefined is False unless both sides are Undefined.
func eq(a, b any) bool {
	_, aIsU := a.(undefined)
	_, bIsU := b.(undefined)
	if aIsU || bIsU {
		return aIsU && bIsU
	}
	if an, ok := asNumber(a); ok {
		if bn, ok2 := asNumber(b); ok2 {
			return an == bn
		}
		return false
	}
	switch at := a.(type) {
	case nil:
		return b == nil
	case bool:
		bt, ok := b.(bool)
		return ok && at == bt
	case string:
		bt, ok := b.(string)
		return ok && at == bt
	}
	return reflect.DeepEqual(a, b)
}

// ord orders comparable pairs like Python; anything else is an error.
func ord(op string, a, b any) (bool, error) {
	if an, ok := asNumber(a); ok {
		if bn, ok2 := asNumber(b); ok2 {
			return compareOrdered(op, an, bn), nil
		}
	}
	if as, ok := a.(string); ok {
		if bs, ok2 := b.(string); ok2 {
			return compareOrdered(op, as, bs), nil
		}
	}
	return false, fmt.Errorf("unsupported operand types for %s: %T and %T", op, a, b)
}

func compareOrdered[T float64 | string](op string, a, b T) bool {
	switch op {
	case "<":
		return a < b
	case "<=":
		return a <= b
	case ">":
		return a > b
	case ">=":
		return a >= b
	}
	return false
}

// contains implements `x in y`. Membership in Undefined is False (jinja renders
// such conditions without raising).
func contains(haystack, needle any) (bool, error) {
	switch t := haystack.(type) {
	case undefined:
		return false, nil
	case string:
		ns, ok := needle.(string)
		if !ok {
			return false, fmt.Errorf("argument of type %T is not a substring", needle)
		}
		return strings.Contains(t, ns), nil
	case []any:
		for _, e := range t {
			if eq(e, needle) {
				return true, nil
			}
		}
		return false, nil
	case Dict:
		if ns, ok := needle.(string); ok {
			_, present := t.Vals[ns]
			return present, nil
		}
		return false, nil
	case map[string]any:
		if ns, ok := needle.(string); ok {
			_, present := t[ns]
			return present, nil
		}
		return false, nil
	case nil:
		return false, fmt.Errorf("argument of type 'NoneType' is not a container or iterable")
	default:
		return false, fmt.Errorf("argument of type %T is not a container or iterable", haystack)
	}
}

// --- stringification --------------------------------------------------------

// strValue renders v where a template outputs a value ({{ ... }}, ~, |string).
// Undefined renders as the empty string, matching jinja's default Undefined.
func strValue(v any) (string, error) {
	switch t := v.(type) {
	case undefined:
		return "", nil
	case nil:
		return "None", nil
	case bool:
		if t {
			return "True", nil
		}
		return "False", nil
	case int64:
		return strconv.FormatInt(t, 10), nil
	case float64:
		return pyFloat(t)
	case string:
		return t, nil
	case []any, Dict, map[string]any:
		return reprValue(v)
	case *Namespace:
		return "<namespace>", nil
	default:
		return "", fmt.Errorf("cannot convert %T to string", v)
	}
}

// reprValue renders v for a container context: strings gain quotes.
func reprValue(v any) (string, error) {
	switch t := v.(type) {
	case undefined:
		return "", fmt.Errorf("cannot represent undefined in a container")
	case map[string]any:
		return "", fmt.Errorf("cannot represent an unordered map (parse context with runner.ParseJSON)")
	case string:
		return reprString(t), nil
	case []any:
		var b strings.Builder
		b.WriteByte('[')
		for i, e := range t {
			if i > 0 {
				b.WriteString(", ")
			}
			s, err := reprValue(e)
			if err != nil {
				return "", err
			}
			b.WriteString(s)
		}
		b.WriteByte(']')
		return b.String(), nil
	case Dict:
		var b strings.Builder
		b.WriteByte('{')
		for i, k := range t.Keys {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteString(reprString(k))
			b.WriteString(": ")
			s, err := reprValue(t.Vals[k])
			if err != nil {
				return "", err
			}
			b.WriteString(s)
		}
		b.WriteByte('}')
		return b.String(), nil
	default:
		return strValue(v)
	}
}

// reprString is Python's str repr: single quotes, escaping as Python does.
func reprString(s string) string {
	quote := byte('\'')
	if strings.ContainsRune(s, '\'') && !strings.ContainsRune(s, '"') {
		quote = '"'
	}
	var b strings.Builder
	b.WriteByte(quote)
	for _, r := range s {
		switch r {
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\a':
			b.WriteString(`\a`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		case '\v':
			b.WriteString(`\v`)
		case rune(quote):
			b.WriteString(`\` + string(rune(quote)))
		default:
			if r < 0x20 || r == 0x7f {
				fmt.Fprintf(&b, `\x%02x`, r)
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte(quote)
	return b.String()
}

// pyFloat formats like Python's repr(float): shortest round-trip, fixed
// notation for exponents in [-4, 16), otherwise e-notation.
func pyFloat(f float64) (string, error) {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return "", fmt.Errorf("cannot represent non-finite float %v", f)
	}
	if f == 0 {
		if math.Signbit(f) {
			return "-0.0", nil
		}
		return "0.0", nil
	}
	es := strconv.FormatFloat(f, 'e', -1, 64)
	exp := 0
	if i := strings.LastIndexByte(es, 'e'); i >= 0 {
		exp, _ = strconv.Atoi(es[i+1:])
	}
	if exp >= -4 && exp < 16 {
		fs := strconv.FormatFloat(f, 'f', -1, 64)
		if !strings.ContainsAny(fs, ".e") {
			fs += ".0"
		}
		return fs, nil
	}
	return es, nil
}

// jsonString is json.dumps of a string with ensure_ascii=False: only JSON
// escapes, raw UTF-8 (transformers' tojson does not HTML-escape or \u-escape).
func jsonString(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		default:
			if r < 0x20 {
				fmt.Fprintf(&b, `\u%04x`, r)
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
	return b.String()
}

// tojsonValue is the tojson filter: json.dumps(value, ensure_ascii=False,
// sort_keys=False) with default separators (", ", ": ").
func tojsonValue(v any) (string, error) {
	switch t := v.(type) {
	case nil:
		return "null", nil
	case bool:
		if t {
			return "true", nil
		}
		return "false", nil
	case int64:
		return strconv.FormatInt(t, 10), nil
	case float64:
		return pyFloat(t)
	case string:
		return jsonString(t), nil
	case undefined:
		return "", fmt.Errorf("undefined is not JSON serializable")
	case []any:
		var b strings.Builder
		b.WriteByte('[')
		for i, e := range t {
			if i > 0 {
				b.WriteString(", ")
			}
			s, err := tojsonValue(e)
			if err != nil {
				return "", err
			}
			b.WriteString(s)
		}
		b.WriteByte(']')
		return b.String(), nil
	case Dict:
		var b strings.Builder
		b.WriteByte('{')
		for i, k := range t.Keys {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteString(jsonString(k))
			b.WriteString(": ")
			s, err := tojsonValue(t.Vals[k])
			if err != nil {
				return "", err
			}
			b.WriteString(s)
		}
		b.WriteByte('}')
		return b.String(), nil
	case map[string]any:
		return "", fmt.Errorf("tojson on an unordered map: parse context with runner.ParseJSON (key order must be preserved)")
	default:
		return "", fmt.Errorf("not JSON serializable: %T", v)
	}
}
