package runner

import (
	"fmt"
	"strings"
	"time"
)

// Render applies the template to vars. vars keys are the template's global
// names (messages, tools, add_generation_prompt, special tokens…). now supplies
// strftime_now; nil uses Python-strftime semantics on the local clock.
//
// The returned string is the prompt bytes: same inputs must render the same
// bytes as today's serving (transformers-compatible environment).
func (t *Template) Render(vars map[string]any, now func(string) (string, error)) (string, error) {
	ctx := make(map[string]any, len(vars))
	for k, v := range vars {
		nv, err := normalize(v)
		if err != nil {
			return "", fmt.Errorf("template context %q: %w", k, err)
		}
		ctx[k] = nv
	}
	ev := &evaluator{ctx: ctx, now: now}
	if ev.now == nil {
		ev.now = defaultStrftime
	}
	sig, err := ev.run(t.body, &frame{vars: map[string]any{}})
	if err != nil {
		return "", err
	}
	if sig != sigNone {
		return "", fmt.Errorf("template: loop control outside of a loop")
	}
	return ev.buf.String(), nil
}

// --- evaluator --------------------------------------------------------------

type signal int

const (
	sigNone signal = iota
	sigBreak
	sigContinue
)

type frame struct {
	parent *frame
	vars   map[string]any
}

type evaluator struct {
	buf strings.Builder
	ctx map[string]any
	now func(string) (string, error)
}

func (ev *evaluator) lookup(f *frame, name string) any {
	for fr := f; fr != nil; fr = fr.parent {
		if v, ok := fr.vars[name]; ok {
			return v
		}
	}
	if v, ok := ev.ctx[name]; ok {
		return v
	}
	switch name {
	case "namespace", "strftime_now", "raise_exception":
		return builtinFunc{name: name}
	}
	return undefined{name: name}
}

func (ev *evaluator) run(nodes []node, f *frame) (signal, error) {
	for _, n := range nodes {
		switch t := n.(type) {
		case textNode:
			ev.buf.WriteString(t.s)
		case outputNode:
			v, err := ev.eval(t.e, f)
			if err != nil {
				return sigNone, err
			}
			s, err := strValue(v)
			if err != nil {
				return sigNone, perr(t.line, "%s", err)
			}
			ev.buf.WriteString(s)
		case ifNode:
			sig, err := ev.runIf(t, f)
			if err != nil || sig != sigNone {
				return sig, err
			}
		case forNode:
			sig, err := ev.runFor(t, f)
			if err != nil || sig != sigNone {
				return sig, err
			}
		case setNode:
			v, err := ev.eval(t.value, f)
			if err != nil {
				return sigNone, err
			}
			if t.obj == nil {
				f.vars[t.name] = v
				continue
			}
			obj, err := ev.eval(t.obj, f)
			if err != nil {
				return sigNone, err
			}
			ns, ok := obj.(*Namespace)
			if !ok {
				return sigNone, perr(t.line, "cannot set %q on %T", t.name, obj)
			}
			ns.Vals[t.name] = v
		case genNode:
			// {% generation %} renders transparently; the span it marks is the
			// interface concern of continue_final_message (a later slice).
			sig, err := ev.run(t.body, f)
			if err != nil || sig != sigNone {
				return sig, err
			}
		case breakNode:
			return sigBreak, nil
		case continueNode:
			return sigContinue, nil
		default:
			return sigNone, fmt.Errorf("template: unexpected statement %T", n)
		}
	}
	return sigNone, nil
}

func (ev *evaluator) runIf(n ifNode, f *frame) (signal, error) {
	for _, b := range n.branches {
		c, err := ev.eval(b.cond, f)
		if err != nil {
			return sigNone, err
		}
		if truthy(c) {
			return ev.run(b.body, f)
		}
	}
	return ev.run(n.elseBody, f)
}

func (ev *evaluator) runFor(n forNode, f *frame) (signal, error) {
	iter, err := ev.eval(n.iter, f)
	if err != nil {
		return sigNone, err
	}
	var items []any
	switch t := iter.(type) {
	case undefined:
		items = nil // jinja iterates Undefined as empty
	case []any:
		items = t
	case string:
		items = []any{}
		for _, r := range t {
			items = append(items, string(r))
		}
	case Dict:
		items = make([]any, 0, len(t.Keys))
		for _, k := range t.Keys {
			items = append(items, k)
		}
	default:
		return sigNone, perr(n.line, "cannot iterate over %T", iter)
	}
	length := int64(len(items))
	for i, item := range items {
		child := &frame{parent: f, vars: map[string]any{
			n.target: item,
			"loop":   loopInfo{index0: int64(i), length: length},
		}}
		sig, err := ev.run(n.body, child)
		if err != nil {
			return sigNone, err
		}
		switch sig {
		case sigBreak:
			return sigNone, nil
		case sigContinue:
			continue
		}
	}
	return sigNone, nil
}

// --- expression evaluation --------------------------------------------------

func (ev *evaluator) eval(e expr, f *frame) (any, error) {
	switch t := e.(type) {
	case litExpr:
		return t.v, nil
	case nameExpr:
		return ev.lookup(f, t.name), nil
	case attrExpr:
		return ev.evalAttr(t, f)
	case itemExpr:
		return ev.evalItem(t, f)
	case sliceExpr:
		return ev.evalSlice(t, f)
	case callExpr:
		return ev.evalCall(t, f)
	case filterExpr:
		return ev.evalFilter(t, f)
	case testExpr:
		v, err := ev.eval(t.e, f)
		if err != nil {
			return nil, err
		}
		res, err := applyTest(v, t.name)
		if err != nil {
			return nil, perr(t.line, "%s", err)
		}
		if t.neg {
			res = !res
		}
		return res, nil
	case unaryExpr:
		v, err := ev.eval(t.e, f)
		if err != nil {
			return nil, err
		}
		switch t.op {
		case "not":
			return !truthy(v), nil
		case "-":
			switch n := v.(type) {
			case int64:
				return -n, nil
			case float64:
				return -n, nil
			default:
				return nil, fmt.Errorf("unsupported operand type for unary -: %T", v)
			}
		}
		return nil, fmt.Errorf("unknown unary operator %q", t.op)
	case binaryExpr:
		return ev.evalBinary(t, f)
	case ternaryExpr:
		c, err := ev.eval(t.cond, f)
		if err != nil {
			return nil, err
		}
		if truthy(c) {
			return ev.eval(t.yes, f)
		}
		return ev.eval(t.no, f)
	default:
		return nil, fmt.Errorf("template: unexpected expression %T", e)
	}
}

// evalBinary covers every operator parseOr/parseAdd can produce:
// == != < <= > >= in "not in" and or + - ~ (see binaryExpr.op).
func (ev *evaluator) evalBinary(t binaryExpr, f *frame) (any, error) {
	switch t.op {
	case "and":
		l, err := ev.eval(t.l, f)
		if err != nil {
			return nil, err
		}
		if !truthy(l) {
			return l, nil
		}
		return ev.eval(t.r, f)
	case "or":
		l, err := ev.eval(t.l, f)
		if err != nil {
			return nil, err
		}
		if truthy(l) {
			return l, nil
		}
		return ev.eval(t.r, f)
	}
	l, err := ev.eval(t.l, f)
	if err != nil {
		return nil, err
	}
	r, err := ev.eval(t.r, f)
	if err != nil {
		return nil, err
	}
	switch t.op {
	case "==":
		return eq(l, r), nil
	case "!=":
		return !eq(l, r), nil
	case "<", "<=", ">", ">=":
		return ord(t.op, l, r)
	case "in":
		return contains(r, l)
	case "not in":
		ok, err := contains(r, l)
		if err != nil {
			return nil, err
		}
		return !ok, nil
	case "~":
		ls, err := strValue(l)
		if err != nil {
			return nil, err
		}
		rs, err := strValue(r)
		if err != nil {
			return nil, err
		}
		return ls + rs, nil
	case "+":
		if ls, ok := l.(string); ok {
			if rs, ok := r.(string); ok {
				return ls + rs, nil
			}
		}
		if ll, ok := l.([]any); ok {
			if rl, ok := r.([]any); ok {
				return append(append([]any{}, ll...), rl...), nil
			}
		}
		return binaryArith("+", l, r)
	case "-":
		return binaryArith("-", l, r)
	}
	return nil, fmt.Errorf("unknown binary operator %q", t.op)
}

// binaryArith follows Python: int64 op int64 stays int64, anything else
// widens to float64; non-numbers are a template error.
func binaryArith(op string, l, r any) (any, error) {
	if li, ok := l.(int64); ok {
		if ri, ok := r.(int64); ok {
			if op == "+" {
				return li + ri, nil
			}
			return li - ri, nil
		}
	}
	lf, ok := asNumber(l)
	if !ok {
		return nil, fmt.Errorf("unsupported operand types for %s: %T and %T", op, l, r)
	}
	rf, ok := asNumber(r)
	if !ok {
		return nil, fmt.Errorf("unsupported operand types for %s: %T and %T", op, l, r)
	}
	if op == "+" {
		return lf + rf, nil
	}
	return lf - rf, nil
}

func (ev *evaluator) evalAttr(t attrExpr, f *frame) (any, error) {
	obj, err := ev.eval(t.obj, f)
	if err != nil {
		return nil, err
	}
	switch o := obj.(type) {
	case undefined:
		return nil, fmt.Errorf("'%s' is undefined", o.name)
	case Dict:
		if v, ok := o.Vals[t.name]; ok {
			return v, nil
		}
		return undefined{name: t.name}, nil
	case map[string]any:
		if v, ok := o[t.name]; ok {
			return v, nil
		}
		return undefined{name: t.name}, nil
	case *Namespace:
		if v, ok := o.Vals[t.name]; ok {
			return v, nil
		}
		return undefined{name: t.name}, nil
	case string:
		if stringMethods[t.name] {
			return boundMethod{recv: o, name: t.name}, nil
		}
		return undefined{name: t.name}, nil
	case loopInfo:
		switch t.name {
		case "index0":
			return o.index0, nil
		case "index":
			return o.index0 + 1, nil
		case "first":
			return o.index0 == 0, nil
		case "last":
			return o.index0 == o.length-1, nil
		case "length":
			return o.length, nil
		default:
			return nil, perr(t.line, "unsupported loop attribute %q", t.name)
		}
	default:
		return undefined{name: t.name}, nil
	}
}

func (ev *evaluator) evalItem(t itemExpr, f *frame) (any, error) {
	obj, err := ev.eval(t.obj, f)
	if err != nil {
		return nil, err
	}
	idx, err := ev.eval(t.idx, f)
	if err != nil {
		return nil, err
	}
	return itemGet(obj, idx, t.line)
}

func itemGet(obj, idx any, line int) (any, error) {
	switch o := obj.(type) {
	case undefined:
		return nil, perr(line, "'%s' is undefined", o.name)
	case []any:
		i, ok := idx.(int64)
		if !ok {
			return nil, perr(line, "list index must be an integer, got %T", idx)
		}
		if i < 0 {
			i += int64(len(o))
		}
		if i < 0 || i >= int64(len(o)) {
			return nil, perr(line, "list index %d out of range", idx.(int64))
		}
		return o[i], nil
	case string:
		i, ok := idx.(int64)
		if !ok {
			return nil, perr(line, "string index must be an integer, got %T", idx)
		}
		r := []rune(o)
		if i < 0 {
			i += int64(len(r))
		}
		if i < 0 || i >= int64(len(r)) {
			return nil, perr(line, "string index %d out of range", idx.(int64))
		}
		return string(r[i]), nil
	case Dict:
		k, ok := idx.(string)
		if !ok {
			return nil, perr(line, "object key must be a string, got %T", idx)
		}
		if v, present := o.Vals[k]; present {
			return v, nil
		}
		return undefined{name: k}, nil
	case map[string]any:
		k, ok := idx.(string)
		if !ok {
			return nil, perr(line, "object key must be a string, got %T", idx)
		}
		if v, present := o[k]; present {
			return v, nil
		}
		return undefined{name: k}, nil
	case *Namespace:
		k, ok := idx.(string)
		if !ok {
			return nil, perr(line, "object key must be a string, got %T", idx)
		}
		if v, present := o.Vals[k]; present {
			return v, nil
		}
		return undefined{name: k}, nil
	default:
		return nil, perr(line, "cannot subscript %T", obj)
	}
}

func (ev *evaluator) evalSlice(t sliceExpr, f *frame) (any, error) {
	obj, err := ev.eval(t.obj, f)
	if err != nil {
		return nil, err
	}
	component := func(e expr) (int64, bool, error) {
		if e == nil {
			return 0, false, nil
		}
		v, err := ev.eval(e, f)
		if err != nil {
			return 0, false, err
		}
		i, ok := v.(int64)
		if !ok {
			return 0, false, perr(t.line, "slice components must be integers, got %T", v)
		}
		return i, true, nil
	}
	lo, hasLo, err := component(t.lo)
	if err != nil {
		return nil, err
	}
	hi, hasHi, err := component(t.hi)
	if err != nil {
		return nil, err
	}
	st, hasSt, err := component(t.st)
	if err != nil {
		return nil, err
	}
	_ = hasLo
	_ = hasHi
	if !hasSt {
		st = 1
	}
	if st == 0 {
		return nil, perr(t.line, "slice step cannot be zero")
	}
	switch o := obj.(type) {
	case string:
		runes := []rune(o)
		start, stop, err := sliceIndices(int64(len(runes)), lo, hasLo, hi, hasHi, st)
		if err != nil {
			return nil, err
		}
		var b strings.Builder
		for i := start; inRange(i, stop, st); i += st {
			b.WriteRune(runes[i])
		}
		return b.String(), nil
	case []any:
		start, stop, err := sliceIndices(int64(len(o)), lo, hasLo, hi, hasHi, st)
		if err != nil {
			return nil, err
		}
		out := []any{}
		for i := start; inRange(i, stop, st); i += st {
			out = append(out, o[i])
		}
		return out, nil
	default:
		return nil, perr(t.line, "cannot slice %T", obj)
	}
}

// sliceIndices mirrors Python's slice.indices(length).
func sliceIndices(length, lo int64, hasLo bool, hi int64, hasHi bool, step int64) (start, stop int64, err error) {
	var lower, upper, defStart, defStop int64
	if step < 0 {
		lower, upper = -1, length-1
		defStart, defStop = upper, lower
	} else {
		lower, upper = 0, length
		defStart, defStop = lower, upper
	}
	normalizeIdx := func(v int64) int64 {
		if v < 0 {
			v += length
		}
		if v < lower {
			return lower
		}
		if v > upper {
			return upper
		}
		return v
	}
	start, stop = defStart, defStop
	if hasLo {
		start = normalizeIdx(lo)
	}
	if hasHi {
		stop = normalizeIdx(hi)
	}
	return start, stop, nil
}

func inRange(i, stop, step int64) bool {
	if step > 0 {
		return i < stop
	}
	return i > stop
}

// --- calls ------------------------------------------------------------------

func (ev *evaluator) evalCall(t callExpr, f *frame) (any, error) {
	fn, err := ev.eval(t.fn, f)
	if err != nil {
		return nil, err
	}
	args := make([]any, len(t.args))
	for i, a := range t.args {
		v, err := ev.eval(a, f)
		if err != nil {
			return nil, err
		}
		args[i] = v
	}
	kwargs := map[string]any{}
	for _, kw := range t.kwargs {
		v, err := ev.eval(kw.e, f)
		if err != nil {
			return nil, err
		}
		kwargs[kw.name] = v
	}
	switch c := fn.(type) {
	case builtinFunc:
		return ev.callBuiltin(c.name, args, kwargs, t.line)
	case boundMethod:
		return callMethod(c, args, kwargs, t.line)
	case undefined:
		return nil, perr(t.line, "'%s' is undefined", c.name)
	default:
		return nil, perr(t.line, "%T is not callable", fn)
	}
}

func (ev *evaluator) callBuiltin(name string, args []any, kwargs map[string]any, line int) (any, error) {
	switch name {
	case "namespace":
		if len(args) > 0 {
			return nil, perr(line, "namespace() takes keyword arguments only")
		}
		return &Namespace{Vals: kwargs}, nil
	case "strftime_now":
		if len(args) != 1 || len(kwargs) > 0 {
			return nil, perr(line, "strftime_now() takes one format argument")
		}
		format, ok := args[0].(string)
		if !ok {
			return nil, perr(line, "strftime_now() format must be a string")
		}
		return ev.now(format)
	case "raise_exception":
		if len(args) != 1 || len(kwargs) > 0 {
			return nil, perr(line, "raise_exception() takes one message")
		}
		msg, ok := args[0].(string)
		if !ok {
			return nil, perr(line, "raise_exception() message must be a string")
		}
		return nil, perr(line, "%s", msg)
	default:
		return nil, perr(line, "unknown builtin %q", name)
	}
}

// --- filters and tests ------------------------------------------------------

func (ev *evaluator) evalFilter(t filterExpr, f *frame) (any, error) {
	v, err := ev.eval(t.e, f)
	if err != nil {
		return nil, err
	}
	if len(t.kwargs) > 0 {
		return nil, perr(t.line, "filter %q takes no keyword arguments", t.name)
	}
	switch t.name {
	case "string":
		if len(t.args) > 0 {
			return nil, perr(t.line, "filter %q takes no arguments", t.name)
		}
		return strValue(v)
	case "length":
		if len(t.args) > 0 {
			return nil, perr(t.line, "filter %q takes no arguments", t.name)
		}
		return lengthOf(v, t.line)
	case "tojson":
		if len(t.args) > 0 {
			return nil, perr(t.line, "filter %q takes no arguments", t.name)
		}
		return tojsonValue(v)
	default:
		return nil, perr(t.line, "unsupported filter %q", t.name)
	}
}

func lengthOf(v any, line int) (any, error) {
	switch t := v.(type) {
	case string:
		return int64(len([]rune(t))), nil
	case []any:
		return int64(len(t)), nil
	case Dict:
		return int64(len(t.Keys)), nil
	case map[string]any:
		return int64(len(t)), nil
	case undefined:
		return nil, perr(line, "'%s' is undefined", t.name)
	default:
		return nil, perr(line, "no length for %T", v)
	}
}

func applyTest(v any, name string) (bool, error) {
	switch name {
	case "defined":
		return !isUndefined(v), nil
	case "string":
		_, ok := v.(string)
		return ok, nil
	case "number":
		_, ok := asNumber(v)
		return ok, nil
	case "integer":
		_, ok := v.(int64)
		return ok, nil
	case "boolean":
		_, ok := v.(bool)
		return ok, nil
	case "none":
		return v == nil, nil
	}
	return false, fmt.Errorf("unsupported test %q", name)
}

// --- string methods ---------------------------------------------------------

var stringMethods = map[string]bool{
	"split": true, "strip": true, "lstrip": true, "rstrip": true,
	"replace": true, "startswith": true, "endswith": true,
}

func callMethod(m boundMethod, args []any, kwargs map[string]any, line int) (any, error) {
	if len(kwargs) > 0 {
		return nil, perr(line, "%s() takes no keyword arguments", m.name)
	}
	s, _ := m.recv.(string)
	argString := func(i int) (string, error) {
		if i >= len(args) {
			return "", fmt.Errorf("missing argument")
		}
		str, ok := args[i].(string)
		if !ok {
			return "", fmt.Errorf("argument %d must be a string, got %T", i, args[i])
		}
		return str, nil
	}
	fail := func(err error) (any, error) { return nil, perr(line, "%s(): %s", m.name, err) }
	switch m.name {
	case "split":
		sep := ""
		hasSep := false
		if len(args) > 0 {
			if args[0] == nil {
				hasSep = false
			} else {
				v, err := argString(0)
				if err != nil {
					return fail(err)
				}
				sep, hasSep = v, true
			}
		}
		maxsplit := int64(-1)
		if len(args) > 1 {
			n, ok := args[1].(int64)
			if !ok {
				return fail(fmt.Errorf("maxsplit must be an integer, got %T", args[1]))
			}
			maxsplit = n
		}
		if len(args) > 2 {
			return fail(fmt.Errorf("too many arguments"))
		}
		var parts []string
		if !hasSep {
			parts = strings.Fields(s)
		} else {
			parts = strings.Split(s, sep)
			if maxsplit >= 0 && len(parts) > int(maxsplit)+1 {
				rest := strings.Join(parts[maxsplit+1:], sep)
				parts = append(parts[:maxsplit+1], rest)
			}
		}
		out := make([]any, len(parts))
		for i, p := range parts {
			out[i] = p
		}
		return out, nil
	case "strip", "lstrip", "rstrip":
		if len(args) > 1 {
			return fail(fmt.Errorf("too many arguments"))
		}
		if len(args) == 0 {
			switch m.name {
			case "strip":
				return strings.TrimSpace(s), nil
			case "lstrip":
				return strings.TrimLeftFunc(s, isPySpace), nil
			default:
				return strings.TrimRightFunc(s, isPySpace), nil
			}
		}
		chars, err := argString(0)
		if err != nil {
			return fail(err)
		}
		cut := func(r rune) bool { return strings.ContainsRune(chars, r) }
		switch m.name {
		case "strip":
			return strings.TrimFunc(s, cut), nil
		case "lstrip":
			return strings.TrimLeftFunc(s, cut), nil
		default:
			return strings.TrimRightFunc(s, cut), nil
		}
	case "replace":
		if len(args) < 2 || len(args) > 3 {
			return fail(fmt.Errorf("takes 2 or 3 arguments"))
		}
		old, err := argString(0)
		if err != nil {
			return fail(err)
		}
		neu, err := argString(1)
		if err != nil {
			return fail(err)
		}
		count := int64(-1)
		if len(args) == 3 {
			n, ok := args[2].(int64)
			if !ok {
				return fail(fmt.Errorf("count must be an integer, got %T", args[2]))
			}
			count = n
		}
		if count < 0 {
			return strings.ReplaceAll(s, old, neu), nil
		}
		return strings.Replace(s, old, neu, int(count)), nil
	case "startswith", "endswith":
		if len(args) != 1 {
			return fail(fmt.Errorf("takes one prefix argument"))
		}
		p, err := argString(0)
		if err != nil {
			return fail(err)
		}
		if m.name == "startswith" {
			return strings.HasPrefix(s, p), nil
		}
		return strings.HasSuffix(s, p), nil
	}
	return nil, perr(line, "unsupported string method %q", m.name)
}

func isPySpace(r rune) bool {
	switch r {
	case ' ', '\t', '\n', '\r', '\f', '\v':
		return true
	}
	return false
}

// --- time -------------------------------------------------------------------

// defaultStrftime implements Python's strftime directives we support; anything
// else fails loudly rather than rendering a wrong date.
func defaultStrftime(format string) (string, error) {
	t := time.Now()
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
		directive := format[i]
		var layout string
		switch directive {
		case '%':
			b.WriteByte('%')
			continue
		case 'd':
			layout = "02"
		case 'e':
			layout = "_2"
		case 'm':
			layout = "01"
		case 'B':
			layout = "January"
		case 'b':
			layout = "Jan"
		case 'A':
			layout = "Monday"
		case 'a':
			layout = "Mon"
		case 'Y':
			layout = "2006"
		case 'y':
			layout = "06"
		case 'H':
			layout = "15"
		case 'I':
			layout = "03"
		case 'M':
			layout = "04"
		case 'S':
			layout = "05"
		case 'p':
			layout = "PM"
		case 'z':
			layout = "-0700"
		default:
			return "", fmt.Errorf("unsupported time directive %%%c", directive)
		}
		b.WriteString(t.Format(layout))
	}
	return b.String(), nil
}
