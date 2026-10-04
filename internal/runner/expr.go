package runner

import (
	"strconv"
	"strings"
)

// --- expression AST ---------------------------------------------------------

type expr interface{ exprNode() }

type litExpr struct{ v any } // string, int64, float64, bool or nil

type nameExpr struct {
	name string
	line int
}

type attrExpr struct {
	obj  expr
	name string
	line int
}

type itemExpr struct {
	obj  expr
	idx  expr
	line int
}

type sliceExpr struct {
	obj        expr
	lo, hi, st expr // any may be nil for omitted components
	line       int
}

type kwarg struct {
	name string
	e    expr
}

type callExpr struct {
	fn     expr
	args   []expr
	kwargs []kwarg
	line   int
}

type filterExpr struct {
	e      expr
	name   string
	args   []expr
	kwargs []kwarg
	line   int
}

type testExpr struct {
	e    expr
	neg  bool
	name string
	line int
}

type unaryExpr struct {
	op string // "not" or "-"
	e  expr
}

type binaryExpr struct {
	op   string // == != < <= > >= in "not in" and or + - ~
	l, r expr
}

type ternaryExpr struct{ cond, yes, no expr }

func (litExpr) exprNode()     {}
func (nameExpr) exprNode()    {}
func (attrExpr) exprNode()    {}
func (itemExpr) exprNode()    {}
func (sliceExpr) exprNode()   {}
func (callExpr) exprNode()    {}
func (filterExpr) exprNode()  {}
func (testExpr) exprNode()    {}
func (unaryExpr) exprNode()   {}
func (binaryExpr) exprNode()  {}
func (ternaryExpr) exprNode() {}

// --- supported names (validated at parse time) ------------------------------

var filterNames = map[string]bool{"string": true, "length": true, "tojson": true}
var testNames = map[string]bool{
	"defined": true, "string": true, "number": true,
	"integer": true, "boolean": true, "none": true,
}

// --- statement parser -------------------------------------------------------

type parser struct {
	toks     []token
	pos      int
	forDepth int
}

func (p *parser) cur() (token, bool) {
	if p.pos >= len(p.toks) {
		return token{}, false
	}
	return p.toks[p.pos], true
}

// parseBlock parses statements until one of the stop tags (or template end
// when stop is nil). The stop tag itself is left unconsumed for the caller.
func (p *parser) parseBlock(stop []string) ([]node, error) {
	var out []node
	for {
		t, ok := p.cur()
		if !ok {
			if stop != nil {
				return nil, perr(0, "unexpected end of template, expected %q", stop[0])
			}
			return out, nil
		}
		switch t.kind {
		case tokText:
			p.pos++
			out = append(out, textNode{s: t.body})
			continue
		case tokComment:
			p.pos++
			continue
		case tokOutput:
			p.pos++
			e, err := parseExprText(t.body, t.line)
			if err != nil {
				return nil, err
			}
			out = append(out, outputNode{e: e, line: t.line})
			continue
		}
		// block tag
		name, rest := splitName(t.body)
		if stop != nil && containsName(stop, name) {
			return out, nil
		}
		p.pos++
		switch name {
		case "if":
			n, err := p.parseIf(rest, t.line)
			if err != nil {
				return nil, err
			}
			out = append(out, n)
		case "for":
			n, err := p.parseFor(rest, t.line)
			if err != nil {
				return nil, err
			}
			out = append(out, n)
		case "set":
			n, err := p.parseSet(rest, t.line)
			if err != nil {
				return nil, err
			}
			out = append(out, n)
		case "generation":
			body, err := p.parseBlock([]string{"endgeneration"})
			if err != nil {
				return nil, err
			}
			if err := p.expectEnd("endgeneration", t.line); err != nil {
				return nil, err
			}
			out = append(out, genNode{body: body})
		case "break":
			if p.forDepth == 0 {
				return nil, perr(t.line, "%q outside of a loop", "break")
			}
			out = append(out, breakNode{line: t.line})
		case "continue":
			if p.forDepth == 0 {
				return nil, perr(t.line, "%q outside of a loop", "continue")
			}
			out = append(out, continueNode{line: t.line})
		default:
			if stop == nil {
				return nil, perr(t.line, "unexpected block %q", name)
			}
			return nil, perr(t.line, "unsupported tag %q", name)
		}
	}
}

func (p *parser) expectEnd(name string, line int) error {
	t, ok := p.cur()
	if !ok {
		return perr(line, "missing %q", name)
	}
	got, _ := splitName(t.body)
	if t.kind != tokBlock || got != name {
		return perr(t.line, "expected %q, got %q", name, got)
	}
	p.pos++
	return nil
}

func (p *parser) parseIf(rest string, line int) (node, error) {
	cond, err := parseExprText(rest, line)
	if err != nil {
		return nil, err
	}
	body, err := p.parseBlock([]string{"elif", "else", "endif"})
	if err != nil {
		return nil, err
	}
	n := ifNode{branches: []condBranch{{cond: cond, body: body}}}
	for {
		t, ok := p.cur()
		if !ok {
			return nil, perr(line, "missing %q", "endif")
		}
		name, sub := splitName(t.body)
		p.pos++
		switch name {
		case "endif":
			return n, nil
		case "elif":
			c, err := parseExprText(sub, t.line)
			if err != nil {
				return nil, err
			}
			b, err := p.parseBlock([]string{"elif", "else", "endif"})
			if err != nil {
				return nil, err
			}
			n.branches = append(n.branches, condBranch{cond: c, body: b})
		case "else":
			b, err := p.parseBlock([]string{"endif"})
			if err != nil {
				return nil, err
			}
			n.elseBody = b
			if err := p.expectEnd("endif", t.line); err != nil {
				return nil, err
			}
			return n, nil
		default:
			return nil, perr(t.line, "expected %q, got %q", "endif", name)
		}
	}
}

func (p *parser) parseFor(rest string, line int) (node, error) {
	in := strings.Index(rest, " in ")
	if in < 0 {
		return nil, perr(line, "%q statement needs \"<name> in <expr>\"", "for")
	}
	target := strings.TrimSpace(rest[:in])
	if target == "" || strings.ContainsAny(target, ", \t") {
		return nil, perr(line, "unsupported for-target %q (single name only)", target)
	}
	for i := range len(target) {
		c := target[i]
		if !isIdentChar(c) || (i == 0 && c >= '0' && c <= '9') {
			return nil, perr(line, "invalid for-target %q", target)
		}
	}
	iter, err := parseExprText(rest[in+4:], line)
	if err != nil {
		return nil, err
	}
	p.forDepth++
	body, err := p.parseBlock([]string{"endfor"})
	p.forDepth--
	if err != nil {
		return nil, err
	}
	if err := p.expectEnd("endfor", line); err != nil {
		return nil, err
	}
	return forNode{target: target, iter: iter, body: body, line: line}, nil
}

func (p *parser) parseSet(rest string, line int) (node, error) {
	i := strings.IndexByte(rest, '=')
	if i < 0 {
		return nil, perr(line, "%q needs \"<name> = <expr>\"", "set")
	}
	target := strings.TrimSpace(rest[:i])
	if target == "" {
		return nil, perr(line, "%q needs a target", "set")
	}
	if strings.ContainsAny(target, ", \t\n\r") {
		return nil, perr(line, "unsupported set-target %q (tuple unpacking and whitespace are outside the subset)", target)
	}
	value, err := parseExprText(rest[i+1:], line)
	if err != nil {
		return nil, err
	}
	parts := strings.Split(target, ".")
	for _, part := range parts {
		if part == "" {
			return nil, perr(line, "invalid set-target %q", target)
		}
	}
	if len(parts) == 1 {
		return setNode{name: parts[0], value: value, line: line}, nil
	}
	// Attribute target: rebuild the object expression from the leading names.
	obj, err := parseExprText(parts[0], line)
	if err != nil {
		return nil, perr(line, "invalid set-target %q", target)
	}
	e := expr(obj)
	for _, part := range parts[1 : len(parts)-1] {
		e = attrExpr{obj: e, name: part, line: line}
	}
	// The container chain holds every part but the leaf; name carries the leaf.
	return setNode{name: parts[len(parts)-1], obj: e, value: value, line: line}, nil
}

// --- expression parser ------------------------------------------------------

// parseExprText parses a full expression and requires the whole text consumed.
func parseExprText(src string, line int) (expr, error) {
	ep := &exprParser{src: strings.TrimSpace(src), line: line}
	e, err := ep.parseTernary()
	if err != nil {
		return nil, err
	}
	ep.skipWS()
	if ep.pos < len(ep.src) {
		return nil, perr(line, "unexpected %q in expression", ep.src[ep.pos:])
	}
	return e, nil
}

type exprParser struct {
	src  string
	pos  int
	line int
}

func (ep *exprParser) skipWS() {
	for ep.pos < len(ep.src) && isSpaceByte(ep.src[ep.pos]) {
		ep.pos++
	}
}

func (ep *exprParser) peek() byte {
	ep.skipWS()
	if ep.pos >= len(ep.src) {
		return 0
	}
	return ep.src[ep.pos]
}

// tryKW matches a keyword followed by a non-identifier character.
func (ep *exprParser) tryKW(kw string) bool {
	ep.skipWS()
	if !strings.HasPrefix(ep.src[ep.pos:], kw) {
		return false
	}
	end := ep.pos + len(kw)
	if end < len(ep.src) && isIdentChar(ep.src[end]) {
		return false
	}
	ep.pos = end
	return true
}

func (ep *exprParser) tryOp(op string) bool {
	ep.skipWS()
	if !strings.HasPrefix(ep.src[ep.pos:], op) {
		return false
	}
	// "<=" must not match before "<"; handled by caller order. "=" alone is
	// not an operator outside kwargs (caller does not tryOp("=")).
	ep.pos += len(op)
	return true
}

func (ep *exprParser) parseName() (string, error) {
	ep.skipWS()
	if ep.pos >= len(ep.src) {
		return "", perr(ep.line, "unexpected end of expression")
	}
	c := ep.src[ep.pos]
	if !isIdentStart(c) {
		return "", perr(ep.line, "unexpected %q in expression", string(c))
	}
	start := ep.pos
	for ep.pos < len(ep.src) && isIdentChar(ep.src[ep.pos]) {
		ep.pos++
	}
	return ep.src[start:ep.pos], nil
}

func (ep *exprParser) parseTernary() (expr, error) {
	yes, err := ep.parseOr()
	if err != nil {
		return nil, err
	}
	if !ep.tryKW("if") {
		return yes, nil
	}
	cond, err := ep.parseOr()
	if err != nil {
		return nil, err
	}
	if !ep.tryKW("else") {
		return nil, perr(ep.line, "expected %q in conditional expression", "else")
	}
	no, err := ep.parseTernary()
	if err != nil {
		return nil, err
	}
	return ternaryExpr{cond: cond, yes: yes, no: no}, nil
}

func (ep *exprParser) parseOr() (expr, error) {
	l, err := ep.parseAnd()
	if err != nil {
		return nil, err
	}
	for ep.tryKW("or") {
		r, err := ep.parseAnd()
		if err != nil {
			return nil, err
		}
		l = binaryExpr{op: "or", l: l, r: r}
	}
	return l, nil
}

func (ep *exprParser) parseAnd() (expr, error) {
	l, err := ep.parseNot()
	if err != nil {
		return nil, err
	}
	for ep.tryKW("and") {
		r, err := ep.parseNot()
		if err != nil {
			return nil, err
		}
		l = binaryExpr{op: "and", l: l, r: r}
	}
	return l, nil
}

func (ep *exprParser) parseNot() (expr, error) {
	if ep.tryKW("not") {
		e, err := ep.parseNot()
		if err != nil {
			return nil, err
		}
		return unaryExpr{op: "not", e: e}, nil
	}
	return ep.parseCmp()
}

func (ep *exprParser) parseCmp() (expr, error) {
	l, err := ep.parseAdd()
	if err != nil {
		return nil, err
	}
	for {
		switch {
		case ep.tryOp("<="):
			r, err := ep.parseAdd()
			if err != nil {
				return nil, err
			}
			l = binaryExpr{op: "<=", l: l, r: r}
		case ep.tryOp(">="):
			r, err := ep.parseAdd()
			if err != nil {
				return nil, err
			}
			l = binaryExpr{op: ">=", l: l, r: r}
		case ep.tryOp("=="):
			r, err := ep.parseAdd()
			if err != nil {
				return nil, err
			}
			l = binaryExpr{op: "==", l: l, r: r}
		case ep.tryOp("!="):
			r, err := ep.parseAdd()
			if err != nil {
				return nil, err
			}
			l = binaryExpr{op: "!=", l: l, r: r}
		case ep.tryOp("<"):
			r, err := ep.parseAdd()
			if err != nil {
				return nil, err
			}
			l = binaryExpr{op: "<", l: l, r: r}
		case ep.tryOp(">"):
			r, err := ep.parseAdd()
			if err != nil {
				return nil, err
			}
			l = binaryExpr{op: ">", l: l, r: r}
		case ep.tryKW("not"):
			if !ep.tryKW("in") {
				return nil, perr(ep.line, "expected %q after %q", "in", "not")
			}
			r, err := ep.parseAdd()
			if err != nil {
				return nil, err
			}
			l = binaryExpr{op: "not in", l: l, r: r}
		case ep.tryKW("in"):
			r, err := ep.parseAdd()
			if err != nil {
				return nil, err
			}
			l = binaryExpr{op: "in", l: l, r: r}
		case ep.tryKW("is"):
			neg := ep.tryKW("not")
			name, err := ep.parseName()
			if err != nil {
				return nil, err
			}
			if !testNames[name] {
				return nil, perr(ep.line, "unsupported test %q", name)
			}
			l = testExpr{e: l, neg: neg, name: name, line: ep.line}
		default:
			return l, nil
		}
	}
}

func (ep *exprParser) parseAdd() (expr, error) {
	l, err := ep.parseMul()
	if err != nil {
		return nil, err
	}
	for {
		var op string
		switch {
		case ep.tryOp("~"):
			op = "~"
		case ep.tryOp("+"):
			op = "+"
		case ep.tryOp("-"):
			op = "-"
		default:
			return l, nil
		}
		r, err := ep.parseMul()
		if err != nil {
			return nil, err
		}
		l = binaryExpr{op: op, l: l, r: r}
	}
}

func (ep *exprParser) parseMul() (expr, error) {
	l, err := ep.parseUnary()
	if err != nil {
		return nil, err
	}
	for {
		ep.skipWS()
		if ep.pos < len(ep.src) {
			switch ep.src[ep.pos] {
			case '*', '/', '%':
				return nil, perr(ep.line, "unsupported operator %q", string(ep.src[ep.pos]))
			}
			if strings.HasPrefix(ep.src[ep.pos:], "//") {
				return nil, perr(ep.line, "unsupported operator %q", "//")
			}
		}
		return l, nil
	}
}

func (ep *exprParser) parseUnary() (expr, error) {
	if ep.tryOp("-") {
		e, err := ep.parseUnary()
		if err != nil {
			return nil, err
		}
		return unaryExpr{op: "-", e: e}, nil
	}
	return ep.parseFilter()
}

func (ep *exprParser) parseFilter() (expr, error) {
	e, err := ep.parsePostfix()
	if err != nil {
		return nil, err
	}
	for {
		if !ep.tryOp("|") {
			return e, nil
		}
		name, err := ep.parseName()
		if err != nil {
			return nil, err
		}
		if !filterNames[name] {
			return nil, perr(ep.line, "unsupported filter %q", name)
		}
		args, kwargs, err := ep.parseArgs()
		if err != nil {
			return nil, err
		}
		e = filterExpr{e: e, name: name, args: args, kwargs: kwargs, line: ep.line}
	}
}

func (ep *exprParser) parsePostfix() (expr, error) {
	e, err := ep.parsePrimary()
	if err != nil {
		return nil, err
	}
	for {
		switch {
		case ep.tryOp("."):
			name, err := ep.parseName()
			if err != nil {
				return nil, err
			}
			if ep.peek() == '(' {
				args, kwargs, err := ep.parseArgs()
				if err != nil {
					return nil, err
				}
				e = callExpr{fn: attrExpr{obj: e, name: name, line: ep.line}, args: args, kwargs: kwargs, line: ep.line}
				continue
			}
			e = attrExpr{obj: e, name: name, line: ep.line}
		case ep.peek() == '[':
			ep.pos++ // consume '['
			idx, err := ep.parseIndexOrSlice()
			if err != nil {
				return nil, err
			}
			if sl, ok := idx.(sliceExpr); ok {
				sl.obj = e
				e = sl
			} else {
				e = itemExpr{obj: e, idx: idx, line: ep.line}
			}
		case ep.peek() == '(':
			args, kwargs, err := ep.parseArgs()
			if err != nil {
				return nil, err
			}
			e = callExpr{fn: e, args: args, kwargs: kwargs, line: ep.line}
		default:
			return e, nil
		}
	}
}

// parseIndexOrSlice parses "[expr]" or "[a:b:c]"; the '[' is already consumed.
func (ep *exprParser) parseIndexOrSlice() (expr, error) {
	components := make([]expr, 0, 3) // nil entries mark omitted slice parts
	omit := make([]bool, 0, 3)
	for {
		if ep.peek() == ':' {
			ep.pos++
			components = append(components, nil)
			omit = append(omit, true)
			if ep.peek() == ']' {
				ep.pos++
				break
			}
			continue
		}
		if ep.peek() == ']' {
			ep.pos++
			break
		}
		e, err := ep.parseTernary()
		if err != nil {
			return nil, err
		}
		components = append(components, e)
		omit = append(omit, false)
		if ep.tryOp(":") {
			if ep.peek() == ']' {
				ep.pos++
				break
			}
			continue
		}
		if ep.tryOp("]") {
			break
		}
		return nil, perr(ep.line, "expected %q in subscript", "]")
	}
	if len(components) == 1 && !omit[0] {
		return components[0], nil // plain item (no slice)
	}
	sl := sliceExpr{line: ep.line}
	set := func(dst *expr, i int) {
		if i < len(components) && !omit[i] {
			*dst = components[i]
		}
	}
	set(&sl.lo, 0)
	set(&sl.hi, 1)
	set(&sl.st, 2)
	if len(components) > 3 {
		return nil, perr(ep.line, "too many slice components")
	}
	return sl, nil
}

// parseArgs parses "(pos..., name=expr...)".
func (ep *exprParser) parseArgs() ([]expr, []kwarg, error) {
	if !ep.tryOp("(") {
		return nil, nil, nil
	}
	var args []expr
	var kwargs []kwarg
	for {
		ep.skipWS()
		if ep.tryOp(")") {
			return args, kwargs, nil
		}
		// keyword argument? name followed by '=' but not '=='
		save := ep.pos
		isKw := false
		if name, err := ep.parseName(); err == nil {
			ep.skipWS()
			if ep.pos < len(ep.src) && ep.src[ep.pos] == '=' &&
				!(ep.pos+1 < len(ep.src) && ep.src[ep.pos+1] == '=') {
				ep.pos++
				isKw = true
				e, err := ep.parseTernary()
				if err != nil {
					return nil, nil, err
				}
				kwargs = append(kwargs, kwarg{name: name, e: e})
			} else {
				ep.pos = save
			}
		} else {
			ep.pos = save
		}
		if !isKw {
			if len(kwargs) > 0 {
				return nil, nil, perr(ep.line, "positional argument after keyword argument")
			}
			e, err := ep.parseTernary()
			if err != nil {
				return nil, nil, err
			}
			args = append(args, e)
		}
		if ep.tryOp(",") {
			continue
		}
		if ep.tryOp(")") {
			return args, kwargs, nil
		}
		return nil, nil, perr(ep.line, "expected %q or %q in argument list", ",", ")")
	}
}

func (ep *exprParser) parsePrimary() (expr, error) {
	ep.skipWS()
	if ep.pos >= len(ep.src) {
		return nil, perr(ep.line, "unexpected end of expression")
	}
	c := ep.src[ep.pos]
	switch {
	case c == '\'' || c == '"':
		s, err := ep.parseString()
		if err != nil {
			return nil, err
		}
		return litExpr{v: s}, nil
	case c == '(':
		ep.pos++
		e, err := ep.parseTernary()
		if err != nil {
			return nil, err
		}
		if !ep.tryOp(")") {
			return nil, perr(ep.line, "expected %q", ")")
		}
		return e, nil
	case c >= '0' && c <= '9':
		return ep.parseNumber()
	}
	// keywords-as-literals, then names
	for _, lit := range []struct {
		word string
		v    any
	}{{"true", true}, {"True", true}, {"false", false}, {"False", false},
		{"none", nil}, {"None", nil}} {
		if ep.tryKW(lit.word) {
			return litExpr{v: lit.v}, nil
		}
	}
	if isIdentStart(c) {
		name, err := ep.parseName()
		if err != nil {
			return nil, err
		}
		return nameExpr{name: name, line: ep.line}, nil
	}
	return nil, perr(ep.line, "unexpected %q in expression", string(c))
}

func (ep *exprParser) parseNumber() (expr, error) {
	start := ep.pos
	for ep.pos < len(ep.src) && isDigit(ep.src[ep.pos]) {
		ep.pos++
	}
	isFloat := false
	if ep.pos < len(ep.src) && ep.src[ep.pos] == '.' {
		isFloat = true
		ep.pos++
		for ep.pos < len(ep.src) && isDigit(ep.src[ep.pos]) {
			ep.pos++
		}
	}
	if ep.pos < len(ep.src) && (ep.src[ep.pos] == 'e' || ep.src[ep.pos] == 'E') {
		save := ep.pos
		ep.pos++
		if ep.pos < len(ep.src) && (ep.src[ep.pos] == '+' || ep.src[ep.pos] == '-') {
			ep.pos++
		}
		digits := ep.pos
		for ep.pos < len(ep.src) && isDigit(ep.src[ep.pos]) {
			ep.pos++
		}
		if ep.pos == digits {
			ep.pos = save // not an exponent after all
		} else {
			isFloat = true
		}
	}
	text := ep.src[start:ep.pos]
	if isFloat {
		f, err := strconv.ParseFloat(text, 64)
		if err != nil {
			return nil, perr(ep.line, "invalid number %q", text)
		}
		return litExpr{v: f}, nil
	}
	n, err := strconv.ParseInt(text, 10, 64)
	if err != nil {
		return nil, perr(ep.line, "invalid number %q", text)
	}
	return litExpr{v: n}, nil
}

func (ep *exprParser) parseString() (string, error) {
	quote := ep.src[ep.pos]
	// triple-quoted strings are outside the subset
	if ep.pos+2 < len(ep.src) && ep.src[ep.pos+1] == quote && ep.src[ep.pos+2] == quote {
		return "", perr(ep.line, "triple-quoted strings are unsupported")
	}
	ep.pos++
	var b strings.Builder
	for {
		if ep.pos >= len(ep.src) {
			return "", perr(ep.line, "unterminated string literal")
		}
		c := ep.src[ep.pos]
		switch {
		case c == quote:
			ep.pos++
			return b.String(), nil
		case c == '\n':
			return "", perr(ep.line, "unterminated string literal")
		case c == '\\':
			ep.pos++
			if ep.pos >= len(ep.src) {
				return "", perr(ep.line, "unterminated escape sequence")
			}
			e := ep.src[ep.pos]
			ep.pos++
			switch e {
			case 'n':
				b.WriteByte('\n')
			case 't':
				b.WriteByte('\t')
			case 'r':
				b.WriteByte('\r')
			case 'a':
				b.WriteByte('\a')
			case 'b':
				b.WriteByte('\b')
			case 'f':
				b.WriteByte('\f')
			case 'v':
				b.WriteByte('\v')
			case '\\', '\'', '"':
				b.WriteByte(e)
			case 'x':
				v, err := ep.parseHex(2)
				if err != nil {
					return "", err
				}
				b.WriteByte(byte(v))
			case 'u':
				v, err := ep.parseHex(4)
				if err != nil {
					return "", err
				}
				b.WriteString(string(rune(v)))
			case 'U':
				v, err := ep.parseHex(8)
				if err != nil {
					return "", err
				}
				b.WriteString(string(rune(v)))
			default:
				// Python keeps unknown escapes verbatim (e.g. "\q" is backslash q).
				b.WriteByte('\\')
				b.WriteByte(e)
			}
		default:
			b.WriteByte(c)
			ep.pos++
		}
	}
}

func (ep *exprParser) parseHex(n int) (uint64, error) {
	if ep.pos+n > len(ep.src) {
		return 0, perr(ep.line, "truncated hex escape")
	}
	text := ep.src[ep.pos : ep.pos+n]
	v, err := strconv.ParseUint(text, 16, 32)
	if err != nil {
		return 0, perr(ep.line, "invalid hex escape \\x%s", text)
	}
	ep.pos += n
	return v, nil
}

// --- lexer helpers ----------------------------------------------------------

func splitName(body string) (name, rest string) {
	body = strings.TrimLeft(body, " \t\r\n")
	i := 0
	for i < len(body) && isIdentChar(body[i]) {
		i++
	}
	return body[:i], body[i:]
}

func containsName(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

func isIdentStart(c byte) bool {
	return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

func isIdentChar(c byte) bool {
	return isIdentStart(c) || isDigit(c)
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }

func isSpaceByte(c byte) bool {
	return c == ' ' || c == '\t' || c == '\r' || c == '\n'
}
