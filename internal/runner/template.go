// Package runner is stone-llama's own serving runner (phase 6): it replaces
// the vendored TabbyAPI process with code we own.
//
// This file implements the chat-template renderer: a bounded, device-independent
// subset of the Jinja2 language as used by HuggingFace chat templates. It is
// reimplemented from the interface contract (how apply_chat_template renders),
// not by translating any AGPL source. The contract, verified against the two
// environments that matter, is:
//
//   - ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
//     with the tags if/elif/else/endif, for/endfor, set, generation/endgeneration,
//     break/continue (loopcontrols), and comments;
//   - globals: namespace, strftime_now, raise_exception;
//   - filters: string, length, tojson (json.dumps with ensure_ascii=False,
//     sort_keys=False, default separators — insertion key order, raw UTF-8);
//   - tests: defined, string, number, integer, boolean, none.
//
// Anything outside that subset fails at parse time with a loud error naming the
// construct; nothing is silently approximated.
package runner

import (
	"fmt"
	"strings"
)

// --- tokens -----------------------------------------------------------------

type tokKind int

const (
	tokText tokKind = iota
	tokOutput
	tokBlock
	tokComment
)

type token struct {
	kind      tokKind
	body      string // text run, or content inside {{ }}/{% %}/{# #}
	dashLeft  bool   // {{- / {%- / {#-
	dashRight bool   // -}} / -%} / -#}
	line      int    // 1-based line of the token start
}

// --- statement AST ----------------------------------------------------------

type node interface{ stmtNode() }

type textNode struct{ s string }

type outputNode struct {
	e    expr
	line int
}

type condBranch struct {
	cond expr
	body []node
}

type ifNode struct {
	branches []condBranch // one per if/elif
	elseBody []node       // nil when absent
}

type forNode struct {
	target string
	iter   expr
	body   []node
	line   int
}

type setNode struct {
	name  string // variable name, or the leaf attribute when obj is set
	obj   expr   // container expression for `a.b = v`; nil for a plain name
	value expr
	line  int
}

type genNode struct{ body []node }

type breakNode struct{ line int }
type continueNode struct{ line int }

func (textNode) stmtNode()     {}
func (outputNode) stmtNode()   {}
func (ifNode) stmtNode()       {}
func (forNode) stmtNode()      {}
func (setNode) stmtNode()      {}
func (genNode) stmtNode()      {}
func (breakNode) stmtNode()    {}
func (continueNode) stmtNode() {}

// --- template ---------------------------------------------------------------

// Template is a parsed chat template.
type Template struct {
	src  string
	body []node
}

// ParseTemplate parses a chat-template string in the supported Jinja2 subset.
// Unsupported constructs are parse errors naming the construct.
func ParseTemplate(src string) (*Template, error) {
	toks, err := lex(src)
	if err != nil {
		return nil, err
	}
	for i, s := range applyWhitespace(toks) {
		if toks[i].kind == tokText {
			toks[i].body = s
		}
	}
	p := &parser{toks: toks}
	body, err := p.parseBlock(nil)
	if err != nil {
		return nil, err
	}
	if p.pos < len(p.toks) {
		t := p.toks[p.pos]
		return nil, perr(t.line, "unexpected %s", describeTok(t))
	}
	return &Template{src: src, body: body}, nil
}

// Source returns the template text as parsed.
func (t *Template) Source() string { return t.src }

// --- lexer ------------------------------------------------------------------

// lex splits src into raw tokens. Whitespace control (the '-' flags) is applied
// later, when text runs are turned into nodes, because it depends on both
// neighbours of each run.
func lex(src string) ([]token, error) {
	var toks []token
	line := 1
	i := 0
	textStart := 0
	emitText := func(end int) {
		if end > textStart {
			toks = append(toks, token{kind: tokText, body: src[textStart:end], line: line})
		}
	}
	for i < len(src) {
		if src[i] != '{' || i+1 >= len(src) || (src[i+1] != '{' && src[i+1] != '%' && src[i+1] != '#') {
			if src[i] == '\n' {
				line++
			}
			i++
			continue
		}
		open := src[i : i+2] // "{{", "{%" or "{#"
		close := map[string]string{"{{": "}}", "{%": "%}", "{#": "#}"}[open]
		j := i + 2
		dashLeft := false
		if j < len(src) && src[j] == '-' {
			dashLeft = true
			j++
		}
		// Scan the tag body; string literals may contain the closing delimiter.
		bodyStart := j
		var quote byte
		dashRight := false
		closed := false
		bodyEnd := j // set where the tag body stops
		for j < len(src) {
			c := src[j]
			if quote != 0 {
				if c == '\\' && j+1 < len(src) {
					j += 2
					continue
				}
				if c == quote {
					quote = 0
				}
				j++
				continue
			}
			if open != "{#" && (c == '\'' || c == '"') {
				quote = c
				j++
				continue
			}
			if c == '-' && j+1 < len(src) && strings.HasPrefix(src[j+1:], close) {
				dashRight = true
				bodyEnd = j // the '-' itself is not part of the body
				j++
				closed = true
				break
			}
			if strings.HasPrefix(src[j:], close) {
				bodyEnd = j
				closed = true
				break
			}
			if c == '\n' {
				line++
			}
			j++
		}
		if !closed {
			return nil, perr(line, "unterminated %s tag", open)
		}
		body := src[bodyStart:bodyEnd]
		if quote != 0 {
			return nil, perr(line, "unterminated string literal in %s tag", open)
		}
		j += len(close)
		emitText(i)
		toks = append(toks, token{
			kind: map[string]tokKind{"{{": tokOutput, "{%": tokBlock, "{#": tokComment}[open],
			body: body, dashLeft: dashLeft, dashRight: dashRight, line: line,
		})
		textStart = j
		i = j
	}
	emitText(len(src))
	return toks, nil
}

// applyWhitespace post-processes text runs against their neighbouring tags.
// Rules (matching trim_blocks / lstrip_blocks / '-' control):
//   - a previous tag's '-}}'/-%}' strips all whitespace at the run's start;
//   - else a previous block/comment tag under trim_blocks drops one newline;
//   - a following tag's '-'/{{-' strips all whitespace at the run's end;
//   - else a following block/comment tag under lstrip_blocks strips the
//     run's trailing spaces/tabs when they are the whole last line.
func applyWhitespace(toks []token) []string {
	texts := make([]string, len(toks))
	for idx, t := range toks {
		if t.kind != tokText {
			continue // texts[idx] stays ""
		}
		s := t.body
		var prev, next *token
		if idx > 0 {
			prev = &toks[idx-1]
		}
		if idx+1 < len(toks) {
			next = &toks[idx+1]
		}
		// left side
		if prev != nil {
			switch {
			case prev.dashRight:
				s = trimAllSpaceLeft(s)
			case (prev.kind == tokBlock || prev.kind == tokComment):
				if strings.HasPrefix(s, "\r\n") {
					s = s[2:]
				} else if strings.HasPrefix(s, "\n") {
					s = s[1:]
				}
			}
		}
		// right side
		if next != nil {
			switch {
			case next.dashLeft:
				s = trimAllSpaceRight(s)
			case next.kind == tokBlock || next.kind == tokComment:
				if nl := strings.LastIndexByte(s, '\n'); isSpacesTabs(s[nl+1:]) {
					s = s[:nl+1]
				}
			}
		}
		texts[idx] = s
	}
	// Keep positions: texts aligns with toks, but only text tokens matter below.
	return texts
}

func isSpacesTabs(s string) bool {
	for i := range len(s) {
		if s[i] != ' ' && s[i] != '\t' {
			return false
		}
	}
	return true
}

func trimAllSpaceLeft(s string) string {
	return strings.TrimLeft(s, " \t\r\n\f\v")
}

func trimAllSpaceRight(s string) string {
	return strings.TrimRight(s, " \t\r\n\f\v")
}

func describeTok(t token) string {
	switch t.kind {
	case tokOutput:
		return "output expression"
	case tokBlock:
		name, _ := splitName(t.body)
		return fmt.Sprintf("block %q", name)
	case tokComment:
		return "comment"
	default:
		return "text"
	}
}

func perr(line int, format string, args ...any) error {
	return fmt.Errorf("template line %d: %s", line, fmt.Sprintf(format, args...))
}
