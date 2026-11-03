#!/usr/bin/env python3
"""Generate templates.json — the golden fixture for internal/runner's
differential tests against transformers' apply_chat_template.

Oracle: transformers.AutoTokenizer.apply_chat_template(tokenize=False) for
every entry (chat templates and the small expression snippets alike; snippets
simply ignore `messages`). Only local tokenizer/template files are read
(local_files_only=True) — nothing is ever downloaded, no model weights load.

Regenerate from the repo root (the asdf shim resolves the interpreter per
working directory, so pin the one the fixture was recorded with):

    ASDF_PYTHON_VERSION=3.14.3 python3 internal/runner/testdata/gen_golden.py

The transformers/jinja2 versions below are asserted: if they no longer match,
the script fails loudly instead of silently regenerating the fixture against
a different oracle. Re-verify against the Go engine before replacing
templates.json when bumping versions.

strftime_now is overridden with a fixed clock (2026-01-01 12:00) so the
fixture is date-independent; the Go test injects the same clock.

Output: internal/runner/testdata/templates.json (UTF-8, ensure_ascii=False).
"""

import datetime
import json
import pathlib
import platform
import sys

import jinja2
import transformers

WANT_TRANSFORMERS = "5.13.1"
WANT_JINJA2 = "3.1.4"

FIXED = datetime.datetime(2026, 1, 1, 12, 0, 0)
REPO_MODELS = pathlib.Path.home() / ".local/share/stone-llama/models"
OUT = pathlib.Path(__file__).with_name("templates.json")

CHAT_TEMPLATES = [
    ("smollm3_chat", "SmolLM3-3B-exl3_4.0bpw", "chat_template.jinja"),
    ("smollm3_tabby", "SmolLM3-3B-exl3_4.0bpw", "tabby_template.jinja"),
    ("qwen3_2507", "Qwen_Qwen3-4B-Instruct-2507-EXL3", None),
]

TOOL_SCHEMAS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]


def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


if transformers.__version__ != WANT_TRANSFORMERS:
    die(f"need transformers {WANT_TRANSFORMERS}, got {transformers.__version__} "
        "(run with ASDF_PYTHON_VERSION=3.14.3)")
if jinja2.__version__ != WANT_JINJA2:
    die(f"need jinja2 {WANT_JINJA2}, got {jinja2.__version__} "
        "(run with ASDF_PYTHON_VERSION=3.14.3)")


def fixed_strftime(fmt):
    return FIXED.strftime(fmt)


def m(role, content, **kw):
    d = {"role": role, "content": content}
    d.update(kw)
    return d


SYSTEM = m("system", "You are a helpful assistant. Answer briefly.")
USER = m("user", "What is the capital of France?")
ASST = m("assistant", "Paris is the capital of France.")
USER2 = m("user", "And what is its population?")
MULTI = [SYSTEM, USER, ASST, USER2]

TOOL_MESSAGES = [
    SYSTEM,
    m("user", "What's the weather in Paris?"),
    m("assistant", "", tool_calls=[{
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
    }]),
    m("tool", '{"temperature": 21, "unit": "celsius"}', tool_call_id="call_1"),
    m("user", "Thanks."),
]

CHAT_CASES = [
    {"name": "sys_multi_gen", "messages": MULTI, "add_generation_prompt": True},
    {"name": "sys_multi_nogen", "messages": MULTI, "add_generation_prompt": False},
    {"name": "no_system_gen", "messages": [USER, ASST, USER2], "add_generation_prompt": True},
    {"name": "single_turn_gen", "messages": [SYSTEM, USER], "add_generation_prompt": True},
    {"name": "empty_system_gen", "messages": [m("system", ""), USER], "add_generation_prompt": True},
    {"name": "unicode_gen", "messages": [
        SYSTEM,
        m("user", "héllo — 日本語 مرحبا 🚀 naïve straße?"),
        m("assistant", "Café ☕ : Ünïcode OK ✔"),
        m("user", "Danke!🙏"),
    ], "add_generation_prompt": True},
    {"name": "special_chars_gen", "messages": [
        SYSTEM,
        m("user", 'She said "hi" & <tag> \'single\' \\ back\\slash\n'
                  "newline\ttab | pipe $var {{jinja}} %s %d {# not a comment #}"),
        ASST,
        m("user", "Continue, please."),
    ], "add_generation_prompt": True},
    {"name": "tool_turns", "messages": TOOL_MESSAGES, "add_generation_prompt": True,
     "tools": TOOL_SCHEMAS},
    {"name": "thinking_off", "messages": MULTI, "add_generation_prompt": True,
     "enable_thinking": False},
]

SNIPPETS = [
    {"name": "and_or_operand",
     "template": '{{ 1 and "b" }}|{{ 0 or "z" }}|{{ "" and "x" }}|{{ "a" or 2 }}|{{ 3 and 4 }}|{{ 0 and 4 }}',
     "vars": {}},
    {"name": "eq_ne",
     "template": '{{ 1 == 1 }}|{{ 1 == 1.0 }}|{{ true == 1 }}|{{ "1" == 1 }}|{{ none == 0 }}|{{ 1 != 2 }}|{{ "a" == "a" }}',
     "vars": {}},
    {"name": "ordered_comparisons",
     "template": '{{ 1 < 2 }}|{{ 2 <= 2 }}|{{ "b" > "a" }}|{{ 3 >= 4 }}|{{ 1.5 < 2 }}',
     "vars": {}},
    {"name": "membership",
     "template": '{{ "b" in "abc" }}|{{ 2 in lst }}|{{ "x" not in "abc" }}|{{ "k" in obj }}|{{ 4 in lst }}',
     "vars": {"lst": [1, 2, 3], "obj": {"k": 1}}},
    {"name": "concat_tilde",
     "template": '{{ "x" ~ 1 ~ true ~ none }}|{{ "v=" ~ 1.5 }}',
     "vars": {}},
    {"name": "arith_int64",
     "template": "{{ 1 + 2 }}|{{ 10 - 3 }}|{{ 1 + 1.5 }}|{{ 2 - 0.5 }}",
     "vars": {}},
    {"name": "set_local",
     "template": "{% set x = 7 %}{{ x }}{{ x + 1 }}",
     "vars": {}},
    {"name": "namespace_attr_set",
     "template": "{% set ns = namespace(n=1) %}{% set ns.n = ns.n + 41 %}{{ ns.n }}",
     "vars": {}},
    {"name": "tojson_order_utf8",
     "template": "{{ obj | tojson }}",
     "vars": {"obj": {"b": 1, "a": "ü", "n": [1, 2], "s": "a'b"}}},
    {"name": "ternary_undefined",
     "template": '{{ "y" if flag else "n" }}|{{ "z" if missing_var else "n" }}',
     "vars": {"flag": True}},
    {"name": "filters_length_string",
     "template": '{{ "abc" | length }}|{{ 42 | string }}|{{ 3.5 | string }}',
     "vars": {}},
    {"name": "loop_index",
     "template": "{% for i in lst %}{{ loop.index0 }},{% endfor %}",
     "vars": {"lst": [1, 2, 3]}},
    {"name": "tests_defined_none",
     "template": "{{ z is defined }}|{{ y is defined }}|{{ x is none }}",
     "vars": {"x": None, "z": 1}},
    {"name": "strftime_fixed_clock",
     "template": 'Today: {{ strftime_now("%d %B %Y") }}',
     "vars": {}},
    {"name": "whitespace_control",
     "template": "{% if true %}\n  hello\n{% endif %}\n{%- if true %}\nworld{% endif %}",
     "vars": {}},
    {"name": "generation_tag",
     "template": 'A{% generation %}B{{ "C" }}{% endgeneration %}D',
     "vars": {}},
]


def load_tokenizer(subdir):
    from transformers import AutoTokenizer

    path = REPO_MODELS / subdir
    if not path.is_dir():
        die(f"missing local model dir: {path}")
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def main():
    entries = []
    for name, subdir, fname in CHAT_TEMPLATES:
        model_dir = REPO_MODELS / subdir
        tok = load_tokenizer(subdir)
        if fname is not None:
            src = (model_dir / fname).read_text(encoding="utf-8")
        else:
            src = json.loads((model_dir / "tokenizer_config.json").read_text())["chat_template"]
        cases, skipped = [], []
        for case in CHAT_CASES:
            kwargs = {k: v for k, v in case.items() if k not in ("name", "messages")}
            try:
                out = tok.apply_chat_template(
                    case["messages"], tokenize=False, chat_template=src,
                    strftime_now=fixed_strftime, **kwargs)
            except Exception as e:  # template genuinely rejects this input
                skipped.append({"case": case["name"],
                                "error": f"{type(e).__name__}: {e}"})
                continue
            cases.append({
                "name": case["name"],
                "vars": {"messages": case["messages"], **kwargs},
                "expected": out,
            })
        entry = {"name": name, "kind": "chat", "template": src, "cases": cases}
        if skipped:
            entry["skipped"] = skipped
        entries.append(entry)

    # Expression snippets: one synthetic entry each; they ignore `messages`.
    tok = load_tokenizer("Qwen_Qwen3-4B-Instruct-2507-EXL3")
    for snip in SNIPPETS:
        try:
            out = tok.apply_chat_template(
                [m("user", "")], tokenize=False, add_generation_prompt=False,
                chat_template=snip["template"], strftime_now=fixed_strftime,
                **snip["vars"])
        except Exception as e:
            die(f"snippet {snip['name']} failed in oracle: {type(e).__name__}: {e}")
        entries.append({
            "name": f"snippet_{snip['name']}",
            "kind": "snippet",
            "template": snip["template"],
            "cases": [{"name": "render", "vars": snip["vars"], "expected": out}],
        })

    fixture = {
        "meta": {
            "fixture_version": 1,
            "generator": "internal/runner/testdata/gen_golden.py",
            "command": "ASDF_PYTHON_VERSION=3.14.3 python3 internal/runner/testdata/gen_golden.py",
            "transformers": transformers.__version__,
            "jinja2": jinja2.__version__,
            "python": platform.python_version(),
            "fixed_clock": "2026-01-01T12:00:00 (strftime_now kwarg override)",
            "oracle": "AutoTokenizer.apply_chat_template(tokenize=False), local_files_only=True, no downloads",
        },
        "entries": entries,
    }
    OUT.write_text(json.dumps(fixture, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    n = sum(len(e["cases"]) for e in entries)
    nk = sum(len(e.get("skipped", [])) for e in entries)
    print(f"wrote {OUT.name}: {len(entries)} entries, {n} cases, {nk} skipped")


if __name__ == "__main__":
    main()
