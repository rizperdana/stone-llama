# stone-llama site — style guide

Binding for everything under `site/`. The CSS token layer
(`site/assets/styles.css`, `:root` block) is the machine-readable form of this
file: **change a value here first, then in `:root`, never the reverse.**
Variable names below are the literal custom-property names.

## 1. Design principles

**Engineering-honest spec-sheet, not a SaaS landing page.** The page's identity
is real terminal output plus tabular, monospaced fact — the same voice as the
CLI. **The visual identity must not read as AI-generated**: no gradients,
blobs, glows, glassmorphism, drop-shadow depth, emoji, decorative animation, or
feature-card grids with icons; no superlatives or invented numbers in copy
either. Every element earns its place: if a block repeats its neighbour,
merge or delete it. One accent colour; status colours exist only to carry
honesty signals (`✓/⚠/✗`, `[est]`), never decoration.

The page is a scan, not an essay: one claim per block, tables are ✓/✗
matrices with a two- or three-word row label, and every section links to the
docs tree instead of explaining itself. When a sentence restates its heading
or its neighbour, delete it — depth lives in `README.md` and `docs/`.

## 2. Token layer

### 2.1 Colour

Dark is the default (`:root`); light is applied by `html[data-theme="light"]`
and, without JS, by `@media (prefers-color-scheme: light)` on
`:root:not([data-theme])`. Terminal captures use a **fixed dark palette in both
themes** — they are verbatim output from a dark terminal.

Ratios are WCAG 2.1 (sRGB relative luminance, alpha composited over backdrop),
measured 2026-09-26. Floor: **4.5:1 text, 3:1 control boundaries/focus.**
Decorative hairlines (section rules, table row separators — no control or
state conveyed) are exempt from 1.4.11 and listed for transparency.

| Token | Dark | Light | Pair (on `--bg` unless noted) | Ratio |
|---|---|---|---|---|
| `--bg` | `#0c1013` | `#f6f4ef` | ground | — |
| `--bg-raise` | `#12171b` | `#efece4` | footer/anchored surfaces | — |
| `--panel` | `#161c21` | `#eae7de` | table header surface | — |
| `--fg` | `#e9eef1` | `#14191d` | body text | 16.34 / 16.10 |
| `--fg-2` | `#c5ced4` | `#39434a` | secondary text | 11.97 / 9.20 |
| `--muted` | `#97a3ab` | `#5a646b` | fine print, table headers (on `--panel`: 6.66 / 4.89) | 7.41 / 5.51 |
| `--accent` | `#4cc3d6` | `#0d6b70` | links, eyebrows, focus ring | 9.17 / 5.70 |
| `--accent-ink` | `#06242a` | `#ffffff` | label on `--accent` (buttons) | 7.80 / 6.26 |
| `--ok` | `#5fca8e` | `#146b45` | pass verdicts | 9.39 / 5.94 |
| `--warn` | `#eab45f` | `#7a5100` | warnings, caveat rule | 10.18 / 6.36 |
| `--bad` | `#f28b7d` | `#a33520` | refusals, unsupported | 7.97 / 6.19 |
| `--est` | `#8fb4c4` | `#3a6a8a` | `[est]` marker (on `--panel`: 7.77 / 4.71) | 8.64 / 5.29 |
| `--line` | `#232b31` | `#d5d0c4` | decorative hairline | 1.33 / 1.40 *(exempt)* |
| `--line-strong` | `#606a71` | `#8a867a` | control borders (on `--bg-raise`: 3.26 / 3.08) | 3.45 / 3.31 |
| `--row-tint` | `color-mix(#4cc3d6 12%, #12171b)` | `color-mix(#0d6b70 9%, #efece4)` | tinted rows (worst text on tint: `--muted` = 5.63 / 4.52) | — |

Terminal (both themes, `:root`): `--term-bg #070b0e`, `--term-bar #0e1418`,
`--term-line #1c262d` *(decorative, 1.28)*, `--term-fg #cfd8dc` (13.65),
`--term-cmd #74cfe0` (11.05), `--term-c #8497a1` (6.51; on bar 6.12),
`--term-ok #62cc8e` (9.92), `--term-warn #f0b45f` (10.72),
`--term-bad #ff9a8a` (9.63).

Tints: `--accent-soft` (inline `code`, alpha) and `--row-tint` (self/baseline
rows, opaque so sticky cells don't bleed). Code chips inside tinted rows use
`--panel` — accent-on-accent over the darker tint measures 4.13:1 in light.
Tags and callouts use `color-mix(in srgb, …)` on top of existing surfaces —
never a new flat hex. **Worst gated pair: 3.08:1** (light control border on
`--bg-raise`). **Worst rendered text pair: 4.52:1** (light `--muted` on
`--row-tint`). One combination is deliberately unused: `--est` on `--row-tint`
would be 4.35:1 in light, so `?`/`[est]` markers never sit in a tinted cell
(`.self`, `.row-anchor`) — no current cell does.

Recompute any pair:

```python
def lum(h):
    h = h.lstrip('#'); v = [int(h[i:i+2], 16)/255 for i in (0, 2, 4)]
    v = [c/12.92 if c <= .04045 else ((c+.055)/1.055)**2.4 for c in v]
    return .2126*v[0] + .7152*v[1] + .0722*v[2]
ratio = (max(lum(a), lum(b)) + .05) / (min(lum(a), lum(b)) + .05)
```

### 2.2 Type scale

Two families only: `--sans` (system UI stack), `--mono` (system mono stack).
No web fonts, no `@font-face`.

| Token | Size | Weight | Line height | Use |
|---|---|---|---|---|
| `--fs-xs` | 0.75rem | 600 | 1.4 | eyebrows, table headers, tags, term bars, footnotes |
| `--fs-sm` | 0.875rem | 400 | 1.6 | fine print, table cells (≥561px), terminal pre, buttons (600) |
| `--fs-base` | 1rem | 400 | `--lh-body` (1.6) | body, lede paragraphs |
| `--fs-lg` | 1.125rem | 400 | 1.55 | `.lede` |
| `--fs-xl` | 1.375rem | 600 | `--lh-tight` (1.15) | `h3` |
| `--fs-2xl` | `clamp(1.5rem, 3.2vw, 2rem)` | 600 | `--lh-tight` | `h2` |
| `--fs-3xl` | `clamp(2rem, 5vw, 3rem)` | 600 | `--lh-tight` | `h1` |

Fixed micro-sizes outside the scale, each with one job: `0.9em` (inline
`code`), `0.85em` (`.est`), `1.75rem` (`.stat-n`), `0.625rem`
(comparison-matrix column headers, ≤560px only — five tool names must fit
375px).

Weights in use: 400 (body), 600 (headings, labels, buttons, verdicts), 700
nowhere — emphasis is `strong` at 600 or a colour token. Letter-spacing: only
`0.14em`/`0.08em` on uppercase mono labels, `-0.01em`/`-0.02em` on headings.
Do not introduce a size or weight outside this table without editing it here.

### 2.3 Spacing

4px base, `--sp-1 … --sp-8`: `0.25 / 0.5 / 0.75 / 1 / 1.5 / 2 / 3 / 4 rem`.
Section rhythm: `.sect { padding-block: clamp(2.75rem, 7vw, 5rem) }`; hero
`clamp(2.5rem, 6vw, 4.5rem)`. Table cell padding `--sp-3 --sp-4` (`--sp-2`
6px at ≤560px). Use tokens, never raw px, for anything vertical.
never raw px, for anything vertical.

### 2.4 Radii, borders, depth

- `--r-sm` 4px: buttons, code, tags, toggle, install strip.
- `--r-md` 8px: terminal blocks, callouts, table wraps.
- Borders: 1px `--line` (structure) / 1px `--line-strong` (anything you click
  or that can focus). Two widths only: 1px, plus the caveat's 3px `--warn`
  left rule (the single deliberate accent rule on the page).
- **No box-shadows. No gradients. Depth = surface change** (`--bg` →
  `--bg-raise` → `--panel`) + a hairline.

### 2.5 Motion

`--dur: 0.15s`, `ease`, and only on colour/border/background of interactive
elements. Nothing animates on scroll or load; no transforms, no parallax.
`@media (prefers-reduced-motion: reduce)` zeroes all transitions globally
(including smooth scrolling, which is otherwise gated behind
`prefers-reduced-motion: no-preference`). Adding a keyframe animation requires
editing this section first.

## 3. Components

**Terminal block (`.term`)** — use for verbatim CLI output. What not to do:
never recolour per theme (fixed dark tokens), never restyle the text inside
beyond the semantic spans (`.cmd`, `.c`, `.ok`, `.warn`, `.bad`), never wrap
(`pre` scrolls; `tabindex="0"` keeps it keyboard reachable), never add window
dots/traffic lights — the `.term-bar` filename label is the only chrome.

**Capture figure (`figure.term > figcaption.term-bar + pre`)** — use for the
four live captures (`doctor-list`, `fit`, `gate`, `serve`). What not to do: no
`screenshot` images of terminals, no paraphrasing capture text, no removing
`[est]`/`calibration pending` markers, no `reveal`/fade classes (content is
never hidden pending JS). Length is cut only by eliding a contiguous block:
one `<span class="c">[…]</span>` line replacing the cut lines; every
displayed line stays byte-identical to `docs/screenshots/*.txt`.

**Comparison matrix (`.table-wrap > table.table-cmp`)** — rows are
capabilities (2–5-word row header), columns are the five tools, cells are only
`✓`/`✗`/`~`/`?` in `.ok`/`.bad`/`.dim`/`.est`; one legend line + one compact
sources line (both `.fine`); `table-layout: fixed` with the stone-llama
column tinted (`.self`), glyphs centred. Still a real `<table>` with
`role="region"`, `aria-label` and `tabindex="0"`. What not to do: never
card-ify rows (comparison needs aligned rows), never state an unsourced
competitor behaviour, never drop the losing rows, never re-sort to bury a
stone-llama loss, never put an `.est`/`?` marker inside a tinted cell (light
`--est` on `--row-tint` measures 4.35:1), never add per-cell footnote links —
the single sources line carries provenance.

**Feature item (`.feature`)** — top hairline `--line-strong` + `h3` + prose.
Use for the two mechanisms. What not to do: no card background, no icon, no
shadow, no more than one row of these per section.

**Callout/caveat (`.caveat`)** — the one warn-ruled box: genuine hazard or
platform hard-stop only (currently exactly one: prefill OOM). What not to do:
not for emphasis, not for asides, not for links, never a second callout style
or a "good news" variant — good news is plain prose.

**Tags (`.tag .tag-ok/.tag-tight`)** — border-only, `currentColor` at 45%,
uppercase mono, for verdicts only (`fits`, `fits-tight`). What not to do: no
filled backgrounds, no new variants beyond these two.

**Nav/header (`.site-head`)** — sticky, solid `--bg`, hairline bottom, no
blur/transparency; nav is muted links, which become a horizontally scrollable
strip below 860px (the section TOC never simply vanishes). What not to do: no
mega-menu, no animated underline, no second CTA in the header.

**Tables** — every table is sized to fit 375px (smaller type, 6px padding,
headers allowed to break anywhere), so the page has no horizontal scroll on a
phone. The wrapper still scrolls (`overflow-x:auto`, `role="region"`,
`tabindex="0"`) as a safety net below ~360px, where the first column turns
`position:sticky; left:0` (opaque background, right hairline) so row identity
survives; keep any first-cell background rules in sync when tinting rows.

**Buttons (`.btn`, `.btn-primary`/`.btn-ghost`)** — one primary per screen
(View on GitHub), ghost for the rest; `--r-sm`, no transform on hover.
Copy button and theme toggle are JS-only (`html.js`) and hidden otherwise.

**Footer (`.site-foot`)** — raised surface, three columns collapsing to one;
carries licence + AGPL note. What not to do: no newsletter/social rows, no
repeating the nav verbatim.

## 4. Honesty rules for content

1. **`[est]` markers always stay.** Every estimated number keeps `[est]` (or a
   `tag-tight`/`dim` qualifier); measured values are called out as measured in
   prose (currently: 42.7 tok/s, the prefill OOM at ~2.5k tokens/~20 MiB, the
   4096 MiB card, driver/build facts) — never marked `[est]`, never rounded
   into a cleaner number.
2. **Never imply unsupported platforms work.** Linux/amd64 is the only
   supported platform; Windows stays "published but untested at runtime";
   macOS stays "can run `doctor`/`list`/`fit`, never serve".
3. **Never invent numbers.** A number on this page exists in the repo, in the
   captures, or in the cited peer research — or it is not printed. No
   benchmarks beyond the one measured anchor; no invented percentages.
4. **Captures must be real output**: displayed lines byte-identical to
   `docs/screenshots/`, including `EXIT=`/`exit=` lines and refusal text;
   cutting for length is one `<span class="c">[…]</span>` elision line.
5. **Competitor claims cite their source** in the single `.sources` line under
   each matrix; anything the sources don't cover is `?`. No
   runtime-adoption/`update`/`uninstall` claims — none of that is in HEAD.
6. **Dependency claims are checkable**: "zero third-party dependencies —
   `go.mod` has no `require`, no `go.sum`". Never write "one dependency" or
   name a library the module doesn't have.

## 5. Accessibility requirements

- **Contrast floor**: 4.5:1 text, 3:1 control boundaries/focus — measured
  pairs in §2.1; any palette edit must re-measure before commit.
- **Structure**: exactly one `<h1>`; no skipped heading levels; landmarks
  (`header`/`main`/`footer`, named `nav`s); tables are real tables with
  `scope` on headers; wide tables are labelled scroll regions (`role=region`,
  `aria-label`, `tabindex=0`) so keyboards can scroll them.
- **No horizontal scroll at 375px**: all three tables are sized to fit
  (`≤ 375px` columns sum), verified at 1440/375 — only the scroll-region
  wrapper (below ~360px) may scroll, and it stays keyboard-scrollable.
- **Focus**: `:focus-visible` = 2px `--accent` outline, 2px offset, never
  removed; skip link (`.skip`) is the first focusable element and jumps to
  `#main`.
- **Images**: meaningful images get `alt` (the hero-adjacent icons are
  decorative next to their text label → `alt=""`).
- **Controls**: every `<button>` has visible text or `aria-label`; the install
  strip is a labelled `role="group"`; copy button text swaps ("copied") as
  feedback.
- **Motion**: `prefers-reduced-motion` zeroes transitions; no content depends
  on animation.
- **No-JS**: without JavaScript the theme follows the OS preference, the
  toggle/copy button hide themselves, and nothing is hidden that JS reveals —
  all content is plain HTML.

## 6. Cross-surface: terminal output conventions (conventions only)

*Conventions-only: this documents how the site renders CLI output so site and
CLI read as one product. The Go formatting code is out of scope for the site.*

- **Colour degrades when piped**: CLI colours are ANSI and disappear under
  redirection/pipes; the site's spans (`.cmd`, `.ok`, `.warn`, `.bad`, `.c`)
  are the same roles. Text alone must carry the meaning — glyphs (`✓ ⚠ ✗`),
  words (`refused`, `warning`), and `EXIT=n` are present in monochrome, colour
  only reinforces. Never encode a verdict in colour alone.
- **Table alignment**: CLI tables are left-aligned columns, uppercase mono
  headers, one space padding; the site mirrors this in `thead th` (mono,
  uppercase, `--fs-xs`) and `.num` cells (`tabular-nums`, `white-space:
  nowrap`) so numbers stack exactly as they do in the terminal.
- **`fit` verdict glyphs**: `✓` proceed (arch/quant/fit pass), `⚠`
  proceed-with-warning (thin margin / reduced ctx — read the printed
  arithmetic), `✗` refused (nothing fits — the message includes the largest ctx
  that *would* fit and the full breakdown). Site verdict tags (`fits`,
  `fits-tight`) and these glyphs use the same `--ok`/`--warn`/`--bad` roles.
- **`[est]`**: the tool's own estimate marker; reproduced verbatim everywhere,
  never promoted to a plain number.
