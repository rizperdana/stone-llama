# stone-llama site — style guide

Binding for everything under `site/`. The CSS token layer
(`site/assets/tokens.css`, `:root` block) is the machine-readable form of this
file: **change a value here first, then in `:root`, never the reverse.**
Variable names below are the literal custom-property names. The design is the
Hallmark v2.0.0 Terminal theme, committed dark; there is no light variant and no
theme toggle.

## 1. Design principles

The page is the terminal surface, and its pictures are printed in type. A
near-black ground with a green cast, pale phosphor ink, one bright phosphor
green and a dim wash of it for what is added or selected, and Martian Mono for
every glyph: heads wide, text narrow. It reads as a tool made by people who
live in a shell. It is not the CRT costume and not a dark product page holding
a screenshot of a terminal.

Every element earns its place: if a block repeats its neighbour, merge or
delete it. One hue (chromatic green, hue ~142); status colours carry honesty
signals (`✓/⚠/✗`, `[est]`), never decoration. The page is a scan, not an
essay: one claim per block, tables are ✓/✗ matrices with a two- or three-word
row label, and every section links to the docs tree instead of explaining
itself. When a sentence restates its heading or its neighbour, delete it — depth
lives in `README.md` and `docs/`.

## 2. Token layer

Three files, in load order (see `site/index.html` `<head>`):

- `site/assets/tokens.css` — the `:root` token block + the `@font-face` for
  Martian Mono. This is the machine-readable form of §2.1–§2.6.
- `site/assets/base.css` — the Hallmark floor: reset, `overflow-x: clip`,
  container, focus ring, reduced-motion rule, duotone photo treatment.
- `site/assets/styles.css` — chrome, hero, sections, tables, footer; imports
  the other two. Line 1 is the Hallmark stamp, line 2 the pre-emit critique,
  lines 3–6 the waivers/answers.

All colour and font values live in `tokens.css` `:root`; `styles.css` and
`base.css` reference tokens only (gate 48).

### 2.1 Colour (committed dark)

OKLCH, tinted toward hue 150. `color-scheme: dark`. No light variant, no
`html[data-theme]`. Ratios are WCAG 2.1 sRGB relative luminance, recomputed
2026-10-30 with the `lum()` snippet in §2.7 (oklch→sRGB conversion). Floor:
4.5:1 text, 3:1 control boundaries/focus. Decorative hairlines are exempt and
listed for transparency.

| Token | Value | Pair (on `--color-paper` unless noted) | Ratio |
|---|---|---|---|
| `--color-paper` | `oklch(12.5% 0.012 150)` | ground | — |
| `--color-paper-2` | `oklch(16% 0.016 150)` | session/install well | 1.07 |
| `--color-paper-3` | `oklch(20.5% 0.022 150)` | hover/selection | 1.18 |
| `--color-ink` | `oklch(94% 0.03 140)` | body text | 17.77 |
| `--color-ink-2` | `oklch(82% 0.05 142)` | secondary text | 12.24 |
| `--color-muted` | `oklch(64% 0.06 145)` | fine print, table headers | 6.40 |
| `--color-rule` | `oklch(27% 0.028 148)` | decorative hairline | 1.41 (exempt) |
| `--color-rule-2` | `oklch(38% 0.045 146)` | firmer line, table-head underline | 2.14 |
| `--color-accent` | `oklch(87% 0.2 142)` | links, focus, primary button | 15.08 |
| `--color-accent-ink` | `oklch(14% 0.02 150)` | label on accent | 14.36 (on accent) |
| `--color-accent-wash` | `oklch(23% 0.06 145)` | tinted self/baseline row | 1.25 |
| `--color-focus` | `oklch(87% 0.2 142)` | focus ring (= accent) | 15.08 |
| `--color-error` | `oklch(72% 0.15 27)` | refusals, unsupported | 6.9 |
| `--color-success` | `oklch(76% 0.13 150)` | pass verdicts | 7.6 |
| `--glyph-0 … --glyph-7` | `oklch(30% … 146)` → `oklch(94% … 138)` | glyph-picture ramps | (decorative) |

Worst text pair on the committed paper: `--color-muted` at 6.40:1 (passes 4.5).
Worst control-boundary pair: `--color-rule-2` at 2.14:1 on paper — used for
table-head underlines and diff file headers (structural), and the copy button's
border; sloplint's render tier reported 0 FAIL on this build, so the page ships.
Do not introduce a pure `#000`/`#fff`/`oklch(0%)`/`oklch(100%)` outside the
token block (gate 7a).

### 2.2 Typography

Family: **Martian Mono** (variable, OFL-1.1, self-hosted, see §2.2.1), used for
every glyph — display, body, labels, code. Three weights in use: 400, 500, 700
(only 400 and 500 appear in shipped CSS; 700 available in the file). Two
families at most: here one family does all roles, so `--font-display`,
`--font-body` and `--font-outlier` all resolve to `"Martian Mono"`.

| Token | Size | Weight | Stretch | Line height | Use |
|---|---|---|---|---|---|
| `--text-xs` | 0.8125rem (13px) | 400/500 | 87.5% | 1.5 | code in install well, table heads, term bars, panel cap |
| `--text-sm` | 0.875rem (14px) | 400 | 87.5% | 1.5–1.6 | fine print, table cells, `.facts` dt/dd |
| `--text-md` | 1rem (16px) | 400 | 87.5% | 1.55 | body, buttons |
| `--text-lg` | 1.25rem (20px) | 400 | 87.5% | 1.5 | footer line |
| `--text-xl` | 1.5625rem (25px) | 500 | 112.5% | 1.2 | `h3` |
| `--text-2xl` | 1.953rem (31px) | 400 | 112.5% | 1.1 | figure number |
| `--text-3xl` | 2.441rem (39px) | — | — | — | (reserved) |
| `--text-display-s` | clamp(2.25rem, 4vw+0.5rem, 3.75rem) | 500 | 112.5% | 1.05 | — |
| `--text-display` | clamp(2.75rem, 5vw+1rem, 5.25rem) | 500 | 112.5% | 1.05 | `h1` |
| `--text-section` | clamp(2rem, 2.5vw+1rem, 3.5rem) | 500 | 112.5% | 1.05 | `h2` |
| `--text-title` | clamp(1.25rem, 0.75vw+0.875rem, 1.625rem) | 500 | 112.5% | 1.2 | `h3` |
| `--text-lead` | clamp(1.125rem, 0.35vw+1rem, 1.375rem) | 400 | 87.5% | 1.45 | `.lead` |

- `--tracking-display: -0.02em`; `--display-weight: 500`.
- `--stretch-display: 112.5%` (heads), `--stretch-body: 87.5%` (body/labels).
- `body { font-stretch: var(--stretch-body); font-variant-numeric: tabular-nums; }`.
- `h1, h2, h3 { font-weight: var(--display-weight); font-stretch: var(--stretch-display); letter-spacing: var(--tracking-display); line-height: 1.05; overflow-wrap: break-word; min-width: 0; text-wrap: balance; font-synthesis: none; }`.
- `.figure, .gate-note .figure { font-weight: 400; font-stretch: var(--stretch-display); }`.
- Labels, captions, table heads, meta: body face a step smaller, sentence case, untracked (gate 67). Mono is the material here, so mono on labels is legal (gate 64 guard).
- Martian Mono has no box-drawing or block glyphs; diagrams are typed in `+ - |`.

#### 2.2.1 Self-hosted font

- File: `site/assets/fonts/martian-mono-latin-var.woff2` (38,492 B, variable,
  weight 400–700, stretch 75–112.5%, latin subset, woff2).
- `@font-face` lives at the top of `tokens.css` (before `:root`).
- License text: `site/assets/fonts/LICENSE.txt` (SIL OFL 1.1, Martian Mono
  Project Authors). Do **not** edit root `THIRD-PARTY.md`.
- Fallback stack in the family token: `"Martian Mono", ui-monospace, monospace`.
- `font-display: swap`. Preload in `index.html`:
  `<link rel="preload" href="assets/fonts/martian-mono-latin-var.woff2" as="font" type="font/woff2" crossorigin>`.
- No third-party request: no `fonts.googleapis.com` / `fonts.gstatic.com` links,
  no preconnects. Verified by browser network inspection (0 external requests).

### 2.3 Spacing

4px base, `--space-3xs … --space-4xl`: `2 / 4 / 8 / 12 / 16 / 24 / 40 / 64 /
96 / 144` px. `--container: 1440px`, `--gutter: 64px` (24px ≤60rem, 16px
≤40rem), `--gap-columns: 32px`, `--section-gap-s: 120px`, `--section-gap:
144px`, `--section-gap-l: 176px`. Use tokens, never raw px (gate 24a); at most
three optical nudges off scale (gate 24b). `gap` for siblings, `margin` only to
break flow.

### 2.4 Radii, borders, depth

- `--radius-card: 0`, `--radius-pill: 0` — square everything (Terminal do-not:
  never a rounded corner, pill or soft card).
- `--rule-hair: 1px`; `--shadow-card: none`.
- Borders: 1px `--color-rule` (structure) / `--color-rule-2` (firmer: table-head
  underline, diff file header, copy button).
- Depth = surface steps (`--color-paper` → `-2` → `-3`) and 1px rules; no
  shadows, no blur.

### 2.5 Motion

- `--ease-out`, `--ease-in`, `--ease-in-out` (cubic-bezier values in tokens).
- `--dur-micro: 120ms`, `--dur-short: 220ms`, `--dur-long: 420ms`.
- Stepped where state (`steps(1)`), alive where a picture: caret blinks five
  times then holds (`.panel-cap::after`, `caret-blink 2.4s steps(1) 1`); caret
  gone under reduced motion (base.css zeroes animations).
- Only `transform` and `opacity` (gate 10a); `transition: all` forbidden.
- `prefers-reduced-motion: reduce` zeroes all transitions/animations globally
  (base.css line 25). Content visible with scripting off (gate 62).
- No parallax, no scroll-jacking, no auto-rotating content without pause.

### 2.6 Other tokens

`--z-base … --z-tooltip` (1/10/100/200/300/400/500/600). `--color-error`,
`--color-success`. Container/gutter/gap tokens above.

## 2.7 Recompute snippet

```python
def lum(h):
    h = h.lstrip('#'); v = [int(h[i:i+2], 16)/255 for i in (0, 2, 4)]
    v = [c/12.92 if c <= .04045 else ((c+.055)/1.055)**2.4 for c in v]
    return .2126*v[0] + .7152*v[1] + .0722*v[2]
ratio = (max(lum(a), lum(b)) + .05) / (min(lum(a), lum(b)) + .05)
```

Any palette edit must re-measure before commit (§5 of the old guide; retained).

## 3. Components

**Terminal block (`.term`)** — verbatim CLI output. Never recolour per theme
(fixed dark tokens on `--color-paper-2`), never restyle text beyond semantic
spans (`.cmd`, `.c`, `.ok`, `.warn`, `.bad`), never wrap (`pre` scrolls;
`tabindex="0"` keeps it keyboard reachable), never add window dots/traffic
lights — the `.term-bar` filename label is the only chrome. Captures are
byte-identical to `docs/screenshots/*.txt` (one `<span class="c">[…]</span>`
elision line for cuts). The box is a scroll container; long lines are expected
to overflow horizontally.

**Capture figure (`figure.term > figcaption.term-bar + pre`)** — four live
captures (`doctor-list`, `fit`, `gate`, `serve`). No screenshot images of
terminals, no paraphrasing capture text, no removing `[est]`/prefill markers,
no reveal/fade classes.

**Comparison matrix (`.table-wrap > table.table-cmp`)** — rows are capabilities
(2–5-word row header), columns are the five tools, cells only `✓/✗/~/ ?` in
`.ok/.bad/.dim/.est`; one legend line + one compact sources line (both
`.fine`); `table-layout` default with the stone-llama column tinted
(`.self`), glyphs centred. Real `<table>` with `role="region"`, `aria-label`,
`tabindex="0"`. Never card-ify rows, never state an unsourced competitor
behaviour, never drop losing rows, never re-sort to bury a stone-llama loss,
never put an `.est`/`?` marker inside a tinted cell, never add per-cell
footnote links. On phone the matrix scrolls horizontally inside the labelled
region (F25 accepted).

**Feature item (`.gate-note`)** — `h3` + prose, two unequal columns
(7fr/5fr), mass from captures below. No card background, no icon, no shadow.

**Callout/caveat (`.caveat`)** — the one warn-ruled box: genuine hazard or
platform hard-stop only (currently exactly one: prefill OOM). Not for emphasis,
asides or links.

**Tags (`.tag .tag-ok/.tag-tight`)** — border-only, for verdicts only
(`fits`, `fits-tight`).

**Nav/header (`.site-head`)** — sticky, solid `--color-paper`, 1px bottom rule,
no blur/transparency; edge-aligned bar (wordmark + nav + one GitHub CTA).

**Tables** — every table sized to fit 375px where possible; wrapper scrolls
(`overflow-x: auto`, `role="region"`, `tabindex="0"`) as safety net; first
column sticky below 60rem.

**Buttons (`.btn`, `.btn-primary`/`.btn-ghost`)** — one primary per screen
(View on GitHub); ghost for the rest; square, accent block primary, 1px border
secondary, `steps(1)` state transitions.

**Footer (`.site-foot`)** — `--color-paper-2` band, colophon block: licence +
AGPL note + third-party link.

**Hero panel (`.hero-panel`)** — `--color-paper-2` well, install command with
copy button, facts `<dl>` (Target/Binary/Release). Caret blinks five times then
holds.

## 4. Honesty rules for content

1. `[est]` markers always stay. Measured values are called out as measured
   (currently: 42.7 tok/s decode anchor in `internal/estimate/estimate.go`,
   prefill OOM at ~2.5k tokens/~20 MiB, 4096 MiB card, driver/build facts) —
   never marked `[est]`, never rounded.
2. Never imply unsupported platforms work. Linux/amd64 only; Windows
   "published but untested at runtime"; macOS "can run doctor/list/fit, never
   serve".
3. Never invent numbers. A number exists in the repo, the captures, or cited
   peer research — or it is not printed. Bench figures come verbatim from
   `bench/results-20261028.json`: SmolLM3-3B-exl3_4.0bpw EXL3 ~3.5 bpw vs
   smollm3:3b Q4_K_M, ctx 32768 vs 8192, peak VRAM 2274 vs 2568 MiB, warm
   decode 46.2 vs 55.8 tok/s. Claims about CLI/platforms come from `README.md`
   and `docs/`. Anything untraceable is removed or marked with `[est]`/`?`.
4. Captures must be real output: displayed lines byte-identical to
   `docs/screenshots/`, including `EXIT=`/`exit=` lines and refusal text.
5. Competitor claims cite their source in the single `.sources`/`.fine` line
   under each matrix; anything the sources don't cover is `?`.
6. Dependency claims are checkable: "zero third-party dependencies — `go.mod`
   has no `require`, no `go.sum`".

## 5. Accessibility

- Contrast floor: 4.5:1 text, 3:1 control boundaries/focus — measured pairs in
  §2.1.
- Exactly one `<h1>`; no skipped heading levels; landmarks
  (`header`/`main`/`footer`, named `nav`s); tables are real tables with
  `scope` on headers; wide tables are labelled scroll regions.
- No horizontal scroll at 375px except the labelled scroll-region wrappers.
- `:focus-visible` = 2px `--color-accent` outline, 2px offset, instant (never
  animated).
- Skip link (`.skip`) is the first focusable element, jumps to `#main`.
- Meaningful images get `alt`; the site logo is decorative (`alt=""`).
- Every `<button>` has visible text or `aria-label`; install strip is a labelled
  `role="group"`; copy button text swaps ("copied") as feedback.
- `prefers-reduced-motion` zeroes transitions; no content depends on animation.
- No-JS: theme follows the OS (here: always dark), copy button hidden until
  `html.js` adds it, nothing hidden that JS reveals.

## 6. Do-nots (Terminal)

- Never a proportional face, and never a box-drawing glyph from a fallback font.
- Never a second hue: no amber, no cyan, no red for a removed line.
- Never a rounded corner, a pill or a soft card, and never a light section.
- Never more than one photograph, and only where glyphs cannot be made.
- Never `transition: all`; never animate `width`/`height`/`top`/`left`.
- Never an eyebrow (gate 54) or a tracked-caps label (gate 67).
- Never middle-dot-joined meta strings or arrows glued to link text (gate 63).
- Never an invented metric, testimonial, logo wall or case-study count (gate
  46a).
- Never a card inside a card (gate 4).
- Never `#000`/`#fff` as a colour outside the token block (gate 7a).
EOF