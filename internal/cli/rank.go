package cli

import (
	"flag"
	"fmt"
	"io"
	"net/url"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/rizperdana/stone-llama/internal/hf"
	"github.com/rizperdana/stone-llama/internal/pull"
	"github.com/rizperdana/stone-llama/internal/store"
)

// rankRow is one evaluated candidate; tier orders fit(0) < warn(1) <
// refuse(2) < error(3), decode speed breaks ties inside a tier.
type rankRow struct {
	repo    string
	quant   string
	verdict string
	decode  float64
	est     string
	estPre  string
	rating  string
	tier    int
	err     string
}

// runRank evaluates a candidate set on this GPU from metadata only and
// prints a fit/speed ordering. Ratings are external estimates, never
// measurements made by stone-llama.
func runRank(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("rank", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	collection := fs.String("collection", "", "HuggingFace collection id (owner/name) to rank")
	file := fs.String("file", "", "file of model refs, one per line")
	ratingsPath := fs.String("ratings", "", "file of capability ratings (external estimates)")
	if err := fs.Parse(args); err != nil || fs.NArg() > 0 {
		fmt.Fprintln(stderr, "usage: stone-llama rank (--collection <owner/name> | --file <refs.txt>) [--ratings <file>]")
		return 2
	}
	if (*collection == "") == (*file == "") {
		fmt.Fprintln(stderr, "usage: stone-llama rank (--collection <owner/name> | --file <refs.txt>) [--ratings <file>]")
		return 2
	}

	var refs []string
	if *collection != "" {
		var err error
		refs, err = collectionIDs(hfBaseURL(), hf.LoadToken(os.Getenv("HF_TOKEN"), tokenFilePath()), *collection)
		if err != nil {
			fmt.Fprintf(stderr, "stone-llama rank: %v\n", err)
			return 1
		}
	} else {
		data, err := os.ReadFile(*file)
		if err != nil {
			fmt.Fprintf(stderr, "stone-llama rank: %v\n", err)
			return 1
		}
		for _, line := range strings.Split(string(data), "\n") {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			refs = append(refs, line)
		}
		if len(refs) == 0 {
			fmt.Fprintf(stderr, "stone-llama rank: %s has no model refs\n", *file)
			return 1
		}
	}

	var ratings map[string]float64
	if *ratingsPath != "" {
		var err error
		ratings, err = parseRatings(*ratingsPath)
		if err != nil {
			fmt.Fprintf(stderr, "stone-llama rank: %v\n", err)
			return 1
		}
	}

	cfg, rep, ok := gpuPreamble("rank", stdout, stderr)
	if !ok {
		return 1
	}
	gpuName := rep.GPUs[0].Name

	rows := make([]rankRow, 0, len(refs))
	for _, ref := range refs {
		opts := dryOpts(cfg, rep)
		opts.Ref = ref
		opts.Quiet = true
		opts.Out = io.Discard
		opts.Stdin = strings.NewReader("")
		res, err := pull.Run(opts)
		if err != nil {
			fmt.Fprintf(stderr, "stone-llama rank: %s: %v\n", ref, err)
			rows = append(rows, rankRow{
				repo: ref, quant: "-", verdict: "error",
				est: "-", estPre: "-", rating: ratingCell(ratings, ref), tier: 3, err: err.Error(),
			})
			continue
		}
		est, estPre := estCells(res, gpuName)
		decode := -1.0
		if est != "-" {
			decode, _ = strconv.ParseFloat(est, 64)
		}
		tier := 2
		switch res.Verdict.Status {
		case store.VerdictOK:
			tier = 0
		case store.VerdictWarn:
			tier = 1
		}
		rows = append(rows, rankRow{
			repo:    res.RepoID,
			quant:   orDash(res.QuantLabel),
			verdict: formatVerdict(&res.Verdict),
			decode:  decode,
			est:     est,
			estPre:  estPre,
			rating:  ratingCell(ratings, res.RepoID),
			tier:    tier,
		})
	}

	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].tier != rows[j].tier {
			return rows[i].tier < rows[j].tier
		}
		return rows[i].decode > rows[j].decode
	})

	w := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "REPO\tQUANT\tVERDICT\tEST T/S\tEST PRE T/S\tRATING")
	for _, r := range rows {
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\n", r.repo, r.quant, r.verdict, r.est, r.estPre, r.rating)
	}
	w.Flush()
	fmt.Fprintln(stdout, "~ values are estimates [est] from metadata + GPU spec — not measured (calibration pending)")
	if *ratingsPath != "" {
		fmt.Fprintf(stdout, "* RATING values from %s are external estimates, not measurements by stone-llama\n", *ratingsPath)
	}

	errs := 0
	for _, r := range rows {
		if r.tier == 3 {
			errs++
		}
	}
	if len(rows) == 0 || errs == len(rows) {
		return 1
	}
	return 0
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

// ratingCell formats a matched external rating for one repo.
func ratingCell(ratings map[string]float64, repoID string) string {
	if len(ratings) == 0 {
		return "-"
	}
	if v, ok := ratingLookup(ratings, repoID); ok {
		return fmt.Sprintf("%.1f *", v)
	}
	return "-"
}

// ratingLookup matches a repo id against rating keys: exact id, then
// basename, then a ≥10-char prefix either way (repos often carry
// "-exl3_4.0bpw" suffixes the ratings file won't spell out).
func ratingLookup(ratings map[string]float64, repoID string) (float64, bool) {
	id := strings.ToLower(repoID)
	if v, ok := ratings[id]; ok {
		return v, true
	}
	base := id
	if i := strings.LastIndex(id, "/"); i >= 0 {
		base = id[i+1:]
	}
	if v, ok := ratings[base]; ok {
		return v, true
	}
	for key, v := range ratings {
		if len(key) >= 10 && (strings.HasPrefix(base, key) || strings.HasPrefix(key, base)) {
			return v, true
		}
	}
	return 0, false
}

// collectionIDs lists model ids in a HuggingFace collection
// (owner/name), following pagination (≤5 pages, deduped).
func collectionIDs(baseURL, token, id string) ([]string, error) {
	if !hf.ValidRepoID(id) {
		return nil, fmt.Errorf("invalid collection id %q (want owner/name)", id)
	}
	client := hf.NewClient(baseURL, token)
	type page struct {
		Items []struct {
			Item struct {
				ID   string `json:"id"`
				Type string `json:"type"`
			} `json:"item"`
		} `json:"items"`
		Next         string `json:"next"`
		Continuation string `json:"continuation"`
	}
	var ids []string
	seen := map[string]bool{}
	next := "/api/collections/" + id + "?limit=100"
	for hops := 0; next != "" && hops < 5; hops++ {
		var p page
		if err := client.GetJSON(next, &p); err != nil {
			return nil, fmt.Errorf("fetch collection %s: %w", id, err)
		}
		for _, it := range p.Items {
			switch {
			case it.Item.ID == "":
				continue
			case it.Item.Type != "" && it.Item.Type != "model":
				continue
			}
			if !seen[it.Item.ID] {
				seen[it.Item.ID] = true
				ids = append(ids, it.Item.ID)
			}
		}
		switch {
		case p.Next != "":
			resolved, err := url.Parse(p.Next)
			if err != nil {
				return nil, fmt.Errorf("collection %s: bad pagination link %q", id, p.Next)
			}
			if resolved.IsAbs() {
				next = resolved.RequestURI()
			} else {
				next = p.Next
			}
		case p.Continuation != "":
			next = "/api/collections/" + id + "?limit=100&continuation=" + url.QueryEscape(p.Continuation)
		default:
			next = ""
		}
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("collection %s has no models", id)
	}
	return ids, nil
}

var (
	ratingColumnRe  = regexp.MustCompile(`(?i)rating|score|capability`)
	ratingNumberRe  = regexp.MustCompile(`\d+(?:\.\d+)?`)
	rankingPrefixRe = regexp.MustCompile(`^\d+[.)]\s+`)
)

// parseRatings reads capability ratings written by a human or another
// model — a markdown table with a rating/score/capability column, or
// "name: 7.5" / "name<TAB>7.5" lines. Keys are lowercased full ids and
// basenames. Values are external estimates, never stone-llama's own.
func parseRatings(path string) (map[string]float64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	ratings := map[string]float64{}
	add := func(name string, v float64) {
		name = cleanRatingName(name)
		if name == "" || v < 0 || v > 100 {
			return
		}
		key := strings.ToLower(name)
		ratings[key] = v
		if i := strings.LastIndex(key, "/"); i >= 0 {
			ratings[key[i+1:]] = v
		}
	}
	scoreCol := -1
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "|") {
			cells := splitTableRow(line)
			if scoreCol < 0 {
				for i, c := range cells {
					if ratingColumnRe.MatchString(c) {
						scoreCol = i
						break
					}
				}
				continue
			}
			if scoreCol < len(cells) && len(cells) > 0 {
				if v, ok := ratingValue(cells[scoreCol]); ok {
					add(cells[0], v)
				}
			}
			continue
		}
		if strings.Contains(line, "|") {
			continue // a table row we can't header-map
		}
		if name, v, ok := ratingKV(line); ok {
			add(name, v)
		}
	}
	if len(ratings) == 0 {
		return nil, fmt.Errorf("no ratings found in %s (want a markdown table with a rating/score/capability column, or 'name: value' lines)", path)
	}
	return ratings, nil
}

// splitTableRow splits a markdown table row into trimmed cells.
func splitTableRow(line string) []string {
	line = strings.Trim(line, "|")
	var cells []string
	for _, c := range strings.Split(line, "|") {
		cells = append(cells, strings.TrimSpace(c))
	}
	return cells
}

// ratingValue extracts the first number out of a cell like "8.5",
// "8.5/10", "85%".
func ratingValue(cell string) (float64, bool) {
	m := ratingNumberRe.FindString(cell)
	if m == "" {
		return 0, false
	}
	v, err := strconv.ParseFloat(m, 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

// ratingKV parses "name: 7.5", "- name: 7.5" or "name<TAB>7.5".
func ratingKV(line string) (string, float64, bool) {
	line = strings.TrimPrefix(line, "- ")
	name, value, cut := strings.Cut(line, ":")
	if !cut {
		name, value, cut = strings.Cut(line, "\t")
		if !cut {
			return "", 0, false
		}
	}
	v, ok := ratingValue(value)
	if !ok || strings.TrimSpace(name) == "" {
		return "", 0, false
	}
	return name, v, true
}

// cleanRatingName strips ranking decorations: markdown links/emphasis,
// "1." numbering, trailing notes.
func cleanRatingName(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.Index(s, "]("); i > 0 {
		if j := strings.Index(s, "["); j >= 0 && j < i {
			s = s[j+1 : i]
		}
	}
	s = strings.Trim(s, "`* ")
	s = rankingPrefixRe.ReplaceAllString(s, "")
	if i := strings.Index(s, " ("); i > 0 {
		s = s[:i]
	}
	return strings.TrimSpace(s)
}
