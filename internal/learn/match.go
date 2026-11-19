package learn

import (
	"cmp"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// A similarity search asks for at most maxTerms words of a prompt, the
// rarest first, of the first maxWords distinct informative ones; it skips a
// word once the indexed prompts holding the words asked for would pass the
// postings budget, the larger of minPostings and a twentieth of the
// indexed prompts. bm25 ranks a document by its rare words, and every
// document a common word matches has to be ranked, so dropping the common
// words costs little precision and bounds the work of a search. When every
// known word is too common, as for a task that recurs often, the search
// still asks for the fallbackTerms rarest of them, and its deadline bounds
// it.
const (
	maxTerms      = 16
	maxWords      = 64
	minPostings   = 100
	fallbackTerms = 3
)

// Postings is the postings budget of an index holding indexed prompts.
func Postings(indexed int) int {
	return max(minPostings, indexed/20)
}

// minTermRunes is the length below which a word says too little to search
// for; maxTermRunes the length past which it is a hash or an encoded blob.
const (
	minTermRunes = 3
	maxTermRunes = 40
)

// stopwords are common English words that say nothing about a task.
var stopwords = map[string]bool{}

func init() {
	for _, word := range strings.Fields(`the and for are but not you all any can had has her was one our out
		his how its may new now see two who did get him let put say she too use with that this from have they
		will would there their what when which while into than then them these those been were your about
		could should just also some such only other more most very here where after before please make sure
		want need like each does doing done being over under again`) {
		stopwords[word] = true
	}
}

// diacritic reports whether a rune is a combining diacritical mark, which
// the prompt index's unicode61 tokenizer removes from within a word; any
// other mark separates words there, as punctuation does.
func diacritic(r rune) bool {
	return r >= 0x300 && r <= 0x36f
}

// fold lowercases a word and removes its diacritics, as the prompt
// index's unicode61 tokenizer with remove_diacritics 2 does: a Latin
// letter becomes the base letter of its canonical decomposition (ộ, ǖ and
// ḗ lose every accent), combining diacritical marks go, and letters of
// other scripts and those without a decomposition (ά, ø, ł, đ, ß) stay as
// they are.
func fold(word string) string {
	var folded strings.Builder
	for _, r := range strings.ToLower(word) {
		switch {
		case diacritic(r):
		case r >= utf8.RuneSelf && unicode.Is(unicode.Latin, r):
			for _, part := range norm.NFD.String(string(r)) {
				if !unicode.Is(unicode.Mn, part) {
					folded.WriteRune(part)
				}
			}
		default:
			folded.WriteRune(r)
		}
	}
	return folded.String()
}

// words are a prompt's distinct informative words in the order they first
// appear, at most maxWords: runs of letters, numbers and marks, as SQLite's
// unicode61 tokenizer splits text, lowercased and without diacritics,
// without stopwords, bare numbers and words too short or too long to name
// anything.
func words(prompt string) []string {
	seen := map[string]bool{}
	var list []string
	for _, word := range strings.FieldsFunc(prompt, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r) && !unicode.Is(unicode.Co, r) && !diacritic(r)
	}) {
		word = fold(word)
		if n := utf8.RuneCountInString(word); n < minTermRunes || n > maxTermRunes || stopwords[word] || seen[word] ||
			strings.IndexFunc(word, unicode.IsLetter) < 0 {
			continue
		}
		seen[word] = true
		list = append(list, word)
		if len(list) == maxWords {
			break
		}
	}
	return list
}

// Match is an FTS5 query for the decisions whose prompts share any of the
// prompt's rarest informative words, given how many indexed prompts hold
// each word (documents); it matches the first prompt and the prompt as
// judged both, so a decision from before first prompts were kept matches
// too: each word quoted as a literal string, never as FTS syntax, any of
// them matching, rarest first. documents counts the very index searched,
// so a word it does not hold can find no neighbour and is left out; only
// without any counts are the words searched in the order they appear.
// Empty when the prompt has no word to search for.
func Match(prompt string, documents map[string]int, budget int) string {
	list := words(prompt)
	if len(documents) > 0 {
		list = slices.DeleteFunc(list, func(word string) bool { return documents[word] == 0 })
	}
	order := make(map[string]int, len(list))
	for i, word := range list {
		order[word] = i
	}
	slices.SortStableFunc(list, func(a, b string) int {
		return cmp.Or(cmp.Compare(documents[a], documents[b]), cmp.Compare(order[a], order[b]))
	})
	var terms []string
	postings := 0
	for _, word := range list {
		if len(terms) == maxTerms {
			break
		}
		if postings+documents[word] > budget {
			// Too common; a rarer word after it may still fit.
			continue
		}
		postings += documents[word]
		// The word holds no quote, so quoting makes it a literal string.
		terms = append(terms, `"`+word+`"`)
	}
	if len(terms) == 0 {
		// Every word is common: the rarest few still find the task's
		// earlier runs.
		for _, word := range list[:min(fallbackTerms, len(list))] {
			terms = append(terms, `"`+word+`"`)
		}
	}
	if len(terms) == 0 {
		return ""
	}
	return "(" + strings.Join(terms, " OR ") + ")"
}
