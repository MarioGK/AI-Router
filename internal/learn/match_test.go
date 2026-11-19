package learn

import (
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/MarioGK/AI-Router/internal/pin"
)

func TestMatchQuotesEveryWordAndNoSyntax(t *testing.T) {
	cases := []struct {
		prompt, want string
	}{
		{`Fix the "flaky" race in pin/store.go`, `("fix" OR "flaky" OR "race" OR "pin" OR "store")`},
		// FTS operators and syntax are words or nothing.
		{`NEAR(auth token) OR refresh AND NOT expiry* ^start col:value {a b}`,
			`("near" OR "auth" OR "token" OR "refresh" OR "expiry" OR "start" OR "col" OR "value")`},
		{`'); DROP TABLE events; --`, `("drop" OR "table" OR "events")`},
		// Accents and scripts other than Latin stay words; bare numbers,
		// short words and stopwords go, and repeats count once.
		{`Corrigir a função de autenticação funcao 2026 404 ab`, `("corrigir" OR "funcao" OR "autenticacao")`},
		{`ログイン画面 を 修正 して`, `("ログイン画面")`},
		{`Ünïcödé naïve café café`, `("unicode" OR "naive" OR "cafe")`},
		{`the and of 42 ?!`, ``},
		{``, ``},
	}
	for _, c := range cases {
		if got := Match(c.prompt, nil, minPostings); got != c.want {
			t.Errorf("Match(%q) = %s, want %s", c.prompt, got, c.want)
		}
	}
	long := strings.Repeat("word ", 3) + strings.Repeat("x", maxTermRunes+1)
	if got := Match(long, nil, minPostings); got != `("word")` {
		t.Errorf("a blob is a word: %s", got)
	}
	var many []string
	for i := range 100 {
		many = append(many, "term"+string(rune('a'+i%26))+string(rune('a'+i/26)))
	}
	if got := Match(strings.Join(many, " "), nil, minPostings); strings.Count(got, `"`) != 2*maxTerms {
		t.Errorf("Match kept %d terms, want %d", strings.Count(got, `"`)/2, maxTerms)
	}
}

func TestMatchAsksForTheRarestWords(t *testing.T) {
	counted := map[string]int{"fix": 250, "flaky": 3, "race": 40, "store": 50}
	// The rarest known first; fix would pass the postings bound, and the
	// index holds no scheduler for it to match.
	if got := Match("Fix the flaky race in the store and the scheduler", counted, minPostings); got !=
		`("flaky" OR "race" OR "store")` {
		t.Errorf("Match = %s", got)
	}
	// A prompt of common words still asks for its rarest few.
	common := map[string]int{"fix": 500, "retry": 300, "flaky": 200, "test": 900}
	if got := Match("fix the flaky retry test", common, minPostings); got != `("flaky" OR "retry" OR "fix")` {
		t.Errorf("a prompt of common words searches for %s", got)
	}
	if Postings(10_000) != 500 || Postings(10) != minPostings {
		t.Errorf("postings budgets = %d, %d", Postings(10_000), Postings(10))
	}
	// Novel identifiers cannot crowd out a word the history knows.
	var novel []string
	for i := range maxTerms {
		novel = append(novel, "novelident"+string(rune('a'+i)))
	}
	if got := Match(strings.Join(novel, " ")+" deadlock", map[string]int{"deadlock": 10}, minPostings); got != `("deadlock")` {
		t.Errorf("Match = %s, want deadlock alone", got)
	}
	// Nor turn off the fallback to the rarest common words.
	if got := Match("flaky retry novelword", map[string]int{"flaky": 200, "retry": 300}, minPostings); got !=
		`("flaky" OR "retry")` {
		t.Errorf("Match = %s, want the common words", got)
	}
	// Accented words are folded as the index folds them.
	if got := fold("Função"); got != "funcao" {
		t.Errorf("fold = %s", got)
	}
}

func TestACommonWordLateInPromptsIsCountedAsCommon(t *testing.T) {
	store, err := pin.Open(filepath.Join(t.TempDir(), "router.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	filler := strings.Repeat("lorem ipsum dolor amet consectetur ", 1000)
	for i := range minPostings + 1 {
		key := fmt.Sprintf("t%d", i)
		if _, err := store.Create(ctx, pin.Pin{Key: key, Model: "glm-5.3"}, pin.Event{Key: key, Kind: pin.Decision,
			Model: "glm-5.3", FirstPrompt: filler + " Função"}); err != nil {
			t.Fatal(err)
		}
	}
	counts, indexed, err := store.PromptTerms(ctx)
	if err != nil || counts["funcao"] != minPostings+1 || indexed != minPostings+1 {
		t.Fatalf("counts = %v, %v", counts["funcao"], err)
	}
	// Counted as common, the word is searched only as the fallback.
	if got := Match("função flaky", counts, minPostings); got != `("funcao")` {
		t.Fatalf("Match = %s, want the common word as the fallback", got)
	}
	if got := Match("função flaky ipsum", counts, minPostings); got != `("funcao" OR "ipsum")` {
		t.Fatalf("Match = %s, want the rarest common words", got)
	}
}

func TestWordsAreFoldedAsTheIndexFoldsThem(t *testing.T) {
	store, err := pin.Open(filepath.Join(t.TempDir(), "router.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	prompts := []string{
		"Sửa lỗi đăng nhập trong ứng dụng",
		"Corrigir a função de autenticação",
		"Ǖber ḗmulator ộcean naïve café café xáyz",
		"Straße ølstue łódka άλφα ログイン画面 ガラス",
	}
	for i, prompt := range prompts {
		key := fmt.Sprintf("p%d", i)
		if _, err := store.Create(ctx, pin.Pin{Key: key, Model: "glm-5.3"}, pin.Event{Key: key, Kind: pin.Decision,
			Model: "glm-5.3", FirstPrompt: prompt}); err != nil {
			t.Fatal(err)
		}
	}
	vocabulary, _, err := store.PromptTerms(ctx)
	if err != nil {
		t.Fatal(err)
	}
	for _, prompt := range prompts {
		for _, word := range words(prompt) {
			if vocabulary[word] == 0 {
				t.Errorf("%q folds to %q, which the index does not hold (it holds %v)", prompt, word, vocabulary)
			}
		}
	}
	for _, want := range []string{"sua", "loi", "đang", "nhap", "trong", "ung", "dung", "funcao", "autenticacao",
		"uber", "emulator", "ocean", "naive", "cafe", "xayz"} {
		if !slices.Contains(words(strings.Join(prompts, " ")), want) && !stopwords[want] && len(want) >= minTermRunes {
			t.Errorf("no word folds to %q", want)
		}
	}
}

func TestEveryMatchRunsOnThePromptIndex(t *testing.T) {
	store, err := pin.Open(filepath.Join(t.TempDir(), "router.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	if _, err := store.Create(ctx, pin.Pin{Key: "a", Model: "glm-5.3"}, pin.Event{Key: "a", Kind: pin.Decision,
		Model: "glm-5.3", FirstPrompt: "Fix the flaky race in the pin store"}); err != nil {
		t.Fatal(err)
	}
	for _, prompt := range []string{`"quoted" OR NOT NEAR(x y) AND col:val*`, `ünïcödé`, `fix race`, `") OR rowid:1 --`} {
		match := Match(prompt, nil, minPostings)
		if match == "" {
			continue
		}
		if _, err := store.SimilarPrompts(ctx, match, 0, "", 10, 0); err != nil {
			t.Errorf("search %q: %v", prompt, err)
		}
	}
	found, err := store.SimilarPrompts(ctx, Match("why is the race in store flaky", nil, minPostings), 0, "", 10, 0)
	if err != nil || len(found) != 1 || found[0].Key != "a" || found[0].Score <= 0 {
		t.Fatalf("found = %+v, %v", found, err)
	}
	// Redecisions whose prompts match better take no place within the
	// limit.
	for i := range 5 {
		key := fmt.Sprintf("redecided-%d", i)
		if _, err := store.Create(ctx, pin.Pin{Key: key, Model: "glm-5.3"}, pin.Event{Key: key, Kind: pin.Redecision,
			Model: "glm-5.3", FirstPrompt: "flaky race flaky race store"}); err != nil {
			t.Fatal(err)
		}
	}
	found, err = store.SimilarPrompts(ctx, Match("flaky race store", nil, minPostings), 0, "", 2, 0)
	if err != nil || len(found) != 1 || found[0].Key != "a" {
		t.Fatalf("found among redecisions = %+v, %v", found, err)
	}
}
