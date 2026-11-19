package pin

import (
	"database/sql"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestPromptSearchMigrationAndUpdates(t *testing.T) {
	path := filepath.Join(t.TempDir(), "router.db")
	// A released database already contains prompts before the index exists.
	db, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatal(err)
	}
	for i, migration := range migrations[:11] {
		if _, err := db.Exec(migration); err != nil {
			t.Fatalf("migration %d: %v", i+1, err)
		}
	}
	if _, err := db.Exec(`INSERT INTO events (at, key, kind, model, previous, decider, choice,
		probabilities, escalated, latency_ms, detail, prompt, first_prompt)
		VALUES (1000, 'existing', 'decision', 'glm-5.3', '', 'jev', 'glm-5.3', '{}', 0, 0, '',
		'images=false', 'Corrigir navegação CAFÉ 中文搜索'); PRAGMA user_version = 11;`); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	store := open(t, path)
	for _, search := range []string{"NAVegacao cafe", `"café"`, "中文搜索", "corrigir, café!", "cafe OR navegacao", "!!!", strings.Repeat("a", MaxSearchChars+1)} {
		page, err := store.Decisions(t.Context(), DecisionFilter{Search: search})
		if len(search) > MaxSearchChars {
			if err == nil {
				t.Fatal("oversized search was accepted")
			}
			continue
		}
		want := 1
		if search == "!!!" || search == "cafe OR navegacao" {
			want = 0 // OR is a literal required word, never query syntax.
		}
		if err != nil || page.Total != want {
			t.Fatalf("search %q = %+v, %v, want %d", search, page, err, want)
		}
	}
	if _, err := store.write.Exec(`UPDATE events SET prompt = 'Fix parser escapes', first_prompt = '' WHERE id = 1`); err != nil {
		t.Fatal(err)
	}
	assertSearchCount(t, store, "cafe", 0)
	assertSearchCount(t, store, "parser", 1)
	if n, err := store.PrunePrompts(t.Context(), time.Now()); err != nil || n != 1 {
		t.Fatalf("prune = %d, %v", n, err)
	}
	assertSearchCount(t, store, "parser", 0)
	if _, err := store.write.Exec(`UPDATE events SET first_prompt = 'delete trigger' WHERE id = 1`); err != nil {
		t.Fatal(err)
	}
	assertSearchCount(t, store, "trigger", 1)
	if _, err := store.write.Exec(`DELETE FROM events WHERE id = 1`); err != nil {
		t.Fatal(err)
	}
	assertSearchCount(t, store, "trigger", 0)
}

func assertSearchCount(t *testing.T, store *Store, search string, want int) {
	t.Helper()
	page, err := store.Decisions(t.Context(), DecisionFilter{Search: search})
	if err != nil || page.Total != want {
		t.Fatalf("search %q = %+v, %v, want %d", search, page, err, want)
	}
}

func TestPromptSearchCombinesFiltersAndLogs(t *testing.T) {
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	for _, e := range []Event{
		{Key: "a", Kind: Decision, Model: "glm-5.3", Decider: "jev", Prompt: "fix parser", FirstPrompt: "Make the invoice parser work"},
		{Key: "b", Kind: Decision, Model: "gpt-6.1-sol", Decider: "judge", Prompt: "fix parser"},
		{Key: "c", Kind: Decision, Model: "glm-5.3", Decider: "jev", Prompt: "fix menu"},
		{Key: "d", Kind: Override, Model: "glm-5.3", Prompt: "fix parser"},
	} {
		if _, err := store.Create(t.Context(), Pin{Key: e.Key, Model: e.Model}, e); err != nil {
			t.Fatal(err)
		}
	}
	if err := store.SaveReview(t.Context(), Review{Event: 1, Model: "glm-5.3"}); err != nil {
		t.Fatal(err)
	}
	page, err := store.Decisions(t.Context(), DecisionFilter{Search: "fix parser", Limit: 1, Offset: 1})
	if err != nil || page.Total != 2 || len(page.Decisions) != 1 || page.Decisions[0].Key != "a" {
		t.Fatalf("search pagination = %+v, %v", page, err)
	}
	page, err = store.Decisions(t.Context(), DecisionFilter{Search: "parser", Model: "glm-5.3", Decider: "jev", Review: "right"})
	if err != nil || page.Total != 1 || page.Decisions[0].Key != "a" {
		t.Fatalf("combined search = %+v, %v", page, err)
	}
	logRequests(t, store, RequestLog{Key: "a", Model: "glm-5.3"}, RequestLog{Key: "b", Model: "gpt-6.1-sol"},
		RequestLog{Key: "a", Model: "gpt-6.1-sol"}, RequestLog{Key: "c", Model: "glm-5.3"}, RequestLog{Key: "d", Model: "glm-5.3"})
	for _, c := range []struct {
		filter RequestFilter
		want   []int64
	}{
		{RequestFilter{Search: "parser"}, []int64{3, 2, 1}},
		{RequestFilter{Search: "parser", Model: "gpt-6.1-sol"}, []int64{3, 2}},
		{RequestFilter{Search: "invoice", Key: "a", AfterID: 1}, []int64{3}},
		{RequestFilter{Search: "fix parser", Limit: 1}, []int64{3}},
		{RequestFilter{Search: "!!!"}, []int64{}},
	} {
		got, err := store.Requests(t.Context(), c.filter)
		if err != nil || !slices.Equal(ids(got), c.want) {
			t.Fatalf("Requests(%+v) = %v, %v, want %v", c.filter, ids(got), err, c.want)
		}
	}
}

func BenchmarkPromptSearch(b *testing.B) {
	store, err := Open(filepath.Join(b.TempDir(), "router.db"))
	if err != nil {
		b.Fatal(err)
	}
	defer store.Close()
	if _, err := store.write.Exec(`WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 10000)
		INSERT INTO events (at, key, kind, model, previous, decider, choice, probabilities, escalated, latency_ms, detail, first_prompt)
		SELECT i, 'thread-' || i, 'decision', 'glm-5.3', '', 'jev', '', '{}', 0, 0, '',
		'Routine prompt for a project with ' || CASE WHEN i % 100 = 0 THEN 'parser' ELSE 'menu' END || ' changes' FROM n`); err != nil {
		b.Fatal(err)
	}
	for b.Loop() {
		page, err := store.Decisions(b.Context(), DecisionFilter{Search: "parser project"})
		if err != nil || page.Total != 100 {
			b.Fatalf("search = %+v, %v", page, err)
		}
	}
}
