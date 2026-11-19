package pin

import (
	"database/sql"
	"fmt"
	"path/filepath"
	"slices"
	"testing"

	"github.com/MarioGK/AI-Router/internal/usage"
)

func TestMeteringRepairCorrectsOnlyThePrefixRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "router.db")
	db, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatal(err)
	}
	last := slices.Index(migrations, meteringRepairMigration)
	for i, migration := range migrations[:last] {
		if _, err := db.Exec(migration); err != nil {
			t.Fatalf("migration %d: %v", i+1, err)
		}
	}
	// Around 01:00 UTC on 2026-09-29 and on 2026-09-30.
	const before, after = 1790640000000, 1790730000000
	rows := []struct {
		id                                   int
		at                                   int64
		model                                string
		status                               int
		input, output, cacheRead, cacheWrite int
	}{
		{1, before, "gpt-6.1-sol", 200, 100355, 500, 108032, 0},    // the live case: an estimate a little below its cache reads
		{11, before, "gpt-6.1-sol", 200, 4999, 300, 10000, 0},      // fresh input under half of what was cached: left alone
		{8, before, "gpt-6.1-sol", 200, 1000, 300, 1000, 0},        // the reviewer's case: 1000 fresh beside 1000 cached
		{9, before, "gpt-6.1-sol", 200, 20000, 300, 9999, 0},       // too little cached to tell
		{10, before, "gpt-6.1-sol", 200, 6000, 300, 10000, 2000},   // exactly at both bounds
		{2, before, "gpt-6.1-sol", 200, 120000, 300, 90000, 2000},  // the estimate above them: some fresh input remains
		{3, before, "claude-opus-5-5", 200, 90000, 300, 100000, 0}, // Claude streams were always metered right
		{4, before, "gpt-6.1-sol", 200, 200, 300, 90000, 0},        // fresh input only
		{5, before, "gpt-6.1-sol", 200, 50000, 300, 0, 0},          // nothing cached: the estimate was the input
		{6, before, "gpt-6.1-sol", 429, 100355, 0, 108032, 0},      // failed
		{7, after, "gpt-6.1-sol", 200, 100355, 500, 108032, 0},     // logged by the fixed build
	}
	for _, r := range rows {
		if _, err := db.Exec(`INSERT INTO requests (id, at, key, kind, requested, model, reason, status, failover,
			latency_ms, error, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, cost_usd)
			VALUES (?, ?, 'k', 'messages', 'auto', ?, 'pinned', ?, 0, 1, '', ?, ?, ?, ?, 9.5)`,
			r.id, r.at, r.model, r.status, r.input, r.output, r.cacheRead, r.cacheWrite); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := db.Exec(fmt.Sprintf(`PRAGMA user_version = %d`, last)); err != nil {
		t.Fatal(err)
	}
	db.Close()

	store := open(t, path)
	// Opening the store only adds the columns; the repair is opt-in.
	var flagged int
	if err := store.write.QueryRow(`SELECT count(*) FROM requests WHERE repaired = 1`).Scan(&flagged); err != nil || flagged != 0 {
		t.Fatalf("the migration repaired %d rows, %v; want none", flagged, err)
	}
	if n, err := store.RepairMetering(t.Context(), false); err != nil || n != 3 {
		t.Fatalf("RepairMetering(dry run) = %d, %v, want 3", n, err)
	}
	if n, err := store.RepairMetering(t.Context(), true); err != nil || n != 3 {
		t.Fatalf("RepairMetering(apply) = %d, %v, want 3", n, err)
	}
	if n, err := store.RepairMetering(t.Context(), true); err != nil || n != 0 {
		t.Fatalf("a second RepairMetering = %d, %v, want 0", n, err)
	}
	type result struct {
		input    int
		repaired bool
		cost     sql.NullFloat64
		original sql.NullInt64
		before   sql.NullFloat64
	}
	read := func() map[int]result {
		out := map[int]result{}
		rs, err := store.write.Query(`SELECT id, input_tokens, repaired, cost_usd, original_input_tokens, original_cost_usd FROM requests`)
		if err != nil {
			t.Fatal(err)
		}
		defer rs.Close()
		for rs.Next() {
			var id int
			var r result
			if err := rs.Scan(&id, &r.input, &r.repaired, &r.cost, &r.original, &r.before); err != nil {
				t.Fatal(err)
			}
			out[id] = r
		}
		return out
	}
	kept := sql.NullFloat64{Float64: 9.5, Valid: true}
	got := read()
	// A repaired row keeps what was logged; every other row has no original.
	repaired := func(input, original int) result {
		return result{input, true, sql.NullFloat64{}, sql.NullInt64{Int64: int64(original), Valid: true}, kept}
	}
	untouched := func(input int) result { return result{input: input, cost: kept} }
	for id, want := range map[int]result{
		1: repaired(0, 100355), 2: repaired(28000, 120000), 3: untouched(90000), 4: untouched(200),
		5: untouched(50000), 6: untouched(100355), 7: untouched(100355), 8: untouched(1000),
		9: untouched(20000), 10: repaired(0, 6000), 11: untouched(4999),
	} {
		if got[id] != want {
			t.Errorf("request %d = %+v, want %+v", id, got[id], want)
		}
	}

	price := func(model string) (usage.Price, bool) {
		return usage.Price{Input: 2, Output: 10, CacheRead: 0.2}, model == "gpt-6.1-sol"
	}
	if n, err := store.RepriceRepaired(t.Context(), price); err != nil || n != 3 {
		t.Fatalf("RepriceRepaired = %d, %v, want 3", n, err)
	}
	got = read()
	p, _ := price("gpt-6.1-sol")
	if want := p.Cost(usage.Tokens{Input: 28000, Output: 300, CacheRead: 90000, CacheWrite: 2000}); !got[2].cost.Valid || got[2].cost.Float64 != want {
		t.Errorf("request 2 cost = %+v, want %v", got[2].cost, want)
	}
	if got[2].before != kept {
		t.Errorf("request 2 lost its original cost: %+v", got[2].before)
	}
	if got[3].cost != kept {
		t.Errorf("an untouched request was repriced: %+v", got[3].cost)
	}
	if n, err := store.RepriceRepaired(t.Context(), price); err != nil || n != 0 {
		t.Fatalf("a second RepriceRepaired = %d, %v, want 0", n, err)
	}
}
