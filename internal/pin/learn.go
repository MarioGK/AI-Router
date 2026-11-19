package pin

import (
	"context"
	"database/sql"
	"encoding/json"
	"time"

	"github.com/MarioGK/AI-Router/internal/decide"
)

// Precedent is a decision as the router learns from it: its first prompt,
// every decider's attempt, what it was based on and an operator's review.
type Precedent struct {
	ID  int64
	At  time.Time
	Key string
	// Prompt is the thread's first prompt, cut to the characters a search
	// reads.
	Prompt   string
	Attempts []decide.Attempt
	Basis    Basis
	// Review is nil until an operator judged the decision.
	Review *Review
}

// maxPrecedentChars bounds the prompt a precedent keeps: a search reads
// only its first words.
const maxPrecedentChars = 4000

// Precedents reads the decisions with the given ids in one statement,
// each by its row id.
func (s *Store) Precedents(ctx context.Context, ids []int64) ([]Precedent, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	encoded, err := json.Marshal(ids)
	if err != nil {
		return nil, err
	}
	rows, err := s.read.QueryContext(ctx, `SELECT events.id, events.at, events.key,
			substr(COALESCE(NULLIF(events.first_prompt, ''), events.prompt), 1, ?), events.attempts, events.basis,
			reviews.model, reviews.level
		FROM json_each(?) AS wanted CROSS JOIN events ON events.id = wanted.value
		LEFT JOIN reviews ON reviews.event = events.id
		WHERE events.kind = ? ORDER BY events.id`, maxPrecedentChars, string(encoded), Decision)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var list []Precedent
	for rows.Next() {
		var p Precedent
		var at int64
		var attempts, basis string
		var model, level sql.NullString
		if err := rows.Scan(&p.ID, &at, &p.Key, &p.Prompt, &attempts, &basis, &model, &level); err != nil {
			return nil, err
		}
		p.At = time.UnixMilli(at)
		if err := json.Unmarshal([]byte(attempts), &p.Attempts); err != nil {
			return nil, err
		}
		if err := json.Unmarshal([]byte(basis), &p.Basis); err != nil {
			return nil, err
		}
		if model.Valid {
			p.Review = &Review{Event: p.ID, Model: model.String, Level: level.String}
		}
		list = append(list, p)
	}
	return list, rows.Err()
}

// promptTermsMigration exposes the prompt index's vocabulary: how many
// events' prompts hold each word, as the index folds it.
const promptTermsMigration = `
CREATE VIRTUAL TABLE prompt_terms USING fts5vocab(prompt_search, row);
`

// PromptTerms counts the events whose prompts hold each word of the
// prompt index, as the index folds it, in one pass over its vocabulary,
// and the events the index holds.
func (s *Store) PromptTerms(ctx context.Context) (map[string]int, int, error) {
	var indexed int
	if err := s.read.QueryRowContext(ctx, `SELECT COUNT(*) FROM prompt_search_docsize`).Scan(&indexed); err != nil {
		return nil, 0, err
	}
	rows, err := s.read.QueryContext(ctx, `SELECT term, doc FROM prompt_terms`)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()
	counts := map[string]int{}
	for rows.Next() {
		var term string
		var documents int
		if err := rows.Scan(&term, &documents); err != nil {
			return nil, 0, err
		}
		counts[term] = documents
	}
	return counts, indexed, rows.Err()
}

// Similar is a decision whose first prompt matched a search, with its
// thread and how well it matched: the negated bm25 rank, higher for a
// closer match.
type Similar struct {
	Event int64
	Key   string
	Score float64
}

// SimilarPrompts lists up to limit decisions from the event id from on,
// other than those of the thread exclude, whose prompts match match, an
// FTS5 query the caller built from literal words, best first, after
// skipping the offset best; other events that match take no place. The
// full-text index finds and ranks the matches and their events are read
// by row id, so it never scans the log.
func (s *Store) SimilarPrompts(ctx context.Context, match string, from int64, exclude string, limit, offset int) ([]Similar, error) {
	if match == "" || limit <= 0 {
		return nil, nil
	}
	// Only eligible decisions take a place within the limit: a
	// redecision's prompt is indexed too, and so is the thread's own.
	rows, err := s.read.QueryContext(ctx, `SELECT events.id, events.key, -prompt_search.rank
		FROM prompt_search CROSS JOIN events ON events.id = prompt_search.rowid
		WHERE prompt_search MATCH ? AND prompt_search.rowid >= ? AND events.kind = ? AND events.key != ?
		ORDER BY prompt_search.rank LIMIT ? OFFSET ?`, match, from, Decision, exclude, limit, offset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var list []Similar
	for rows.Next() {
		var similar Similar
		if err := rows.Scan(&similar.Event, &similar.Key, &similar.Score); err != nil {
			return nil, err
		}
		list = append(list, similar)
	}
	return list, rows.Err()
}
