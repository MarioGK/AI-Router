package pin

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"time"
)

// Review is an operator's judgement of a decision: the model and reasoning
// level its task should have had, and why.
type Review struct {
	Event int64
	At    time.Time
	Model string
	// Level is empty when the task should have run at the model's fixed
	// effort.
	Level string
	Note  string
}

// Review filters of a DecisionFilter.
const (
	Unreviewed = "unreviewed"
	Right      = "right"
	Wrong      = "wrong"
)

// ErrNoDecision is a review of an event that is not a decision.
var ErrNoDecision = errors.New("no such decision")

const (
	// decided is the model a decision chose, before a busy model spilled
	// the thread elsewhere.
	decided = `COALESCE(NULLIF(events.basis->>'spilled', ''), events.model)`
	// unreviewed matches the decisions without a review.
	unreviewed = `NOT EXISTS (SELECT 1 FROM reviews WHERE reviews.event = events.id)`
	// maxReviewed bounds the reviewed decisions read at once.
	maxReviewed = 10_000
)

// reviewed matches the decisions of each review filter.
var reviewed = map[string]string{
	Unreviewed: unreviewed,
	Right: `EXISTS (SELECT 1 FROM reviews WHERE reviews.event = events.id
		AND reviews.model = ` + decided + ` AND reviews.level = events.level)`,
	Wrong: `EXISTS (SELECT 1 FROM reviews WHERE reviews.event = events.id
		AND (reviews.model != ` + decided + ` OR reviews.level != events.level))`,
}

// Decided is the model a decision chose, before a busy model spilled the
// thread elsewhere.
func (e Event) Decided() string {
	return cmp.Or(e.Basis.Spilled, e.Model)
}

// Right reports whether a review agrees with the decision it judges: on
// the model, and on the reasoning level.
func (r Review) Right(decision Event) (model, level bool) {
	return r.Model == decision.Decided(), r.Level == decision.Level
}

// SaveReview records an operator's review of a decision, replacing any
// earlier one; ErrNoDecision when the event is not a decision.
func (s *Store) SaveReview(ctx context.Context, review Review) error {
	result, err := s.write.ExecContext(ctx, `INSERT INTO reviews (event, at, model, level, note)
		SELECT id, ?, ?, ?, ? FROM events WHERE id = ? AND kind = ?
		ON CONFLICT (event) DO UPDATE SET at = excluded.at, model = excluded.model, level = excluded.level,
			note = excluded.note`,
		s.now().UnixMilli(), review.Model, review.Level, review.Note, review.Event, Decision)
	if err != nil {
		return err
	}
	if saved, err := result.RowsAffected(); err != nil {
		return err
	} else if saved == 0 {
		return ErrNoDecision
	}
	return nil
}

// DeleteReview forgets the review of a decision.
func (s *Store) DeleteReview(ctx context.Context, event int64) error {
	_, err := s.write.ExecContext(ctx, `DELETE FROM reviews WHERE event = ?`, event)
	return err
}

// Reviewed lists the reviewed decisions, newest first, at most ten
// thousand.
func (s *Store) Reviewed(ctx context.Context) ([]Event, error) {
	return events(ctx, s.read, ` WHERE kind = ? AND NOT `+unreviewed+` ORDER BY id DESC LIMIT ?`, Decision, maxReviewed)
}

// ReviewSummary is how the reviewed decisions fared: how many chose the
// right model and the right reasoning level, and the misses by the model
// decided and the model it should have been ("glm-5.3->gpt-6.1-sol").
type ReviewSummary struct {
	Reviewed   int            `json:"reviewed"`
	RightModel int            `json:"right_model"`
	RightLevel int            `json:"right_level"`
	Misses     map[string]int `json:"misses"`
	// Unreviewed is how many decisions with a recorded prompt wait for a
	// review.
	Unreviewed int `json:"unreviewed"`
}

// Reviews summarizes the reviews of the decisions made since the given
// time.
func (s *Store) Reviews(ctx context.Context, since time.Time) (ReviewSummary, error) {
	summary := ReviewSummary{Misses: map[string]int{}}
	if err := s.read.QueryRowContext(ctx, `SELECT COUNT(*) FROM events WHERE kind = ? AND at >= ? AND `+
		recorded+` AND `+unreviewed, Decision, since.UnixMilli()).Scan(&summary.Unreviewed); err != nil {
		return summary, err
	}
	rows, err := s.read.QueryContext(ctx, `SELECT `+decided+`, events.level, reviews.model, reviews.level
		FROM events JOIN reviews ON reviews.event = events.id WHERE events.kind = ? AND events.at >= ?`,
		Decision, since.UnixMilli())
	if err != nil {
		return summary, err
	}
	defer rows.Close()
	for rows.Next() {
		var model, level, rightModel, rightLevel string
		if err := rows.Scan(&model, &level, &rightModel, &rightLevel); err != nil {
			return summary, err
		}
		summary.Reviewed++
		if model == rightModel {
			summary.RightModel++
		} else {
			summary.Misses[model+"->"+rightModel]++
		}
		if level == rightLevel {
			summary.RightLevel++
		}
	}
	return summary, rows.Err()
}

// readReview decodes the review a decision is read with.
func readReview(event int64, encoded string) (*Review, error) {
	var fields struct {
		At    int64  `json:"at"`
		Model string `json:"model"`
		Level string `json:"level"`
		Note  string `json:"note"`
	}
	if err := json.Unmarshal([]byte(encoded), &fields); err != nil {
		return nil, err
	}
	return &Review{Event: event, At: time.UnixMilli(fields.At), Model: fields.Model, Level: fields.Level,
		Note: fields.Note}, nil
}
