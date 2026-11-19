package pin

import (
	"context"
	"database/sql"
	"encoding/json/v2"
	"errors"
	"fmt"

	"github.com/MarioGK/AI-Router/internal/decide"
)

const routingSettingsMigration = `
CREATE TABLE routing_settings (
	id         INTEGER PRIMARY KEY CHECK (id = 1),
	settings   TEXT NOT NULL,
	updated_at INTEGER NOT NULL
);
CREATE TABLE thread_image_requirements (
	key        TEXT PRIMARY KEY,
	created_at INTEGER NOT NULL
);
`

// RoutingSettings is the operator's persistent routing policy. Models is
// the automatic model group; an empty group leaves configured routing intact.
type RoutingSettings struct {
	Mode        string             `json:"mode"`
	FixedModel  string             `json:"fixed_model"`
	Models      []string           `json:"models"`
	JevEnabled  bool               `json:"jev_enabled"`
	Preferences decide.Preferences `json:"preferences"`
}

// RoutingSettings returns the saved settings, or nil before the first save.
func (s *Store) RoutingSettings(ctx context.Context) (*RoutingSettings, error) {
	var encoded string
	if err := s.read.QueryRowContext(ctx, `SELECT settings FROM routing_settings WHERE id = 1`).Scan(&encoded); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil
		}
		return nil, err
	}
	var settings RoutingSettings
	if err := json.Unmarshal([]byte(encoded), &settings); err != nil {
		return nil, fmt.Errorf("read routing settings: %w", err)
	}
	return &settings, nil
}

// SaveRoutingSettings atomically replaces the global settings. Validation
// against configured models belongs to the resolver.
func (s *Store) SaveRoutingSettings(ctx context.Context, settings RoutingSettings) error {
	encoded, err := json.Marshal(settings)
	if err != nil {
		return err
	}
	_, err = s.write.ExecContext(ctx, `INSERT INTO routing_settings (id, settings, updated_at) VALUES (1, ?, ?)
		ON CONFLICT (id) DO UPDATE SET settings = excluded.settings, updated_at = excluded.updated_at`,
		string(encoded), s.now().UnixMilli())
	return err
}

// ImagesRequired combines fixed-mode observations with the automatic pin's
// durable capability flag, without changing its model or reasoning level.
func (s *Store) ImagesRequired(ctx context.Context, key string) (bool, error) {
	var required bool
	err := s.write.QueryRowContext(ctx, `SELECT
		EXISTS (SELECT 1 FROM thread_image_requirements WHERE key = ?1) OR
		EXISTS (SELECT 1 FROM pins WHERE key = ?1 AND upgraded = 1)`, key).Scan(&required)
	return required, err
}

// RequireImages remembers a session's images even when fixed mode has never
// created an automatic pin. Requirements, like pins, are permanent.
func (s *Store) RequireImages(ctx context.Context, key string) error {
	if key == "" {
		return fmt.Errorf("image requirement needs a thread key")
	}
	_, err := s.write.ExecContext(ctx, `INSERT OR IGNORE INTO thread_image_requirements (key, created_at) VALUES (?, ?)`,
		key, s.now().UnixMilli())
	return err
}
