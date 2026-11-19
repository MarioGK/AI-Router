package pin

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"path/filepath"
	"slices"
)

// ExportRecord is one legacy row prepared for a TerraBase document. Key is
// its natural identity; the SDK importer will derive the document UUID from
// Collection and Key. Numeric IDs remain in Document for reviews and tails.
type ExportRecord struct {
	Collection string          `json:"collection"`
	Key        string          `json:"key"`
	Document   json.RawMessage `json:"document"`
}

// ExportManifest ends a completed export. Importers must require this final
// line and verify every collection's count before accepting the artifact.
type ExportManifest struct {
	Format        int            `json:"format"`
	SchemaVersion int            `json:"schema_version"`
	Counts        map[string]int `json:"counts"`
}

var exportCollections = []struct {
	name, key string
	json      []string
	booleans  []string
}{
	{"pins", "key", nil, []string{"failed_over", "upgraded", "compacted"}},
	{"events", "id", []string{"probabilities", "levels", "attempts", "basis"}, []string{"escalated"}},
	{"requests", "id", nil, []string{"failover", "repaired"}},
	{"policies", "fingerprint", []string{"policy"}, nil},
	{"reviews", "event", nil, nil},
	{"routing_settings", "id", []string{"settings"}, nil},
	{"thread_image_requirements", "key", nil, nil},
}

// Export streams all application rows as JSON lines from one read-only
// snapshot. It never creates or migrates the source. Search tables are
// derived indexes, rebuilt from the exported events after cutover.
func Export(ctx context.Context, database string, output io.Writer) error {
	path, err := filepath.Abs(database)
	if err != nil {
		return err
	}
	source := url.URL{Scheme: "file", Path: path}
	query := url.Values{"mode": {"ro"}, "_pragma": {"query_only(1)", "busy_timeout(5000)"}}
	source.RawQuery = query.Encode()
	db, err := sql.Open("sqlite", source.String())
	if err != nil {
		return err
	}
	defer db.Close()
	tx, err := db.BeginTx(ctx, &sql.TxOptions{ReadOnly: true})
	if err != nil {
		return err
	}
	defer tx.Rollback()
	var version int
	if err := tx.QueryRowContext(ctx, "PRAGMA user_version").Scan(&version); err != nil {
		return err
	}
	if version != len(migrations) {
		return fmt.Errorf("export needs schema version %d; source is version %d (source left unchanged)", len(migrations), version)
	}
	encode := json.NewEncoder(output)
	manifest := ExportManifest{Format: 1, SchemaVersion: version, Counts: map[string]int{}}
	for _, collection := range exportCollections {
		manifest.Counts[collection.name] = 0
		// Both identifiers come from the fixed table above, never input.
		rows, err := tx.QueryContext(ctx, "SELECT * FROM "+collection.name+" ORDER BY "+collection.key)
		if err != nil {
			return err
		}
		columns, err := rows.Columns()
		if err != nil {
			rows.Close()
			return err
		}
		for rows.Next() {
			values := make([]any, len(columns))
			targets := make([]any, len(columns))
			for i := range values {
				targets[i] = &values[i]
			}
			if err := rows.Scan(targets...); err != nil {
				rows.Close()
				return err
			}
			document := make(map[string]any, len(columns))
			var key string
			for i, name := range columns {
				value := values[i]
				if name == collection.key {
					key = fmt.Sprint(value)
				}
				if slices.Contains(collection.json, name) {
					text, ok := value.(string)
					if !ok || !json.Valid([]byte(text)) {
						rows.Close()
						return fmt.Errorf("invalid JSON in %s.%s", collection.name, name)
					}
					value = json.RawMessage(text)
				}
				if slices.Contains(collection.booleans, name) {
					number, ok := value.(int64)
					if !ok || number < 0 || number > 1 {
						rows.Close()
						return fmt.Errorf("invalid boolean in %s.%s", collection.name, name)
					}
					value = number == 1
				}
				document[name] = value
			}
			body, err := json.Marshal(document)
			if err == nil {
				err = encode.Encode(ExportRecord{Collection: collection.name, Key: key, Document: body})
			}
			if err != nil {
				rows.Close()
				return err
			}
			manifest.Counts[collection.name]++
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return err
		}
		if err := rows.Close(); err != nil {
			return err
		}
	}
	if err := tx.Commit(); err != nil {
		return err
	}
	return encode.Encode(struct {
		Manifest ExportManifest `json:"manifest"`
	}{manifest})
}
