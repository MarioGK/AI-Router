package pin

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/usage"
)

func TestExportPreservesAllEntitiesAndMetering(t *testing.T) {
	path := filepath.Join(t.TempDir(), "router.db")
	store := open(t, path)
	ctx := t.Context()
	key := "thread:session/agent"
	if _, err := store.Create(ctx, Pin{Key: key, Model: "sol", Level: "high", Parent: "parent"},
		Event{Key: key, Kind: Decision, Model: "sol", Level: "high", Prompt: "quoted \"text\"\n中文 café",
			Probabilities: map[string]float64{"sol": 0.9}, Levels: map[string]float64{"high": 1},
			Attempts: []decide.Attempt{{Decider: "judge", Outcome: decide.Decided, Response: "raw reply"}}}); err != nil {
		t.Fatal(err)
	}
	if err := store.Seen(ctx, key); err != nil {
		t.Fatal(err)
	}
	if err := store.Compacted(ctx, key); err != nil {
		t.Fatal(err)
	}
	if err := store.RequireImages(ctx, "fixed-only"); err != nil {
		t.Fatal(err)
	}
	if err := store.SavePolicy(ctx, decide.Policy{}); err != nil {
		t.Fatal(err)
	}
	if err := store.SaveReview(ctx, Review{Event: 1, Model: "sol", Level: "high", Note: "keep this"}); err != nil {
		t.Fatal(err)
	}
	if err := store.SaveRoutingSettings(ctx, RoutingSettings{Mode: "fixed", FixedModel: "sol", JevEnabled: true,
		Preferences: decide.Preferences{Priority: decide.PriorityCost, Ranking: []string{"sol"}}}); err != nil {
		t.Fatal(err)
	}
	logRequests(t, store, RequestLog{Key: key, At: base, Model: "sol", Tokens: usage.Tokens{Input: 41, Output: 7,
		CacheRead: 108032, CacheWrite: 2000}, Latency: 3 * time.Second, Queue: 150 * time.Millisecond},
		RequestLog{Key: key, At: base, Model: "sol", Cost: 0, Priced: true})
	// Values above float64's integer precision and the repair audit trail
	// must survive; NULL cost and priced zero must remain distinct.
	if _, err := store.write.Exec(`UPDATE requests SET id = 9007199254740993, repaired = 1,
		original_input_tokens = 100355, original_cost_usd = 9.5 WHERE id = 1`); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := Export(ctx, path, &output); err != nil {
		t.Fatal(err)
	}
	var again bytes.Buffer
	if err := Export(ctx, path, &again); err != nil || !bytes.Equal(output.Bytes(), again.Bytes()) {
		t.Fatalf("repeated export changed: %v", err)
	}
	records := map[string]map[string]json.RawMessage{}
	counts := map[string]int{}
	var manifest *ExportManifest
	decoder := json.NewDecoder(&output)
	for {
		var record struct {
			ExportRecord
			Manifest *ExportManifest `json:"manifest"`
		}
		if err := decoder.Decode(&record); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			t.Fatal(err)
		}
		if manifest != nil {
			t.Fatal("records after the manifest")
		}
		if record.Manifest != nil {
			manifest = record.Manifest
			continue
		}
		var document map[string]json.RawMessage
		if err := json.Unmarshal(record.Document, &document); err != nil {
			t.Fatal(err)
		}
		records[record.Collection+"/"+record.Key] = document
		counts[record.Collection]++
	}
	if manifest == nil || manifest.Format != 1 || manifest.SchemaVersion != len(migrations) || len(manifest.Counts) != 7 {
		t.Fatalf("manifest = %+v", manifest)
	}
	for _, collection := range exportCollections {
		var want int
		if err := store.read.QueryRow("SELECT count(*) FROM " + collection.name).Scan(&want); err != nil {
			t.Fatal(err)
		}
		if want == 0 || counts[collection.name] != want || manifest.Counts[collection.name] != want {
			t.Errorf("%s: exported %d, want %d", collection.name, counts[collection.name], want)
		}
	}
	request := records["requests/9007199254740993"]
	for field, want := range map[string]string{"id": "9007199254740993", "input_tokens": "41", "output_tokens": "7",
		"cache_read_tokens": "108032", "cache_write_tokens": "2000", "cost_usd": "null", "repaired": "true",
		"original_input_tokens": "100355", "original_cost_usd": "9.5", "queue_ms": "150", "latency_ms": "3000"} {
		if string(request[field]) != want {
			t.Errorf("request %s = %s, want %s", field, request[field], want)
		}
	}
	if string(records["requests/2"]["cost_usd"]) != "0" || string(records["requests/2"]["original_cost_usd"]) != "null" {
		t.Fatal("priced zero or unrepaired NULL was lost")
	}
	if string(records["pins/"+key]["compacted"]) != "true" || string(records["pins/"+key]["turns"]) != "1" ||
		string(records["pins/"+key]["decided_level"]) != `"high"` {
		t.Fatal("pin redecision state was lost")
	}
	var event struct {
		Prompt        string
		Probabilities map[string]float64
		Attempts      []decide.Attempt
	}
	// Structured columns must be JSON objects/arrays, not encoded strings.
	data, _ := json.Marshal(records["events/1"])
	if err := json.Unmarshal(data, &event); err != nil || event.Prompt != "quoted \"text\"\n中文 café" ||
		event.Probabilities["sol"] != 0.9 || len(event.Attempts) != 1 || event.Attempts[0].Response != "raw reply" {
		t.Fatalf("event = %+v, %v", event, err)
	}
	var settings RoutingSettings
	if err := json.Unmarshal(records["routing_settings/1"]["settings"], &settings); err != nil ||
		settings.Mode != "fixed" || settings.Preferences.Priority != decide.PriorityCost {
		t.Fatalf("settings = %+v, %v", settings, err)
	}
}

type firstWrite struct {
	bytes.Buffer
	change func()
}

func (w *firstWrite) Write(data []byte) (int, error) {
	if w.change != nil {
		change := w.change
		w.change = nil
		change()
	}
	return w.Buffer.Write(data)
}

func TestExportUsesOneSnapshotWhileRouterWrites(t *testing.T) {
	path := filepath.Join(t.TempDir(), "router.db")
	store := open(t, path)
	if _, err := store.Create(t.Context(), Pin{Key: "first", Model: "sol"}, Event{Key: "first", Kind: Decision, Model: "sol"}); err != nil {
		t.Fatal(err)
	}
	output := &firstWrite{change: func() {
		if _, err := store.Create(t.Context(), Pin{Key: "later", Model: "sol"}, Event{Key: "later", Kind: Decision, Model: "sol"}); err != nil {
			t.Fatal(err)
		}
		logRequests(t, store, RequestLog{Key: "later", Model: "sol"})
	}}
	if err := Export(t.Context(), path, output); err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(output.Bytes(), []byte("later")) || bytes.Contains(output.Bytes(), []byte(`"collection":"requests"`)) {
		t.Fatal("export mixed rows written after its snapshot")
	}
	if pin, err := store.Get(t.Context(), "later"); err != nil || pin == nil {
		t.Fatalf("concurrent write did not finish: %+v, %v", pin, err)
	}
}

func TestExportRejectsMissingIncompatibleAndCorruptSources(t *testing.T) {
	path := filepath.Join(t.TempDir(), "router.db")
	if err := Export(t.Context(), path, io.Discard); err == nil {
		t.Fatal("missing source accepted")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("export created missing source: %v", err)
	}
	store := open(t, path)
	if _, err := store.write.Exec("PRAGMA user_version = 1"); err != nil {
		t.Fatal(err)
	}
	if err := Export(t.Context(), path, io.Discard); err == nil {
		t.Fatal("incompatible source accepted")
	}
	var version int
	if err := store.read.QueryRow("PRAGMA user_version").Scan(&version); err != nil || version != 1 {
		t.Fatalf("source migrated: version %d, %v", version, err)
	}
	if _, err := store.write.Exec("PRAGMA user_version = 999"); err != nil {
		t.Fatal(err)
	}
	if err := Export(t.Context(), path, io.Discard); err == nil {
		t.Fatal("newer source accepted")
	}
	if _, err := store.write.Exec(fmt.Sprintf("PRAGMA user_version = %d", len(migrations))); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Create(t.Context(), Pin{Key: "k", Model: "sol"}, Event{Key: "k", Kind: Decision, Model: "sol"}); err != nil {
		t.Fatal(err)
	}
	for _, statement := range []string{`UPDATE events SET attempts = 'corrupt'`, `UPDATE events SET attempts = '[]', escalated = 2`} {
		if _, err := store.write.Exec(statement); err != nil {
			t.Fatal(err)
		}
		var output bytes.Buffer
		if err := Export(t.Context(), path, &output); err == nil || bytes.Contains(output.Bytes(), []byte(`"manifest"`)) {
			t.Fatalf("corrupt row accepted: %v", err)
		}
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestExportEscapesSourcePathAndPropagatesCancellation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "router.db")
	store := open(t, path)
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	escaped := path + " ?#copy.db"
	if err := os.Rename(path, escaped); err != nil {
		t.Fatal(err)
	}
	if err := Export(t.Context(), escaped, io.Discard); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := Export(t.Context(), escaped, &output); err != nil {
		t.Fatal(err)
	}
	var line struct {
		Manifest ExportManifest `json:"manifest"`
	}
	if err := json.Unmarshal(output.Bytes(), &line); err != nil || len(line.Manifest.Counts) != 7 {
		t.Fatalf("empty export = %s, %v", output.Bytes(), err)
	}
	for collection, count := range line.Manifest.Counts {
		if count != 0 {
			t.Errorf("empty %s count = %d", collection, count)
		}
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err := Export(ctx, escaped, io.Discard); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled export = %v", err)
	}
}

type failedExport struct{ err error }

func (w failedExport) Write([]byte) (int, error) { return 0, w.err }

func TestExportPropagatesOutputFailure(t *testing.T) {
	path := filepath.Join(t.TempDir(), "router.db")
	open(t, path)
	want := errors.New("output unavailable")
	if err := Export(t.Context(), path, failedExport{want}); !errors.Is(err, want) {
		t.Fatalf("export error = %v, want %v", err, want)
	}
}
