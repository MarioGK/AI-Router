package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/MarioGK/AI-Router/internal/pin"
)

func TestExportStateWritesPrivateFileWithoutConfig(t *testing.T) {
	directory := t.TempDir()
	database := filepath.Join(directory, "router.db")
	store, err := pin.Open(database)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Create(t.Context(), pin.Pin{Key: "k", Model: "sol"},
		pin.Event{Key: "k", Kind: pin.Decision, Model: "sol", Prompt: "private prompt"}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	to := filepath.Join(directory, "export.ndjson")
	var stdout, stderr bytes.Buffer
	if code := run([]string{"export-state", "--db", database, "--to", to, "--config", "/does/not/exist"}, &stdout, &stderr); code != 0 {
		t.Fatalf("export-state = %d, %s", code, stderr.String())
	}
	data, err := os.ReadFile(to)
	if err != nil || !bytes.Contains(data, []byte(`"collection":"pins"`)) || !bytes.Contains(data, []byte(`"manifest"`)) {
		t.Fatalf("export = %s, %v", data, err)
	}
	if info, err := os.Stat(to); err != nil || info.Mode().Perm() != 0o600 {
		t.Fatalf("private file = %v, %v", info, err)
	}
	if stdout.Len() != 0 || stderr.Len() != 0 {
		t.Fatal("export printed data instead of writing the private file")
	}
	if code := run([]string{"export-state", "--db", database, "--to", to}, &stdout, &stderr); code != 1 {
		t.Fatalf("existing export accepted: %d", code)
	}
	again, err := os.ReadFile(to)
	if err != nil || !bytes.Equal(again, data) {
		t.Fatalf("existing export replaced: %v", err)
	}
}

func TestExportStateRemovesFailedOutputAndKeepsSource(t *testing.T) {
	directory := t.TempDir()
	database := filepath.Join(directory, "missing.db")
	to := filepath.Join(directory, "export.ndjson")
	var stdout, stderr bytes.Buffer
	if code := run([]string{"export-state", "--db", database, "--to", to}, &stdout, &stderr); code != 1 {
		t.Fatalf("missing source accepted: %d", code)
	}
	for _, path := range []string{database, to} {
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("failed export left %s: %v", path, err)
		}
	}
	broken := filepath.Join(directory, "broken.db")
	if err := os.WriteFile(broken, []byte("not a database"), 0o600); err != nil {
		t.Fatal(err)
	}
	if code := run([]string{"export-state", "--db", broken, "--to", to}, &stdout, &stderr); code != 1 {
		t.Fatalf("corrupt source accepted: %d", code)
	}
	if _, err := os.Stat(to); !os.IsNotExist(err) {
		t.Fatalf("corrupt source left output: %v", err)
	}
	if data, err := os.ReadFile(broken); err != nil || string(data) != "not a database" {
		t.Fatalf("corrupt source changed: %s, %v", data, err)
	}
	if code := run([]string{"export-state", "--db", database}, &stdout, &stderr); code != 2 || !strings.Contains(stderr.String(), "needs --to") {
		t.Fatalf("missing destination = %d, %s", code, stderr.String())
	}
	store, err := pin.Open(database)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(database)
	if err != nil {
		t.Fatal(err)
	}
	if code := run([]string{"export-state", "--db", database, "--to", database}, &stdout, &stderr); code != 1 {
		t.Fatalf("source overwrite accepted: %d", code)
	}
	after, err := os.ReadFile(database)
	if err != nil || !bytes.Equal(before, after) {
		t.Fatalf("source changed: %v", err)
	}
}

func TestExportStateRejectsSQLiteSidecarsThroughAliases(t *testing.T) {
	directory := t.TempDir()
	database := filepath.Join(directory, "router.db")
	store, err := pin.Open(database)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	alias := filepath.Join(t.TempDir(), "source")
	if err := os.Symlink(directory, alias); err != nil {
		t.Fatal(err)
	}
	fileAlias := filepath.Join(directory, "alias.db")
	if err := os.Symlink(database, fileAlias); err != nil {
		t.Fatal(err)
	}
	for _, source := range []string{database, filepath.Join(alias, "router.db"), fileAlias} {
		for _, suffix := range []string{"-wal", "-shm", "-journal"} {
			for _, to := range []string{database + suffix, filepath.Join(alias, "router.db"+suffix), source + suffix} {
				var stdout, stderr bytes.Buffer
				if code := run([]string{"export-state", "--db", source, "--to", to}, &stdout, &stderr); code != 1 ||
					!strings.Contains(stderr.String(), "SQLite sidecar") {
					t.Fatalf("sidecar source=%s to=%s = %d, %s", source, to, code, stderr.String())
				}
				if _, err := os.Stat(to); !os.IsNotExist(err) {
					t.Fatalf("export created %s: %v", to, err)
				}
			}
		}
	}
	// The rejected paths must not damage a subsequent source open.
	reopened, err := pin.Open(database)
	if err != nil {
		t.Fatal(err)
	}
	if err := reopened.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestExportStateNormalizesSourceBeforeResolvingAliases(t *testing.T) {
	directory := t.TempDir()
	database := filepath.Join(directory, "router.db")
	store, err := pin.Open(database)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	aliasDirectory := t.TempDir()
	otherDirectory := t.TempDir()
	if err := os.Mkdir(filepath.Join(otherDirectory, "sub"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(otherDirectory, "router.db"), []byte("different source"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(database, filepath.Join(aliasDirectory, "router.db")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(otherDirectory, "sub"), filepath.Join(aliasDirectory, "link")); err != nil {
		t.Fatal(err)
	}
	// Preserve '..' here: Export cleans it before SQLite resolves symlinks.
	source := aliasDirectory + "/link/../router.db"
	for _, suffix := range []string{"-wal", "-shm", "-journal"} {
		to := database + suffix
		var stdout, stderr bytes.Buffer
		if code := run([]string{"export-state", "--db", source, "--to", to}, &stdout, &stderr); code != 1 ||
			!strings.Contains(stderr.String(), "SQLite sidecar") {
			t.Fatalf("normalized source=%s to=%s = %d, %s", source, to, code, stderr.String())
		}
		if _, err := os.Stat(to); !os.IsNotExist(err) {
			t.Fatalf("export created %s: %v", to, err)
		}
	}
}
