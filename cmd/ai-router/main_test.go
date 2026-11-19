package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/MarioGK/AI-Router/internal/pin"
)

func fakes(t *testing.T, models string) (cpa, jev *httptest.Server) {
	cpa = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			_, _ = fmt.Fprintf(w, `{"data":[%s]}`, models)
		case "/v1/chat/completions":
			_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"{\"probabilities\":{\"deepseek-flash\":1}}"}}]}`))
		}
	}))
	jev = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// The router's decider state starts with the images line, and
		// nothing else.
		var request struct {
			State string `json:"state"`
		}
		if json.NewDecoder(r.Body).Decode(&request) != nil || !strings.HasPrefix(request.State, "images=no\n\n") {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		_, _ = w.Write([]byte(`{"answers":{"model":{"choice":"deepseek-flash","probabilities":{"deepseek-flash":0.97,"glm-5.3":0.03}}}}`))
	}))
	t.Cleanup(cpa.Close)
	t.Cleanup(jev.Close)
	return cpa, jev
}

func writeConfig(t *testing.T, cpa, jev string) string {
	t.Setenv("CLIPROXYAPI_API_KEY", "proxy-key")
	t.Setenv("JEV_API_KEY", "jev-key")
	path := filepath.Join(t.TempDir(), "config.yaml")
	document := fmt.Sprintf(`
upstream: {base_url: %q}
auto:
  default_model: glm-5.3
  candidates:
    - {id: deepseek-flash, description: cheap}
    - {id: glm-5.3, description: everyday}
deciders:
  - {name: jev, type: systemone, url: %q, model: jev-latest, api_key_env: JEV_API_KEY}
  - {name: judge, type: llm, model: gpt-6-luna}
`, cpa, jev)
	if err := os.WriteFile(path, []byte(document), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestCheckProbesCatalogAndDeciders(t *testing.T) {
	cpa, jev := fakes(t, `{"id":"deepseek-flash"},{"id":"glm-5.3"}`)
	var stdout, stderr bytes.Buffer
	if code := run([]string{"check", "--config", writeConfig(t, cpa.URL, jev.URL)}, &stdout, &stderr); code != 0 {
		t.Fatalf("check exited %d: %s %s", code, stdout.String(), stderr.String())
	}
	var report struct {
		Catalog struct {
			Loaded bool `json:"loaded"`
		} `json:"catalog"`
		Deciders map[string]map[string]any `json:"deciders"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil || !report.Catalog.Loaded {
		t.Fatalf("report = %s, %v", stdout.String(), err)
	}
	if report.Deciders["jev"]["model"] != "deepseek-flash" || report.Deciders["judge"]["model"] != "deepseek-flash" {
		t.Fatalf("deciders = %v", report.Deciders)
	}
}

func TestCheckFailsOnMissingModels(t *testing.T) {
	cpa, jev := fakes(t, `{"id":"deepseek-flash"}`)
	var stdout, stderr bytes.Buffer
	if code := run([]string{"check", "--config", writeConfig(t, cpa.URL, jev.URL)}, &stdout, &stderr); code != 1 || !strings.Contains(stdout.String(), `"glm-5.3"`) {
		t.Fatalf("check exited %d: %s", code, stdout.String())
	}
}

func TestUsage(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := run(nil, &stdout, &stderr); code != 2 || !strings.Contains(stderr.String(), "usage") {
		t.Fatalf("no args exited %d", code)
	}
	if code := run([]string{"frobnicate"}, &stdout, &stderr); code != 2 {
		t.Fatalf("unknown command exited %d", code)
	}
	if code := run([]string{"serve", "--config", "/does/not/exist"}, &stdout, &stderr); code != 1 {
		t.Fatalf("missing config exited %d", code)
	}
}

func TestBenchFailsBelowTheLeastAccuracy(t *testing.T) {
	cpa, jev := fakes(t, `{"id":"deepseek-flash"},{"id":"glm-5.3"}`)
	cases := filepath.Join(t.TempDir(), "cases.yaml")
	document := "cases:\n  - {name: tiny, expect: [deepseek-flash], prompt: rename x}\n  - {name: feature, expect: [glm-5.3], prompt: add pagination}\n"
	if err := os.WriteFile(cases, []byte(document), 0o600); err != nil {
		t.Fatal(err)
	}
	config := writeConfig(t, cpa.URL, jev.URL)
	var stdout, stderr bytes.Buffer
	if code := run([]string{"bench", "--config", config, "--cases", cases, "--min-accuracy", "0.5"}, &stdout, &stderr); code != 0 {
		t.Fatalf("bench at 50%% exited %d: %s", code, stderr.String())
	}
	stderr.Reset()
	if code := run([]string{"bench", "--config", config, "--cases", cases, "--min-accuracy", "0.9"}, &stdout, &stderr); code != 1 ||
		!strings.Contains(stderr.String(), "chain(escalate) routed 50% of the cases correctly, below 90%") {
		t.Fatalf("bench below the least accuracy exited %d: %s", code, stderr.String())
	}
}

func TestBackupWritesASnapshot(t *testing.T) {
	directory := t.TempDir()
	database := filepath.Join(directory, "router.db")
	store, err := pin.Open(database)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	snapshot := filepath.Join(directory, "snapshot.db")
	var stdout, stderr bytes.Buffer
	if code := run([]string{"backup", "--db", database, "--to", snapshot}, &stdout, &stderr); code != 0 {
		t.Fatalf("backup exited %d: %s", code, stderr.String())
	}
	if info, err := os.Stat(snapshot); err != nil || info.Size() == 0 {
		t.Fatalf("snapshot = %v, %v", info, err)
	}
	if code := run([]string{"backup", "--db", database, "--to", snapshot}, &stdout, &stderr); code != 1 {
		t.Fatalf("a backup over an existing file exited %d", code)
	}
	missing := filepath.Join(directory, "missing.db")
	if code := run([]string{"backup", "--db", missing, "--to", filepath.Join(directory, "other.db")}, &stdout, &stderr); code != 1 {
		t.Fatalf("a backup of a missing database exited %d", code)
	}
	if _, err := os.Stat(missing); !os.IsNotExist(err) {
		t.Fatalf("a backup of a missing database created it: %v", err)
	}
	if code := run([]string{"backup", "--db", database}, &stdout, &stderr); code != 2 {
		t.Fatalf("a backup without --to exited %d", code)
	}
}
