package accounts

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"sync"
	"testing"
	"time"
)

const managementKey = "management-key-for-tests"

// management is a fake CLIProxy management API.
type management struct {
	server  *httptest.Server
	mu      sync.Mutex
	files   []map[string]any
	models  map[string][]string
	patches []map[string]any
	calls   []string

	// modelsStatus fails every model listing when set.
	modelsStatus int
}

func newManagement(t *testing.T) *management {
	m := &management{models: map[string][]string{}}
	m.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		m.mu.Lock()
		defer m.mu.Unlock()
		m.calls = append(m.calls, r.Method+" "+r.URL.Path)
		if r.Header.Get("Authorization") != "Bearer "+managementKey {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/v0/management/auth-files":
			_ = json.NewEncoder(w).Encode(map[string]any{"files": m.files})
		case r.Method == http.MethodGet && r.URL.Path == "/v0/management/auth-files/models":
			if m.modelsStatus != 0 {
				w.WriteHeader(m.modelsStatus)
				return
			}
			var models []map[string]string
			for _, id := range m.models[r.URL.Query().Get("name")] {
				models = append(models, map[string]string{"id": id})
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"models": models})
		case r.Method == http.MethodPatch && r.URL.Path == "/v0/management/auth-files/fields":
			body, _ := io.ReadAll(r.Body)
			var patch map[string]any
			_ = json.Unmarshal(body, &patch)
			m.patches = append(m.patches, patch)
			for _, f := range m.files {
				if f["name"] == patch["name"] {
					f["priority"] = patch["priority"]
				}
			}
			_, _ = w.Write([]byte(`{"status":"ok"}`))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	t.Cleanup(m.server.Close)
	return m
}

func (m *management) patched() []map[string]any {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.patches)
}

// observe sets the quota CLIProxy last saw on an account's responses.
func (m *management) observe(name string, signals map[string]string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, f := range m.files {
		if f["name"] == name {
			f["quota"] = map[string]any{"observed_at": time.Now().Add(-time.Minute), "signals": signals}
		}
	}
}

// forget drops what CLIProxy saw of every account's quota, as a CLIProxy
// restart does.
func (m *management) forget() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, f := range m.files {
		delete(f, "quota")
	}
}

// in is a Unix time the given duration from now, as the quota headers
// carry it.
func in(d time.Duration) string {
	return strconv.FormatInt(time.Now().Add(d).Unix(), 10)
}

// claude is a Claude account's unified rate limit headers, as CLIProxy
// records them: the five-hour and weekly windows used and their resets.
func claude(fiveHour, weekly float64, fiveHourReset, weeklyReset time.Duration) map[string]string {
	status := func(used float64) string {
		if used >= 1 {
			return "rejected"
		}
		return "allowed"
	}
	return map[string]string{
		"Anthropic-Ratelimit-Unified-5h-Utilization":          strconv.FormatFloat(fiveHour, 'f', -1, 64),
		"Anthropic-Ratelimit-Unified-5h-Reset":                in(fiveHourReset),
		"Anthropic-Ratelimit-Unified-5h-Status":               status(fiveHour),
		"Anthropic-Ratelimit-Unified-7d-Utilization":          strconv.FormatFloat(weekly, 'f', -1, 64),
		"Anthropic-Ratelimit-Unified-7d-Reset":                in(weeklyReset),
		"Anthropic-Ratelimit-Unified-7d-Status":               status(weekly),
		"Anthropic-Ratelimit-Unified-Status":                  status(max(fiveHour, weekly)),
		"Anthropic-Ratelimit-Unified-Representative-Claim":    "five_hour",
		"Anthropic-Ratelimit-Unified-Overage-Status":          "rejected",
		"Anthropic-Ratelimit-Unified-Overage-Disabled-Reason": "org_level_disabled",
		"Anthropic-Ratelimit-Unified-Fallback-Percentage":     "0.5",
	}
}

// codex is a ChatGPT account's rate limit headers: a weekly primary window
// and no secondary one, as a Pro plan reports them.
func codex(used int, reset time.Duration) map[string]string {
	return map[string]string{
		"X-Codex-Plan-Type":                     "pro",
		"X-Codex-Active-Limit":                  "premium",
		"X-Codex-Credits-Has-Credits":           "False",
		"X-Codex-Primary-Used-Percent":          strconv.Itoa(used),
		"X-Codex-Primary-Window-Minutes":        "10080",
		"X-Codex-Primary-Reset-At":              in(reset),
		"X-Codex-Primary-Reset-After-Seconds":   strconv.Itoa(int(reset.Seconds())),
		"X-Codex-Secondary-Used-Percent":        "0",
		"X-Codex-Secondary-Window-Minutes":      "0",
		"X-Codex-Secondary-Reset-After-Seconds": "0",
	}
}

// team is the deployment's shape: two Claude accounts, one at its weekly
// limit, and three ChatGPT accounts, two spent.
func team(t *testing.T) *management {
	m := newManagement(t)
	m.files = []map[string]any{
		{"name": "claude-a.json", "provider": "claude", "email": "a@example.com"},
		{"name": "claude-b.json", "provider": "claude", "email": "b@example.com", "priority": -10},
		{"name": "codex-a.json", "provider": "codex", "account": "x@example.com",
			"id_token": map[string]any{"plan_type": "plus"}},
		{"name": "codex-b.json", "provider": "codex", "account": "y@example.com",
			"id_token": map[string]any{"plan_type": "plus"}},
		{"name": "codex-c.json", "provider": "codex", "account": "z@example.com", "priority": "0",
			"id_token": map[string]any{"plan_type": "pro"}},
		{"name": "gemini.json", "provider": "gemini"},
	}
	for _, name := range []string{"claude-a.json", "claude-b.json"} {
		m.models[name] = []string{"claude-opus-5-5", "claude-sonnet-5-5"}
	}
	for _, name := range []string{"codex-a.json", "codex-b.json", "codex-c.json"} {
		m.models[name] = []string{"gpt-6.1-sol", "gpt-6-luna"}
	}
	m.observe("claude-a.json", claude(0, 1, 3*time.Hour, 50*time.Hour))
	m.observe("claude-b.json", claude(0.04, 0.16, 2*time.Hour, 6*24*time.Hour))
	m.observe("codex-a.json", codex(100, 101*time.Hour))
	m.observe("codex-b.json", codex(100, 102*time.Hour))
	m.observe("codex-c.json", codex(93, 125*time.Hour))
	return m
}

func watcher(m *management) *Watcher {
	return New(m.server.URL, managementKey, []string{Claude, Codex}, -10, m.server.Client(), slog.New(slog.DiscardHandler))
}

func byName(w *Watcher) map[string]Account {
	accounts := map[string]Account{}
	for _, a := range w.Snapshot().Accounts {
		accounts[a.Name] = a
	}
	return accounts
}

func TestRefreshReadsQuotaAndMovesLimitedAccountsBack(t *testing.T) {
	m := team(t)
	w := watcher(m)
	if !w.Available("gpt-6.1-sol") {
		t.Fatal("a model is unavailable before the first refresh")
	}
	if err := w.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	snapshot := w.Snapshot()
	if len(snapshot.Accounts) != 5 || snapshot.CheckedAt.IsZero() {
		t.Fatalf("snapshot = %+v", snapshot)
	}
	accounts := byName(w)
	claudeA := accounts["claude-a.json"]
	if !claudeA.Limited || time.Until(claudeA.Until) < 49*time.Hour || claudeA.Priority != -10 ||
		len(claudeA.Windows) != 2 || claudeA.Observed.IsZero() {
		t.Fatalf("claude a = %+v", claudeA)
	}
	if claudeB := accounts["claude-b.json"]; claudeB.Limited || claudeB.Priority != 0 || claudeB.Windows[0].Name != "five_hour" ||
		claudeB.Windows[0].Used != 0.04 || claudeB.Windows[1].Name != "seven_day" {
		t.Fatalf("claude b = %+v", claudeB)
	}
	if codexA := accounts["codex-a.json"]; !codexA.Limited || codexA.Plan != "pro" || codexA.Windows[0].Name != "weekly" ||
		time.Until(codexA.Until) < 4*24*time.Hour {
		t.Fatalf("codex a = %+v", codexA)
	}
	if codexC := accounts["codex-c.json"]; codexC.Limited || codexC.Windows[0].Used != 0.93 || len(codexC.Windows) != 1 {
		t.Fatalf("codex c = %+v", codexC)
	}

	// The limited accounts went behind the others, the recovered Claude
	// account came back, and the rest kept their priority.
	var moved []string
	for _, patch := range m.patched() {
		moved = append(moved, patch["name"].(string)+"="+jsonNumber(patch["priority"]))
	}
	slices.Sort(moved)
	if want := []string{"claude-a.json=-10", "claude-b.json=0", "codex-a.json=-10", "codex-b.json=-10"}; !slices.Equal(moved, want) {
		t.Fatalf("priority changes = %q, want %q", moved, want)
	}
	if err := w.Refresh(context.Background()); err != nil || len(m.patched()) != 4 {
		t.Fatalf("a second refresh changed priorities again: %v, %d", err, len(m.patched()))
	}
}

func TestClaudeResetReadsFreshManagementQuota(t *testing.T) {
	m := team(t)
	m.observe("claude-a.json", claude(1, 0.2, time.Hour, 48*time.Hour))
	w := watcher(m)
	now := time.Now()
	reset, ok := w.ClaudeReset(context.Background(), "claude-opus-5-5", now)
	if !ok || reset.Before(now.Add(50*time.Minute)) || reset.After(now.Add(70*time.Minute)) {
		t.Fatalf("ClaudeReset = %v, %v", reset, ok)
	}
	for _, call := range m.calls {
		if call != "GET /v0/management/auth-files" {
			t.Errorf("fresh reset made an unexpected management call: %s", call)
		}
	}
}

func TestTheWatcherNeverCallsTheProviders(t *testing.T) {
	m := team(t)
	if err := watcher(m).Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	for _, call := range m.calls {
		if call != "GET /v0/management/auth-files" && call != "GET /v0/management/auth-files/models" &&
			call != "PATCH /v0/management/auth-files/fields" {
			t.Errorf("the watcher called %s", call)
		}
	}
}

func jsonNumber(value any) string {
	encoded, _ := json.Marshal(value)
	return string(encoded)
}

func TestAvailabilityFollowsTheAccountsThatServeAModel(t *testing.T) {
	m := team(t)
	w := watcher(m)
	if err := w.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	for model, want := range map[string]bool{
		"gpt-6.1-sol": true, "claude-opus-5-5": true, "glm-5.3": true, "deepseek-flash": true,
	} {
		if got := w.Available(model); got != want {
			t.Errorf("Available(%s) = %v, want %v", model, got, want)
		}
	}

	// The last ChatGPT account reaches its limit, and Claude b's Opus window
	// fills: Sol and Opus go, Sonnet stays.
	m.observe("codex-c.json", map[string]string{
		"X-Codex-Primary-Used-Percent": "100", "X-Codex-Primary-Window-Minutes": "300",
		"X-Codex-Primary-Reset-After-Seconds": "600",
	})
	opus := claude(0.04, 0.4, 2*time.Hour, 6*24*time.Hour)
	opus["Anthropic-Ratelimit-Unified-7d_opus-Utilization"] = "1.0"
	opus["Anthropic-Ratelimit-Unified-7d_opus-Status"] = "rejected"
	opus["Anthropic-Ratelimit-Unified-7d_opus-Reset"] = in(60 * time.Hour)
	m.observe("claude-b.json", opus)
	if err := w.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	for model, want := range map[string]bool{
		"gpt-6.1-sol": false, "gpt-6-luna": false, "claude-opus-5-5": false, "claude-sonnet-5-5": true, "glm-5.3": true,
	} {
		if got := w.Available(model); got != want {
			t.Errorf("after the limits, Available(%s) = %v, want %v", model, got, want)
		}
	}
	accounts := byName(w)
	if c := accounts["codex-c.json"]; c.Windows[0].Name != "5h" || time.Until(c.Until) > 11*time.Minute {
		t.Errorf("codex c = %+v", c)
	}
	if b := accounts["claude-b.json"]; b.Limited || b.Windows[2].Name != "seven_day_opus" || !b.Windows[2].Full() {
		t.Errorf("claude b = %+v", b)
	}
}

func TestWindowsThatResetNoLongerLimit(t *testing.T) {
	m := team(t)
	// Claude a's weekly window was spent, but it has reset since CLIProxy
	// last saw a response of it: the account is usable again.
	m.observe("claude-a.json", claude(0.2, 1, 3*time.Hour, -time.Minute))
	w := watcher(m)
	if err := w.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	if a := byName(w)["claude-a.json"]; a.Limited || a.Priority != 0 || len(a.Windows) != 1 {
		t.Fatalf("claude a = %+v", a)
	}
}

func TestAnAccountWithoutQuotaCountsAsUsableButKeepsItsPriority(t *testing.T) {
	m := team(t)
	m.forget()
	w := watcher(m)
	if err := w.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	for _, a := range w.Snapshot().Accounts {
		if a.Limited || len(a.Windows) != 0 || !a.Observed.IsZero() {
			t.Errorf("account %+v", a)
		}
	}
	if !w.Available("gpt-6.1-sol") || !w.Available("claude-opus-5-5") {
		t.Fatal("unknown quota made a model unavailable")
	}
	// Claude b sits behind the others and nothing says it may return.
	if moved := m.patched(); len(moved) != 0 {
		t.Fatalf("an unknown quota moved accounts: %v", moved)
	}
	// Once CLIProxy sees it serve with quota left, it returns.
	m.observe("claude-b.json", claude(0.04, 0.16, 2*time.Hour, 6*24*time.Hour))
	if err := w.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	if moved := m.patched(); len(moved) != 1 || moved[0]["name"] != "claude-b.json" || jsonNumber(moved[0]["priority"]) != "0" {
		t.Fatalf("priority changes = %v", moved)
	}
}

func TestALostQuotaKeepsWhatTheLastRefreshFound(t *testing.T) {
	m := team(t)
	m.observe("codex-c.json", codex(100, 10*time.Minute))
	w := watcher(m)
	if err := w.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	spent := byName(w)
	if w.Available("gpt-6.1-sol") {
		t.Fatal("every ChatGPT account is spent, yet Sol is available")
	}

	// CLIProxy restarts, so it knows no quota until the accounts serve
	// again, and the model listing fails: the spent accounts stay behind
	// the others until their windows reset.
	patched := len(m.patched())
	m.forget()
	m.mu.Lock()
	m.modelsStatus = http.StatusBadGateway
	m.mu.Unlock()
	if err := w.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	if moved := m.patched()[patched:]; len(moved) != 0 {
		t.Fatalf("a lost quota moved accounts: %v", moved)
	}
	for _, a := range w.Snapshot().Accounts {
		if a.Provider != Codex {
			continue
		}
		last := spent[a.Name]
		if !a.Limited || !a.Until.Equal(last.Until) || !slices.Equal(a.Windows, last.Windows) ||
			!slices.Equal(a.Models, last.Models) || a.Err == "" {
			t.Errorf("account %+v, was %+v", a, last)
		}
	}
	if w.Available("gpt-6.1-sol") {
		t.Fatal("a lost quota made Sol available")
	}

	// Past its reset, nothing of the limit is held.
	last := spent["codex-c.json"]
	var reset Account
	reset.hold(last, last.Until.Add(time.Second))
	if reset.Limited || len(reset.Windows) != 0 {
		t.Fatalf("held past the reset: %+v", reset)
	}
}

func TestModelListsAreReadEveryRefresh(t *testing.T) {
	m := team(t)
	m.observe("codex-c.json", codex(100, 10*time.Minute))
	listed := m.models
	m.models = map[string][]string{} // CLIProxy has not registered the models yet
	w := watcher(m)
	if err := w.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !w.Available("gpt-6.1-sol") {
		t.Fatal("a model no account lists is unavailable")
	}
	m.mu.Lock()
	m.models = listed
	m.mu.Unlock()
	if err := w.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	if w.Available("gpt-6.1-sol") {
		t.Fatal("the late model lists were never read")
	}
}

func TestAStalledCallCannotStopTheWatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v0/management/auth-files":
			_, _ = w.Write([]byte(`{"files":[{"name":"claude-a.json","provider":"claude"}]}`))
		default:
			// The model listing never answers, until the watcher hangs up.
			<-r.Context().Done()
		}
	}))
	t.Cleanup(server.Close)
	w := New(server.URL, managementKey, []string{Claude}, -10, server.Client(), slog.New(slog.DiscardHandler))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go w.Run(ctx, 50*time.Millisecond)
	var first time.Time
	for deadline := time.Now().Add(5 * time.Second); ; time.Sleep(10 * time.Millisecond) {
		snapshot := w.Snapshot()
		if first.IsZero() && len(snapshot.Accounts) == 1 && snapshot.Accounts[0].Err != "" {
			first = snapshot.CheckedAt
		}
		if !first.IsZero() && snapshot.CheckedAt.After(first) {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("the watch stalled: %+v", snapshot)
		}
	}
}

func TestAFailedListingKeepsTheLastView(t *testing.T) {
	m := team(t)
	w := watcher(m)
	if err := w.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	m.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
	})
	if err := w.Refresh(context.Background()); err == nil {
		t.Fatal("a failed listing refreshed")
	}
	if snapshot := w.Snapshot(); len(snapshot.Accounts) != 5 || snapshot.Err == "" {
		t.Fatalf("snapshot = %+v", snapshot)
	}
	if w.Available("gpt-6.1-sol") != true {
		t.Fatal("the last view was lost")
	}
}

func TestWindowNames(t *testing.T) {
	for seconds, want := range map[int64]string{18000: "5h", 604800: "weekly", 86400: "1d", 0: "window"} {
		if got := windowName(seconds); got != want {
			t.Errorf("windowName(%d) = %q, want %q", seconds, got, want)
		}
	}
	for window, want := range map[string]string{"5h": "five_hour", "7d": "seven_day", "7d_opus": "seven_day_opus", "1d": "1d"} {
		if got := claudeWindowName(window); got != want {
			t.Errorf("claudeWindowName(%q) = %q, want %q", window, got, want)
		}
	}
}
