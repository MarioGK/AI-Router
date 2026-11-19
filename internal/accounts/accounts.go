// Package accounts watches the quota of CLIProxy's OAuth subscription
// accounts (Claude and ChatGPT) through its management API. It reads the
// quota CLIProxy observes on the responses each account serves, and never
// calls the providers itself: a quota probe is traffic no real Claude Code
// or Codex client sends. CLIProxy fills one account at a time; the watcher
// moves an account at its limit behind the others, so new conversations go
// to the next one, moves it back once its window resets, and tells the
// router which models no account can serve.
package accounts

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"sync"
	"time"
)

// Providers the watcher can read quota for.
const (
	Claude = "claude"
	Codex  = "codex"
)

// Window is one quota window of an account.
type Window struct {
	// Name is the provider's window name, e.g. five_hour, seven_day or
	// seven_day_opus for Claude, 5h or weekly for ChatGPT.
	Name string
	// Used is the fraction of the window spent, 0 to 1.
	Used     float64
	ResetsAt time.Time
	// refused records an explicit request refusal scoped to this family.
	refused bool
}

// Full reports whether the window is spent.
func (w Window) Full() bool {
	return w.Used >= 1
}

// Credits is the last passive observation of provider credits or extra usage.
// Unknown metadata has Known false and never unlocks an exhausted window.
type Credits struct {
	Known     bool
	Available bool
	Unlimited bool
	// Balance is the provider's validated nonnegative credit amount. An empty
	// balance is unknown, not zero; it is never converted to a currency.
	Balance string
	// DisabledReason is Anthropic's restriction on extra usage, when reported.
	DisabledReason string
	Observed       time.Time
}

// Account is one OAuth account and its quota.
type Account struct {
	// Name is the CLIProxy auth file.
	Name     string
	Provider string
	Email    string
	Plan     string
	// Priority is the CLIProxy selection priority the account has now.
	Priority int
	Disabled bool
	Windows  []Window
	Credits  Credits
	// Limited is set when the provider refuses the account further use
	// until a window resets, at Until.
	Limited bool
	Until   time.Time
	// Observed is when CLIProxy last saw the account's quota, on a
	// response the account served; zero when neither it nor the last
	// refresh knows any. An account whose quota CLIProxy lost keeps what
	// the last refresh found; with nothing known it counts as usable for
	// routing but keeps its priority, so an account never returns to
	// rotation without evidence that it may.
	Observed time.Time
	// Models are the models CLIProxy serves through the account.
	Models []string
	// Err is why the last model listing failed; a failed listing keeps the
	// previous models.
	Err string
	// refused distinguishes a provider refusal from spent included quota.
	refused bool
}

// Blocked reports whether the account cannot serve model: it is disabled,
// refused, or an applicable window is spent without available extra usage.
func (a Account) Blocked(model string) bool {
	if a.Disabled || a.Limited {
		return true
	}
	for _, w := range a.Windows {
		if family := windowFamily(w.Name); w.Full() && (family == "" || strings.Contains(model, family)) && (w.refused || !a.Credits.Available) {
			return true
		}
	}
	return false
}

// UsesCredits reports whether some models need credits because an included
// quota window is spent. It does not claim a measured amount has been charged.
func (a Account) UsesCredits() bool {
	if a.Disabled || a.Limited || !a.Credits.Available {
		return false
	}
	return slices.ContainsFunc(a.Windows, func(w Window) bool { return w.Full() && !w.refused })
}

// hold keeps what the last refresh learned about an account whose quota
// CLIProxy no longer knows, as after it restarted: when it was seen, the
// windows that have not reset and, until it resets, the limit.
func (a *Account) hold(last Account, now time.Time) {
	a.Observed = last.Observed
	a.Credits = last.Credits
	a.holdWindows(last, now)
	a.refused = last.refused && (last.Until.IsZero() || last.Until.After(now))
	a.applyLimits(now)
}

func (a *Account) holdWindows(last Account, now time.Time) {
	for _, w := range last.Windows {
		if w.ResetsAt.IsZero() || w.ResetsAt.After(now) {
			a.Windows = append(a.Windows, w)
		}
	}
}

// Snapshot is the last view of every watched account.
type Snapshot struct {
	Accounts  []Account
	CheckedAt time.Time
	// Err is why the last refresh failed as a whole.
	Err string
}

// Watcher polls the accounts and steers CLIProxy's selection priority.
type Watcher struct {
	base, key string
	client    *http.Client
	providers []string
	// limitedPriority is below credit-backed accounts, which in turn sit
	// below accounts with included quota remaining.
	limitedPriority int
	log             *slog.Logger

	mu       sync.RWMutex
	snapshot Snapshot
}

// refreshTimeout bounds one refresh, so a stalled management call cannot
// stop the watch.
const refreshTimeout = time.Minute

// New returns a watcher of the providers' accounts behind the CLIProxy at
// base, authenticated with its management key.
func New(base, key string, providers []string, limitedPriority int, client *http.Client, log *slog.Logger) *Watcher {
	return &Watcher{
		base: strings.TrimSuffix(base, "/") + "/v0/management", key: key, client: client,
		providers: providers, limitedPriority: limitedPriority, log: log,
	}
}

// Run refreshes now and then every interval until ctx ends, each refresh
// within the interval and at most a minute.
func (w *Watcher) Run(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		refreshCtx, cancel := context.WithTimeout(ctx, min(interval, refreshTimeout))
		err := w.Refresh(refreshCtx)
		cancel()
		if err != nil && ctx.Err() == nil {
			w.log.Warn("account quota refresh failed", "error", err)
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

// Snapshot returns the last view of the accounts.
func (w *Watcher) Snapshot() Snapshot {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.snapshot
}

// Available reports whether some account can serve model. Models no
// watched account serves (API-key upstreams such as GLM or DeepSeek) are
// always available, and so is everything before the first refresh.
func (w *Watcher) Available(model string) bool {
	w.mu.RLock()
	defer w.mu.RUnlock()
	serving := false
	for _, a := range w.snapshot.Accounts {
		if !slices.Contains(a.Models, model) {
			continue
		}
		if !a.Blocked(model) {
			return true
		}
		serving = true
	}
	return !serving
}

// ClaudeReset returns the earliest reset at which a Claude subscription
// account may serve model again. It only reads CLIProxy's management API; it
// never probes Anthropic. The direct read is needed after a router restart or
// immediately after CLIProxy records a refusal, before the periodic watcher
// refresh has incorporated the new quota observation.
func (w *Watcher) ClaudeReset(ctx context.Context, model string, now time.Time) (time.Time, bool) {
	if reset, ok := claudeReset(w.Snapshot().Accounts, model, now); ok {
		return reset, true
	}
	probeCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	files, err := w.authFiles(probeCtx)
	if err != nil {
		return time.Time{}, false
	}
	var earliest time.Time
	for _, file := range files {
		if file.Provider != Claude || file.Disabled {
			continue
		}
		account := Account{Provider: Claude}
		if !readQuota(file.Quota, &account, Account{}, now) {
			continue
		}
		if reset, ok := accountClaudeReset(account, model, now); ok && (earliest.IsZero() || reset.Before(earliest)) {
			earliest = reset
		}
	}
	return earliest, !earliest.IsZero()
}

func claudeReset(accounts []Account, model string, now time.Time) (time.Time, bool) {
	var earliest time.Time
	for _, account := range accounts {
		if account.Provider != Claude || len(account.Models) > 0 && !slices.Contains(account.Models, model) {
			continue
		}
		if reset, ok := accountClaudeReset(account, model, now); ok && (earliest.IsZero() || reset.Before(earliest)) {
			earliest = reset
		}
	}
	return earliest, !earliest.IsZero()
}

func accountClaudeReset(account Account, model string, now time.Time) (time.Time, bool) {
	if !account.Blocked(model) {
		return time.Time{}, false
	}
	reset := account.Until
	for _, window := range account.Windows {
		family := windowFamily(window.Name)
		if !window.Full() || family != "" && !strings.Contains(model, family) ||
			!window.ResetsAt.After(now) || !window.refused && account.Credits.Available {
			continue
		}
		if window.ResetsAt.After(reset) {
			reset = window.ResetsAt
		}
	}
	return reset, reset.After(now)
}

// Refresh reads every watched account's models and quota and moves
// accounts at their limit behind the others. An account whose model
// listing fails, or whose quota CLIProxy no longer knows, keeps what the
// previous refresh found. It reports ctx's error when ctx ended before
// every account was read.
func (w *Watcher) Refresh(ctx context.Context) error {
	files, err := w.authFiles(ctx)
	if err != nil {
		w.mu.Lock()
		w.snapshot.Err = err.Error()
		w.mu.Unlock()
		return err
	}
	last := map[string]Account{}
	for _, a := range w.Snapshot().Accounts {
		last[a.Name] = a
	}
	var accounts []Account
	for _, f := range files {
		if !slices.Contains(w.providers, f.Provider) {
			continue
		}
		account := Account{Name: f.Name, Provider: f.Provider, Email: f.email(), Plan: f.IDToken.PlanType,
			Priority: f.priority(), Disabled: f.Disabled}
		if account.Models, err = w.modelsOf(ctx, f.Name); err != nil {
			account.Models, account.Err = last[f.Name].Models, err.Error()
		}
		if !account.Disabled {
			if now := time.Now(); !readQuota(f.Quota, &account, last[f.Name], now) {
				account.hold(last[f.Name], now)
			}
			w.steer(ctx, &account)
		}
		accounts = append(accounts, account)
	}
	w.mu.Lock()
	w.snapshot = Snapshot{Accounts: accounts, CheckedAt: time.Now()}
	w.mu.Unlock()
	return ctx.Err()
}

// steer prefers included quota, then credits, then limited accounts.
// An account nothing is
// known about keeps its priority: when CLIProxy and the router both
// restarted, one moved behind the others may still be spent.
func (w *Watcher) steer(ctx context.Context, account *Account) {
	creditPriority := w.limitedPriority / 2
	want := max(0, creditPriority+1)
	if account.Limited {
		want = w.limitedPriority
	} else if account.UsesCredits() {
		want = creditPriority
	}
	if account.Priority == want || account.Observed.IsZero() {
		return
	}
	body, _ := json.Marshal(map[string]any{"name": account.Name, "priority": want})
	if err := w.call(ctx, http.MethodPatch, "/auth-files/fields", body, nil); err != nil {
		w.log.Warn("account priority change failed", "account", account.Email, "error", err)
		return
	}
	if account.Limited {
		w.log.Info("account at its limit moved behind the others", "provider", account.Provider, "account", account.Email,
			"until", account.Until)
	} else if account.UsesCredits() {
		w.log.Info("account using credits moved behind included quota", "provider", account.Provider, "account", account.Email)
	} else {
		w.log.Info("account back in rotation", "provider", account.Provider, "account", account.Email)
	}
	account.Priority = want
}

// authFile is one entry of CLIProxy's auth file listing.
type authFile struct {
	Name     string          `json:"name"`
	Provider string          `json:"provider"`
	Email    string          `json:"email"`
	Account  string          `json:"account"`
	Disabled bool            `json:"disabled"`
	Priority json.RawMessage `json:"priority"`
	IDToken  struct {
		PlanType string `json:"plan_type"`
	} `json:"id_token"`
	Quota quota `json:"quota"`
}

func (f authFile) email() string {
	if f.Email != "" {
		return f.Email
	}
	return f.Account
}

// priority reads the listing's priority, a number or a numeric string.
func (f authFile) priority() int {
	var value json.Number
	if json.Unmarshal(bytes.Trim(f.Priority, `"`), &value) != nil {
		return 0
	}
	n, _ := value.Int64()
	return int(n)
}

func (w *Watcher) authFiles(ctx context.Context) ([]authFile, error) {
	var listing struct {
		Files []authFile `json:"files"`
	}
	if err := w.call(ctx, http.MethodGet, "/auth-files", nil, &listing); err != nil {
		return nil, err
	}
	return listing.Files, nil
}

// modelsOf lists the models CLIProxy serves through an account.
func (w *Watcher) modelsOf(ctx context.Context, name string) ([]string, error) {
	var listing struct {
		Models []struct {
			ID string `json:"id"`
		} `json:"models"`
	}
	if err := w.call(ctx, http.MethodGet, "/auth-files/models?name="+url.QueryEscape(name), nil, &listing); err != nil {
		return nil, err
	}
	var models []string
	for _, m := range listing.Models {
		models = append(models, m.ID)
	}
	return models, nil
}

// call sends a management API request and decodes a JSON response into out
// when out is not nil.
func (w *Watcher) call(ctx context.Context, method, path string, body []byte, out any) error {
	request, err := http.NewRequestWithContext(ctx, method, w.base+path, bytes.NewReader(body))
	if err != nil {
		return err
	}
	request.Header.Set("Authorization", "Bearer "+w.key)
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	response, err := w.client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, 4<<20))
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("management %s %s: status %d", method, strings.SplitN(path, "?", 2)[0], response.StatusCode)
	}
	if out == nil {
		return nil
	}
	return json.Unmarshal(data, out)
}
