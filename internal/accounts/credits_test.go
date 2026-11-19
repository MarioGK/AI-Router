package accounts

import (
	"maps"
	"strconv"
	"testing"
	"time"
)

func TestCodexCreditsAndIncludedQuota(t *testing.T) {
	now := time.Now()
	for _, tc := range []struct {
		name      string
		credits   map[string]string
		known     bool
		available bool
		unlimited bool
		balance   string
	}{
		{name: "unknown"},
		{name: "positive", credits: map[string]string{"Has-Credits": "True", "Unlimited": "False", "Balance": "123.456"}, known: true, available: true, balance: "123.456"},
		{name: "availability without balance", credits: map[string]string{"Has-Credits": "true"}, known: true, available: true},
		{name: "balance alone", credits: map[string]string{"Balance": "2.5"}, known: true, available: true, balance: "2.5"},
		{name: "unlimited", credits: map[string]string{"Has-Credits": "False", "Unlimited": "True", "Balance": "0"}, known: true, available: true, unlimited: true, balance: "0"},
		{name: "numeric booleans", credits: map[string]string{"Has-Credits": "1", "Unlimited": "0", "Balance": " 42.0 "}, known: true, available: true, balance: "42.0"},
		{name: "zero", credits: map[string]string{"Has-Credits": "true", "Balance": "0.0"}, known: true, balance: "0.0"},
		{name: "explicit no credits", credits: map[string]string{"Has-Credits": "false", "Balance": "10"}, known: true, balance: "10"},
		{name: "unlimited false only", credits: map[string]string{"Unlimited": "false"}},
		{name: "malformed boolean", credits: map[string]string{"Has-Credits": "yes", "Balance": "10"}, balance: "10"},
		{name: "malformed unlimited", credits: map[string]string{"Has-Credits": "true", "Unlimited": "yes", "Balance": "10"}, balance: "10"},
		{name: "nan", credits: map[string]string{"Has-Credits": "true", "Balance": "NaN"}},
		{name: "infinity", credits: map[string]string{"Has-Credits": "true", "Balance": "+Inf"}},
		{name: "negative", credits: map[string]string{"Has-Credits": "true", "Balance": "-1"}},
		{name: "overflow", credits: map[string]string{"Has-Credits": "true", "Balance": "1e999"}},
		{name: "nondecimal", credits: map[string]string{"Balance": "0x1p3"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, used := range []int{25, 100} {
				signals := codex(used, time.Hour)
				delete(signals, "X-Codex-Credits-Has-Credits")
				for name, value := range tc.credits {
					signals["x-codex-credits-"+name] = value
				}
				a := Account{Provider: Codex}
				if !readQuota(quota{ObservedAt: now, Signals: signals}, &a, Account{}, now) {
					t.Fatal("valid included quota was not observed")
				}
				if c := a.Credits; c.Known != tc.known || c.Available != tc.available || c.Unlimited != tc.unlimited || c.Balance != tc.balance {
					t.Fatalf("credits = %+v", c)
				}
				blocked := used == 100 && !tc.available
				if a.Limited != blocked || a.Blocked("gpt-6.1-sol") != blocked || a.UsesCredits() != (used == 100 && tc.available) {
					t.Fatalf("used %d: account = %+v", used, a)
				}
			}
		})
	}
}

func TestCreditBackedAccountStillHonorsProviderRefusal(t *testing.T) {
	now := time.Now()
	for _, refusal := range []map[string]string{
		{"X-Codex-Allowed": "false"},
		{"X-Codex-Limit-Reached": "1"},
	} {
		signals := codex(100, time.Hour)
		signals["X-Codex-Credits-Unlimited"] = "true"
		maps.Copy(signals, refusal)
		a := Account{Provider: Codex}
		readQuota(quota{ObservedAt: now, Signals: signals}, &a, Account{}, now)
		if !a.Limited || !a.Blocked("gpt-6.1-sol") || a.UsesCredits() {
			t.Fatalf("provider refusal was bypassed: %+v", a)
		}
		var reset Account
		reset.hold(a, a.Until.Add(time.Second))
		if reset.Limited || len(reset.Windows) != 0 {
			t.Fatalf("reset refusal was retained: %+v", reset)
		}
		var reread Account
		reread.Provider = Codex
		readQuota(quota{ObservedAt: now, Signals: signals}, &reread, a, a.Until.Add(time.Second))
		if reread.Limited {
			t.Fatalf("expired observed refusal was reapplied: %+v", reread)
		}
	}
}

func TestReachedIncludedQuotaWithExplicitAllowedCredits(t *testing.T) {
	now := time.Now()
	for _, tc := range []struct {
		name    string
		credits map[string]string
		blocked bool
	}{
		{name: "positive", credits: map[string]string{"Has-Credits": "true", "Balance": "10"}},
		{name: "unlimited", credits: map[string]string{"Unlimited": "true"}},
		{name: "zero", credits: map[string]string{"Has-Credits": "true", "Balance": "0"}, blocked: true},
		{name: "no credits", credits: map[string]string{"Has-Credits": "false"}, blocked: true},
		{name: "unknown", blocked: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			signals := codex(100, time.Hour)
			delete(signals, "X-Codex-Credits-Has-Credits")
			signals["X-Codex-Allowed"], signals["X-Codex-Limit-Reached"] = "true", "true"
			for name, value := range tc.credits {
				signals["X-Codex-Credits-"+name] = value
			}
			a := Account{Provider: Codex}
			readQuota(quota{ObservedAt: now, Signals: signals}, &a, Account{}, now)
			if a.Limited != tc.blocked || a.Blocked("gpt-6.1-sol") != tc.blocked || a.UsesCredits() == tc.blocked {
				t.Fatalf("allowed credit-backed request = %+v", a)
			}
		})
	}
}

func TestCreditPrioritiesPreferIncludedQuota(t *testing.T) {
	for _, limitedPriority := range []int{-10, -2, -1} {
		t.Run(strconv.Itoa(limitedPriority), func(t *testing.T) {
			m := newManagement(t)
			m.files = []map[string]any{
				{"name": "included", "provider": Codex},
				{"name": "credits", "provider": Codex},
				{"name": "limited", "provider": Codex},
			}
			for _, name := range []string{"included", "credits", "limited"} {
				m.models[name] = []string{"gpt-6.1-sol"}
			}
			m.observe("included", codex(25, time.Hour))
			credits := codex(100, time.Hour)
			credits["X-Codex-Credits-Has-Credits"] = "true"
			credits["X-Codex-Credits-Balance"] = "10"
			m.observe("credits", credits)
			m.observe("limited", codex(100, time.Hour))
			w := watcher(m)
			w.limitedPriority = limitedPriority
			if err := w.Refresh(t.Context()); err != nil {
				t.Fatal(err)
			}
			accounts := byName(w)
			if accounts["included"].Priority <= accounts["credits"].Priority || accounts["credits"].Priority <= accounts["limited"].Priority || !w.Available("gpt-6.1-sol") {
				t.Fatalf("priority order or availability = %+v", accounts)
			}
			// Every included window is now full, yet the credit-backed account
			// keeps the model available. Exhausting its credits withholds it.
			m.observe("included", codex(100, time.Hour))
			if err := w.Refresh(t.Context()); err != nil || !w.Available("gpt-6.1-sol") {
				t.Fatalf("credit-backed model unavailable: %v", err)
			}
			m.observe("credits", codex(100, time.Hour))
			if err := w.Refresh(t.Context()); err != nil || w.Available("gpt-6.1-sol") {
				t.Fatalf("exhausted model available: %v", err)
			}
		})
	}
}

func TestClaudeExtraUsageAndFamilyWindows(t *testing.T) {
	now := time.Now()
	for _, tc := range []struct {
		status, reason string
		available      bool
	}{
		{status: "allowed", available: true},
		{status: "allowed", reason: "org_level_disabled"},
		{status: "rejected", reason: "org_level_disabled"},
		{status: "unknown"},
		{},
	} {
		t.Run(tc.status+tc.reason, func(t *testing.T) {
			signals := claude(0.2, 0.5, time.Hour, 24*time.Hour)
			signals["Anthropic-Ratelimit-Unified-Overage-Status"] = tc.status
			signals["Anthropic-Ratelimit-Unified-Overage-Disabled-Reason"] = tc.reason
			signals["Anthropic-Ratelimit-Unified-Fallback"] = "available"
			signals["Anthropic-Ratelimit-Unified-7d_opus-Utilization"] = "1"
			signals["Anthropic-Ratelimit-Unified-7d_opus-Reset"] = in(time.Hour)
			a := Account{Provider: Claude}
			readQuota(quota{ObservedAt: now, Signals: signals}, &a, Account{}, now)
			if a.Limited || a.Blocked("claude-sonnet-5-5") || a.Blocked("claude-opus-5-5") == tc.available || a.Credits.Available != tc.available || a.Credits.Balance != "" {
				t.Fatalf("family quota and extra usage = %+v", a)
			}
			// An allowed overall response and available overage keep a spent
			// included window usable; overall refusal still blocks it.
			signals["Anthropic-Ratelimit-Unified-7d-Utilization"] = "1"
			read := Account{Provider: Claude}
			readQuota(quota{ObservedAt: now, Signals: signals}, &read, Account{}, now)
			if read.Limited == tc.available || read.Blocked("claude-sonnet-5-5") == tc.available {
				t.Fatalf("global quota = %+v", read)
			}
			signals["Anthropic-Ratelimit-Unified-Status"] = "rejected"
			read = Account{Provider: Claude}
			readQuota(quota{ObservedAt: now, Signals: signals}, &read, Account{}, now)
			if !read.Limited || !read.Blocked("claude-sonnet-5-5") {
				t.Fatalf("Claude refusal was bypassed: %+v", read)
			}
		})
	}
}

func TestLostAndPartialQuotaPreserveCreditIntersections(t *testing.T) {
	now := time.Now()
	signals := codex(100, time.Hour)
	signals["X-Codex-Credits-Has-Credits"] = "true"
	signals["X-Codex-Credits-Balance"] = "10.25"
	a := Account{Provider: Codex}
	readQuota(quota{ObservedAt: now, Signals: signals}, &a, Account{}, now)
	var held Account
	held.hold(a, now.Add(time.Minute))
	if held.Limited || !held.UsesCredits() || held.Credits != a.Credits || !held.Observed.Equal(now) {
		t.Fatalf("lost quota did not retain last credits: %+v", held)
	}
	var reset Account
	reset.hold(a, now.Add(2*time.Hour))
	if reset.Limited || reset.UsesCredits() || len(reset.Windows) != 0 || reset.Credits != a.Credits {
		t.Fatalf("included window reset changed the credit balance: %+v", reset)
	}
	// A fresh full-window observation without credits cannot borrow the old
	// positive balance. A credit-only update must retain the unreset window.
	fresh := Account{Provider: Codex}
	missing := codex(100, time.Hour)
	delete(missing, "X-Codex-Credits-Has-Credits")
	readQuota(quota{ObservedAt: now.Add(time.Minute), Signals: missing}, &fresh, a, now)
	if !fresh.Limited || fresh.Credits.Known {
		t.Fatalf("missing credits reused an old balance: %+v", fresh)
	}
	fresh = Account{Provider: Codex}
	readQuota(quota{ObservedAt: now.Add(time.Minute), Signals: map[string]string{"X-Codex-Credits-Balance": "0"}}, &fresh, a, now)
	if !fresh.Limited || len(fresh.Windows) != 1 || fresh.Credits.Balance != "0" {
		t.Fatalf("credit-only exhaustion lost included quota: %+v", fresh)
	}
	fresh = Account{Provider: Codex}
	readQuota(quota{ObservedAt: now.Add(time.Minute), Signals: map[string]string{"X-Codex-Credits-Balance": "NaN"}}, &fresh, a, now)
	if !fresh.Limited || fresh.Credits.Known || fresh.Credits.Available || fresh.Credits.Balance != "" {
		t.Fatalf("invalid fresh balance reused an old positive balance: %+v", fresh)
	}
}

func TestClaudeFamilyRefusalVetoesCreditsUntilReset(t *testing.T) {
	now := time.Now()
	signals := claude(0.2, 0.5, time.Hour, 24*time.Hour)
	signals["Anthropic-Ratelimit-Unified-Overage-Status"] = "allowed"
	delete(signals, "Anthropic-Ratelimit-Unified-Overage-Disabled-Reason")
	signals["Anthropic-Ratelimit-Unified-7d_opus-Utilization"] = "1"
	signals["Anthropic-Ratelimit-Unified-7d_opus-Status"] = "rejected"
	signals["Anthropic-Ratelimit-Unified-7d_opus-Reset"] = in(30 * time.Minute)
	signals["Anthropic-Ratelimit-Unified-Status"] = "rejected"
	a := Account{Provider: Claude}
	readQuota(quota{ObservedAt: now, Signals: signals}, &a, Account{}, now)
	if a.Limited || !a.Blocked("claude-opus-5-5") || a.Blocked("claude-sonnet-5-5") || a.UsesCredits() {
		t.Fatalf("scoped explicit refusal was bypassed: %+v", a)
	}
	var held Account
	held.hold(a, now.Add(time.Minute))
	if !held.Blocked("claude-opus-5-5") || held.Blocked("claude-sonnet-5-5") {
		t.Fatalf("lost quota changed scoped refusal: %+v", held)
	}
	var reset Account
	reset.hold(a, now.Add(time.Hour))
	if reset.Blocked("claude-opus-5-5") {
		t.Fatalf("family refusal persisted after reset: %+v", reset)
	}
	signals["Anthropic-Ratelimit-Unified-Status"] = "allowed"
	cleared := Account{Provider: Claude}
	readQuota(quota{ObservedAt: now.Add(time.Minute), Signals: signals}, &cleared, a, now)
	if cleared.Blocked("claude-opus-5-5") || !cleared.UsesCredits() {
		t.Fatalf("fresh permission did not clear the scoped refusal: %+v", cleared)
	}
}

func TestRefusalWithoutResetPersistsUntilNewEvidence(t *testing.T) {
	now := time.Now()
	last := Account{Provider: Codex}
	readQuota(quota{ObservedAt: now, Signals: map[string]string{"X-Codex-Allowed": "false"}}, &last, Account{}, now)
	var held Account
	held.hold(last, now.Add(24*time.Hour))
	if !held.Limited || !held.Until.IsZero() {
		t.Fatalf("refusal without reset was lost: %+v", held)
	}
	partial := Account{Provider: Codex}
	readQuota(quota{ObservedAt: now.Add(time.Minute), Signals: map[string]string{"X-Codex-Credits-Balance": "10"}}, &partial, last, now)
	if !partial.Limited || partial.UsesCredits() {
		t.Fatalf("credits alone cleared a provider refusal: %+v", partial)
	}
	read := Account{Provider: Codex}
	readQuota(quota{ObservedAt: now.Add(time.Minute), Signals: map[string]string{"X-Codex-Allowed": "true", "X-Codex-Credits-Balance": "10"}}, &read, last, now)
	if read.Limited || read.Blocked("gpt-6.1-sol") {
		t.Fatalf("new allowed evidence did not clear refusal: %+v", read)
	}
}

func TestMalformedQuotaDoesNotRestoreSpentAccounts(t *testing.T) {
	now := time.Now()
	last := Account{Provider: Codex}
	readQuota(quota{ObservedAt: now, Signals: codex(100, time.Hour)}, &last, Account{}, now)
	for _, value := range []string{"NaN", "+Inf", "-1", "1e999", "invalid"} {
		signals := codex(100, time.Hour)
		signals["X-Codex-Primary-Used-Percent"] = value
		delete(signals, "X-Codex-Credits-Has-Credits")
		a := Account{Provider: Codex}
		if readQuota(quota{ObservedAt: now.Add(time.Minute), Signals: signals}, &a, last, now) {
			t.Fatalf("malformed quota %q was accepted: %+v", value, a)
		}
		a.hold(last, now)
		if !a.Limited || !a.Blocked("gpt-6.1-sol") {
			t.Fatalf("malformed quota %q restored a spent account: %+v", value, a)
		}
	}
}
