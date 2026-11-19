package dashboard

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/accounts"
)

func TestAccountCreditMetadataIsExplicit(t *testing.T) {
	for _, test := range []struct {
		name     string
		provider string
		credits  accounts.Credits
		want     []string
		absent   []string
	}{
		{"unknown", accounts.Codex, accounts.Credits{}, []string{"Credits:", "Not reported"}, []string{"0 credits remaining", "$"}},
		{"balance", accounts.Codex, accounts.Credits{Known: true, Available: true, Balance: "12.75"}, []string{"12.75 credits remaining"}, []string{"$"}},
		{"observed", accounts.Codex, accounts.Credits{Known: true, Available: true, Observed: time.Date(2026, time.October, 1, 12, 0, 0, 0, time.UTC)}, []string{"As of", "<time"}, []string{"@when"}},
		{"zero", accounts.Codex, accounts.Credits{Known: true, Balance: "0"}, []string{"0 credits remaining", "Unavailable for requests"}, []string{"Not reported", "$"}},
		{"unlimited", accounts.Codex, accounts.Credits{Known: true, Available: true, Unlimited: true}, []string{"Unlimited"}, []string{"0 credits remaining", "$"}},
		{"extra usage", accounts.Claude, accounts.Credits{Known: true, Available: true}, []string{"Extra usage:", "Available; balance not reported"}, []string{"credits remaining", "$"}},
		{"restricted extra usage", accounts.Claude, accounts.Credits{Known: true, DisabledReason: "organization_restricted"}, []string{"Extra usage:", "Unavailable; balance not reported", "organization_restricted"}, []string{"credits remaining", "$"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			var rendered bytes.Buffer
			if err := accountCredits(accounts.Account{Provider: test.provider, Credits: test.credits}).Render(t.Context(), &rendered); err != nil {
				t.Fatal(err)
			}
			body := rendered.String()
			for _, want := range test.want {
				if !strings.Contains(body, want) {
					t.Errorf("credit metadata lacks %q: %s", want, body)
				}
			}
			for _, absent := range test.absent {
				if strings.Contains(body, absent) {
					t.Errorf("credit metadata invents %q: %s", absent, body)
				}
			}
		})
	}
}

func TestCreditBackedAccountRemainsDistinctFromBlockedAccount(t *testing.T) {
	account := accounts.Account{Provider: accounts.Codex, Priority: -5,
		Credits: accounts.Credits{Known: true, Available: true, Balance: "12"},
		Windows: []accounts.Window{{Name: "weekly", Used: 1, ResetsAt: time.Now().Add(time.Hour)}}}
	for _, limited := range []bool{false, true} {
		account.Limited = limited
		var rendered bytes.Buffer
		if err := accountState(account).Render(t.Context(), &rendered); err != nil {
			t.Fatal(err)
		}
		body := rendered.String()
		if limited {
			if !strings.Contains(body, "at limit") || strings.Contains(body, "credit-backed") {
				t.Fatalf("blocked account state = %s", body)
			}
		} else if !strings.Contains(body, "credit-backed") || !strings.Contains(body, "CLIProxy priority -5") {
			t.Fatalf("credit-backed account state = %s", body)
		}
	}
}
