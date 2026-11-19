package dashboard

import (
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/MarioGK/AI-Router/internal/accounts"
	"github.com/MarioGK/AI-Router/internal/limit"
)

// Accounts is the watch over CLIProxy's subscription accounts;
// *accounts.Watcher is one.
type Accounts interface {
	Snapshot() accounts.Snapshot
	Available(model string) bool
}

// accountGroup is one provider's accounts and the candidates they leave
// out of new decisions.
type accountGroup struct {
	Title    string
	Accounts []accounts.Account
	Usable   int
	Withheld []string
}

// accountsView is the subscription accounts card.
type accountsView struct {
	Groups    []accountGroup
	CheckedAt time.Time
	Err       string
}

// providerTitles names the watched providers, in display order.
var providerTitles = []struct{ provider, title string }{
	{accounts.Codex, "ChatGPT"},
	{accounts.Claude, "Claude"},
}

// accountsOf groups the watched accounts by provider; nil when the watch is
// off.
func (d *Dashboard) accountsOf() *accountsView {
	if d.accounts == nil {
		return nil
	}
	snapshot := d.accounts.Snapshot()
	view := &accountsView{CheckedAt: snapshot.CheckedAt, Err: snapshot.Err}
	for _, p := range providerTitles {
		group := accountGroup{Title: p.title}
		served := map[string]bool{}
		for _, a := range snapshot.Accounts {
			if a.Provider != p.provider {
				continue
			}
			group.Accounts = append(group.Accounts, a)
			if !a.Disabled && !a.Limited {
				group.Usable++
			}
			for _, model := range a.Models {
				served[model] = true
			}
		}
		if len(group.Accounts) == 0 {
			continue
		}
		for _, candidate := range d.auto.Candidates {
			if served[candidate.ID] && !d.accounts.Available(candidate.ID) {
				group.Withheld = append(group.Withheld, candidate.ID)
			}
		}
		view.Groups = append(view.Groups, group)
	}
	return view
}

// quotaWindows are the windows of an account worth showing: the ones that
// decide whether it may be used, in a stable order.
func quotaWindows(a accounts.Account) []accounts.Window {
	var shown []accounts.Window
	for _, w := range a.Windows {
		if windowLabel(w.Name) != "" {
			shown = append(shown, w)
		}
	}
	slices.SortStableFunc(shown, func(x, y accounts.Window) int { return windowRank(x.Name) - windowRank(y.Name) })
	return shown
}

// windowLabel names a quota window, or "" for one that does not limit
// routing.
func windowLabel(name string) string {
	switch name {
	case "five_hour", "5h":
		return "5-hour"
	case "seven_day", "weekly":
		return "Weekly"
	}
	for _, family := range []string{"opus", "sonnet", "haiku", "fable"} {
		if strings.HasSuffix(name, "_"+family) {
			return "Weekly " + strings.ToUpper(family[:1]) + family[1:]
		}
	}
	if strings.HasSuffix(name, "h") || strings.HasSuffix(name, "d") {
		return name
	}
	return ""
}

func windowRank(name string) int {
	switch name {
	case "five_hour", "5h":
		return 0
	case "seven_day", "weekly":
		return 1
	}
	return 2
}

// windowTone colors a window's bar by how much of it is spent.
func windowTone(w accounts.Window) string {
	switch {
	case w.Full():
		return "bg-critical"
	case w.Used >= 0.8:
		return "bg-serious"
	case w.Used >= 0.5:
		return "bg-warning"
	}
	return "bg-good"
}

// until says when a time comes, relative to now: "in 3d", "in 4h".
func until(t time.Time) string {
	left := time.Until(t)
	switch {
	case t.IsZero():
		return ""
	case left <= time.Minute:
		return "now"
	case left < time.Hour:
		return "in " + strconv.Itoa(int(left.Round(time.Minute).Minutes())) + "m"
	case left < 48*time.Hour:
		return "in " + strconv.Itoa(int(left.Round(time.Hour).Hours())) + "h"
	}
	return "in " + strconv.Itoa(int(left.Round(24*time.Hour).Hours()/24)) + "d"
}

// itoaSigned writes an integer with its sign.
func itoaSigned(n int) string {
	if n > 0 {
		return "+" + strconv.Itoa(n)
	}
	return strconv.Itoa(n)
}

// Limits are the upstream concurrency gates; *limit.Limiter is one.
type Limits interface {
	Stats() []limit.Stats
}

// slotTone colors one slot of a gate: running, open, or closed by a
// throttle.
func slotTone(g limit.Stats, slot int) string {
	switch {
	case slot < g.InFlight:
		return swatch(g.Model)
	case slot < g.Limit:
		return "bg-raised"
	}
	return "bg-critical/30"
}
