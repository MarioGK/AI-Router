package accounts

import (
	"fmt"
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"
)

// families scope Claude windows named after a model family to it.
var families = []string{"opus", "sonnet", "haiku", "fable"}

// windowFamily is the model family a window is limited to, or "" for a
// window that covers every model.
func windowFamily(name string) string {
	for _, family := range families {
		if strings.HasSuffix(name, "_"+family) {
			return family
		}
	}
	return ""
}

// quota is what CLIProxy last saw of an account's quota: the rate limit
// headers of the latest upstream response the account served, by canonical
// header name, and when it saw them.
type quota struct {
	ObservedAt time.Time         `json:"observed_at"`
	Signals    map[string]string `json:"signals"`
}

// signal returns a quota header's value, matching its name without regard
// to case.
func (q quota) signal(name string) string {
	for key, value := range q.Signals {
		if strings.EqualFold(key, name) {
			return value
		}
	}
	return ""
}

// readQuota uses valid passive observations. A credit-only update retains
// unreset windows; a fresh quota without credits does not retain an old balance.
func readQuota(q quota, account *Account, last Account, now time.Time) bool {
	if q.ObservedAt.IsZero() || len(q.Signals) == 0 {
		return false
	}
	windowsKnown, refusalKnown := false, false
	switch account.Provider {
	case Claude:
		account.Credits = claudeCredits(q)
		windowsKnown = claudeWindows(q, account)
		status := strings.ToLower(strings.TrimSpace(q.signal("Anthropic-Ratelimit-Unified-Status")))
		refusalKnown = status == "allowed" || status == "allowed_warning" || status == "rejected"
		account.refused = status == "rejected"
		// A family-specific rejection does not refuse unrelated models.
		if account.refused && slices.ContainsFunc(account.Windows, func(w Window) bool { return w.Full() && windowFamily(w.Name) != "" }) &&
			!slices.ContainsFunc(account.Windows, func(w Window) bool { return w.Full() && windowFamily(w.Name) == "" }) {
			account.refused = false
			for i := range account.Windows {
				w := &account.Windows[i]
				w.refused = w.Full() && windowFamily(w.Name) != ""
			}
		}
	case Codex:
		account.Credits = codexCredits(q)
		windowsKnown = codexWindows(q, account)
		allowed, hasAllowed := signalBool(q.signal("X-Codex-Allowed"))
		reached, hasReached := signalBool(q.signal("X-Codex-Limit-Reached"))
		refusalKnown = hasAllowed || hasReached
		// Limit-Reached can describe spent included quota while Allowed
		// explicitly permits credit-backed requests. Without that permission
		// the reached flag remains a conservative refusal.
		account.refused = hasAllowed && !allowed || !hasAllowed && hasReached && reached
	}
	if !windowsKnown && !refusalKnown && account.Credits.Observed.IsZero() {
		return false
	}
	account.Observed = q.ObservedAt
	if !windowsKnown {
		account.holdWindows(last, now)
		if !refusalKnown && last.refused && (last.Until.IsZero() || last.Until.After(now)) {
			account.refused = true
		}
	}
	account.applyLimits(now)
	account.Windows = slices.DeleteFunc(account.Windows, func(w Window) bool { return !w.ResetsAt.IsZero() && !w.ResetsAt.After(now) })
	sortWindows(account.Windows)
	return true
}

// codexCredits preserves the amount as text while checking it is finite and
// nonnegative. Explicit no-credits or a zero balance veto finite availability;
// unlimited credits do not depend on a finite balance.
func codexCredits(q quota) Credits {
	var credits Credits
	hasRaw, unlimitedRaw := q.signal("X-Codex-Credits-Has-Credits"), q.signal("X-Codex-Credits-Unlimited")
	has, hasKnown := signalBool(hasRaw)
	unlimited, unlimitedKnown := signalBool(unlimitedRaw)
	balanceRaw := strings.TrimSpace(q.signal("X-Codex-Credits-Balance"))
	balance, balanceKnown := nonnegativeFloat(balanceRaw)
	balanceKnown = balanceKnown && len(balanceRaw) <= 512 && creditAmount.MatchString(balanceRaw)
	if balanceKnown {
		credits.Balance = balanceRaw
	}
	if hasRaw != "" || unlimitedRaw != "" || balanceRaw != "" {
		credits.Observed = q.ObservedAt
	}
	if hasRaw != "" && !hasKnown || unlimitedRaw != "" && !unlimitedKnown {
		return credits
	}
	if unlimitedKnown && unlimited {
		credits.Known, credits.Available, credits.Unlimited = true, true, true
		return credits
	}
	if hasKnown && !has || balanceKnown && balance == 0 {
		credits.Known = true
		return credits
	}
	if balanceRaw != "" && !balanceKnown {
		return credits
	}
	credits.Known = hasKnown || balanceKnown
	credits.Available = hasKnown && has || balanceKnown && balance > 0
	return credits
}

var creditAmount = regexp.MustCompile(`^(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$`)

// claudeCredits reads extra usage availability only. These passive headers do
// not report a monetary balance, and fallback availability is unrelated.
func claudeCredits(q quota) Credits {
	status := strings.ToLower(strings.TrimSpace(q.signal("Anthropic-Ratelimit-Unified-Overage-Status")))
	reason := strings.TrimSpace(q.signal("Anthropic-Ratelimit-Unified-Overage-Disabled-Reason"))
	credits := Credits{DisabledReason: reason}
	if status != "" || reason != "" {
		credits.Observed = q.ObservedAt
	}
	credits.Known = status == "allowed" || status == "rejected" || reason != ""
	credits.Available = status == "allowed" && reason == ""
	return credits
}

func signalBool(value string) (bool, bool) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "true", "1":
		return true, true
	case "false", "0":
		return false, true
	default:
		return false, false
	}
}

func nonnegativeFloat(value string) (float64, bool) {
	n, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
	return n, err == nil && !math.IsNaN(n) && !math.IsInf(n, 0) && n >= 0
}

// claudeWindow matches Anthropic's unified rate limit headers of one
// window: anthropic-ratelimit-unified-<window>-<field>, the window being a
// length (5h, 7d) with an optional model family (7d_opus).
var claudeWindow = regexp.MustCompile(`(?i)^anthropic-ratelimit-unified-([0-9]+[hd](?:_[a-z]+)?)-(utilization|reset|status)$`)

// claudeWindows reads the unified rate limit Anthropic reports on every
// response of a subscription account: per window, the fraction used, its
// reset in Unix seconds and whether it still allows requests.
func claudeWindows(q quota, account *Account) bool {
	windows := map[string]*Window{}
	known := map[string]bool{}
	for key, value := range q.Signals {
		match := claudeWindow.FindStringSubmatch(key)
		if match == nil {
			continue
		}
		name := claudeWindowName(strings.ToLower(match[1]))
		w := windows[name]
		if w == nil {
			w = &Window{Name: name}
			windows[name] = w
		}
		switch strings.ToLower(match[2]) {
		case "utilization":
			if used, valid := nonnegativeFloat(value); valid {
				w.Used = max(w.Used, used)
				known[name] = true
			}
		case "reset":
			w.ResetsAt = unixTime(value)
		case "status":
			if strings.EqualFold(strings.TrimSpace(value), "rejected") {
				w.Used = max(w.Used, 1)
				known[name] = true
			} else if strings.EqualFold(strings.TrimSpace(value), "allowed") || strings.EqualFold(strings.TrimSpace(value), "allowed_warning") {
				known[name] = true
			}
		}
	}
	for _, w := range windows {
		if !known[w.Name] {
			continue
		}
		account.Windows = append(account.Windows, *w)
	}
	return len(account.Windows) > 0
}

// claudeWindowName names a window as Anthropic's usage report does, which
// the dashboard and the family scoping read: five_hour, seven_day,
// seven_day_opus.
func claudeWindowName(window string) string {
	length, family, _ := strings.Cut(window, "_")
	name := length
	switch length {
	case "5h":
		name = "five_hour"
	case "7d":
		name = "seven_day"
	}
	if family != "" {
		name += "_" + family
	}
	return name
}

// codexWindows reads the rate limit ChatGPT reports on every Codex response:
// a primary and a secondary window, each with the percent used, its length
// in minutes (0 when the plan has no such window) and its reset, and the
// plan.
func codexWindows(q quota, account *Account) bool {
	if plan := q.signal("X-Codex-Plan-Type"); plan != "" {
		account.Plan = plan
	}
	known := false
	for _, prefix := range []string{"X-Codex-Primary-", "X-Codex-Secondary-"} {
		minutes, err := strconv.ParseInt(strings.TrimSpace(q.signal(prefix+"Window-Minutes")), 10, 64)
		if err == nil && minutes == 0 {
			known = known || prefix == "X-Codex-Primary-"
			continue
		}
		used, valid := nonnegativeFloat(q.signal(prefix + "Used-Percent"))
		if err != nil || minutes <= 0 || minutes > math.MaxInt64/60 || !valid {
			continue
		}
		known = true
		w := Window{Name: windowName(minutes * 60), Used: used / 100, ResetsAt: unixTime(q.signal(prefix + "Reset-At"))}
		if w.ResetsAt.IsZero() {
			if after, err := strconv.ParseInt(strings.TrimSpace(q.signal(prefix+"Reset-After-Seconds")), 10, 64); err == nil && after >= 0 && after <= math.MaxInt64/int64(time.Second) {
				w.ResetsAt = q.ObservedAt.Add(time.Duration(after) * time.Second)
			}
		}
		account.Windows = append(account.Windows, w)
	}
	return known
}

// applyLimits distinguishes spent included quota from explicit refusal. A
// refusal tied to a known reset expires with it; one without a reset is held
// until a new valid observation clears it.
func (a *Account) applyLimits(now time.Time) {
	a.Limited, a.Until = false, time.Time{}
	full, unknownReset := false, false
	var latestFull, soonest time.Time
	for _, w := range a.Windows {
		if !w.ResetsAt.IsZero() && (soonest.IsZero() || w.ResetsAt.Before(soonest)) {
			soonest = w.ResetsAt
		}
		if !w.Full() || windowFamily(w.Name) != "" {
			continue
		}
		full = true
		unknownReset = unknownReset || w.ResetsAt.IsZero()
		if w.ResetsAt.After(latestFull) {
			latestFull = w.ResetsAt
		}
		if (w.ResetsAt.IsZero() || w.ResetsAt.After(now)) && !a.Credits.Available {
			a.limit(w.ResetsAt, now)
		}
	}
	if !a.refused {
		return
	}
	reset := soonest
	if full {
		reset = latestFull
		if unknownReset {
			reset = time.Time{}
		}
	}
	if !reset.IsZero() && !reset.After(now) {
		a.refused = false
		return
	}
	a.limit(reset, now)
}

// unixTime reads a time in Unix seconds, or the zero time.
func unixTime(value string) time.Time {
	seconds, err := strconv.ParseInt(strings.TrimSpace(value), 10, 64)
	if err != nil || seconds <= 0 {
		return time.Time{}
	}
	return time.Unix(seconds, 0)
}

// limit marks the account limited until the window reset, keeping the
// latest reset when several windows are spent.
func (a *Account) limit(resetsAt, now time.Time) {
	a.Limited = true
	if resetsAt.After(now) && resetsAt.After(a.Until) {
		a.Until = resetsAt
	}
}

// windowName names a ChatGPT window by its length.
func windowName(seconds int64) string {
	switch {
	case seconds <= 0:
		return "window"
	case seconds%(7*24*3600) == 0:
		return "weekly"
	case seconds%(24*3600) == 0:
		return fmt.Sprintf("%dd", seconds/(24*3600))
	}
	return fmt.Sprintf("%dh", seconds/3600)
}

// sortWindows orders windows by name, so snapshots read the same every
// time.
func sortWindows(windows []Window) {
	slices.SortFunc(windows, func(a, b Window) int { return strings.Compare(a.Name, b.Name) })
}
