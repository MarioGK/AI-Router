package proxy

import (
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"
)

const claudeLimitStatus = "Anthropic-Ratelimit-Unified-Status"
const claudeLimitReset = "Anthropic-Ratelimit-Unified-Reset"

// ClaudeLimit supplies the Claude Code rate-limit signal for a CLIProxy
// cooldown response. CLIProxy can return a 429 with only Retry-After when
// every credential is cooling down; Claude Code needs the unified reset
// timestamp to emit rate_limit_event with resetsAt for T3's resume timer.
// Native Anthropic signals remain authoritative. Call only for Claude models.
// fallback is used when CLIProxy omitted Retry-After, such as its final 429
// after all subscription credentials entered cooldown.
func ClaudeLimit(response *http.Response, now time.Time, fallback ...time.Time) {
	if response.StatusCode != http.StatusTooManyRequests || response.Header.Get(claudeLimitStatus) != "" {
		return
	}
	hint := strings.TrimSpace(response.Header.Get("Retry-After"))
	reset, err := http.ParseTime(hint)
	if err != nil {
		seconds, err := strconv.ParseInt(hint, 10, 64)
		if err != nil || seconds <= 0 || seconds > math.MaxInt64/int64(time.Second) {
			reset = time.Time{}
		} else {
			reset = now.Add(time.Duration(seconds) * time.Second)
		}
	}
	if !reset.After(now) && len(fallback) > 0 {
		reset = fallback[0]
	}
	if !reset.After(now) {
		return
	}
	response.Header.Set(claudeLimitStatus, "rejected")
	if response.Header.Get(claudeLimitReset) == "" {
		response.Header.Set(claudeLimitReset, strconv.FormatInt(reset.Unix(), 10))
	}
	if response.Header.Get("Retry-After") == "" {
		seconds := int64(math.Ceil(reset.Sub(now).Seconds()))
		if seconds > 0 {
			response.Header.Set("Retry-After", strconv.FormatInt(seconds, 10))
		}
	}
}
