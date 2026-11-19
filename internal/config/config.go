// Package config loads, defaults and validates the ai-router configuration.
package config

import (
	"bytes"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net/url"
	"os"
	"regexp"
	"slices"
	"strings"
	"time"

	"go.yaml.in/yaml/v3"

	"github.com/MarioGK/AI-Router/internal/accounts"
	"github.com/MarioGK/AI-Router/internal/auth"
	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/usage"
)

// effort is a CLIProxy reasoning suffix: a level name or a token budget.
var effort = regexp.MustCompile(`^([a-z]+|[0-9]+)?$`)

var secretEnvName = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// ValidEffort reports whether text is a CLIProxy effort: a level name or a
// token budget.
func ValidEffort(text string) bool {
	return text != "" && effort.MatchString(text)
}

// Suffixed is the model value CLIProxy runs at effort: the `model(effort)`
// suffix, or the bare model without an effort.
func Suffixed(model, effort string) string {
	if effort == "" {
		return model
	}
	return model + "(" + effort + ")"
}

// maxLevels is the most reasoning levels a systemone score question takes.
const maxLevels = 10

// Config is the whole router configuration file.
type Config struct {
	Listen          string        `yaml:"listen"`
	Database        string        `yaml:"database"`
	SecretsFiles    []string      `yaml:"secrets_files"`
	LogLevel        string        `yaml:"log_level"`
	MaxBodyBytes    int64         `yaml:"max_body_bytes"`
	CatalogInterval time.Duration `yaml:"catalog_interval"`
	LogRetention    time.Duration `yaml:"log_retention"`
	// PromptRetention is how long decisions keep their prompt and the
	// deciders' raw replies; zero keeps them as long as the decisions.
	PromptRetention time.Duration `yaml:"prompt_retention"`
	// ShutdownGrace is how long a stopping router waits for the responses
	// still streaming.
	ShutdownGrace time.Duration `yaml:"shutdown_grace"`
	Upstream      Upstream      `yaml:"upstream"`
	Accounts      Accounts      `yaml:"accounts"`
	Dashboard     Dashboard     `yaml:"dashboard"`
	// Limits caps concurrent requests per model.
	Limits   map[string]Limit `yaml:"limits"`
	Auto     Auto             `yaml:"auto"`
	Deciders []Decider        `yaml:"deciders"`
	// Prices are list prices in US dollars per million tokens, used to
	// estimate what requests and threads cost; subscription upstreams spend
	// quota roughly in proportion.
	Prices map[string]usage.Price `yaml:"prices"`

	secrets map[string]string
}

// Upstream is the CLIProxy endpoint every request is forwarded to. Its key
// authenticates clients and the router's own catalog and judge calls.
type Upstream struct {
	BaseURL   string `yaml:"base_url"`
	APIKeyEnv string `yaml:"api_key_env"`
	// IdleTimeout ends a request whose upstream stays silent that long:
	// before its response headers or between two chunks of its body.
	IdleTimeout time.Duration `yaml:"idle_timeout"`
}

// Dashboard configures the separate operator username/password sign-in.
type Dashboard struct {
	Username string `yaml:"username"`
	// PasswordHashEnv names a bcrypt hash from a secrets file or the process
	// environment. An absent hash disables login without stopping routing.
	PasswordHashEnv string `yaml:"password_hash_env"`
}

// Accounts configures the watch over CLIProxy's OAuth subscription
// accounts: their quota is read through the management API, accounts at
// their limit move behind the others, and a model no account can serve is
// left out of new decisions.
type Accounts struct {
	// ManagementKeyEnv names the secret holding CLIProxy's management key;
	// empty turns the watch off.
	ManagementKeyEnv string        `yaml:"management_key_env"`
	Interval         time.Duration `yaml:"interval"`
	Providers        []string      `yaml:"providers"`
	// LimitedPriority is the CLIProxy priority of an account at its limit;
	// included quota takes precedence over available provider credits.
	LimitedPriority int `yaml:"limited_priority"`
}

// Limit caps one model's concurrent requests. Requests over the cap wait
// for a slot in arrival order, at most QueueTimeout. A 429 carrying one of
// ThrottleCodes shrinks the cap to the requests still running and is
// retried from the queue, at most Retries times, instead of failing over;
// the cap grows back by one every Recovery. While the cap is reached, new
// threads decided for the model go to Spill, when it can serve them.
type Limit struct {
	Concurrency   int           `yaml:"concurrency"`
	ThrottleCodes []string      `yaml:"throttle_codes"`
	Recovery      time.Duration `yaml:"recovery"`
	QueueTimeout  time.Duration `yaml:"queue_timeout"`
	// Retries defaults to 4 when unset; 0 turns retries off.
	Retries *int   `yaml:"retries"`
	Spill   string `yaml:"spill"`
}

// Auto configures the virtual model and how threads using it are routed.
type Auto struct {
	ModelID             string  `yaml:"model_id"`
	DefaultModel        string  `yaml:"default_model"`
	AuxModel            string  `yaml:"aux_model"`
	EscalationThreshold float64 `yaml:"escalation_threshold"`
	Doubt               string  `yaml:"doubt"`
	FailoverStatuses    []int   `yaml:"failover_statuses"`
	// A server error (5xx, 529) among FailoverStatuses fails a thread over
	// only once its model has answered FailoverTrip of them in a row; until
	// then the error reaches the client, which retries on the same model
	// and account and keeps its prompt cache. Any other status fails over
	// at once. 3 when unset.
	FailoverTrip  *int     `yaml:"failover_trip"`
	StripTags     []string `yaml:"strip_tags"`
	StateMaxChars int      `yaml:"state_max_chars"`
	// RecordPrompts keeps each decision's prompt, as the user wrote it and
	// as the deciders judged it, to review decisions; true when unset.
	RecordPrompts *bool         `yaml:"record_prompts"`
	DecideTimeout time.Duration `yaml:"decide_timeout"`
	// A decider that fails DeciderTrip turns in a row sits out DeciderRest,
	// so an outage does not delay every new thread by its timeout; 3 when
	// unset, 0 never rests one.
	DeciderTrip *int              `yaml:"decider_trip"`
	DeciderRest time.Duration     `yaml:"decider_rest"`
	Redecide    Redecide          `yaml:"redecide"`
	Reasoning   Reasoning         `yaml:"reasoning"`
	Cost        Cost              `yaml:"cost"`
	Outcomes    Outcomes          `yaml:"outcomes"`
	Learn       Learn             `yaml:"learn"`
	Candidates  []Candidate       `yaml:"candidates"`
	Fallback    map[string]string `yaml:"fallback"`
	// Retired are models taken out of service: a thread pinned to one moves
	// at its next request to the model's fallback, else the default.
	Retired []string `yaml:"retired"`
}

// Redecide makes a thread's decision run again, on a new prompt from the
// user, where its prompt cache is cold anyway: after Idle without a request
// (5m when unset, 0 for never) or after a compaction. A thread pinned below
// the top of the ladder that made GrowRequests requests since its last
// decision (60 when unset, 0 for never) is decided again at its next cold
// point even when Idle is 0, where a compaction is its only cold point.
type Redecide struct {
	Idle         *time.Duration `yaml:"idle"`
	GrowRequests *int           `yaml:"grow_requests"`
}

// IdleAfter is the idle time after which a new prompt decides a thread
// again; 0 when it never does.
func (r Redecide) IdleAfter() time.Duration {
	if r.Idle == nil {
		return 0
	}
	return *r.Idle
}

// GrowAfter is how many requests make a thread pinned below the top of the
// ladder be decided again at its next cold point; 0 when none do.
func (r Redecide) GrowAfter() int {
	if r.GrowRequests == nil {
		return 0
	}
	return *r.GrowRequests
}

// Reasoning configures the reasoning chooser: the deciders also judge how
// much reasoning a new thread needs, on Levels ordered lightest first, and
// the thread runs at that effort on every candidate that accepts it. No
// levels turn the chooser off.
type Reasoning struct {
	Levels []Level `yaml:"levels"`
}

// Level is one reasoning effort the chooser can pick. Its id is the effort
// CLIProxy applies (the `model(effort)` suffix); its description and
// examples are the criteria the deciders weigh.
type Level struct {
	ID          string   `yaml:"id"`
	Description string   `yaml:"description"`
	Examples    []string `yaml:"examples"`
}

// Cost makes the router choose the ladder step with the least expected
// cost to get a task done, instead of escalating by the threshold: a
// candidate's cost per task is learned from the request log, per reasoning
// level, and priced at its list price times its cost_factor; a task sent
// to a model too weak for it costs its redo on the right model plus Miss
// times that on top.
type Cost struct {
	Enabled bool `yaml:"enabled"`
	// Miss is 3 when unset.
	Miss *float64 `yaml:"miss"`
	// Window is how far back threads teach the costs; Weight is how many
	// threads the estimate a cell starts from counts as.
	Window time.Duration `yaml:"window"`
	Weight *float64      `yaml:"weight"`
	// TimeValue is what an hour of a task's wall-clock time is worth, in
	// US dollars, so a slower model pays for the time its tasks take; 0,
	// the default, leaves time out.
	TimeValue float64 `yaml:"time_value"`
}

// Outcomes are the request counts at which a settled thread's size says
// its decision was wrong: a thread on the lighter half of the ladder that
// ran more than Long requests, and one on the heavier half that stayed
// within Tiny. Long is 200 and Tiny 3 when unset; 0 turns either off.
type Outcomes struct {
	Long *int `yaml:"long"`
	Tiny *int `yaml:"tiny"`
}

// Learn is how the router learns what its own decisions alone cannot show.
// It only ever sees how the step it chose fared, never whether a cheaper
// one would have done, so with Explore set (a share from 0 to 0.2, 0 the
// default turning it off) it routes that share of borderline automatic
// decisions one ladder step cheaper: those where the economy scored the
// step below the one it chose within ExploreMargin of it, relative to
// its score (0.25 when unset).
//
// With Enabled (true when unset) settled threads label where their tasks
// belonged, and a new thread's decision blends in what the Neighbours (20
// when unset) most similar labelled first prompts suggest, by a weight the
// history earns on its own labels.
type Learn struct {
	Explore       float64  `yaml:"explore"`
	ExploreMargin *float64 `yaml:"explore_margin"`
	Enabled       *bool    `yaml:"enabled"`
	Neighbours    *int     `yaml:"neighbours"`
}

// Margin is how close, relative to the chosen step's score, the step below
// must score for a decision to be borderline; 0.25 when unset.
func (l Learn) Margin() float64 {
	if l.ExploreMargin == nil {
		return 0.25
	}
	return *l.ExploreMargin
}

// Criteria are the reasoning levels as the deciders see them; nil when the
// chooser is off.
func (r Reasoning) Criteria() []decide.Level {
	if len(r.Levels) == 0 {
		return nil
	}
	levels := make([]decide.Level, len(r.Levels))
	for i, level := range r.Levels {
		levels[i] = decide.Level{ID: level.ID, Description: level.Description, Examples: level.Examples}
	}
	return levels
}

// IDs are the level ids, lightest first.
func (r Reasoning) IDs() []string {
	ids := make([]string, len(r.Levels))
	for i, level := range r.Levels {
		ids[i] = level.ID
	}
	return ids
}

// Rank is the position of a level, lightest first, or -1.
func (r Reasoning) Rank(id string) int {
	return slices.IndexFunc(r.Levels, func(l Level) bool { return l.ID == id })
}

// Candidate is one model `auto` can pin a thread to. Candidates form an
// escalation ladder in list order, from the lightest to the most capable;
// a specialist (e.g. the frontend model) is chosen only for its own kind of
// work and is never an escalation step.
type Candidate struct {
	ID      string `yaml:"id"`
	Context int    `yaml:"context"`
	Output  int    `yaml:"output"`
	Vision  bool   `yaml:"vision"`
	// ExpectedLatency is an operator estimate from comparable requests, used
	// only as guidance for speed priority. Zero means unknown.
	ExpectedLatency time.Duration `yaml:"expected_latency"`
	// Reasoning is the effort (e.g. high, xhigh) CLIProxy applies when the
	// router sends this candidate and the chooser picked no level for the
	// thread; empty leaves the client's own setting.
	Reasoning string `yaml:"reasoning"`
	// Efforts are the chooser's levels this candidate accepts: a thread's
	// level runs at the nearest of them. Empty keeps Reasoning fixed.
	Efforts    []string `yaml:"efforts"`
	Specialist bool     `yaml:"specialist"`
	// CostFactor is what a dollar of the candidate's list price costs: 1
	// when unset, as for pay-per-use and quota subscriptions, less for a
	// flat-rate plan.
	CostFactor *float64 `yaml:"cost_factor"`
	// Description is what the candidate is for, NotFor what it is not
	// for, and Examples are requests it should get; the deciders weigh
	// them.
	Description string   `yaml:"description"`
	NotFor      string   `yaml:"not_for"`
	Examples    []string `yaml:"examples"`
}

// Criterion is the candidate as the deciders see it.
func (c Candidate) Criterion() decide.Candidate {
	return decide.Candidate{ID: c.ID, Description: c.Description, NotFor: c.NotFor, Examples: c.Examples, Specialist: c.Specialist,
		ExpectedLatencyMS: c.ExpectedLatency.Milliseconds()}
}

// Decider is one step of the decision chain.
type Decider struct {
	Name      string `yaml:"name"`
	Type      string `yaml:"type"`
	URL       string `yaml:"url"`
	Model     string `yaml:"model"`
	APIKeyEnv string `yaml:"api_key_env"`
	// Timeout is how long the chain waits for this decider's answer before
	// it fails and the next decider is tried; auto.decide_timeout when
	// unset, and at most that.
	Timeout time.Duration `yaml:"timeout"`
}

// Decider types.
const (
	SystemOne = "systemone"
	LLM       = "llm"
)

// DefaultDecider is how a decision no decider made records its decider.
const DefaultDecider = "default"

// EnvVar holds the whole configuration document; when set it wins over the
// configuration file (Incus passes it as an instance environment key).
const EnvVar = "TS_ROUTER_CONFIG"

// Load reads, defaults and validates the configuration from EnvVar, or else
// from the file at path.
func Load(path string) (*Config, error) {
	if document := os.Getenv(EnvVar); document != "" {
		return Parse([]byte(document))
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(data)
}

// Parse decodes a configuration document strictly, applies defaults, loads
// the secrets files and validates the result.
func Parse(data []byte) (*Config, error) {
	var cfg Config
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	cfg.applyDefaults()
	cfg.secrets = map[string]string{}
	for _, path := range cfg.SecretsFiles {
		if err := readSecrets(path, cfg.secrets); err != nil {
			return nil, err
		}
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	return &cfg, nil
}

// Secret returns the named secret from the secrets files, falling back to
// the process environment.
func (c *Config) Secret(name string) string {
	if value, ok := c.secrets[name]; ok {
		return value
	}
	return os.Getenv(name)
}

// Price is the list price of a model id, ignoring an effort suffix.
func (c *Config) Price(model string) (usage.Price, bool) {
	price, ok := c.Prices[Bare(model)]
	return price, ok
}

// Bare is a model id without CLIProxy's `(effort)` suffix.
func Bare(model string) string {
	if i := strings.IndexByte(model, '('); i > 0 && strings.HasSuffix(model, ")") {
		return model[:i]
	}
	return model
}

// CostFactor is what a dollar of a model's list price costs: its
// candidate's cost factor, else 1.
func (c *Config) CostFactor(model string) float64 {
	return c.Auto.CostFactor(model)
}

// CostFactor is what a dollar of a model's list price costs: its
// candidate's cost factor, else 1.
func (a *Auto) CostFactor(model string) float64 {
	if candidate, ok := a.Candidate(model); ok && candidate.CostFactor != nil {
		return *candidate.CostFactor
	}
	return 1
}

// DashboardPasswordHash is the operator's bcrypt password hash. It is never
// derived from an upstream API key; empty disables dashboard login.
func (c *Config) DashboardPasswordHash() string { return c.Secret(c.Dashboard.PasswordHashEnv) }

// UpstreamKey is the CLIProxy client key.
func (c *Config) UpstreamKey() string {
	return c.Secret(c.Upstream.APIKeyEnv)
}

// ManagementKey is the CLIProxy management key, empty when the account
// watch is off.
func (c *Config) ManagementKey() string {
	if c.Accounts.ManagementKeyEnv == "" {
		return ""
	}
	return c.Secret(c.Accounts.ManagementKeyEnv)
}

// Level is the configured slog level.
func (c *Config) Level() slog.Level {
	var level slog.Level
	_ = level.UnmarshalText([]byte(c.LogLevel))
	return level
}

// Candidate returns the candidate with the given id.
func (a *Auto) Candidate(id string) (Candidate, bool) {
	for _, candidate := range a.Candidates {
		if candidate.ID == id {
			return candidate, true
		}
	}
	return Candidate{}, false
}

// Effort is the reasoning effort model runs at for a thread whose chosen
// level is level: the accepted effort nearest to the level (the stronger
// on a tie), or the candidate's fixed Reasoning when it accepts none or no
// level was chosen. A single accepted effort always fixes the model to
// that effort. It is empty for a model that is not a candidate, which keeps
// the client's own setting.
func (a *Auto) Effort(model, level string) string {
	candidate, ok := a.Candidate(model)
	if !ok {
		return ""
	}
	if len(candidate.Efforts) == 1 {
		return candidate.Efforts[0]
	}
	want := a.Reasoning.Rank(level)
	if want < 0 || len(candidate.Efforts) == 0 {
		return candidate.Reasoning
	}
	best, distance := "", 0
	for _, accepted := range candidate.Efforts {
		rank := a.Reasoning.Rank(accepted)
		d := rank - want
		if d < 0 {
			// Below the level: weaker at the same distance, so it loses ties.
			d = -2*d + 1
		} else {
			d *= 2
		}
		if best == "" || d < distance {
			best, distance = accepted, d
		}
	}
	return best
}

// Ladder is the ids of the non-specialist candidates, lightest first.
func (a *Auto) Ladder() []string {
	var ids []string
	for _, c := range a.Candidates {
		if !c.Specialist {
			ids = append(ids, c.ID)
		}
	}
	return ids
}

// CandidateIDs are the candidate ids, in ladder order.
func (a *Auto) CandidateIDs() []string {
	ids := make([]string, len(a.Candidates))
	for i, candidate := range a.Candidates {
		ids[i] = candidate.ID
	}
	return ids
}

// FallsBackTo reports whether some model fails over to model.
func (a *Auto) FallsBackTo(model string) bool {
	for _, to := range a.Fallback {
		if to == model {
			return true
		}
	}
	return false
}

// IsRetired reports whether model is taken out of service.
func (a *Auto) IsRetired(model string) bool {
	return slices.Contains(a.Retired, model)
}

// Rank is the position of a candidate in the cheapest-first order, or -1.
func (a *Auto) Rank(id string) int {
	return slices.IndexFunc(a.Candidates, func(c Candidate) bool { return c.ID == id })
}

// MinContext and MinOutput are the limits advertised for the virtual model:
// a thread may land on any candidate, so clients must plan for the smallest.
func (a *Auto) MinContext() int {
	return minimum(a.Candidates, func(c Candidate) int { return c.Context })
}

func (a *Auto) MinOutput() int {
	return minimum(a.Candidates, func(c Candidate) int { return c.Output })
}

// Required lists every model id the configuration routes to.
func (a *Auto) Required() []string {
	ids := []string{a.DefaultModel, a.AuxModel}
	for _, candidate := range a.Candidates {
		ids = append(ids, candidate.ID)
	}
	for from, to := range a.Fallback {
		if !a.IsRetired(from) {
			ids = append(ids, from)
		}
		ids = append(ids, to)
	}
	slices.Sort(ids)
	return slices.Compact(ids)
}

func minimum(candidates []Candidate, value func(Candidate) int) int {
	result := 0
	for _, candidate := range candidates {
		if v := value(candidate); v > 0 && (result == 0 || v < result) {
			result = v
		}
	}
	return result
}

func (c *Config) applyDefaults() {
	setDefault(&c.Listen, ":8080")
	setDefault(&c.Database, "/var/lib/ai-router/router.db")
	setDefault(&c.LogLevel, "info")
	setDefault(&c.Upstream.APIKeyEnv, "CLIPROXYAPI_API_KEY")
	setDefault(&c.Dashboard.Username, "mariogk")
	setDefault(&c.Dashboard.PasswordHashEnv, "TS_ROUTER_DASHBOARD_PASSWORD_HASH")
	if c.MaxBodyBytes == 0 {
		c.MaxBodyBytes = 64 << 20
	}
	if c.CatalogInterval == 0 {
		c.CatalogInterval = 5 * time.Minute
	}
	if c.LogRetention == 0 {
		c.LogRetention = 30 * 24 * time.Hour
	}
	if c.ShutdownGrace == 0 {
		c.ShutdownGrace = time.Minute
	}
	if c.Upstream.IdleTimeout == 0 {
		c.Upstream.IdleTimeout = 10 * time.Minute
	}
	if c.Accounts.Interval == 0 {
		c.Accounts.Interval = 5 * time.Minute
	}
	if c.Accounts.Providers == nil {
		c.Accounts.Providers = []string{accounts.Claude, accounts.Codex}
	}
	if c.Accounts.LimitedPriority == 0 {
		c.Accounts.LimitedPriority = -10
	}
	for model, limit := range c.Limits {
		if limit.Recovery == 0 {
			limit.Recovery = time.Minute
		}
		if limit.QueueTimeout == 0 {
			limit.QueueTimeout = 5 * time.Minute
		}
		if limit.Retries == nil {
			retries := 4
			limit.Retries = &retries
		}
		c.Limits[model] = limit
	}
	a := &c.Auto
	setDefault(&a.ModelID, "auto")
	setDefault(&a.Doubt, decide.DoubtEscalate)
	setDefault(&a.AuxModel, a.DefaultModel)
	if a.EscalationThreshold == 0 {
		a.EscalationThreshold = 0.10
	}
	if a.FailoverStatuses == nil {
		a.FailoverStatuses = []int{429, 500, 502, 503, 504, 529}
	}
	if a.FailoverTrip == nil {
		trip := 3
		a.FailoverTrip = &trip
	}
	if a.StripTags == nil {
		a.StripTags = []string{"system-reminder", "environment_context", "user_instructions"}
	}
	if a.StateMaxChars == 0 {
		a.StateMaxChars = 12000
	}
	if a.DecideTimeout == 0 {
		a.DecideTimeout = 30 * time.Second
	}
	if a.RecordPrompts == nil {
		record := true
		a.RecordPrompts = &record
	}
	if a.DeciderTrip == nil {
		trip := 3
		a.DeciderTrip = &trip
	}
	if a.DeciderRest == 0 {
		a.DeciderRest = time.Minute
	}
	if a.Redecide.Idle == nil {
		idle := 5 * time.Minute
		a.Redecide.Idle = &idle
	}
	if a.Redecide.GrowRequests == nil {
		grow := 60
		a.Redecide.GrowRequests = &grow
	}
	if a.Cost.Miss == nil {
		miss := 3.0
		a.Cost.Miss = &miss
	}
	if a.Cost.Window == 0 {
		a.Cost.Window = 30 * 24 * time.Hour
	}
	if a.Cost.Weight == nil {
		weight := 20.0
		a.Cost.Weight = &weight
	}
	if a.Learn.ExploreMargin == nil {
		margin := 0.25
		a.Learn.ExploreMargin = &margin
	}
	if a.Outcomes.Long == nil {
		long := 200
		a.Outcomes.Long = &long
	}
	if a.Outcomes.Tiny == nil {
		tiny := 3
		a.Outcomes.Tiny = &tiny
	}
	if a.Learn.Enabled == nil {
		enabled := true
		a.Learn.Enabled = &enabled
	}
	if a.Learn.Neighbours == nil {
		neighbours := 20
		a.Learn.Neighbours = &neighbours
	}
	for i := range c.Deciders {
		d := &c.Deciders[i]
		if d.Timeout == 0 {
			d.Timeout = a.DecideTimeout
		}
		if d.Type == LLM && d.URL == "" {
			d.URL = strings.TrimSuffix(c.Upstream.BaseURL, "/") + "/v1/chat/completions"
		}
		if d.Type == LLM && d.APIKeyEnv == "" {
			d.APIKeyEnv = c.Upstream.APIKeyEnv
		}
	}
}

func setDefault(field *string, value string) {
	if *field == "" {
		*field = value
	}
}

// nonnegativeFinite reports whether every value is safe to use in a cost
// calculation and serialize in a decision or policy.
func nonnegativeFinite(values ...float64) bool {
	for _, value := range values {
		if value < 0 || math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func (c *Config) validate() error {
	var problems []error
	add := func(format string, args ...any) { problems = append(problems, fmt.Errorf(format, args...)) }

	if err := httpURL(c.Upstream.BaseURL); err != nil {
		add("upstream.base_url: %w", err)
	}
	if c.UpstreamKey() == "" {
		add("upstream key %s is not set", c.Upstream.APIKeyEnv)
	}
	for model, price := range c.Prices {
		if !nonnegativeFinite(price.Input, price.Output, price.CacheRead, price.CacheWrite) {
			add("prices.%s: prices must not be negative or non-finite", model)
		}
	}
	if c.MaxBodyBytes < 0 || c.CatalogInterval < 0 || c.LogRetention < 0 || c.PromptRetention < 0 {
		add("max_body_bytes, catalog_interval, log_retention and prompt_retention must be positive")
	}
	if c.ShutdownGrace < time.Second || c.Upstream.IdleTimeout < time.Second {
		add("shutdown_grace and upstream.idle_timeout must be at least 1s")
	}
	if len(c.Dashboard.PasswordHashEnv) > 256 || !secretEnvName.MatchString(c.Dashboard.PasswordHashEnv) {
		add("dashboard.password_hash_env must name a secret environment variable")
	}
	if _, err := auth.NewPassword(c.Dashboard.Username, c.DashboardPasswordHash()); err != nil {
		add("%w", err)
	}
	var level slog.Level
	if err := level.UnmarshalText([]byte(c.LogLevel)); err != nil {
		add("log_level: %w", err)
	}
	for model, limit := range c.Limits {
		switch {
		case limit.Concurrency < 1:
			add("limits.%s.concurrency must be at least 1", model)
		case limit.Recovery < time.Second || limit.QueueTimeout < time.Second || *limit.Retries < 0 || *limit.Retries > 10:
			add("limits.%s: recovery and queue_timeout must be at least 1s, retries between 0 and 10", model)
		case Bare(model) != model:
			add("limits.%s: cap the bare model; every effort shares its gate", model)
		case limit.Spill == model:
			add("limits.%s.spill is the model itself", model)
		case limit.Spill != "" && c.Auto.Rank(limit.Spill) < 0:
			add("limits.%s.spill %q is not a candidate", model, limit.Spill)
		}
	}
	if c.Accounts.ManagementKeyEnv != "" {
		if c.ManagementKey() == "" {
			add("accounts: management key %s is not set", c.Accounts.ManagementKeyEnv)
		}
		if c.Accounts.Interval < 30*time.Second {
			add("accounts.interval must be at least 30s")
		}
		if c.Accounts.LimitedPriority >= 0 {
			add("accounts.limited_priority must be negative")
		}
		for _, provider := range c.Accounts.Providers {
			if provider != accounts.Claude && provider != accounts.Codex {
				add("accounts.providers: %q reports no quota (claude, codex)", provider)
			}
		}
	}

	a := c.Auto
	if len(a.Candidates) == 0 {
		add("auto.candidates is empty")
	}
	levels := map[string]bool{}
	for _, level := range a.Reasoning.Levels {
		switch {
		case level.ID == "" || !effort.MatchString(level.ID):
			add("auto.reasoning.levels: %q is not an effort level or token budget", level.ID)
		case levels[level.ID]:
			add("auto.reasoning.levels: %q is duplicated", level.ID)
		case strings.TrimSpace(level.Description) == "":
			add("auto.reasoning.levels: %q has no description", level.ID)
		}
		levels[level.ID] = true
	}
	if n := len(a.Reasoning.Levels); n == 1 || n > maxLevels {
		add("auto.reasoning.levels: the chooser needs 2 to %d levels (got %d)", maxLevels, n)
	}
	seen := map[string]bool{a.ModelID: true}
	for _, candidate := range a.Candidates {
		previous := -1
		for _, level := range candidate.Efforts {
			rank := a.Reasoning.Rank(level)
			switch {
			case rank < 0:
				add("auto.candidates: %q effort %q is not one of auto.reasoning.levels", candidate.ID, level)
			case rank <= previous:
				add("auto.candidates: %q efforts must follow auto.reasoning.levels, lightest first, once each", candidate.ID)
			}
			previous = max(previous, rank)
		}
		if len(candidate.Efforts) > 0 && candidate.Reasoning != "" && !slices.Contains(candidate.Efforts, candidate.Reasoning) {
			add("auto.candidates: %q reasoning %q is not one of its efforts", candidate.ID, candidate.Reasoning)
		}
		switch {
		case candidate.ID == "":
			add("auto.candidates: a candidate has no id")
		case seen[candidate.ID]:
			add("auto.candidates: %q is duplicated or equals auto.model_id", candidate.ID)
		case strings.TrimSpace(candidate.Description) == "":
			add("auto.candidates: %q has no description", candidate.ID)
		case candidate.Context < 0 || candidate.Output < 0:
			add("auto.candidates: %q has a negative limit", candidate.ID)
		case candidate.ExpectedLatency < 0:
			add("auto.candidates: %q expected_latency must not be negative", candidate.ID)
		case !effort.MatchString(candidate.Reasoning):
			add("auto.candidates: %q reasoning %q is not an effort level or token budget", candidate.ID, candidate.Reasoning)
		}
		seen[candidate.ID] = true
	}
	if a.Rank(a.DefaultModel) < 0 {
		add("auto.default_model %q is not a candidate", a.DefaultModel)
	}
	if a.DecideTimeout <= 0 {
		add("auto.decide_timeout must be positive")
	}
	for _, status := range a.FailoverStatuses {
		if status < 400 || status > 599 {
			add("auto.failover_statuses: %d is not an error status", status)
		}
	}
	if slices.ContainsFunc(a.StripTags, func(tag string) bool { return strings.TrimSpace(tag) == "" }) {
		add("auto.strip_tags: a tag is empty")
	}
	if math.IsNaN(a.EscalationThreshold) || a.EscalationThreshold <= 0 || a.EscalationThreshold > 1 {
		add("auto.escalation_threshold must be finite and in (0, 1]")
	}
	if a.Doubt != decide.DoubtEscalate && a.Doubt != decide.DoubtPair {
		add("auto.doubt must be escalate or pair")
	}
	if a.StateMaxChars < 200 {
		add("auto.state_max_chars must be at least 200")
	}
	if *a.FailoverTrip < 1 {
		add("auto.failover_trip must be at least 1")
	}
	if *a.DeciderTrip < 0 || a.DeciderRest < time.Second {
		add("auto.decider_trip must not be negative and auto.decider_rest must be at least 1s")
	}
	if idle := *a.Redecide.Idle; idle < 0 || idle > 0 && idle < time.Minute {
		add("auto.redecide.idle must be 0 (never) or at least 1m")
	}
	if *a.Redecide.GrowRequests < 0 {
		add("auto.redecide.grow_requests must not be negative")
	}
	for from, to := range a.Fallback {
		if from == "" || to == "" || from == to {
			add("auto.fallback: %q -> %q is not a switch to another model", from, to)
		}
	}
	if !nonnegativeFinite(*a.Cost.Miss, *a.Cost.Weight, a.Cost.TimeValue) || a.Cost.Window < time.Hour {
		add("auto.cost: miss, weight and time_value must not be negative or non-finite, window at least 1h")
	}
	if !nonnegativeFinite(a.Learn.Explore, *a.Learn.ExploreMargin) || a.Learn.Explore > 0.2 {
		add("auto.learn: explore must be from 0 to 0.2 and explore_margin must not be negative or non-finite")
	}
	if *a.Outcomes.Long < 0 || *a.Outcomes.Tiny < 0 {
		add("auto.outcomes: long and tiny must not be negative")
	}
	if *a.Learn.Neighbours < 1 || *a.Learn.Neighbours > 200 {
		add("auto.learn.neighbours must be between 1 and 200")
	}
	if a.Cost.Window > c.LogRetention {
		add("auto.cost.window must not exceed log_retention: older threads have lost requests")
	}
	for _, candidate := range a.Candidates {
		if candidate.CostFactor != nil && !nonnegativeFinite(*candidate.CostFactor) {
			add("auto.candidates: %q cost_factor must not be negative or non-finite", candidate.ID)
		}
		if _, priced := c.Price(candidate.ID); a.Cost.Enabled && !candidate.Specialist && !priced {
			add("auto.cost: ladder candidate %q has no price", candidate.ID)
		}
	}
	for _, model := range a.Retired {
		switch {
		case a.Rank(model) >= 0:
			add("auto.retired: %q is still a candidate", model)
		case a.FallsBackTo(model):
			add("auto.retired: %q is still a fallback", model)
		case model == a.AuxModel:
			add("auto.retired: %q is the aux model", model)
		}
	}

	if len(c.Deciders) == 0 {
		add("deciders is empty")
	}
	names := map[string]bool{}
	for _, d := range c.Deciders {
		switch {
		case d.Name == "" || names[d.Name]:
			add("deciders: names must be unique and non-empty (got %q)", d.Name)
		case d.Name == DefaultDecider:
			add("deciders: %q names the default model, not a decider", d.Name)
		}
		names[d.Name] = true
		switch d.Type {
		case SystemOne:
			if d.Model == "" || d.APIKeyEnv == "" {
				add("deciders.%s: systemone needs model and api_key_env", d.Name)
			}
		case LLM:
			if d.Model == "" {
				add("deciders.%s: llm needs model", d.Name)
			}
		default:
			add("deciders.%s: type must be %s or %s", d.Name, SystemOne, LLM)
			continue
		}
		if err := httpURL(d.URL); err != nil {
			add("deciders.%s.url: %w", d.Name, err)
		}
		if c.Secret(d.APIKeyEnv) == "" {
			add("deciders.%s: secret %s is not set", d.Name, d.APIKeyEnv)
		}
		if d.Timeout <= 0 || d.Timeout > a.DecideTimeout {
			add("deciders.%s: timeout must be positive and at most auto.decide_timeout (%s)", d.Name, a.DecideTimeout)
		}
	}
	return errors.Join(problems...)
}

func httpURL(raw string) error {
	parsed, err := url.Parse(raw)
	if err != nil {
		return err
	}
	if (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return fmt.Errorf("%q is not an http(s) URL", raw)
	}
	return nil
}
