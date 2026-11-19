// Package catalog tracks which model ids CLIProxy currently serves, so new
// threads are never pinned to a model it cannot route.
package catalog

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"
)

// Catalog is the last model list fetched from CLIProxy.
type Catalog struct {
	url, key string
	client   *http.Client

	mu        sync.RWMutex
	models    map[string]bool
	loaded    bool
	lastError string
	checkedAt time.Time
	// fetchedAt is when the last refresh succeeded.
	fetchedAt time.Time
}

const (
	// stale is how long the catalog stays healthy after its last good
	// refresh while refreshes fail.
	stale = 10 * time.Minute
	// retry is how soon a failed refresh is tried again.
	retry = 30 * time.Second
)

// New returns a catalog for the CLIProxy base URL.
func New(baseURL, key string, client *http.Client) *Catalog {
	return &Catalog{url: strings.TrimSuffix(baseURL, "/") + "/v1/models", key: key, client: client}
}

// Refresh fetches the model list.
func (c *Catalog) Refresh(ctx context.Context) error {
	models, err := c.fetch(ctx)
	c.mu.Lock()
	defer c.mu.Unlock()
	c.checkedAt = time.Now()
	if err != nil {
		c.lastError = err.Error()
		return err
	}
	c.models, c.loaded, c.lastError, c.fetchedAt = models, true, "", c.checkedAt
	return nil
}

func (c *Catalog) fetch(ctx context.Context) (map[string]bool, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, c.url, nil)
	if err != nil {
		return nil, err
	}
	request.Header.Set("Authorization", "Bearer "+c.key)
	response, err := c.client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GET /v1/models: status %d", response.StatusCode)
	}
	var list struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 8<<20)).Decode(&list); err != nil {
		return nil, fmt.Errorf("GET /v1/models: %w", err)
	}
	models := map[string]bool{}
	for _, model := range list.Data {
		models[model.ID] = true
	}
	return models, nil
}

// Run refreshes the catalog every interval until ctx ends, and sooner
// while refreshes fail, the first one before Run included.
func (c *Catalog) Run(ctx context.Context, interval time.Duration, log *slog.Logger) {
	c.mu.RLock()
	first := interval
	if !c.loaded || c.lastError != "" {
		first = min(interval, retry)
	}
	c.mu.RUnlock()
	timer := time.NewTimer(first)
	defer timer.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-timer.C:
			refreshCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
			err := c.Refresh(refreshCtx)
			cancel()
			next := interval
			if err != nil && ctx.Err() == nil {
				log.Warn("catalog refresh failed", "error", err)
				next = min(interval, retry)
			}
			timer.Reset(next)
		}
	}
}

// Has reports whether CLIProxy serves id. Before the first successful fetch
// every id counts as served, so a slow CLIProxy start blocks nothing.
func (c *Catalog) Has(id string) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return !c.loaded || c.models[id]
}

// Status describes the catalog against the ids the configuration needs.
type Status struct {
	Loaded  bool     `json:"loaded"`
	Missing []string `json:"missing,omitempty"`
	// Error is why the latest refresh failed; Stale is set once refreshes
	// have failed for longer than the catalog stays trusted.
	Error     string    `json:"error,omitempty"`
	Stale     bool      `json:"stale,omitempty"`
	CheckedAt time.Time `json:"checked_at"`
}

// Healthy reports whether every required model is served, as far as a
// recent model list tells.
func (s Status) Healthy() bool {
	return s.Loaded && !s.Stale && len(s.Missing) == 0
}

// Check reports which of the required ids CLIProxy does not serve.
func (c *Catalog) Check(required []string) Status {
	c.mu.RLock()
	defer c.mu.RUnlock()
	status := Status{Loaded: c.loaded, Error: c.lastError, CheckedAt: c.checkedAt,
		Stale: c.loaded && c.lastError != "" && c.checkedAt.Sub(c.fetchedAt) > stale}
	if c.loaded {
		for _, id := range required {
			if !c.models[id] {
				status.Missing = append(status.Missing, id)
			}
		}
		slices.Sort(status.Missing)
	}
	return status
}
