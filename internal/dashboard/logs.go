package dashboard

import (
	"net/http"
	"net/url"
	"strconv"

	"github.com/starfederation/datastar-go/datastar"

	"github.com/MarioGK/AI-Router/internal/pin"
)

// logQuery selects the requests the log shows.
type logQuery struct {
	Search string `json:"search"`
	Model  string `json:"model"`
	Status string `json:"status"`
	Paused bool   `json:"paused"`
}

var outcomes = []option{{"errors", "Errors only"}, {"success", "Success"}}

// tail follows the requests the query selects.
func (q logQuery) tail(source Source) *tail {
	return &tail{source: source, filter: pin.RequestFilter{Model: q.Model, Status: q.Status, Search: q.Search}}
}

// values is the query as page URL parameters.
func (q logQuery) values() url.Values {
	values := url.Values{}
	for name, value := range map[string]string{"model": q.Model, "status": q.Status, "search": q.Search} {
		if value != "" {
			values.Set(name, value)
		}
	}
	return values
}

// logs shows the latest requests for the filters in the URL; the page then
// follows them live.
func (d *Dashboard) logs(w http.ResponseWriter, r *http.Request) {
	values := r.URL.Query()
	query := logQuery{Model: values.Get("model"), Status: values.Get("status"), Search: values.Get("search")}
	t := query.tail(d.source)
	t.revision = d.logRevision.Load
	t.load(r.Context())
	deleted := ""
	if count, err := strconv.ParseInt(values.Get("deleted"), 10, 64); err == nil && count >= 0 {
		deleted = strconv.FormatInt(count, 10)
	}
	d.render(w, r, http.StatusOK, logsPage(newFrame(r, "Logs", navLogs), query, t.view(), d.historyModels(), deleted, ""))
}

// tailLogs streams the log for the page's filters until they change; a
// paused log gets nothing, which ends the stream it had.
func (d *Dashboard) tailLogs(w http.ResponseWriter, r *http.Request) {
	var query logQuery
	if err := datastar.ReadSignals(r, &query); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if query.Paused {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	ctx := r.Context()
	sse := datastar.NewSSE(w, r)
	t := query.tail(d.source)
	t.revision = d.logRevision.Load
	t.load(ctx)
	d.patch(sse, logRows(t.view()))
	d.replaceURL(sse, "/dashboard/logs", query.values())
	d.follow(ctx, func(tick int) {
		if t.poll(ctx) || tick%retimeTicks == 0 {
			d.patch(sse, logRows(t.view()))
		}
	})
}
