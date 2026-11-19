package dashboard

import (
	"context"
	"net/http"
	"net/url"
	"strconv"

	"github.com/a-h/templ"
	"github.com/starfederation/datastar-go/datastar"

	"github.com/MarioGK/AI-Router/internal/pin"
)

// threadsPerPage is the length of a page of threads.
const threadsPerPage = 25

// threadQuery selects a page of threads.
type threadQuery struct {
	Model string `json:"model"`
	Page  int    `json:"-"`
}

// values is the query as page URL parameters.
func (q threadQuery) values() url.Values {
	values := url.Values{}
	for name, value := range map[string]string{"model": q.Model} {
		if value != "" {
			values.Set(name, value)
		}
	}
	if q.Page > 1 {
		values.Set("page", strconv.Itoa(q.Page))
	}
	return values
}

// threadsView is one page of threads.
type threadsView struct {
	Query threadQuery
	pin.ThreadPage
	Err error
}

// pages is how many pages the matching threads fill.
func (v threadsView) pages() int {
	return max(1, (v.Total+threadsPerPage-1)/threadsPerPage)
}

// first and last number the threads on the page.
func (v threadsView) first() int {
	return (v.Query.Page-1)*threadsPerPage + 1
}

func (v threadsView) last() int {
	return v.first() + len(v.Threads) - 1
}

// threadURL is a thread's page.
func threadURL(key string) templ.SafeURL {
	return templ.SafeURL("/dashboard/threads/" + url.PathEscape(key))
}

// pageURL is the stream that shows another page of the same threads.
func pageURL(page int) string {
	return "/dashboard/sse/threads?page=" + strconv.Itoa(page)
}

// threads lists the pinned threads for the filters and page in the URL.
func (d *Dashboard) threads(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query()
	page, _ := strconv.Atoi(query.Get("page"))
	view := d.listThreads(r.Context(), threadQuery{Model: query.Get("model"), Page: page})
	d.render(w, r, http.StatusOK, threadsPage(newFrame(r, "Threads", navThreads), view, d.historyModels()))
}

// threadRows re-renders the thread table for the filters and page asked for.
func (d *Dashboard) threadRows(w http.ResponseWriter, r *http.Request) {
	var query threadQuery
	if err := datastar.ReadSignals(r, &query); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	query.Page, _ = strconv.Atoi(r.URL.Query().Get("page"))
	view := d.listThreads(r.Context(), query)
	sse := datastar.NewSSE(w, r)
	d.patch(sse, threadsTable(view))
	d.replaceURL(sse, "/dashboard/threads", view.Query.values())
}

// listThreads reads a page of threads; a page past the end shows the last.
func (d *Dashboard) listThreads(ctx context.Context, query threadQuery) threadsView {
	query.Page = max(1, query.Page)
	view := threadsView{Query: query}
	for {
		filter := pin.ThreadFilter{
			Model: query.Model, Limit: threadsPerPage, Offset: (view.Query.Page - 1) * threadsPerPage,
		}
		view.ThreadPage, view.Err = d.source.Threads(ctx, filter)
		if view.Err != nil {
			d.log.Warn("dashboard threads failed", "error", view.Err)
			return view
		}
		if len(view.Threads) > 0 || view.Query.Page <= view.pages() {
			return view
		}
		view.Query.Page = view.pages()
	}
}
