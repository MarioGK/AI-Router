package dashboard

import (
	"errors"
	"fmt"
	"mime"
	"net/http"
	"strconv"
	"time"
)

// deleteLogs only removes request history after an explicit form confirmation.
// Pins and decisions survive; deletion never changes the routing controls.
func (d *Dashboard) deleteLogs(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 4<<10)
	status, problem := http.StatusBadRequest, ""
	mediaType, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || mediaType != "application/x-www-form-urlencoded" {
		status, problem = http.StatusUnsupportedMediaType, "Use the delete logs form on this page."
	} else if err := r.ParseForm(); err != nil {
		problem = "The deletion form could not be read. No logs were deleted."
		if _, large := errors.AsType[*http.MaxBytesError](err); large {
			status = http.StatusRequestEntityTooLarge
		}
	}
	query := logQuery{Model: r.PostForm.Get("model"), Status: r.PostForm.Get("status"), Search: r.PostForm.Get("search")}
	before := time.Now()
	if problem == "" {
		for name, entries := range r.PostForm {
			if len(entries) != 1 || (name != "age" && name != "confirm" && name != "model" && name != "status" && name != "search") {
				problem = "Use the delete logs controls on this page. No logs were deleted."
				break
			}
		}
		switch r.PostForm.Get("age") {
		case "24h":
			before = before.Add(-24 * time.Hour)
		case "7d":
			before = before.Add(-7 * 24 * time.Hour)
		case "30d":
			before = before.Add(-30 * 24 * time.Hour)
		case "all":
		default:
			problem = "Choose which age of request logs to delete. No logs were deleted."
		}
		if r.PostForm.Get("confirm") != "delete" {
			problem = "Check the confirmation box before deleting request logs. No logs were deleted."
		}
	}
	if problem == "" {
		deleted, err := d.source.Prune(r.Context(), before)
		// Even a partial deletion must disappear from other open live tables.
		if deleted > 0 {
			d.logRevision.Add(1)
		}
		if err == nil {
			values := query.values()
			values.Set("deleted", strconv.FormatInt(deleted, 10))
			http.Redirect(w, r, "/dashboard/logs?"+values.Encode(), http.StatusSeeOther)
			return
		}
		d.log.Warn("dashboard log deletion failed", "deleted", deleted, "error", err)
		status = http.StatusInternalServerError
		problem = fmt.Sprintf("Deleted %d request logs before the operation stopped. Try again to finish deleting the remaining logs.", deleted)
	}
	t := query.tail(d.source)
	t.load(r.Context())
	d.render(w, r, status, logsPage(newFrame(r, "Logs", navLogs), query, t.view(), d.historyModels(), "", problem))
}
