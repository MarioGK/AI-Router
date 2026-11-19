package route

import "sync"

// outages counts the server errors each model answered in a row, so one
// that passes does not move a thread off its model, and its prompt cache,
// for good.
type outages struct {
	mu     sync.Mutex
	errors map[string]int
}

// answered records a reply of model and returns how many server errors it
// has answered in a row, this one included.
func (o *outages) answered(model string, status int) int {
	o.mu.Lock()
	defer o.mu.Unlock()
	if status < 500 {
		delete(o.errors, model)
		return 0
	}
	if o.errors == nil {
		o.errors = map[string]int{}
	}
	o.errors[model]++
	return o.errors[model]
}
