package proxy

import (
	"encoding/json"
	"net/http"
)

// Error answers with an error in the Anthropic Messages API's shape.
func Error(w http.ResponseWriter, status int, kind, message string) {
	body := map[string]any{"type": "error", "error": map[string]string{"type": kind, "message": message}}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}
