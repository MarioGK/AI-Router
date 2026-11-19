package proxy

import (
	"encoding/json"
	"fmt"
	"slices"
	"time"
)

// Virtual describes the virtual model added to model lists.
type Virtual struct {
	ID      string
	Context int
	Output  int
}

// AddModel prepends the virtual model to a /v1/models list body, in the
// shape of the list's own entries (OpenAI or Anthropic), omitting retired
// models from the selectable catalog.
func AddModel(body []byte, model Virtual, retired []string) ([]byte, error) {
	var list map[string]json.RawMessage
	if err := json.Unmarshal(body, &list); err != nil {
		return nil, fmt.Errorf("model list: %w", err)
	}
	var data []json.RawMessage
	if err := json.Unmarshal(list["data"], &data); err != nil {
		return nil, fmt.Errorf("model list data: %w", err)
	}
	var shape map[string]json.RawMessage
	present, removed := false, false
	kept := make([]json.RawMessage, 0, len(data))
	for _, entry := range data {
		var fields map[string]json.RawMessage
		if json.Unmarshal(entry, &fields) != nil {
			kept = append(kept, entry)
			continue
		}
		if shape == nil {
			shape = fields
		}
		var id string
		if json.Unmarshal(fields["id"], &id) == nil {
			if slices.Contains(retired, id) {
				removed = true
				continue
			}
			present = present || id == model.ID
		}
		kept = append(kept, entry)
	}
	if present && !removed {
		return body, nil
	}
	data = kept

	if !present {
		entry := map[string]any{"id": model.ID}
		if _, anthropic := shape["type"]; anthropic {
			entry["type"] = "model"
			entry["display_name"] = "Auto"
			entry["created_at"] = time.Unix(0, 0).UTC().Format(time.RFC3339)
		} else {
			entry["object"] = "model"
			entry["created"] = 0
			entry["owned_by"] = "ai-router"
		}
		if model.Context > 0 {
			entry["context_length"] = model.Context
		}
		if model.Output > 0 {
			entry["max_output_tokens"] = model.Output
		}
		encoded, err := json.Marshal(entry)
		if err != nil {
			return nil, err
		}
		data = append([]json.RawMessage{encoded}, data...)
	}
	var err error
	if list["data"], err = json.Marshal(data); err != nil {
		return nil, err
	}
	return json.Marshal(list)
}
