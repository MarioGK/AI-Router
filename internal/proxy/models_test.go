package proxy

import (
	"encoding/json"
	"slices"
	"testing"
)

func TestAddModelMatchesTheListShape(t *testing.T) {
	virtual := Virtual{ID: "auto", Context: 200000, Output: 64000}
	openai, err := AddModel([]byte(`{"object":"list","data":[{"id":"glm-5.3","object":"model","created":1790609949,"owned_by":"anthropic"}]}`), virtual, nil)
	if err != nil {
		t.Fatal(err)
	}
	var list struct {
		Object string           `json:"object"`
		Data   []map[string]any `json:"data"`
	}
	if err := json.Unmarshal(openai, &list); err != nil || list.Object != "list" || len(list.Data) != 2 {
		t.Fatalf("list = %s, %v", openai, err)
	}
	if first := list.Data[0]; first["id"] != "auto" || first["object"] != "model" || first["context_length"] != 200000.0 {
		t.Fatalf("auto entry = %v", first)
	}
	if second := list.Data[1]; second["created"] != 1790609949.0 {
		t.Fatalf("existing entry changed: %v", second)
	}

	anthropic, err := AddModel([]byte(`{"data":[{"type":"model","id":"claude-sonnet-5","display_name":"Claude Sonnet 5"}],"has_more":false}`), virtual, nil)
	if err != nil {
		t.Fatal(err)
	}
	var anthropicList struct {
		Data []map[string]any `json:"data"`
	}
	_ = json.Unmarshal(anthropic, &anthropicList)
	if first := anthropicList.Data[0]; first["type"] != "model" || first["display_name"] != "Auto" {
		t.Fatalf("anthropic auto entry = %v", first)
	}

	present := []byte(`{"data":[{"id":"auto"}]}`)
	if same, err := AddModel(present, virtual, nil); err != nil || string(same) != string(present) {
		t.Fatalf("AddModel changed a list that already has auto: %s, %v", same, err)
	}
	if _, err := AddModel([]byte(`{"error":"nope"}`), virtual, nil); err == nil {
		t.Fatal("AddModel accepted a body without data")
	}
}

func TestAddModelOmitsRetiredModels(t *testing.T) {
	for _, body := range []string{
		`{"data":[{"id":"claude-sonnet-5-5"},{"id":"claude-sonnet-5"},{"id":"retired-model"},{"id":"gpt-6.1-sol"}],"has_more":false}`,
		`{"data":[{"id":"auto"},{"id":"claude-sonnet-5-5"},{"id":"claude-sonnet-5"},{"id":"retired-model"},{"id":"gpt-6.1-sol"}],"has_more":false}`,
	} {
		filtered, err := AddModel([]byte(body), Virtual{ID: "auto"}, []string{"retired-model", "claude-sonnet-5-5"})
		if err != nil {
			t.Fatal(err)
		}
		var list struct {
			Data []struct {
				ID string `json:"id"`
			} `json:"data"`
			HasMore bool `json:"has_more"`
		}
		if err := json.Unmarshal(filtered, &list); err != nil {
			t.Fatal(err)
		}
		var ids []string
		for _, entry := range list.Data {
			ids = append(ids, entry.ID)
		}
		if !slices.Equal(ids, []string{"auto", "claude-sonnet-5", "gpt-6.1-sol"}) || list.HasMore {
			t.Fatalf("filtered list = %s", filtered)
		}
	}
}
