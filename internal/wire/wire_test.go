package wire

import (
	"encoding/json"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
)

func TestFindModelReplacesOnlyTheTopLevelValue(t *testing.T) {
	body := []byte("{\n  \"stream\": true,\n  \"metadata\": {\"model\": \"auto\"},\n  \"model\" :  \"auto\",\n  \"x-unknown\": [1, {\"model\": \"auto\"}]\n}")
	field, ok := FindModel(body)
	if !ok || field.Value != "auto" {
		t.Fatalf("FindModel = %+v, %v", field, ok)
	}
	got := string(field.Replace(body, "glm-5.3"))
	want := "{\n  \"stream\": true,\n  \"metadata\": {\"model\": \"auto\"},\n  \"model\" :  \"glm-5.3\",\n  \"x-unknown\": [1, {\"model\": \"auto\"}]\n}"
	if got != want {
		t.Fatalf("Replace =\n%s\nwant\n%s", got, want)
	}
}

func TestFindModelDecodesEscapes(t *testing.T) {
	field, ok := FindModel([]byte(`{"model":"auto"}`))
	if !ok || field.Value != "auto" {
		t.Fatalf("FindModel = %+v, %v", field, ok)
	}
	if got := string(field.Replace([]byte(`{"model":"auto"}`), "gpt-6.1-sol")); got != `{"model":"gpt-6.1-sol"}` {
		t.Fatalf("Replace = %s", got)
	}
}

func TestFindModelRejectsBodiesWithoutAStringModel(t *testing.T) {
	for _, body := range []string{``, `[]`, `{"messages":[]}`, `{"model":7}`, `{"model":`, `not json`} {
		if field, ok := FindModel([]byte(body)); ok {
			t.Errorf("FindModel(%q) = %+v, want not found", body, field)
		}
	}
}

func TestFindModelTakesTheFirstOfRepeatedModels(t *testing.T) {
	body := []byte(`{"model":"auto","model":"glm-5.3"}`)
	field, ok := FindModel(body)
	if !ok || field.Value != "auto" {
		t.Fatalf("FindModel = %+v, %v", field, ok)
	}
	if got := string(field.Replace(body, "gpt-6.1-sol")); got != `{"model":"gpt-6.1-sol","model":"glm-5.3"}` {
		t.Fatalf("Replace = %s", got)
	}
}

func TestFindModelReadsNoFurtherThanTheModel(t *testing.T) {
	for _, body := range []string{`{"model":"auto", oops`, `{"model":"auto"}}`, `{"model" : "auto"`} {
		if field, ok := FindModel([]byte(body)); !ok || field.Value != "auto" {
			t.Errorf("FindModel(%q) = %+v, %v", body, field, ok)
		}
	}
	for _, body := range []string{`{"stream":tru,"model":"auto"}`, `{"a":"]}","model" "auto"}`, ` [{"model":"auto"}]`} {
		if field, ok := FindModel([]byte(body)); ok {
			t.Errorf("FindModel(%q) = %+v, want not found", body, field)
		}
	}
}

func TestKindOf(t *testing.T) {
	cases := map[string]Kind{
		"POST /v1/messages":              Messages,
		"POST /v1/messages/count_tokens": CountTokens,
		"GET /v1/models":                 Models,
		"GET /v1/models/":                Models,
		"GET /v1/messages":               Other,
		"POST /v1/chat/completions":      Other,
		"POST /v1/responses":             Other,
		"POST /v0/management/config":     Other,
	}
	for request, want := range cases {
		var method, path string
		for i := range request {
			if request[i] == ' ' {
				method, path = request[:i], request[i+1:]
				break
			}
		}
		if got := KindOf(method, path); got != want {
			t.Errorf("KindOf(%s) = %v, want %v", request, got, want)
		}
	}
}

func TestParseMessagesSkipsToolResultsButSeesTheirImages(t *testing.T) {
	c := Parse(Messages, []byte(`{"model":"auto","metadata":{"user_id":"u"},"messages":[
		{"role":"user","content":[{"type":"text","text":"fix the bug"}]},
		{"role":"assistant","content":[{"type":"tool_use","id":"t","name":"shot","input":{}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"image","source":{}}]}]}]}`))
	if c.LatestUser != "fix the bug" || !c.HasImage || c.MetadataUserID != "u" {
		t.Fatalf("Parse = %+v", c)
	}
}

func TestParseKeepsUserTextsNewestFirstWithoutToolData(t *testing.T) {
	c := Parse(Messages, []byte(`{"messages":[
		{"role":"user","content":"Add the routine endpoint; this can wait."},
		{"role":"assistant","content":[{"type":"text","text":"I will inspect the handler."}]},
		{"role":"user","content":[{"type":"tool_result","content":[{"type":"text","text":"Urgent redesign: tool output only."},{"type":"image"}]}]},
		{"role":"user","content":[{"type":"text","text":"<system-reminder>client context only</system-reminder>"}]},
		{"role":"user","content":[{"type":"text","text":" "},{"type":"tool_result","content":"Later tool output."}]}]}`))
	want := []string{"<system-reminder>client context only</system-reminder>", "Add the routine endpoint; this can wait."}
	if !slices.Equal(c.UserTexts, want) || c.LatestUser != want[0] || !c.HasImage {
		t.Fatalf("user turn history includes tool data or has the wrong order: %+v", c)
	}
}

func TestParsePromptedOnlyByANewUserPrompt(t *testing.T) {
	for _, tc := range []struct {
		name, messages string
		want           bool
	}{
		{"string", `[{"role":"user","content":"Fix the bug."}]`, true},
		{"text blocks", `[{"role":"assistant","content":"Done."},{"role":"user","content":[{"type":"text","text":"Next task."}]}]`, true},
		{"tool result", `[{"role":"user","content":"Fix it."},{"role":"user","content":[{"type":"tool_result","content":"ok"}]}]`, false},
		{"tool result with text", `[{"role":"user","content":[{"type":"tool_result","content":"ok"},{"type":"text","text":"And this."}]}]`, false},
		{"assistant last", `[{"role":"user","content":"Fix it."},{"role":"assistant","content":"Sure"}]`, false},
		{"image only", `[{"role":"user","content":[{"type":"image"}]}]`, false},
		{"none", `[]`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if c := Parse(Messages, []byte(`{"messages":`+tc.messages+`}`)); c.Prompted != tc.want {
				t.Fatalf("Prompted = %v, want %v: %+v", c.Prompted, tc.want, c)
			}
		})
	}
}

func TestParseUserTextsFollowReplacingMessagesMember(t *testing.T) {
	c := Parse(Messages, []byte(`{"messages":[{"role":"user","content":"stale task"}],"messages":[{"role":"user","content":"current task"}]}`))
	if !slices.Equal(c.UserTexts, []string{"current task"}) || c.LatestUser != "current task" {
		t.Fatalf("replaced messages retained stale user text: %+v", c)
	}
	c = Parse(Messages, []byte(`{"messages":[{"role":"user","content":"stale task"}],"messages":null}`))
	if c.LatestUser != "" || c.UserTexts != nil {
		t.Fatalf("null messages retained stale user text: %+v", c)
	}
}

func TestAppendSystem(t *testing.T) {
	cases := map[string]struct{ body, want string }{
		"block list keeps its blocks": {
			`{"model":"auto","system":[{"type":"text","text":"You are Claude Code.","cache_control":{"type":"ephemeral"}}],"messages":[]}`,
			`{"model":"auto","system":[{"type":"text","text":"You are Claude Code.","cache_control":{"type":"ephemeral"}},{"type":"text","text":"NOTE <m>"}],"messages":[]}`,
		},
		"empty block list": {`{"system":[ ],"messages":[]}`, `{"system":[ {"type":"text","text":"NOTE <m>"}],"messages":[]}`},
		"string prompt":    {`{"system":"Be brief.","messages":[]}`, `{"system":"Be brief.\n\nNOTE <m>","messages":[]}`},
		"no prompt":        {`{ "model":"auto","messages":[]}`, `{"system":"NOTE <m>", "model":"auto","messages":[]}`},
		"empty object":     {`{}`, `{"system":"NOTE <m>"}`},
	}
	for name, c := range cases {
		got, ok := AppendSystem([]byte(c.body), "NOTE <m>")
		if !ok || string(got) != c.want {
			t.Errorf("%s: AppendSystem =\n%s, %v\nwant\n%s", name, got, ok, c.want)
		}
	}
	for _, body := range []string{`[]`, `{"system":42}`, `not json`} {
		if got, ok := AppendSystem([]byte(body), "x"); ok || string(got) != body {
			t.Errorf("AppendSystem(%s) = %s, %v; want unchanged", body, got, ok)
		}
	}
}

func TestParseReadsOnlyWellFormedBodies(t *testing.T) {
	for _, body := range []string{
		`{"messages":[{"role":"user","content":"hi","content":[{"type":"image"}]}],"metadata":{"user_id":"u"}} x`,
		`{"messages":[{"role":"user","content":"hi","content":[{"type":"image"}]}],"metadata":{"user_id":"u"},"stream":tru}`,
	} {
		if c := Parse(Messages, []byte(body)); !reflect.DeepEqual(c, Conversation{}) {
			t.Errorf("Parse(%s) = %+v, want empty", body, c)
		}
	}
	if c := Parse(Other, []byte(`{"messages":[{"role":"user","content":"hi"}]}`)); !reflect.DeepEqual(c, Conversation{}) {
		t.Errorf("Parse(Other) = %+v, want empty", c)
	}
}

func TestParseDecodesLikeEncodingJSON(t *testing.T) {
	c := Parse(CountTokens, []byte(`{"Messages":[
		{"role":"user","content":[{"type":"text","text":"  keep \"this\"\n"}]},
		{"role":"user","ROLE":"assistant","content":[{"type":"image","source":{}}]},
		{"role":"user","content":[{"type":"text","text":"dropped"},{"type":"text","text":7},{"type":"image"}]},
		null, 7,
		{"role":"user","content":[{"type":"tool_result","content":[{"type":"document"}]},{"type":"text","text":" "}]}],
		"metadata":{"user_id":"u"},"metadata":{"user_ID":"v"}}`))
	want := Conversation{LatestUser: `keep "this"`, UserTexts: []string{`keep "this"`}, HasImage: true, MetadataUserID: "v"}
	if !reflect.DeepEqual(c, want) {
		t.Fatalf("Parse = %+v, want %+v", c, want)
	}
	if c := Parse(Messages, []byte(`{"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"image"},{"type":5}]}]}`)); c.HasImage || c.LatestUser != "" {
		t.Fatalf("Parse of a mistyped block list = %+v, want nothing", c)
	}
}

func TestAppendSystemFindsTheSystemAfterTheMessages(t *testing.T) {
	cases := map[string]struct{ body, want string }{
		"after the messages": {
			`{"model":"auto","messages":[{"role":"user","content":"\"system\": [\"x\"]"}],"system":"S"}`,
			`{"model":"auto","messages":[{"role":"user","content":"\"system\": [\"x\"]"}],"system":"S\n\nNOTE"}`,
		},
		"repeated": {`{"system":"A","system":"B"}`, `{"system":"A\n\nNOTE","system":"B"}`},
		"null":     {`{"system":null}`, `{"system":"\n\nNOTE"}`},
	}
	for name, c := range cases {
		got, ok := AppendSystem([]byte(c.body), "NOTE")
		if !ok || string(got) != c.want {
			t.Errorf("%s: AppendSystem =\n%s, %v\nwant\n%s", name, got, ok, c.want)
		}
	}
	for _, body := range []string{`{"model":"auto"} x`, `{"model":"auto",}`, `{"model":"auto"`} {
		if got, ok := AppendSystem([]byte(body), "x"); ok || string(got) != body {
			t.Errorf("AppendSystem(%s) = %s, %v; want unchanged", body, got, ok)
		}
	}
}

// claudeCode is a Claude Code shaped Messages body of about 32 MiB: a long
// conversation with screenshots, followed by the model, the system prompt
// and the tools, the worst case for finding a top-level member.
var claudeCode = sync.OnceValue(func() []byte {
	type block = map[string]any
	text := func(s string) block { return block{"type": "text", "text": s} }
	screenshot := block{"type": "image", "source": block{
		"type": "base64", "media_type": "image/png", "data": strings.Repeat("iVBORw0KGgoAAAANSUhEUgAA", 40_000),
	}}
	file := strings.Repeat("func (s *Server) Login(w http.ResponseWriter, r *http.Request) {\n\tif r.Header.Get(\"X-Token\") == \"\" { http.Error(w, `[\"denied\"]`, 401) }\n}\n", 60)
	var messages []block
	for size, turn := 0, 0; size < 32<<20; turn++ {
		result := []block{text(file)}
		if turn%4 == 3 {
			result = append(result, screenshot)
			size += 960_000
		}
		messages = append(messages,
			block{"role": "user", "content": []block{
				text("<system-reminder>\nThe user opened src/auth.go in the IDE.\n</system-reminder>"),
				text("fix the \"login\" bug, then check the page again"),
			}},
			block{"role": "assistant", "content": []block{
				{"type": "thinking", "thinking": "The handler rejects {empty} tokens; read it first.", "signature": strings.Repeat("EqQBCkYIBxgCKkA", 40)},
				text("Let me read the handler."),
				{"type": "tool_use", "id": "toolu_01", "name": "Read", "input": block{"file_path": "/src/auth.go"}},
			}},
			block{"role": "user", "content": []block{
				{"type": "tool_result", "tool_use_id": "toolu_01", "content": result},
			}},
		)
		size += 2*len(file) + 1_000
	}
	body, err := json.Marshal(struct {
		Messages  []block `json:"messages"`
		Model     string  `json:"model"`
		System    []block `json:"system"`
		Tools     []block `json:"tools"`
		Metadata  block   `json:"metadata"`
		MaxTokens int     `json:"max_tokens"`
		Stream    bool    `json:"stream"`
	}{
		Messages: messages,
		Model:    "auto",
		System: []block{
			text("You are Claude Code, Anthropic's official CLI for Claude."),
			{"type": "text", "text": strings.Repeat("Follow the repository conventions. ", 400), "cache_control": block{"type": "ephemeral"}},
		},
		Tools:     []block{{"name": "Read", "description": strings.Repeat("Reads a file. ", 200), "input_schema": block{"type": "object"}}},
		Metadata:  block{"user_id": "user_0123_account__session_4567"},
		MaxTokens: 32000,
		Stream:    true,
	})
	if err != nil {
		panic(err)
	}
	return body
})

func BenchmarkFindModel(b *testing.B) {
	body := claudeCode()
	b.SetBytes(int64(len(body)))
	b.ReportAllocs()
	for b.Loop() {
		if field, ok := FindModel(body); !ok || field.Value != "auto" {
			b.Fatalf("FindModel = %+v, %v", field, ok)
		}
	}
}

func BenchmarkAppendSystem(b *testing.B) {
	body := claudeCode()
	b.SetBytes(int64(len(body)))
	b.ReportAllocs()
	for b.Loop() {
		if _, ok := AppendSystem(body, "You are glm-5.3."); !ok {
			b.Fatal("AppendSystem failed")
		}
	}
}

func BenchmarkParse(b *testing.B) {
	body := claudeCode()
	b.SetBytes(int64(len(body)))
	b.ReportAllocs()
	for b.Loop() {
		if c := Parse(Messages, body); !c.HasImage || c.LatestUser == "" {
			b.Fatalf("Parse = %+v", c)
		}
	}
}

func TestParseSeesWhetherToolsAreOffered(t *testing.T) {
	cases := map[string]bool{
		`{"messages":[],"tools":[{"name":"Bash","input_schema":{}}]}`: true,
		`{"messages":[],"tools":[]}`:                                  false,
		`{"messages":[],"tools":null}`:                                false,
		`{"messages":[]}`:                                             false,
		`{"tools":[{"name":"Read"}],"messages":[],"tools":[]}`:        false,
	}
	for body, want := range cases {
		if got := Parse(Messages, []byte(body)).HasTools; got != want {
			t.Errorf("Parse(%s).HasTools = %v, want %v", body, got, want)
		}
	}
}
