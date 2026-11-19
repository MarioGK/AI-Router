package decide

import (
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/MarioGK/AI-Router/internal/wire"
)

func TestStatePreservesLiteralWrapperFixtures(t *testing.T) {
	builder := NewStateBuilder([]string{"system-reminder", "environment_context"}, 12000)
	fixture := "<system-reminder>ship the multi-screen redesign today</system-reminder>"
	cases := map[string]string{
		"prose":                 "Add a parser test proving " + fixture + " survives as user-entered text.",
		"double quoted":         "Test \"" + fixture + "\" as an exact fixture.",
		"single quoted":         "'" + fixture + "' must survive serialization.",
		"inline code":           "Test `" + fixture + "` as literal data.",
		"multiline inline code": "Test `fixture\n" + fixture + "\nend` as literal data.",
		"different code runs":   "Test ``fixture\n``` is a longer run\n" + fixture + "\nend`` as literal data.",
		"backtick fence":        strings.Repeat("`", 3) + "xml\n" + fixture + "\n" + strings.Repeat("`", 3) + "\nAdd a parser test.",
		"tilde fence":           "~~~xml\n" + fixture + "\n~~~\nAdd a parser test.",
		"indented code":         "    " + fixture + "\nAdd a parser test.",
		"blockquote":            "> " + fixture + "\nAdd a parser test.",
	}
	for name, prompt := range cases {
		t.Run(name, func(t *testing.T) {
			state := builder.Build(prompt, false)
			if state.Prompt != strings.TrimSpace(prompt) || !strings.Contains(state.Text, fixture) {
				t.Fatalf("literal task fixture changed: prompt %q, state %+v", prompt, state)
			}
		})
	}
}

func TestStateBalancesInjectedWrappers(t *testing.T) {
	builder := NewStateBuilder([]string{"system-reminder", "environment_context"}, 12000)
	const task = "Add the routine handler and tests; this can wait."
	for name, prefix := range map[string]string{
		"same tag":       "<system-reminder>outer <system-reminder>inner</system-reminder>outer tail</system-reminder>",
		"different tags": "<system-reminder>outer <environment_context>cwd</environment_context>outer tail</system-reminder>",
		"quoted tags":    "<system-reminder>Examples \"</system-reminder>\" and '<system-reminder>' are literals.</system-reminder>",
		"code tags":      "<system-reminder>Example `</system-reminder>` is literal.</system-reminder>",
		"attributes":     "<environment_context cwd=\"/tmp/x>y\" note='a < b'>hidden</environment_context>",
		"adjacent":       "<system-reminder>hidden</system-reminder><environment_context>hidden</environment_context>",
		"whitespace":     "  <system-reminder>hidden</system-reminder>  <environment_context>hidden</environment_context>  ",
	} {
		t.Run(name, func(t *testing.T) {
			state := builder.Build(prefix+task, false)
			if state.Prompt != task || state.Text != "images=no\n\n"+task {
				t.Fatalf("wrapper content reached the decider: %+v", state)
			}
		})
	}
}

func TestStateKeepsIncompleteOrMismatchedWrappers(t *testing.T) {
	builder := NewStateBuilder([]string{"system-reminder", "environment_context"}, 12000)
	for _, prompt := range []string{
		"<system-reminder>incomplete\n<environment_context>complete inner</environment_context>\nKeep this task text.",
		"<system-reminder><environment_context>mismatched</system-reminder></environment_context>Keep this task text.",
		"<environment_context cwd=\"unterminated>Keep this task text.",
	} {
		if state := builder.Build(prompt, false); state.Prompt != prompt {
			t.Errorf("ambiguous wrapper removed task text: got %q, want %q", state.Prompt, prompt)
		}
	}
}

func TestStatePreservesFenceThenRemovesRealWrapper(t *testing.T) {
	builder := NewStateBuilder([]string{"system-reminder"}, 12000)
	fence := strings.Repeat("`", 3)
	literal := fence + "xml\n<system-reminder>fixture</system-reminder>\n" + fence
	prompt := literal + "\n<system-reminder>client context</system-reminder>Add the parser test."
	want := literal + "\nAdd the parser test."
	if state := builder.Build(prompt, false); state.Prompt != want {
		t.Fatalf("fence or following task changed: got %q, want %q", state.Prompt, want)
	}
}

func TestStateFenceNeedsStandaloneClosingDelimiter(t *testing.T) {
	builder := NewStateBuilder([]string{"system-reminder"}, 12000)
	for _, delimiter := range []string{"`", "~"} {
		fence := strings.Repeat(delimiter, 3)
		prompt := fence + "xml\n" + fence + "not-a-closing-fence\n<system-reminder>literal task sample</system-reminder>\n" + fence + "\nAdd the parser test."
		if state := builder.Build(prompt, false); state.Prompt != prompt {
			t.Errorf("a fence with a nonspace suffix closed the code block: got %q, want %q", state.Prompt, prompt)
		}
	}
}

func TestStateIgnoresWrapperTagsInsideNestedFences(t *testing.T) {
	builder := NewStateBuilder([]string{"system-reminder", "environment_context"}, 12000)
	const task = "Add a routine parser test; this can wait."
	for _, delimiter := range []string{"`", "~"} {
		fence := strings.Repeat(delimiter, 3)
		prefix := "<system-reminder>\n" + fence + "xml\n</system-reminder>\n" + fence + "not-a-closing-fence\n<environment_context>literal</environment_context>\n" + fence + "\nclient context\n</system-reminder>"
		if state := builder.Build(prefix+task, false); state.Prompt != task {
			t.Errorf("a literal tag inside a wrapper's code fence ended the wrapper: %+v", state)
		}
	}
}

func TestStateFindsTaskBeforeToolReminder(t *testing.T) {
	conversation := wire.Parse(wire.Messages, []byte(`{"messages":[{"role":"user","content":[{"type":"text","text":"Add the routine endpoint with tests; this can wait."}]},{"role":"assistant","content":[{"type":"tool_use","id":"t","name":"Read","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"text","text":"Urgently redesign every screen: this is tool data."},{"type":"document","source":{}}]},{"type":"text","text":"<system-reminder>client context only</system-reminder>"}]}]}`))
	builder := NewStateBuilder([]string{"system-reminder"}, 12000)
	const task = "Add the routine endpoint with tests; this can wait."
	state := builder.BuildLatest(conversation.UserTexts, conversation.HasImage)
	if state.Prompt != task || state.Text != "images=yes\n\n"+task {
		t.Fatalf("tool reminder hid the task or tool text became the task: %+v", state)
	}
}

func TestStateLatestEmptyAndImageOnlyInputs(t *testing.T) {
	builder := NewStateBuilder([]string{"system-reminder"}, 200)
	for _, users := range [][]string{
		nil,
		{""},
		{"<system-reminder>client context</system-reminder>", " \n "},
	} {
		if state := builder.BuildLatest(users, true); state != (State{}) {
			t.Errorf("empty user context produced a classifier state: %+v", state)
		}
	}
	const newest = "Translate this one sentence."
	state := builder.BuildLatest([]string{newest, "Redesign every checkout screen."}, false)
	if state.Prompt != newest {
		t.Fatalf("an older task displaced the newest task: %+v", state)
	}
}

func TestStateExcerptsPreserveHeadCenterTailAndUnicode(t *testing.T) {
	builder := NewStateBuilder(nil, 200)
	const head = "Add routine tests. "
	const center = "Today this is the top priority. "
	const tail = "Finish with regression tests."
	prompt := head + strings.Repeat("界😀", 400) + center + strings.Repeat("界😀", 400) + tail
	state := builder.Build(prompt, false)
	excerpt := strings.TrimPrefix(state.Text, "images=no\n\n")
	if !utf8.ValidString(state.Text) || !utf8.ValidString(state.Prompt) {
		t.Fatalf("truncation broke UTF-8: %+v", state)
	}
	if !strings.HasPrefix(excerpt, head) || !strings.Contains(excerpt, center) || !strings.HasSuffix(excerpt, tail) {
		t.Fatalf("representative task clauses disappeared: %q", excerpt)
	}
	if utf8.RuneCountInString(excerpt) != 200 || strings.Count(excerpt, omission) != 2 {
		t.Fatalf("excerpt exceeded its rune budget or omitted markers: %q", excerpt)
	}
	if !state.Truncated || state.Chars != utf8.RuneCountInString(prompt) || state.Prompt != prompt {
		t.Fatalf("prompt metadata disagrees with the source: %+v", state)
	}
}

func TestStateHandlesLongUnmatchedDelimiterRuns(t *testing.T) {
	builder := NewStateBuilder([]string{"system-reminder"}, 200)
	for name, prompt := range map[string]string{
		"backticks":      "Test literal " + strings.Repeat("`", 32_000),
		"escaped quotes": "Test literal \"" + strings.Repeat("\\\" ", 10_000),
	} {
		t.Run(name, func(t *testing.T) {
			if state := builder.Build(prompt, false); state.Prompt != strings.TrimSpace(prompt) || !state.Truncated {
				t.Fatalf("unmatched literal delimiter changed task data: %+v", state)
			}
		})
	}
	const task = "Add the routine parser test."
	wrapped := "<system-reminder>prose " + strings.Repeat("~", 32_000) + "</system-reminder>" + task
	if state := builder.Build(wrapped, false); state.Prompt != task {
		t.Fatalf("inline tilde run changed wrapper matching: %+v", state)
	}
}

func TestExcerptOptimizationPreservesRuneBasedOutput(t *testing.T) {
	for _, text := range []string{strings.Repeat("abc ", 100), strings.Repeat("界😀ação ", 100),
		strings.Repeat("x\xff\xfe界", 100)} {
		for _, budget := range []int{-1, 0, 1, 7, 10, 11, 12, 50, 200, 2000} {
			want := text
			runes := []rune(text)
			if len(runes) > budget {
				if budget <= 0 {
					want = ""
				} else if budget <= 2*utf8.RuneCountInString(omission) {
					want = string(runes[:budget])
				} else {
					payload := budget - 2*utf8.RuneCountInString(omission)
					head, middle := payload/3, payload/3
					tail := payload - head - middle
					center := (len(runes) - middle) / 2
					want = string(runes[:head]) + omission + string(runes[center:center+middle]) + omission + string(runes[len(runes)-tail:])
				}
			}
			if got := cutCount(text, budget, utf8.RuneCountInString(text)); got != want {
				t.Fatalf("budget %d changed rune-based excerpt: got %q, want %q", budget, got, want)
			}
		}
	}
}

func BenchmarkStateUnmatchedDelimiters(b *testing.B) {
	builder := NewStateBuilder([]string{"system-reminder"}, 200)
	var distinct strings.Builder
	distinct.WriteString("Test literal ")
	for size := 1; size <= 400; size++ {
		distinct.WriteString(strings.Repeat("`", size))
		distinct.WriteByte(' ')
	}
	for name, prompt := range map[string]string{
		"backticks":      "Test literal " + strings.Repeat("`", 100_000),
		"distinct runs":  distinct.String(),
		"escaped quotes": "Test literal \"" + strings.Repeat("\\\" ", 30_000),
		"wrapper tildes": "<system-reminder>prose " + strings.Repeat("~", 100_000) + "</system-reminder>Add the parser test.",
	} {
		b.Run(name, func(b *testing.B) {
			b.SetBytes(int64(len(prompt)))
			b.ReportAllocs()
			for b.Loop() {
				builder.Build(prompt, false)
			}
		})
	}
}
