package wire

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestScannerAgreesWithEncodingJSON(t *testing.T) {
	deep := func(n int) string { return strings.Repeat("[", n) + strings.Repeat("]", n) }
	for _, text := range []string{
		`""`, `"a\"b"`, `"a\\"`, `"a\\\"b"`, `"\\\\"`, `"\/\b\f\n\r\t"`, `"]}{["`, `"é 😀"`, "\"\xff\"",
		`"\u00e9"`, `"\uD83D\uDE00"`, `"\ud800"`, `"\u12"`, `"\u12g4"`, `"\x"`, `"\'"`, `"a`, `"a\"`, `"a\`,
		"\"tab\there\"", "\"new\nline\"", "\"\x00\"", "\"\x7f\"",
		`0`, `-0`, `7`, `-12.5e+3`, `1E9`, `0.5`, `01`, `-`, `+1`, `.5`, `1.`, `1e`, `1e+`, `0x1`, `1.5.2`, `--1`,
		`true`, `false`, `null`, `tru`, `nul`, `True`, `truex`, `nullnull`,
		`{}`, `[]`, `{ }`, "[\t\n\r ]", `{"a":1}`, `{"a":[1,{"b":"]"}]}`, `[1,[2,[3]]]`, `[{"a":"}"},"{"]`,
		`{"a":1,}`, `[1,]`, `{,}`, `[,1]`, `{"a"}`, `{"a":}`, `{"a" 1}`, `{"a":1 "b":2}`, `{1:2}`, `{a:1}`,
		`[}`, `{]`, `[1}`, `{"a":1]`, `[`, `{`, `{"a":1`, `[1,2`, `{"a":{"b":[1,2]}`, `]`, `}`,
		` {} `, "\n\t{}\r\n", `{} {}`, `{}x`, `[] ]`, "\f{}", "\v[]", "\u00a0[]", "\ufeff{}", ``, ` `,
		deep(maxDepth), deep(maxDepth + 1), `{"a":` + deep(maxDepth-1) + `}`, `{"a":` + deep(maxDepth) + `}`,
	} {
		s := scanner{data: []byte(text)}
		if got, want := s.value() && s.end(), json.Valid([]byte(text)); got != want {
			t.Errorf("scanning %.40q = %v, encoding/json says %v", text, got, want)
		}
	}
}

func TestMember(t *testing.T) {
	cases := []struct {
		body, name, value string
		ok                bool
	}{
		{`{"a":"x\"}\\","model":"m"}`, "model", `"m"`, true},
		{`{"a":"\\\"model\\\":1","model":"m"}`, "model", `"m"`, true},
		{`{"a":{"b":["]}{[",{"model":"inner"}]},"model":["[",{"}":"]"}]}`, "model", `["[",{"}":"]"}]`, true},
		{`{"mod\u0065l":"m\u00e9"}`, "model", `"m\u00e9"`, true},
		{`{"Model":"m"}`, "model", ``, true},
		{"\t{\n\"model\"\r:\t\"m\" ,\"a\":1}\n", "model", `"m"`, true},
		{`{"model":-1.5e3}`, "model", `-1.5e3`, true},
		{`{"model":"a","model":"b"}`, "model", `"a"`, true},
		{`{"model":"m"}x`, "model", `"m"`, true},
		{`{"model":"m", "a":`, "model", `"m"`, true},
		{`{"a":1} x`, "model", ``, false},
		{`{"a":1}{}`, "model", ``, false},
		{`{"a":"x`, "model", ``, false},
		{`{"a":1,`, "model", ``, false},
		{`{"model":"m`, "model", ``, false},
		{`{"model":`, "model", ``, false},
		{`{"a":01,"model":"m"}`, "model", ``, false},
		{`{"a":[}],"model":"m"}`, "model", ``, false},
		{`[{"model":"m"}]`, "model", ``, false},
		{`"model"`, "model", ``, false},
		{`null`, "model", ``, false},
		{``, "model", ``, false},
	}
	for _, c := range cases {
		start, end, found, ok := member([]byte(c.body), c.name)
		if value := c.body[start:end]; value != c.value || found != (c.value != "") || ok != c.ok {
			t.Errorf("member(%q, %q) = %q, %v, %v; want %q, ok %v", c.body, c.name, value, found, ok, c.value, c.ok)
		}
	}
}

func TestNamedFoldsCaseAndEscapes(t *testing.T) {
	for key, want := range map[string]bool{
		`"messages"`: true, `"MESSAGES"`: true, `"Messages"`: true, `"meſſages"`: true, `"m\u0065ssages"`: true,
		`"message"`: false, `"messages "`: false, `"messages\u0000"`: false, `""`: false,
	} {
		if got := named([]byte(key), "messages"); got != want {
			t.Errorf("named(%s) = %v, want %v", key, got, want)
		}
	}
}
