package wire

import (
	"bytes"
	"encoding/json"
)

// ModelField is the location of the top-level "model" string in a request
// body. Only these bytes change when a request is routed; everything else
// (field order, unknown fields, whitespace) reaches CLIProxy untouched.
type ModelField struct {
	Value      string
	start, end int
}

// FindModel locates the top-level "model" string of a JSON object body.
func FindModel(body []byte) (ModelField, bool) {
	start, end, found, _ := member(body, "model")
	field := ModelField{start: start, end: end}
	if !found || json.Unmarshal(body[start:end], &field.Value) != nil {
		return ModelField{}, false
	}
	return field, true
}

// Replace returns a copy of body with the model value set to model.
func (f ModelField) Replace(body []byte, model string) []byte {
	return splice(body, f.start, f.end, quote(model))
}

// splice returns a copy of body with body[start:end] replaced by insert.
func splice(body []byte, start, end int, insert []byte) []byte {
	out := make([]byte, 0, len(body)-(end-start)+len(insert))
	out = append(out, body[:start]...)
	out = append(out, insert...)
	return append(out, body[end:]...)
}

// quote encodes s as a JSON string without HTML escaping.
func quote(s string) []byte {
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	_ = encoder.Encode(s)
	return bytes.TrimRight(buffer.Bytes(), "\n")
}
