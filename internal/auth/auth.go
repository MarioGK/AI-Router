// Package auth checks programmatic API keys and separate dashboard credentials.
package auth

import (
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"net/http"
	"strings"
)

// Checker compares presented credentials with the configured key.
type Checker struct {
	key         []byte
	fingerprint string
}

// New returns a checker for key.
func New(key string) *Checker {
	sum := sha256.Sum256([]byte(key))
	return &Checker{key: []byte(key), fingerprint: hex.EncodeToString(sum[:8])}
}

// Valid reports whether the request carries the key as a Bearer token or an
// x-api-key header (Anthropic clients).
func (c *Checker) Valid(header http.Header) bool {
	presented := Presented(header)
	return presented != "" && subtle.ConstantTimeCompare([]byte(presented), c.key) == 1
}

// Fingerprint identifies the key without revealing it; it scopes pin keys.
func (c *Checker) Fingerprint() string {
	return c.fingerprint
}

// Presented returns the credential a request carries, if any.
func Presented(header http.Header) string {
	if value := header.Get("Authorization"); len(value) > 7 && strings.EqualFold(value[:7], "bearer ") {
		return strings.TrimSpace(value[7:])
	}
	return strings.TrimSpace(header.Get("X-Api-Key"))
}
