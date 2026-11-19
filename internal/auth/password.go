package auth

import (
	"crypto/sha256"
	"crypto/subtle"
	"fmt"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/crypto/bcrypt"
)

// Passwords longer than bcrypt's input limit must never be silently truncated.
const MaxPasswordBytes = 72

// MaxUsernameBytes bounds both configured and presented operator usernames.
const MaxUsernameBytes = 128

// maxPasswordCost limits work for a configured hash to keep login requests
// bounded. HashPassword uses bcrypt's default cost (10).
const maxPasswordCost = 14

var passwordHashFormat = regexp.MustCompile(`^\$2[aby]\$[0-9]{2}\$[./A-Za-z0-9]{21}[.Oeu][./A-Za-z0-9]{30}[.CGKOSWaeimquy26]$`)

// Password verifies the single configured dashboard account. It is independent
// of the programmatic upstream API key; an absent hash disables all logins.
type Password struct {
	username [sha256.Size]byte
	hash     []byte
}

// NewPassword validates the username and bcrypt hash without exposing either
// hash contents or credential input in its errors. An empty hash returns a
// disabled checker, allowing routing and benchmarks to run without a login.
func NewPassword(username, hash string) (*Password, error) {
	if !validUsername(username) {
		return nil, fmt.Errorf("dashboard username must be 1 to %d bytes without surrounding whitespace or control characters", MaxUsernameBytes)
	}
	checker := &Password{username: sha256.Sum256([]byte(username))}
	if hash == "" {
		return checker, nil
	}
	if err := ValidatePasswordHash(hash); err != nil {
		return nil, err
	}
	checker.hash = []byte(hash)
	return checker, nil
}

func validUsername(username string) bool {
	return username != "" && len(username) <= MaxUsernameBytes && utf8.ValidString(username) && strings.TrimSpace(username) == username &&
		!strings.ContainsFunc(username, unicode.IsControl)
}

// ValidatePasswordHash accepts only canonical bcrypt hashes with a bounded
// cost. Malformed or excessively expensive hashes are configuration errors.
func ValidatePasswordHash(hash string) error {
	if !passwordHashFormat.MatchString(hash) {
		return fmt.Errorf("dashboard password hash must be a valid bcrypt hash")
	}
	cost, err := bcrypt.Cost([]byte(hash))
	if err != nil || cost < bcrypt.MinCost || cost > maxPasswordCost {
		return fmt.Errorf("dashboard password hash bcrypt cost must be between %d and %d", bcrypt.MinCost, maxPasswordCost)
	}
	return nil
}

// Enabled reports whether a dashboard password hash was configured.
func (p *Password) Enabled() bool { return p != nil && len(p.hash) != 0 }

// Valid checks both credentials. Password verification also runs for a wrong
// username so account-name guesses do not bypass the bcrypt work.
func (p *Password) Valid(username, password string) bool {
	if !p.Enabled() || !validUsername(username) || password == "" || len(password) > MaxPasswordBytes {
		return false
	}
	usernameSum := sha256.Sum256([]byte(username))
	usernameOK := subtle.ConstantTimeCompare(usernameSum[:], p.username[:]) == 1
	secret := []byte(password)
	defer clear(secret)
	return bcrypt.CompareHashAndPassword(p.hash, secret) == nil && usernameOK
}

// HashPassword creates a bcrypt hash suitable for the secret loader. Errors
// contain only input bounds and never repeat the supplied password.
func HashPassword(password []byte) (string, error) {
	if len(password) == 0 || len(password) > MaxPasswordBytes {
		return "", fmt.Errorf("password must be 1 to %d bytes", MaxPasswordBytes)
	}
	hash, err := bcrypt.GenerateFromPassword(password, bcrypt.DefaultCost)
	if err != nil {
		return "", fmt.Errorf("could not hash password")
	}
	return string(hash), nil
}
