package auth

import (
	"strings"
	"testing"

	"golang.org/x/crypto/bcrypt"
)

// These credentials are fixture data, unrelated to the operator password.
const fixturePassword = "fixture-dashboard-password"
const fixtureHash = "$2a$04$/30ZGeo.5yuV7h1jOMHe3eD9AAeEnWTPFaNstiRnB8Kj633R2P5ou"

func TestPasswordRequiresBothCredentials(t *testing.T) {
	checker, err := NewPassword("mariogk", fixtureHash)
	if err != nil {
		t.Fatal(err)
	}
	if !checker.Enabled() || !checker.Valid("mariogk", fixturePassword) {
		t.Fatal("the fixture credentials were rejected")
	}
	for _, credentials := range [][2]string{
		{"another-user", fixturePassword}, {"mariogk", "wrong-fixture-password"},
		{" mariogk", fixturePassword}, {"mariogk", ""}, {"", fixturePassword},
		{strings.Repeat("u", MaxUsernameBytes+1), fixturePassword},
		{"mariogk", strings.Repeat("p", MaxPasswordBytes+1)},
	} {
		if checker.Valid(credentials[0], credentials[1]) {
			t.Fatal("invalid fixture credentials were accepted")
		}
	}
}

func TestPasswordMissingHashFailsClosed(t *testing.T) {
	checker, err := NewPassword("mariogk", "")
	if err != nil {
		t.Fatal(err)
	}
	if checker.Enabled() || checker.Valid("mariogk", fixturePassword) || checker.Valid("mariogk", "upstream-fixture-key") {
		t.Fatal("a missing dashboard hash allowed login")
	}
	var absent *Password
	if absent.Enabled() || absent.Valid("mariogk", fixturePassword) {
		t.Fatal("a nil checker allowed login")
	}
}

func TestPasswordRejectsBcryptPrefixTruncation(t *testing.T) {
	password := strings.Repeat("p", MaxPasswordBytes)
	hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.MinCost)
	if err != nil {
		t.Fatal("could not generate fixture hash")
	}
	checker, err := NewPassword("mariogk", string(hash))
	if err != nil {
		t.Fatal(err)
	}
	if !checker.Valid("mariogk", password) || checker.Valid("mariogk", password+"extra") {
		t.Fatal("password input was rejected at the limit or silently truncated beyond it")
	}
}

func TestPasswordRejectsUnsafeConfigurationWithoutEcho(t *testing.T) {
	for _, hash := range []string{
		"plaintext-fixture-password", fixtureHash + "extra", fixtureHash[:len(fixtureHash)-1],
		strings.Replace(fixtureHash, "$2a$", "$2z$", 1), strings.Replace(fixtureHash, "$04$", "$31$", 1),
		strings.Replace(fixtureHash, "$04$", "$00$", 1), strings.Replace(fixtureHash, "/30", "!30", 1),
		fixtureHash[:len(fixtureHash)-1] + "A",
	} {
		if _, err := NewPassword("mariogk", hash); err == nil {
			t.Fatal("an unsafe password hash was accepted")
		} else if strings.Contains(err.Error(), hash) {
			t.Fatal("hash validation disclosed supplied credential contents")
		}
	}
	for _, username := range []string{"", " mariogk", "mariogk\n", strings.Repeat("u", MaxUsernameBytes+1), string([]byte{0xff})} {
		if _, err := NewPassword(username, fixtureHash); err == nil {
			t.Fatal("an invalid username was accepted")
		}
	}
}

func TestPasswordAcceptsSupportedBcryptVersions(t *testing.T) {
	for _, version := range []string{"2a", "2b", "2y"} {
		checker, err := NewPassword("mariogk", strings.Replace(fixtureHash, "2a", version, 1))
		if err != nil || !checker.Valid("mariogk", fixturePassword) {
			t.Fatalf("bcrypt version %s was rejected", version)
		}
	}
}

func TestHashPasswordProducesConfiguredHash(t *testing.T) {
	hash, err := HashPassword([]byte(fixturePassword))
	if err != nil {
		t.Fatal(err)
	}
	cost, err := bcrypt.Cost([]byte(hash))
	if err != nil || cost != bcrypt.DefaultCost {
		t.Fatal("hash generation did not use the configured cost")
	}
	checker, err := NewPassword("mariogk", hash)
	if err != nil || !checker.Valid("mariogk", fixturePassword) {
		t.Fatal("generated hash could not authenticate its fixture password")
	}
	for _, password := range [][]byte{nil, []byte(strings.Repeat("fixture-", 20))} {
		if _, err := HashPassword(password); err == nil {
			t.Fatal("hash generation accepted an empty or overlong password")
		} else if len(password) > 0 && strings.Contains(err.Error(), string(password)) {
			t.Fatal("hash generation disclosed supplied password contents")
		}
	}
}
