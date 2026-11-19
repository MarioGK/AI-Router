package main

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/MarioGK/AI-Router/internal/auth"
)

func TestHashPasswordReadsOnlyStdinAndPreservesSpaces(t *testing.T) {
	const fixture = " fixture password with spaces "
	var stdout, stderr bytes.Buffer
	if code := hashPassword(strings.NewReader(fixture+"\r\n"), &stdout, &stderr); code != 0 {
		t.Fatalf("hash-password exited %d: %s", code, stderr.String())
	}
	checker, err := auth.NewPassword("mariogk", strings.TrimSuffix(stdout.String(), "\n"))
	if err != nil || !checker.Valid("mariogk", fixture) || checker.Valid("mariogk", strings.TrimSpace(fixture)) {
		t.Fatal("stdin hashing changed the password or produced an unusable hash")
	}
	if strings.Contains(stdout.String(), fixture) || stderr.Len() != 0 {
		t.Fatal("hash-password emitted plaintext or unexpected diagnostics")
	}
}

func TestHashPasswordRejectsArgumentsWithoutEcho(t *testing.T) {
	for _, arguments := range [][]string{
		{"fixture-sensitive-argument"}, {"--password=fixture-sensitive-argument"},
		{"--parallel=fixture-sensitive-argument"}, {"--config", "fixture-sensitive-argument"},
	} {
		var stdout, stderr bytes.Buffer
		if code := run(append([]string{"hash-password"}, arguments...), &stdout, &stderr); code != 2 {
			t.Fatalf("hash-password accepted command-line input, exit %d", code)
		}
		if stdout.Len() != 0 || strings.Contains(stderr.String(), "fixture-sensitive-argument") || !strings.Contains(stderr.String(), "only through standard input") {
			t.Fatal("hash-password echoed an argument or accepted shared flags")
		}
	}
}

func TestHashPasswordBoundsAndSingleLine(t *testing.T) {
	for _, input := range []string{"", "\n", "\r\n", "first\nsecond", strings.Repeat("p", auth.MaxPasswordBytes+1), strings.Repeat("p", auth.MaxPasswordBytes+2) + "\r\n"} {
		var stdout, stderr bytes.Buffer
		if code := hashPassword(strings.NewReader(input), &stdout, &stderr); code != 2 || stdout.Len() != 0 {
			t.Fatal("hash-password accepted empty, multiline or overlong input")
		}
		if input != "" && len(input) > 2 && strings.Contains(stderr.String(), input) {
			t.Fatal("hash-password disclosed rejected plaintext input")
		}
	}
	fixture := strings.Repeat("p", auth.MaxPasswordBytes)
	var stdout, stderr bytes.Buffer
	if code := hashPassword(strings.NewReader(fixture+"\r\n"), &stdout, &stderr); code != 0 {
		t.Fatalf("hash-password rejected the exact byte limit: %s", stderr.String())
	}
	checker, err := auth.NewPassword("mariogk", strings.TrimSuffix(stdout.String(), "\n"))
	if err != nil || !checker.Valid("mariogk", fixture) || checker.Valid("mariogk", fixture+"suffix") {
		t.Fatal("hash-password silently truncated at the bcrypt input limit")
	}
}

type passwordReaderError struct{}

func (passwordReaderError) Read([]byte) (int, error) {
	return 0, errors.New("fixture-sensitive-reader-error")
}

type passwordWriterError struct{}

func (passwordWriterError) Write([]byte) (int, error) {
	return 0, errors.New("fixture-sensitive-writer-error")
}

func TestHashPasswordErrorsDoNotEchoInputOrIOErrors(t *testing.T) {
	var stderr bytes.Buffer
	if code := hashPassword(passwordReaderError{}, io.Discard, &stderr); code != 1 || strings.Contains(stderr.String(), "fixture-sensitive") {
		t.Fatal("hash-password exposed a reader error")
	}
	stderr.Reset()
	if code := hashPassword(strings.NewReader("fixture-password"), passwordWriterError{}, &stderr); code != 1 || strings.Contains(stderr.String(), "fixture-sensitive") || strings.Contains(stderr.String(), "fixture-password") {
		t.Fatal("hash-password exposed a writer error or plaintext input")
	}
}
