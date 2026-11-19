package main

import (
	"bytes"
	"fmt"
	"io"
	"os"

	"github.com/mattn/go-isatty"

	"github.com/MarioGK/AI-Router/internal/auth"
)

// hashPassword accepts a single password through a pipe, never a command
// argument or an echoing terminal. The hash is intentionally written only to
// stdout so an operator can redirect it to the secret materializer.
func hashPassword(stdin io.Reader, stdout, stderr io.Writer) int {
	if file, ok := stdin.(*os.File); ok && (isatty.IsTerminal(file.Fd()) || isatty.IsCygwinTerminal(file.Fd())) {
		fmt.Fprintln(stderr, "ai-router: hash-password needs a non-terminal input; pipe a password from a non-echoing reader")
		return 2
	}
	secret, err := io.ReadAll(io.LimitReader(stdin, auth.MaxPasswordBytes+3))
	if err != nil {
		fmt.Fprintln(stderr, "ai-router: could not read password input")
		return 1
	}
	defer clear(secret)
	password := secret
	if trimmed, ok := bytes.CutSuffix(password, []byte("\n")); ok {
		password, _ = bytes.CutSuffix(trimmed, []byte("\r"))
	}
	if bytes.ContainsAny(password, "\r\n") {
		fmt.Fprintln(stderr, "ai-router: hash-password needs exactly one password line")
		return 2
	}
	hash, err := auth.HashPassword(password)
	if err != nil {
		fmt.Fprintf(stderr, "ai-router: %v\n", err)
		return 2
	}
	if _, err := fmt.Fprintln(stdout, hash); err != nil {
		fmt.Fprintln(stderr, "ai-router: could not write password hash")
		return 1
	}
	return 0
}
