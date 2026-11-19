package config

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"strings"
)

// readSecrets adds the entries of a dotenv file (KEY=VALUE lines, #
// comments, optional export prefix and quotes) to secrets; later files win.
func readSecrets(path string, secrets map[string]string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read secrets file: %w", err)
	}
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for line := 1; scanner.Scan(); line++ {
		text := strings.TrimSpace(scanner.Text())
		if text == "" || strings.HasPrefix(text, "#") {
			continue
		}
		key, value, ok := strings.Cut(strings.TrimPrefix(text, "export "), "=")
		key = strings.TrimSpace(key)
		if !ok || key == "" {
			return fmt.Errorf("secrets file %s line %d is not KEY=VALUE", path, line)
		}
		value = strings.TrimSpace(value)
		if len(value) >= 2 && (value[0] == '"' || value[0] == '\'') && value[len(value)-1] == value[0] {
			value = value[1 : len(value)-1]
		}
		secrets[key] = value
	}
	return scanner.Err()
}
