package pin

import (
	"errors"
	"strings"
	"unicode"
	"unicode/utf8"
)

// MaxSearchChars bounds dashboard prompt searches.
const MaxSearchChars = 256

const promptSearchMigration = `
CREATE VIRTUAL TABLE prompt_search USING fts5(
	prompt, first_prompt, content='events', content_rowid='id',
	tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER prompt_search_insert AFTER INSERT ON events BEGIN
	INSERT INTO prompt_search(rowid, prompt, first_prompt) VALUES (new.id, new.prompt, new.first_prompt);
END;
CREATE TRIGGER prompt_search_delete AFTER DELETE ON events BEGIN
	INSERT INTO prompt_search(prompt_search, rowid, prompt, first_prompt)
		VALUES ('delete', old.id, old.prompt, old.first_prompt);
END;
CREATE TRIGGER prompt_search_update AFTER UPDATE OF prompt, first_prompt ON events BEGIN
	INSERT INTO prompt_search(prompt_search, rowid, prompt, first_prompt)
		VALUES ('delete', old.id, old.prompt, old.first_prompt);
	INSERT INTO prompt_search(rowid, prompt, first_prompt) VALUES (new.id, new.prompt, new.first_prompt);
END;
INSERT INTO prompt_search(prompt_search) VALUES ('rebuild');
`

// promptMatch treats the input as literal words, all required, never as
// FTS syntax. SQLite matches case and accents without scanning prompts.
func promptMatch(search string) (string, error) {
	if utf8.RuneCountInString(search) > MaxSearchChars {
		return "", errors.New("prompt search must be at most 256 characters")
	}
	words := strings.FieldsFunc(search, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r) && !unicode.IsMark(r)
	})
	for i, word := range words {
		words[i] = `"` + word + `"`
	}
	return strings.Join(words, " AND "), nil
}

func (c *conditions) search(column, search string) error {
	if strings.TrimSpace(search) == "" {
		return nil
	}
	match, err := promptMatch(search)
	if err != nil {
		return err
	}
	if match == "" {
		c.add("0")
		return nil
	}
	c.add(column+` IN (SELECT rowid FROM prompt_search WHERE prompt_search MATCH ?)`, match)
	return nil
}
