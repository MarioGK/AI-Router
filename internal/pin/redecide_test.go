package pin

import (
	"path/filepath"
	"testing"
	"time"
)

func TestRedecideOnlyMovesTheColdPinItRead(t *testing.T) {
	ctx := t.Context()
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	now := time.UnixMilli(1_000_000)
	store.now = func() time.Time { return now }
	created, err := store.Create(ctx, Pin{Key: "agent", Model: "light", Level: "low", Parent: "session"},
		Event{Key: "agent", Kind: Decision, Model: "light"})
	if err != nil || created.Parent != "session" || !created.LastRequest.Equal(now) || created.Turns != 0 {
		t.Fatalf("Create = %+v, %v", created, err)
	}
	now = now.Add(time.Second)
	if err := store.Seen(ctx, "agent"); err != nil {
		t.Fatal(err)
	}
	now = now.Add(time.Second)
	if err := store.Compacted(ctx, "agent"); err != nil {
		t.Fatal(err)
	}
	cold, err := store.Get(ctx, "agent")
	if err != nil || !cold.Compacted || cold.Turns != 1 || !cold.LastRequest.Equal(now) {
		t.Fatalf("cold pin = %+v, %v", cold, err)
	}
	// A request the thread made since the pin was read warms its cache: the
	// stale redecision changes nothing and logs nothing.
	now = now.Add(time.Second)
	if err := store.Seen(ctx, "agent"); err != nil {
		t.Fatal(err)
	}
	p, err := store.Redecide(ctx, *cold, "strong", "high", false,
		Event{Key: "agent", Kind: Redecision, Model: "strong", Previous: "light"})
	if err != nil || p.Model != "light" || p.Turns != 2 {
		t.Fatalf("stale Redecide = %+v, %v", p, err)
	}
	cold = p
	now = now.Add(time.Minute)
	p, err = store.Redecide(ctx, *cold, "strong", "high", false,
		Event{Key: "agent", Kind: Redecision, Model: "strong", Previous: "light", Prompt: "new task", FirstPrompt: "new task"})
	if err != nil || p.Model != "strong" || p.Level != "high" || p.Decided != "light" || p.Turns != 0 ||
		p.Compacted || !p.LastRequest.Equal(now) {
		t.Fatalf("Redecide = %+v, %v", p, err)
	}
	detail, err := store.Thread(ctx, "agent")
	if err != nil || detail == nil || len(detail.Events) != 2 || detail.Events[1].Kind != Redecision ||
		detail.Parent != "session" {
		t.Fatalf("thread = %+v, %v", detail, err)
	}
	if overridden, err := store.Overridden(ctx, "agent"); err != nil || overridden {
		t.Fatalf("Overridden = %v, %v", overridden, err)
	}
	if pruned, err := store.PrunePrompts(ctx, now.Add(time.Hour)); err != nil || pruned != 1 {
		t.Fatalf("PrunePrompts = %d, %v; want the redecision's prompts thinned", pruned, err)
	}
}
