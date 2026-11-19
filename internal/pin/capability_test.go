package pin

import (
	"path/filepath"
	"testing"
)

func TestCreateKeepsInitialCapabilityRequirement(t *testing.T) {
	ctx := t.Context()
	path := filepath.Join(t.TempDir(), "router.db")
	store := open(t, path)
	p, err := store.Create(ctx, Pin{Key: "image-thread", Model: "vision", Upgraded: true},
		Event{Key: "image-thread", Kind: Decision, Model: "vision"})
	if err != nil || p == nil || !p.Upgraded || p.FailedOver {
		t.Fatalf("Create = %+v, %v", p, err)
	}
	p, err = store.Create(ctx, Pin{Key: "image-thread", Model: "text"},
		Event{Key: "image-thread", Kind: Decision, Model: "text"})
	if err != nil || p == nil || !p.Upgraded || p.Model != "vision" {
		t.Fatalf("second Create = %+v, %v", p, err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store = open(t, path)
	p, err = store.Get(ctx, "image-thread")
	if err != nil || p == nil || !p.Upgraded || p.Model != "vision" {
		t.Fatalf("Get after reopen = %+v, %v", p, err)
	}
	p, err = store.Move(ctx, "image-thread", "vision", "other-vision", "", true,
		Event{Key: "image-thread", Kind: Migration, Model: "other-vision", Previous: "vision"})
	if err != nil || p == nil || !p.Upgraded || p.Model != "other-vision" {
		t.Fatalf("Move = %+v, %v", p, err)
	}
	p, err = store.Upgrade(ctx, "image-thread", "other-vision", "unexpected",
		Event{Key: "image-thread", Kind: CapabilityUpgrade, Model: "unexpected", Previous: "other-vision"})
	if err != nil || p == nil || !p.Upgraded || p.Model != "other-vision" {
		t.Fatalf("second capability upgrade = %+v, %v", p, err)
	}
	text, err := store.Create(ctx, Pin{Key: "text-thread", Model: "text"},
		Event{Key: "text-thread", Kind: Decision, Model: "text"})
	if err != nil || text == nil || text.Upgraded {
		t.Fatalf("text Create = %+v, %v", text, err)
	}
	text, err = store.Move(ctx, "text-thread", "text", "other-text", "", false,
		Event{Key: "text-thread", Kind: Migration, Model: "other-text", Previous: "text"})
	if err != nil || text == nil || text.Upgraded || text.Model != "other-text" {
		t.Fatalf("text Move = %+v, %v", text, err)
	}
}

func TestImageRequirementRejectsStaleTextTransitions(t *testing.T) {
	for _, transition := range []string{"override", "failover"} {
		t.Run(transition, func(t *testing.T) {
			ctx := t.Context()
			store := open(t, filepath.Join(t.TempDir(), "router.db"))
			stale, err := store.Create(ctx, Pin{Key: "k", Model: "vision", Level: "high"},
				Event{Key: "k", Kind: Decision, Model: "vision"})
			if err != nil {
				t.Fatal(err)
			}
			marked, err := store.Upgrade(ctx, "k", "vision", "vision",
				Event{Key: "k", Kind: CapabilityUpgrade, Model: "vision", Previous: "vision"})
			if err != nil || marked == nil || !marked.Upgraded || stale.Upgraded {
				t.Fatalf("image observation = %+v, %v; stale pin = %+v", marked, err, stale)
			}
			change := func(model string, supportsImages bool) (*Pin, error) {
				if transition == "override" {
					return store.Move(ctx, "k", stale.Model, model, "low", supportsImages,
						Event{Key: "k", Kind: Override, Model: model, Previous: stale.Model})
				}
				return store.Failover(ctx, "k", stale.Model, model, supportsImages,
					Event{Key: "k", Kind: Failover, Model: model, Previous: stale.Model})
			}
			p, err := change("text", false)
			if err != nil || p == nil || p.Model != "vision" || !p.Upgraded || p.FailedOver || p.Level != "high" {
				t.Fatalf("stale text transition = %+v, %v", p, err)
			}
			detail, err := store.Thread(ctx, "k")
			if err != nil || detail == nil || len(detail.Events) != 2 {
				t.Fatalf("rejected transition was logged: %+v, %v", detail, err)
			}
			p, err = change("other-vision", true)
			if err != nil || p == nil || p.Model != "other-vision" || !p.Upgraded || p.FailedOver != (transition == "failover") {
				t.Fatalf("vision transition = %+v, %v", p, err)
			}
		})
	}
}
