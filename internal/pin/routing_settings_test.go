package pin

import (
	"context"
	"path/filepath"
	"reflect"
	"sync"
	"testing"

	"github.com/MarioGK/AI-Router/internal/decide"
)

func TestRoutingSettingsAreAtomicAndSurviveReopen(t *testing.T) {
	path := filepath.Join(t.TempDir(), "router.db")
	store := open(t, path)
	if settings, err := store.RoutingSettings(t.Context()); err != nil || settings != nil {
		t.Fatalf("initial settings = %+v, %v", settings, err)
	}
	want := RoutingSettings{Mode: "auto", FixedModel: "sol", Models: []string{"glm", "opus"}, JevEnabled: false,
		Preferences: decide.Preferences{Priority: decide.PrioritySpeed, Ranking: []string{"opus", "glm"}}}
	if err := store.SaveRoutingSettings(t.Context(), want); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store = open(t, path)
	got, err := store.RoutingSettings(t.Context())
	if err != nil || got == nil || !reflect.DeepEqual(*got, want) {
		t.Fatalf("settings after reopen = %+v, %v", got, err)
	}
	got.Models[0] = "mutated"
	got.Preferences.Ranking[0] = "mutated"
	again, err := store.RoutingSettings(t.Context())
	if err != nil || again == nil || !reflect.DeepEqual(*again, want) {
		t.Fatalf("caller mutation reached stored settings = %+v, %v", again, err)
	}
}

func TestRoutingSettingsConcurrentReadersSeeWholeRows(t *testing.T) {
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	a := RoutingSettings{Mode: "fixed", FixedModel: "sol", Models: []string{"glm"}, JevEnabled: false}
	b := RoutingSettings{Mode: "auto", FixedModel: "opus", Models: []string{"flash", "opus"}, JevEnabled: true}
	if err := store.SaveRoutingSettings(t.Context(), a); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for range 3 {
		wg.Go(func() {
			for range 30 {
				got, err := store.RoutingSettings(t.Context())
				if err != nil || got == nil || !reflect.DeepEqual(*got, a) && !reflect.DeepEqual(*got, b) {
					t.Errorf("partial settings snapshot = %+v, %v", got, err)
					return
				}
			}
		})
	}
	wg.Go(func() {
		for i := range 30 {
			settings := a
			if i%2 == 0 {
				settings = b
			}
			if err := store.SaveRoutingSettings(t.Context(), settings); err != nil {
				t.Error(err)
				return
			}
		}
	})
	wg.Wait()
}

func TestIndependentImageRequirementPreservesPinsAndGuardsTransitions(t *testing.T) {
	path := filepath.Join(t.TempDir(), "router.db")
	store := open(t, path)
	before, err := store.Create(t.Context(), Pin{Key: "k", Model: "vision", Level: "high"},
		Event{Key: "k", Kind: Decision, Model: "vision", Level: "high"})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.RequireImages(t.Context(), "k"); err != nil {
		t.Fatal(err)
	}
	if err := store.RequireImages(t.Context(), "fixed-only"); err != nil {
		t.Fatal(err)
	}
	after, err := store.Get(t.Context(), "k")
	if err != nil || !reflect.DeepEqual(after, before) {
		t.Fatalf("image observation changed automatic pin = %+v, %v", after, err)
	}
	if p, err := store.Get(t.Context(), "fixed-only"); err != nil || p != nil {
		t.Fatalf("image observation made automatic pin = %+v, %v", p, err)
	}
	for _, transition := range []string{"move", "failover"} {
		var p *Pin
		if transition == "move" {
			p, err = store.Move(t.Context(), "k", "vision", "text", "low", false,
				Event{Key: "k", Kind: Override, Model: "text", Previous: "vision"})
		} else {
			p, err = store.Failover(t.Context(), "k", "vision", "text", false,
				Event{Key: "k", Kind: Failover, Model: "text", Previous: "vision"})
		}
		if err != nil || !reflect.DeepEqual(p, before) {
			t.Fatalf("%s bypassed independent image guard = %+v, %v", transition, p, err)
		}
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store = open(t, path)
	for _, key := range []string{"k", "fixed-only"} {
		if required, err := store.ImagesRequired(t.Context(), key); err != nil || !required {
			t.Fatalf("requirement after reopen %q = %v, %v", key, required, err)
		}
	}
	if p, err := store.Move(t.Context(), "k", "vision", "other-vision", "high", true,
		Event{Key: "k", Kind: Migration, Model: "other-vision", Previous: "vision"}); err != nil || p.Model != "other-vision" {
		t.Fatalf("image guard rejected a compatible transition = %+v, %v", p, err)
	}
}

func TestRoutingSettingsAndCapabilityErrorsRemainVisible(t *testing.T) {
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err := store.SaveRoutingSettings(ctx, RoutingSettings{Mode: "fixed", FixedModel: "sol"}); err == nil {
		t.Fatal("canceled settings write succeeded")
	}
	if err := store.RequireImages(ctx, "k"); err == nil {
		t.Fatal("canceled image write succeeded")
	}
	if err := store.SaveRoutingSettings(t.Context(), RoutingSettings{Mode: "auto", JevEnabled: true}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.write.ExecContext(t.Context(), `UPDATE routing_settings SET settings = '{'`); err != nil {
		t.Fatal(err)
	}
	if _, err := store.RoutingSettings(t.Context()); err == nil {
		t.Fatal("malformed settings were accepted")
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := store.RoutingSettings(t.Context()); err == nil {
		t.Fatal("closed settings store was hidden")
	}
	if _, err := store.ImagesRequired(t.Context(), "k"); err == nil {
		t.Fatal("closed image requirement store was hidden")
	}
}
