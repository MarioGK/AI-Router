package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/MarioGK/AI-Router/internal/auth"
)

const dashboardFixtureHash = "$2a$04$/30ZGeo.5yuV7h1jOMHe3eD9AAeEnWTPFaNstiRnB8Kj633R2P5ou"

func dashboardFixtureEnv(t *testing.T) {
	t.Helper()
	t.Setenv("CLIPROXYAPI_API_KEY", "upstream-fixture-key")
	t.Setenv("JEV_API_KEY", "jev-fixture-key")
	t.Setenv("TS_ROUTER_DASHBOARD_PASSWORD_HASH", "")
}

func TestDashboardDefaultsDisableLoginWithoutHash(t *testing.T) {
	dashboardFixtureEnv(t)
	cfg, err := Parse([]byte(minimal))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Dashboard.Username != "mariogk" || cfg.Dashboard.PasswordHashEnv != "TS_ROUTER_DASHBOARD_PASSWORD_HASH" || cfg.DashboardPasswordHash() != "" {
		t.Fatal("dashboard defaults are incorrect")
	}
	checker, err := auth.NewPassword(cfg.Dashboard.Username, cfg.DashboardPasswordHash())
	if err != nil || checker.Enabled() || checker.Valid("mariogk", cfg.UpstreamKey()) {
		t.Fatal("dashboard login fell back to a programmatic API key")
	}
}

func TestDashboardPasswordHashUsesIndependentSecret(t *testing.T) {
	dashboardFixtureEnv(t)
	t.Setenv("TS_ROUTER_DASHBOARD_PASSWORD_HASH", dashboardFixtureHash)
	cfg, err := Parse([]byte(minimal))
	if err != nil {
		t.Fatal(err)
	}
	checker, err := auth.NewPassword(cfg.Dashboard.Username, cfg.DashboardPasswordHash())
	if err != nil || !checker.Valid("mariogk", "fixture-dashboard-password") || checker.Valid("mariogk", cfg.UpstreamKey()) {
		t.Fatal("dashboard credentials are not independent of the upstream key")
	}
}

func TestDashboardPasswordHashSecretFilesWin(t *testing.T) {
	dashboardFixtureEnv(t)
	t.Setenv("OPERATOR_PASSWORD_HASH", "invalid-environment-fixture")
	secretPath := filepath.Join(t.TempDir(), "dashboard.env")
	if err := os.WriteFile(secretPath, []byte("OPERATOR_PASSWORD_HASH='"+dashboardFixtureHash+"'\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	document := "dashboard: {username: operator, password_hash_env: OPERATOR_PASSWORD_HASH}\nsecrets_files: [" + secretPath + "]\n" + minimal
	cfg, err := Parse([]byte(document))
	if err != nil {
		t.Fatal(err)
	}
	checker, err := auth.NewPassword(cfg.Dashboard.Username, cfg.DashboardPasswordHash())
	if err != nil || !checker.Valid("operator", "fixture-dashboard-password") || checker.Valid("mariogk", "fixture-dashboard-password") {
		t.Fatal("dashboard username or secret-file precedence was incorrect")
	}
}

func TestDashboardRejectsMalformedCredentialConfiguration(t *testing.T) {
	dashboardFixtureEnv(t)
	for _, hash := range []string{"plaintext-fixture-password", dashboardFixtureHash + "extra", strings.Replace(dashboardFixtureHash, "$04$", "$31$", 1)} {
		t.Setenv("TS_ROUTER_DASHBOARD_PASSWORD_HASH", hash)
		if _, err := Parse([]byte(minimal)); err == nil {
			t.Fatal("malformed dashboard hash was accepted")
		} else if strings.Contains(err.Error(), hash) {
			t.Fatal("configuration validation disclosed credential input")
		}
	}
	t.Setenv("TS_ROUTER_DASHBOARD_PASSWORD_HASH", "")
	for _, dashboard := range []string{
		"{username: ' admin'}", "{username: \"admin\\nname\"}", "{username: '" + strings.Repeat("u", auth.MaxUsernameBytes+1) + "'}",
		"{password_hash_env: 'ENV WITH SPACE'}", "{password_hash_env: 'KEY=value'}",
	} {
		if _, err := Parse([]byte("dashboard: " + dashboard + "\n" + minimal)); err == nil {
			t.Fatal("invalid dashboard credential configuration was accepted")
		}
	}
}
