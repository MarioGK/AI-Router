"""Credential and policy boundaries for deployment-derived comparisons."""

import copy
import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location("compare_config", Path(__file__).with_name("compare-config.py"))
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


class ComparisonConfigTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "upstream": {"base_url": "https://proxy.example", "api_key_env": "DEPLOYED_PROXY_KEY"},
            "secrets_files": ["/run/secrets/router.env"],
            "dashboard": {"key_env": "RETIRED_DASHBOARD_KEY"},
            "auto": {"default_model": "retired-model", "candidates": [{"id": "retired-model"}]},
            "limits": {"retired-model": {"concurrency": 1}},
            "deciders": [{"name": "jev", "type": "systemone", "model": "jev-latest",
                          "url": "https://jev.example/v1/systemone", "api_key_env": "DEPLOYED_JEV_KEY"}],
        }

    def test_reviewed_policy_preserves_deployed_access(self):
        original = copy.deepcopy(self.base)
        policy = {
            "auto": {"default_model": "gpt-6.1-sol", "candidates": [{"id": "gpt-6.1-sol"}]},
            "prices": {"gpt-6.1-sol": {"input": 1}},
            "limits": {"gpt-6.1-sol": {"concurrency": 2}},
            "upstream": {"base_url": "https://wrong.example"},
            "secrets_files": ["/wrong/example.env"],
            "deciders": [{"name": "wrong"}],
        }
        original_policy = copy.deepcopy(policy)
        result = COMPARE.comparison_config(self.base, policy=policy)
        self.assertEqual(result["auto"]["default_model"], "gpt-6.1-sol")
        self.assertEqual(result["auto"]["decide_timeout"], "120s")
        self.assertEqual(result["prices"], policy["prices"])
        self.assertEqual(result["limits"], policy["limits"])
        self.assertEqual(result["upstream"], original["upstream"])
        self.assertEqual(result["secrets_files"], original["secrets_files"])
        self.assertEqual(result["deciders"][0]["api_key_env"], "DEPLOYED_JEV_KEY")
        self.assertEqual(result["deciders"][0]["url"], "https://jev.example/v1/systemone")
        for judge in result["deciders"][1:]:
            self.assertEqual(judge["url"], "https://proxy.example/v1/chat/completions")
            self.assertEqual(judge["api_key_env"], "DEPLOYED_PROXY_KEY")
        self.assertEqual([judge["model"] for judge in result["deciders"]],
                         ["jev-latest", "gpt-6-luna", "gpt-6.1-sol(high)"])
        self.assertNotIn("dashboard", result)
        self.assertEqual(self.base, original)
        self.assertEqual(policy, original_policy)

    def test_policy_removes_absent_retired_limits_and_prices(self):
        result = COMPARE.comparison_config(self.base, policy={"auto": {}})
        self.assertEqual(result["limits"], {})
        self.assertEqual(result["prices"], {})

    def test_policy_retention_is_coherent_with_its_learning_window(self):
        self.base["log_retention"] = "24h"
        self.base["auto"]["cost"] = {"window": "24h"}
        policy = {"auto": {"cost": {"window": "720h"}}, "log_retention": "720h"}
        result = COMPARE.comparison_config(self.base, policy=policy)
        self.assertEqual(result["log_retention"], "720h")
        self.assertEqual(result["auto"]["cost"]["window"], "720h")
        del policy["log_retention"]
        result = COMPARE.comparison_config(self.base, policy=policy)
        self.assertNotIn("log_retention", result)

    def test_invalid_policy_fails_before_output(self):
        for policy in ([], {}, {"auto": []}, {"auto": {}, "prices": []}, {"auto": {}, "limits": []}):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                COMPARE.comparison_config(self.base, policy=policy)

    def test_deployment_inline_credentials_are_rejected(self):
        for field in ("password", "api_key", "token"):
            base = copy.deepcopy(self.base)
            base["dashboard"][field] = "test-secret"
            with self.subTest(field=field), self.assertRaises(ValueError):
                COMPARE.comparison_config(base)

    def test_multiple_jev_deciders_need_explicit_selection(self):
        base = copy.deepcopy(self.base)
        alternate = dict(base["deciders"][0], name="jev-other")
        base["deciders"].append(alternate)
        with self.assertRaises(ValueError):
            COMPARE.comparison_config(base)
        result = COMPARE.comparison_config(base, jev_name="jev-other")
        self.assertEqual(result["deciders"][0]["name"], "jev-other")


if __name__ == "__main__":
    unittest.main()
