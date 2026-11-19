import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("teacher_cases", Path(__file__).with_name("teacher-cases.py"))
teacher_cases = importlib.util.module_from_spec(spec)
spec.loader.exec_module(teacher_cases)


def report_fixture():
    return {"schema_version": 2, "deciders": ["jev", "sol61"], "cases": [], "inputs": [], "outcomes": [],
            "policies": {"policy": {"deciders": [{"name": "sol61", "model": "gpt-6.1-sol", "kind": "llm"}],
                                   "reasoning": {"levels": [{"id": "low"}, {"id": "high"}]}}}}


def add_trial(report, name, trial, choice="cheap", level="low", **changes):
    report["cases"].append({"name": name, "trial": trial, "group": "routine", "expect": ["hard"],
                            "reasoning": ["high"], "images": True, "offered": [],
                            "prompt": "Complete the original request\nwith image context: ação"})
    report["inputs"].append({"state": "prepared state", "candidates": [{"id": "cheap"}, {"id": "hard"}]})
    teacher = {"started": True, "cancelled": False, "choice": choice, "task_depth": level, "model": "hard", "policy": "policy",
               "attempts": [{"outcome": "doubted", "choice": choice, "level": level, "model": "gpt-6.1-sol", "digest": f"request-{name}-{trial}",
                             "probabilities": {"cheap": 0.8, "hard": 0.2}}]}
    teacher.update(changes)
    report["outcomes"].append([{"started": True, "cancelled": False, "choice": "cheap", "task_depth": "low", "model": "cheap"}, teacher])


class TeacherFreezeTests(unittest.TestCase):
    def test_freeze_uses_stable_primary_choices_and_preserves_provenance(self):
        report = report_fixture()
        add_trial(report, "stable/task", 1)
        add_trial(report, "stable/task", 2)
        original = copy.deepcopy(report)
        cases, provenance = teacher_cases.freeze(report, "sol61", "report-hash")
        self.assertEqual(report, original)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["expect"], ["cheap"])
        self.assertEqual(cases[0]["reasoning"], ["low"])
        self.assertEqual(cases[0]["prompt"], original["cases"][0]["prompt"])
        self.assertTrue(cases[0]["images"])
        self.assertEqual(cases[0]["offered"], ["cheap", "hard"])
        self.assertEqual(provenance["report_sha256"], "report-hash")
        self.assertFalse(provenance["canonical_labels_modified"])
        self.assertEqual(provenance["evaluation_scores"]["cli_score"], "initial")
        source = provenance["sources"][0]
        self.assertEqual(source["trials"], [1, 2])
        self.assertEqual(source["canonical_expect"], ["hard"])
        self.assertEqual(source["canonical_reasoning"], ["high"])
        self.assertEqual(source["configured_teacher_models"], ["gpt-6.1-sol"])
        self.assertEqual(source["observed_teacher_models"], ["gpt-6.1-sol"])
        self.assertEqual(len(source["policy_sha256"]["policy"]), 64)
        self.assertEqual(len(source["prepared_state_sha256"][0]), 64)
        self.assertEqual(len(source["request_digests"]), 2)

    def test_freeze_excludes_any_failed_missing_cancelled_or_unstable_repeat(self):
        report = report_fixture()
        add_trial(report, "good", 1)
        add_trial(report, "good", 2)
        add_trial(report, "one-trial", 1)
        add_trial(report, "fails", 1)
        add_trial(report, "fails", 2, error="paid malformed response")
        add_trial(report, "missing", 1, level="")
        add_trial(report, "cancelled", 1, started=False, cancelled=True)
        add_trial(report, "unstable", 1)
        add_trial(report, "unstable", 2, choice="hard")
        add_trial(report, "different-depth", 1)
        add_trial(report, "different-depth", 2, level="high")
        add_trial(report, "unoffered", 1, choice="unknown")
        add_trial(report, "no-policy", 1, policy="unknown")
        add_trial(report, "no-evidence", 1, attempts=[])
        cases, provenance = teacher_cases.freeze(report, "sol61", "hash")
        self.assertEqual([case["name"] for case in cases], ["good"])
        excluded = {entry["case"]: entry["reasons"] for entry in provenance["exclusions"]}
        self.assertEqual(len(excluded), 9)
        self.assertIn("at least two matching successful trials required for stability", excluded["one-trial"])
        self.assertIn("teacher trial failed", excluded["fails"])
        self.assertIn("missing task depth", excluded["missing"])
        self.assertIn("teacher trial did not start", excluded["cancelled"])
        self.assertIn("teacher model choice or task depth was unstable", excluded["unstable"])
        self.assertIn("teacher model choice or task depth was unstable", excluded["different-depth"])
        self.assertIn("teacher choice was not offered", excluded["unoffered"])
        self.assertIn("missing recorded teacher policy", excluded["no-policy"])
        self.assertIn("missing matching primary request evidence", excluded["no-evidence"])

    def test_split_keeps_explicit_semantic_variants_together(self):
        cases = [{"name": "routine/import", "group": "routine"}, {"name": "urgent/import", "group": "priority"},
                 {"name": "pt/import", "group": "routine"}, {"name": "complex/security", "group": "difficult"}]
        families = {"routine/import": "import", "urgent/import": "import", "pt/import": "import", "complex/security": "security"}
        train, evaluation, details = teacher_cases.split_cases(cases, families, 0.25, "stable-seed")
        train_names = {case["name"] for case in train}
        eval_names = {case["name"] for case in evaluation}
        self.assertFalse(train_names & eval_names)
        self.assertTrue({"routine/import", "urgent/import", "pt/import"} <= train_names or
                        {"routine/import", "urgent/import", "pt/import"} <= eval_names)
        self.assertFalse(set(details["train_families"]) & set(details["eval_families"]))
        self.assertEqual(teacher_cases.split_cases(list(reversed(cases)), families, 0.25, "stable-seed")[2]["eval_families"], details["eval_families"])
        self.assertTrue(details["train_group_support"])
        self.assertTrue(details["eval_group_support"])
        with self.assertRaisesRegex(ValueError, "missing included cases"):
            teacher_cases.split_cases(cases, {"routine/import": "import"}, 0.25, "seed")
        with self.assertRaisesRegex(ValueError, "at least two"):
            teacher_cases.split_cases(cases, {case["name"]: "one-family" for case in cases}, 0.25, "seed")

    def test_split_covers_rare_groups_and_target_tiers_on_both_sides(self):
        cases = [{"name": "routine/a", "group": "routine", "expect": ["cheap"]},
                 {"name": "routine/b", "group": "routine", "expect": ["cheap"]},
                 {"name": "ui/a", "group": "substantial-ui", "expect": ["hard"]},
                 {"name": "ui/b", "group": "substantial-ui", "expect": ["hard"]}]
        families = {case["name"]: case["name"] for case in cases}
        train, evaluation, details = teacher_cases.split_cases(cases, families, 0.25, "teacher-eval-v1")
        self.assertEqual(len(train), 2)
        self.assertEqual(len(evaluation), 2)
        self.assertEqual(details["requested_eval_families"], 1)
        self.assertTrue(details["family_count_adjusted_for_coverage"])
        self.assertEqual(details["realized_eval_family_fraction"], 0.5)
        self.assertEqual(details["realized_eval_case_fraction"], 0.5)
        for key in ("train_group_support", "eval_group_support"):
            self.assertEqual(details[key], {"routine": 1, "substantial-ui": 1})
        for key in ("train_model_support", "eval_model_support"):
            self.assertEqual(details[key], {"cheap": 1, "hard": 1})
        self.assertFalse(details["unsupported_singleton_strata"])
        reverse = teacher_cases.split_cases(list(reversed(cases)), dict(reversed(list(families.items()))), 0.25, "teacher-eval-v1")
        self.assertEqual(reverse[2]["eval_families"], details["eval_families"])

    def test_split_exposes_singleton_strata_without_splitting_their_family(self):
        cases = [{"name": "routine/a", "group": "routine", "expect": ["cheap"]},
                 {"name": "routine/b", "group": "routine", "expect": ["cheap"]},
                 {"name": "ui/en", "group": "substantial-ui", "expect": ["hard"]},
                 {"name": "ui/pt", "group": "substantial-ui", "expect": ["hard"]}]
        families = {"routine/a": "routine-a", "routine/b": "routine-b", "ui/en": "same-ui", "ui/pt": "same-ui"}
        train, evaluation, details = teacher_cases.split_cases(cases, families, 0.25, "seed")
        unsupported = {(entry["kind"], entry["label"]) for entry in details["unsupported_singleton_strata"]}
        self.assertEqual(unsupported, {("group", "substantial-ui"), ("target_model", "hard")})
        train_names, eval_names = {case["name"] for case in train}, {case["name"] for case in evaluation}
        self.assertTrue({"ui/en", "ui/pt"} <= train_names or {"ui/en", "ui/pt"} <= eval_names)
        self.assertFalse(set(details["train_families"]) & set(details["eval_families"]))

    def test_coverage_solver_backtracks_and_reports_jointly_impossible_constraints(self):
        # These overlapping family pairs force opposite sides; a pure hash
        # assignment can miss a group even though a valid coloring exists.
        cases, families = [], {}
        for group, members in {"ab": ("a", "b"), "bc": ("b", "c"), "cd": ("c", "d"), "da": ("d", "a")}.items():
            for family in members:
                name = group + "/" + family
                cases.append({"name": name, "group": group, "expect": ["cheap"]})
                families[name] = family
        _, _, details = teacher_cases.split_cases(cases, families, 0.25, "seed")
        self.assertTrue(all(entry["train_cases"] > 0 and entry["eval_cases"] > 0 for entry in details["coverage"]))
        impossible = cases[:4] + [{"name": "ac/a", "group": "ac", "expect": ["cheap"]},
                                {"name": "ac/c", "group": "ac", "expect": ["cheap"]}]
        impossible_families = {case["name"]: case["name"].split("/")[1] for case in impossible}
        with self.assertRaisesRegex(ValueError, "cannot jointly cover"):
            teacher_cases.split_cases(impossible, impossible_families, 0.25, "seed")
        with self.assertRaisesRegex(ValueError, "search budget exhausted"):
            teacher_cases.covered_assignment(4, 2, [0b0011, 0b1100], [0])

    def test_cli_exports_unsplit_without_manifest_and_refuses_unreviewed_split(self):
        report = report_fixture()
        add_trial(report, "safe", 1)
        add_trial(report, "safe", 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "report.json"
            raw = json.dumps(report).encode()
            report_path.write_bytes(raw)
            output = root / "export"
            args = ["--report", str(report_path), "--teacher", "sol61", "--out", str(output)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(teacher_cases.main(args), 0)
            self.assertTrue((output / "teacher-cases.json").exists())
            self.assertFalse((output / "teacher-train.json").exists())
            self.assertFalse((output / "teacher-eval.json").exists())
            provenance = json.loads((output / "teacher-provenance.json").read_text())
            self.assertEqual(provenance["split"]["method"], "unsplit")
            self.assertEqual(provenance["report_sha256"], teacher_cases.hashlib.sha256(raw).hexdigest())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as rejected:
                teacher_cases.main(args + ["--split"])
            self.assertEqual(rejected.exception.code, 2)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                teacher_cases.main(args)

    def test_cli_records_manifest_hash_and_separate_split_files(self):
        report = report_fixture()
        add_trial(report, "routine/import", 1)
        add_trial(report, "routine/import", 2)
        add_trial(report, "priority/import", 1)
        add_trial(report, "priority/import", 2)
        add_trial(report, "complex/security", 1, choice="hard", level="high")
        add_trial(report, "complex/security", 2, choice="hard", level="high")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "report.json"
            report_path.write_text(json.dumps(report))
            family_path = root / "families.json"
            family_path.write_text(json.dumps({"routine/import": "import", "priority/import": "import", "complex/security": "security"}))
            output = root / "export"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(teacher_cases.main(["--report", str(report_path), "--teacher", "sol61", "--out", str(output),
                                                    "--split", "--families", str(family_path)]), 0)
            train = json.loads((output / "teacher-train.json").read_text())["cases"]
            evaluation = json.loads((output / "teacher-eval.json").read_text())["cases"]
            self.assertEqual(len(train) + len(evaluation), 3)
            provenance = json.loads((output / "teacher-provenance.json").read_text())
            self.assertEqual(provenance["split"]["manifest_sha256"], teacher_cases.hashlib.sha256(family_path.read_bytes()).hexdigest())

    def test_freeze_rejects_changed_state_policy_identity_and_malformed_policy(self):
        report = report_fixture()
        for name in ("state-changes", "policy-changes", "identity-changes", "malformed-policy"):
            add_trial(report, name, 1)
            add_trial(report, name, 2)
        report["inputs"][1]["state"] = "different prepared state"
        report["policies"]["new-policy"] = copy.deepcopy(report["policies"]["policy"])
        report["outcomes"][3][1]["policy"] = "new-policy"
        report["outcomes"][5][1]["attempts"][0]["model"] = "different-teacher-version"
        report["policies"]["malformed"] = {"deciders": None, "reasoning": "invalid"}
        report["outcomes"][6][1]["policy"] = "malformed"
        report["outcomes"][7][1]["policy"] = "malformed"
        cases, provenance = teacher_cases.freeze(report, "sol61", "hash")
        self.assertEqual(cases, [])
        reasons = {entry["case"]: entry["reasons"] for entry in provenance["exclusions"]}
        self.assertIn("prepared state changed or is missing across trials", reasons["state-changes"])
        self.assertIn("teacher policy changed or is missing across trials", reasons["policy-changes"])
        self.assertIn("teacher model identity changed across trials", reasons["identity-changes"])
        self.assertIn("teacher identity absent from policy", reasons["malformed-policy"])
        self.assertIn("teacher task depth absent from policy", reasons["malformed-policy"])


if __name__ == "__main__":
    unittest.main()
