#!/usr/bin/env python3
"""Freeze auditable teacher choices without modifying canonical routing cases.

An unsplit export needs a schema-version-2 benchmark report with at least two
matching, successful teacher trials per case under the same state and policy.
A train/evaluation split additionally requires an explicit JSON mapping from
every included case name to a semantic family. Keep urgency variants,
translations and related tasks in the same split. Teacher agreement is a tuning
target, not a claim of
task correctness or an independent replacement for canonical accuracy.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def freeze(report, teacher, report_sha256):
    """Return trusted, stable teacher cases plus their provenance/exclusions."""
    if report.get("schema_version") != 2:
        raise ValueError("teacher export requires benchmark schema_version 2")
    deciders = report.get("deciders", [])
    if not isinstance(deciders, list) or deciders.count(teacher) != 1:
        raise ValueError(f"teacher {teacher!r} must name exactly one report decider")
    teacher_index = deciders.index(teacher)
    cases, outcomes, inputs = (report.get(key) for key in ("cases", "outcomes", "inputs"))
    if not all(isinstance(rows, list) for rows in (cases, outcomes, inputs)) or not cases:
        raise ValueError("report needs nonempty cases, outcomes and prepared inputs")
    if len(cases) != len(outcomes) or len(cases) != len(inputs):
        raise ValueError("cases, outcomes and prepared inputs must have identical row counts")
    policies = report.get("policies", {})
    if not isinstance(policies, dict):
        raise ValueError("report policies must be an object")
    grouped = {}
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or not isinstance(case.get("name"), str) or not case["name"].strip():
            raise ValueError(f"case row {index} has no valid base name")
        grouped.setdefault(case["name"], []).append(index)

    accepted, sources, exclusions = [], [], []
    configured_models, observed_models = set(), set()
    for name, rows in grouped.items():
        reasons, signatures, answers, trials = set(), set(), set(), []
        source = {"case": name, "trials": [], "policy_fingerprints": [], "policy_sha256": {},
                  "request_digests": [], "prepared_state_sha256": [],
                  "configured_teacher_models": [], "observed_teacher_models": []}
        case_models, case_observed = set(), set()
        frozen_case = None
        for index in rows:
            case, input_row, outcome_row = cases[index], inputs[index], outcomes[index]
            trial = case.get("trial", 0)
            if not isinstance(trial, int) or isinstance(trial, bool) or trial < 0 or trial in trials:
                reasons.add("invalid or duplicated trial identity")
            trials.append(trial)
            if not isinstance(outcome_row, list) or teacher_index >= len(outcome_row) or not isinstance(outcome_row[teacher_index], dict):
                reasons.add("missing teacher outcome")
                continue
            outcome = outcome_row[teacher_index]
            if outcome.get("started") is not True:
                reasons.add("teacher trial did not start")
            if outcome.get("cancelled") is not False:
                reasons.add("teacher trial cancelled or cancellation state missing")
            if outcome.get("error"):
                reasons.add("teacher trial failed")
            choice, level = outcome.get("choice"), outcome.get("task_depth")
            if not isinstance(choice, str) or not choice.strip():
                reasons.add("missing primary model choice")
            if not isinstance(level, str) or not level.strip():
                reasons.add("missing task depth")
            if not isinstance(input_row, dict):
                reasons.add("missing prepared input")
                continue
            candidates = input_row.get("candidates", [])
            offered = [candidate.get("id") for candidate in candidates if isinstance(candidate, dict)] if isinstance(candidates, list) else []
            if not offered or len(offered) != len(candidates) or any(not isinstance(model, str) or not model for model in offered) or len(set(offered)) != len(offered):
                reasons.add("invalid offered candidates")
            if choice not in offered:
                reasons.add("teacher choice was not offered")
            prompt, images = case.get("prompt"), case.get("images")
            group = case.get("group", name.split("/", 1)[0] if "/" in name else "ungrouped")
            if not isinstance(prompt, str) or not prompt.strip() or not isinstance(images, bool) or not isinstance(group, str):
                reasons.add("invalid original prompt, images or group")
            else:
                signatures.add(digest({"prompt": prompt, "images": images, "offered": offered, "group": group}))
            state = input_row.get("state")
            if not isinstance(state, str) or not state.strip():
                reasons.add("missing prepared state")
            else:
                source["prepared_state_sha256"].append(hashlib.sha256(state.encode("utf-8")).hexdigest())
            fingerprint = outcome.get("policy")
            policy = policies.get(fingerprint) if isinstance(fingerprint, str) else None
            if not isinstance(fingerprint, str) or not fingerprint or not isinstance(policy, dict):
                reasons.add("missing recorded teacher policy")
            else:
                decider_infos = policy.get("deciders", [])
                infos = [info for info in decider_infos if isinstance(info, dict) and info.get("name") == teacher] if isinstance(decider_infos, list) else []
                if len(infos) != 1 or not isinstance(infos[0].get("model"), str) or not infos[0]["model"]:
                    reasons.add("teacher identity absent from policy")
                else:
                    case_models.add(infos[0]["model"])
                reasoning = policy.get("reasoning")
                policy_levels = reasoning.get("levels", []) if isinstance(reasoning, dict) else []
                allowed_levels = [entry.get("id") for entry in policy_levels if isinstance(entry, dict)] if isinstance(policy_levels, list) else []
                if level not in allowed_levels:
                    reasons.add("teacher task depth absent from policy")
                source["policy_fingerprints"].append(fingerprint)
                source["policy_sha256"][fingerprint] = digest(policy)
            attempts = outcome.get("attempts", [])
            primary = next((attempt for attempt in attempts if isinstance(attempt, dict) and not attempt.get("pair") and attempt.get("outcome") in ("decided", "doubted")), None) if isinstance(attempts, list) else None
            if primary is None or primary.get("choice") != choice or primary.get("level") != level or not isinstance(primary.get("digest"), str) or not primary["digest"]:
                reasons.add("missing matching primary request evidence")
            else:
                source["request_digests"].append(primary["digest"])
                if isinstance(primary.get("model"), str) and primary["model"]:
                    case_observed.add(primary["model"])
            if isinstance(choice, str) and isinstance(level, str):
                answers.add((choice, level))
                frozen_case = {"name": name, "group": group, "expect": [choice], "reasoning": [level],
                               "images": images, "offered": offered, "prompt": prompt,
                               "note": f"Frozen {teacher} primary choice and task depth; source report sha256 {report_sha256}."}
        if len(signatures) != 1:
            reasons.add("task or offered candidates changed across trials")
        if len(rows) < 2:
            reasons.add("at least two matching successful trials required for stability")
        if len(set(source["prepared_state_sha256"])) != 1:
            reasons.add("prepared state changed or is missing across trials")
        if len(set(source["policy_fingerprints"])) != 1:
            reasons.add("teacher policy changed or is missing across trials")
        if len(answers) != 1:
            reasons.add("teacher model choice or task depth was unstable")
        if len(case_models) > 1 or len(case_observed) > 1:
            reasons.add("teacher model identity changed across trials")
        if reasons:
            exclusions.append({"case": name, "trials": trials, "reasons": sorted(reasons)})
            continue
        source["trials"] = sorted(trials)
        for key in ("policy_fingerprints", "request_digests", "prepared_state_sha256"):
            source[key] = sorted(set(source[key]))
        source["configured_teacher_models"] = sorted(case_models)
        source["observed_teacher_models"] = sorted(case_observed)
        source["canonical_expect"] = cases[rows[0]].get("expect", [])
        source["canonical_reasoning"] = cases[rows[0]].get("reasoning", [])
        accepted.append(frozen_case)
        sources.append(source)
        configured_models.update(case_models)
        observed_models.update(case_observed)
    provenance = {"schema_version": 1, "source_schema_version": 2, "report_sha256": report_sha256,
                  "teacher": teacher, "configured_teacher_models": sorted(configured_models),
                  "observed_teacher_models": sorted(observed_models), "included_cases": len(accepted),
                  "excluded_cases": len(exclusions), "sources": sources, "exclusions": exclusions,
                  "target": "primary model choice and task depth, before economy or escalation",
                  "evaluation_scores": {"model": "choice_accuracy", "joint": "choice_joint_accuracy",
                                        "cli_score": "initial", "final_route_scores_are_separate": True},
                  "canonical_labels_modified": False}
    return accepted, provenance


def split_cases(cases, families, eval_fraction, seed):
    """Stratify groups/targets while retaining explicit semantic families."""
    if not isinstance(families, dict) or not all(isinstance(name, str) and isinstance(family, str) and family.strip() and family == family.strip() for name, family in families.items()):
        raise ValueError("families must be a JSON object mapping case names to nonempty semantic-family strings")
    missing = [case["name"] for case in cases if case["name"] not in families]
    if missing:
        raise ValueError("family manifest is missing included cases: " + ", ".join(missing))
    if not math.isfinite(eval_fraction) or not 0 < eval_fraction < 1:
        raise ValueError("eval fraction must be finite and strictly between 0 and 1")
    family_names = {families[case["name"]] for case in cases}
    if len(family_names) < 2:
        raise ValueError("a train/eval split needs at least two explicit semantic families")
    ordered = sorted(family_names, key=lambda family: hashlib.sha256((seed + "\0" + family).encode("utf-8")).hexdigest())
    requested_count = min(len(ordered) - 1, max(1, math.ceil(len(ordered) * eval_fraction)))
    strata = {}
    for case in cases:
        labels = [("group", case["group"])] + [("target_model", model) for model in case.get("expect", [])]
        for kind, label in labels:
            strata.setdefault((kind, label), set()).add(families[case["name"]])
    positions = {family: index for index, family in enumerate(ordered)}
    constraints = [sum(1 << positions[family] for family in members)
                   for _, members in sorted(strata.items()) if len(members) >= 2]
    # A family can span several groups/targets, so a greedy per-group sample
    # can falsely declare a possible split impossible. Exact propagation and
    # deterministic backtracking keep overlapping families intact. The search
    # budget bounds pathological manifests; exhaustion is explicit, not a
    # claim that no valid split exists.
    budget = [250_000]
    eval_mask = None
    for count in sorted(range(1, len(ordered)), key=lambda value: (abs(value - requested_count), value)):
        eval_mask = covered_assignment(len(ordered), count, constraints, budget)
        if eval_mask is not None:
            break
    if eval_mask is None:
        raise ValueError("semantic families cannot jointly cover every eligible group and target model in both train and evaluation; add independent families or revise the manifest")
    eval_families = {family for index, family in enumerate(ordered) if eval_mask & (1 << index)}
    train = [case for case in cases if families[case["name"]] not in eval_families]
    evaluation = [case for case in cases if families[case["name"]] in eval_families]
    def support(rows, key):
        counts = {}
        for case in rows:
            labels = [case["group"]] if key == "group" else case.get("expect", [])
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))
    train_groups, eval_groups = support(train, "group"), support(evaluation, "group")
    train_models, eval_models = support(train, "target_model"), support(evaluation, "target_model")
    coverage = []
    for (kind, label), members in sorted(strata.items()):
        train_support, eval_support = (train_groups, eval_groups) if kind == "group" else (train_models, eval_models)
        coverage.append({"kind": kind, "label": label, "families": len(members),
                         "train_cases": train_support.get(label, 0), "eval_cases": eval_support.get(label, 0),
                         "both_sides_supported": len(members) >= 2})
    details = {"method": "coverage-constrained explicit semantic families, sha256 tie order", "seed": seed, "requested_eval_fraction": eval_fraction,
               "requested_eval_families": requested_count, "realized_eval_family_fraction": len(eval_families) / len(ordered),
               "realized_eval_case_fraction": len(evaluation) / len(cases), "family_count_adjusted_for_coverage": len(eval_families) != requested_count,
               "train_families": sorted(family_names - eval_families), "eval_families": sorted(eval_families),
               "case_families": {case["name"]: families[case["name"]] for case in cases},
               "train_group_support": train_groups, "eval_group_support": eval_groups,
               "train_model_support": train_models, "eval_model_support": eval_models,
               "coverage": coverage, "unsupported_singleton_strata": [entry for entry in coverage if not entry["both_sides_supported"]],
               "train_cases": len(train), "eval_cases": len(evaluation)}
    return train, evaluation, details


def covered_assignment(family_count, eval_count, constraints, budget):
    """Find a two-sided coverage assignment with exactly eval_count families."""
    all_families = (1 << family_count) - 1
    def search(evaluation, training):
        budget[0] -= 1
        if budget[0] < 0:
            raise ValueError("family coverage search budget exhausted; no infeasibility conclusion was made")
        while True:
            unknown = all_families & ~(evaluation | training)
            if evaluation.bit_count() > eval_count or training.bit_count() > family_count - eval_count:
                return None
            force_eval, force_train = 0, 0
            if evaluation.bit_count() == eval_count:
                force_train |= unknown
            if training.bit_count() == family_count - eval_count:
                force_eval |= unknown
            for members in constraints:
                remaining = members & unknown
                has_eval, has_train = bool(members & evaluation), bool(members & training)
                if not remaining and not (has_eval and has_train):
                    return None
                if remaining.bit_count() == 1:
                    if not has_eval:
                        force_eval |= remaining
                    if not has_train:
                        force_train |= remaining
            if force_eval & force_train:
                return None
            if not (force_eval | force_train):
                break
            evaluation |= force_eval
            training |= force_train
        if not unknown:
            return evaluation
        pending = [members & unknown for members in constraints
                   if not (members & evaluation and members & training)]
        choices = min(pending, key=lambda mask: (mask.bit_count(), mask)) if pending else unknown
        family = choices & -choices
        # Prefer the branch closest to the requested proportion in affected
        # strata. This is only search order: backtracking still considers the
        # other branch and never sacrifices coverage for approximate balance.
        fraction = eval_count / family_count
        def imbalance(to_eval):
            assigned = (evaluation | training).bit_count() + 1
            cost = (evaluation.bit_count() + to_eval - fraction * assigned) ** 2 / family_count
            for members in constraints:
                if members & family:
                    assigned = (members & (evaluation | training)).bit_count() + 1
                    count = (members & evaluation).bit_count() + to_eval
                    cost += (count - fraction * assigned) ** 2 / members.bit_count()
            return cost
        prefer_eval = imbalance(1) <= imbalance(0)
        for to_eval in (prefer_eval, not prefer_eval):
            found = search(evaluation | family if to_eval else evaluation, training if to_eval else training | family)
            if found is not None:
                return found
        return None
    return search(0, 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, required=True, help="frozen schema-version-2 benchmark JSON")
    parser.add_argument("--teacher", required=True, help="exact reference decider name, e.g. sol61")
    parser.add_argument("--out", type=Path, required=True, help="directory for new export files")
    parser.add_argument("--split", action="store_true", help="write train/eval files; requires --families")
    parser.add_argument("--families", type=Path, help="explicit JSON case-name to semantic-family mapping")
    parser.add_argument("--eval-fraction", type=float, default=0.25)
    parser.add_argument("--seed", default="teacher-eval-v1")
    args = parser.parse_args(argv)
    if args.split and args.families is None:
        parser.error("--split requires --families; semantic task families cannot be inferred safely from case names")
    try:
        raw = args.report.read_bytes()
        report = json.loads(raw)
        if not isinstance(report, dict):
            raise ValueError("benchmark report must be a JSON object")
        cases, provenance = freeze(report, args.teacher, hashlib.sha256(raw).hexdigest())
        if not cases:
            raise ValueError("no stable, auditable teacher cases remain after exclusions")
        files = {"teacher-cases.json": {"cases": cases}}
        if args.families is not None:
            family_raw = args.families.read_bytes()
            families = json.loads(family_raw)
            known_names = {case["name"] for case in report["cases"]}
            if isinstance(families, dict) and set(families) - known_names:
                raise ValueError("family manifest contains case names absent from the report")
            train, evaluation, details = split_cases(cases, families, args.eval_fraction, args.seed)
            details["manifest_sha256"] = hashlib.sha256(family_raw).hexdigest()
            provenance["split"] = details
            files["teacher-train.json"] = {"cases": train}
            files["teacher-eval.json"] = {"cases": evaluation}
        else:
            provenance["split"] = {"method": "unsplit", "reason": "explicit semantic-family manifest required before splitting"}
        files["teacher-provenance.json"] = provenance
        conflicts = [str(args.out / name) for name in files if (args.out / name).exists()]
        if conflicts:
            raise ValueError("export files already exist: " + ", ".join(conflicts))
        encoded = {name: json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n" for name, value in files.items()}
        args.out.mkdir(parents=True, exist_ok=True)
        for name, body in encoded.items():
            with (args.out / name).open("x", encoding="utf-8") as stream:
                stream.write(body)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(f"Exported {len(cases)} teacher cases; excluded {provenance['excluded_cases']}; provenance: {args.out / 'teacher-provenance.json'}")
    if args.families is None:
        print("Unsplit export only. Create an explicit semantic-family manifest before training/evaluation splitting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
