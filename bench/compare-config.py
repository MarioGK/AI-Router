#!/usr/bin/env python3
"""Build a Jev/Luna/Sol benchmark configuration from an existing router config.

JSON input needs only Python's standard library; YAML input needs PyYAML.
The output is JSON, which the router's YAML configuration reader also accepts.
Only secret file paths and environment variable names are carried over: this
program never loads dotenv files or reads credential environment variables.
Run it from the original router working directory to preserve relative
secret-file references.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import sys


def read_config(path):
    text = path.read_text(encoding="utf-8")
    try:
        config = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as error:
            raise ValueError("YAML input needs PyYAML; install it with python3 -m pip install PyYAML") from error

        class UniqueKeysLoader(yaml.SafeLoader):
            pass

        def mapping(loader, node, deep=False):
            result = {}
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node, deep=deep)
                if key in result:
                    raise ValueError(f"duplicate configuration key {key!r}")
                result[key] = loader.construct_object(value_node, deep=deep)
            return result

        UniqueKeysLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
        try:
            config = yaml.load(text, Loader=UniqueKeysLoader)
        except yaml.YAMLError as error:
            raise ValueError("invalid YAML configuration") from error
    if not isinstance(config, dict):
        raise ValueError("configuration must be an object")
    return config


def comparison_config(base, jev_name=None, policy=None):
    config = copy.deepcopy(base)
    deciders = config.get("deciders", [])
    if not isinstance(deciders, list) or not all(isinstance(d, dict) for d in deciders):
        raise ValueError("deciders must be a list of objects")
    names = [d.get("name") for d in deciders]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("configured decider names must be nonempty and unique")
    # The router accepts environment-variable references for credentials,
    # never literal keys. Do not carry invalid inline secrets into the output.
    inline_secret_fields = {"api_key", "key", "management_key", "authorization", "token", "access_token", "refresh_token", "password", "secret", "secrets"}
    scopes = [("configuration", config)] + [(name, config.get(name, {})) for name in ("upstream", "accounts", "dashboard")]
    scopes += [("decider " + d["name"], d) for d in deciders]
    for name, scope in scopes:
        if isinstance(scope, dict) and inline_secret_fields.intersection(scope):
            raise ValueError(f"{name} contains an inline credential; use secret files and key environment-variable names")
    if policy is not None:
        if not isinstance(policy, dict) or not isinstance(policy.get("auto"), dict):
            raise ValueError("policy configuration must contain an auto object")
        # Test the reviewed policy while retaining the deployment's access
        # configuration. Never import example URLs, secrets or deciders.
        config["auto"] = copy.deepcopy(policy["auto"])
        for field in ("prices", "limits"):
            value = policy.get(field, {})
            if not isinstance(value, dict):
                raise ValueError(f"policy {field} must be an object")
            config[field] = copy.deepcopy(value)
        # Config validation ties the cost-learning window to log retention.
        # Benchmarks use cold estimates, but still need a coherent config;
        # keep the reviewed policy's retention (or its standard default).
        config.pop("log_retention", None)
        if "log_retention" in policy:
            config["log_retention"] = copy.deepcopy(policy["log_retention"])
    # Benchmarks never serve a dashboard. In particular, a deployed key-based
    # sign-in configuration must not leak into the new password-only schema.
    config.pop("dashboard", None)
    jevs = [d for d in deciders if d.get("type") == "systemone" and isinstance(d.get("model"), str)
            and (d["model"] == "jev" or d["model"].startswith("jev-"))
            and (jev_name is None or d.get("name") == jev_name)]
    if len(jevs) != 1:
        raise ValueError("select one configured Jev systemone decider with --jev-name")
    jev = jevs[0]
    if jev["name"] in {"luna", "sol61"}:
        raise ValueError("the Jev decider name must differ from luna and sol61")
    upstream = config.get("upstream")
    if not isinstance(upstream, dict) or not isinstance(upstream.get("base_url"), str) or not upstream["base_url"]:
        raise ValueError("upstream.base_url is required")
    url = upstream["base_url"].rstrip("/") + "/v1/chat/completions"
    key_env = upstream.get("api_key_env") or "CLIPROXYAPI_API_KEY"
    jev["timeout"] = "60s"
    config["deciders"] = [
        jev,
        {"name": "luna", "type": "llm", "url": url, "model": "gpt-6-luna", "api_key_env": key_env, "timeout": "60s"},
        {"name": "sol61", "type": "llm", "url": url, "model": "gpt-6.1-sol(high)", "api_key_env": key_env, "timeout": "60s"},
    ]
    auto = config.setdefault("auto", {})
    if not isinstance(auto, dict):
        raise ValueError("auto must be an object")
    auto["decide_timeout"] = "120s"
    files = config.get("secrets_files", [])
    if not isinstance(files, list) or not all(isinstance(path, str) for path in files):
        raise ValueError("secrets_files must be a list of paths")
    config["secrets_files"] = [
        # config.Load resolves relative secret paths against its working
        # directory, even when the original config is in a subdirectory.
        str(Path(path).resolve()) if not Path(path).is_absolute() else path
        for path in files
    ]
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="existing router JSON or YAML configuration")
    parser.add_argument("--output", type=Path, required=True, help="new comparison JSON configuration file")
    parser.add_argument("--jev-name", help="Jev systemone decider to retain when the base config has several")
    parser.add_argument("--policy", type=Path, help="reviewed config supplying auto policy, prices and limits; credentials remain from the base config")
    args = parser.parse_args()
    try:
        base_path = args.config.resolve()
        policy = read_config(args.policy) if args.policy is not None else None
        output = comparison_config(read_config(base_path), args.jev_name, policy)
        payload = json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        # Never overwrite the base file or an earlier comparison configuration.
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            destination.write(payload)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Created {args.output}. Run with TS_ROUTER_CONFIG unset and --reference sol61.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
