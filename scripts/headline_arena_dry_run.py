#!/usr/bin/env python3
"""P2.1 — Headline Arena dual-agent dry-run ($0, offline).

Validates the example (or local) dual-arm config: two distinct arena agents
with separate credential *slots* (env names / file paths — no secrets read).
Writes a synthetic forecast + scorecard stub under
``benchmark/results/headline_arena/dry_run.json``.

No network. Live registration/forecast needs the headlinearena plugin and
real credentials (see docs/HEADLINE_ARENA.md). ``--live`` without both env
pairs exits 2 with a clear message.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "config" / "headline_arena.example.yaml"
LOCAL = REPO / "config" / "headline_arena.local.yaml"
OUT = REPO / "benchmark" / "results" / "headline_arena" / "dry_run.json"
FIXTURE_GENERATED_AT = "2026-01-01T00:00:00+00:00"


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError:
        # Minimal subset parser for our committed example (no PyYAML required):
        return _parse_simple_yaml(path.read_text(encoding="utf-8"))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _parse_simple_yaml(text: str) -> dict:
    """Tiny indented-key parser sufficient for the committed example file."""
    # Prefer stdlib-only: use json after a hand conversion is brittle; instead
    # embed the canonical structure when reading the example path.
    # For robustness on the example file, return the known schema and verify
    # arena_agent lines match arms.py.
    arms: dict = {}
    current = None
    cred = None
    dry: dict = {}
    section = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        key, _, val = line.strip().partition(":")
        val = val.strip().strip('"').strip("'")
        if indent == 0 and key == "arms":
            section = "arms"
            continue
        if indent == 0 and key == "dry_run":
            section = "dry_run"
            current = None
            continue
        if section == "arms" and indent == 2 and val == "":
            current = key
            arms[current] = {"credentials": {}}
            cred = arms[current]["credentials"]
            continue
        if section == "arms" and current and indent == 4:
            if key == "arena_agent":
                arms[current]["arena_agent"] = val
            elif key == "analysts":
                inner = val.strip("[]")
                arms[current]["analysts"] = [x.strip() for x in inner.split(",") if x.strip()]
            elif key == "credentials":
                continue
        if section == "arms" and current and indent == 6 and cred is not None:
            cred[key] = val
        if section == "dry_run" and indent == 2:
            dry[key] = int(val) if val.isdigit() else (float(val) if val.replace(".", "", 1).isdigit() else val)
    return {"arms": arms, "dry_run": dry}


def validate_config(cfg: dict) -> list[str]:
    errs: list[str] = []
    arms = cfg.get("arms") or {}
    if set(arms) != {"macro", "no_macro"}:
        errs.append(f"arms must be {{macro, no_macro}}, got {sorted(arms)}")
        return errs
    names = [arms[k].get("arena_agent") for k in ("macro", "no_macro")]
    if len(set(names)) != 2 or not all(names):
        errs.append("arena_agent names must be present and distinct")
    files = [arms[k].get("credentials", {}).get("credentials_file") for k in ("macro", "no_macro")]
    envs_id = [arms[k].get("credentials", {}).get("agent_id_env") for k in ("macro", "no_macro")]
    envs_sec = [arms[k].get("credentials", {}).get("client_secret_env") for k in ("macro", "no_macro")]
    if files[0] == files[1]:
        errs.append("credentials_file paths must differ (separate scorecards)")
    if envs_id[0] == envs_id[1] or envs_sec[0] == envs_sec[1]:
        errs.append("credential env var names must differ per arm")
    # Cross-check arms.py names when importable
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import headline_arena_arms as ha  # noqa: E402

        for k in ("macro", "no_macro"):
            if arms[k]["arena_agent"] != ha.ARMS[k]["arena_agent"]:
                errs.append(
                    f"{k}: config arena_agent {arms[k]['arena_agent']!r} != arms.py {ha.ARMS[k]['arena_agent']!r}"
                )
    except Exception as e:  # pragma: no cover
        errs.append(f"could not import headline_arena_arms: {e}")
    return errs


def build_dry_run_payload(cfg: dict) -> dict:
    dr = cfg.get("dry_run") or {}
    arms_out = {}
    for name, arm in cfg["arms"].items():
        arms_out[name] = {
            "arena_agent": arm["arena_agent"],
            "analysts": arm.get("analysts"),
            "credential_slots": {
                "agent_id_env": arm["credentials"]["agent_id_env"],
                "client_secret_env": arm["credentials"]["client_secret_env"],
                "credentials_file": arm["credentials"]["credentials_file"],
            },
            "forecast": {
                "challenge_id": dr.get("challenge_id"),
                "asset": dr.get("asset"),
                "value": dr.get("forecast_value"),
                "stake_credits": dr.get("stake_credits"),
                "status": "dry_run_not_submitted",
            },
            "scorecard": {
                "calibration_public": False,
                "note": "Appears on headlinearena.com after live forecasts settle",
            },
        }
    return {
        # Keep the committed offline fixture deterministic so smoke tests do not
        # rewrite it on every run.
        "generated_at": FIXTURE_GENERATED_AT,
        "mode": "dry_run",
        "network": False,
        "issue": "https://github.com/RAFAELDCOELHO/autonomous-hedge-fund/issues/3",
        "arms": arms_out,
    }


def live_ready(cfg: dict) -> bool:
    for arm in cfg["arms"].values():
        c = arm["credentials"]
        if not os.environ.get(c["agent_id_env"]) or not os.environ.get(c["client_secret_env"]):
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live", action="store_true", help="require real env credentials (no submit)")
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args(argv)
    path = args.config or (LOCAL if LOCAL.exists() else EXAMPLE)
    if not path.exists():
        print(f"missing config {path}", file=sys.stderr)
        return 1
    cfg = _load_yaml(path)
    errs = validate_config(cfg)
    if errs:
        print("config invalid:", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        return 1
    if args.live:
        if not live_ready(cfg):
            print(
                "live: set both arms' AGENT_ID and CLIENT_SECRET env vars "
                "(see config/headline_arena.example.yaml). No submit performed.",
                file=sys.stderr,
            )
            return 2
        print("live: credential env vars present for both arms; submit via plugin ha.py (not implemented here).")
        return 0
    payload = build_dry_run_payload(cfg)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    for name, arm in payload["arms"].items():
        print(f"  {name:9s} agent={arm['arena_agent']} forecast={arm['forecast']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
