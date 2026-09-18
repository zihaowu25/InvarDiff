"""Shared cache-preset resolution without changing sampler algorithms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


PRESET_PATH = Path(__file__).with_name("cache_presets.json")


def load_presets(path: Path = PRESET_PATH) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported cache preset schema")
    return payload["policies"]


def same_execution_config(saved: object, expected: dict) -> bool:
    """Compare Cache Book settings without treating a preset label as a setting."""
    if not isinstance(saved, dict):
        return False
    return {
        key: value for key, value in saved.items() if key != "cache_preset"
    } == {
        key: value for key, value in expected.items() if key != "cache_preset"
    }


def add_preset_argument(
    parser: argparse.ArgumentParser,
    policy: str,
    tiers: Iterable[str] = ("fast", "balanced", "slow"),
) -> None:
    parser.add_argument(
        "--cache-preset",
        choices=tuple(tiers),
        default="fast" if "fast" in tiers else next(iter(tiers)),
        help=f"Threshold preset from cache_presets.json for {policy}.",
    )


def apply_preset(
    args: argparse.Namespace,
    policy: str,
    flag_to_attr: dict[str, str],
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Apply preset values except for threshold flags explicitly supplied.

    argparse does not retain whether a value came from a default. Inspecting
    the original option tokens preserves the documented precedence:
    explicit threshold > selected preset > parser legacy default.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    tier = args.cache_preset
    policies = load_presets()
    if policy not in policies or tier not in policies[policy]["presets"]:
        raise ValueError(f"Missing cache preset {policy}:{tier}")
    values = policies[policy]["presets"][tier]["thresholds"]
    explicit = {token.split("=", 1)[0] for token in argv if token.startswith("--")}
    for flag, attr in flag_to_attr.items():
        if flag not in explicit:
            if attr not in values:
                raise ValueError(f"Preset {policy}:{tier} lacks {attr}")
            setattr(args, attr, float(values[attr]))
    args.cache_preset_resolved = tier
    args.cache_resolved_thresholds = {
        attr: float(getattr(args, attr)) for attr in flag_to_attr.values()
    }
    return args
