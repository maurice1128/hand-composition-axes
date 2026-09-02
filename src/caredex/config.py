"""Minimal YAML config with dotted command-line overrides.

Deliberately not hydra/omegaconf: the whole config is one flat-ish dict, every
script takes ``--set a.b=c``, and the resolved dict is written into every
checkpoint so a result can be traced back to the settings that produced it.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    for item in overrides or []:
        _apply_override(cfg, item)
    return cfg


def _apply_override(cfg: dict, item: str) -> None:
    if "=" not in item:
        raise ValueError(f"override must be key.path=value, got {item!r}")
    key, raw = item.split("=", 1)
    node: Any = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = _parse(raw)


def _parse(raw: str) -> Any:
    """Interpret an override value as Python literal, falling back to string."""
    lowered = raw.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def dump(cfg: dict) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True).rstrip()
