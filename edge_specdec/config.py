from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelPair:
    name: str
    target: str
    draft: str


def load_model_pairs(config_path: str | Path) -> list[ModelPair]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    pairs = []
    for item in data.get("model_pairs", []):
        pairs.append(
            ModelPair(
                name=str(item["name"]),
                target=str(item["target"]),
                draft=str(item["draft"]),
            )
        )
    if not pairs:
        raise ValueError(f"No model_pairs found in {path}")
    return pairs


def select_model_pairs(pairs: list[ModelPair], names: list[str] | None) -> list[ModelPair]:
    if not names:
        return pairs

    by_name = {pair.name: pair for pair in pairs}
    missing = [name for name in names if name not in by_name]
    if missing:
        known = ", ".join(sorted(by_name))
        raise ValueError(f"Unknown model pair(s): {missing}. Known pairs: {known}")
    return [by_name[name] for name in names]
