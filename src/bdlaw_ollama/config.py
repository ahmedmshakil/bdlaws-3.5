from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class AppConfig:
    root: Path
    project: dict[str, Any]
    rag: dict[str, Any]
    train: dict[str, Any]

    @property
    def paths(self) -> dict[str, Path]:
        raw_paths = self.project["paths"]
        return {key: (self.root / value).resolve() for key, value in raw_paths.items() if key.endswith("_dir") or key.endswith("_path") or key.endswith("_root")}

    def path(self, key: str) -> Path:
        raw_value = self.project["paths"][key]
        return (self.root / raw_value).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_app_config(root: str | Path | None = None) -> AppConfig:
    root_path = Path(root or ".").resolve()
    return AppConfig(
        root=root_path,
        project=load_yaml(root_path / "configs" / "project.yaml"),
        rag=load_yaml(root_path / "configs" / "rag.yaml"),
        train=load_yaml(root_path / "configs" / "train.yaml"),
    )
