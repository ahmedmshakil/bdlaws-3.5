from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Iterable, Iterator


BANGLA_DIGITS = "০১২৩৪৫৬৭৮৯"
ASCII_DIGITS = "0123456789"
BANGLA_TO_ASCII = str.maketrans(BANGLA_DIGITS, ASCII_DIGITS)
ASCII_TO_BANGLA = str.maketrans(ASCII_DIGITS, BANGLA_DIGITS)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_pickle(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def normalize_digits(text: str) -> str:
    return text.translate(BANGLA_TO_ASCII)


def contains_bangla(text: str) -> bool:
    return bool(re.search(r"[\u0980-\u09FF]", text))


def slugify(text: str) -> str:
    lowered = normalize_digits(text).lower()
    lowered = re.sub(r"[^a-z0-9\u0980-\u09ff]+", "-", lowered)
    lowered = re.sub(r"-{2,}", "-", lowered)
    return lowered.strip("-") or "item"


def compact_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_year_like(text: str) -> bool:
    clean = normalize_digits(text.strip())
    return bool(re.fullmatch(r"(19|20)\d{2}", clean))


def is_numeric_name(text: str) -> bool:
    clean = normalize_digits(text.strip())
    return bool(re.fullmatch(r"[0-9ivxlcdmIVXLCDM]+", clean))


def choose_best_title(candidates: Iterable[str]) -> str:
    cleaned = [candidate.strip() for candidate in candidates if candidate and candidate.strip()]
    preferred = [candidate for candidate in cleaned if not is_year_like(candidate) and not is_numeric_name(candidate)]
    pool = preferred or cleaned
    return max(pool, key=len, default="Unknown Law")
