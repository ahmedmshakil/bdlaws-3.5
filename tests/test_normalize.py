from pathlib import Path

from bdlaw_ollama.normalize import clean_page_text, derive_act_no, derive_year, find_best_txt_fallback
from bdlaw_ollama.utils import choose_best_title, normalize_digits


def test_choose_best_title_skips_year_and_number() -> None:
    assert choose_best_title(["1972", "6", "The Bangladesh Bank Order, 1972"]) == "The Bangladesh Bank Order, 1972"


def test_clean_page_text_strips_headers_and_titles() -> None:
    raw = "\n".join(
        [
            "THE BANGLADESH CODE VOLUME - XIII",
            "The Bangladesh Bank Order, 1972 407",
            "The Bangladesh Bank Order, 1972",
            "1. This Order may be called the Bangladesh Bank Order, 1972.",
        ]
    )
    cleaned = clean_page_text(raw, title="The Bangladesh Bank Order, 1972")
    assert "THE BANGLADESH CODE" not in cleaned
    assert cleaned.startswith("1. This Order")


def test_derive_metadata_from_english_and_bangla_text() -> None:
    assert derive_year("বাংলাদেশ শিল্পকলা একাডেমী আইন, ১৯৮৯") == "1989"
    assert derive_act_no("Act No. IX of 1990") == "IX"


def test_find_best_txt_fallback(tmp_path: Path) -> None:
    root = tmp_path / "bdcode_txt" / "volume_xiii"
    root.mkdir(parents=True)
    target = root / "The_Bangladesh_Bank_Order_1972.txt"
    target.write_text("sample", encoding="utf-8")
    found = find_best_txt_fallback(tmp_path / "bdcode_txt", "roman", "XIII", "The Bangladesh Bank Order, 1972")
    assert found == target


def test_normalize_digits() -> None:
    assert normalize_digits("১২৩") == "123"
