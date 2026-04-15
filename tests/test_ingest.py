from pathlib import Path

from bdlaw_ollama.ingest import canonicalize_group


def test_canonicalize_group_deduplicates_pages_and_uses_fallback(tmp_path: Path) -> None:
    fallback_root = tmp_path / "bdcode_txt" / "volume_xiii"
    fallback_root.mkdir(parents=True)
    fallback_file = fallback_root / "The_Bangladesh_Bank_Order_1972.txt"
    fallback_file.write_text("Fallback law text", encoding="utf-8")

    entries = [
        {
            "name": "The Bangladesh Bank Order, 1972",
            "filename": "bb-order.pdf",
            "pdf_path": "downloads/bb-order.pdf",
            "ocr_language": "eng",
            "_volume_name": "VOLUME-XIII",
            "_volume_type": "roman",
            "_volume_number": "XIII",
            "pages": [],
        },
        {
            "name": "127",
            "filename": "bb-order.pdf",
            "pdf_path": "downloads/bb-order.pdf",
            "ocr_language": "eng",
            "_volume_name": "VOLUME-XIII",
            "_volume_type": "roman",
            "_volume_number": "XIII",
            "pages": [],
        },
    ]

    law = canonicalize_group(entries, fallback_txt_root=tmp_path / "bdcode_txt")
    assert law["title"] == "The Bangladesh Bank Order, 1972"
    assert law["source_fallback_txt"] is not None
    assert law["pages"][0]["text"] == "Fallback law text"


def test_canonicalize_group_keeps_longest_page_text(tmp_path: Path) -> None:
    entries = [
        {
            "name": "Sample Law, 2000",
            "filename": "sample.pdf",
            "pdf_path": "downloads/sample.pdf",
            "ocr_language": "eng",
            "_volume_name": "VOLUME-I",
            "_volume_type": "roman",
            "_volume_number": "I",
            "pages": [{"page_number": 1, "text": "short"}],
        },
        {
            "name": "2000",
            "filename": "sample.pdf",
            "pdf_path": "downloads/sample.pdf",
            "ocr_language": "eng",
            "_volume_name": "VOLUME-I",
            "_volume_type": "roman",
            "_volume_number": "I",
            "pages": [{"page_number": 1, "text": "This is the longer body text."}],
        },
    ]
    law = canonicalize_group(entries, fallback_txt_root=tmp_path)
    assert law["pages"][0]["text"] == "This is the longer body text."
