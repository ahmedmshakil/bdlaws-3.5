from bdlaw_ollama.chunking import build_law_chunks


def test_build_law_chunks_preserves_section_metadata() -> None:
    law = {
        "law_id": "sample-law",
        "title": "Sample Law, 2024",
        "year": "2024",
        "act_no": "1",
        "language": "eng",
        "pages": [
            {
                "page_number": 1,
                "text": "1. Short title. This Act may be called Sample Law, 2024.\n2. Definitions. In this Act...",
            }
        ],
    }
    rag_config = {
        "chunking": {
            "max_chars": 1200,
            "overlap": 120,
            "section_markers": [
                r"(?im)^\s*\d+[.)]\s+",
            ],
        }
    }
    chunks = build_law_chunks(law, rag_config)
    assert len(chunks) >= 2
    assert chunks[0]["section_id"] == "1"
    assert "Section 1" in chunks[0]["citation_label"]


def test_build_law_chunks_handles_bangla_markers() -> None:
    law = {
        "law_id": "bangla-law",
        "title": "উদাহরণ আইন, ২০২৪",
        "year": "2024",
        "act_no": "১",
        "language": "ben",
        "pages": [
            {
                "page_number": 1,
                "text": "১। সংক্ষিপ্ত শিরোনামা। এই আইনের নাম উদাহরণ আইন, ২০২৪।\n২। সংজ্ঞা। এই আইনে...",
            }
        ],
    }
    rag_config = {
        "chunking": {
            "max_chars": 1200,
            "overlap": 120,
            "section_markers": [
                r"(?im)^\s*[০-৯]+[।.)]\s+",
            ],
        }
    }
    chunks = build_law_chunks(law, rag_config)
    assert len(chunks) >= 2
    assert chunks[0]["section_id"] == "1"
