from bdlaw_ollama.retrieval import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_prefers_shared_top_items() -> None:
    scores = reciprocal_rank_fusion([[5, 2, 1], [2, 5, 9]], k=60)
    assert scores[5] > scores[1]
    assert scores[2] > scores[1]
