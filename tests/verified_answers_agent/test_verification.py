"""Tests for source support annotation."""

from verified_answers_agent.verification import Candidate, render_answers, render_final_answer


def test_all_statements_remain_in_order_with_support_or_unavailable() -> None:
    """Both supported and unmatched statements remain visible with their annotations."""
    candidates = [
        Candidate(sentence="First.", quote="First source."),
        Candidate(sentence="Second.", quote="Missing."),
        Candidate(sentence="Third.", quote="Third source."),
    ]
    assert render_answers(candidates, [(0.9, 0.8), (None, None), (0.2, 0.3)]) == (
        "First.\n===\nQUOTE: First source.\nSUPPORT: 0.9 (quote), 0.8 (document)\n\n"
        "Second.\n===\nQUOTE: Missing.\nSource support unavailable: quote not found\n\n"
        "Third.\n===\nQUOTE: Third source.\nSUPPORT: 0.2 (quote), 0.3 (document)"
    )


def test_final_answer_includes_separate_question_relevance() -> None:
    """The final answer retains source annotations and shows its relevance score."""
    assembled = "First.\n===\nQUOTE: First source.\nSUPPORT: 0.9 (quote), 0.8 (document)"
    assert render_final_answer(assembled, 0.75) == (assembled + "\n\nANSWERS QUESTION: 0.75")
