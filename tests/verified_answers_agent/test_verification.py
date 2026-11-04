"""Tests for source support annotation."""

import pytest
from verified_answers_agent.verification import Candidate, render_answers, render_final_answer


def test_unmatched_quote_is_warned_and_still_has_support_score() -> None:
    """A missing quote is visible alongside its independently judged support score."""
    candidates = [
        Candidate(sentence="First.", quotes=["First source.", "Other source."]),
        Candidate(sentence="Second.", quotes=["First source.", "Missing.", "Also missing."]),
    ]
    assert render_answers(candidates, [0.9, 0.2], "First source. Other source.") == (
        "First.\n===\nQUOTE: First source.\nQUOTE: Other source.\nSUPPORT: 0.9\n\n"
        "Second.\n===\nQUOTE: First source.\nQUOTE: Missing.\n"
        "WARNING: quote 2 not found in document\nQUOTE: Also missing.\n"
        "WARNING: quote 3 not found in document\nSUPPORT: 0.2"
    )


def test_final_answer_includes_separate_question_relevance() -> None:
    """The final answer retains source annotations and shows its relevance score."""
    assembled = "First.\n===\nQUOTE: First source.\nSUPPORT: 0.9"
    assert render_final_answer(assembled, 0.75) == (assembled + "\n\n===\nANSWERS QUESTION: 0.75")


def test_formatting_variants_match_without_changing_displayed_quotes() -> None:
    """Formatting differences match while the original quote stays visible."""
    quote = '"STRASSE café" fine-tuning softword zero width'
    document = "“Straße cafe\u0301” fine-\n tuning soft\u00adword ze\u200bro\t width"
    candidate = Candidate(sentence="Answer.", quotes=[quote])
    assert render_answers([candidate], [0.9], document) == (
        f"Answer.\n===\nQUOTE: {quote}\nSUPPORT: 0.9"
    )
    assert render_answers([Candidate(sentence="Answer.", quotes=[document])], [0.9], quote) == (
        f"Answer.\n===\nQUOTE: {document}\nSUPPORT: 0.9"
    )


def test_changed_math_symbol_still_warns() -> None:
    """Normalization preserves meaningful mathematical differences."""
    candidate = Candidate(sentence="Answer.", quotes=["model πteacher"])
    assert render_answers([candidate], [0.9], "model teacher") == (
        "Answer.\n===\nQUOTE: model πteacher\nWARNING: quote 1 not found in document\nSUPPORT: 0.9"
    )


@pytest.mark.parametrize(
    ("sentence", "count"),
    [
        ("First. Second? Third!", 3),
        ("No punctuation", 0),
        ("One.", 1),
        ("One?", 1),
        ("One!", 1),
        ("The value is 3.14.", 1),
        ("The value is .5!", 1),
        ("First. 2 follow.", 2),
    ],
)
def test_sentence_count_warning_preserves_answer_and_ignores_quotes(sentence, count) -> None:
    """Flag answer punctuation counts other than one without counting source quotes."""
    quote = "First. Second? Third!"
    warning = (
        f"WARNING: answer has {count} sentence-ending marks (.?!); expected 1.\n"
        if count != 1
        else ""
    )
    assert render_answers([Candidate(sentence=sentence, quotes=[quote])], [0.9], quote) == (
        f"{sentence}\n===\n{warning}QUOTE: {quote}\nSUPPORT: 0.9"
    )
