"""Tests for deterministic committee ranking and answer anonymization."""

import json
import random

import pytest
from committee_agent.agent import AnswererAnswer, Source
from committee_agent.committee import (
    anonymize_answers,
    ranking_view,
    select_winner,
    serialize_candidate,
)


def test_ranking_view_hides_models_and_shuffles_answers() -> None:
    """Ranking inputs must contain no provider identity or stable answer order."""
    answers = {
        "openai/gpt-5.6-sol": AnswererAnswer(
            answer="OpenAI answer",
            sources=[Source(url="https://example.com/openai", title="OpenAI", snippet="OA")],
        ),
        "gemini/gemini-3.1-pro-preview": AnswererAnswer(
            answer="Gemini answer",
            sources=[Source(url="https://example.com/gemini", title="Gemini", snippet="GE")],
        ),
        "anthropic/claude-fable-5": AnswererAnswer(
            answer="Anthropic answer",
            sources=[Source(url="https://example.com/anthropic", title="Anthropic", snippet="AN")],
        ),
    }
    candidates = anonymize_answers(answers, random.Random(7))

    first_view = ranking_view(candidates, random.Random(1))
    second_view = ranking_view(candidates, random.Random(5))
    serialized = json.dumps([serialize_candidate(candidate) for candidate in first_view])

    assert [candidate.candidate_id for candidate in first_view] != [
        candidate.candidate_id for candidate in second_view
    ]
    assert all(model not in serialized for model in answers)
    assert {candidate.answer for candidate in first_view} == {
        answer.answer for answer in answers.values()
    }


def test_structured_sources_preserve_order_duplicates_and_exact_values() -> None:
    """Anonymous candidates retain source order, duplicates, and field spelling."""
    sources = [
        Source(
            url="HTTPS://Example.com/path%2Fitem?b=2&a=1#Fragment",
            title="First title",
            snippet="Snippet with exact spacing",
        ),
        Source(
            url="https://example.com/path%2Fitem?b=2&a=1#Fragment",
            title="Second title",
            snippet="Second snippet",
        ),
        Source(
            url="HTTPS://Example.com/path%2Fitem?b=2&a=1#Fragment",
            title="First title",
            snippet="Snippet with exact spacing",
        ),
    ]
    answer = AnswererAnswer(answer="[Source](https://example.com)", sources=sources)

    candidate = anonymize_answers({"openai/gpt-5.6-sol": answer}, random.Random(7))[0]
    ranked = ranking_view([candidate], random.Random(1))[0]

    assert ranked.sources == sources
    assert [source.model_dump() for source in ranked.sources] == [
        source.model_dump() for source in sources
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"url": "ftp://example.com", "title": "Title", "snippet": "Snippet"},
        {"url": "example.com", "title": "Title", "snippet": "Snippet"},
        {"url": "https://", "title": "Title", "snippet": "Snippet"},
        {"url": "https://example.com:not-a-port", "title": "Title", "snippet": "Snippet"},
        {"url": "https://example.com:99999", "title": "Title", "snippet": "Snippet"},
        {"url": " https://example.com", "title": "Title", "snippet": "Snippet"},
        {"url": "https://example.com ", "title": "Title", "snippet": "Snippet"},
        {"url": "https://example.com", "title": "", "snippet": "Snippet"},
        {"url": "https://example.com", "title": " ", "snippet": "Snippet"},
        {"url": "https://example.com", "title": "Title", "snippet": "\t"},
    ],
)
def test_source_rejects_invalid_or_whitespace_fields(kwargs: dict[str, str]) -> None:
    """Source fields must be nonblank and preserve accepted values without trimming."""
    with pytest.raises(ValueError):
        Source(**kwargs)


def test_answerer_answer_accepts_empty_sources_and_preserves_values() -> None:
    """Answerers may return no sources while retaining exact accepted source values."""
    source = Source(
        url="HTTPS://Example.com/path%2Fitem?b=2&a=1#Fragment",
        title="First title",
        snippet="Snippet with exact spacing",
    )

    answer = AnswererAnswer(answer="[Source](https://example.com)", sources=[source])
    empty = AnswererAnswer(answer="No citations.", sources=[])

    assert answer.sources == [source]
    assert answer.sources[0].model_dump() == {
        "url": "HTTPS://Example.com/path%2Fitem?b=2&a=1#Fragment",
        "title": "First title",
        "snippet": "Snippet with exact spacing",
    }
    assert empty.sources == []


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com:0/path",
        "https://example.com:65535/path",
    ],
)
def test_source_accepts_boundary_ports_without_rewriting(url: str) -> None:
    """Valid boundary ports remain accepted and byte-for-byte unchanged."""
    source = Source(url=url, title="Title", snippet="Snippet")

    assert source.url == url


def test_select_winner_uses_borda_scores() -> None:
    """The candidate with the best aggregate rank must win."""
    ballots = [
        ["response_a", "response_b", "response_c"],
        ["response_b", "response_a", "response_c"],
        ["response_a", "response_c", "response_b"],
    ]

    assert select_winner(ballots) == "response_a"


def test_select_winner_breaks_ties_by_first_place_votes() -> None:
    """First-place votes must break equal Borda scores."""
    ballots = [
        ["response_b", "response_a", "response_c"],
        ["response_b", "response_c", "response_a"],
        ["response_a", "response_c", "response_b"],
    ]

    assert select_winner(ballots) == "response_b"
