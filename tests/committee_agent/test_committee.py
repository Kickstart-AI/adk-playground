"""Tests for deterministic committee ranking and answer anonymization."""

import json
import random

from committee_agent.committee import (
    anonymize_answers,
    ranking_view,
    select_winner,
)


def test_ranking_view_hides_models_and_shuffles_answers() -> None:
    """Ranking inputs must contain no provider identity or stable answer order."""
    answers = {
        "openai/gpt-5.6-sol": "OpenAI answer",
        "gemini/gemini-3.1-pro-preview": "Gemini answer",
        "anthropic/claude-fable-5": "Anthropic answer",
    }
    candidates = anonymize_answers(answers, random.Random(7))

    first_view = ranking_view(candidates, random.Random(1))
    second_view = ranking_view(candidates, random.Random(5))
    serialized = json.dumps([candidate.__dict__ for candidate in first_view])

    assert [candidate.candidate_id for candidate in first_view] != [
        candidate.candidate_id for candidate in second_view
    ]
    assert all(model not in serialized for model in answers)
    assert {candidate.answer for candidate in first_view} == set(answers.values())


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
