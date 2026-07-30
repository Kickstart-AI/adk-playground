"""Anonymous candidate handling and rank aggregation for the committee agent."""

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """An answer and its private originating model."""

    candidate_id: str
    model: str
    answer: str


@dataclass(frozen=True)
class RankedCandidate:
    """A model-neutral answer presented to a ranking agent."""

    candidate_id: str
    answer: str


def anonymize_answers(answers: dict[str, str], rng: random.Random) -> list[Candidate]:
    """Shuffle answers and assign opaque identifiers."""
    items = list(answers.items())
    rng.shuffle(items)
    return [
        Candidate(
            candidate_id=f"response_{rng.getrandbits(32):08x}",
            model=model,
            answer=answer,
        )
        for model, answer in items
    ]


def ranking_view(candidates: list[Candidate], rng: random.Random) -> list[RankedCandidate]:
    """Return an independently shuffled view without model identities."""
    view = [
        RankedCandidate(candidate_id=candidate.candidate_id, answer=candidate.answer)
        for candidate in candidates
    ]
    rng.shuffle(view)
    return view


def select_winner(ballots: list[list[str]]) -> str:
    """Select a candidate with Borda count and deterministic tie breakers."""
    candidate_ids = set(ballots[0])
    scores = dict.fromkeys(candidate_ids, 0)
    first_place_votes = dict.fromkeys(candidate_ids, 0)

    for ballot in ballots:
        for rank, candidate_id in enumerate(ballot):
            scores[candidate_id] += len(ballot) - rank - 1
        first_place_votes[ballot[0]] += 1

    return min(
        candidate_ids,
        key=lambda candidate_id: (
            -scores[candidate_id],
            -first_place_votes[candidate_id],
            candidate_id,
        ),
    )
