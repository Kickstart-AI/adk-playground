"""Anonymous candidate handling and rank aggregation for the committee agent."""

import random
from dataclasses import dataclass

from .schemas import AnswererAnswer, Source


@dataclass(frozen=True)
class Candidate:
    """An answer and its private originating model."""

    candidate_id: str
    model: str
    answer: str
    sources: list[Source]


@dataclass(frozen=True)
class RankedCandidate:
    """A model-neutral answer presented to a ranking agent."""

    candidate_id: str
    answer: str
    sources: list[Source]


def anonymize_answers(
    answers: dict[str, AnswererAnswer],
    rng: random.Random,
) -> list[Candidate]:
    """Shuffle answers and assign opaque identifiers."""
    items = list(answers.items())
    rng.shuffle(items)
    return [
        Candidate(
            candidate_id=f"response_{rng.getrandbits(32):08x}",
            model=model,
            answer=answer.answer,
            sources=answer.sources,
        )
        for model, answer in items
    ]


def ranking_view(candidates: list[Candidate], rng: random.Random) -> list[RankedCandidate]:
    """Return an independently shuffled view without model identities."""
    view = [
        RankedCandidate(
            candidate_id=candidate.candidate_id,
            answer=candidate.answer,
            sources=candidate.sources,
        )
        for candidate in candidates
    ]
    rng.shuffle(view)
    return view


def serialize_candidate(candidate: Candidate | RankedCandidate) -> dict:
    """Serialize an anonymous candidate with only provider-neutral source fields."""
    return {
        "candidate_id": candidate.candidate_id,
        "answer": candidate.answer,
        "sources": [source.model_dump() for source in candidate.sources],
    }


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
