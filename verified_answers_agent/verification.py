"""Render document-grounded answer candidates and support scores."""

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    """One answer sentence and its supporting verbatim document quote."""

    sentence: str = Field(description="One complete answer sentence.")
    quote: str = Field(description="An exact, contiguous quote from the document.")


class Candidates(BaseModel):
    """The answer agent's proposed sentences and quotes."""

    candidates: list[Candidate]


def render_answers(
    candidates: list[Candidate], probabilities: list[tuple[float | None, float | None]]
) -> str:
    """Preserve each statement and append its source support assessment."""
    lines = []
    for candidate, probability in zip(candidates, probabilities, strict=True):
        annotation = (
            "Source support unavailable: quote not found"
            if (probability[0] is None and probability[1] is None)
            else f"SUPPORT: {probability[0]} (quote), {probability[1]} (document)"
        )
        lines.append(f"{candidate.sentence}\n===\nQUOTE: {candidate.quote}\n{annotation}")
    return "\n\n".join(lines) if lines else "No answer found."


def render_final_answer(answer: str, relevance: float) -> str:
    """Append the answer-to-question judgment without changing the answer."""
    return f"{answer}\n\nANSWERS QUESTION: {relevance}"
