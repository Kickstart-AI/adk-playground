"""Render document-grounded answer candidates and support scores."""

import re
import unicodedata

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    """A list of verbatim quotes from the document and the
    answer sentence that is supported by the quotes."""

    quotes: list[str] = Field(
        description="A list of exact, contiguous quotes from the document "
        "that support the answer sentence."
    )

    sentence: str = Field(
        description="One self-contained sentence stating one independently verifiable fact."
    )


class Candidates(BaseModel):
    """The answer agent's proposed sentences and quotes."""

    candidates: list[Candidate]


def repair_line_breaks(text: str) -> str:
    """Join a word after a line-ending hyphen, preserving the hyphen."""
    return re.sub(r"(?<=\w)-[^\S\r\n]*\r?\n[^\S\r\n]*(?=\w)", "-", text)


def normalize_quote(text: str) -> str:
    """Normalize formatting for source matching without changing displayed text."""
    text = text.translate(
        str.maketrans(
            {
                "“": '"',
                "”": '"',
                "\u2018": "'",
                "\u2019": "'",
                "\u00ad": "",
                "\u200b": "",
            }
        )
    )
    text = unicodedata.normalize("NFC", text.casefold())
    return " ".join(repair_line_breaks(text).split())


def render_answers(candidates: list[Candidate], probabilities: list[float], document: str) -> str:
    """Preserve each statement and append its source support assessment."""
    normalized_document = normalize_quote(document)
    lines = []
    for candidate, probability in zip(candidates, probabilities, strict=True):
        chunk = f"{candidate.sentence}\n===\n"
        count = len(re.findall(r"[?!]|\.(?!\d)", candidate.sentence))
        if count != 1:
            chunk += f"WARNING: answer has {count} sentence-ending marks (.?!); expected 1.\n"
        for index, quote in enumerate(candidate.quotes, start=1):
            chunk += f"QUOTE: {quote}\n"
            normalized_quote = normalize_quote(quote)
            if not normalized_quote or normalized_quote not in normalized_document:
                chunk += f"WARNING: quote {index} not found in document\n"

        chunk += f"SUPPORT: {probability}"

        lines.append(chunk)
    return "\n\n".join(lines) if lines else "No answer found."


def render_final_answer(answer: str, relevance: float) -> str:
    """Append the answer-to-question judgment without changing the answer."""
    return f"{answer}\n\n===\nANSWERS QUESTION: {relevance}"
