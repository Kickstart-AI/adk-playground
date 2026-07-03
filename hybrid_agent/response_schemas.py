"""Structured response schemas for the hybrid agent's LLM nodes."""

from pydantic import BaseModel, Field


class SpeakerOption(BaseModel):
    """A user-facing label for one YAML answer option."""

    label: str


class SpeakerResponse(BaseModel):
    """User-facing speaker text, or a resolution when the transcript already answers.

    When skipping is allowed and the conversation already clearly answers the
    question, the speaker sets answered/answer (plain questions) or
    selected_index (multiple choice) instead of writing a message.
    """

    message: str = ""
    answer_options: list[SpeakerOption] = Field(default_factory=list)
    answered: bool = False
    answer: str = ""
    selected_index: int = -1


class OptionSelection(BaseModel):
    """Selected answer option by zero-based index, or -1 when unclear."""

    selected_index: int
    reason: str


class IntakeResponse(BaseModel):
    """Triage result: a matching flow name, or the message to send instead."""

    flow: str = ""  # empty if no clear match yet
    message: str = ""


class Verdict(BaseModel):
    """Result of a reflection check."""

    passed: bool
    reason: str
