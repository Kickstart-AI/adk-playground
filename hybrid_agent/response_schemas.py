"""Structured response schemas for the hybrid agent's LLM nodes."""

from pydantic import BaseModel, Field


class SpeakerOption(BaseModel):
    """A user-facing label for one YAML answer option."""

    label: str


class SpeakerResponse(BaseModel):
    """User-facing speaker text and localized option labels."""

    message: str
    answer_options: list[SpeakerOption] = Field(default_factory=list)


class OptionSelection(BaseModel):
    """Selected answer option by zero-based index, or -1 when unclear."""

    selected_index: int
    reason: str


class IntakeDecision(BaseModel):
    """Which flow matches the user's request."""

    flow: str = ""  # empty if no clear match yet


class Resolution(BaseModel):
    """Whether the conversation already answers a question we were about to ask."""

    answered: bool
    answer: str = ""
    reason: str


class Verdict(BaseModel):
    """Result of a reflection check."""

    passed: bool
    reason: str
