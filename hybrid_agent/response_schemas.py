"""Structured response schemas for the hybrid agent's LLM nodes."""

from pydantic import BaseModel, Field


class SpeakerOption(BaseModel):
    """A user-facing label for one YAML answer option."""

    label: str = Field(..., description="Localized label to show for one YAML answer option.")


class SpeakerResponse(BaseModel):
    """Speaker output, or an in-transcript resolution when skipping is allowed."""

    message: str = Field(
        "",
        description="User-facing text to send. Leave empty when resolving from transcript.",
    )
    answer_options: list[SpeakerOption] = Field(
        default_factory=list,
        description="Localized labels for YAML-provided answer options, in the same order.",
    )
    resolved_answer: str = Field(
        "",
        description="Answer already present in the transcript for a plain message action.",
    )
    selected_index: int = Field(
        -1,
        description="Zero-based option index already selected in the transcript, or -1.",
    )
    transfer_to_agent: str = Field(
        "",
        description="Workflow target to route to instead of sending a message, currently intake.",
    )


class OptionSelection(BaseModel):
    """Selected answer option by zero-based index, or -1 when unclear."""

    selected_index: int = Field(
        ...,
        description="Zero-based selected option index, or -1 when no option is clear.",
    )
    reason: str = Field(..., description="Brief explanation for the selected index.")
    transfer_to_agent: str = Field(
        "",
        description="Workflow target to route to instead of selecting, currently intake.",
    )


class IntakeResponse(BaseModel):
    """Triage result: a matching flow name, or the message to send instead."""

    flow: str = Field("", description="Matched flow name, or empty when no flow is clear.")
    message: str = Field("", description="User-facing intake message when no flow is routed.")


class Verdict(BaseModel):
    """Result of a reflection check."""

    passed: bool = Field(..., description="Whether the validation instruction holds.")
    reason: str = Field(..., description="Brief explanation of the verdict.")
    transfer_to_agent: str = Field(
        "",
        description="Workflow target to route to instead of judging, currently intake.",
    )
