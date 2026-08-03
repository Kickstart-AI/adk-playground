"""Structured response schemas shared by committee workflow components."""

from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator


class Source(BaseModel):
    """A provider-neutral web source supporting an answer."""

    url: str = Field(description="The source URL, using HTTP or HTTPS.")
    title: str = Field(description="The descriptive source title.")
    snippet: str = Field(description="The source snippet supporting the answer.")

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        """Require an unchanged, syntactically valid HTTP(S) URL."""
        if not value or value != value.strip() or any(character.isspace() for character in value):
            raise ValueError("url must be nonblank and contain no whitespace")
        try:
            parsed = urlsplit(value)
        except ValueError as error:
            raise ValueError("url must be a valid HTTP(S) URL") from error
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ValueError("url must be a valid HTTP(S) URL")
        try:
            if not parsed.hostname:
                raise ValueError("url must be a valid HTTP(S) URL")
        except ValueError as error:
            raise ValueError("url must be a valid HTTP(S) URL") from error
        return value

    @field_validator("title", "snippet")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Require nonblank text without trimming accepted values."""
        if not value or value != value.strip():
            raise ValueError("source text must be nonblank and have no surrounding whitespace")
        return value


class AnswererAnswer(BaseModel):
    """An answerer response with ordered provider-neutral sources."""

    answer: str = Field(description="The complete proposed answer with inline Markdown links.")
    sources: list[Source] = Field(
        description="Ordered supporting sources from provider-native search; may be empty."
    )
