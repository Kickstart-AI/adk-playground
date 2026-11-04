"""Callbacks for supplying session conversation history to LLM agents."""

from __future__ import annotations

import json
from collections.abc import Collection

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.genai import types


class ConversationHistoryCallback:
    """Populate an LLM request from visible session events and its node payload.

    Workflow node payloads are represented as the latest request content by ADK. The
    callback removes history fields from that payload, while retaining task-specific
    fields such as the current question, and reconstructs prior turns from events.
    """

    def __init__(self, agent_names: Collection[str], *, prefixes: Collection[str] = ()) -> None:
        """Create a callback for the selected assistant event authors."""
        self.agent_names = frozenset(agent_names)
        self.prefixes = tuple(prefixes)

    def __call__(self, callback_context: CallbackContext, llm_request: LlmRequest) -> None:
        """Prepend visible session history while retaining the complete node request."""
        contents = []
        for event in callback_context.session.events:
            # ADK 2.2.0's single_turn workflow wrapper records synthetic node inputs
            # as user events with an empty invocation ID. Runner-created user turns
            # have a non-empty ID and remain part of the visible conversation.
            if event.author == "user" and not getattr(event, "invocation_id", ""):
                continue
            # A real runner-created user turn can be scoped while a delegated
            # workflow is active. Keep that turn, but hide scoped internal
            # agent events from sibling model calls.
            if event.isolation_scope and event.author != "user":
                continue
            if event.author not in self.agent_names and not any(
                event.author.startswith(prefix) for prefix in self.prefixes
            ):
                continue
            event_content = _event_content(event)
            if event_content is not None:
                contents.append(event_content)

        for content in llm_request.contents:
            cleaned = _clean_payload_content(content)
            if cleaned is not None and not (contents and _same_content(contents[-1], cleaned)):
                contents.append(cleaned)
        llm_request.contents = contents


def _event_content(event) -> types.Content | None:
    """Convert a visible event to content, hiding historical tool protocol parts."""
    if event.content is not None:
        text_parts = []
        for part in event.content.parts or []:
            if part.text:
                text_parts.append(types.Part(text=part.text))
                continue
            if (not part.function_call or part.function_call.name != "adk_request_input") and (
                not part.function_response or part.function_response.name != "adk_request_input"
            ):
                continue
            if part.function_call:
                message = (part.function_call.args or {}).get("message")
            else:
                message = (part.function_response.response or {}).get("result")
            if message is not None:
                text_parts.append(types.Part(text=str(message)))
        if text_parts:
            return types.Content(role=event.content.role, parts=text_parts)
        return None
    if not getattr(event, "route", None) and isinstance(event.output, str) and event.output:
        return types.Content(role="model", parts=[types.Part(text=event.output)])
    return None


def _same_content(left: types.Content, right: types.Content) -> bool:
    """Compare content values without depending on generated model equality."""
    return left.model_dump(exclude_none=True) == right.model_dump(exclude_none=True)


def _clean_payload_content(content: types.Content | None) -> types.Content | None:
    """Remove callback-managed history keys from JSON text parts only."""
    if content is None or not content.parts:
        return None
    cleaned_parts = []
    for part in content.parts:
        if not part.text:
            cleaned_parts.append(part)
            continue
        try:
            payload = json.loads(part.text)
        except json.JSONDecodeError:
            cleaned_parts.append(part)
            continue
        if not isinstance(payload, dict):
            cleaned_parts.append(part)
            continue
        payload.pop("conversation_history", None)
        payload.pop("transcript", None)
        if payload:
            cleaned_parts.append(types.Part(text=json.dumps(payload)))
    if not cleaned_parts:
        return None
    return content.model_copy(update={"parts": cleaned_parts})
