from google.adk.events.event import Event
from google.genai import types

from .flow_schema import Action


def route_event(target: str, output, facts: dict) -> Event:
    """Build an event that routes to another node.

    Routing to intake also points current_step there, so follow-up turns
    land in intake instead of re-entering the step that routed away.
    """
    state: dict = {"facts": facts}
    if target == "intake":
        state["current_step"] = "intake"
    return Event(output=output, route=target, state=state)


def answered_event(action: Action, memo: dict):
    """Return a pass-route event when a message answer completes one."""
    if action.result and action.result.passed:
        return route_event(action.result.passed, memo["facts"], memo["facts"])
    return None


def message_event(
    text: str,
    state: dict | None = None,
    output=None,
    answer_options: list[dict] | None = None,
) -> Event:
    """Build an event that shows a message to the user."""
    if output is None and answer_options:
        output = {"message": text, "answer_options": answer_options}
    return Event(
        output=output,
        custom_metadata={"answer_options": answer_options} if answer_options else None,
        content=types.Content(role="model", parts=[types.Part.from_text(text=text)]),
        state=state or {},
    )
