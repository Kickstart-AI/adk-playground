"""Regression tests for conversation history callback construction."""

from types import SimpleNamespace
from typing import cast

from agent_utils.conversation import ConversationHistoryCallback
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.genai import types


def event(
    author,
    **kwargs,
):
    """Create the small event surface used by the callback."""
    return SimpleNamespace(
        author=author,
        content=kwargs.get("content"),
        output=kwargs.get("output"),
        isolation_scope=kwargs.get("isolation_scope"),
        route=kwargs.get("route"),
        invocation_id=kwargs.get("invocation_id", "invocation"),
    )


def content(*parts, role="user"):
    """Build a model content value from parts."""
    return types.Content(role=role, parts=list(parts))


def parts(item: types.Content) -> list[types.Part]:
    """Return the parts of content after asserting the test fixture has them."""
    assert item.parts is not None
    return item.parts


def callback_context(events) -> CallbackContext:
    """Adapt the minimal callback fixture to ADK's callback context type."""
    return cast(CallbackContext, SimpleNamespace(session=SimpleNamespace(events=events)))


def test_callback_preserves_current_tool_chain_and_cleans_only_json_history_fields():
    """Current function call/response parts must survive callback reconstruction."""
    current = [
        content(types.Part(text='{"conversation_history": [], "question": "now"}')),
        content(
            types.Part(function_call=types.FunctionCall(name="lookup", args={"q": "now"})),
            role="model",
        ),
        content(
            types.Part(
                function_response=types.FunctionResponse(
                    name="lookup", response={"result": "found"}
                )
            )
        ),
    ]
    request = LlmRequest(contents=current)
    callback = ConversationHistoryCallback({"user"})
    callback(
        callback_context([event("user", content=content(types.Part(text="before")))]),
        request,
    )

    assert [part.text for part in parts(request.contents[0])] == ["before"]
    assert parts(request.contents[1])[0].text == '{"question": "now"}'
    function_call = parts(request.contents[2])[0].function_call
    function_response = parts(request.contents[3])[0].function_response
    assert function_call is not None
    assert function_call.name == "lookup"
    assert function_response is not None
    assert function_response.name == "lookup"


def test_callback_converts_output_and_hybrid_input_tool_events_to_visible_content():
    """Workflow output and adk_request_input protocol events become plain text."""
    callback = ConversationHistoryCallback({"user", "hybrid_agent", "speaker", "tool"})
    request = LlmRequest(contents=[content(types.Part(text='{"current": true}'))])
    callback(
        callback_context(
            [
                event("hybrid_agent", output="A workflow reply"),
                event("hybrid_agent", output={"route": "secret", "state": {}}),
                event(
                    "user",
                    content=content(types.Part(text="isolated")),
                    isolation_scope="internal",
                    invocation_id="",
                ),
                event(
                    "speaker",
                    content=content(
                        types.Part(
                            function_call=types.FunctionCall(
                                name="adk_request_input",
                                args={"message": "What is your account?"},
                            )
                        )
                    ),
                ),
                event(
                    "tool",
                    content=content(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name="adk_request_input",
                                response={"result": "A-123"},
                            )
                        )
                    ),
                ),
                event(
                    "tool",
                    content=content(
                        types.Part(function_call=types.FunctionCall(name="secret", args={}))
                    ),
                ),
            ]
        ),
        request,
    )

    assert [part.text for item in request.contents[:-1] for part in parts(item)] == [
        "A workflow reply",
        "What is your account?",
        "A-123",
    ]
    assert all(
        not any(part.function_call or part.function_response for part in parts(item))
        for item in request.contents[:-1]
    )


def test_callback_excludes_routed_string_output():
    """Internal route reasons must not become visible conversation history."""
    request = LlmRequest(contents=[content(types.Part(text='{"current": true}'))])
    ConversationHistoryCallback({"hybrid_agent"})(
        callback_context([event("hybrid_agent", output="internal verdict", route="next")]),
        request,
    )

    assert all(part.text != "internal verdict" for item in request.contents for part in parts(item))


def test_callback_skips_synthetic_empty_invocation_user_event():
    """Workflow node-input events are not conversation turns."""
    payload = content(types.Part(text='{"task": "Answer the latest user question."}'))
    request = LlmRequest(contents=[content(types.Part(text='{"task": "next"}'))])
    ConversationHistoryCallback({"user"})(
        callback_context(
            [
                event("user", content=content(types.Part(text="real question"))),
                event("user", content=payload, invocation_id=""),
            ]
        ),
        request,
    )

    assert [part.text for item in request.contents for part in parts(item)] == [
        "real question",
        '{"task": "next"}',
    ]


def test_callback_keeps_user_event_with_nonempty_invocation_id():
    """Runner-created user turns remain visible even when they look synthetic."""
    request = LlmRequest(contents=[content(types.Part(text='{"task": "next"}'))])
    ConversationHistoryCallback({"user"})(
        callback_context(
            [
                event(
                    "user",
                    content=content(types.Part(text="real question")),
                    invocation_id="runner-turn",
                )
            ]
        ),
        request,
    )

    assert parts(request.contents[0])[0].text == "real question"


def test_callback_deduplicates_current_payload_already_recorded_as_latest_event():
    """The current user payload occurs once while its complete tool chain remains."""
    payload = content(types.Part(text='{"current_request": "now"}'))
    request = LlmRequest(
        contents=[
            payload,
            content(types.Part(function_call=types.FunctionCall(name="lookup")), role="model"),
            content(
                types.Part(
                    function_response=types.FunctionResponse(
                        name="lookup", response={"result": "done"}
                    )
                )
            ),
        ]
    )
    callback = ConversationHistoryCallback({"user"})
    callback(
        callback_context([event("user", content=payload)]),
        request,
    )

    assert (
        sum(
            part.text == '{"current_request": "now"}'
            for item in request.contents
            for part in parts(item)
        )
        == 1
    )
    function_call = parts(request.contents[-2])[0].function_call
    function_response = parts(request.contents[-1])[0].function_response
    assert function_call is not None
    assert function_call.name == "lookup"
    assert function_response is not None
    assert function_response.name == "lookup"


def test_callback_excludes_scoped_committee_payloads_but_keeps_current_payload_once():
    """Scoped committee node payloads stay out of later model history."""
    current_payload = '{"task": "Answer the latest user question."}'
    request = LlmRequest(contents=[content(types.Part(text=current_payload))])
    ConversationHistoryCallback({"user", "committee_agent"})(
        callback_context(
            [
                event(
                    "user",
                    content=content(types.Part(text=current_payload)),
                    isolation_scope="committee:answer:0",
                    invocation_id="",
                ),
                event(
                    "user",
                    content=content(types.Part(text=current_payload)),
                    isolation_scope="committee:ranking:0",
                    invocation_id="",
                ),
            ]
        ),
        request,
    )

    assert [part.text for item in request.contents for part in parts(item)] == [current_payload]


def test_callback_keeps_scoped_genuine_user_and_filters_scoped_synthetic_and_internal_events():
    """Only runner-created scoped user turns belong in reconstructed history."""
    request = LlmRequest(contents=[content(types.Part(text='{"task": "answer"}'))])
    ConversationHistoryCallback({"user", "committee_agent"})(
        callback_context(
            [
                event(
                    "user",
                    content=content(
                        types.Part(text="would you believe it if I told you agent swarms...")
                    ),
                    isolation_scope="committee:answer:0",
                    invocation_id="0f312d0e0f10441d3df643440a99cb64",
                ),
                event(
                    "user",
                    content=content(types.Part(text='{"task": "synthetic"}')),
                    isolation_scope="committee:answer:0",
                    invocation_id="",
                ),
                event(
                    "committee_agent",
                    content=content(types.Part(text="private committee output")),
                    isolation_scope="committee:answer:0",
                    invocation_id="child-invocation",
                ),
            ]
        ),
        request,
    )

    assert [part.text for item in request.contents for part in parts(item) if part.text] == [
        "would you believe it if I told you agent swarms...",
        '{"task": "answer"}',
    ]
