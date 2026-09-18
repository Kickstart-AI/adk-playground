"""Regression coverage for history callback installation and privacy boundaries."""

import asyncio
from types import SimpleNamespace
from typing import cast

import committee_agent.agent as committee
import deep_research_agent.agent as deep_research
import hybrid_agent.agent as hybrid
import recursive_agent.agent as recursive
from agent_utils.conversation import ConversationHistoryCallback
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.adk.tools.agent_tool import AgentTool
from google.genai import types


def test_history_callbacks_are_wired_to_visible_agents_only():
    """Every agent receiving reconstructed history has the shared callback."""
    assert deep_research.planner_agent.before_model_callback is not None
    for agent in (
        hybrid.intake_agent,
        hybrid.speaker,
        hybrid.reflector,
        hybrid.option_selector,
        *hybrid.extractors.values(),
    ):
        assert agent.before_model_callback is hybrid.HISTORY_CALLBACK
        assert agent.before_model_callback.agent_names == frozenset({"user", "hybrid_agent"})
        assert not agent.before_model_callback.prefixes

    for member in committee.MEMBERS:
        for agent in (member.answerer, member.ranker, member.synthesizer):
            callback = cast(ConversationHistoryCallback, agent.before_model_callback)
            assert callback.agent_names == frozenset({"user", "committee_agent"})
            assert not any("member_" in name for name in callback.agent_names)


def test_recursive_dynamic_parent_and_nested_agents_get_history_callback():
    """Runtime-created recursive agents, including AgentTool children, retain wiring."""
    schema = recursive.AgentSchema(
        name="parent",
        description="parent",
        instruction="parent instruction",
        sub_agents=[
            recursive.AgentSchema(
                name="child", description="child", instruction="child instruction"
            )
        ],
    )
    agent = asyncio.run(recursive.construct_agent(schema, 1))

    nested = cast(AgentTool, agent.tools[0]).agent
    for candidate in (
        recursive.initial_agent,
        agent,
        cast(LlmAgent, nested),
    ):
        callback = cast(ConversationHistoryCallback, candidate.before_model_callback)
        assert callback.agent_names == frozenset({"user", "recursive_agent"})
        assert not callback.prefixes


def test_committee_callback_filters_private_member_events():
    """Committee history contains only user and anonymous root-agent dialogue."""
    callback = ConversationHistoryCallback({"user", "committee_agent"})
    request = LlmRequest(contents=[])

    def make_event(author, text, isolation_scope=None, invocation_id="runner-turn"):
        return SimpleNamespace(
            author=author,
            content=types.Content(parts=[types.Part(text=text)]),
            output=None,
            isolation_scope=isolation_scope,
            route=None,
            invocation_id=invocation_id,
        )

    callback(
        cast(
            CallbackContext,
            SimpleNamespace(
                session=SimpleNamespace(
                    events=[
                        make_event("user", "Question"),
                        make_event("committee_agent", "Answer"),
                        make_event("member_model_answerer", "Private answer"),
                        make_event(
                            "user",
                            '{"internal": true}',
                            "committee/internal",
                            invocation_id="",
                        ),
                    ]
                )
            ),
        ),
        request,
    )

    assert [part.text for item in request.contents for part in (item.parts or [])] == [
        "Question",
        "Answer",
    ]
