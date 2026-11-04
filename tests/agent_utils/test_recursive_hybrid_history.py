"""Exercise production history callbacks through the ADK Runner lifecycle."""

import asyncio
import json
from typing import cast

import hybrid_agent.agent as hybrid
import pytest
import recursive_agent.agent as recursive
from google.adk import Workflow
from google.adk.agents import LlmAgent
from google.adk.agents.base_agent import BaseAgent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import node
from google.genai import types
from pydantic import Field


class ScriptedModel(BaseLlm):
    """Capture requests and return deterministic protocol responses."""

    responses: list[types.Content]
    requests: list[LlmRequest] = Field(default_factory=list)

    @classmethod
    def supported_models(cls) -> list[str]:
        """Accept the test model name."""
        return ["scripted"]

    async def generate_content_async(self, llm_request, stream=False):
        """Record callback output before emitting the next scripted response."""
        self.requests.append(llm_request.model_copy(deep=True))
        yield LlmResponse(content=self.responses.pop(0))


def reply(payload: dict | str) -> types.Content:
    """Build a deterministic assistant response."""
    return types.Content(
        role="model",
        parts=[types.Part(text=json.dumps(payload) if isinstance(payload, dict) else payload)],
    )


def text_of(request: LlmRequest) -> list[str]:
    """Extract request text without dropping role or tool data from the request."""
    return [
        part.text for content in request.contents for part in (content.parts or []) if part.text
    ]


async def run_turn(runner: Runner, question: str):
    """Run one real user turn without invoking a remote model."""
    async for _ in runner.run_async(
        user_id="user",
        session_id="history-test",
        new_message=types.Content(role="user", parts=[types.Part(text=question)]),
        yield_user_message=True,
    ):
        pass


def runner_for(agent) -> Runner:
    """Create an isolated session store for a production workflow."""
    return Runner(
        app_name=agent.name,
        agent=cast(BaseAgent, agent),
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )


@pytest.mark.parametrize("rewritten", ["Explain the energy tradeoff.", "Question one"])
def test_recursive_production_workflow_keeps_dialogue_and_explicit_rewrite(monkeypatch, rewritten):
    """Rewrites survive recursion while original requests and schema outputs do not repeat."""
    schema = {"name": "improved", "description": "Focused", "instruction": "Focus on energy."}
    model = ScriptedModel(
        model="scripted",
        responses=[
            reply({"agent_schema": schema, "user_prompt": rewritten}),
            reply({"agent_schema": schema}),
            reply({"direct_answer": "Visible answer one"}),
            reply({"direct_answer": "Visible answer two"}),
        ],
    )
    monkeypatch.setattr(recursive.initial_agent, "model", model)
    monkeypatch.setattr(recursive, "state", {})
    original_construct = recursive.construct_agent

    async def construct(schema, iteration):
        """Keep production construction and replace only remote model access."""
        agent = await original_construct(schema, iteration)
        agent.model = model
        return agent

    monkeypatch.setattr(recursive, "construct_agent", construct)

    async def scenario():
        """Use one event loop for successive turns in the real production workflow."""
        runner = runner_for(recursive.root_agent)
        await run_turn(runner, "Question one")
        await run_turn(runner, "Question two")

    asyncio.run(scenario())
    assert len(model.requests) == 4
    for index, request in enumerate(model.requests):
        texts = text_of(request)
        assert texts.count("Question one") == 1
        assert all("current_request" not in text and "agent_schema" not in text for text in texts)
        payload = json.loads(texts[-1])
        assert payload["task"] == "Answer the latest user request or improve the agent."
        if index in (1, 2) and rewritten != "Question one":
            assert payload["rewritten_request"] == rewritten
            assert sum(rewritten in text for text in texts) == 1
        else:
            assert "rewritten_request" not in payload
    assert text_of(model.requests[-1]).count("Question two") == 1
    assert text_of(model.requests[-1]).count("Visible answer one") == 1


def test_hybrid_production_workflow_keeps_visible_replies_only(monkeypatch):
    """Routing and speaker JSON stay private while messages survive subsequent turns."""
    model = ScriptedModel(
        model="scripted",
        responses=[
            reply({"flow": "return_item"}),
            reply({"message": "Which order?"}),
            reply({"order_number": "", "transfer_to_agent": "intake"}),
            reply({"message": "How else can I help?"}),
            reply({"message": "Goodbye"}),
        ],
    )
    for agent in (hybrid.intake_agent, hybrid.speaker, *hybrid.extractors.values()):
        monkeypatch.setattr(agent, "model", model)

    async def scenario():
        """Exercise intake, speaker, extractor handoff and another intake turn."""
        runner = runner_for(hybrid.build_workflow())
        await run_turn(runner, "I want to return an item")
        await run_turn(runner, "Cancel that")
        await run_turn(runner, "Nothing else")

    asyncio.run(scenario())
    assert len(model.requests) == 5
    for request in model.requests:
        texts = text_of(request)
        assert texts.count("I want to return an item") == 1
        assert not any('"flow": "return_item"' in text for text in texts)
        assert not any('"message": "Which order?"' in text for text in texts)
        assert not any('"transfer_to_agent": "intake"' in text for text in texts)
    for request in model.requests[2:]:
        texts = text_of(request)
        assert texts.count("Cancel that") == 1
        assert texts.count("Which order?") == 1
    latest = text_of(model.requests[-1])
    assert latest.count("Nothing else") == 1
    assert latest.count("How else can I help?") == 1


@pytest.mark.parametrize(
    "callback", [recursive.initial_agent.before_model_callback, hybrid.HISTORY_CALLBACK]
)
def test_active_tool_continuation_survives_history_filter(callback):
    """An active tool call and its result remain paired through the real ADK flow."""
    model = ScriptedModel(
        model="scripted",
        responses=[
            types.Content(
                role="model",
                parts=[types.Part(function_call=types.FunctionCall(name="lookup", args={}))],
            ),
            reply("Tool-backed answer"),
        ],
    )

    def lookup() -> dict:
        """Return a deterministic tool result."""
        return {"value": "looked-up"}

    child = LlmAgent(
        name="Agent_tool_test",
        model=model,
        tools=[lookup],
        include_contents="none",
        before_model_callback=callback,
    )

    async def workflow_node(ctx, node_input):
        """Invoke the tool-enabled child as a workflow node."""
        return await ctx.run_node(child, {"task": "Use lookup."})

    workflow = Workflow(
        name="tool_test", edges=[("START", node(workflow_node, rerun_on_resume=True))]
    )
    asyncio.run(run_turn(runner_for(workflow), "Find the value"))
    assert len(model.requests) == 2
    parts = [part for content in model.requests[-1].contents for part in (content.parts or [])]
    assert sum(part.function_call is not None for part in parts) == 1
    assert sum(part.function_response is not None for part in parts) == 1
    assert text_of(model.requests[-1]).count("Find the value") == 1
