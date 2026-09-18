"""Regression test for synthetic workflow node-input history."""

import asyncio
from types import SimpleNamespace
from typing import ClassVar, cast

import deep_research_agent.agent as deep_research
from agent_utils.conversation import ConversationHistoryCallback
from google.adk import Workflow
from google.adk.agents import LlmAgent
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.context import Context
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import node
from google.genai import types
from pydantic import Field


class CapturingModel(BaseLlm):
    """Deterministic model that records every callback-mutated request."""

    requests: ClassVar[list[LlmRequest]] = []

    @classmethod
    def supported_models(cls) -> list[str]:
        """Accept the local test model name."""
        return ["capture"]

    async def generate_content_async(self, llm_request, stream=False):
        """Record the request and return valid structured output."""
        self.requests.append(llm_request.model_copy(deep=True))
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text='{"answer": "ok"}')],
            )
        )


class QueuedModel(BaseLlm):
    """Return queued responses while retaining every callback-mutated request."""

    responses: list[types.Content]
    requests: list[LlmRequest] = Field(default_factory=list)

    @classmethod
    def supported_models(cls) -> list[str]:
        """Accept the local test model name."""
        return ["queued"]

    async def generate_content_async(self, llm_request, stream=False):
        """Capture a request and emit the next deterministic response."""
        self.requests.append(llm_request.model_copy(deep=True))
        yield LlmResponse(content=self.responses.pop(0))


def parts(item: types.Content) -> list[types.Part]:
    """Return fixture content parts after asserting they are present."""
    assert item.parts is not None
    return item.parts


def test_runner_workflow_child_calls_do_not_replay_synthetic_payloads():
    """Each single-turn child sees the question and only its current payload."""
    CapturingModel.requests.clear()
    model = CapturingModel(model="capture")
    child = LlmAgent(
        name="child",
        model=model,
        mode="single_turn",
        include_contents="none",
        instruction="Return the answer field.",
        before_model_callback=ConversationHistoryCallback({"user"}),
        output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )

    async def workflow_node(ctx, node_input):
        """Run three children with distinct static node payloads."""
        for index in range(3):
            await ctx.run_node(
                child,
                {"task": f"child-{index}"},
                run_id=f"child-{index}",
                override_isolation_scope=f"committee:answer:{index}",
            )
        return "done"

    workflow = Workflow(
        name="lifecycle_test",
        edges=[("START", node(workflow_node, rerun_on_resume=True))],
    )
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="lifecycle_test",
        agent=cast(BaseAgent, workflow),
        session_service=session_service,
        auto_create_session=True,
    )

    async def run(question):
        """Drive the real Runner, workflow, and single-turn child lifecycle."""
        async for _ in runner.run_async(
            user_id="user",
            session_id="session",
            invocation_id=question,
            new_message=types.Content(role="user", parts=[types.Part(text=question)]),
            yield_user_message=True,
        ):
            pass

    asyncio.run(run("What is frontier AI?"))
    first_turn = CapturingModel.requests[:]
    asyncio.run(run("How should frontier AI be governed?"))
    session = asyncio.run(
        session_service.get_session(app_name="lifecycle_test", user_id="user", session_id="session")
    )
    assert session is not None
    assert len(CapturingModel.requests) == 6
    user_turns = [
        part.text
        for event in session.events
        if event.author == "user" and event.content
        for part in parts(event.content)
        if part.text
    ]
    assert user_turns == [
        "What is frontier AI?",
        "How should frontier AI be governed?",
    ]
    for index, request in enumerate(first_turn):
        text = [part.text for content in request.contents for part in parts(content) if part.text]
        assert text.count("What is frontier AI?") == 1
        assert text.count(f'{{"task": "child-{index}"}}') == 1
        assert all(f'"child-{other}"' not in item for other in range(index) for item in text)
    for index, request in enumerate(CapturingModel.requests[3:]):
        text = [part.text for content in request.contents for part in parts(content) if part.text]
        assert text.count("How should frontier AI be governed?") == 1
        assert text.count("What is frontier AI?") == 1
        assert text.count(f'{{"task": "child-{index}"}}') == 1
        assert all(
            f'"child-{other}"' not in item for other in range(3) if other != index for item in text
        )


def test_planner_static_task_uses_callback_history_for_two_turns():
    """Planner requests contain each user turn once, without current_request JSON."""
    CapturingModel.requests.clear()
    planner = LlmAgent(
        name="Planner_Agent",
        model=CapturingModel(model="capture"),
        mode="single_turn",
        include_contents="none",
        instruction="Return the answer field.",
        before_model_callback=ConversationHistoryCallback({"user", "Planner_Agent"}),
        output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )

    async def workflow_node(ctx, node_input):
        """Invoke the planner with its static workflow directive."""
        del node_input
        return await ctx.run_node(
            planner,
            {"task": "Plan the deep-research workflow for the latest user request."},
        )

    workflow = Workflow(
        name="planner_history_test",
        edges=[("START", node(workflow_node, rerun_on_resume=True))],
    )
    runner = Runner(
        app_name="planner_history_test",
        agent=cast(BaseAgent, workflow),
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )

    async def run(question):
        """Drive one planner turn through the real Runner lifecycle."""
        async for _ in runner.run_async(
            user_id="user",
            session_id="session",
            invocation_id=question,
            new_message=types.Content(role="user", parts=[types.Part(text=question)]),
            yield_user_message=True,
        ):
            pass

    asyncio.run(run("What is frontier AI?"))
    asyncio.run(run("How should frontier AI be governed?"))

    assert len(CapturingModel.requests) == 2
    for request, question in zip(
        CapturingModel.requests,
        ("What is frontier AI?", "How should frontier AI be governed?"),
        strict=True,
    ):
        text = [part.text for content in request.contents for part in parts(content) if part.text]
        assert sum(item == question for item in text) == 1
        assert (
            text.count('{"task": "Plan the deep-research workflow for the latest user request."}')
            == 1
        )
        assert all("current_request" not in item for item in text)


def test_deep_research_workflow_passes_static_planner_task():
    """Production workflow sends only the static planner task to its child node."""
    planner_payloads = []

    class StubContext:
        """Capture production workflow child payloads without calling model APIs."""

        state: ClassVar[dict] = {}
        actions = SimpleNamespace(state_delta={})

        async def run_node(self, agent, payload, **_kwargs):
            if agent is deep_research.planner_agent:
                planner_payloads.append(payload)
                return {
                    "ready_to_research": False,
                    "clarifying_questions": ["Please clarify the research request."],
                    "research_goal": "",
                    "subtasks": [],
                }
            raise AssertionError(f"Unexpected child agent: {agent.name}")

    async def run_workflow(question):
        """Run the production workflow node with a deterministic planner stub."""
        node_input = SimpleNamespace(model_dump=lambda: {"parts": [{"text": question}]})
        results = []
        async for result in deep_research.deep_research_workflow._run_impl(
            ctx=cast(Context, StubContext()), node_input=node_input
        ):
            results.append(result)
        return results

    first = asyncio.run(run_workflow("What is frontier AI?"))
    second = asyncio.run(run_workflow("How should frontier AI be governed?"))

    assert [event.output for event in first] == ["Please clarify the research request."]
    assert [event.output for event in second] == ["Please clarify the research request."]
    assert planner_payloads == [
        {"task": "Plan the deep-research workflow for the latest user request."},
        {"task": "Plan the deep-research workflow for the latest user request."},
    ]
    assert all("current_request" not in payload for payload in planner_payloads)


def test_deep_research_production_history_hides_internal_outputs(monkeypatch):
    """Later planner turns retain visible root replies without replaying agent JSON."""
    planner_model = QueuedModel(
        model="queued",
        responses=[
            types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=(
                            '{"ready_to_research": false, '
                            '"clarifying_questions": ["Which deployment target?"]}'
                        )
                    )
                ],
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=(
                            '{"ready_to_research": true, "research_goal": "Compare targets", '
                            '"subtasks": ["Compare deployment targets"]}'
                        )
                    )
                ],
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=(
                            '{"ready_to_research": false, '
                            '"clarifying_questions": ["Which region?"]}'
                        )
                    )
                ],
            ),
        ],
    )
    researcher_model = QueuedModel(
        model="gemini-3.5-flash",
        responses=[types.Content(role="model", parts=[types.Part(text="Research finding")])],
    )
    writer_model = QueuedModel(
        model="queued",
        responses=[types.Content(role="model", parts=[types.Part(text="Final research report")])],
    )
    monkeypatch.setattr(deep_research.planner_agent, "model", planner_model)
    monkeypatch.setattr(deep_research.researcher_agent, "model", researcher_model)
    monkeypatch.setattr(deep_research.writer_agent, "model", writer_model)

    async def run_scenario():
        """Run clarification, research, and a follow-up through the production Runner."""
        runner = Runner(
            app_name=deep_research.root_agent.name,
            agent=cast(BaseAgent, deep_research.root_agent),
            session_service=InMemorySessionService(),
            auto_create_session=True,
        )
        for question in ("What should I research?", "Use cloud targets.", "And Europe?"):
            async for _ in runner.run_async(
                user_id="user",
                session_id="deep-research-history",
                new_message=types.Content(role="user", parts=[types.Part(text=question)]),
                yield_user_message=True,
            ):
                pass

    asyncio.run(run_scenario())
    assert len(planner_model.requests) == 3

    first = [
        part.text
        for content in planner_model.requests[0].contents
        for part in parts(content)
        if part.text
    ]
    second = [
        part.text
        for content in planner_model.requests[1].contents
        for part in parts(content)
        if part.text
    ]
    third = [
        part.text
        for content in planner_model.requests[2].contents
        for part in parts(content)
        if part.text
    ]
    assert first.count("What should I research?") == 1
    assert second.count("What should I research?") == 1
    assert second.count("Which deployment target?") == 1
    assert second.count("Use cloud targets.") == 1
    assert third.count("What should I research?") == 1
    assert third.count("Which deployment target?") == 1
    assert third.count("Use cloud targets.") == 1
    assert third.count("Final research report") == 1
    assert third.count("And Europe?") == 1
    for texts in (first, second, third):
        assert not any("ready_to_research" in text for text in texts)
        assert not any("Research finding" in text for text in texts)
