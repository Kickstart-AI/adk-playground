"""Tests for committee workflow privacy boundaries."""

import asyncio
import json
import random
from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast

import committee_agent.agent as agent_module
import pytest
from committee_agent.agent import (
    MEMBERS,
    MODEL_IDS,
    collect_ballots,
    run_committee_workflow,
)
from committee_agent.committee import Candidate
from google.adk.agents.context import Context
from google.genai import types


def test_ranking_payloads_contain_no_model_ids() -> None:
    """Every judge must receive only shuffled candidate IDs and answer text."""
    candidates = [
        Candidate(f"response_{index}", model_id, f"Answer {index}")
        for index, model_id in enumerate(MODEL_IDS)
    ]
    payloads = []

    class FakeContext:
        async def run_node(self, _agent, payload, run_id, **kwargs):
            payloads.append((run_id, payload, kwargs))
            return {
                "candidate_ids": [candidate["candidate_id"] for candidate in payload["candidates"]],
                "reasoning": "Test ranking.",
            }

    ballots = asyncio.run(
        collect_ballots(
            cast(Context, FakeContext()),
            {"task": "Answer the latest user question."},
            candidates,
            random.Random(3),
        )
    )

    assert len(ballots) == len(MODEL_IDS)
    serialized = json.dumps(payloads)
    assert all(model_id not in serialized for model_id in MODEL_IDS)
    assert [kwargs["override_isolation_scope"] for _, _, kwargs in payloads] == [
        f"committee:ranking:{index}" for index in range(len(MODEL_IDS))
    ]


def test_collect_answers_scopes_each_member_call() -> None:
    """Each independent answer call gets a private committee isolation scope."""
    calls = []

    class FakeContext:
        async def run_node(self, _agent, _payload, run_id, **kwargs):
            calls.append((run_id, kwargs))
            return {"answer": f"Answer for {run_id}"}

    answers = asyncio.run(
        agent_module.collect_answers(
            cast(Context, FakeContext()),
            {"task": "Answer the latest user question."},
        )
    )

    assert len(answers) == len(MODEL_IDS)
    assert [kwargs["override_isolation_scope"] for _, kwargs in calls] == [
        f"committee:answer:{index}" for index in range(len(MODEL_IDS))
    ]


def test_committee_synthesis_call_has_private_isolation_scope(monkeypatch) -> None:
    """The final synthesis payload is isolated from other committee calls."""
    calls = []

    class FakeContext:
        session = SimpleNamespace(id="session-1", events=[])

        async def run_node(self, _agent, _payload, run_id, **kwargs):
            calls.append((run_id, kwargs))
            return {"answer": "Final answer"}

    monkeypatch.setattr(
        agent_module,
        "langfuse",
        SimpleNamespace(
            update_current_span=lambda **_kwargs: None,
            flush=lambda: None,
        ),
    )
    monkeypatch.setattr(agent_module, "propagate_attributes", lambda **_kwargs: nullcontext())

    async def fake_collect_answers(_ctx, _payload):
        return {MEMBERS[0].model_id: "Answer"}

    async def fake_collect_ballots(*_args):
        return [["winner"]]

    monkeypatch.setattr(
        agent_module,
        "collect_answers",
        fake_collect_answers,
    )
    monkeypatch.setattr(agent_module, "collect_ballots", fake_collect_ballots)
    monkeypatch.setattr(
        agent_module,
        "anonymize_answers",
        lambda _answers, _rng: [Candidate("winner", MEMBERS[0].model_id, "Answer")],
    )
    monkeypatch.setattr(agent_module, "select_winner", lambda _ballots: "winner")

    node_input = types.Content(parts=[types.Part(text="Current question")])
    result = asyncio.run(run_committee_workflow(cast(Context, FakeContext()), node_input))

    assert result == "Final answer"
    assert calls == [("synthesis", {"override_isolation_scope": "committee:synthesis"})]


def test_committee_workflow_flushes_langfuse_after_failure(monkeypatch) -> None:
    """Langfuse must receive the question and flush when the workflow fails."""

    class FakeLangfuse:
        def __init__(self) -> None:
            self.inputs = []
            self.flush_count = 0

        def update_current_span(self, *, input) -> None:
            self.inputs.append(input)

        def flush(self) -> None:
            self.flush_count += 1

    payloads = []

    async def fail_collect_answers(_ctx, payload):
        payloads.append(payload)
        raise RuntimeError("model failure")

    fake_langfuse = FakeLangfuse()
    monkeypatch.setattr(agent_module, "langfuse", fake_langfuse)
    monkeypatch.setattr(agent_module, "propagate_attributes", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(agent_module, "collect_answers", fail_collect_answers)
    ctx = SimpleNamespace(session=SimpleNamespace(id="session-1", events=[]))
    node_input = types.Content(parts=[types.Part(text="Current question")])

    with pytest.raises(RuntimeError, match="model failure"):
        asyncio.run(
            run_committee_workflow(
                cast(Context, ctx),
                node_input,
            )
        )

    assert fake_langfuse.inputs == [{"question": "Current question"}]
    assert fake_langfuse.flush_count == 1
    assert payloads == [{"task": "Answer the latest user question."}]
    assert "Current question" not in json.dumps(payloads)
