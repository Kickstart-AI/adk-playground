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
    MODEL_IDS,
    collect_ballots,
    run_committee_workflow,
    visible_conversation,
)
from committee_agent.committee import Candidate
from google.adk.agents.context import Context
from google.genai import types


def text_event(author: str, text: str, isolation_scope: str | None = None) -> SimpleNamespace:
    """Create the event fields used by visible conversation extraction."""
    content = SimpleNamespace(parts=[SimpleNamespace(text=text)])
    return SimpleNamespace(author=author, content=content, isolation_scope=isolation_scope)


def test_visible_conversation_excludes_internal_events_and_agent_names() -> None:
    """Only visible dialogue may be forwarded, with assistant identity removed."""
    ctx = SimpleNamespace(
        session=SimpleNamespace(
            events=[
                text_event("user", "First question"),
                text_event("member_0_answerer", "private answer", "committee/internal"),
                text_event("committee_agent", "Visible response"),
                text_event("user", "Follow-up question"),
            ]
        )
    )

    assert visible_conversation(cast(Context, ctx)) == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "Visible response"},
        {"role": "user", "content": "Follow-up question"},
    ]


def test_ranking_payloads_contain_no_model_ids() -> None:
    """Every judge must receive only shuffled candidate IDs and answer text."""
    candidates = [
        Candidate(f"response_{index}", model_id, f"Answer {index}")
        for index, model_id in enumerate(MODEL_IDS)
    ]
    payloads = []

    class FakeContext:
        async def run_node(self, _agent, payload, run_id):
            payloads.append((run_id, payload))
            return {
                "candidate_ids": [candidate["candidate_id"] for candidate in payload["candidates"]]
            }

    ballots = asyncio.run(
        collect_ballots(
            cast(Context, FakeContext()),
            {"current_question": "Question", "conversation_history": []},
            candidates,
            random.Random(3),
        )
    )

    assert len(ballots) == len(MODEL_IDS)
    serialized = json.dumps(payloads)
    assert all(model_id not in serialized for model_id in MODEL_IDS)


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

    async def fail_collect_answers(_ctx, _payload):
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
