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
    Answer,
    AnswererAnswer,
    AnthropicAnswererClient,
    Source,
    collect_answers,
    collect_ballots,
    create_member,
    run_committee_workflow,
    visible_conversation,
)
from committee_agent.anthropic import AnthropicAnswererModel
from committee_agent.committee import Candidate
from google.adk.agents.context import Context
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.genai import types
from litellm.types.utils import ModelResponse


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
        Candidate(f"response_{index}", model_id, f"Answer {index}", [])
        for index, model_id in enumerate(MODEL_IDS)
    ]
    payloads = []

    class FakeContext:
        async def run_node(self, _agent, payload, run_id):
            payloads.append((run_id, payload))
            return {
                "candidate_ids": [candidate["candidate_id"] for candidate in payload["candidates"]],
                "reasoning": "Test ranking.",
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


@pytest.mark.parametrize(
    ("model_id", "expected_route", "expected_args"),
    [
        (
            "openai/gpt-5.6-sol",
            "openai/responses/gpt-5.6-sol",
            {
                "reasoning_effort": "none",
                "extra_body": {"tools": [{"type": "web_search", "search_context_size": "low"}]},
            },
        ),
        (
            "anthropic/claude-fable-5",
            "anthropic/claude-fable-5",
            {"web_search_options": {"search_context_size": "low"}},
        ),
        (
            "gemini/gemini-3.1-pro-preview",
            "gemini/gemini-3.1-pro-preview",
            {"web_search_options": {"search_context_size": "low"}},
        ),
    ],
)
def test_answerer_models_use_provider_native_search(
    model_id: str,
    expected_route: str,
    expected_args: dict,
) -> None:
    """Each answerer gets its exact provider-native hosted search configuration."""
    member = create_member(0, model_id)
    answerer_model = member.answerer.model

    assert isinstance(answerer_model, LiteLlm)
    assert answerer_model.model == expected_route
    assert all(
        answerer_model._additional_args.get(key) == value for key, value in expected_args.items()
    )
    assert "include" not in answerer_model._additional_args
    assert member.answerer.tools == []


def test_search_isolated_to_answerer_models() -> None:
    """Rankers and synthesizers stay search-free and use another model object."""
    for index, model_id in enumerate(MODEL_IDS):
        member = create_member(index, model_id)
        answerer_model = member.answerer.model
        ranker_model = cast(LiteLlm, member.ranker.model)
        synthesizer_model = cast(LiteLlm, member.synthesizer.model)

        assert answerer_model is not ranker_model
        assert answerer_model is not synthesizer_model
        assert ranker_model is synthesizer_model
        assert ranker_model._additional_args == {}
        assert synthesizer_model._additional_args == {}
        assert member.ranker.tools == []
        assert member.synthesizer.tools == []


def test_answerer_and_synthesizer_instructions_require_markdown_citations() -> None:
    """Search and descriptive Markdown citation requirements remain explicit."""
    member = create_member(0, MODEL_IDS[0])
    answerer_instruction = cast(str, member.answerer.instruction)
    synthesizer_instruction = cast(str, member.synthesizer.instruction)

    assert "provider-native web search" in answerer_instruction
    assert "descriptive Markdown links" in answerer_instruction
    assert "bare URL" in answerer_instruction
    assert "preserve useful source links" in synthesizer_instruction
    assert "descriptive Markdown links" in synthesizer_instruction
    assert "bare URL" in synthesizer_instruction
    ranker_instruction = cast(str, member.ranker.instruction)
    assert "provider-native web search" in ranker_instruction
    assert "relevance" in ranker_instruction
    assert "quality" in ranker_instruction
    assert "snippet" in ranker_instruction
    assert member.answerer.output_schema is AnswererAnswer
    assert member.synthesizer.output_schema is Answer


def test_anthropic_client_forces_targeted_request_to_non_streaming() -> None:
    """The targeted hosted-search request forwards exactly one completed call."""
    captured = {}
    calls = []
    intermediate_query = {"query": "latest"}
    with pytest.raises(ValueError):
        AnswererAnswer.model_validate(intermediate_query)

    class FakeClient(AnthropicAnswererClient):
        async def _provider_acompletion(self, model, messages, tools, **kwargs):
            calls.append(1)
            captured.update(model=model, messages=messages, tools=tools, kwargs=kwargs)
            return ModelResponse(
                model=model,
                choices=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"answer":"Final","sources":[]}',
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "id": "search-1",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": '{"query":"latest"}',
                                    },
                                },
                                {
                                    "type": "function",
                                    "id": "other-1",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"key":"value"}',
                                    },
                                },
                            ],
                        },
                        "finish_reason": "stop",
                    }
                ],
            )

    result = asyncio.run(
        FakeClient().acompletion(
            model="anthropic/claude-fable-5",
            messages=[{"role": "user", "content": "Question"}],
            tools=[],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "AnswererAnswer",
                    "strict": True,
                    "schema": {"type": "object", "required": ["answer", "sources"]},
                },
            },
            web_search_options={"search_context_size": "low"},
            sentinel="preserved",
            stream=True,
        )
    )

    assert len(calls) == 1
    assert captured["kwargs"]["stream"] is False
    assert captured["kwargs"]["sentinel"] == "preserved"
    assert "response_format" not in captured["kwargs"]
    assert captured["kwargs"]["output_config"] == {
        "format": {
            "type": "json_schema",
            "schema": {"type": "object", "required": ["answer", "sources"]},
        }
    }
    assert result.choices[0].message.content == '{"answer":"Final","sources":[]}'
    assert [call.function.name for call in result.choices[0].message.tool_calls] == ["lookup"]


def test_anthropic_non_target_request_is_delegated_unchanged() -> None:
    """Requests outside the answerer activation matrix retain all arguments."""
    captured = {}

    class FakeClient(AnthropicAnswererClient):
        async def _provider_acompletion(self, model, messages, tools, **kwargs):
            captured.update(model=model, messages=messages, tools=tools, kwargs=kwargs)
            return "response"

    marker = {"type": "json_object"}
    result = asyncio.run(
        FakeClient().acompletion(
            model="anthropic/other-model",
            messages=["message"],
            tools=["tool"],
            response_format=marker,
            stream=True,
            sentinel="preserved",
        )
    )

    assert result == "response"
    assert captured == {
        "model": "anthropic/other-model",
        "messages": ["message"],
        "tools": ["tool"],
        "kwargs": {
            "response_format": marker,
            "stream": True,
            "sentinel": "preserved",
        },
    }


def test_anthropic_answerer_emits_one_completed_response_for_streaming_request(
    monkeypatch,
) -> None:
    """The adapter yields one final response instead of an intermediate query."""
    member = create_member(0, "anthropic/claude-fable-5")
    model = cast(AnthropicAnswererModel, member.answerer.model)
    client = cast(AnthropicAnswererClient, model.llm_client)
    response = ModelResponse(
        model="anthropic/claude-fable-5",
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": '{"answer":"Final","sources":[]}',
                },
                "finish_reason": "stop",
            }
        ],
    )

    async def fake_completion(*_args, **_kwargs):
        return response

    monkeypatch.setattr(client, "_provider_acompletion", fake_completion)
    request = LlmRequest(
        model="anthropic/claude-fable-5",
        contents=[types.Content(role="user", parts=[types.Part.from_text(text="Question")])],
    )
    request.set_output_schema(AnswererAnswer)

    async def collect_responses():
        return [response async for response in model.generate_content_async(request, stream=True)]

    results = asyncio.run(collect_responses())

    assert len(results) == 1
    assert results[0].partial is False
    assert results[0].turn_complete is True
    assert AnswererAnswer.model_validate_json(results[0].content.parts[0].text).answer == "Final"


def test_anthropic_malformed_final_response_is_not_repaired(monkeypatch) -> None:
    """A malformed completed answer remains a schema validation failure."""
    member = create_member(0, "anthropic/claude-fable-5")
    model = cast(AnthropicAnswererModel, member.answerer.model)
    client = cast(AnthropicAnswererClient, model.llm_client)
    response = ModelResponse(
        model="anthropic/claude-fable-5",
        choices=[
            {
                "message": {"role": "assistant", "content": '{"query":"still malformed"}'},
                "finish_reason": "stop",
            }
        ],
    )

    async def fake_completion(*_args, **_kwargs):
        return response

    monkeypatch.setattr(client, "_provider_acompletion", fake_completion)
    request = LlmRequest(
        model="anthropic/claude-fable-5",
        contents=[types.Content(role="user", parts=[types.Part.from_text(text="Question")])],
    )
    request.set_output_schema(AnswererAnswer)

    async def collect_responses():
        return [item async for item in model.generate_content_async(request, stream=True)]

    result = asyncio.run(collect_responses())
    content = result[0].content.parts[0].text
    with pytest.raises(ValueError):
        AnswererAnswer.model_validate_json(content)


def test_collect_answers_preserves_cited_answer_text() -> None:
    """Collection keeps the complete structured answer, including source links."""
    cited_answers = [
        AnswererAnswer(
            answer="[Official source](https://example.com/one)",
            sources=[
                Source(url="https://example.com/one", title="One", snippet="First"),
                Source(url="https://example.com/one", title="One duplicate", snippet="Second"),
            ],
        ),
        AnswererAnswer(
            answer="[Reference source](https://example.com/two)",
            sources=[Source(url="https://example.com/two", title="Two", snippet="Second")],
        ),
        AnswererAnswer(
            answer="[Primary source](https://example.com/three)",
            sources=[],
        ),
    ]

    class FakeContext:
        async def run_node(self, member, _payload, run_id):
            index = int(run_id.rsplit("_", 1)[1])
            assert member is agent_module.MEMBERS[index].answerer
            return cited_answers[index].model_dump()

    answers = asyncio.run(
        collect_answers(
            cast(Context, FakeContext()),
            {"current_question": "Question", "conversation_history": []},
        )
    )

    assert list(answers.values()) == cited_answers


def test_collect_ballots_serializes_ordered_sources_without_provider_metadata() -> None:
    """Ranker payloads contain only exact provider-neutral source dictionaries."""
    sources = [
        Source(url="HTTPS://Example.com/one?b=2&a=1", title="First", snippet="One"),
        Source(url="https://example.com/two#fragment", title="Second", snippet="Two"),
        Source(url="HTTPS://Example.com/one?b=2&a=1", title="First", snippet="One"),
    ]
    candidates = [
        Candidate(
            candidate_id="response_a",
            model="openai/gpt-5.6-sol",
            answer="[One](https://example.com/one)",
            sources=sources,
        )
    ]
    payloads = []

    class FakeContext:
        async def run_node(self, _agent, payload, run_id):
            payloads.append((run_id, payload))
            return {"candidate_ids": ["response_a"], "reasoning": "Supported."}

    ballots = asyncio.run(
        collect_ballots(
            cast(Context, FakeContext()),
            {"current_question": "Question", "conversation_history": []},
            candidates,
            random.Random(3),
        )
    )

    assert ballots == [["response_a"]] * len(agent_module.MEMBERS)
    for _run_id, payload in payloads:
        assert payload["candidates"] == [
            {
                "candidate_id": "response_a",
                "answer": "[One](https://example.com/one)",
                "sources": [source.model_dump() for source in sources],
            }
        ]
        serialized = json.dumps(payload)
        assert "openai/gpt-5.6-sol" not in serialized
        assert "member_0_answerer" not in serialized


def test_collect_answers_propagates_provider_failure_without_fallback() -> None:
    """A failed search-enabled answerer is called once and its error propagates."""
    calls = []

    class FakeContext:
        async def run_node(self, member, _payload, run_id):
            calls.append((member, run_id))
            raise RuntimeError("provider unavailable")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        asyncio.run(
            collect_answers(
                cast(Context, FakeContext()),
                {"current_question": "Question", "conversation_history": []},
            )
        )

    assert len(calls) == len(agent_module.MEMBERS)
    assert {run_id for _member, run_id in calls} == {
        f"answer_{index}" for index in range(len(agent_module.MEMBERS))
    }
    assert {id(member) for member, _run_id in calls} == {
        id(committee_member.answerer) for committee_member in agent_module.MEMBERS
    }


def test_synthesis_receives_all_anonymous_citations_and_returns_them(monkeypatch) -> None:
    """Synthesis receives cited text without model identity and preserves final output."""
    cited_answers = [
        "[Official source](https://example.com/one)",
        "[Reference source](https://example.com/two)",
        "[Primary source](https://example.com/three)",
    ]
    final_answer = "[Best source](https://example.com/final)"
    captured_payload = {}

    async def fake_collect_answers(_ctx, _payload):
        return dict(
            zip(
                (member.model_id for member in agent_module.MEMBERS),
                [
                    AnswererAnswer(
                        answer=answer,
                        sources=[
                            Source(
                                url=f"https://example.com/{index}",
                                title=f"Source {index}",
                                snippet=f"Snippet {index}",
                            )
                        ],
                    )
                    for index, answer in enumerate(cited_answers)
                ],
                strict=True,
            )
        )

    async def fake_collect_ballots(_ctx, _payload, candidates, _rng):
        return [[candidate.candidate_id for candidate in candidates] for _ in agent_module.MEMBERS]

    class FakeContext:
        session = SimpleNamespace(id="session-1", events=[])

        async def run_node(self, member, payload, run_id):
            assert run_id == "synthesis"
            assert member in [
                committee_member.synthesizer for committee_member in agent_module.MEMBERS
            ]
            captured_payload.update(payload)
            return {"answer": final_answer}

    monkeypatch.setattr(agent_module, "collect_answers", fake_collect_answers)
    monkeypatch.setattr(agent_module, "collect_ballots", fake_collect_ballots)
    monkeypatch.setattr(agent_module, "propagate_attributes", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(agent_module.langfuse, "update_current_span", lambda **_kwargs: None)
    monkeypatch.setattr(agent_module.langfuse, "flush", lambda: None)

    result = asyncio.run(
        run_committee_workflow(
            cast(Context, FakeContext()),
            types.Content(parts=[types.Part(text="Question")]),
        )
    )

    assert result == final_answer
    candidate_answers = captured_payload["candidate_answers"]
    assert {candidate["answer"] for candidate in candidate_answers} == set(cited_answers)
    assert all(
        set(source) == {"url", "title", "snippet"}
        for candidate in candidate_answers
        for source in candidate["sources"]
    )
    assert {
        source["url"] for candidate in candidate_answers for source in candidate["sources"]
    } == {f"https://example.com/{index}" for index in range(len(cited_answers))}
    serialized = json.dumps(candidate_answers)
    assert all(model_id not in serialized for model_id in MODEL_IDS)
    assert all(member.answerer.name not in serialized for member in agent_module.MEMBERS)
