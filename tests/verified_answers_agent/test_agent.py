"""Tests for uploaded documents and Cloudflare Jev routing."""

import asyncio
import io
import json
from unittest.mock import AsyncMock, patch

from google.genai import types
from verified_answers_agent.agent import (
    extract_document,
    judge_answer,
    judge_support,
    run_verified_answers,
)


def test_uploaded_text_and_question_are_extracted() -> None:
    """A text attachment and question are used as separate inputs."""
    content = types.Content(
        parts=[
            types.Part(text="What happened?"),
            types.Part.from_bytes(data=b"  It\t rained.\n Next  line.  ", mime_type="text/plain"),
        ]
    )
    assert extract_document(content) == ("What happened?", "It rained. Next line.")


def test_support_check_uses_cloudflare_gateway() -> None:
    """A sentence and all its quotes receive a Jev support score through Cloudflare."""
    response = io.BytesIO(
        json.dumps(
            {"result": {"state": "Completed", "result": {"answers": {"supported": {"noul": 0.93}}}}}
        ).encode()
    )
    with (
        patch.dict(
            "os.environ",
            {"CLOUDFLARE_ACCOUNT_ID": "test-account", "CLOUDFLARE_API_TOKEN": "test-token"},
        ),
        patch("verified_answers_agent.agent.urlopen", return_value=response) as open_url,
    ):
        assert judge_support("It rained.", ["It rained.", "The ground was wet."]) == 0.93
    request = open_url.call_args.args[0]
    assert request.full_url == "https://api.cloudflare.com/client/v4/accounts/test-account/ai/run"
    assert request.get_header("Authorization") == "Bearer test-token"
    payload = json.loads(request.data)
    assert payload["model"] == "typesafe/jev-1.14.0"
    assert payload["input"]["state"] == {
        "sentence": "It rained.",
        "quotes": ["It rained.", "The ground was wet."],
    }
    assert payload["input"]["questions"]["supported"]["type"] == "noul"


def test_final_answer_check_sends_question_and_assembled_answer() -> None:
    """Jev judges the full assembled answer against the original question."""
    response = io.BytesIO(json.dumps({"answers": {"answers_question": {"noul": 0.75}}}).encode())
    answer = "It rained.\n===\nQUOTE: It rained.\nSUPPORT: 0.9"
    with (
        patch.dict("os.environ", {"CLOUDFLARE_ACCOUNT_ID": "test", "CLOUDFLARE_API_TOKEN": "test"}),
        patch("verified_answers_agent.agent.urlopen", return_value=response) as open_url,
    ):
        assert judge_answer("What happened?", answer) == 0.75
    payload = json.loads(open_url.call_args.args[0].data)
    assert payload["input"]["state"] == {"question": "What happened?", "answer": answer}
    assert payload["input"]["questions"]["answers_question"]["type"] == "noul"


def test_workflow_judges_the_assembled_answer() -> None:
    """The final relevance score applies to the answer shown to the user."""
    content = types.Content(
        parts=[
            types.Part(text="What happened?"),
            types.Part.from_bytes(data=b"It rained.", mime_type="text/plain"),
        ]
    )
    context = AsyncMock()
    context.run_node.return_value = {
        "candidates": [{"sentence": "It rained.", "quotes": ["It rained."]}]
    }
    answer = "It rained.\n===\nQUOTE: It rained.\nSUPPORT: 0.9"
    with (
        patch("verified_answers_agent.agent.judge_support", return_value=0.9),
        patch("verified_answers_agent.agent.judge_answer", return_value=0.75) as judge,
    ):
        result = asyncio.run(run_verified_answers(context, content))
    judge.assert_called_once_with("What happened?", "It rained.")
    assert result == answer + "\n\n===\nANSWERS QUESTION: 0.75"
