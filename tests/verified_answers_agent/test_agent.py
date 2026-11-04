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
    score_candidate,
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


def test_support_check_uses_cloudflare_workers_ai() -> None:
    """Support checks send the full sentence and quote to Clef via Cloudflare."""
    response = io.BytesIO(
        json.dumps(
            {
                "answers": {
                    "supported_by_quote": {"noul": 0.93},
                    "supported_by_document": {"noul": 0.87},
                }
            }
        ).encode()
    )
    with (
        patch.dict(
            "os.environ",
            {"CLOUDFLARE_ACCOUNT_ID": "test-account", "CLOUDFLARE_API_TOKEN": "test-token"},
        ),
        patch("verified_answers_agent.agent.urlopen", return_value=response) as open_url,
    ):
        assert judge_support("It rained.", "It rained.", "it rained.") == (0.93, 0.87)
    request = open_url.call_args.args[0]
    assert (
        request.full_url
        == "https://api.cloudflare.com/client/v4/accounts/test-account/ai/run/@cf/cloudflare/clef"
    )
    assert request.get_header("Authorization") == "Bearer test-token"
    payload = json.loads(request.data)
    assert payload["model"] == "clef"
    assert payload["state"] == {
        "sentence": "It rained.",
        "quote": "It rained.",
        "document": "it rained.",
    }
    assert payload["questions"]["supported_by_quote"]["type"] == "noul"


def test_quote_must_match_before_support_check() -> None:
    """An invented quote remains unscored while an exact quote gets a probability."""
    with patch("verified_answers_agent.agent.judge_support", return_value=(0.93, 0.87)) as judge:
        assert asyncio.run(score_candidate("It rained.", "It rained.", "It rained.")) == (
            0.93,
            0.87,
        )
        assert asyncio.run(score_candidate("It rained.", "It snowed.", "It snowed.")) == (
            None,
            None,
        )
        assert asyncio.run(score_candidate("It rained.", "It rained.", "it rained.")) == (
            None,
            None,
        )
    judge.assert_called_once_with("It rained.", "It rained.", "It rained.")


def test_final_answer_check_sends_question_and_assembled_answer() -> None:
    """Clef judges the full assembled answer against the original question."""
    response = io.BytesIO(json.dumps({"answers": {"answers_question": {"noul": 0.75}}}).encode())
    answer = "It rained.\n===\nQUOTE: it rained.\nSUPPORT: 0.9 (quote), 0.8 (document)"
    with (
        patch.dict("os.environ", {"CLOUDFLARE_ACCOUNT_ID": "test", "CLOUDFLARE_API_TOKEN": "test"}),
        patch("verified_answers_agent.agent.urlopen", return_value=response) as open_url,
    ):
        assert judge_answer("What happened?", answer) == 0.75
    payload = json.loads(open_url.call_args.args[0].data)
    assert payload["state"] == {"question": "What happened?", "answer": answer}
    assert payload["questions"]["answers_question"]["type"] == "noul"


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
        "candidates": [{"sentence": "It rained.", "quote": "It rained."}]
    }
    answer = "It rained.\n===\nQUOTE: It rained.\nSUPPORT: 0.9 (quote), 0.8 (document)"
    with (
        patch("verified_answers_agent.agent.judge_support", return_value=(0.9, 0.8)),
        patch("verified_answers_agent.agent.judge_answer", return_value=0.75) as judge,
    ):
        result = asyncio.run(run_verified_answers(context, content))
    judge.assert_called_once_with("What happened?", answer)
    assert result == answer + "\n\nANSWERS QUESTION: 0.75"
