"""ADK workflow for answering questions with verified document citations."""

import asyncio
import json
import os
import subprocess
from urllib.request import Request, urlopen

from google.adk import Workflow
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.workflow import node

from .verification import Candidates, render_answers, render_final_answer

answer_agent = LlmAgent(
    name="document_answerer",
    model="gemini-3.5-flash-lite",
    include_contents="none",
    instruction=(
        "Answer the user prompt using only the supplied document. "
        "Return the answer in structured sentence quote pairs: "
        "each must contain exactly one complete answer sentence and one exact contiguous "
        "quote from the document that fully supports that sentence. Do not invent quotes. "
        "Return an empty list if the document does not answer the question."
    ),
    output_schema=Candidates,
)


def extract_document(node_input) -> tuple[str, str]:
    """Read the question and an attached PDF or UTF-8 text file from the input."""
    question = "\n".join(part.text for part in node_input.parts or [] if part.text)
    documents = [part.inline_data for part in node_input.parts or [] if part.inline_data]
    if not question.strip() or len(documents) != 1:
        raise ValueError("Attach one PDF or text file and include a question in the same message.")
    blob = documents[0]
    if blob.mime_type == "application/pdf":
        document = subprocess.run(
            ["pdftotext", "-layout", "-", "-"], input=blob.data, capture_output=True, check=True
        ).stdout.decode("utf-8")
    elif blob.mime_type and blob.mime_type.startswith("text/"):
        document = blob.data.decode("utf-8")
    else:
        raise ValueError("Attach a PDF or UTF-8 text file.")
    if not document.strip():
        raise ValueError("The attached document contains no extractable text.")

    return question, " ".join(document.split())


def judge(state: dict[str, str], questions: dict[str, str]) -> dict[str, float]:
    """Ask Clef for probabilities for the supplied questions."""
    payload = {
        "model": "clef",
        "state": state,
        "questions": {
            name: {"type": "noul", "instructions": instructions}
            for name, instructions in questions.items()
        },
    }
    account_id = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    request = Request(
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/cloudflare/clef",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['CLOUDFLARE_API_TOKEN']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        result = json.load(response)
    answers = result.get("result", result)["answers"]
    probabilities = {}
    for name in questions:
        probability = answers[name]["noul"]
        if (
            not isinstance(probability, (float, int))
            or isinstance(probability, bool)
            or not 0 <= probability <= 1
        ):
            raise ValueError(f"Clef returned an invalid probability for {name}.")
        probabilities[name] = float(probability)
    return probabilities


def judge_support(sentence: str, quote: str, document: str) -> tuple[float, float]:
    """Ask Clef whether the quote and document support the answer sentence."""
    probabilities = judge(
        {"sentence": sentence, "quote": quote, "document": document},
        {
            "supported_by_quote": "Does the quote fully support every claim in the sentence?",
            "supported_by_document": "Does the document fully support every claim in the sentence?",
        },
    )
    return probabilities["supported_by_quote"], probabilities["supported_by_document"]


def judge_answer(question: str, answer: str) -> float:
    """Ask Clef whether the assembled answer answers the user's question."""
    return judge(
        {"question": question, "answer": answer},
        {"answers_question": "Does the answer answer the user's question?"},
    )["answers_question"]


async def run_verified_answers(ctx: Context, node_input) -> str:
    """Generate candidates and annotate each with source support."""
    question, document = extract_document(node_input)
    result = await ctx.run_node(
        answer_agent,
        {"question": question, "document": document},
        override_isolation_scope="answer_candidates",
    )
    candidates = Candidates.model_validate(result).candidates
    probabilities = [
        await score_candidate(document, candidate.sentence, candidate.quote)
        for candidate in candidates
    ]
    answer = render_answers(candidates, probabilities)
    relevance = await asyncio.to_thread(judge_answer, question, answer)
    return render_final_answer(answer, relevance)


async def score_candidate(
    document: str, sentence: str, quote: str
) -> tuple[float | None, float | None]:
    """Score a candidate only when its quote occurs exactly in the document."""
    if not quote.strip() or quote.strip() not in document:
        return None, None
    return await asyncio.to_thread(judge_support, sentence, quote, document)


root_agent = Workflow(
    name="verified_answers_agent",
    description="Answers questions using exact quotes verified against a document and Clef.",
    edges=[("START", node(rerun_on_resume=True)(run_verified_answers))],
)
