"""ADK workflow for document answers with source-match warnings and support scores."""

import asyncio
import json
import os
import subprocess
from urllib.request import Request, urlopen

from google.adk import Workflow
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.workflow import node
from google.genai import types

from .verification import Candidates, render_answers, render_final_answer, repair_line_breaks

answer_agent = LlmAgent(
    name="document_answerer",
    model="gemini-3.1-flash-lite",
    generate_content_config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
    ),
    include_contents="none",
    instruction=(
        "Answer the user prompt using only the supplied document. "
        "Return the answer in structured sentence-quotes pairs: "
        "each must contain exactly one complete, self-contained answer sentence "
        "and one or more exact contiguous "
        "quotes from the document that fully support that sentence. "
        "Each sentence must state exactly one distinct, independently verifiable fact. "
        "Split different facts into separate sentence-quotes pairs, even when they concern "
        "the same subject. Keep qualifiers and conditions with the fact they qualify. "
        "Each pair must include only quotes supporting its own fact. "
        "First collect the verbatim quotes from the document, then generate the answer sentence. "
        "Return all the relevant quotes that were used to generate the sentence. "
        "Do not invent quotes. "
        "Return an empty list if the document does not answer the question. "
        "All answer sentences together should form a coherent answer."
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
        document = repair_line_breaks(document)
    elif blob.mime_type and blob.mime_type.startswith("text/"):
        document = blob.data.decode("utf-8")
    else:
        raise ValueError("Attach a PDF or UTF-8 text file.")
    if not document.strip():
        raise ValueError("The attached document contains no extractable text.")

    return question, " ".join(document.split())


def judge(state: dict[str, str | list[str]], questions: dict[str, str]) -> dict[str, float]:
    """Ask Jev for probabilities for the supplied questions."""
    payload = {
        "model": "typesafe/jev-1.14.0",
        "input": {
            "state": state,
            "questions": {
                name: {"type": "noul", "instructions": instructions}
                for name, instructions in questions.items()
            },
        },
    }
    account_id = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    request = Request(
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['CLOUDFLARE_API_TOKEN']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        result = json.load(response)
    result = result.get("result", result)
    answers = result.get("result", result)["answers"]
    probabilities = {}
    for name in questions:
        probability = answers[name]["noul"]
        if (
            not isinstance(probability, (float, int))
            or isinstance(probability, bool)
            or not 0 <= probability <= 1
        ):
            raise ValueError(f"Jev returned an invalid probability for {name}.")
        probabilities[name] = float(probability)
    return probabilities


def judge_support(sentence: str, quotes: list[str]) -> float:
    """Ask Jev whether the quotes support the answer sentence."""
    probabilities = judge(
        {"sentence": sentence, "quotes": quotes},
        {
            "supported": "Do the quotes fully support every claim in the sentence?",
        },
    )
    return probabilities["supported"]


def judge_answer(question: str, answer: str) -> float:
    """Ask Jev whether the joined answer sentences answer the user's question."""
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
        await asyncio.to_thread(judge_support, candidate.sentence, candidate.quotes)
        for candidate in candidates
    ]
    answer = render_answers(candidates, probabilities, document)
    sentences = "\n".join(candidate.sentence for candidate in candidates) or "No answer found."
    relevance = await asyncio.to_thread(judge_answer, question, sentences)
    return render_final_answer(answer, relevance)


root_agent = Workflow(
    name="verified_answers_agent",
    description="Answers document questions with normalized quote checks and Jev support scores.",
    edges=[("START", node(rerun_on_resume=True)(run_verified_answers))],
)
