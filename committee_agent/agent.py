"""ADK workflow that answers through an anonymous three-model committee."""

import asyncio
import random
from dataclasses import dataclass

from google.adk import Workflow
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.models.lite_llm import LiteLlm
from google.adk.workflow import node
from langfuse import get_client, observe, propagate_attributes
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from pydantic import BaseModel, Field

from .committee import Candidate, anonymize_answers, ranking_view, select_winner

langfuse = get_client()
GoogleADKInstrumentor().instrument()

MODEL_IDS = (
    "openai/gpt-5.6-sol",
    "gemini/gemini-3.1-pro-preview",
    "anthropic/claude-fable-5",
)


class Answer(BaseModel):
    """A committee member's response to the user."""

    answer: str = Field(description="The complete proposed answer.")


class Ranking(BaseModel):
    """An ordered ballot over anonymous answers."""

    candidate_ids: list[str] = Field(
        description="Every candidate ID exactly once, ordered from best to worst."
    )


@dataclass(frozen=True)
class Member:
    """One model's private workflow resources."""

    model_id: str
    answerer: LlmAgent
    ranker: LlmAgent
    synthesizer: LlmAgent


def create_member(index: int, model_id: str) -> Member:
    """Create generic agents for one privately identified model."""
    model = LiteLlm(model=model_id)
    return Member(
        model_id=model_id,
        answerer=LlmAgent(
            name=f"member_{index}_answerer",
            model=model,
            include_contents="none",
            instruction=(
                "Answer the user's current question using the supplied user-visible "
                "conversation history. Produce a complete, accurate, and useful answer. "
                "Return only the structured answer field."
            ),
            output_schema=Answer,
        ),
        ranker=LlmAgent(
            name=f"member_{index}_ranker",
            model=model,
            include_contents="none",
            instruction=(
                "Rank the anonymous candidate answers by correctness, relevance, reasoning, "
                "and usefulness. The candidates are deliberately shuffled. Do not infer or "
                "mention who wrote them. Return every candidate ID exactly once from best "
                "to worst."
            ),
            output_schema=Ranking,
        ),
        synthesizer=LlmAgent(
            name=f"member_{index}_synthesizer",
            model=model,
            include_contents="none",
            instruction=(
                "Write the final answer to the user using all supplied anonymous candidate "
                "answers and their collective rankings. Reconcile disagreements and retain "
                "the strongest viewpoints. Do not mention candidates, rankings, models, or "
                "the committee. Return only the structured answer field."
            ),
            output_schema=Answer,
        ),
    )


MEMBERS = tuple(create_member(index, model_id) for index, model_id in enumerate(MODEL_IDS))
MEMBER_BY_MODEL = {member.model_id: member for member in MEMBERS}


def visible_conversation(ctx: Context) -> list[dict[str, str]]:
    """Extract only user-visible text and hide internal agent identities."""
    conversation = []
    for event in ctx.session.events:
        if event.isolation_scope or not event.content:
            continue
        text = "\n".join(
            part.text.strip()
            for part in event.content.parts or []
            if part.text and part.text.strip()
        )
        if text:
            role = "user" if event.author == "user" else "assistant"
            conversation.append({"role": role, "content": text})
    return conversation


def content_text(node_input) -> str:
    """Extract text from the current ADK workflow input."""
    return "\n".join(
        part.text.strip() for part in node_input.parts or [] if part.text and part.text.strip()
    )


async def collect_answers(ctx: Context, payload: dict) -> dict[str, str]:
    """Ask all committee members for independent answers in parallel."""
    responses = await asyncio.gather(
        *(
            ctx.run_node(member.answerer, payload, run_id=f"answer_{index}")
            for index, member in enumerate(MEMBERS)
        )
    )
    return {
        member.model_id: Answer.model_validate(response).answer
        for member, response in zip(MEMBERS, responses, strict=True)
    }


async def collect_ballots(
    ctx: Context,
    payload: dict,
    candidates: list[Candidate],
    rng: random.Random,
) -> list[list[str]]:
    """Ask every member to rank an independently shuffled anonymous view."""
    ranking_payloads = [
        {
            **payload,
            "candidates": [candidate.__dict__ for candidate in ranking_view(candidates, rng)],
        }
        for _ in MEMBERS
    ]
    responses = await asyncio.gather(
        *(
            ctx.run_node(
                member.ranker,
                ranking_payload,
                run_id=f"ranking_{index}",
            )
            for index, (member, ranking_payload) in enumerate(
                zip(MEMBERS, ranking_payloads, strict=True)
            )
        )
    )
    return [Ranking.model_validate(response).candidate_ids for response in responses]


@observe(
    name="committee-workflow",
    as_type="chain",
    capture_input=False,
    capture_output=True,
)
async def run_committee_workflow(ctx: Context, node_input) -> str:
    """Run independent answers, anonymous voting, and winner-led synthesis."""
    question = content_text(node_input)
    with propagate_attributes(
        session_id=ctx.session.id,
        tags=["google-adk", "committee"],
        metadata={"agent": "committee_agent", "workflow": "answer-rank-synthesize"},
    ):
        langfuse.update_current_span(input={"question": question})
        try:
            rng = random.SystemRandom()
            payload = {
                "current_question": question,
                "conversation_history": visible_conversation(ctx),
            }
            answers = await collect_answers(ctx, payload)
            candidates = anonymize_answers(answers, rng)
            ballots = await collect_ballots(ctx, payload, candidates, rng)
            winner_id = select_winner(ballots)
            winner = next(
                candidate for candidate in candidates if candidate.candidate_id == winner_id
            )
            synthesis = await ctx.run_node(
                MEMBER_BY_MODEL[winner.model].synthesizer,
                {
                    **payload,
                    "candidate_answers": [
                        candidate.__dict__ for candidate in ranking_view(candidates, rng)
                    ],
                    "collective_rankings": ballots,
                },
                run_id="synthesis",
            )
            return Answer.model_validate(synthesis).answer
        finally:
            langfuse.flush()


committee_workflow = node(rerun_on_resume=True)(run_committee_workflow)

root_agent = Workflow(
    name="committee_agent",
    description="Combines anonymous answers selected by a three-model committee.",
    edges=[("START", committee_workflow)],
)
