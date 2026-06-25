"""Python ADK definition for the recursive agent."""

import logging

from google.adk import Context, Workflow
from google.adk.agents import context
from google.adk.agents.llm_agent import Agent
from google.adk.workflow import node
from langfuse import get_client, observe, propagate_attributes
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from pydantic import BaseModel, Field

langfuse = get_client()
GoogleADKInstrumentor().instrument()

MODEL = "gemini-3.5-flash"

logging.basicConfig(level=logging.INFO)


class RecursiveAgentSchema(BaseModel):
    """Agent output to specify whether to answer directly or rewrite the prompt."""

    direct_answer: str | None = Field(
        default=None,
        description=(
            "If the agent decides to answer directly, put the answer here, if not leave it empty."
        ),
    )
    system_instruction: str | None = Field(
        default=None,
        description="New system instructions to improve the agent.",
    )
    user_prompt: str | None = Field(
        default=None,
        description=(
            "User prompt. The user message can be passed unchanged, or modified if it helps "
            "the agent give a better answer."
        ),
    )


initial_agent = Agent(
    name="Agent_0",
    model=MODEL,
    instruction=(
        "You are a recursive AI agent that is tasked to help the user with their problem. "
        "To achieve that, you must continuously improve yourself: you can rewrite your own "
        "system prompts to increase the probability that you give the best answer. You can "
        "also choose to answer directly. When rewriting your prompt you can choose to pass "
        "on the instruction for recursive self-improvement or give a final prompt. Your "
        "choice. At every user question, make this evaluation again to answer directly or "
        "improve yourself. Hints for improving your prompt are when you keep repeating "
        "yourself, you are going in circles with the user, or the user expresses "
        "dissatisfaction."
    ),
    output_schema=RecursiveAgentSchema,
)


state = {}


@node(rerun_on_resume=True)
@observe(
    name="recursive-workflow",
    as_type="chain",
    capture_input=False,
    capture_output=True,
)
async def recursive_workflow(ctx: context.Context, node_input) -> str:
    """ """
    with propagate_attributes(
        trace_name="recursive-workflow",
        tags=["google-adk", "recursive"],
        metadata={"agent": "recursive_agent", "model": MODEL},
    ):
        langfuse.update_current_span(
            input={"request": node_input},
            metadata={"workflow": "recursive-workflow"},
        )
        try:
            session_id = ctx.session.id
            if session_id not in state:
                state[session_id] = {"current_agent": initial_agent, "iteration": 0}

            history = await extract_history(ctx)

            recursion_limit = 3
            i = 0
            current_request = node_input.model_dump()
            fallback_answer = "I could not produce a final answer within the recursion limit."
            while i < recursion_limit:
                i += 1

                agent_answer = await ctx.run_node(
                    state[session_id]["current_agent"],
                    {
                        "conversation_history": history,
                        "current_request": current_request,
                    },
                )
                agent_answer = RecursiveAgentSchema.model_validate(agent_answer)
                new_system_prompt = (
                    agent_answer.system_instruction
                    or state[session_id]["current_agent"].instruction
                )
                current_request = agent_answer.user_prompt or current_request

                if new_system_prompt != state[session_id]["current_agent"].instruction:
                    state[session_id]["iteration"] += 1
                    iteration = state[session_id]["iteration"]
                    logging.info(
                        f"Updating agent {iteration} for session "
                        f"{ctx.session.id} with new system prompt."
                    )
                    state[session_id]["current_agent"] = Agent(
                        name=f"Agent_{iteration}",
                        model=MODEL,
                        instruction=new_system_prompt,
                        output_schema=RecursiveAgentSchema,
                    )

                if agent_answer.direct_answer:
                    return agent_answer.direct_answer

            return fallback_answer

        finally:
            langfuse.flush()


async def extract_history(ctx: Context) -> list[dict[str, str]]:
    history = []
    for event in ctx.session.events:
        content = None
        if event.content and event.author == "user":
            content = "\n".join(part.text for part in event.content.parts or [] if part.text)
        elif event.output and event.author == "recursive_agent":
            content = event.output
        if content:
            history.append({"author": event.author, "text": content})
    return history


root_agent = Workflow(
    name="recursive_agent",
    description="Recursive AI agent that helps the user with their problem.",
    edges=[("START", recursive_workflow)],
)
