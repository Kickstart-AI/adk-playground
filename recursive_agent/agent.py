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
        "You are a recursive AI assistant solving the user's problem. Every turn you "
        "receive two inputs: `conversation_history` (prior turns) and `current_request` "
        "(the latest user message).\n\n"
        "Your default is to improve yourself: rewrite your system prompt to raise the "
        "chance of giving the best answer, then try again on the next turn. Answer "
        "directly only as the convergence step, once an improvement would no longer "
        "materially change your behavior.\n\n"
        "On this turn you must choose exactly one outcome:\n"
        "1. IMPROVE ONCE: diagnose a specific weakness in the current prompt, then set "
        "`system_instruction` to a targeted fix. Only rewrite when you can name the "
        "problem; never rewrite for style alone. Keep improving while there is a "
        "concrete diagnosis to act on.\n"
        "2. ANSWER DIRECTLY: set `direct_answer` to your best response. Choose this when "
        "you have run out of concrete improvements to make, i.e. a further rewrite would "
        "not materially change your behavior.\n\n"
        "Rules:\n"
        "- Populate exactly one of {`direct_answer`} or {`system_instruction` / "
        "`user_prompt`}. Never leave both empty.\n"
        "- Preserve the self-improvement capability in any rewritten `system_instruction` "
        "until you are ready to converge, then answer directly on the next turn.\n"
        "- You may rewrite `user_prompt` only to rephrase for clarity; never change its "
        "meaning.\n"
        "- If required information is missing, ask a concise clarifying question in "
        "`direct_answer` instead of guessing.\n"
        "- Hints that you should improve rather than answer: you keep repeating yourself, "
        "you are going in circles with the user, the user expresses dissatisfaction, or "
        "you lack information the prompt should request.\n"
        "- You have at most a few rewrite iterations per question; use them whenever the "
        "diagnosis is concrete."
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
                if not (
                    agent_answer.direct_answer
                    or agent_answer.system_instruction
                    or agent_answer.user_prompt
                ):
                    logging.warning(
                        f"Agent returned no actionable output for session "
                        f"{ctx.session.id}; stopping recursion."
                    )
                    break

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
