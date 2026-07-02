"""Python ADK definition for the recursive agent."""

import logging
from typing import Literal

from google.adk import Context, Workflow
from google.adk import tools as adk_tools
from google.adk.agents import LlmAgent, context
from google.adk.agents.llm_agent import Agent
from google.adk.workflow import node
from langfuse import get_client, observe, propagate_attributes
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from pydantic import BaseModel, Field

langfuse = get_client()
GoogleADKInstrumentor().instrument()

MODEL = "gemini-3.5-flash"

COMMON_AGENT_INSTRUCTION = (
    "You are part of a recursive agent workflow. Every turn you receive two inputs: "
    "`conversation_history` (prior turns) and `current_request` (the latest user "
    "message).\n\n"
    "You must choose exactly one outcome:\n"
    "1. ANSWER DIRECTLY: set `direct_answer` to your best response when you can answer "
    "well enough or no further prompt rewrite would materially improve the result.\n"
    "2. IMPROVE ONCE: set `agent_schema` to a complete replacement agent definition "
    "when you can name a concrete weakness in the current agent and a rewrite is likely "
    "to improve the next attempt.\n\n"
    "When improving, `agent_schema` must include `name`, `description`, and "
    "`instruction`. It may include `model`, `tools`, and `sub_agents`. Use `user_prompt` "
    "only with `agent_schema`, and only to rephrase the request without changing its "
    "meaning. Preserve this recursive self-improvement contract in rewritten top-level "
    "agent instructions until you are ready to converge.\n\n"
    "When improving, prefer defining focused sub-agents if the request has separable "
    "parts, needs multiple specialist perspectives, benefits from independent "
    "verification, or requires research across different domains. Keep sub-agents "
    "narrow and give each one a clear description and instruction. Do not create "
    "sub-agents for simple requests where one agent can answer directly.\n\n"
    "Never populate both `direct_answer` and `agent_schema`. Never leave both empty. "
    "If required information is missing, ask a concise clarifying question in "
    "`direct_answer`."
)

logging.basicConfig(level=logging.INFO)

AgentTool = Literal["google_search", "url_context"]
Model = Literal[
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    # "gemini-3.1-pro-preview"
]


class AgentSchema(BaseModel):
    name: str = Field(
        description="Unique name of the agent. Use underscores to separate words.",
    )
    model: Model = Field(default=MODEL)
    description: str = Field()
    instruction: str = Field(
        description="System instructions for the agent.",
    )
    sub_agents: list[AgentSchema] = Field(
        default_factory=list,
        description="Subagents that the agent can delegate to.",
    )
    tools: list[AgentTool] = Field(
        default_factory=list,
    )


class AgentResponseSchema(BaseModel):
    """Agent output to specify whether to answer directly or rewrite the prompt."""

    direct_answer: str | None = Field(
        default=None,
        description=(
            "If the agent decides to answer directly, put the answer here, if not leave it empty."
        ),
    )
    agent_schema: AgentSchema | None = Field(
        default=None,
        description=(
            "If the agent decides to rewrite the prompt, put the new agent schema here, "
            "if not leave it empty."
        ),
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
        COMMON_AGENT_INSTRUCTION + "\n\n"
        "You are the bootstrap recursive assistant. Start from a general-purpose "
        "assistant role, then improve the next top-level agent only when you can "
        "identify a concrete weakness in the current behavior. You may define "
        "specialist sub-agents and tools when they materially improve the answer. "
        "Prefer answering directly once further rewrites are unlikely to change the "
        "result."
    ),
    output_schema=AgentResponseSchema,
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

            recursion_limit = 5
            i = 0
            current_request = node_input.model_dump(exclude_none=True)
            fallback_answer = "I could not produce a final answer within the recursion limit."
            while i < recursion_limit:
                i += 1

                agent_response_dict = await ctx.run_node(
                    state[session_id]["current_agent"],
                    {
                        "conversation_history": history,
                        "current_request": current_request,
                    },
                )

                agent_response = AgentResponseSchema.model_validate(agent_response_dict)
                if not (
                    agent_response.direct_answer
                    or agent_response.agent_schema
                    or agent_response.user_prompt
                ):
                    logging.warning(
                        f"Agent returned no actionable output for session "
                        f"{ctx.session.id}; stopping recursion."
                    )
                    break

                if agent_response.agent_schema:
                    if not agent_response.agent_schema.instruction:
                        agent_response.agent_schema.instruction = state[session_id][
                            "current_agent"
                        ].instruction
                elif agent_response.direct_answer:
                    return agent_response.direct_answer
                else:
                    return "I could not produce an answer."

                current_request = agent_response.user_prompt or current_request

                state[session_id]["iteration"] += 1
                iteration = state[session_id]["iteration"]
                logging.info(
                    f"Updating agent {iteration} for session "
                    f"{ctx.session.id} with new system prompt."
                )
                state[session_id]["current_agent"] = await construct_agent(
                    agent_response.agent_schema, iteration
                )

                if agent_response.direct_answer:
                    return agent_response.direct_answer

            return fallback_answer

        finally:
            langfuse.flush()


async def construct_agent(agent_schema: AgentSchema, iteration) -> LlmAgent:
    agent_kwargs = agent_schema.model_dump()
    agent_kwargs["name"] = f"Agent_{iteration}"
    agent_kwargs["instruction"] = COMMON_AGENT_INSTRUCTION + "\n\n" + agent_kwargs["instruction"]
    agent_kwargs["output_schema"] = AgentResponseSchema
    agent_kwargs = construct_agent_kwargs(agent_kwargs)
    new_agent = Agent(**agent_kwargs)
    return new_agent


def construct_agent_kwargs(agent_kwargs: dict) -> dict:
    tools = agent_kwargs.pop("tools")
    tools = [getattr(adk_tools, tool) for tool in tools]

    if tools:
        agent_kwargs["tools"] = tools
        agent_kwargs["generate_content_config"] = {
            "tool_config": {"include_server_side_tool_invocations": True}
        }

    if agent_kwargs.get("sub_agents"):
        sub_agents = []
        for sub_agent_kwargs in agent_kwargs.pop("sub_agents"):
            sub_agent = Agent(**construct_agent_kwargs(sub_agent_kwargs))
            sub_agents.append(adk_tools.AgentTool(sub_agent))
        if "tools" not in agent_kwargs:
            agent_kwargs["tools"] = []
        agent_kwargs["tools"].extend(sub_agents)
    return agent_kwargs


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
