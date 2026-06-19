"""Python ADK definition for the deep research agent."""

import asyncio

from google.adk import Workflow
from google.adk.agents.llm_agent import Agent
from google.adk.tools import google_search, url_context
from google.adk.workflow import node
from google.genai import types
from langfuse import get_client, observe, propagate_attributes
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from pydantic import BaseModel, Field

langfuse = get_client()
GoogleADKInstrumentor().instrument()

MODEL = "gemini-3.5-flash"


class ResearchPlan(BaseModel):
    """Planner output for either clarification or executable research."""

    ready_to_research: bool = Field(
        description="Whether the user's request is clear enough to start research."
    )
    clarifying_questions: list[str] = Field(
        default_factory=list,
        description="Questions to ask the user before research can start.",
    )
    research_goal: str = Field(
        default="",
        description="The clarified research goal to give to the writer.",
    )
    subtasks: list[str] = Field(
        default_factory=list,
        description="Independent research subtasks that can be run in parallel.",
    )


planner_agent = Agent(
    name="Planner_Agent",
    model=MODEL,
    instruction=(
        "You are the only agent that talks directly to the user before research starts. "
        "Decide whether the user's request is clear enough for deep research. If it is not "
        "clear, set ready_to_research to false and provide concise clarifying_questions. "
        "If it is clear, set ready_to_research to true, write a precise research_goal, and "
        "create independent subtasks that can be researched in parallel. Keep the number of "
        "subtasks appropriate to the topic complexity."
    ),
    output_schema=ResearchPlan,
)

researcher_agent = Agent(
    name="Researcher_Agent",
    model=MODEL,
    instruction=(
        "You execute one assigned research subtask by finding and gathering information "
        "from reliable sources. Use Google Search to find candidate sources, then use "
        "URL context to read and extract the relevant facts. Return concise findings "
        "with source URLs."
    ),
    generate_content_config=types.GenerateContentConfig(
        tool_config=types.ToolConfig(include_server_side_tool_invocations=True),
    ),
    tools=[google_search, url_context],
)

writer_agent = Agent(
    name="Writer_Agent",
    model=MODEL,
    instruction=(
        "You synthesize the provided research goal, plan, and findings into a final, coherent, "
        "and well-cited report. Use only the provided findings and include source URLs."
    ),
)


@node(rerun_on_resume=True)
@observe(
    name="deep-research-workflow",
    as_type="chain",
    capture_input=False,
    capture_output=True,
)
async def deep_research_workflow(ctx, node_input) -> str:
    """Run planner, dynamic researcher fan-out, and writer."""
    with propagate_attributes(
        trace_name="deep-research-workflow",
        tags=["google-adk", "deep-research"],
        metadata={"agent": "deep_research_agent", "model": MODEL},
    ):
        langfuse.update_current_span(
            input={"request": node_input},
            metadata={"workflow": "planner-researcher-writer"},
        )
        try:
            plan = await ctx.run_node(planner_agent, node_input)
            plan = ResearchPlan.model_validate(plan)
            if not plan.ready_to_research:
                return "\n".join(plan.clarifying_questions)

            research_tasks = [
                ctx.run_node(
                    researcher_agent,
                    {"research_goal": plan.research_goal, "subtask": subtask},
                    run_id=f"research-{index}",
                )
                for index, subtask in enumerate(plan.subtasks)
            ]
            findings = await asyncio.gather(*research_tasks)

            return await ctx.run_node(
                writer_agent,
                {
                    "research_goal": plan.research_goal,
                    "subtasks": plan.subtasks,
                    "findings": findings,
                },
            )
        finally:
            langfuse.flush()


root_agent = Workflow(
    name="deep_research_agent",
    description=(
        "Collects research requirements, dynamically fans out researchers, and writes "
        "the final report."
    ),
    edges=[("START", deep_research_workflow)],
)
