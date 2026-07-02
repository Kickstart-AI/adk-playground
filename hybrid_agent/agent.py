"""Hybrid ADK agent: flow graph built from agent_flows.yaml, conversation driven by LLMs."""

import logging
import pathlib

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.workflow import FunctionNode, Workflow
from google.genai import types
from langfuse import get_client
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from pydantic import BaseModel

from .flow_schema import Action, Step, load_config

logger = logging.getLogger(__name__)

langfuse = get_client()
GoogleADKInstrumentor().instrument()

MODEL = "gemini-3.5-flash"
FLOWS_PATH = pathlib.Path(__file__).parent / "agent_flows.yaml"
CONFIG = load_config(FLOWS_PATH)

FAKE_ORDERS = {
    "1001": {
        "order_number": "1001",
        "items": [
            {"name": "running shoes", "returnable": True},
            {"name": "socks", "returnable": True},
            {"name": "protein bars", "returnable": False, "reason": "food is exempt from return"},
        ],
        "delivered_days_ago": 5,
    },
    "1002": {
        "order_number": "1002",
        "items": [{"name": "winter jacket", "returnable": True}],
        "delivered_days_ago": 45,
    },
}


def fetch_order(order_number: str) -> dict:
    """Fetch order details by order number."""
    order = FAKE_ORDERS.get(order_number)
    if order is None:
        raise ValueError(f"Order {order_number} not found.")
    return order


def validate_order_eligibility(order_number: str) -> dict:
    """Check that the order is within the 30-day return window."""
    order = fetch_order(order_number)
    if order["delivered_days_ago"] > 30:
        raise ValueError(f"Order {order_number} is outside the 30-day return window.")
    return {"eligible": True}


def register_return(order_number: str, item: str) -> dict:
    """Register the return of an item and issue a return label."""
    match = next((i for i in fetch_order(order_number)["items"] if i["name"] == item), None)
    if match is None:
        raise ValueError(f"Item '{item}' is not part of order {order_number}.")
    if not match["returnable"]:
        raise ValueError(f"Item '{item}' is not returnable: {match['reason']}.")
    return {
        "return_id": f"R-{order_number}",
        "label_url": f"https://returns.example.com/R-{order_number}.pdf",
    }


class FetchOrderArgs(BaseModel):
    """Arguments for fetch_order."""

    order_number: str


class ValidateArgs(BaseModel):
    """Arguments for validate_order_eligibility."""

    order_number: str


class RegisterReturnArgs(BaseModel):
    """Arguments for register_return."""

    order_number: str
    item: str


TOOLS = {
    "fetch_order": (fetch_order, FetchOrderArgs),
    "validate_order_eligibility": (validate_order_eligibility, ValidateArgs),
    "register_return": (register_return, RegisterReturnArgs),
}


class IntakeDecision(BaseModel):
    """Which flow matches the user's request."""

    flow: str = ""  # empty if no clear match yet


class Resolution(BaseModel):
    """Whether the conversation already answers a question we were about to ask."""

    answered: bool
    answer: str = ""


class Verdict(BaseModel):
    """Result of a reflection check."""

    passed: bool
    reason: str


speaker = LlmAgent(
    name="speaker",
    model=MODEL,
    instruction=(
        f"{CONFIG.persona.strip()} "
        "Given an instruction, the conversation transcript, and the known facts, "
        "write the exact message to send to the user, and nothing else."
    ),
)

resolver = LlmAgent(
    name="resolver",
    model=MODEL,
    instruction=(
        "You are given a question the agent is about to ask the user, plus the conversation "
        "transcript and known facts. Decide whether the answer is already clearly provided. "
        "Only set answered=true if the user unambiguously gave the answer; never guess."
    ),
    output_schema=Resolution,
)

reflector = LlmAgent(
    name="reflector",
    model=MODEL,
    instruction=(
        "Judge whether the given validation instruction holds based on the known facts. "
        "Explain your verdict briefly in the reason."
    ),
    output_schema=Verdict,
)

extractors = {
    tool_name: LlmAgent(
        name=f"{tool_name}_args",
        model=MODEL,
        instruction=f"Extract the arguments for tool '{tool_name}' from the conversation facts.",
        output_schema=args_model,
    )
    for tool_name, (_, args_model) in TOOLS.items()
}


intake_speaker = LlmAgent(
    name="intake_speaker",
    model=MODEL,
    instruction=f"{CONFIG.persona.strip()} {CONFIG.instruction.strip()}",
)

intake_router = LlmAgent(
    name="intake_router",
    model=MODEL,
    instruction=(
        "Decide which of the available flows matches the user's request based on the "
        "conversation transcript. Return the flow name, or an empty string if no flow "
        "clearly matches yet."
    ),
    output_schema=IntakeDecision,
)

INTERNAL_AGENTS = {
    "speaker",
    "reflector",
    "resolver",
    "intake_router",
    "intake_speaker",
    *extractors,
}

FLOWS = CONFIG.flows
STEPS = [step for flow in FLOWS for step in flow.steps]

# State delta that sends the conversation back to intake for a fresh request.
INTAKE_RESET = {
    "current_step": "intake",
    "facts": {},
    **{
        f"asked:{step.name}:{index}": False
        for step in STEPS
        for index, action in enumerate(step.actions)
        if action.message is not None
    },
    **{f"done:{step.name}:{index}": False for step in STEPS for index in range(len(step.actions))},
}


def conversation_transcript(ctx: Context) -> list[dict]:
    """Reconstruct the user-visible dialogue from the session events."""
    lines = []
    for event in ctx.session.events:
        if event.author not in ["user", "hybrid_agent"] or not event.content:
            continue
        for part in event.content.parts or []:
            if part.text and part.text.strip():
                lines.append({"author": event.author, "text": part.text.strip()})
            elif part.function_call and part.function_call.name == "adk_request_input":
                lines.append(
                    {"author": event.author, "text": (part.function_call.args or {}).get("message")}
                )
            elif part.function_response and part.function_response.name == "adk_request_input":
                reply = (part.function_response.response or {}).get("result")
                lines.append({"author": event.author, "text": str(reply)})
    return lines


def message_event(text: str, state: dict | None = None, output=None) -> Event:
    """Build an event that shows a message to the user."""
    return Event(
        output=output,
        content=types.Content(role="model", parts=[types.Part.from_text(text=text)]),
        state=state or {},
    )


def text_of(output: str) -> str:
    """Extract the plain text from an LLM agent's output."""
    return output.strip()


def route_event(target: str, output, facts: dict) -> Event:
    """Build an event that routes to another node.

    Routing to intake also points current_step there, so follow-up turns
    land in intake instead of re-entering the step that routed away.
    """
    state = {"facts": facts}
    if target == "intake":
        state["current_step"] = "intake"
    return Event(output=output, route=target, state=state)


async def run_message_action(step_name: str, run_id: str, action: Action, ctx: Context, memo: dict):
    """Ask the user a message action's question, unless already asked or answerable.

    Asks as a normal message ending the turn; the answer arrives in the next
    turn's transcript and the dispatcher routes back to this step. Non-required
    questions are skipped when the resolver finds the answer in the conversation.
    """

    def on_answered():
        """Route onward if a pass target is set, otherwise continue with the next action."""
        if action.result and action.result.passed:
            return route_event(action.result.passed, memo["facts"], memo["facts"])
        return None

    if ctx.state.get(f"asked:{run_id}"):
        if action.result is None or action.result.fail is None:
            return on_answered()
        # Judge whether the user's reply actually answers the question positively.
        verdict = Verdict(
            **await ctx.run_node(
                reflector,
                {
                    "instruction": f"The agent asked the user: '{action.message}'. "
                    "Judge whether the user's latest reply answers it affirmatively.",
                    **memo,
                },
                run_id=f"{run_id}:verify",
            )
        )
        if verdict.passed:
            return on_answered()
        return route_event(action.result.fail, verdict.reason, memo["facts"])
    if not action.required:
        resolution = Resolution(
            **await ctx.run_node(
                resolver, {"question": action.message, **memo}, run_id=f"{run_id}:resolve"
            )
        )
        if resolution.answered:
            memo["facts"][action.message] = resolution.answer
            return on_answered()
    utterance = await ctx.run_node(
        speaker, {"instruction": action.message, **memo}, run_id=f"{run_id}:speak"
    )
    return message_event(
        text_of(utterance),
        state={f"asked:{run_id}": True, "current_step": step_name},
    )


async def run_action(step: Step, index: int, action: Action, ctx: Context, memo: dict):
    """Run one action; return an Event that ends the step, or None to continue.

    memo holds the shared "transcript" and "facts" passed to the LLM calls.
    """
    run_id = f"{step.name}:{index}"
    if action.message is not None:
        return await run_message_action(step.name, run_id, action, ctx, memo)
    facts = memo["facts"]
    # Guaranteed by flow_schema route validation.
    assert action.result is not None and action.result.fail is not None
    if action.reflect is not None:
        verdict = Verdict(
            **await ctx.run_node(reflector, {"instruction": action.reflect, **memo}, run_id=run_id)
        )
        if not verdict.passed:
            return route_event(action.result.fail, verdict.reason, facts)
    else:
        assert action.tool_call is not None  # only action kind left after message and reflect
        tool, args_model = TOOLS[action.tool_call]
        args = args_model(**await ctx.run_node(extractors[action.tool_call], memo, run_id=run_id))
        try:
            facts[f"{action.tool_call}_result"] = tool(**args.model_dump())
        except ValueError as error:
            return route_event(action.result.fail, str(error), facts)
    # Success: route onward if a pass target is set, otherwise continue with the next action.
    if action.result.passed:
        return route_event(action.result.passed, facts, facts)
    return None


def is_terminal(step: Step) -> bool:
    """A terminal step has no onward "pass" route."""
    return not any(action.result and action.result.passed for action in step.actions)


async def execute_step(step: Step, ctx: Context) -> Event:
    """Run one YAML flow step and return its final event."""
    terminal = is_terminal(step)
    facts = dict(ctx.state.get("facts", {}))
    memo = {"transcript": conversation_transcript(ctx), "facts": facts}

    # Terminal steps combine their messages into one closing message instead.
    # Completed actions are skipped, so a rerun resumes at the next pending action.
    done_delta = {}
    for index, action in [] if terminal else list(enumerate(step.actions)):
        run_id = f"{step.name}:{index}"
        if ctx.state.get(f"done:{run_id}"):
            continue
        event = await run_action(step, index, action, ctx, memo)
        if event is None:
            done_delta[f"done:{run_id}"] = True
            continue
        event.actions.state_delta = {**done_delta, **(event.actions.state_delta or {})}
        return event

    if not terminal:
        raise ValueError(f"Step {step.name} finished without routing anywhere.")
    instruction = " Then: ".join(a.message for a in step.actions if a.message is not None)
    utterance = await ctx.run_node(
        speaker, {"instruction": instruction, **memo}, run_id=f"{step.name}:close"
    )
    text = text_of(utterance)
    return message_event(text, output=text, state=INTAKE_RESET)


def make_step_node(step: Step) -> FunctionNode:
    """Build a workflow node that executes one YAML flow step."""

    async def run_step(ctx: Context, node_input):
        try:
            return await execute_step(step, ctx)
        except Exception:
            # Fallback: hand the conversation back to intake instead of dying.
            logger.error("Step %s failed, falling back to intake.", step.name, exc_info=True)
            utterance = await ctx.run_node(
                intake_speaker,
                {"transcript": conversation_transcript(ctx)},
                run_id=f"{step.name}:fallback",
            )
            return message_event(text_of(utterance), state=INTAKE_RESET)

    return FunctionNode(func=run_step, name=step.name, rerun_on_resume=True)


async def intake(ctx: Context, node_input):
    """Match the user's request to a flow, or ask what they need."""
    transcript = conversation_transcript(ctx)
    menu = [{"name": flow.name, "description": flow.description} for flow in FLOWS]
    decision = IntakeDecision(
        **await ctx.run_node(
            intake_router, {"flows": menu, "transcript": transcript}, run_id="intake:route"
        )
    )
    first_steps = {flow.name: flow.steps[0].name for flow in FLOWS}
    if decision.flow in first_steps:
        yield Event(output=decision.flow, route=first_steps[decision.flow])
        return
    utterance = await ctx.run_node(intake_speaker, {"transcript": transcript}, run_id="intake:ask")
    yield message_event(text_of(utterance))


async def exit_flow(ctx: Context, node_input):
    """Terminal node for flows that cannot be completed."""
    utterance = await ctx.run_node(
        speaker,
        {
            "instruction": "Apologize that the request cannot be completed, explain the reason, "
            "and ask if there is anything else you can help with.",
            "reason": node_input,
            "transcript": conversation_transcript(ctx),
        },
        run_id="exit:close",
    )
    text = text_of(utterance)
    yield message_event(text, output=text, state=INTAKE_RESET)


def build_workflow() -> Workflow:
    """Construct the ADK workflow graph from the YAML flow definitions."""
    nodes = {step.name: make_step_node(step) for step in STEPS}
    nodes["exit"] = FunctionNode(func=exit_flow, name="exit", rerun_on_resume=True)
    intake_node = nodes["intake"] = FunctionNode(func=intake, name="intake", rerun_on_resume=True)

    def dispatch(ctx: Context, node_input):
        """Route each new turn to the step the conversation is currently in."""
        current = ctx.state.get("current_step", "intake")
        if current != "intake" and current not in nodes:
            logger.warning("Unknown current_step %r, falling back to intake.", current)
            current = "intake"
        return Event(output="", route=current)

    dispatcher = FunctionNode(func=dispatch, name="dispatch")
    edges: list = [
        ("START", dispatcher),
        (
            dispatcher,
            {"intake": intake_node, **{step.name: nodes[step.name] for step in STEPS}},
        ),
        (
            intake_node,
            {flow.steps[0].name: nodes[flow.steps[0].name] for flow in FLOWS},
        ),
    ]
    for step in STEPS:
        targets = {
            target
            for action in step.actions
            if action.result
            for target in (action.result.passed, action.result.fail)
            if target
        }
        if targets:
            edges.append((nodes[step.name], {target: nodes[target] for target in targets}))

    return Workflow(
        name="hybrid_agent",
        description="Customer-service agent driven by the flows in agent_flows.yaml.",
        edges=edges,
    )


root_agent = build_workflow()
