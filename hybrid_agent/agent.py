"""Hybrid ADK agent: flow graph built from agent_flows.yaml, conversation driven by LLMs."""

import logging
import pathlib

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.workflow import FunctionNode, Workflow
from langfuse import get_client
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from .events import answered_event, message_event, route_event
from .flow_schema import Action, AnswerOption, Step, load_config
from .response_schemas import (
    IntakeDecision,
    OptionSelection,
    Resolution,
    SpeakerResponse,
    Verdict,
)
from .tools import TOOLS

logger = logging.getLogger(__name__)

langfuse = get_client()
GoogleADKInstrumentor().instrument()

MODEL = "gemini-3.5-flash"
FLOWS_PATH = pathlib.Path(__file__).parent / "agent_flows.yaml"
CONFIG = load_config(FLOWS_PATH)

speaker = LlmAgent(
    name="speaker",
    model=MODEL,
    instruction=(
        f"{CONFIG.persona.strip()} "
        "Given an instruction, the conversation transcript, and the known facts, "
        "write the exact message to send to the user. If answer_options are provided, "
        "return exactly one user-facing label for each option, in the same order. "
        "Do not invent route IDs or option values."
    ),
    output_schema=SpeakerResponse,
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

option_selector = LlmAgent(
    name="option_selector",
    model=MODEL,
    instruction=(
        "Given the conversation transcript and the available options, return the zero-based "
        "index of the option the user has clearly chosen. Return -1 when no option has "
        "clearly been chosen. Never guess."
    ),
    output_schema=OptionSelection,
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
    "option_selector",
    *extractors,
}

FLOWS = CONFIG.flows
STEPS = [step for flow in FLOWS for step in flow.steps]
FLOW_BY_STEP = {step.name: flow for flow in FLOWS for step in flow.steps}


def step_reset(step: Step) -> dict:
    """State delta clearing a step's per-action flags so it re-runs from the top."""
    delta = {}
    for index, action in enumerate(step.actions):
        delta[f"done:{step.name}:{index}"] = False
        if action.message is not None:
            delta[f"asked:{step.name}:{index}"] = False
        if action.answer_options:
            delta[f"options:{step.name}:{index}"] = []
    return delta


# State delta that sends the conversation back to intake for a fresh request.
INTAKE_RESET = {
    "current_step": "intake",
    "facts": {},
    **{key: value for step in STEPS for key, value in step_reset(step).items()},
}


def conversation_transcript(ctx: Context) -> list[dict]:
    """Reconstruct the user-visible dialogue from the session events."""
    lines = []
    for event in ctx.session.events:
        # Internal LLM-node inputs are appended as user-authored events with an
        # isolation scope; skip them so payload JSON doesn't leak into the transcript.
        internal = event.author not in ["user", "hybrid_agent"] or event.isolation_scope
        if internal or not event.content:
            continue
        for part in event.content.parts or []:
            text = (part.text or "").strip()
            if text.startswith("{"):
                # Internal LLM-node inputs are appended to the session as user-authored
                # events carrying our JSON payloads; keep them out of the transcript.
                continue
            if text:
                lines.append({"author": event.author, "text": text})
            elif part.function_call and part.function_call.name == "adk_request_input":
                lines.append(
                    {"author": event.author, "text": (part.function_call.args or {}).get("message")}
                )
            elif part.function_response and part.function_response.name == "adk_request_input":
                reply = (part.function_response.response or {}).get("result")
                lines.append({"author": event.author, "text": str(reply)})
    return lines


def text_of(output: str) -> str:
    """Extract the plain text from an LLM agent's output."""
    return output.strip()


def speaker_response_of(output: dict) -> SpeakerResponse:
    """Parse structured speaker output from an LLM node."""
    return SpeakerResponse(**output)


async def run_speaker(ctx: Context, payload: dict, run_id: str) -> SpeakerResponse:
    """Run the speaker with transcript context guaranteed."""
    if "transcript" not in payload:
        payload = {**payload, "transcript": conversation_transcript(ctx)}
    return speaker_response_of(await ctx.run_node(speaker, payload, run_id=run_id))


def labels_for_options(response: SpeakerResponse, options: list[AnswerOption]) -> list[str]:
    """Return one user-facing label for each YAML option."""
    labels = [option.label for option in response.answer_options]
    labels.extend(option.name for option in options[len(labels) :])
    return labels[: len(options)]


def runtime_options(options: list[AnswerOption], labels: list[str]) -> list[dict]:
    """Pair generated labels with deterministic YAML option effects."""
    return [
        {
            "name": option.name,
            "label": label,
            **({"route": option.route} if option.route else {}),
            **({"value": option.value} if option.value else {}),
        }
        for option, label in zip(options, labels, strict=True)
    ]


async def select_answer_option(run_id: str, ctx: Context, memo: dict, options: list[dict]):
    """Match the conversation to one of the runtime options, or None when unclear."""
    selection = OptionSelection(
        **await ctx.run_node(
            option_selector,
            {
                "options": [
                    {"index": index, "label": option["label"], "name": option["name"]}
                    for index, option in enumerate(options)
                ],
                **memo,
            },
            run_id=f"{run_id}:select",
        )
    )
    if 0 <= selection.selected_index < len(options):
        return options[selection.selected_index]
    return None


async def ask_message_action(step_name: str, run_id: str, action: Action, ctx: Context, memo: dict):
    """Generate and send a message action to the user."""
    response = await run_speaker(
        ctx,
        {
            "instruction": action.message,
            "answer_options": [{"name": option.name} for option in action.answer_options],
            **memo,
        },
        run_id=f"{run_id}:speak",
    )
    labels = labels_for_options(response, action.answer_options)
    options = runtime_options(action.answer_options, labels)
    return message_event(
        response.message,
        state={
            f"asked:{run_id}": True,
            f"options:{run_id}": options,
            "current_step": step_name,
        },
        answer_options=options,
    )


async def ask_to_choose_option(
    step_name: str, run_id: str, ctx: Context, memo: dict, options: list[dict]
):
    """Ask the user to choose one of the existing options again."""
    response = await run_speaker(
        ctx,
        {
            "instruction": "Ask the user to choose one of the available options again.",
            "available_options": [
                {"name": option["name"], "label": option["label"]} for option in options
            ],
            **memo,
        },
        run_id=f"{run_id}:retry_options",
    )
    return message_event(
        response.message,
        state={"current_step": step_name},
        answer_options=options,
    )


def apply_selected_option(action: Action, selected: dict, memo: dict):
    """Apply a chosen option's deterministic effect: route away or store its value."""
    if selected.get("route"):
        return route_event(selected["route"], selected, memo["facts"])
    memo["facts"][action.message] = selected
    return answered_event(action, memo)


async def handle_answer_option_reply(
    step_name: str, run_id: str, action: Action, ctx: Context, memo: dict
):
    """Process a user's reply to a message with answer options."""
    options = ctx.state.get(f"options:{run_id}", [])
    selected = await select_answer_option(run_id, ctx, memo, options)
    if not selected:
        return await ask_to_choose_option(step_name, run_id, ctx, memo, options)
    return apply_selected_option(action, selected, memo)


async def handle_asked_message(
    step_name: str, run_id: str, action: Action, ctx: Context, memo: dict
):
    """Process the user's reply to a message action."""
    if action.answer_options:
        return await handle_answer_option_reply(step_name, run_id, action, ctx, memo)
    if action.result is None or action.result.fail is None:
        return answered_event(action, memo)

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
        return answered_event(action, memo)
    return route_event(action.result.fail, verdict.reason, memo["facts"])


async def resolve_existing_answer(action: Action, run_id: str, ctx: Context, memo: dict):
    """Resolve an optional question from existing transcript context."""
    resolution = Resolution(
        **await ctx.run_node(
            resolver, {"question": action.message, **memo}, run_id=f"{run_id}:resolve"
        )
    )
    if not resolution.answered:
        return False, None
    memo["facts"][action.message] = resolution.answer
    return True, answered_event(action, memo)


async def resolve_existing_option(action: Action, run_id: str, ctx: Context, memo: dict):
    """Resolve an optional multiple-choice question from existing transcript context."""
    options = runtime_options(action.answer_options, [o.name for o in action.answer_options])
    selected = await select_answer_option(run_id, ctx, memo, options)
    if selected is None:
        return False, None
    return True, apply_selected_option(action, selected, memo)


async def run_message_action(step_name: str, run_id: str, action: Action, ctx: Context, memo: dict):
    """Ask the user a message action's question, unless already asked or answerable.

    Asks as a normal message ending the turn; the answer arrives in the next
    turn's transcript and the dispatcher routes back to this step. Non-required
    questions are skipped when the resolver finds the answer in the conversation.
    """

    if ctx.state.get(f"asked:{run_id}"):
        return await handle_asked_message(step_name, run_id, action, ctx, memo)

    if not action.required:
        resolve = resolve_existing_option if action.answer_options else resolve_existing_answer
        resolved, event = await resolve(action, run_id, ctx, memo)
        if resolved:
            return event

    return await ask_message_action(step_name, run_id, action, ctx, memo)


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
    """A terminal step has only closing messages and no onward route."""
    return (
        all(action.message is not None for action in step.actions)
        and not any(action.answer_options for action in step.actions)
        and not any(action.result and action.result.passed for action in step.actions)
    )


async def execute_step(step: Step, ctx: Context) -> Event:
    """Run one YAML flow step and return its final event."""
    terminal = is_terminal(step)
    facts = dict(ctx.state.get("facts", {}))
    flow = FLOW_BY_STEP[step.name]
    memo = {
        "transcript": conversation_transcript(ctx),
        "facts": facts,
        "flow_instructions": flow.instructions,
        "step_task": step.task,
    }

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
        # Persist facts gathered this turn even when the step ends with a plain message.
        delta = {**done_delta, "facts": facts, **(event.actions.state_delta or {})}
        event.actions.state_delta = delta
        return event

    if not terminal:
        raise ValueError(f"Step {step.name} finished without routing anywhere.")
    instruction = " Then: ".join(a.message for a in step.actions if a.message is not None)
    response = await run_speaker(ctx, {"instruction": instruction, **memo}, f"{step.name}:close")
    return message_event(response.message, output=response.message, state=INTAKE_RESET)


async def report_step_failure(step: Step, ctx: Context, reason: str) -> Event:
    """Explain a fail-route entry to the user and end the turn.

    Running the target step's actions in the same turn would re-derive the same
    result from an unchanged transcript and loop forever (same class as the
    intake re-triage loop). Instead the step's flags are reset so the next user
    turn re-enters it from the top with fresh input.
    """
    response = await run_speaker(
        ctx,
        {
            "instruction": "Explain the problem to the user and ask them to try again.",
            "problem": reason,
            "step_task": step.task,
        },
        run_id=f"{step.name}:fail",
    )
    return message_event(response.message, state={**step_reset(step), "current_step": step.name})


def make_step_node(step: Step) -> FunctionNode:
    """Build a workflow node that executes one YAML flow step."""

    async def run_step(ctx: Context, node_input):
        # A non-empty string node_input is a failure reason from a fail route.
        if isinstance(node_input, str) and node_input:
            return await report_step_failure(step, ctx, node_input)
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
    """Match the user's request to a flow, or ask what they need.

    When reached via a fail route (node_input carries the failure reason),
    intake must not re-route in the same turn — deterministic re-triage of an
    unchanged transcript loops forever. It informs the user and ends the turn.
    """
    transcript = conversation_transcript(ctx)
    if isinstance(node_input, str) and node_input:
        utterance = await ctx.run_node(
            intake_speaker, {"transcript": transcript, "problem": node_input}, run_id="intake:fail"
        )
        # Consistent with report_step_failure: the abandoned flow's flags must not
        # linger, or re-entering it later would skip its done actions.
        yield message_event(text_of(utterance), state=INTAKE_RESET)
        return
    menu = [
        {
            "name": flow.name,
            "description": flow.description,
            "instructions": flow.instructions,
        }
        for flow in FLOWS
    ]
    decision = IntakeDecision(
        **await ctx.run_node(
            intake_router, {"flows": menu, "transcript": transcript}, run_id="intake:route"
        )
    )
    first_steps = {flow.name: flow.steps[0].name for flow in FLOWS}
    if decision.flow in first_steps:
        # output=None: a string output would be mistaken for a fail-route reason.
        yield Event(output=None, route=first_steps[decision.flow])
        return
    utterance = await ctx.run_node(intake_speaker, {"transcript": transcript}, run_id="intake:ask")
    yield message_event(text_of(utterance))


async def exit_flow(ctx: Context, node_input):
    """Terminal node for flows that cannot be completed."""
    instruction = (
        "Apologize that the request cannot be completed, explain the reason, "
        "and ask if there is anything else you can help with."
    )
    response = await run_speaker(
        ctx,
        {
            "instruction": instruction,
            "reason": node_input,
        },
        run_id="exit:close",
    )
    yield message_event(response.message, output=response.message, state=INTAKE_RESET)


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
            for target in [
                action.result.passed if action.result else None,
                action.result.fail if action.result else None,
                *(option.route for option in action.answer_options),
            ]
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
