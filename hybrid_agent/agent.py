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
    IntakeResponse,
    OptionSelection,
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
        "use flow_instructions for scope, tone, and handoff decisions, then write "
        "the exact message to send to the user. If answer_options are provided, "
        "return exactly one user-facing label for each option, in the same order. "
        "Do not invent route IDs or option values. "
        "Exception: when skip_if_already_answered is set and the conversation already "
        "unambiguously answers the question you were told to ask, do not write a message; "
        "set resolved_answer for plain questions or selected_index for answer_options instead. "
        "If the latest user message clearly cancels the current request or asks for a "
        "different request outside current_flow, set transfer_to_agent=intake and leave "
        "message empty. Do not transfer when the latest user message matches current_flow. "
        "Never guess — when in doubt, ask. "
        "Always write in the language the user uses in the transcript."
    ),
    output_schema=SpeakerResponse,
)

reflector = LlmAgent(
    name="reflector",
    model=MODEL,
    instruction=(
        "Judge whether the given validation instruction holds based on the known facts. "
        "Use flow_instructions for scope and handoff decisions. Explain your verdict "
        "briefly in the reason. If the latest user message clearly "
        "cancels the current request or asks for a different request outside current_flow, "
        "set transfer_to_agent=intake instead of judging. Do not transfer when the latest "
        "user message matches current_flow."
    ),
    output_schema=Verdict,
)

option_selector = LlmAgent(
    name="option_selector",
    model=MODEL,
    instruction=(
        "Given the conversation transcript and the available options, return the zero-based "
        "index of the option the user has clearly chosen. Return -1 when no option has "
        "clearly been chosen. Use flow_instructions for scope and handoff decisions. "
        "If the latest user message clearly cancels the current "
        "request or asks for a different request outside current_flow, set "
        "transfer_to_agent=intake instead of selecting. Do not transfer when the latest "
        "user message matches current_flow. Never guess."
    ),
    output_schema=OptionSelection,
)

extractors = {
    tool_name: LlmAgent(
        name=f"{tool_name}_args",
        model=MODEL,
        instruction=(
            f"Extract the arguments for tool '{tool_name}' from the conversation facts. "
            "Use flow_instructions for scope and handoff decisions. "
            "If the latest user message clearly cancels the current request or asks for a "
            "different request outside current_flow, set transfer_to_agent=intake instead "
            "of extracting arguments. Do not transfer when the latest user message matches "
            "current_flow."
        ),
        output_schema=args_model,
    )
    for tool_name, (_, args_model) in TOOLS.items()
}


intake_agent = LlmAgent(
    name="intake_agent",
    model=MODEL,
    instruction=(
        f"{CONFIG.persona.strip()} {CONFIG.instruction.strip()} "
        "When available flows are provided and the user's request clearly matches one, "
        "return its name in flow and leave message empty. Otherwise leave flow empty "
        "and write the message to send to the user. When a problem is provided, explain "
        "it, ask how else you can help, and leave flow empty. "
        "Always write in the language the user uses in the transcript."
    ),
    output_schema=IntakeResponse,
)

INTERNAL_AGENTS = {
    "speaker",
    "reflector",
    "intake_agent",
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


def speaker_response_of(output: dict) -> SpeakerResponse:
    """Parse structured speaker output from an LLM node."""
    return SpeakerResponse(**output)


async def run_speaker(ctx: Context, payload: dict, run_id: str) -> SpeakerResponse:
    """Run the speaker with transcript context guaranteed."""
    if "transcript" not in payload:
        payload = {**payload, "transcript": conversation_transcript(ctx)}
    return speaker_response_of(await ctx.run_node(speaker, payload, run_id=run_id))


def handoff_event(response):
    """Route to intake when structured output returns a transfer target."""
    if response.transfer_to_agent == "intake":
        return Event(output=None, route="intake", state=INTAKE_RESET)
    return None


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
    if event := handoff_event(selection):
        return event
    if 0 <= selection.selected_index < len(options):
        return options[selection.selected_index]
    return None


async def ask_message_action(step_name: str, run_id: str, action: Action, ctx: Context, memo: dict):
    """One speaker call: resolve the question from the transcript if allowed, else ask it."""
    response = await run_speaker(
        ctx,
        {
            "instruction": action.message,
            "answer_options": [{"name": option.name} for option in action.answer_options],
            **({"skip_if_already_answered": True} if not action.required else {}),
            **memo,
        },
        run_id=f"{run_id}:speak",
    )
    if not action.required:
        resolved, event = resolved_event(action, response, memo)
        if resolved:
            return event
    if event := handoff_event(response):
        return event
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


def resolved_event(action: Action, response: SpeakerResponse, memo: dict):
    """Apply the speaker's in-transcript resolution.

    Returns (resolved, event); the event may be None, meaning continue with
    the next action.
    """
    if action.answer_options:
        options = runtime_options(action.answer_options, [o.name for o in action.answer_options])
        if 0 <= response.selected_index < len(options):
            return True, apply_selected_option(action, options[response.selected_index], memo)
        return False, None
    if response.resolved_answer:
        memo["facts"][action.message] = response.resolved_answer
        return True, answered_event(action, memo)
    return False, None


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
    if event := handoff_event(response):
        return event
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
    if isinstance(selected, Event):
        return selected
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
    if event := handoff_event(verdict):
        return event
    if verdict.passed:
        return answered_event(action, memo)
    return route_event(action.result.fail, verdict.reason, memo["facts"])


async def run_message_action(step_name: str, run_id: str, action: Action, ctx: Context, memo: dict):
    """Ask the user a message action's question, unless already asked or answerable.

    Asks as a normal message ending the turn; the answer arrives in the next
    turn's transcript and the dispatcher routes back to this step. Non-required
    questions are skipped when the speaker finds the answer in the conversation.
    """
    if ctx.state.get(f"asked:{run_id}"):
        return await handle_asked_message(step_name, run_id, action, ctx, memo)
    return await ask_message_action(step_name, run_id, action, ctx, memo)


async def run_reflect_action(run_id: str, action: Action, ctx: Context, memo: dict):
    """Run a reflect action, including same-call handoff detection."""
    assert action.result is not None and action.result.fail is not None
    facts = memo["facts"]
    verdict = Verdict(
        **await ctx.run_node(reflector, {"instruction": action.reflect, **memo}, run_id=run_id)
    )
    if event := handoff_event(verdict):
        return event
    if not verdict.passed:
        return route_event(action.result.fail, verdict.reason, facts)
    return None


async def run_tool_action(run_id: str, action: Action, ctx: Context, memo: dict):
    """Run a tool action, including same-call handoff detection."""
    assert action.tool_call is not None
    assert action.result is not None and action.result.fail is not None
    facts = memo["facts"]
    tool, args_model = TOOLS[action.tool_call]
    args = args_model(**await ctx.run_node(extractors[action.tool_call], memo, run_id=run_id))
    if event := handoff_event(args):
        return event
    try:
        facts[f"{action.tool_call}_result"] = tool(**args.model_dump(exclude={"transfer_to_agent"}))
    except ValueError as error:
        return route_event(action.result.fail, str(error), facts)
    return None


async def run_action(step: Step, index: int, action: Action, ctx: Context, memo: dict):
    """Run one action; return an Event that ends the step, or None to continue.

    memo holds the shared "transcript" and "facts" passed to the LLM calls.
    """
    run_id = f"{step.name}:{index}"
    if action.message is not None:
        return await run_message_action(step.name, run_id, action, ctx, memo)
    # Guaranteed by flow_schema route validation.
    assert action.result is not None and action.result.fail is not None
    if action.reflect is not None:
        event = await run_reflect_action(run_id, action, ctx, memo)
    else:
        assert action.tool_call is not None  # only action kind left after message and reflect
        event = await run_tool_action(run_id, action, ctx, memo)
    if event:
        return event
    # Success: route onward if a pass target is set, otherwise continue with the next action.
    if action.result.passed:
        return route_event(action.result.passed, memo["facts"], memo["facts"])
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
        "current_flow": flow.name,
        "current_flow_description": flow.description,
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
    if event := handoff_event(response):
        return event
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
    if event := handoff_event(response):
        return event
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
            response = IntakeResponse(
                **await ctx.run_node(
                    intake_agent,
                    {"transcript": conversation_transcript(ctx)},
                    run_id=f"{step.name}:fallback",
                )
            )
            return message_event(response.message, state=INTAKE_RESET)

    return FunctionNode(func=run_step, name=step.name, rerun_on_resume=True)


async def intake(ctx: Context, node_input):
    """Match the user's request to a flow, or ask what they need.

    When reached via a fail route (node_input carries the failure reason),
    intake must not re-route in the same turn — deterministic re-triage of an
    unchanged transcript loops forever. It informs the user and ends the turn.
    """
    transcript = conversation_transcript(ctx)
    if isinstance(node_input, str) and node_input:
        # No flows in the payload: entered via a fail route, intake must not
        # re-route in the same turn. It explains the problem and ends the turn.
        response = IntakeResponse(
            **await ctx.run_node(
                intake_agent,
                {"transcript": transcript, "problem": node_input},
                run_id="intake:fail",
            )
        )
        # Consistent with report_step_failure: the abandoned flow's flags must not
        # linger, or re-entering it later would skip its done actions.
        yield message_event(response.message, state=INTAKE_RESET)
        return
    menu = [
        {
            "name": flow.name,
            "description": flow.description,
            "instructions": flow.instructions,
        }
        for flow in FLOWS
    ]
    response = IntakeResponse(
        **await ctx.run_node(
            intake_agent, {"flows": menu, "transcript": transcript}, run_id="intake:triage"
        )
    )
    first_steps = {flow.name: flow.steps[0].name for flow in FLOWS}
    if response.flow in first_steps:
        # output=None: a string output would be mistaken for a fail-route reason.
        yield Event(output=None, route=first_steps[response.flow])
        return
    yield message_event(response.message)


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
    if event := handoff_event(response):
        yield event
        return
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
        targets.add("intake")
        if targets:
            edges.append((nodes[step.name], {target: nodes[target] for target in targets}))
    edges.append((nodes["exit"], {"intake": intake_node}))

    return Workflow(
        name="hybrid_agent",
        description="Customer-service agent driven by the flows in agent_flows.yaml.",
        edges=edges,
    )


root_agent = build_workflow()
