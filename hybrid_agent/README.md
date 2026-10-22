# Hybrid Agent

A customer-service agent that combines a deterministic flow graph with LLM-driven conversation.
The flows (steps, tool calls, routing) are declared in `agent_flows.yaml`, validated by the
Pydantic schema in `flow_schema.py`, and compiled into an ADK `Workflow` graph in `agent.py`.

## Motivation
We want to combine the flexibility of GenAI with the predictability of a rule-based bot, in particular:

- Have guarantees that required flow actions run in YAML order, unless an optional question is already answered by the transcript.
- Natural conversation.
- Have the answers tailored to the user (no hardcoded answers, language flexibility).
- Have guarantees that the bot will ask or say a required message (like a disclaimer, asking for confirmation, ...).
- Have the ability to infer intent and context from the conversation history rather than explicitly asking for every bit of information.
- Change tone of voice or instructions based on the situation rather than one general system prompt.
- Be able to exit a flow when the user changes intent or is in the wrong flow.

## How it works

- **dispatch** routes each turn to the current step (`current_step` in session state), defaulting to **intake**.
- **intake** is a single LLM call that either routes into the matching flow or returns the conversational reply to send.
- Each YAML step becomes a workflow node. Actions:
  - `message`: phrased by the LLM speaker and sent as a normal reply ending the turn; the answer is read from the transcript next turn. The same speaker call may instead resolve the question from the conversation (skipping the ask), unless marked `required: true`. Optional `result.fail` routes away when the reply is judged negative.
  - `reflect`: an LLM verdict on a validation instruction; `result.fail` routes away on failure.
  - `tool_call`: LLM-extracted arguments, executed against the (stubbed) tools; routes to `result.pass`/`result.fail`.
  - `result.pass` is optional everywhere: if omitted, execution continues with the next action.
- Per-action `done:` flags make reruns resume at the next pending action instead of re-executing the step.
- **exit** apologizes with the failure reason and fully resets state back to intake; unexpected node errors also fall back to intake.
- `result.fail: intake` abandons the flow: intake tells the user the problem, fully resets state (like exit), and triages the next user message normally. A fail route targeting a step likewise ends the turn — the speaker explains the problem and the step's flags are reset so it re-runs from the top next turn (in-turn re-execution of an unchanged transcript would loop).
- `message` actions can carry `answer_options` with a deterministic `route` or `value` per option; labels are LLM-phrased, selection is matched by an LLM against the transcript (also pre-ask, to skip already-answered choices).
- All user-facing text is LLM-generated; every LLM call receives callback-built
  visible conversation history.
- Langfuse tracing is enabled via OpenInference instrumentation.

## Learnings from this session

- LLM agents used as workflow nodes run in `single_turn` mode with `include_contents='none'` — they see no history automatically. The shared `ConversationHistoryCallback` is installed in each LLM node's `before_model_callback` and writes visible history to `LlmRequest.contents`; node JSON carries only current-stage task data and facts.
- The history callback includes only genuine user events and visible `hybrid_agent` replies. Internal intake, speaker, selector, reflector, and extractor outputs are not conversation messages. Facts and routing data reach each node through its current-stage payload. Active tool calls and their results remain in the current model request.
- `ctx.run_node()` on an `LlmAgent` without `output_schema` returns a plain `str`, even though the docs' node-output table says `types.Content`.
- `Event(state=...)` is consumed at construction into `actions.state_delta`; to amend an already-built event, mutate `event.actions.state_delta`, not `event.state`.
- Routed edges must be `(node, {route: target})` dicts — the 3-tuple `(node, target, route)` form from the cheatsheet fails Workflow validation in ADK 2.2.0.
- `RequestInput` interrupts must be answered with a `FunctionResponse` carrying the interrupt id (`adk web`/CLI do this automatically); plain-text replies never resolve them. We dropped interrupts in favor of normal messages plus a dispatcher that routes each turn to `current_step`.
- Graph validation rejects unreachable nodes, which surfaced a routing bug in the original YAML (`get_order_details` skipping `validate_order_eligibility`).
- ADK 2.2.0 appends synthetic single-turn node inputs as user-authored events with empty invocation IDs. The callback excludes those synthetic events, retains genuine Runner-created user events with non-empty invocation IDs even when scoped, and excludes scoped internal-agent events. This prevents internal JSON payloads from snowballing into later calls while preserving real user turns. The shared helper can reconstruct human-input replies from `adk_request_input` function-call/response parts; this hybrid workflow currently uses ordinary message events for user replies. Imported genuine events without invocation IDs cannot be distinguished from synthetic node inputs.
- Routing a failure back to intake for same-turn re-triage loops forever: the transcript is unchanged, so the router deterministically picks the same flow and the same failure recurs (observed 16 cycles in one turn). Fail-routes into a triage node must end the turn and wait for new user input.

## TODOs

- [x] Add multiple choice possibility (`answer_options` on message actions)
- [x] Each node should have the option to hand off to the intake agent if the user changes their mind or they are in the wrong step or flow or something
- [x] Simplify where possible and sensible the number of LLM calls (intake router+speaker merged; ask-path resolver folded into the speaker call)
- [x] Exercise the `change_account` flow end-to-end (login-number loop, both answer-option kinds)
- [ ] Decide whether `result.fail: intake` should keep facts for a later flow re-entry (currently full reset, same as exit)
