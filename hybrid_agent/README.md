# Hybrid Agent

A customer-service agent that combines a deterministic flow graph with LLM-driven conversation.
The flows (steps, tool calls, routing) are declared in `agent_flows.yaml`, validated by the
Pydantic schema in `flow_schema.py`, and compiled into an ADK `Workflow` graph in `agent.py`.

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
- All user-facing text is LLM-generated; every LLM call receives the conversation transcript.
- Langfuse tracing is enabled via OpenInference instrumentation.

## Learnings from this session

- LLM agents used as workflow nodes run in `single_turn` mode with `include_contents='none'` — they see no conversation history. Anything they need (transcript, facts) must be passed explicitly in the node input.
- `ctx.run_node()` on an `LlmAgent` without `output_schema` returns a plain `str`, even though the docs' node-output table says `types.Content`.
- `Event(state=...)` is consumed at construction into `actions.state_delta`; to amend an already-built event, mutate `event.actions.state_delta`, not `event.state`.
- Routed edges must be `(node, {route: target})` dicts — the 3-tuple `(node, target, route)` form from the cheatsheet fails Workflow validation in ADK 2.0.0.
- `RequestInput` interrupts must be answered with a `FunctionResponse` carrying the interrupt id (`adk web`/CLI do this automatically); plain-text replies never resolve them. We dropped interrupts in favor of normal messages plus a dispatcher that routes each turn to `current_step`.
- Graph validation rejects unreachable nodes, which surfaced a routing bug in the original YAML (`get_order_details` skipping `validate_order_eligibility`).
- The single-turn LlmAgent wrapper appends each node input to the session as a user-authored event. Rebuilding a transcript from session events therefore picks up internal JSON payloads, which snowball recursively into every later call (author filtering isn't enough — `isolation_scope` isn't always set, so we also drop user events whose text starts with `{`).
- Routing a failure back to intake for same-turn re-triage loops forever: the transcript is unchanged, so the router deterministically picks the same flow and the same failure recurs (observed 16 cycles in one turn). Fail-routes into a triage node must end the turn and wait for new user input.

## TODOs

- [x] Add multiple choice possibility (`answer_options` on message actions)
- [ ] Each node should have the option to hand off to the intake agent if the user changes their mind or they are in the wrong step or flow or something
- [x] Simplify where possible and sensible the number of LLM calls (intake router+speaker merged; ask-path resolver folded into the speaker call)
- [ ] Exercise the `change_account` flow end-to-end (login-number loop, both answer-option kinds)
- [ ] Decide whether `result.fail: intake` should keep facts for a later flow re-entry (currently full reset, same as exit)
