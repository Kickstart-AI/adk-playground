# Hybrid Agent

A customer-service agent that combines a deterministic flow graph with LLM-driven conversation.
The flows (steps, tool calls, routing) are declared in `agent_flows.yaml`, validated by the
Pydantic schema in `flow_schema.py`, and compiled into an ADK `Workflow` graph in `agent.py`.

## How it works

- **dispatch** routes each turn to the current step (`current_step` in session state), defaulting to **intake**.
- **intake** matches the user's request against the flow descriptions and routes into the matching flow, or converses naturally.
- Each YAML step becomes a workflow node. Actions:
  - `message`: phrased by the LLM speaker and sent as a normal reply ending the turn; the answer is read from the transcript next turn. Skipped when the resolver finds the answer already in the conversation, unless marked `required: true`. Optional `result.fail` routes away when the reply is judged negative.
  - `reflect`: an LLM verdict on a validation instruction; `result.fail` routes away on failure.
  - `tool_call`: LLM-extracted arguments, executed against the (stubbed) tools; routes to `result.pass`/`result.fail`.
  - `result.pass` is optional everywhere: if omitted, execution continues with the next action.
- Per-action `done:` flags make reruns resume at the next pending action instead of re-executing the step.
- **exit** apologizes with the failure reason and resets state back to intake; unexpected node errors also fall back to intake.
- All user-facing text is LLM-generated; every LLM call receives the conversation transcript.
- Langfuse tracing is enabled via OpenInference instrumentation.

## Learnings from this session

- LLM agents used as workflow nodes run in `single_turn` mode with `include_contents='none'` — they see no conversation history. Anything they need (transcript, facts) must be passed explicitly in the node input.
- `ctx.run_node()` on an `LlmAgent` without `output_schema` returns a plain `str`, even though the docs' node-output table says `types.Content`.
- `Event(state=...)` is consumed at construction into `actions.state_delta`; to amend an already-built event, mutate `event.actions.state_delta`, not `event.state`.
- Routed edges must be `(node, {route: target})` dicts — the 3-tuple `(node, target, route)` form from the cheatsheet fails Workflow validation in ADK 2.0.0.
- `RequestInput` interrupts must be answered with a `FunctionResponse` carrying the interrupt id (`adk web`/CLI do this automatically); plain-text replies never resolve them. We dropped interrupts in favor of normal messages plus a dispatcher that routes each turn to `current_step`.
- Graph validation rejects unreachable nodes, which surfaced a routing bug in the original YAML (`get_order_details` skipping `validate_order_eligibility`).

## TODOs

- [ ] Add multiple choice possibility to ActionResult
- [ ] Each node should have the option to hand off to the intake agent if the user changes their mind or they are in the wrong step or flow or something
- [ ] Simplify where possible and sensible the number of LLM calls, for example intake router and intake speaker could be one call etc.
