# Deep Research Agent

`deep_research_agent` is an ADK 2.0 dynamic `Workflow` agent for research tasks.

The workflow uses three agent roles:

- `Planner_Agent` decides whether the request is clear enough for research and creates parallel subtasks.
- `Researcher_Agent` gathers concise source-backed findings for one assigned subtask.
- `Writer_Agent` synthesizes the plan and findings into the final cited report.

## Run

```bash
uv run adk run deep_research_agent
```

## Learnings

### Passing conversation history to workflow LLM nodes

`ctx.run_node(planner_agent, node_input)` creates a child node context from the parent invocation context. For normal workflow nodes, the child context receives the same `Session` object. For `single_turn` LLM workflow nodes, ADK creates a shallow session copy for the agent context; the `events` list is still shared. In both cases, existing `ctx.session.events` are available to the child execution path.

However, an `Agent` used as a `Workflow` node defaults to `single_turn` mode. ADK sets `include_contents` to `none` for `single_turn` nodes, so the model request does not automatically include prior session events. Setting `include_contents="default"` on a `single_turn` workflow node does not fix this because ADK overwrites it at runtime.

Setting `mode="chat"` lets the model see conversation history, but it is not appropriate here because a chat-mode workflow node can answer the user directly instead of returning a structured node output. In that case `ctx.run_node(...)` can return `None`, causing planner output validation to fail.

The confirmed working approach is to keep the planner in workflow-compatible `single_turn` behavior and pass the needed history explicitly in `node_input`:

```python
history = [
    {
        "author": event.author,
        "text": "\n".join(
            part.text for part in event.content.parts or [] if part.text
        ),
    }
    for event in ctx.session.events
    if event.content
    and event.author in {"user", "Planner_Agent", "Writer_Agent"}
]

plan = await ctx.run_node(
    planner_agent,
    {
        "conversation_history": history,
        "current_request": node_input.model_dump(),
    },
)
```

The planner instruction should mention both `current_request` and `conversation_history` so the structured `ResearchPlan` is based on the current turn plus relevant prior context.
