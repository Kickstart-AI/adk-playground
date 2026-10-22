# Deep Research Agent

`deep_research_agent` is an ADK 2.2.0 dynamic `Workflow` agent for research tasks.

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

Workflow nodes use `include_contents="none"` and receive only current-stage task
data in their JSON payloads. The planner receives only a static task directive;
it does not contain the current request or a serialized history field.
`ConversationHistoryCallback` runs in the planner's `before_model_callback` and
reconstructs the visible dialogue in `LlmRequest.contents` before the model call.

This keeps history as model messages rather than nested JSON and avoids passing
`conversation_history` or `transcript` through workflow state. The researcher
and writer receive their assigned research data and findings as stage payloads.

ADK 2.2.0 records synthetic workflow node inputs as user-authored events with
empty invocation IDs. The callback excludes those events, while retaining genuine
Runner-created user events with non-empty invocation IDs, including scoped user
events. Scoped internal-agent events are excluded.
