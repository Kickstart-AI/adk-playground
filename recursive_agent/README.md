# Recursive Agent

Recursive Agent is an ADK workflow that lets an assistant revise its own
instructions before it answers. The point is not to polish forever; it gets a few
chances to make the next answer better, then it must respond.

## Flow

1. The workflow sends the conversation history and latest request to the current
   session agent.
2. The agent returns structured output with one of two choices: answer directly, or
   rewrite its system instruction with an optional clearer user prompt.
3. If the system instruction changes, the workflow creates the next in-session agent
   and tries again.
4. The workflow stops when the agent returns a direct answer or reaches the recursion
   limit.

The agent is intentionally biased toward self-improvement. When it sees a real
prompt weakness, it rewrites. When the missing piece is user information, it asks
a concise clarifying question in `direct_answer` instead of guessing.

Rewrites are session-local. They live in process memory and are not written back
to source. If the workflow hits the recursion limit without a direct answer, it
returns a fallback message.

## Learnings

### Constructing agents dynamically

`LlmAgent.sub_agents` must contain constructed `Agent` / `LlmAgent` instances,
not dictionaries of constructor keyword arguments. Passing raw dictionaries into
`sub_agents` can produce Pydantic errors such as `sub_agents.0.model`,
`sub_agents.0.instruction`, or `sub_agents.0.tools` being extra inputs. That
usually means ADK is validating the value as the wrong agent shape, not that
those fields are invalid on an `LlmAgent`.

When converting generated schemas into agents, always replace the raw schema
list with the constructed list:

```python
agent_kwargs["sub_agents"] = [
    Agent(**construct_agent_kwargs(sub_agent))
    for sub_agent in agent_kwargs["sub_agents"]
]
```

Building `sub_agents` in a local variable is not enough; if it is not assigned
back into `agent_kwargs`, ADK still receives raw dictionaries.

### Structured output and tools

`output_schema` is useful for the recursive top-level agent because
`ctx.run_node(...)` can validate and return `AgentResponseSchema`. It should be
used carefully once tools or sub-agents enter the graph.

ADK has special handling for `output_schema` with tools. Depending on the model
and tool mix, ADK may inject an internal `set_model_response` tool so the model
can provide the final structured response. This is another tool call in the
event stream, so combining structured output, generated tools, and generated
sub-agents increases the chance of confusing the turn history.

Specialist sub-agents should generally not use the recursive
`AgentResponseSchema`. If a specialist receives the recursive schema, it is
asked to behave like an orchestrator instead of answering its specialist task.
For this agent, keep the recursive schema on the top-level recursive agents and
keep specialist agents plain.

### Prompt propagation

The recursive contract belongs in one common instruction that is prepended to
top-level recursive agents. That contract explains:

- the inputs: `conversation_history` and `current_request`
- the two valid outcomes: `direct_answer` or `agent_schema`
- the rule that both outcomes must not be populated together
- how to use `user_prompt`
- when to preserve the self-improvement behavior

Specialist sub-agents should not automatically receive this common recursive
instruction. They need narrow task instructions and clear descriptions so the
parent can decide when to call them.

Keep the bootstrap prompt small. If it repeats the common schema and recursion
rules, the prompt becomes harder to maintain and can drift from the actual
Pydantic schema.

### Chat sub-agents versus tool-like sub-agents

ADK `sub_agents` default to chat mode. A chat-mode sub-agent is a transfer
target reached through `transfer_to_agent`. This is a conversational handoff,
not a function call that returns a value to the parent.

With chat handoff, the sub-agent answer is emitted as an event in the session.
It is not necessarily returned as the parent agent's structured
`ctx.run_node(...)` result. If the workflow uses chat handoff, the final answer
may need to be extracted from `ctx.session.events`.

`single_turn` sub-agents are different. ADK treats them as tool-like agents that
complete a task without becoming the active conversational agent. They are
useful when the parent should receive a sub-agent result, but they are not
normal `transfer_to_agent` targets.

The practical choice is:

- use chat `sub_agents` for true handoff, then read final answers from events
- use tool-like agents when the parent must synthesize the final answer

### AgentTool wrappers

`AgentTool` is the cleanest option when the parent should call a specialist and
receive the specialist's result as tool output. Construct the specialist as a
plain `Agent` and append `AgentTool(specialist_agent)` to the parent `tools`
list.

Do not set `mode="single_turn"` on an agent wrapped with `AgentTool`.
`AgentTool.run_async(...)` creates a fresh `Runner(agent=self.agent)`, so the
wrapped agent becomes the root agent for that child run. ADK requires a root
`LlmAgent` to have `mode="chat"` or no explicit mode. If the wrapped agent has
`mode="single_turn"`, ADK raises:

```text
ValueError: LlmAgent as root agent must have mode='chat', but got mode='single_turn'.
```

`mode="single_turn"` belongs to ADK's built-in `sub_agents` machinery, not to
agents wrapped directly in `AgentTool`.

`AgentTool` agents are tools, not transfer targets. Use them when
call-and-return specialist behavior is more important than conversational
handoff.

### Event history and debugging

`ctx.run_node(current_agent, ...)` can perform multiple internal ADK steps before
it returns. For an agent with tools or sub-agents, one `run_node` call can
include:

1. a parent model call
2. a tool or sub-agent call
3. a function response event
4. a follow-up parent model call to consume the result

Errors thrown inside `run_node` can therefore happen during ADK's internal
follow-up model request, not only during the first model call.

The SQLite session database can contain both a function call and its matching
function response while ADK still raises:

```text
ValueError: No function call event found for function responses ids: {...}
```

That happens because ADK does not always pass the raw session history to the
model. It filters and slices events into a current-turn view before rearranging
function calls and function responses. In the observed failure, the parent
function call and function response both existed in SQLite, but a sub-agent event
was treated as an "other agent reply." ADK started the current turn at that
sub-agent event, so the filtered slice contained the later function response but
not the earlier matching function call.

`include_contents="none"` does not necessarily avoid this. ADK can still keep
current-turn tool/function history even when broader conversation history is
excluded.

When debugging these issues, check both:

- the raw session events in SQLite
- the ADK code path that filters those events for the current request

The raw database can look correct while the filtered request contents are not.

### Current design rule for this agent

For this recursive agent, prefer this structure:

- top-level recursive agents use `AgentResponseSchema`
- generated specialist agents are plain `Agent` instances
- generated specialist agents do not receive `COMMON_AGENT_INSTRUCTION`
- specialist agents are exposed to the parent through `AgentTool`
- `AgentTool`-wrapped agents do not set `mode="single_turn"`
- the parent recursive agent synthesizes the final `direct_answer`

Use chat `sub_agents` only if true handoff is required and the workflow is
prepared to extract the answer from session events.
