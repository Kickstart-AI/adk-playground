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
