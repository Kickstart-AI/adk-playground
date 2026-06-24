# Recursive Agent

This ADK workflow runs a recursive assistant that can either answer directly or rewrite its
own system instruction and user prompt before trying again.

## Flow

1. The current session agent receives the conversation history and latest request.
2. The agent returns structured output indicating whether to answer directly or improve the prompt.
3. Direct answers are reviewed by a judge agent before being returned.
4. Prompt rewrites create the next session agent and continue up to the recursion limit.

If no answer is approved within the recursion limit, the workflow returns the last candidate answer
or a fallback message.
