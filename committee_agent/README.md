# Committee agent

`committee_agent` is a local ADK workflow that uses three models:

- GPT-5.6 Sol
- Gemini 3.1 Pro
- Claude Fable 5

For each user turn, all models receive the current question and the complete
user-visible conversation. They produce answers in parallel. The workflow assigns
opaque answer IDs and gives each model a separately shuffled list to rank.

A Borda count selects the winning answer. First-place votes break equal scores,
followed by the opaque candidate ID for deterministic final ties. The model that
wrote the winning answer synthesizes all anonymous answers into the final response.

Model names are private workflow state. Ranking and synthesis prompts contain only
opaque candidate IDs and answer text.

## Configure

Copy the root environment example:

```bash
cp .env.example committee_agent/.env
```

Set these values:

```env
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

Do not commit `committee_agent/.env`.

The workflow logs its ADK spans to Langfuse. It groups each run by the ADK
session ID and flushes pending spans before the workflow exits.

## Run

```bash
uv run adk run committee_agent
```
