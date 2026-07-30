# ADK playground

A small repo for trying Google ADK Python agents.

## Table of contents

- [Install](#install)
- [Directory tree](#directory-tree)
- [Agents](#agents)
- [Create an agent](#create-an-agent)
- [Authenticate with an API key](#authenticate-with-an-api-key)
- [Set up Langfuse](#set-up-langfuse)
- [Start an agent](#start-an-agent)

## Install

Install the project dependencies:

```bash
uv sync
```

Python `>=3.13,<3.14` is required.

## Directory tree

```text
.
├── deep_research_agent/   # Dynamic ADK workflow for research tasks
├── committee_agent/       # Anonymous multi-model answer committee
├── hybrid_agent/          # Flow-driven customer-service agent
├── recursive_agent/       # Recursive agent experiment
├── .agents/               # Local agent skills and supporting metadata
├── .adk/                  # Local ADK runtime state and artifacts
├── mlartifacts/           # Local MLflow/Langfuse-related artifacts
├── .env.example           # Example environment variables for agents
├── docker-compose.yaml    # Local Langfuse stack
├── pyproject.toml         # Python project configuration
├── uv.lock                # Locked Python dependencies
└── README.md              # Project overview
```

## Agents

### `deep_research_agent`

`deep_research_agent` is an ADK 2.0 dynamic `Workflow` agent.

It is built for research tasks. The planner checks whether the request is clear
enough to run. If not, it asks clarifying questions. If the request is clear, the
planner breaks it into independent subtasks, the workflow runs one researcher per
subtask in parallel, and the writer turns the findings into a cited report.

### `committee_agent`

`committee_agent` asks GPT-5.6 Sol, Gemini 3.1 Pro, and Claude Fable 5 to answer
the current question using the user-visible conversation history.

It removes model identities, shuffles the answers independently for each model,
and asks every model to rank all answers. A Borda count selects the winning
answer. The model that wrote it then synthesizes the final response from all
viewpoints without exposing models, candidates, or rankings to the user.

### `hybrid_agent`

`hybrid_agent` is a customer-service agent that combines a deterministic YAML
flow graph with LLM-generated conversation.

It supports online-shop tasks such as returns and account changes. The YAML file
defines steps, tool calls, routing, and answer options; the LLM nodes phrase
messages, infer already-provided answers from the transcript, and hand back to
intake when the user changes intent.

### `recursive_agent`

`recursive_agent` is an experiment in recursive agent self-improvement.

Each turn starts with a general agent that either answers directly or proposes a
replacement agent schema. The workflow can rebuild the active agent with new
instructions, tools, code execution, or specialist sub-agents, then retries up to
a recursion limit before returning a final answer.

## Create an agent

Create a new agent folder:

```bash
uv run adk create my_agent
```

Every ADK agent needs a `root_agent` in `my_agent/agent.py`:

```python
from google.adk.agents.llm_agent import Agent

root_agent = Agent(
    name="my_agent",
    model="gemini-flash-latest",
    instruction="You are a helpful assistant.",
)
```

## Authenticate with an API key

Copy the example env file into the agent folder:

```bash
cp .env.example my_agent/.env
```

Then edit `my_agent/.env` and set `GOOGLE_API_KEY`.

`committee_agent` also requires `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`.

Commit `.env.example`. Do not commit `.env`.

## Set up Langfuse

Start the local Langfuse stack:

```bash
docker compose up
```

Open `http://localhost:3033`, create or select a project, and create API keys in
the web interface.

Set the keys in the agent `.env` file:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=http://localhost:3033
```

## Start an agent

Run the current agent in the terminal:

```bash
uv run adk run deep_research_agent
```

Or start the ADK web UI:

```bash
uv run adk web --port 8000
```

Open `http://localhost:8000` and select the agent.
