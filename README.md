# ADK playground

A small repo for trying Google ADK Python agents.

## Install

Install the project dependencies:

```bash
uv sync
```

Python `>=3.14` is required.

## Agents

### `deep_research_agent`

`deep_research_agent` is an ADK 2.0 dynamic `Workflow` agent.

It is built for research tasks. The planner checks whether the request is clear
enough to run. If not, it asks clarifying questions. If the request is clear, the
planner breaks it into independent subtasks, the workflow runs one researcher per
subtask in parallel, and the writer turns the findings into a cited report.

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

Commit `.env.example`. Do not commit `.env`.

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
