"""Pydantic schema and loader for agent_flows.yaml."""

import pathlib

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ActionResult(BaseModel):
    """Routing targets for an action's outcome (step names, or 'exit')."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    passed: str | None = Field(None, alias="pass")
    fail: str | None = None


class Action(BaseModel):
    """One action in a step: exactly one of message, reflect, or tool_call."""

    model_config = ConfigDict(extra="forbid")

    message: str | None = None
    reflect: str | None = None
    tool_call: str | None = None
    required: bool = False  # message only: always ask, never resolve silently
    result: ActionResult | None = None

    @model_validator(mode="after")
    def check_exactly_one_kind(self):
        """Ensure the action is exactly one of the three kinds."""
        kinds = [self.message, self.reflect, self.tool_call]
        if sum(k is not None for k in kinds) != 1:
            raise ValueError("Action must have exactly one of message, reflect, or tool_call.")
        return self

    @model_validator(mode="after")
    def check_routes(self):
        """Reflect and tool_call actions need a fail route; pass is optional (continue)."""
        if (self.reflect is not None or self.tool_call is not None) and not (
            self.result and self.result.fail
        ):
            raise ValueError("Reflect and tool_call actions require result.fail.")
        return self


class Step(BaseModel):
    """A named step consisting of a task description and its actions."""

    model_config = ConfigDict(extra="forbid")

    name: str
    task: str
    actions: list[Action]


class Flow(BaseModel):
    """A conversation flow the intake can route to."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    steps: list[Step]


class AgentConfig(BaseModel):
    """Top-level agent_flows.yaml contents."""

    model_config = ConfigDict(extra="forbid")

    persona: str
    instruction: str
    flows: list[Flow]


def load_config(path: pathlib.Path) -> AgentConfig:
    """Load and validate the agent configuration from YAML."""
    return AgentConfig.model_validate(yaml.safe_load(path.read_text()))
