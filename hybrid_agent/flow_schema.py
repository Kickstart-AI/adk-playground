"""Pydantic schema and loader for agent_flows.yaml."""

import pathlib

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ActionResult(BaseModel):
    """Routing targets for an action's outcome (step names, or 'exit')."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    passed: str | None = Field(None, alias="pass")
    fail: str | None = None


class AnswerOption(BaseModel):
    """A selectable answer option with a deterministic runtime effect."""

    model_config = ConfigDict(extra="forbid")

    name: str
    route: str | None = None
    value: str | None = None

    @model_validator(mode="after")
    def check_exactly_one_effect(self):
        """Ensure the option either routes or stores a value."""
        if sum(effect is not None for effect in [self.route, self.value]) != 1:
            raise ValueError("AnswerOption must have exactly one of route or value.")
        return self


class Action(BaseModel):
    """One action in a step: exactly one of message, reflect, or tool_call."""

    model_config = ConfigDict(extra="forbid")

    message: str | None = None
    reflect: str | None = None
    tool_call: str | None = None
    required: bool = False  # message only: always ask, never resolve silently
    answer_options: list[AnswerOption] = Field(default_factory=list)
    result: ActionResult | None = None

    @model_validator(mode="after")
    def check_exactly_one_kind(self):
        """Ensure the action is exactly one of the three kinds."""
        kinds = [self.message, self.reflect, self.tool_call]
        if sum(k is not None for k in kinds) != 1:
            raise ValueError("Action must have exactly one of message, reflect, or tool_call.")
        return self

    @model_validator(mode="after")
    def check_answer_options(self):
        """Restrict answer options to message actions."""
        if self.answer_options and self.message is None:
            raise ValueError("answer_options are only supported for message actions.")
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
    instructions: str = ""
    steps: list[Step]


class AgentConfig(BaseModel):
    """Top-level agent_flows.yaml contents."""

    model_config = ConfigDict(extra="forbid")

    persona: str
    instruction: str
    flows: list[Flow]

    @model_validator(mode="after")
    def check_routes(self):
        """Ensure all YAML routes point at known workflow targets."""
        step_names = [step.name for flow in self.flows for step in flow.steps]
        duplicate_steps = {name for name in step_names if step_names.count(name) > 1}
        if duplicate_steps:
            raise ValueError(f"Duplicate step names are not supported: {sorted(duplicate_steps)}")

        targets = set(step_names) | {"exit", "intake"}
        for flow in self.flows:
            for step in flow.steps:
                for action in step.actions:
                    routes = []
                    if action.result:
                        routes.extend([action.result.passed, action.result.fail])
                    routes.extend(option.route for option in action.answer_options)
                    unknown = [route for route in routes if route and route not in targets]
                    if unknown:
                        raise ValueError(f"Unknown route target in step {step.name}: {unknown}")
        return self


def load_config(path: pathlib.Path) -> AgentConfig:
    """Load and validate the agent configuration from YAML."""
    return AgentConfig.model_validate(yaml.safe_load(path.read_text()))
