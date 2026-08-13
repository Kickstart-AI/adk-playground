"""Anthropic hosted-search model behavior for committee answerers."""

from copy import deepcopy
from typing import Any

from google.adk.models.lite_llm import LiteLlm, LiteLLMClient


def _tool_call_name(tool_call: Any) -> Any:
    """Read a normalized LiteLLM function-call name."""
    function = (
        tool_call.get("function")
        if isinstance(tool_call, dict)
        else getattr(tool_call, "function", None)
    )
    return function.get("name") if isinstance(function, dict) else getattr(function, "name", None)


def _remove_provider_search_calls(response: Any) -> Any:
    """Remove only normalized provider-owned web-search function calls."""
    filtered = deepcopy(response)
    choices = filtered.get("choices", []) if isinstance(filtered, dict) else filtered.choices
    for choice in choices or []:
        message = choice.get("message") if isinstance(choice, dict) else choice.message
        if message is None:
            continue
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else message.tool_calls
        if tool_calls is None:
            continue
        retained = [call for call in tool_calls if _tool_call_name(call) != "web_search"]
        if isinstance(message, dict):
            message["tool_calls"] = retained
        else:
            message.tool_calls = retained
    return filtered


def _has_application_web_search(tools: list[dict] | None) -> bool:
    """Check whether the request declares its own web_search function."""
    return any(_tool_call_name(tool) == "web_search" for tool in tools or [])


class AnthropicAnswererClient(LiteLLMClient):
    """Force hosted-search answerer requests to complete before returning."""

    async def _provider_acompletion(self, model, messages, tools, **kwargs):
        """Delegate to LiteLLM's ordinary asynchronous completion client."""
        return await super().acompletion(model=model, messages=messages, tools=tools, **kwargs)

    async def acompletion(self, model, messages, tools, **kwargs):
        """Complete the targeted answerer request with provider streaming disabled."""
        response_format = kwargs.get("response_format")
        native_format = (
            response_format.get("json_schema") if isinstance(response_format, dict) else None
        )
        targeted = (
            model == "anthropic/claude-fable-5"
            and bool(kwargs.get("web_search_options"))
            and isinstance(native_format, dict)
            and isinstance(native_format.get("schema"), dict)
        )
        provider_kwargs = dict(kwargs)
        if targeted:
            provider_kwargs["stream"] = False
            provider_kwargs.pop("response_format", None)
            provider_kwargs["output_config"] = {
                **provider_kwargs.get("output_config", {}),
                "format": {"type": "json_schema", "schema": native_format["schema"]},
            }
        response = await self._provider_acompletion(model, messages, tools, **provider_kwargs)
        if targeted and not _has_application_web_search(tools):
            return _remove_provider_search_calls(response)
        return response


class AnthropicAnswererModel(LiteLlm):
    """LiteLLM model that emits one completed response to ADK stream callers."""

    def __init__(self, model: str, **kwargs):
        """Initialize the answerer model with its custom LiteLLM client."""
        super().__init__(model=model, **kwargs)

    async def generate_content_async(self, llm_request, stream=False):
        """Hide provider streaming while preserving ADK's completed response contract."""
        async for response in super().generate_content_async(llm_request, stream=False):
            response.partial = False
            response.turn_complete = True
            yield response
