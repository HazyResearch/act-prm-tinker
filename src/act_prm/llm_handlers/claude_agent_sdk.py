"""
Claude Agent SDK LLM handlers

Two subclasses of LLM using the Claude Agent SDK:
- ClaudeQueryLLM: Uses `query()` for stateless, one-shot interactions
- ClaudeClientLLM: Uses `ClaudeSDKClient` for persistent, multi-turn sessions
"""

import asyncio
import json

from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    AssistantMessage,
    ResultMessage,
    query,
    tool as sdk_tool,
    create_sdk_mcp_server,
)

from .base import LLM
from .types import ActionFromLLM


@dataclass
class ClaudeAgentResponse:
    """
    Response container that mirrors the structure expected by get_actions().
    Collects AssistantMessages from the Claude Agent SDK stream.
    """

    assistant_messages: list[AssistantMessage] = field(default_factory=list)
    result: ResultMessage | None = None
    usage: dict[str, int] | None = None

    @property
    def output(self) -> list[Any]:
        """Flatten all content blocks from all assistant messages."""
        blocks = []
        for msg in self.assistant_messages:
            blocks.extend(msg.content)
        return blocks


def _convert_tools_to_mcp_server(
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Convert OpenAI-style function tool definitions to an in-process MCP server.

    Expects tools in the format:
    [{"type": "function", "name": "tool_name", "description": "...",
      "parameters": {"type": "object", "properties": {...}, "required": [...]}}]
    """
    if not tools:
        return {}, []

    sdk_tools = []
    tool_names = []
    for tool_def in tools:
        name = tool_def.get("name", tool_def.get("function", {}).get("name", ""))
        description = tool_def.get(
            "description", tool_def.get("function", {}).get("description", "")
        )
        parameters = tool_def.get(
            "parameters", tool_def.get("function", {}).get("parameters", {})
        )

        @sdk_tool(name, description, parameters)
        async def _placeholder_handler(args: dict[str, Any]) -> dict[str, Any]:
            # This handler should never be called — the environment handles tool
            # execution. Return an error if somehow invoked.
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: tool '{name}' should be handled by the environment, not the SDK.",
                    }
                ],
                "isError": True,
            }

        sdk_tools.append(_placeholder_handler)
        tool_names.append(f"mcp__env_tools__{name}")

    server = create_sdk_mcp_server(
        name="env_tools",
        version="1.0.0",
        tools=sdk_tools,
    )
    return {"env_tools": server}, tool_names


def _messages_to_prompt(
    system_prompt: str | None,
    messages: list[dict[str, Any]],
) -> str:
    """
    Convert a list of chat messages into a single prompt string for query().

    For single-turn use, we format the conversation history as a prompt.
    """
    parts = []
    if system_prompt:
        parts.append(f"[System]\n{system_prompt}")

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if msg.get("type") == "function_call_output":
            call_id = msg.get("call_id", "unknown")
            parts.append(f"[Tool Result ({call_id})]\n{content}")
        elif role == "assistant":
            parts.append(f"[Assistant]\n{content}")
        elif role == "user":
            parts.append(f"[User]\n{content}")
        elif role == "system":
            parts.append(f"[System]\n{content}")

    return "\n\n".join(parts)


class ClaudeQueryLLM(LLM):
    """
    Claude Agent SDK LLM using `query()` — stateless, one-shot interactions.

    Each call to `sample()` creates a fresh session with no memory of previous
    interactions. Best for independent rollout steps where conversation context
    is managed externally via message lists.

    Example usage:
    ```
    llm = ClaudeQueryLLM(model="claude-sonnet-4-6")
    responses = llm.sample(
        system_prompt="You are a helpful agent.",
        messages=[{"role": "user", "content": "Hello"}],
        tools=[{"type": "function", "name": "search", ...}],
    )
    actions = llm.get_actions(responses[0])
    ```
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        generation_config: Any | None = None,
        max_turns: int | None = 1,
        permission_mode: str = "bypassPermissions",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, generation_config=generation_config, **kwargs)
        self.max_turns = max_turns
        self.permission_mode = permission_mode
        self.last_usage = None

    async def _query_single(
        self,
        system_prompt: str | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_new_tokens: int | None = None,
        **generation_kwargs: Any,
    ) -> ClaudeAgentResponse | None:
        """Run a single query() call and collect the response."""
        prompt = _messages_to_prompt(system_prompt, messages)

        mcp_servers, allowed_tools = _convert_tools_to_mcp_server(tools or [])

        options = ClaudeAgentOptions(
            model=self.model,
            system_prompt=system_prompt,
            max_turns=self.max_turns,
            permission_mode=self.permission_mode,
            mcp_servers=mcp_servers,
            allowed_tools=allowed_tools,
        )

        response = ClaudeAgentResponse()

        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    response.assistant_messages.append(message)
                elif isinstance(message, ResultMessage):
                    response.result = message
                    if message.usage:
                        response.usage = {
                            "input_tokens": message.usage.get("input_tokens", 0),
                            "output_tokens": message.usage.get("output_tokens", 0),
                        }
        except Exception as e:
            print(f"ClaudeQueryLLM query error: {type(e).__name__}: {e}")
            return None

        # Track tokens
        if response.usage:
            self.last_usage = self._track_tokens(
                type("Usage", (), response.usage)()
            )
            if self.last_usage is not None:
                print("Claude Agent SDK token usage:")
                for k, v in self.last_usage.items():
                    print(f"-> {k}: {v}")

        return response

    def _sample_single(
        self,
        system_prompt: str | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_new_tokens: int | None = None,
        **generation_kwargs: Any,
    ) -> ClaudeAgentResponse | None:
        """Synchronous wrapper around async _query_single."""
        return asyncio.run(
            self._query_single(
                system_prompt=system_prompt,
                messages=messages,
                tools=tools,
                max_new_tokens=max_new_tokens,
                **generation_kwargs,
            )
        )

    def sample(
        self,
        system_prompt: str | None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_new_tokens: int | None = None,
        num_return_sequences: int = 1,
        **generation_kwargs: Any,
    ) -> list[ClaudeAgentResponse | None]:
        """Generate response(s) from Claude via query()."""
        return [
            self._sample_single(
                system_prompt=system_prompt,
                messages=messages or [],
                tools=tools or [],
                max_new_tokens=max_new_tokens,
                **generation_kwargs,
            )
            for _ in range(num_return_sequences)
        ]

    def update_messages(
        self,
        messages: list[dict[str, Any]],
        model_response: ClaudeAgentResponse | None,
        prior_messages: list[dict[str, Any]],
        interleave: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Return updated messages for the model.

        Converts ClaudeAgentResponse content blocks back into message dicts
        that can be passed in the next call's messages list.
        """
        if interleave or model_response is None:
            new_messages: list[dict[str, Any]] = []
        else:
            # Convert assistant content blocks to message dicts
            new_messages = _response_to_message_dicts(model_response)

        for message in messages:
            if message.get("type") == "function_call_output":
                msg = {k: v for k, v in message.items() if k != "role"}
                try:
                    image_message = msg.pop("image_output")
                    new_messages.extend([msg, image_message])
                except KeyError:
                    new_messages.append(msg)
            else:
                new_messages.append(message)

        return prior_messages + new_messages

    def get_actions(
        self, response: ClaudeAgentResponse | None
    ) -> list[ActionFromLLM]:
        """Extract actions from a ClaudeAgentResponse."""
        return _extract_actions(response)


class ClaudeClientLLM(LLM):
    """
    Claude Agent SDK LLM using `ClaudeSDKClient` — persistent, multi-turn sessions.

    Maintains a conversation session across multiple `sample()` calls. The client
    remembers previous context, making it suitable for multi-step agent interactions
    where you want Claude to maintain state.

    Example usage:
    ```
    llm = ClaudeClientLLM(model="claude-sonnet-4-6")
    llm.connect()  # Start session

    responses = llm.sample(
        system_prompt="You are a helpful agent.",
        messages=[{"role": "user", "content": "Hello"}],
        tools=[{"type": "function", "name": "search", ...}],
    )
    actions = llm.get_actions(responses[0])

    # Later calls reuse the same session
    responses = llm.sample(
        system_prompt="You are a helpful agent.",
        messages=[{"role": "user", "content": "Follow up"}],
    )

    llm.disconnect()  # End session
    ```
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        generation_config: Any | None = None,
        max_turns: int | None = 1,
        permission_mode: str = "bypassPermissions",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, generation_config=generation_config, **kwargs)
        self.max_turns = max_turns
        self.permission_mode = permission_mode
        self.last_usage = None
        self._client: ClaudeSDKClient | None = None
        self._connected = False

    def _get_options(
        self,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions from current config."""
        mcp_servers, allowed_tools = _convert_tools_to_mcp_server(tools or [])
        return ClaudeAgentOptions(
            model=self.model,
            system_prompt=system_prompt,
            max_turns=self.max_turns,
            permission_mode=self.permission_mode,
            mcp_servers=mcp_servers,
            allowed_tools=allowed_tools,
        )

    def connect(
        self,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        """Connect the client and start a session."""
        options = self._get_options(system_prompt, tools)
        self._client = ClaudeSDKClient(options=options)
        asyncio.run(self._client.connect())
        self._connected = True

    def disconnect(self) -> None:
        """Disconnect the client and end the session."""
        if self._client is not None:
            asyncio.run(self._client.disconnect())
            self._client = None
            self._connected = False

    async def _query_single_async(
        self,
        prompt: str,
    ) -> ClaudeAgentResponse | None:
        """Send a query and collect the response."""
        if self._client is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        await self._client.query(prompt)

        response = ClaudeAgentResponse()
        try:
            async for message in self._client.receive_response():
                if isinstance(message, AssistantMessage):
                    response.assistant_messages.append(message)
                elif isinstance(message, ResultMessage):
                    response.result = message
                    if message.usage:
                        response.usage = {
                            "input_tokens": message.usage.get("input_tokens", 0),
                            "output_tokens": message.usage.get("output_tokens", 0),
                        }
        except Exception as e:
            print(f"ClaudeClientLLM query error: {type(e).__name__}: {e}")
            return None

        # Track tokens
        if response.usage:
            self.last_usage = self._track_tokens(
                type("Usage", (), response.usage)()
            )
            if self.last_usage is not None:
                print("Claude Agent SDK token usage:")
                for k, v in self.last_usage.items():
                    print(f"-> {k}: {v}")

        return response

    def _sample_single(
        self,
        system_prompt: str | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_new_tokens: int | None = None,
        **generation_kwargs: Any,
    ) -> ClaudeAgentResponse | None:
        """Synchronous wrapper around async query."""
        # Auto-connect if not yet connected
        if not self._connected:
            self.connect(system_prompt=system_prompt, tools=tools)

        prompt = _messages_to_prompt(None, messages)  # system_prompt set at connect
        return asyncio.run(self._query_single_async(prompt))

    def sample(
        self,
        system_prompt: str | None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_new_tokens: int | None = None,
        num_return_sequences: int = 1,
        **generation_kwargs: Any,
    ) -> list[ClaudeAgentResponse | None]:
        """Generate response(s) from Claude via ClaudeSDKClient."""
        return [
            self._sample_single(
                system_prompt=system_prompt,
                messages=messages or [],
                tools=tools or [],
                max_new_tokens=max_new_tokens,
                **generation_kwargs,
            )
            for _ in range(num_return_sequences)
        ]

    def update_messages(
        self,
        messages: list[dict[str, Any]],
        model_response: ClaudeAgentResponse | None,
        prior_messages: list[dict[str, Any]],
        interleave: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Return updated messages for the model.

        For ClaudeClientLLM, the session maintains its own context, so this
        primarily formats new messages for the next prompt.
        """
        if interleave or model_response is None:
            new_messages: list[dict[str, Any]] = []
        else:
            new_messages = _response_to_message_dicts(model_response)

        for message in messages:
            if message.get("type") == "function_call_output":
                msg = {k: v for k, v in message.items() if k != "role"}
                try:
                    image_message = msg.pop("image_output")
                    new_messages.extend([msg, image_message])
                except KeyError:
                    new_messages.append(msg)
            else:
                new_messages.append(message)

        return prior_messages + new_messages

    def get_actions(
        self, response: ClaudeAgentResponse | None
    ) -> list[ActionFromLLM]:
        """Extract actions from a ClaudeAgentResponse."""
        return _extract_actions(response)

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        if self._connected:
            try:
                self.disconnect()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _is_tool_use_block(block: Any) -> bool:
    """Check if block is a ToolUseBlock (by structure, not isinstance)."""
    return hasattr(block, "name") and hasattr(block, "input") and hasattr(block, "id")


def _is_text_block(block: Any) -> bool:
    """Check if block is a TextBlock (by structure, not isinstance)."""
    return hasattr(block, "text") and not hasattr(block, "thinking")


def _is_thinking_block(block: Any) -> bool:
    """Check if block is a ThinkingBlock (by structure, not isinstance)."""
    return hasattr(block, "thinking")


def _is_tool_result_block(block: Any) -> bool:
    """Check if block is a ToolResultBlock (by structure, not isinstance)."""
    return hasattr(block, "tool_use_id") and not hasattr(block, "name")


def _extract_actions(response: ClaudeAgentResponse | None) -> list[ActionFromLLM]:
    """
    Extract ActionFromLLM items from a ClaudeAgentResponse.

    Maps Claude Agent SDK content blocks to ActionFromLLM:
    - ToolUseBlock -> type="function_call"
    - TextBlock -> type="message"
    - ThinkingBlock -> type="reasoning"
    """
    actions = []
    if response is None:
        return actions

    for block in response.output:
        if _is_tool_use_block(block):
            text_repr = json.dumps({"name": block.name, "arguments": block.input})
            text_repr = f"<tool_call>\n{text_repr}\n</tool_call>"
            actions.append(
                ActionFromLLM(
                    role="assistant",
                    type="function_call",
                    text=text_repr,
                    call_id=block.id,
                    name=block.name,
                    arguments=block.input,
                )
            )
        elif _is_thinking_block(block):
            actions.append(
                ActionFromLLM(
                    role="assistant",
                    type="reasoning",
                    text=block.thinking,
                    call_id=None,
                    name=None,
                    arguments=None,
                )
            )
        elif _is_text_block(block):
            actions.append(
                ActionFromLLM(
                    role="assistant",
                    type="message",
                    text=block.text,
                    call_id=None,
                    name=None,
                    arguments=None,
                )
            )
        elif _is_tool_result_block(block):
            # Tool results are environment responses, not model actions.
            pass
        else:
            print(f"Warning: unknown content block type: {type(block)}")

    return actions


def _response_to_message_dicts(
    response: ClaudeAgentResponse,
) -> list[dict[str, Any]]:
    """
    Convert a ClaudeAgentResponse's content blocks into message dicts
    suitable for inclusion in a messages list.
    """
    message_dicts = []
    for block in response.output:
        if _is_tool_use_block(block):
            message_dicts.append(
                {
                    "role": "assistant",
                    "type": "function_call",
                    "call_id": block.id,
                    "name": block.name,
                    "arguments": json.dumps(block.input),
                }
            )
        elif _is_thinking_block(block):
            message_dicts.append(
                {
                    "role": "assistant",
                    "type": "reasoning",
                    "content": block.thinking,
                }
            )
        elif _is_text_block(block):
            message_dicts.append(
                {
                    "role": "assistant",
                    "content": block.text,
                }
            )
    return message_dicts
