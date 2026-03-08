"""
Tests for the Claude Agent SDK LLM handlers (ClaudeQueryLLM, ClaudeClientLLM).

Uses mocks to avoid real Claude Agent SDK calls.
"""

import json
from dataclasses import dataclass
from typing import Any


from act_prm.llm_handlers.claude_agent_sdk import (
    ClaudeAgentResponse,
    ClaudeClientLLM,
    ClaudeQueryLLM,
    _convert_tools_to_mcp_server,
    _extract_actions,
    _messages_to_prompt,
    _response_to_message_dicts,
)


# ---------------------------------------------------------------------------
# Mock SDK types — mirror the real dataclass shapes
# ---------------------------------------------------------------------------


@dataclass
class MockTextBlock:
    text: str


@dataclass
class MockToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class MockThinkingBlock:
    thinking: str
    signature: str = ""


@dataclass
class MockToolResultBlock:
    tool_use_id: str
    content: str | None = None
    is_error: bool | None = None


@dataclass
class MockAssistantMessage:
    content: list[Any]
    model: str = "claude-sonnet-4-6"
    parent_tool_use_id: str | None = None
    error: Any = None


@dataclass
class MockResultMessage:
    subtype: str = "result"
    duration_ms: int = 100
    duration_api_ms: int = 80
    is_error: bool = False
    num_turns: int = 1
    session_id: str = "test-session"
    total_cost_usd: float | None = 0.01
    usage: dict[str, Any] | None = None
    result: str | None = None
    stop_reason: str | None = None
    structured_output: Any = None


# ---------------------------------------------------------------------------
# Tests for _messages_to_prompt
# ---------------------------------------------------------------------------


class TestMessagesToPrompt:
    def test_empty_messages(self):
        result = _messages_to_prompt(None, [])
        assert result == ""

    def test_system_prompt_only(self):
        result = _messages_to_prompt("Be helpful.", [])
        assert result == "[System]\nBe helpful."

    def test_user_message(self):
        result = _messages_to_prompt(None, [{"role": "user", "content": "Hello"}])
        assert "[User]\nHello" in result

    def test_assistant_message(self):
        result = _messages_to_prompt(
            None, [{"role": "assistant", "content": "Hi there"}]
        )
        assert "[Assistant]\nHi there" in result

    def test_function_call_output(self):
        result = _messages_to_prompt(
            None,
            [
                {
                    "type": "function_call_output",
                    "call_id": "abc",
                    "content": "result data",
                }
            ],
        )
        assert "[Tool Result (abc)]" in result
        assert "result data" in result

    def test_full_conversation(self):
        messages = [
            {"role": "user", "content": "Search for X"},
            {"role": "assistant", "content": "Let me search."},
            {
                "type": "function_call_output",
                "call_id": "c1",
                "content": "Found X",
            },
        ]
        result = _messages_to_prompt("You are an agent.", messages)
        assert "[System]\nYou are an agent." in result
        assert "[User]\nSearch for X" in result
        assert "[Assistant]\nLet me search." in result
        assert "[Tool Result (c1)]" in result


# ---------------------------------------------------------------------------
# Tests for _convert_tools_to_mcp_server
# ---------------------------------------------------------------------------


class TestConvertTools:
    def test_empty_tools(self):
        servers, names = _convert_tools_to_mcp_server([])
        assert servers == {}
        assert names == []

    def test_single_tool(self):
        tools = [
            {
                "type": "function",
                "name": "search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]
        servers, names = _convert_tools_to_mcp_server(tools)
        assert "env_tools" in servers
        assert "mcp__env_tools__search" in names

    def test_multiple_tools(self):
        tools = [
            {"name": "tool_a", "description": "A", "parameters": {}},
            {"name": "tool_b", "description": "B", "parameters": {}},
        ]
        servers, names = _convert_tools_to_mcp_server(tools)
        assert len(names) == 2
        assert "mcp__env_tools__tool_a" in names
        assert "mcp__env_tools__tool_b" in names

    def test_nested_function_format(self):
        """Test tools with nested function key (some APIs use this)."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up data",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        servers, names = _convert_tools_to_mcp_server(tools)
        assert "mcp__env_tools__lookup" in names


# ---------------------------------------------------------------------------
# Tests for _extract_actions
# ---------------------------------------------------------------------------


class TestExtractActions:
    def test_none_response(self):
        assert _extract_actions(None) == []

    def test_empty_response(self):
        response = ClaudeAgentResponse()
        assert _extract_actions(response) == []

    def test_text_block(self):
        response = ClaudeAgentResponse(
            assistant_messages=[
                MockAssistantMessage(content=[MockTextBlock(text="Hello world")])
            ]
        )
        actions = _extract_actions(response)
        assert len(actions) == 1
        assert actions[0].type == "message"
        assert actions[0].text == "Hello world"
        assert actions[0].role == "assistant"

    def test_tool_use_block(self):
        response = ClaudeAgentResponse(
            assistant_messages=[
                MockAssistantMessage(
                    content=[
                        MockToolUseBlock(
                            id="call_123",
                            name="search",
                            input={"query": "test"},
                        )
                    ]
                )
            ]
        )
        actions = _extract_actions(response)
        assert len(actions) == 1
        assert actions[0].type == "function_call"
        assert actions[0].name == "search"
        assert actions[0].arguments == {"query": "test"}
        assert actions[0].call_id == "call_123"
        assert "<tool_call>" in actions[0].text

    def test_thinking_block(self):
        response = ClaudeAgentResponse(
            assistant_messages=[
                MockAssistantMessage(
                    content=[MockThinkingBlock(thinking="Let me think...")]
                )
            ]
        )
        actions = _extract_actions(response)
        assert len(actions) == 1
        assert actions[0].type == "reasoning"
        assert actions[0].text == "Let me think..."

    def test_tool_result_block_skipped(self):
        response = ClaudeAgentResponse(
            assistant_messages=[
                MockAssistantMessage(
                    content=[
                        MockToolResultBlock(
                            tool_use_id="call_123", content="result"
                        )
                    ]
                )
            ]
        )
        actions = _extract_actions(response)
        assert len(actions) == 0

    def test_mixed_blocks(self):
        response = ClaudeAgentResponse(
            assistant_messages=[
                MockAssistantMessage(
                    content=[
                        MockThinkingBlock(thinking="Hmm..."),
                        MockTextBlock(text="I'll search for that."),
                        MockToolUseBlock(
                            id="c1", name="search", input={"q": "test"}
                        ),
                    ]
                )
            ]
        )
        actions = _extract_actions(response)
        assert len(actions) == 3
        assert actions[0].type == "reasoning"
        assert actions[1].type == "message"
        assert actions[2].type == "function_call"

    def test_multiple_assistant_messages(self):
        response = ClaudeAgentResponse(
            assistant_messages=[
                MockAssistantMessage(content=[MockTextBlock(text="First")]),
                MockAssistantMessage(content=[MockTextBlock(text="Second")]),
            ]
        )
        actions = _extract_actions(response)
        assert len(actions) == 2
        assert actions[0].text == "First"
        assert actions[1].text == "Second"


# ---------------------------------------------------------------------------
# Tests for _response_to_message_dicts
# ---------------------------------------------------------------------------


class TestResponseToMessageDicts:
    def test_text_block(self):
        response = ClaudeAgentResponse(
            assistant_messages=[
                MockAssistantMessage(content=[MockTextBlock(text="Hi")])
            ]
        )
        dicts = _response_to_message_dicts(response)
        assert len(dicts) == 1
        assert dicts[0]["role"] == "assistant"
        assert dicts[0]["content"] == "Hi"

    def test_tool_use_block(self):
        response = ClaudeAgentResponse(
            assistant_messages=[
                MockAssistantMessage(
                    content=[
                        MockToolUseBlock(
                            id="c1", name="search", input={"q": "test"}
                        )
                    ]
                )
            ]
        )
        dicts = _response_to_message_dicts(response)
        assert len(dicts) == 1
        assert dicts[0]["type"] == "function_call"
        assert dicts[0]["call_id"] == "c1"
        assert dicts[0]["name"] == "search"

    def test_thinking_block(self):
        response = ClaudeAgentResponse(
            assistant_messages=[
                MockAssistantMessage(
                    content=[MockThinkingBlock(thinking="reasoning")]
                )
            ]
        )
        dicts = _response_to_message_dicts(response)
        assert len(dicts) == 1
        assert dicts[0]["type"] == "reasoning"


# ---------------------------------------------------------------------------
# Tests for ClaudeAgentResponse
# ---------------------------------------------------------------------------


class TestClaudeAgentResponse:
    def test_output_flattens_blocks(self):
        r = ClaudeAgentResponse(
            assistant_messages=[
                MockAssistantMessage(
                    content=[MockTextBlock(text="A"), MockTextBlock(text="B")]
                ),
                MockAssistantMessage(content=[MockTextBlock(text="C")]),
            ]
        )
        assert len(r.output) == 3

    def test_empty_output(self):
        r = ClaudeAgentResponse()
        assert r.output == []


# ---------------------------------------------------------------------------
# Tests for ClaudeQueryLLM
# ---------------------------------------------------------------------------


class TestClaudeQueryLLM:
    def test_init_defaults(self):
        llm = ClaudeQueryLLM()
        assert llm.model == "claude-sonnet-4-6"
        assert llm.max_turns == 1
        assert llm.permission_mode == "bypassPermissions"

    def test_init_custom(self):
        llm = ClaudeQueryLLM(
            model="claude-opus-4-6", max_turns=5, permission_mode="default"
        )
        assert llm.model == "claude-opus-4-6"
        assert llm.max_turns == 5

    def test_get_actions_delegates(self):
        llm = ClaudeQueryLLM()
        response = ClaudeAgentResponse(
            assistant_messages=[
                MockAssistantMessage(content=[MockTextBlock(text="test")])
            ]
        )
        actions = llm.get_actions(response)
        assert len(actions) == 1
        assert actions[0].text == "test"

    def test_get_actions_none(self):
        llm = ClaudeQueryLLM()
        assert llm.get_actions(None) == []

    def test_update_messages_no_response(self):
        llm = ClaudeQueryLLM()
        result = llm.update_messages(
            messages=[{"role": "user", "content": "hi"}],
            model_response=None,
            prior_messages=[],
        )
        assert len(result) == 1
        assert result[0]["content"] == "hi"

    def test_update_messages_with_response(self):
        llm = ClaudeQueryLLM()
        response = ClaudeAgentResponse(
            assistant_messages=[
                MockAssistantMessage(content=[MockTextBlock(text="reply")])
            ]
        )
        result = llm.update_messages(
            messages=[{"role": "user", "content": "follow up"}],
            model_response=response,
            prior_messages=[{"role": "user", "content": "first"}],
        )
        # prior + response dict + new message
        assert len(result) == 3
        assert result[0]["content"] == "first"
        assert result[1]["content"] == "reply"
        assert result[2]["content"] == "follow up"

    def test_update_messages_function_call_output(self):
        llm = ClaudeQueryLLM()
        result = llm.update_messages(
            messages=[
                {
                    "role": "assistant",
                    "type": "function_call_output",
                    "call_id": "c1",
                    "content": "result",
                }
            ],
            model_response=None,
            prior_messages=[],
        )
        assert len(result) == 1
        assert "role" not in result[0]  # role stripped for function_call_output
        assert result[0]["call_id"] == "c1"

    def test_load_llm_claude_query(self):
        from act_prm.llm_handlers import load_llm

        llm = load_llm("claude_query", model_config={"model": "claude-sonnet-4-6"})
        assert isinstance(llm, ClaudeQueryLLM)

    def test_load_llm_claude_client(self):
        from act_prm.llm_handlers import load_llm

        llm = load_llm("claude_client", model_config={"model": "claude-sonnet-4-6"})
        assert isinstance(llm, ClaudeClientLLM)


# ---------------------------------------------------------------------------
# Tests for ClaudeClientLLM
# ---------------------------------------------------------------------------


class TestClaudeClientLLM:
    def test_init_defaults(self):
        llm = ClaudeClientLLM()
        assert llm.model == "claude-sonnet-4-6"
        assert not llm._connected
        assert llm._client is None

    def test_get_actions_delegates(self):
        llm = ClaudeClientLLM()
        response = ClaudeAgentResponse(
            assistant_messages=[
                MockAssistantMessage(
                    content=[
                        MockToolUseBlock(
                            id="c1", name="search", input={"q": "x"}
                        )
                    ]
                )
            ]
        )
        actions = llm.get_actions(response)
        assert len(actions) == 1
        assert actions[0].type == "function_call"
        assert actions[0].name == "search"

    def test_update_messages_with_response(self):
        llm = ClaudeClientLLM()
        response = ClaudeAgentResponse(
            assistant_messages=[
                MockAssistantMessage(content=[MockTextBlock(text="ok")])
            ]
        )
        result = llm.update_messages(
            messages=[],
            model_response=response,
            prior_messages=[],
        )
        assert len(result) == 1
        assert result[0]["content"] == "ok"


# ---------------------------------------------------------------------------
# Tests for tool call text repr format
# ---------------------------------------------------------------------------


class TestToolCallTextFormat:
    def test_tool_call_json_format(self):
        """Ensure tool call text matches the format used by OpenAI handler."""
        response = ClaudeAgentResponse(
            assistant_messages=[
                MockAssistantMessage(
                    content=[
                        MockToolUseBlock(
                            id="c1",
                            name="expand",
                            input={"result_id": "8868"},
                        )
                    ]
                )
            ]
        )
        actions = _extract_actions(response)
        expected_inner = json.dumps(
            {"name": "expand", "arguments": {"result_id": "8868"}}
        )
        expected = f"<tool_call>\n{expected_inner}\n</tool_call>"
        assert actions[0].text == expected
