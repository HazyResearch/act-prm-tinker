"""
Tests for the Claude Agent SDK LLM handlers (ClaudeQueryLLM, ClaudeClientLLM).

Unit tests use mock objects; live integration tests (marked with @pytest.mark.live)
call the real Claude Agent SDK and require valid credentials.

Run unit tests only:  pytest tests/test_claude_agent_sdk_handler.py -m "not live"
Run live tests only:  pytest tests/test_claude_agent_sdk_handler.py -m live -v
Run all tests:        pytest tests/test_claude_agent_sdk_handler.py -v
"""

import json
from dataclasses import dataclass
from typing import Any

import pytest

from act_prm.llm_handlers.claude_agent_sdk import (
    ClaudeAgentResponse,
    ClaudeClientLLM,
    ClaudeQueryLLM,
    _convert_tools_to_mcp_server,
    _extract_actions,
    _messages_to_prompt,
    _response_to_message_dicts,
)


@pytest.fixture(autouse=True, scope="session")
def _load_env():
    """Load .env file for credentials used by live tests."""
    from dotenv import load_dotenv

    load_dotenv()


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
                        MockToolResultBlock(tool_use_id="call_123", content="result")
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
                        MockToolUseBlock(id="c1", name="search", input={"q": "test"}),
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
                        MockToolUseBlock(id="c1", name="search", input={"q": "test"})
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
                MockAssistantMessage(content=[MockThinkingBlock(thinking="reasoning")])
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
                    content=[MockToolUseBlock(id="c1", name="search", input={"q": "x"})]
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


# ---------------------------------------------------------------------------
# Live integration tests — require valid Claude credentials
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestClaudeQueryLLMLive:
    """Live tests for ClaudeQueryLLM using real Claude Agent SDK calls."""

    def test_simple_text_response(self):
        """query() should return a text response for a simple prompt."""
        llm = ClaudeQueryLLM(model="claude-sonnet-4-6", max_turns=1)
        responses = llm.sample(
            system_prompt="You are a helpful assistant. Respond in one short sentence.",
            messages=[{"role": "user", "content": "What is 2 + 2?"}],
            tools=[],
            num_return_sequences=1,
        )
        assert len(responses) == 1
        response = responses[0]
        assert response is not None, "Response should not be None"
        assert len(response.assistant_messages) > 0, "Should have assistant messages"
        assert response.result is not None, "Should have a ResultMessage"

        # Extract actions and verify we got a text message
        actions = llm.get_actions(response)
        assert len(actions) > 0, "Should have at least one action"
        text_actions = [a for a in actions if a.type == "message"]
        assert len(text_actions) > 0, "Should have at least one text message"
        # The response should mention "4"
        combined_text = " ".join(a.text for a in text_actions if a.text)
        assert "4" in combined_text, f"Expected '4' in response, got: {combined_text}"

    def test_token_tracking(self):
        """Token usage should be tracked after a query."""
        llm = ClaudeQueryLLM(model="claude-sonnet-4-6", max_turns=1)
        responses = llm.sample(
            system_prompt="Respond with exactly one word.",
            messages=[{"role": "user", "content": "Say hello."}],
            tools=[],
        )
        response = responses[0]
        assert response is not None
        # Check that tokens were tracked
        assert llm.prompt_tokens > 0 and llm.completion_tokens > 0, (
            f"Both token counts should be > 0, got prompt={llm.prompt_tokens}, "
            f"completion={llm.completion_tokens}"
        )

    def test_get_actions_from_live_response(self):
        """get_actions() should correctly parse a live response."""
        llm = ClaudeQueryLLM(model="claude-sonnet-4-6", max_turns=1)
        responses = llm.sample(
            system_prompt="You are a calculator. Only respond with the numeric answer.",
            messages=[{"role": "user", "content": "What is 7 * 8?"}],
            tools=[],
        )
        response = responses[0]
        assert response is not None
        actions = llm.get_actions(response)
        assert len(actions) > 0
        # Should have a message action with "56"
        text_actions = [a for a in actions if a.type == "message"]
        assert len(text_actions) > 0
        combined = " ".join(a.text for a in text_actions if a.text)
        assert "56" in combined, f"Expected '56' in response, got: {combined}"

    def test_update_messages_roundtrip(self):
        """update_messages() should integrate live response into message history."""
        llm = ClaudeQueryLLM(model="claude-sonnet-4-6", max_turns=1)
        responses = llm.sample(
            system_prompt="Respond briefly.",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
        )
        response = responses[0]
        assert response is not None

        # Use update_messages to build a conversation history
        updated = llm.update_messages(
            messages=[{"role": "user", "content": "Follow up question"}],
            model_response=response,
            prior_messages=[{"role": "user", "content": "Hi"}],
        )
        # Should have: prior user msg + response content + new user msg
        assert len(updated) >= 3, f"Expected >= 3 messages, got {len(updated)}"
        assert updated[0]["content"] == "Hi"
        assert updated[-1]["content"] == "Follow up question"

    def test_multiple_return_sequences(self):
        """num_return_sequences > 1 should return multiple independent responses."""
        llm = ClaudeQueryLLM(model="claude-sonnet-4-6", max_turns=1)
        responses = llm.sample(
            system_prompt="Respond with one word.",
            messages=[{"role": "user", "content": "Say any color."}],
            tools=[],
            num_return_sequences=2,
        )
        assert len(responses) == 2
        for r in responses:
            assert r is not None
            actions = llm.get_actions(r)
            assert len(actions) > 0


@pytest.mark.live
class TestClaudeQueryLLMLiveWithTools:
    """Live tests for ClaudeQueryLLM with tool definitions."""

    def test_tool_call_response(self):
        """Claude should produce a tool call when given tools and an appropriate prompt."""
        tools = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name",
                        }
                    },
                    "required": ["city"],
                },
            }
        ]
        llm = ClaudeQueryLLM(model="claude-sonnet-4-6", max_turns=1)
        responses = llm.sample(
            system_prompt=(
                "You are a weather assistant. Use the get_weather tool to answer questions. "
                "Always use the tool, never answer directly."
            ),
            messages=[
                {"role": "user", "content": "What's the weather in San Francisco?"}
            ],
            tools=tools,
        )
        response = responses[0]
        assert response is not None, "Response should not be None"

        actions = llm.get_actions(response)
        # Should have at least one function_call action
        tool_actions = [a for a in actions if a.type == "function_call"]
        assert len(tool_actions) > 0, (
            f"Expected tool call, got actions: {[(a.type, a.name) for a in actions]}"
        )

        tool_action = tool_actions[0]
        assert tool_action.name == "get_weather"
        assert "city" in tool_action.arguments
        assert tool_action.call_id is not None


@pytest.mark.live
class TestLoadLLMLive:
    """Test load_llm() factory with live instantiation."""

    def test_load_and_sample_claude_query(self):
        """load_llm('claude_query') should produce a working ClaudeQueryLLM."""
        from act_prm.llm_handlers import load_llm

        llm = load_llm("claude_query", model_config={"model": "claude-sonnet-4-6"})
        assert isinstance(llm, ClaudeQueryLLM)

        responses = llm.sample(
            system_prompt="Respond with one word only.",
            messages=[{"role": "user", "content": "Say yes."}],
            tools=[],
        )
        assert responses[0] is not None
        actions = llm.get_actions(responses[0])
        assert len(actions) > 0


# ---------------------------------------------------------------------------
# Live LLM-as-a-judge tests — ClaudeQueryLLM as grader via LLMGraderForQA
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestLLMGraderWithClaudeQuery:
    """Test LLMGraderForQA using ClaudeQueryLLM as the grader model."""

    def _make_grader(self):
        from act_prm.graders.qa import LLMGraderForQA

        llm = ClaudeQueryLLM(model="claude-sonnet-4-6", max_turns=1)
        return LLMGraderForQA(grader_model=llm, num_samples=1, verbose=True)

    def test_correct_answer(self):
        """Grader should mark a correct answer as correct."""
        grader = self._make_grader()
        is_correct, msg = grader(
            question="What is the capital of France?",
            correct_answer="Paris",
            response="The capital of France is Paris.",
        )
        assert is_correct, f"Expected correct, grader said: {msg}"

    def test_incorrect_answer(self):
        """Grader should mark an incorrect answer as incorrect."""
        grader = self._make_grader()
        is_correct, msg = grader(
            question="What is the capital of France?",
            correct_answer="Paris",
            response="The capital of France is Berlin.",
        )
        assert not is_correct, f"Expected incorrect, grader said: {msg}"

    def test_numeric_answer(self):
        """Grader should handle numeric answers correctly."""
        grader = self._make_grader()
        is_correct, msg = grader(
            question="What is 15 * 23?",
            correct_answer="345",
            response="15 multiplied by 23 equals 345.",
        )
        assert is_correct, f"Expected correct, grader said: {msg}"

    def test_numeric_wrong_answer(self):
        """Grader should reject wrong numeric answers."""
        grader = self._make_grader()
        is_correct, msg = grader(
            question="What is 15 * 23?",
            correct_answer="345",
            response="15 multiplied by 23 equals 350.",
        )
        assert not is_correct, f"Expected incorrect, grader said: {msg}"

    def test_metrics_tracking(self):
        """Grader should track running metrics across calls."""
        grader = self._make_grader()

        grader(
            question="What is 2+2?",
            correct_answer="4",
            response="4",
            sample_id=0,
            split="test",
        )
        grader(
            question="What is 3+3?",
            correct_answer="6",
            response="The answer is 7.",
            sample_id=1,
            split="test",
        )

        assert len(grader.metrics["correct"]) == 2
        assert "test/acc" in grader.running_metrics
        assert "test/total" in grader.running_metrics
        assert grader.running_metrics["test/total"] == 2
        # First should be correct (4==4), second incorrect (7!=6)
        assert grader.running_metrics["test/correct"] == 1, (
            f"Expected 1 correct, got {grader.running_metrics['test/correct']}"
        )

    def test_grader_via_load_llm(self):
        """LLMGraderForQA should work when instantiated via grader_model_config."""
        from act_prm.graders.qa import LLMGraderForQA

        grader = LLMGraderForQA(
            grader_model_config={
                "name": "claude_query",
                "model_config": {"model": "claude-sonnet-4-6"},
            },
            num_samples=1,
            verbose=True,
        )
        is_correct, msg = grader(
            question="What color is the sky on a clear day?",
            correct_answer="Blue",
            response="The sky is blue.",
        )
        assert is_correct, f"Expected correct, grader said: {msg}"


@pytest.mark.live
class TestClaudeClientLLMLive:
    """Live tests for ClaudeClientLLM (persistent session)."""

    def test_connect_sample_disconnect(self):
        """Basic lifecycle: connect, sample, disconnect."""
        llm = ClaudeClientLLM(model="claude-sonnet-4-6", max_turns=1)
        llm.connect(system_prompt="Respond with one word only.")

        try:
            responses = llm.sample(
                system_prompt=None,
                messages=[{"role": "user", "content": "Say hello."}],
                tools=[],
            )
            assert len(responses) == 1
            response = responses[0]
            assert response is not None, "Response should not be None"

            actions = llm.get_actions(response)
            assert len(actions) > 0, "Should have at least one action"
            text_actions = [a for a in actions if a.type == "message"]
            assert len(text_actions) > 0, "Should have at least one text message"
        finally:
            llm.disconnect()


# ---------------------------------------------------------------------------
# Snorkel Finance grading integration test
# ---------------------------------------------------------------------------


def _extract_final_response(action_content: str) -> str | None:
    """Extract the agent's final response text from the last action content.

    Handles two patterns:
    1. respond_user tool call: extracts the text argument
    2. Plain text ending with "Final Answer: ..."
    """
    # Try to parse as a tool call with respond_user
    if "<tool_call>" in action_content:
        try:
            tc_start = action_content.index("<tool_call>") + len("<tool_call>")
            tc_end = action_content.index("</tool_call>")
            tc_json = json.loads(action_content[tc_start:tc_end].strip())
            if tc_json.get("name") == "respond_user":
                return tc_json.get("arguments", {}).get("text", "")
        except (ValueError, json.JSONDecodeError, KeyError):
            pass

    # Try plain text with "Final Answer:"
    if "Final Answer:" in action_content:
        return action_content.split("Final Answer:")[-1].strip()

    # Fall back to returning the whole content (may be the direct response)
    return action_content.strip() if action_content.strip() else None


@pytest.mark.live
class TestSnorkelFinanceGradingWithClaude:
    """Integration test: grade Snorkel Finance traces using SnorkelFinanceGrader
    with ClaudeQueryLLM as the grader model.

    Loads completed (done=True) episodes from the aligned HF dataset and verifies
    that the grader produces sensible results.
    """

    NUM_SAMPLES = 5  # grade this many completed episodes

    @staticmethod
    def _load_done_samples():
        """Load completed episodes from the aligned dataset."""
        from datasets import load_dataset

        ds = load_dataset(
            "mzio/aprm-snorkelai_agent_finance_reasoning-aligned",
            split="train",
        )
        # Filter for done episodes with reward=1 (known-correct trajectories)
        done_correct = [s for s in ds if s["done"] and s["reward"] == 1]
        assert len(done_correct) > 0, "No done+correct samples found in dataset"
        return done_correct

    def test_grade_correct_traces(self):
        """Grading known-correct completed traces should yield mostly correct."""
        from act_prm.graders.snorkel_finance import SnorkelFinanceGrader

        samples = self._load_done_samples()
        grader = SnorkelFinanceGrader(
            grader_model=ClaudeQueryLLM(model="claude-sonnet-4-6", max_turns=1),
            num_samples=1,
            verbose=True,
        )

        results = []
        for i, sample in enumerate(samples[: self.NUM_SAMPLES]):
            # Extract question from first user message in state
            question = None
            for msg in sample["state"]:
                if msg["role"] == "user":
                    question = msg["content"]
                    break
            assert question is not None, f"No user message in sample {i}"

            correct_answer = sample["answer"]

            # The final action content is the agent's response
            action_content = sample["action"]["content_0"]
            response_text = _extract_final_response(action_content)
            assert response_text is not None, (
                f"Could not extract response from sample {i}"
            )

            is_correct, msg = grader(
                question=question,
                correct_answer=correct_answer,
                response=response_text,
                sample_id=sample["unique_data_sample_id"],
                generation_id=sample.get("generation_id", 0),
                split="test",
            )
            results.append(is_correct)
            print(
                f"Sample {i} (id={sample['unique_data_sample_id']}): "
                f"graded={'correct' if is_correct else 'incorrect'}, "
                f"answer={correct_answer!r}"
            )

        n_correct = sum(results)
        print(f"\nGrading results: {n_correct}/{len(results)} correct")
        # These are reward=1 traces, so the grader should agree most of the time
        assert n_correct >= len(results) // 2, (
            f"Expected at least half correct for reward=1 traces, "
            f"got {n_correct}/{len(results)}"
        )

    def test_grade_incorrect_traces(self):
        """Grading known-incorrect traces (reward=0, done=True) should yield mostly incorrect."""
        from datasets import load_dataset

        from act_prm.graders.snorkel_finance import SnorkelFinanceGrader

        ds = load_dataset(
            "mzio/aprm-snorkelai_agent_finance_reasoning-aligned",
            split="train",
        )
        done_incorrect = [s for s in ds if s["done"] and s["reward"] == 0]
        if len(done_incorrect) == 0:
            pytest.skip("No done+incorrect samples in dataset")

        grader = SnorkelFinanceGrader(
            grader_model=ClaudeQueryLLM(model="claude-sonnet-4-6", max_turns=1),
            num_samples=1,
            verbose=True,
        )

        results = []
        for i, sample in enumerate(done_incorrect[: self.NUM_SAMPLES]):
            question = None
            for msg in sample["state"]:
                if msg["role"] == "user":
                    question = msg["content"]
                    break
            if question is None:
                continue

            correct_answer = sample["answer"]
            action_content = sample["action"]["content_0"]
            response_text = _extract_final_response(action_content)
            if response_text is None:
                continue

            is_correct, msg = grader(
                question=question,
                correct_answer=correct_answer,
                response=response_text,
                sample_id=sample["unique_data_sample_id"],
                generation_id=sample.get("generation_id", 0),
                split="test",
            )
            results.append(is_correct)
            print(
                f"Sample {i} (id={sample['unique_data_sample_id']}): "
                f"graded={'correct' if is_correct else 'incorrect'}, "
                f"answer={correct_answer!r}"
            )

        if not results:
            pytest.skip("Could not extract responses from incorrect traces")

        n_incorrect = sum(not r for r in results)
        print(
            f"\nGrading results: {n_incorrect}/{len(results)} correctly identified as wrong"
        )
        assert n_incorrect >= len(results) // 2, (
            f"Expected at least half graded incorrect for reward=0 traces, "
            f"got {n_incorrect}/{len(results)}"
        )
