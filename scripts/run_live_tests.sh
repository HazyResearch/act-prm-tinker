#!/usr/bin/env bash
# Run live integration tests for the Claude Agent SDK LLM handlers.
#
# Usage:
#   ./scripts/run_live_tests.sh           # run all live tests
#   ./scripts/run_live_tests.sh grader    # run only grader tests
#   ./scripts/run_live_tests.sh quick     # run only the simple text response test
#
# Prerequisites:
#   - Valid Claude credentials (ANTHROPIC_API_KEY or claude CLI auth)
#   - uv installed

set -euo pipefail

# Unset CLAUDECODE to allow launching Claude Code subprocess from agent SDK
unset CLAUDECODE 2>/dev/null || true

cd "$(git rev-parse --show-toplevel)"

case "${1:-all}" in
  quick)
    echo "==> Running quick live test (simple text response)..."
    uv run python -m pytest tests/test_claude_agent_sdk_handler.py::TestClaudeQueryLLMLive::test_simple_text_response -v -s
    ;;
  grader)
    echo "==> Running LLM-as-a-judge grader tests..."
    uv run python -m pytest tests/test_claude_agent_sdk_handler.py::TestLLMGraderWithClaudeQuery -v -s
    ;;
  all)
    echo "==> Running all live integration tests..."
    uv run python -m pytest tests/test_claude_agent_sdk_handler.py -m live -v -s
    ;;
  *)
    echo "Usage: $0 [quick|grader|all]"
    exit 1
    ;;
esac

echo "==> Done!"
