"""
Tests for Snorkel Finance environment.

Unit tests (mocked, no API keys needed):
    pytest tests/test_snorkel_finance.py -v -k "not live"

Live integration tests (requires ANTHROPIC_API_KEY):
    pytest tests/test_snorkel_finance.py -v -m live
"""

import json
import os
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_PATH_ENV = "SNORKEL_FINANCE_DATA_PATH"
DEFAULT_DATA_PATH = "./data/snorkel_finance/raw"
BENCHMARK_REASONING_CSV = "./data/snorkel_finance/benchmark/finqa_reasoning.csv"
BENCHMARK_FINQA_CSV = "./data/snorkel_finance/benchmark/finqa.csv"


@pytest.fixture
def data_path():
    path = os.environ.get(DATA_PATH_ENV, DEFAULT_DATA_PATH)
    if not os.path.isdir(path):
        pytest.skip(f"Data not found at {path}")
    return path


@pytest.fixture
def benchmark_reasoning_csv():
    if not os.path.isfile(BENCHMARK_REASONING_CSV):
        pytest.skip(f"Benchmark CSV not found at {BENCHMARK_REASONING_CSV}")
    return BENCHMARK_REASONING_CSV


@pytest.fixture
def benchmark_finqa_csv():
    if not os.path.isfile(BENCHMARK_FINQA_CSV):
        pytest.skip(f"Benchmark CSV not found at {BENCHMARK_FINQA_CSV}")
    return BENCHMARK_FINQA_CSV


@pytest.fixture
def mock_grader():
    """Patch SnorkelFinanceGrader to avoid needing API keys in unit tests."""
    with patch(
        "act_prm.graders.snorkel_finance.SnorkelFinanceGrader.__init__",
        return_value=None,
    ):
        yield


# ---------------------------------------------------------------------------
# DataBackend tests
# ---------------------------------------------------------------------------


class TestDataBackend:
    """Tests for the DataBackend (real SQL queries on local JSON data)."""

    def test_init(self, data_path):
        from act_prm.environments.snorkel_finance.backend import DataBackend

        backend = DataBackend(data_path)
        companies = backend.list_companies()
        assert len(companies) > 0
        assert "at_t" in companies
        assert "alphabet" in companies

    def test_init_missing_path(self):
        from act_prm.environments.snorkel_finance.backend import DataBackend

        with pytest.raises(FileNotFoundError):
            DataBackend("/nonexistent/path")

    def test_get_descriptions(self, data_path):
        from act_prm.environments.snorkel_finance.backend import DataBackend

        backend = DataBackend(data_path)
        result = backend.get_descriptions("at_t")
        tables = json.loads(result)
        assert isinstance(tables, list)
        assert len(tables) > 0

    def test_get_descriptions_invalid_company(self, data_path):
        from act_prm.environments.snorkel_finance.backend import DataBackend

        backend = DataBackend(data_path)
        result = backend.get_descriptions("nonexistent_company")
        assert "not found" in result.lower()

    def test_get_table_info(self, data_path):
        from act_prm.environments.snorkel_finance.backend import DataBackend

        backend = DataBackend(data_path)
        tables = json.loads(backend.get_descriptions("at_t"))
        info = backend.get_table_info("at_t", tables[0])
        parsed = json.loads(info)
        assert "column_names" in parsed or "column_dtypes" in parsed
        assert "unique_vals_per_col" in parsed

    def test_get_table_info_strips_extension(self, data_path):
        from act_prm.environments.snorkel_finance.backend import DataBackend

        backend = DataBackend(data_path)
        tables = json.loads(backend.get_descriptions("at_t"))
        info = backend.get_table_info("at_t", tables[0] + ".json")
        parsed = json.loads(info)
        assert "unique_vals_per_col" in parsed

    def test_sql_query_with_where(self, data_path):
        from act_prm.environments.snorkel_finance.backend import DataBackend

        backend = DataBackend(data_path)
        tables = json.loads(backend.get_descriptions("at_t"))
        result = backend.sql_query(
            "at_t", tables[0], f'SELECT item FROM "{tables[0]}" WHERE rowid <= 3'
        )
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) > 0

    def test_sql_query_rejects_select_star(self, data_path):
        from act_prm.environments.snorkel_finance.backend import DataBackend

        backend = DataBackend(data_path)
        tables = json.loads(backend.get_descriptions("at_t"))
        result = backend.sql_query("at_t", tables[0], f'SELECT * FROM "{tables[0]}"')
        assert "not allowed" in result.lower()

    def test_sql_query_rejects_no_filter(self, data_path):
        from act_prm.environments.snorkel_finance.backend import DataBackend

        backend = DataBackend(data_path)
        tables = json.loads(backend.get_descriptions("at_t"))
        result = backend.sql_query("at_t", tables[0], f'SELECT item FROM "{tables[0]}"')
        assert "not allowed" in result.lower()

    def test_sql_query_invalid_table(self, data_path):
        from act_prm.environments.snorkel_finance.backend import DataBackend

        backend = DataBackend(data_path)
        result = backend.sql_query(
            "at_t",
            "nonexistent_table",
            'SELECT x FROM "nonexistent_table" WHERE rowid = 1',
        )
        assert "not found" in result.lower() or "error" in result.lower()


# ---------------------------------------------------------------------------
# Tools tests
# ---------------------------------------------------------------------------


class TestTools:
    """Tests for the 5 environment tools."""

    def test_calculator(self):
        from act_prm.environments.snorkel_finance.tools import CalculatorTool

        calc = CalculatorTool()
        result = json.loads(calc(expression="(310/6759)*100"))
        assert isinstance(result, list)
        assert len(result) == 2
        assert abs(result[0] - 4.586477) < 0.001

    def test_calculator_math_functions(self):
        from act_prm.environments.snorkel_finance.tools import CalculatorTool

        calc = CalculatorTool()
        result = json.loads(calc(expression="sqrt(144)"))
        assert result[0] == 12.0

    def test_calculator_error(self):
        from act_prm.environments.snorkel_finance.tools import CalculatorTool

        calc = CalculatorTool()
        result = calc(expression="undefined_var + 1")
        assert "error" in result.lower()

    def test_respond_user(self):
        from act_prm.environments.snorkel_finance.tools import RespondUserTool

        tool = RespondUserTool()
        result = tool(text="The answer is 42.")
        assert result == "The answer is 42."

    def test_tool_descriptions(self):
        from act_prm.environments.snorkel_finance.tools import TOOL_CLASSES

        for name, cls in TOOL_CLASSES.items():
            tool = cls()
            desc = tool.get_tool_desc()
            assert desc["name"] == name
            assert "description" in desc
            assert "parameters" in desc

    def test_get_descriptions_tool_calls_backend(self, data_path):
        from act_prm.environments.snorkel_finance.tools import GetDescriptionsTool
        from act_prm.environments.snorkel_finance.backend import DataBackend

        backend = DataBackend(data_path)
        tool = GetDescriptionsTool()
        result = tool(company_name="at_t", backend=backend)
        tables = json.loads(result)
        assert len(tables) > 0

    def test_sql_query_tool_calls_backend(self, data_path):
        from act_prm.environments.snorkel_finance.tools import SqlQueryTool
        from act_prm.environments.snorkel_finance.backend import DataBackend

        backend = DataBackend(data_path)
        tables = json.loads(backend.get_descriptions("at_t"))
        tool = SqlQueryTool()
        result = tool(
            company_name="at_t",
            table_name=tables[0],
            query=f'SELECT item FROM "{tables[0]}" WHERE rowid = 1',
            backend=backend,
        )
        parsed = json.loads(result)
        assert len(parsed) > 0


# ---------------------------------------------------------------------------
# Prompts tests
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_render_prompt(self):
        from act_prm.environments.snorkel_finance.prompts import render_prompt

        result = render_prompt(user_query="What is revenue?", company="apple")
        assert "apple" in result
        assert "What is revenue?" in result

    def test_system_prompts_exist(self):
        from act_prm.environments.snorkel_finance.prompts import SYSTEM_PROMPTS

        assert "finqa" in SYSTEM_PROMPTS
        assert "finqa_reasoning" in SYSTEM_PROMPTS
        assert len(SYSTEM_PROMPTS["finqa"]) > 0
        assert len(SYSTEM_PROMPTS["finqa_reasoning"]) > len(SYSTEM_PROMPTS["finqa"])

    def test_finqa_reasoning_prompt_has_max_turns(self):
        from act_prm.environments.snorkel_finance.prompts import SYSTEM_PROMPTS

        formatted = SYSTEM_PROMPTS["finqa_reasoning"].format(max_turns=50)
        assert "50" in formatted


# ---------------------------------------------------------------------------
# Grader unit tests
# ---------------------------------------------------------------------------


class TestSnorkelFinanceGraderPrompt:
    """Verify the grader prompt matches the reference FinQABenchmark/src/llmj.py."""

    def test_system_prompt_matches_reference(self):
        from act_prm.graders.snorkel_finance import (
            SNORKEL_FINANCE_GRADER_SYSTEM_PROMPT,
        )

        # Key phrases from the reference llmj.py prediction_correctness_prompt
        assert (
            "model response matches the label" in SNORKEL_FINANCE_GRADER_SYSTEM_PROMPT
        )
        assert "two decimal places" in SNORKEL_FINANCE_GRADER_SYSTEM_PROMPT
        assert "without rounding off" in SNORKEL_FINANCE_GRADER_SYSTEM_PROMPT
        assert "boxed" in SNORKEL_FINANCE_GRADER_SYSTEM_PROMPT
        assert "fraction" in SNORKEL_FINANCE_GRADER_SYSTEM_PROMPT

    def test_user_prompt_matches_reference(self):
        from act_prm.graders.snorkel_finance import (
            SNORKEL_FINANCE_GRADER_USER_TEMPLATE,
        )

        # Reference llmj.py formats: "Question: {q}\n\nModel Response: {r}\n\nLabel: {l}"
        formatted = SNORKEL_FINANCE_GRADER_USER_TEMPLATE.format(
            question="test q", response="test r", correct_answer="test a"
        )
        assert "Question: test q" in formatted
        assert "Model Response: test r" in formatted
        assert "Label: test a" in formatted

    def test_system_and_user_are_separate(self):
        """Grader should pass system prompt separately, not combined in user msg."""
        from act_prm.graders.snorkel_finance import (
            SNORKEL_FINANCE_GRADER_SYSTEM_PROMPT,
            SNORKEL_FINANCE_GRADER_USER_TEMPLATE,
        )

        # System prompt should NOT contain Question/Model Response/Label fields
        assert "Question:" not in SNORKEL_FINANCE_GRADER_SYSTEM_PROMPT
        assert "Model Response:" not in SNORKEL_FINANCE_GRADER_SYSTEM_PROMPT
        assert "Label:" not in SNORKEL_FINANCE_GRADER_SYSTEM_PROMPT

        # User template should NOT contain the grading instructions
        user = SNORKEL_FINANCE_GRADER_USER_TEMPLATE.format(
            question="q", response="r", correct_answer="a"
        )
        assert "two decimal places" not in user


# ---------------------------------------------------------------------------
# Environment tests (finqa_reasoning)
# ---------------------------------------------------------------------------


class TestSnorkelFinanceEnvReasoning:
    """Tests for SnorkelFinanceEnv with finqa_reasoning task."""

    def test_init(self, data_path, mock_grader, benchmark_reasoning_csv):
        from act_prm.environments.snorkel_finance.env import SnorkelFinanceEnv

        env = SnorkelFinanceEnv(
            data_path=data_path,
            benchmark_csv=benchmark_reasoning_csv,
            task="finqa_reasoning",
            max_turns=50,
            num_train_samples=47,
            num_val_samples=16,
            num_test_samples=16,
        )
        assert len(env.datasets["train"]) == 47
        assert len(env.datasets["eval"]) == 16
        assert len(env.datasets["test"]) == 16
        assert "step by step" in env.system_prompt

    def test_reset(self, data_path, mock_grader, benchmark_reasoning_csv):
        from act_prm.environments.snorkel_finance.env import SnorkelFinanceEnv

        env = SnorkelFinanceEnv(
            data_path=data_path,
            benchmark_csv=benchmark_reasoning_csv,
            task="finqa_reasoning",
            max_turns=50,
        )
        state = env.reset(sample_idx=0)
        assert state.company != ""
        assert state.question != ""
        assert state.answer != ""
        assert state.timestep == 0
        assert len(state.tools) == 5
        assert state.backend is not None

    def test_reset_wraps_around(self, data_path, mock_grader, benchmark_reasoning_csv):
        from act_prm.environments.snorkel_finance.env import SnorkelFinanceEnv

        env = SnorkelFinanceEnv(
            data_path=data_path,
            benchmark_csv=benchmark_reasoning_csv,
            task="finqa_reasoning",
            max_turns=50,
        )
        n = len(env)
        state1 = env.reset(sample_idx=0)
        state2 = env.reset(sample_idx=n)
        assert state1.question == state2.question

    def test_tools_work_through_state(
        self, data_path, mock_grader, benchmark_reasoning_csv
    ):
        from act_prm.environments.snorkel_finance.env import SnorkelFinanceEnv

        env = SnorkelFinanceEnv(
            data_path=data_path,
            benchmark_csv=benchmark_reasoning_csv,
            task="finqa_reasoning",
            max_turns=50,
        )
        state = env.reset(sample_idx=0)
        result = state.tool_registry["get_descriptions"](
            company_name=state.company, backend=state.backend
        )
        tables = json.loads(result)
        assert len(tables) > 0

    def test_split_deterministic(self, data_path, mock_grader, benchmark_reasoning_csv):
        """Same seed produces same splits."""
        from act_prm.environments.snorkel_finance.env import SnorkelFinanceEnv

        env1 = SnorkelFinanceEnv(
            data_path=data_path,
            benchmark_csv=benchmark_reasoning_csv,
            task="finqa_reasoning",
            max_turns=50,
            seed=42,
        )
        env2 = SnorkelFinanceEnv(
            data_path=data_path,
            benchmark_csv=benchmark_reasoning_csv,
            task="finqa_reasoning",
            max_turns=50,
            seed=42,
        )
        for split in ["train", "eval", "test"]:
            q1 = env1.datasets[split]["question"]
            q2 = env2.datasets[split]["question"]
            assert q1 == q2


# ---------------------------------------------------------------------------
# Environment tests (finqa)
# ---------------------------------------------------------------------------


class TestSnorkelFinanceEnvFinQA:
    """Tests for SnorkelFinanceEnv with finqa task."""

    def test_init(self, data_path, mock_grader, benchmark_finqa_csv):
        from act_prm.environments.snorkel_finance.env import SnorkelFinanceEnv

        env = SnorkelFinanceEnv(
            data_path=data_path,
            benchmark_csv=benchmark_finqa_csv,
            task="finqa",
            max_turns=50,
            num_train_samples=174,
            num_val_samples=58,
            num_test_samples=58,
        )
        assert len(env.datasets["train"]) == 174
        assert len(env.datasets["eval"]) == 58
        assert len(env.datasets["test"]) == 58
        assert env.system_prompt == "Only execute one tool call at a time"

    def test_boxed_answers(self, data_path, mock_grader, benchmark_finqa_csv):
        from act_prm.environments.snorkel_finance.env import SnorkelFinanceEnv

        env = SnorkelFinanceEnv(
            data_path=data_path,
            benchmark_csv=benchmark_finqa_csv,
            task="finqa",
            max_turns=50,
        )
        state = env.reset(sample_idx=0)
        assert (
            "boxed" in state.answer
            or state.answer.replace(".", "").replace("-", "").isdigit()
        )


# ---------------------------------------------------------------------------
# Live grader tests (require ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestLiveGrader:
    """Live integration tests for ClaudeQueryLLM-powered grader.

    Tests use real finqa and finqa_reasoning samples from the benchmark CSVs.

    Run with: pytest tests/test_snorkel_finance.py -v -m live
    """

    @pytest.fixture(autouse=True)
    def grader(self):
        from act_prm.graders.snorkel_finance import SnorkelFinanceGrader

        self._grader = SnorkelFinanceGrader(
            grader_model_config={
                "name": "claude_query",
                "model_config": {"model": "claude-sonnet-4-6"},
            },
            num_samples=1,
            verbose=False,
        )

    # --- finqa: boxed numeric answers ---

    def test_finqa_correct_ratio(self):
        """finqa sample: ratio of Domestic to Foreign Income = 6.118."""
        is_correct, _ = self._grader(
            question=(
                "What is the ratio of Domestic Income to Foreign Income "
                "for continuing operations before income taxes in 2022?"
            ),
            correct_answer=r"\boxed{6.118}",
            response="The ratio of Domestic Income to Foreign Income is approximately 6.118.",
        )
        assert is_correct is True

    def test_finqa_incorrect_ratio(self):
        """finqa sample: wrong numeric answer should be graded incorrect."""
        is_correct, _ = self._grader(
            question=(
                "What is the ratio of Domestic Income to Foreign Income "
                "for continuing operations before income taxes in 2022?"
            ),
            correct_answer=r"\boxed{6.118}",
            response="The ratio is approximately 3.5.",
        )
        assert is_correct is False

    def test_finqa_decimal_matching(self):
        """finqa: verify 2-decimal-place matching rule from llmj.py."""
        is_correct, _ = self._grader(
            question="What is the interest ratio?",
            correct_answer=r"\boxed{0.112}",
            response="The ratio is 0.112345.",
        )
        # 0.11 == 0.11 (truncated to 2 decimal places) → correct
        assert is_correct is True

    # --- finqa_reasoning: paragraph answers ---

    def test_finqa_reasoning_att_correct(self):
        """finqa_reasoning sample 0: AT&T postretirement interest cost ratio."""
        is_correct, _ = self._grader(
            question=(
                "How significant are the company's postretirement benefit "
                "obligations in terms of interest burden, and what does this "
                "indicate about the company's long-term liability management "
                "in 2024?"
            ),
            correct_answer=(
                "The interest cost ratio for postretirement benefits in 2024 "
                "is 4.6%, which indicates a relatively moderate interest burden "
                "on the company's postretirement obligations."
            ),
            response=(
                "AT&T's postretirement benefit interest cost represents about "
                "4.6% of total interest expense in 2024, indicating a moderate "
                "burden and reasonable long-term liability management."
            ),
        )
        assert is_correct is True

    def test_finqa_reasoning_att_incorrect(self):
        """finqa_reasoning: wrong key figure should be graded incorrect."""
        is_correct, _ = self._grader(
            question=(
                "How significant are the company's postretirement benefit "
                "obligations in terms of interest burden, and what does this "
                "indicate about the company's long-term liability management "
                "in 2024?"
            ),
            correct_answer=(
                "The interest cost ratio for postretirement benefits in 2024 "
                "is 4.6%, which indicates a relatively moderate interest burden "
                "on the company's postretirement obligations."
            ),
            response=(
                "The postretirement benefit interest cost ratio is 25.3%, "
                "indicating a very heavy burden on the company's finances."
            ),
        )
        assert is_correct is False

    def test_finqa_reasoning_meta_leases(self):
        """finqa_reasoning sample 1: Meta lease financing strategy."""
        is_correct, _ = self._grader(
            question=(
                "What is the company's lease financing strategy and how "
                "does it impact cash flow obligations?"
            ),
            correct_answer=(
                "The company primarily utilizes operating leases over finance "
                "leases, with finance leases representing only 3.4% of total "
                "undiscounted lease cash flows."
            ),
            response=(
                "Meta primarily relies on operating leases rather than finance "
                "leases. Finance leases account for approximately 3.4% of the "
                "total undiscounted lease cash flows, suggesting a preference "
                "for off-balance-sheet financing."
            ),
        )
        assert is_correct is True
