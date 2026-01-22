"""
Instruction prompts for BabyAI-Text environment
"""

INSTRUCTION_PROMPT_BABYAI_TEXT = """
You are an agent playing BabyAI-Text, a grid-based game with text observations.
Your goal is to follow the mission and reach the win condition.

## Rules:
- You must call exactly one tool per turn to choose an action.
- Only the listed tools are valid actions.
- You have {max_turns} steps to complete the task. Restarting is forbidden.
""".strip()


def get_instruction_prompt(max_turns: int | None = None) -> str:
    """
    Get the instruction prompt for BabyAI-Text
    """
    max_turns = max_turns or 20
    return INSTRUCTION_PROMPT_BABYAI_TEXT.format(max_turns=max_turns).strip()
