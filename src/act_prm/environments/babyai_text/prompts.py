"""
System-level instruction prompts for the BabyAI-Text environment.

Defines the rules and constraints that are prepended to the LLM's system
prompt, telling it how to play BabyAI-Text (one tool call per turn, only
listed tools are valid, limited number of steps).
"""

# Template string with a {max_turns} placeholder that gets filled in at runtime.
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
    Get the instruction prompt for BabyAI-Text with the max turns filled in.

    Args:
        max_turns: Maximum number of steps the agent has. Defaults to 20
            if not provided.

    Returns:
        The instruction prompt string with ``{max_turns}`` replaced.
    """
    max_turns = max_turns or 20
    return INSTRUCTION_PROMPT_BABYAI_TEXT.format(max_turns=max_turns).strip()
