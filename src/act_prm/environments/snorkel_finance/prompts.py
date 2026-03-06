"""
Prompts for Snorkel Agent Finance Reasoning environment.
"""

# System prompt from the original snorkelai/agent-finance-reasoning dataset
SYSTEM_PROMPT = (
    "Only execute one tool call at a time. Think and reason step by step. "
    "Feel free to iteratively gather data as much as you need to answer the question. "
    "Although there is a limit of {max_turns} turns, you can still gather data as much as you "
    "need to answer the question. When you have gathered all the data, answer the question.\n\n"
    "Once you have gathered all the data, then you can gradually start forming an answer. "
    "Once you have formed an answer, generate a 1 paragraph summary of it with all "
    "the relevant figures summarized."
)


def render_prompt(
    user_query: str,
    company: str,
) -> str:
    """Render the user prompt for a snorkel finance task."""
    return (
        f"Here is the question: {user_query}\n\n"
        f"Here are the companies name in the database to query for: {company}\n\n"
        f"Use the available tools to gather financial data and answer the question. "
        f"When ready, use the `respond_user` tool to provide your final answer."
    )
