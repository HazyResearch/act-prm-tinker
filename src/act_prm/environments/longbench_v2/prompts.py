"""
Prompt (templates) for LongBench environment
"""

INITIAL_PROMPT_TEMPLATE = """
You are given a long document and some available tools. Your goal is to answer the question below.
- To do this, you **must** use the tools and think between tool calls.
- This will let you navigate the document to answer the question.

## Question:
{question}

## Possible Answers:
{choices}
---

## Document View (Initial):
'''
{document}
(Scroll down for more...)
'''
---

## Tool Calls
After each tool call, you should reflect on the results. Then, either call another tool or answer the question.

## Submission
When you're ready to answer, put your response as one letter in the following format:

Final Answer: <your chosen answer letter (A, B, C, or D)>
"""
