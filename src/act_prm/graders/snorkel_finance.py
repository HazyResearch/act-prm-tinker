"""
LLM-based grader for Snorkel Finance QA tasks.

Matches the grading prompt from FinQABenchmark/src/llmj.py with
decimal matching rules specific to financial QA.
"""

import re

from .qa import LLMGraderForQA

# From: https://github.com/snorkel-ai/FinQABenchmark/blob/main/src/llmj.py
SNORKEL_FINANCE_GRADER_TEMPLATE = """I am going to give you
- question
- model response
- label

Your task is to tell me if you think the model response matches the label.

The model response might be a larger piece of text, and the label might be in a different format, e.g.

- question : What is the age of dog Kamikaze?
- model response : The age of dog Kamikaze is 52
- label : \\boxed{{52}}

Your response : {{True, 'Yes they both match'}}

It's allowed to have model response as a fraction. You can compute the fraction and check if the prediction matches the fraction within two decimal places.

If model response OR label exceeds more than two decimal places, you will compare only uptill the first two decimal places for the model answer and the label, without rounding off.

Question: {question}

Model Response: {response}

Label: {correct_answer}

Provide your judgement in the following format:
correct: Answer 'yes' if the model response matches the label, 'no' otherwise.
rationale: Brief explanation of why they match or don't match."""


class SnorkelFinanceGrader(LLMGraderForQA):
    """Grader for Snorkel Finance QA using the reference FinQABenchmark prompt.

    Inherits majority voting, metrics tracking from LLMGraderForQA.
    Overrides grade_sample to use decimal-matching-aware prompt.
    """

    def grade_sample(
        self,
        question: str,
        correct_answer: str,
        response: str,
    ) -> tuple[str, str]:
        grader_prompt = SNORKEL_FINANCE_GRADER_TEMPLATE.format(
            question=question,
            correct_answer=correct_answer,
            response=response,
        ).strip()
        prompt_messages = [{"role": "user", "content": grader_prompt}]
        sampler_response = self.grader_model.sample(
            system_prompt="",
            messages=prompt_messages,
            tools=None,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=1,
        )[0]
        grading_response = self.grader_model.get_actions(sampler_response)[-1].text
        match = re.search(
            r"correct:\s*(yes|no)\b", grading_response, flags=re.IGNORECASE
        )
        match = match.group(1).lower() if match else "no"
        return match, grading_response.strip()
