import re
from math_verify import LatexExtractionConfig, parse, verify


def reward_len(completions, ideal_length=50, **kwargs):
    """Reward function that checks if the completion is the correct length."""
    return [-abs(ideal_length - len(completion)) for completion in completions]


def reward_format(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def reward_accuracy(completions, **kwargs):
    """Reward function that checks if the completion is correct in AI-MO/NuminaMath-TIR"""
    solutions = kwargs["solution"]
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(
            solution,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        answer_parsed = parse(
            content,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards
