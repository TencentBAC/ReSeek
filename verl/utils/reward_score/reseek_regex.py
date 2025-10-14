# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
# Copyright 2025 ReSeek Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py

import random
import re
import string
from typing import List, Tuple, Optional, Any, Union


def extract_information_judge_pairs(solution_str: str) -> List[Tuple[str, str]]:
    """Extract information and corresponding judge label pairs from solution"""
    pairs = []

    # Find all information tags
    info_pattern = r"<information>(.*?)</information>"
    info_matches = list(re.finditer(info_pattern, solution_str, re.DOTALL))

    for info_match in info_matches:
        info_content = info_match.group(1).strip()
        info_end = info_match.end()

        # Find nearest judge tag after information tag
        remaining_text = solution_str[info_end:]
        judge_pattern = r"<judge>(Yes|No)</judge>"
        judge_match = re.search(judge_pattern, remaining_text, re.IGNORECASE)

        if judge_match:
            judge_value = judge_match.group(1).strip()
            pairs.append((info_content, judge_value))

    return pairs


def check_label_in_information(label: Union[str, List[str]], information: str) -> bool:
    """
    Check if label is in information
    Supports multiple matching strategies:
    1. Complete inclusion

    Args:
        label: Can be string or string list, if list then match if any one hits
        information: Information text

    Returns:
        bool: Whether matched
    """
    if not label or not information:
        return False

    # If label is string, convert to list
    if isinstance(label, str):
        labels = [label]
    else:
        labels = label

    info_norm = information.lower().strip()

    # Iterate through all labels, return True if any one matches
    for single_label in labels:
        if not single_label:
            continue

        label_norm = single_label.lower().strip()

        # 1. Complete inclusion check
        if label_norm in info_norm:
            return True

    return False


def compute_content_overlap_reward(
    solution_str: str,
    final_answer: str,
    ground_truth,
    answer_correct_boost: float = 0.1,
    correct_judge_reward: float = 0.3,
    wrong_yes_penalty: float = -0.6,
    wrong_no_penalty: float = -0.3,
) -> float:
    """
    Compute content overlap reward (without using embedding model)

    Args:
        solution_str: Complete solution text
        final_answer: Final answer
        ground_truth: Correct answer
        positive_reward: Reward for correct judgment
        negative_reward: Penalty for incorrect judgment

    Returns:
        Content overlap reward score
    """
    if not final_answer:
        return 0.0

    info_judge_pairs = extract_information_judge_pairs(solution_str)
    if not info_judge_pairs:
        return 0.0

    # Check if final answer is correct
    is_answer_correct = em_check(final_answer, ground_truth["target"])

    total_reward = 0.0

    for info_content, judge_value in info_judge_pairs:
        if not info_content:
            continue

        model_said_yes = judge_value.lower() == "yes"

        # If judge is yes and final answer is correct, give positive reward directly without checking info content
        if model_said_yes and is_answer_correct:
            reward = correct_judge_reward + answer_correct_boost
            total_reward += reward
        else:
            # For other cases, check if info content is really useful
            is_info_truly_useful = check_label_in_information(ground_truth["target"], info_content)
            is_judge_correct = (model_said_yes and is_info_truly_useful) or (
                not model_said_yes and not is_info_truly_useful
            )

            if is_judge_correct:
                reward = correct_judge_reward
                if is_answer_correct:
                    reward += answer_correct_boost
                total_reward += reward
            else:
                # Determine what type of error and apply different penalties
                if model_said_yes:  # This means model incorrectly said "Yes"
                    total_reward += wrong_yes_penalty
                else:  # This means model incorrectly said "No"
                    total_reward += wrong_no_penalty

    return total_reward


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0 matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")
    return opening_tags, closing_tags


def compute_score(
    solution_str,
    ground_truth,
    method="strict",
    format_score=0.0,
    score=1.0,
    enable_content_overlap=True,
    answer_correct_boost=0.1,
    correct_judge_reward=0.3,
    wrong_yes_penalty=-0.6,
    wrong_no_penalty=-0.3,
):
    """The scoring function for exact match (EM) with content overlap reward.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
        enable_content_overlap: whether to enable content overlap scoring
        positive_reward: reward for correct judgment
        negative_reward: penalty for incorrect judgment
    """
    answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    base_score = 0.0
    if answer is None:
        base_score = 0.0
    else:
        if em_check(answer, ground_truth["target"]):
            if open_count > 10 or close_count > 10:  # prevent output a lot of </answer>
                base_score = score / 4
            else:
                base_score = score
        else:
            base_score = format_score

    # Compute content overlap reward
    content_overlap_reward = 0.0
    if enable_content_overlap and answer is not None:
        content_overlap_reward = compute_content_overlap_reward(
            solution_str=solution_str,
            final_answer=answer,
            ground_truth=ground_truth,
            answer_correct_boost=answer_correct_boost,
            correct_judge_reward=correct_judge_reward,
            wrong_yes_penalty=wrong_yes_penalty,
            wrong_no_penalty=wrong_no_penalty,
        )

        if do_print:
            print(f"Content overlap reward: {content_overlap_reward}")

    return base_score + content_overlap_reward


def compute_score_subem(
    solution_str,
    ground_truth,
    method="strict",
    format_score=0.0,
    score=1.0,
    enable_content_overlap=True,
    answer_correct_boost=0.1,
    correct_judge_reward=0.3,
    wrong_yes_penalty=-0.6,
    wrong_no_penalty=-0.3,
):
    """The scoring function for substring exact match (EM) with content overlap reward.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
        enable_content_overlap: whether to enable content overlap scoring
        positive_reward: reward for correct judgment
        negative_reward: penalty for incorrect judgment
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    base_score = 0.0
    if answer is None:
        base_score = 0.0
    else:
        if subem_check(answer, ground_truth["target"]):
            base_score = score
        else:
            base_score = format_score

    # Compute content overlap reward
    content_overlap_reward = 0.0
    if enable_content_overlap and answer is not None:
        content_overlap_reward = compute_content_overlap_reward(
            solution_str=solution_str,
            final_answer=answer,
            ground_truth=ground_truth,
            answer_correct_boost=answer_correct_boost,
            correct_judge_reward=correct_judge_reward,
            wrong_yes_penalty=wrong_yes_penalty,
            wrong_no_penalty=wrong_no_penalty,
        )

        if do_print:
            print(f"Content overlap reward: {content_overlap_reward}")

    return base_score + content_overlap_reward


