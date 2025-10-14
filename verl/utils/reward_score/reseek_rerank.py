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
from typing import List, Tuple, Optional, Any
import numpy as np
import requests
import json


# Embedding service client class
class EmbeddingClient:
    """Qwen3 Embedding service client"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def get_embeddings(self, texts: List[str], prompt_name: Optional[str] = None):
        """Get text embedding vectors"""
        payload = {"texts": texts, "prompt_name": prompt_name}

        try:
            response = requests.post(
                f"{self.base_url}/embed",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,  # Add timeout setting
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Failed to get embedding vectors: {e}")
            return None

    def compute_similarity(self, queries: List[str], documents: List[str], prompt_name: Optional[str] = None):
        """Compute similarity between queries and documents"""
        payload = {
            "text_1": queries,
            "text_2": documents,
            "model": "/group/40077/shyuli/models/embedding/bge-reranker-v2-m3",
        }
        try:
            response = requests.post(
                f"{self.base_url}/score",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Failed to compute similarity: {e}")
            return None


# Global variable for caching client
_embedding_client = None


def get_embedding_client():
    """Get or initialize embedding client"""
    global _embedding_client
    if _embedding_client is None:
        # Can configure server address through environment variables
        import os

        server_url = os.getenv("EMBEDDING_SERVER_URL", "http://9.130.192.143:8000")
        _embedding_client = EmbeddingClient(server_url)
    return _embedding_client


def compute_cosine_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts"""
    client = get_embedding_client()
    # If text1 or text2 is a list, take the first element
    if isinstance(text1, list):
        text1 = text1[0] if text1 else ""
    if isinstance(text2, list):
        text2 = text2[0] if text2 else ""

    # Ensure input is string
    text1 = str(text1) if text1 is not None else ""
    text2 = str(text2) if text2 is not None else ""

    # Use server-side similarity computation interface
    result = client.compute_similarity([text1], [text2])
    if result and result.get("similarities") and len(result["similarities"]) > 0:
        # Return first value from scores
        return float(result["similarities"][0])
    else:
        # If service call fails, return 0
        print("Warning: embedding service call failed, returning similarity 0")
        return 0.0


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


def compute_content_overlap_reward(
    solution_str: str,
    final_answer: str,
    ground_truth,
    similarity_threshold: float = 0.5,
    positive_reward: float = 0.4,
    negative_reward: float = -0.2,
) -> float:
    """
    Compute content overlap reward

    Args:
        solution_str: Complete solution text
        final_answer: Final answer
        ground_truth: Correct answer
        similarity_threshold: Similarity threshold
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

        if judge_value.lower() == "yes":
            if is_answer_correct:
                # If judge=Yes and final answer is correct, give positive reward directly without computing similarity
                total_reward += positive_reward
            else:
                # judge=Yes but answer is wrong, need to compute similarity to judge if information is really useful
                similarity = compute_cosine_similarity(ground_truth["target"], info_content)
                is_high_similarity = similarity >= similarity_threshold

                if is_high_similarity:
                    # Information is similar to answer but answer is wrong, judgment might be correct but answer generation has issues, give small negative reward
                    total_reward += negative_reward * 0.5
                else:
                    # Information is not similar to answer and answer is wrong, judgment is incorrect, give negative reward
                    total_reward += negative_reward

        elif judge_value.lower() == "no":
            # For judge=No case, compute similarity
            similarity = compute_cosine_similarity(ground_truth["target"], info_content)
            is_high_similarity = similarity >= similarity_threshold

            if is_high_similarity:
                # Incorrect judgment: said not useful but actually useful
                total_reward += negative_reward
            else:
                # Correct judgment: said not useful and indeed not useful
                total_reward += positive_reward * 0.2

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

    # If there are 0  matches, return None
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
    similarity_threshold=0.5,
    overlap_positive_reward=0.1,
    overlap_negative_reward=-0.05,
):
    """The scoring function for exact match (EM) with content overlap reward.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
        enable_content_overlap: whether to enable content overlap scoring
        similarity_threshold: threshold for determining high similarity
        overlap_positive_reward: reward for correct judgment
        overlap_negative_reward: penalty for incorrect judgment
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
            similarity_threshold=similarity_threshold,
            positive_reward=overlap_positive_reward,
            negative_reward=overlap_negative_reward,
        )

        if do_print:
            print(f"Content overlap reward: {content_overlap_reward}")

    # Return dictionary format with detailed information
    return base_score + content_overlap_reward



def compute_score_subem(
    solution_str,
    ground_truth,
    method="strict",
    format_score=0.0,
    score=1.0,
    enable_content_overlap=True,
    similarity_threshold=0.5,
    overlap_positive_reward=0.1,
    overlap_negative_reward=-0.05,
):
    """The scoring function for substring exact match (EM) with content overlap reward.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
        enable_content_overlap: whether to enable content overlap scoring
        similarity_threshold: threshold for determining high similarity
        overlap_positive_reward: reward for correct judgment
        overlap_negative_reward: penalty for incorrect judgment
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
            similarity_threshold=similarity_threshold,
            positive_reward=overlap_positive_reward,
            negative_reward=overlap_negative_reward,
        )

        if do_print:
            print(f"Content overlap reward: {content_overlap_reward}")

    # Return dictionary format with detailed information
    return base_score + content_overlap_reward
    # {
    #     "score": base_score + content_overlap_reward,
    #     "base_score": base_score,
    #     "content_overlap_reward": content_overlap_reward,
    #     "subem_correct": base_score > 0,
    #     "has_answer": answer is not None
    # }

