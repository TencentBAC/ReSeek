# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
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
    """提取solution中的information和对应的judge标签对"""
    pairs = []
    
    # 查找所有information标签
    info_pattern = r"<information>(.*?)</information>"
    info_matches = list(re.finditer(info_pattern, solution_str, re.DOTALL))
    
    for info_match in info_matches:
        info_content = info_match.group(1).strip()
        info_end = info_match.end()
        
        # 在information标签后查找最近的judge标签
        remaining_text = solution_str[info_end:]
        judge_pattern = r"<judge>(Yes|No)</judge>"
        judge_match = re.search(judge_pattern, remaining_text, re.IGNORECASE)
        
        if judge_match:
            judge_value = judge_match.group(1).strip()
            pairs.append((info_content, judge_value))
    
    return pairs

def check_label_in_information(label: Union[str, List[str]], information: str) -> bool:
    """
    检查label是否在information中
    支持多种匹配策略：
    1. 完全包含
    
    Args:
        label: 可以是字符串或字符串列表，如果是列表则只要命中一个就算匹配
        information: 信息文本
    
    Returns:
        bool: 是否匹配
    """
    if not label or not information:
        return False
    
    # 如果label是字符串，转换为列表
    if isinstance(label, str):
        labels = [label]
    else:
        labels = label
    
    info_norm = information.lower().strip()
    
    # 遍历所有label，只要有一个匹配就返回True
    for single_label in labels:
        if not single_label:
            continue
            
        label_norm = single_label.lower().strip()
        
        # 1. 完全包含检查
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
    计算内容重叠度奖励（不使用embedding model）
    
    Args:
        solution_str: 完整的解答文本
        final_answer: 最终答案
        ground_truth: 正确答案
        positive_reward: 正确判断的奖励
        negative_reward: 错误判断的惩罚
    
    Returns:
        内容重叠度奖励分数
    """
    if not final_answer:
        return 0.0
    
    info_judge_pairs = extract_information_judge_pairs(solution_str)
    if not info_judge_pairs:
        return 0.0
    
    # 检查最终答案是否正确
    is_answer_correct = em_check(final_answer, ground_truth["target"])
    
    total_reward = 0.0
    

    for info_content, judge_value in info_judge_pairs:
        if not info_content:
            continue
        
        model_said_yes = (judge_value.lower() == "yes")
        
        # 如果judge是yes且最终答案正确，直接给正奖励，不检查info内容
        if model_said_yes and is_answer_correct:
            reward = correct_judge_reward + answer_correct_boost
            total_reward += reward
        else:
            # 对于其他情况，检查info内容是否真的有用
            is_info_truly_useful = check_label_in_information(ground_truth["target"], info_content)
            is_judge_correct = (model_said_yes and is_info_truly_useful) or \
                            (not model_said_yes and not is_info_truly_useful)
                            
            if is_judge_correct:
                reward = correct_judge_reward
                if is_answer_correct:
                    reward += answer_correct_boost
                total_reward += reward
            else:
                # 判断是哪种类型的错误，并施加不同的惩罚
                if model_said_yes: # 这意味着模型错误地说了 "Yes"
                    total_reward += wrong_yes_penalty
                else: # 这意味着模型错误地说了 "No"
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

def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0, 
                 enable_content_overlap=True, answer_correct_boost=0.1, correct_judge_reward=0.3, wrong_yes_penalty=-0.6, wrong_no_penalty=-0.3):
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

    # 计算内容重叠度奖励
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

def compute_score_subem(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0,
                       enable_content_overlap=True, answer_correct_boost=0.1, correct_judge_reward=0.3, wrong_yes_penalty=-0.6, wrong_no_penalty=-0.3):
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

    # 计算内容重叠度奖励
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

def compute_batch_content_overlap_reward(
    solution_strs: List[str], 
    final_answers: List[str], 
    ground_truths: List[dict],
    answer_correct_boost: float = 0.1,
    correct_judge_reward: float = 0.3,
    wrong_yes_penalty: float = -0.6,
    wrong_no_penalty: float = -0.3
) -> List[float]:
    """
    批量计算内容重叠度奖励（不使用embedding model）
    
    Args:
        solution_strs: 完整的解答文本列表
        final_answers: 最终答案列表
        ground_truths: 正确答案列表
        answer_correct_boost: 答案正确时的额外奖励
        correct_judge_reward: 正确判断的奖励
        wrong_yes_penalty: 错误说yes的惩罚
        wrong_no_penalty: 错误说no的惩罚
    
    Returns:
        内容重叠度奖励分数列表
    """
    if not solution_strs or not final_answers or not ground_truths:
        return [0.0] * len(solution_strs)
    
    batch_size = len(solution_strs)
    rewards = [0.0] * batch_size
    
    for batch_idx in range(batch_size):
        if not final_answers[batch_idx]:
            continue

        solution_str = solution_strs[batch_idx]
        final_answer = final_answers[batch_idx]
        ground_truth = ground_truths[batch_idx]
        
        info_judge_pairs = extract_information_judge_pairs(solution_str)
        if not info_judge_pairs:
            continue
        
        # 检查最终答案是否正确
        is_answer_correct = em_check(final_answer, ground_truth["target"])
        
        for info_content, judge_value in info_judge_pairs:
            if not info_content:
                continue
            
            model_said_yes = (judge_value.lower() == "yes")
            
            # 如果judge是yes且最终答案正确，直接给正奖励，不检查info内容
            if model_said_yes and is_answer_correct:
                rewards[batch_idx] = 0.0
            else:
                # 对于其他情况，检查info内容是否真的有用
                label_in_info = check_label_in_information(ground_truth["target"], info_content)
                
                if judge_value.lower() == "yes":
                    if label_in_info:
                        # 正确判断：说有用且确实有用
                        reward = correct_judge_reward
                        if is_answer_correct:
                            reward = 0.0
                        rewards[batch_idx] += reward
                    else:
                        # 错误判断：说有用但实际没用
                        rewards[batch_idx] += wrong_yes_penalty
                        
                elif judge_value.lower() == "no":
                    if not label_in_info:
                        # 正确判断：说没用且确实没用
                        rewards[batch_idx] += correct_judge_reward
                    else:
                        # 错误判断：说没用但实际有用
                        rewards[batch_idx] += wrong_no_penalty
    
    return rewards

def compute_batch_score(solution_strs: List[str], ground_truths: List[dict], 
                       method="strict", format_score=0.0, score=1.0, 
                       enable_content_overlap=True, answer_correct_boost=0.1, correct_judge_reward=0.3, wrong_yes_penalty=-0.6, wrong_no_penalty=-0.3) -> List[float]:
    """批量版本的scoring函数，用于exact match (EM) with content overlap reward.

    Args:
        solution_strs: 解答文本列表
        ground_truths: 正确答案列表
        method: 提取解答的方法，选择 'strict' 或 'flexible'
        format_score: 格式分数
        score: 正确答案的分数
        enable_content_overlap: 是否启用内容重叠度评分
        positive_reward: 正确判断的奖励
        negative_reward: 错误判断的惩罚
    
    Returns:
        分数列表
    """
    if not solution_strs:
        return []
    
    batch_size = len(solution_strs)
    batch_scores = []
    
    # 批量提取答案
    answers = []
    for solution_str in solution_strs:
        answer = extract_solution(solution_str=solution_str)
        answers.append(answer)
    
    # 批量计算基础分数
    base_scores = []
    for i in range(batch_size):
        answer = answers[i]
        ground_truth = ground_truths[i]
        solution_str = solution_strs[i]
        
        if answer is None:
            base_score = 0.0
        else:
            if em_check(answer, ground_truth["target"]):
                open_count, close_count = count_answer_tags(solution_str)
                if open_count > 10 or close_count > 10:  # prevent output a lot of </answer>
                    base_score = score / 4
                else:
                    base_score = score
            else:
                base_score = format_score
        base_scores.append(base_score)
    
    # 批量计算内容重叠度奖励
    content_overlap_rewards = [0.0] * batch_size
    if enable_content_overlap:
        # 过滤出有答案的样本
        valid_indices = [i for i in range(batch_size) if answers[i] is not None]
        if valid_indices:
            valid_solution_strs = [solution_strs[i] for i in valid_indices]
            valid_answers = [answers[i] for i in valid_indices]
            valid_ground_truths = [ground_truths[i] for i in valid_indices]
            
            valid_rewards = compute_batch_content_overlap_reward(
                solution_strs=valid_solution_strs,
                final_answers=valid_answers,
                ground_truths=valid_ground_truths,
                answer_correct_boost=answer_correct_boost,
                correct_judge_reward=correct_judge_reward,
                wrong_yes_penalty=wrong_yes_penalty,
                wrong_no_penalty=wrong_no_penalty,
            )
            
            # 将结果映射回原始索引
            for i, valid_idx in enumerate(valid_indices):
                content_overlap_rewards[valid_idx] = valid_rewards[i]
    
    # 合并基础分数和内容重叠奖励
    for i in range(batch_size):
        final_score = base_scores[i] + content_overlap_rewards[i]
        batch_scores.append(final_score)
    
    return batch_scores

def compute_batch_score_subem(solution_strs: List[str], ground_truths: List[dict],
                             method="strict", format_score=0.0, score=1.0,
                             enable_content_overlap=True, answer_correct_boost=0.1, correct_judge_reward=0.3, wrong_yes_penalty=-0.6, wrong_no_penalty=-0.3) -> List[float]:
    """批量版本的scoring函数，用于substring exact match (EM) with content overlap reward.

    Args:
        solution_strs: 解答文本列表
        ground_truths: 正确答案列表
        method: 提取解答的方法，选择 'strict' 或 'flexible'
        format_score: 格式分数
        score: 正确答案的分数
        enable_content_overlap: 是否启用内容重叠度评分
        positive_reward: 正确判断的奖励
        negative_reward: 错误判断的惩罚
    
    Returns:
        分数列表
    """
    if not solution_strs:
        return []
    
    batch_size = len(solution_strs)
    batch_scores = []
    
    # 批量提取答案
    answers = []
    for solution_str in solution_strs:
        answer = extract_solution(solution_str=solution_str)
        answers.append(answer)
    
    # 批量计算基础分数
    base_scores = []
    for i in range(batch_size):
        answer = answers[i]
        ground_truth = ground_truths[i]
        
        if answer is None:
            base_score = 0.0
        else:
            if subem_check(answer, ground_truth["target"]):
                base_score = score
            else:
                base_score = format_score
        base_scores.append(base_score)
    
    # 批量计算内容重叠度奖励
    content_overlap_rewards = [0.0] * batch_size
    if enable_content_overlap:
        # 过滤出有答案的样本
        valid_indices = [i for i in range(batch_size) if answers[i] is not None]
        if valid_indices:
            valid_solution_strs = [solution_strs[i] for i in valid_indices]
            valid_answers = [answers[i] for i in valid_indices]
            valid_ground_truths = [ground_truths[i] for i in valid_indices]
            
            valid_rewards = compute_batch_content_overlap_reward(
                solution_strs=valid_solution_strs,
                final_answers=valid_answers,
                ground_truths=valid_ground_truths,
                answer_correct_boost=answer_correct_boost,
                correct_judge_reward=correct_judge_reward,
                wrong_yes_penalty=wrong_yes_penalty,
                wrong_no_penalty=wrong_no_penalty,
            )
            
            # 将结果映射回原始索引
            for i, valid_idx in enumerate(valid_indices):
                content_overlap_rewards[valid_idx] = valid_rewards[i]
    
    # 合并基础分数和内容重叠奖励
    for i in range(batch_size):
        final_score = base_scores[i] + content_overlap_rewards[i]
        batch_scores.append(final_score)
    
    return batch_scores



