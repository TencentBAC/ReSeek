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
from typing import List, Tuple, Optional, Any
import numpy as np
import requests
import json

# Embedding服务客户端类
class EmbeddingClient:
    """Qwen3 Embedding服务客户端"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def get_embeddings(self, texts: List[str], prompt_name: Optional[str] = None):
        """获取文本嵌入向量"""
        payload = {"texts": texts, "prompt_name": prompt_name}

        try:
            response = requests.post(
                f"{self.base_url}/embed", 
                json=payload, 
                headers={"Content-Type": "application/json"},
                timeout=30  # 添加超时设置
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"获取嵌入向量失败: {e}")
            return None

    def compute_similarity(self, queries: List[str], documents: List[str], prompt_name: Optional[str] = None):
        """计算查询和文档之间的相似度"""
        payload = {"text_1": queries, "text_2": documents, "model":"/group/40077/shyuli/models/embedding/bge-reranker-v2-m3"}
        try:
            response = requests.post(
                f"{self.base_url}/score", 
                json=payload, 
                headers={"Content-Type": "application/json"},            
                )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"计算相似度失败: {e}")
            return None

# 全局变量用于缓存客户端
_embedding_client = None

def get_embedding_client():
    """获取或初始化embedding客户端"""
    global _embedding_client
    if _embedding_client is None:
        # 可以通过环境变量配置服务器地址
        import os
        server_url = os.getenv('EMBEDDING_SERVER_URL', 'http://9.130.192.143:8000')
        _embedding_client = EmbeddingClient(server_url)
    return _embedding_client

def compute_cosine_similarity(text1: str, text2: str) -> float:
    """计算两个文本之间的余弦相似度"""
    client = get_embedding_client()
    # 如果 text1 或 text2 是 list，那么取第一个元素
    if isinstance(text1, list):
        text1 = text1[0] if text1 else ""
    if isinstance(text2, list):
        text2 = text2[0] if text2 else ""
    
    # 确保输入是字符串
    text1 = str(text1) if text1 is not None else ""
    text2 = str(text2) if text2 is not None else ""
    
    # 使用服务端的相似度计算接口
    result = client.compute_similarity([text1], [text2])
    if result and result.get('similarities') and len(result['similarities']) > 0:
        # 返回scores中的第一个值
        return float(result['similarities'][0])
    else:
        # 如果服务调用失败，返回0
        print("警告: embedding服务调用失败，返回相似度0")
        return 0.0

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

def compute_content_overlap_reward(
    solution_str: str, 
    final_answer: str, 
    ground_truth,
    similarity_threshold: float = 0.5,
    positive_reward: float = 0.4,
    negative_reward: float = -0.2
) -> float:
    """
    计算内容重叠度奖励
    
    Args:
        solution_str: 完整的解答文本
        final_answer: 最终答案
        ground_truth: 正确答案
        similarity_threshold: 相似度阈值
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
        
        if judge_value.lower() == "yes":
            if is_answer_correct:
                # 如果judge=Yes且最终答案正确，直接给正奖励，不需要计算相似度
                total_reward += positive_reward
            else:
                # judge=Yes但答案错误，需要计算相似度来判断信息是否真的有用
                similarity = compute_cosine_similarity(ground_truth["target"], info_content)
                is_high_similarity = similarity >= similarity_threshold
                
                if is_high_similarity:
                    # 信息与答案相似但答案错误，说明判断可能对但答案生成有问题，给小的负奖励
                    total_reward += negative_reward * 0.5
                else:
                    # 信息与答案不相似且答案错误，说明判断错误，给负奖励
                    total_reward += negative_reward
                    
        elif judge_value.lower() == "no":
            # 对于judge=No的情况，计算相似度
            similarity = compute_cosine_similarity(ground_truth["target"],info_content)
            is_high_similarity = similarity >= similarity_threshold
            
            if is_high_similarity:
                # 错误判断：说没用但实际有用
                total_reward += negative_reward
            else:
                # 正确判断：说没用且确实没用
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
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

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


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0, 
                 enable_content_overlap=True, similarity_threshold=0.5, 
                 overlap_positive_reward=0.1, overlap_negative_reward=-0.05):
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

    # 计算内容重叠度奖励
    content_overlap_reward = 0.0
    if enable_content_overlap and answer is not None:
        content_overlap_reward = compute_content_overlap_reward(
            solution_str=solution_str,
            final_answer=answer,
            ground_truth=ground_truth,
            similarity_threshold=similarity_threshold,
            positive_reward=overlap_positive_reward,
            negative_reward=overlap_negative_reward
        )
        
        if do_print:
            print(f"Content overlap reward: {content_overlap_reward}")

    # 返回字典格式，包含详细信息
    return base_score + content_overlap_reward
    # {
    #     "score": base_score + content_overlap_reward,
    #     "base_score": base_score,
    #     "content_overlap_reward": content_overlap_reward,
    #     "em_correct": base_score > 0,
    #     "has_answer": answer is not None
    # }


def compute_score_subem(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0,
                       enable_content_overlap=True, similarity_threshold=0.5,
                       overlap_positive_reward=0.1, overlap_negative_reward=-0.05):
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

    # 计算内容重叠度奖励
    content_overlap_reward = 0.0
    if enable_content_overlap and answer is not None:
        content_overlap_reward = compute_content_overlap_reward(
            solution_str=solution_str,
            final_answer=answer,
            ground_truth=ground_truth,
            similarity_threshold=similarity_threshold,
            positive_reward=overlap_positive_reward,
            negative_reward=overlap_negative_reward
        )
        
        if do_print:
            print(f"Content overlap reward: {content_overlap_reward}")

    # 返回字典格式，包含详细信息
    return base_score + content_overlap_reward
    # {
    #     "score": base_score + content_overlap_reward,
    #     "base_score": base_score,
    #     "content_overlap_reward": content_overlap_reward,
    #     "subem_correct": base_score > 0,
    #     "has_answer": answer is not None
    # }

def compute_batch_cosine_similarity(text_pairs: List[Tuple[str, str]]) -> List[float]:
    """批量计算多对文本之间的余弦相似度"""
    if not text_pairs:
        return []
    
    client = get_embedding_client()
    
    # 分离查询和文档
    queries = []
    documents = []
    
    for text1, text2 in text_pairs:
        # 处理 list 类型的输入
        if isinstance(text1, list):
            text1 = text1[0] if text1 else ""
        if isinstance(text2, list):
            text2 = text2[0] if text2 else ""
        queries.append(str(text1))
        documents.append(str(text2))
    
    # 批量计算相似度
    result = client.compute_similarity(queries, documents)
    if result and result.get('data'):
        # 直接返回scores列表（每个元素对应一个query-document对的相似度）
        similarities = [item['score'] for item in result['data']]
        
        # 确保返回的长度与输入匹配
        expected_length = len(text_pairs)
        if len(similarities) != expected_length:
            print(f"警告: embedding服务返回长度不匹配，期望{expected_length}，实际{len(similarities)}")
            if len(similarities) < expected_length:
                # 不足的部分填充0.0
                similarities.extend([0.0] * (expected_length - len(similarities)))
            else:
                # 多余的部分截断
                similarities = similarities[:expected_length]
        
        return similarities
    else:
        # 如果服务调用失败，返回0列表
        print("警告: embedding服务调用失败，返回相似度0列表")
        return [0.0] * len(text_pairs)

def compute_batch_content_overlap_reward(
    solution_strs: List[str], 
    final_answers: List[str], 
    ground_truths: List[dict],
    similarity_threshold: float = 0.5,
    positive_reward: float = 0.4,
    negative_reward: float = -0.2,
    similarity_batch_size: int = 32
) -> List[float]:
    """
    批量计算内容重叠度奖励
    
    Args:
        solution_strs: 完整的解答文本列表
        final_answers: 最终答案列表
        ground_truths: 正确答案列表
        similarity_threshold: 相似度阈值
        positive_reward: 正确判断的奖励
        negative_reward: 错误判断的惩罚
        similarity_batch_size: 相似度计算的批量大小，避免显存爆炸
    
    Returns:
        内容重叠度奖励分数列表
    """
    if not solution_strs or not final_answers or not ground_truths:
        return [0.0] * len(solution_strs)
    
    batch_size = len(solution_strs)
    rewards = [0.0] * batch_size
    
    # 收集所有需要计算相似度的文本对
    similarity_pairs = []
    similarity_indices = []  # 记录每个相似度计算对应的样本索引和信息索引
    
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
        
        for info_idx, (info_content, judge_value) in enumerate(info_judge_pairs):
            if not info_content:
                continue
            
            if judge_value.lower() == "yes":
                if is_answer_correct:
                    # 如果judge=Yes且最终答案正确，直接给正奖励，不需要计算相似度
                    rewards[batch_idx] += positive_reward
                else:
                    # judge=Yes但答案错误，需要计算相似度来判断信息是否真的有用
                    similarity_pairs.append((ground_truth["target"], info_content))
                    similarity_indices.append((batch_idx, info_idx, "yes_wrong_answer"))
                    
            elif judge_value.lower() == "no":
                # 对于judge=No的情况，计算相似度
                similarity_pairs.append((ground_truth["target"], info_content))
                similarity_indices.append((batch_idx, info_idx, "no"))
    
    # 分批次计算相似度，避免显存爆炸
    if similarity_pairs:
        all_similarities = []
        
        for i in range(0, len(similarity_pairs), similarity_batch_size):
            batch_pairs = similarity_pairs[i:i + similarity_batch_size]
            batch_similarities = compute_batch_cosine_similarity(batch_pairs)
            
            # 检查返回的相似度数量是否与输入对数匹配
            if len(batch_similarities) != len(batch_pairs):
                print(f"警告: 相似度计算返回长度不匹配，期望{len(batch_pairs)}，实际{len(batch_similarities)}")
                # 填充或截断到正确长度
                if len(batch_similarities) < len(batch_pairs):
                    batch_similarities.extend([0.0] * (len(batch_pairs) - len(batch_similarities)))
                else:
                    batch_similarities = batch_similarities[:len(batch_pairs)]
            
            all_similarities.extend(batch_similarities)
        
        # 最终检查总长度是否匹配
        if len(all_similarities) != len(similarity_indices):
            print(f"警告: 总相似度长度不匹配，期望{len(similarity_indices)}，实际{len(all_similarities)}")
            # 确保长度匹配
            if len(all_similarities) < len(similarity_indices):
                all_similarities.extend([0.0] * (len(similarity_indices) - len(all_similarities)))
            else:
                all_similarities = all_similarities[:len(similarity_indices)]
        
        # 根据相似度结果更新奖励
        for sim_idx, similarity in enumerate(all_similarities):
            if sim_idx >= len(similarity_indices):
                print(f"警告: 跳过索引{sim_idx}，超出similarity_indices范围")
                break
                
            batch_idx, info_idx, judge_type = similarity_indices[sim_idx]
            is_high_similarity = similarity >= similarity_threshold
            
            if judge_type == "yes_wrong_answer":
                if is_high_similarity:
                    # 信息与答案相似但答案错误，说明判断可能对但答案生成有问题，给小的负奖励
                    rewards[batch_idx] += negative_reward * 0.5
                else:
                    # 信息与答案不相似且答案错误，说明判断错误，给负奖励
                    rewards[batch_idx] += negative_reward
            elif judge_type == "no":
                if is_high_similarity:
                    # 错误判断：说没用但实际有用
                    rewards[batch_idx] += negative_reward
                else:
                    # 正确判断：说没用且确实没用
                    rewards[batch_idx] += positive_reward * 0.2
    
    return rewards


def compute_batch_score(solution_strs: List[str], ground_truths: List[dict], 
                       method="strict", format_score=0.0, score=1.0, 
                       enable_content_overlap=True, similarity_threshold=0.5, 
                       overlap_positive_reward=0.1, overlap_negative_reward=-0.05,
                       similarity_batch_size=16) -> List[float]:
    """批量版本的scoring函数，用于exact match (EM) with content overlap reward.

    Args:
        solution_strs: 解答文本列表
        ground_truths: 正确答案列表
        method: 提取解答的方法，选择 'strict' 或 'flexible'
        format_score: 格式分数
        score: 正确答案的分数
        enable_content_overlap: 是否启用内容重叠度评分
        similarity_threshold: 高相似度判断阈值
        overlap_positive_reward: 正确判断的奖励
        overlap_negative_reward: 错误判断的惩罚
        similarity_batch_size: 相似度计算的批量大小，避免显存爆炸
    
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
                similarity_threshold=similarity_threshold,
                positive_reward=overlap_positive_reward,
                negative_reward=overlap_negative_reward,
                similarity_batch_size=similarity_batch_size
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
                             enable_content_overlap=True, similarity_threshold=0.5,
                             overlap_positive_reward=0.1, overlap_negative_reward=-0.05,
                             similarity_batch_size=16) -> List[float]:
    """批量版本的scoring函数，用于substring exact match (EM) with content overlap reward.

    Args:
        solution_strs: 解答文本列表
        ground_truths: 正确答案列表
        method: 提取解答的方法，选择 'strict' 或 'flexible'
        format_score: 格式分数
        score: 正确答案的分数
        enable_content_overlap: 是否启用内容重叠度评分
        similarity_threshold: 高相似度判断阈值
        overlap_positive_reward: 正确判断的奖励
        overlap_negative_reward: 错误判断的惩罚
        similarity_batch_size: 相似度计算的批量大小，避免显存爆炸
    
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
                similarity_threshold=similarity_threshold,
                positive_reward=overlap_positive_reward,
                negative_reward=overlap_negative_reward,
                similarity_batch_size=similarity_batch_size
            )
            
            # 将结果映射回原始索引
            for i, valid_idx in enumerate(valid_indices):
                content_overlap_rewards[valid_idx] = valid_rewards[i]
    
    # 合并基础分数和内容重叠奖励
    for i in range(batch_size):
        final_score = base_scores[i] + content_overlap_rewards[i]
        batch_scores.append(final_score)
    
    return batch_scores
