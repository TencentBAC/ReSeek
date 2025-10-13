# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                num_examine=self.num_examine,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


@register("batch_naive")
class BatchNaiveRewardManager:
    """批量化的 reward manager，提高 GPU 利用率"""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        初始化 BatchNaiveRewardManager 实例。

        Args:
            tokenizer: 用于将 token IDs 解码为文本的 tokenizer。
            num_examine: 用于调试目的打印到控制台的解码响应批次数。
            compute_score: 计算奖励分数的函数。如果为 None，将使用 `default_compute_score`。
            reward_fn_key: 用于访问非张量批次数据中数据源的键。默认为 "data_source"。
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        """批量处理 reward 计算"""

        # 如果已有 rm score，直接返回
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # 批量解码所有 prompts 和 responses
        batch_size = len(data)
        solution_strs = []
        ground_truths = []
        data_sources = []
        extra_infos = []
        valid_response_lengths = []

        for i in range(batch_size):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            valid_response_lengths.append(valid_response_length)

            # 解码
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            solution_strs.append(response_str)
            ground_truths.append(data_item.non_tensor_batch["reward_model"]["ground_truth"])
            data_sources.append(data_item.non_tensor_batch[self.reward_fn_key])
            
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns
            extra_infos.append(extra_info)

        scores = self._compute_batch_scores(solution_strs, ground_truths, data_sources, extra_infos)

        # 处理分数并填充 reward_tensor
        already_print_data_sources = {}
        
        for i in range(batch_size):
            score = scores[i]
            
            if isinstance(score, dict):
                reward = score["score"]
                # 存储包括原始奖励在内的信息
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_lengths[i] - 1] = reward

            # 打印调试信息
            data_source = data_sources[i]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                # 重新解码用于显示（可以优化）
                data_item = data[i]
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                
                print("[prompt]", prompt_str)
                print("[response]", solution_strs[i])
                print("[ground_truth]", ground_truths[i])
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def _compute_batch_scores(self, solution_strs, ground_truths, data_sources, extra_infos):
        """执行批量分数计算"""
        # 这里需要根据具体的计算函数进行适配
        # 假设我们有批量版本的计算函数
        
        # 导入批量计算函数
        try:
            from verl.utils.reward_score.search_r1_like_qa_em_s3 import compute_batch_score, compute_batch_score_subem
            
            # 根据数据源类型选择合适的批量计算函数
            first_data_source = data_sources[0]
            
            if "subem" in first_data_source:
                scores = compute_batch_score_subem(
                    solution_strs=solution_strs,
                    ground_truths=ground_truths,
                    # 可以从 extra_info 中获取其他参数
                    enable_content_overlap=True,
                    similarity_threshold=0.5,
                    overlap_positive_reward=0.1,
                    overlap_negative_reward=-0.05
                )
            else:
                scores = compute_batch_score(
                    solution_strs=solution_strs,
                    ground_truths=ground_truths,
                    # 可以从 extra_info 中获取其他参数
                    enable_content_overlap=True,
                    similarity_threshold=0.5,
                    overlap_positive_reward=0.1,
                    overlap_negative_reward=-0.05
                )
            
            return scores
            
        except ImportError:
            # 如果导入失败，回退到逐个计算
            scores = []
            for i in range(len(solution_strs)):
                score = self.compute_score(
                    data_source=data_sources[i],
                    solution_str=solution_strs[i],
                    ground_truth=ground_truths[i],
                    extra_info=extra_infos[i],
                    num_examine=self.num_examine,
                )
                scores.append(score)
            return scores
