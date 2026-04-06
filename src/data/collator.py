"""
Data collators for MolDA training and evaluation.

TrainCollator: right-pad with EOS (tokenizer.eos_token_id), build labels with -100 for prompt
EvalCollator: left-pad with PAD (tokenizer.pad_token_id), return prompt-only for generation

Token IDs are derived from the tokenizer at init time — not hardcoded.

Reference: DATASET_SPEC.md, Old_MolDA/data_utils.py
"""

import logging
import re
from typing import List

import torch

logger = logging.getLogger(__name__)

GRAPH_PATTERN = re.compile(r"<GRAPH>.*?</GRAPH>", re.DOTALL)


class TrainCollator:
    """Collate training samples: prompt + target → right-padded, labels masked for prompt."""

    def __init__(self, tokenizer, mol_representation: str = "string_only",
                 max_length: int = 512):
        self.tokenizer = tokenizer
        self.mol_representation = mol_representation
        self.max_length = max_length
        self.eos_token_id = tokenizer.eos_token_id

    def __call__(self, batch: List[dict]) -> dict:
        input_ids_list = []
        labels_list = []
        prompt_lengths_list = []
        tasks = []

        for sample in batch:
            prompt_text = sample["prompt_text"]
            target_text = sample["target_text"]
            task = sample["task"]

            # String-only mode: remove <GRAPH>...</GRAPH> from prompt
            if self.mol_representation == "string_only":
                prompt_text = GRAPH_PATTERN.sub("", prompt_text)

            # Tokenize prompt and target separately (no special tokens)
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)

            # Concatenate: prompt + target + EOS (SFT convention, ref: SMDM/sft/)
            full_ids = prompt_ids + target_ids + [self.eos_token_id]
            if len(full_ids) > self.max_length:
                full_ids = full_ids[:self.max_length]
                # Adjust prompt_length if truncation cuts into target
                prompt_len = min(len(prompt_ids), self.max_length)
            else:
                prompt_len = len(prompt_ids)

            # Labels: -100 for prompt, actual token ids for answer
            labels = [-100] * prompt_len + full_ids[prompt_len:]

            input_ids_list.append(full_ids)
            labels_list.append(labels)
            prompt_lengths_list.append(prompt_len)
            tasks.append(task)

        # Right-pad to max_length with EOS
        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []

        for ids, labs in zip(input_ids_list, labels_list):
            pad_len = self.max_length - len(ids)
            padded_input_ids.append(ids + [self.eos_token_id] * pad_len)
            # LLaDA SFT: padding EOS도 answer region에 포함 (GUIDELINES.md line 80)
            padded_labels.append(labs + [self.eos_token_id] * pad_len)
            padded_attention_mask.append([1] * self.max_length)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "prompt_lengths": torch.tensor(prompt_lengths_list, dtype=torch.long),
            "tasks": tasks,
        }


class EvalCollator:
    """Collate eval samples: prompt-only, left-padded with PAD for generation."""

    def __init__(self, tokenizer, mol_representation: str = "string_only",
                 max_length: int = 512):
        self.tokenizer = tokenizer
        self.mol_representation = mol_representation
        self.max_length = max_length
        # pad_token_id가 없으면 eos_token_id 사용 (LLaDA 기본 동작)
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, batch: List[dict]) -> dict:
        prompt_ids_list = []
        tasks = []
        target_texts = []
        input_mol_strings = []

        for sample in batch:
            prompt_text = sample["prompt_text"]

            # String-only mode: remove <GRAPH>...</GRAPH>
            if self.mol_representation == "string_only":
                prompt_text = GRAPH_PATTERN.sub("", prompt_text)

            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

            # Truncate prompt if needed
            if len(prompt_ids) > self.max_length:
                prompt_ids = prompt_ids[:self.max_length]

            prompt_ids_list.append(prompt_ids)
            tasks.append(sample["task"])
            target_texts.append(sample["target_text"])
            input_mol_strings.append(sample.get("input_mol_string", ""))

        # Left-pad to max prompt length with PAD
        max_prompt_len = max(len(ids) for ids in prompt_ids_list)

        padded_prompt_ids = []
        padded_attention_mask = []

        for ids in prompt_ids_list:
            pad_len = max_prompt_len - len(ids)
            padded_prompt_ids.append([self.pad_token_id] * pad_len + ids)
            padded_attention_mask.append([0] * pad_len + [1] * len(ids))

        return {
            "prompt_input_ids": torch.tensor(padded_prompt_ids, dtype=torch.long),
            "prompt_attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "tasks": tasks,
            "target_texts": target_texts,
            "input_mol_strings": input_mol_strings,
        }
