"""MolPOTrainCollator: chosen/rejected pair → 2B (or 3B) batch for V-MolPO.

Layout (molpo.batch_division=2):
    output["input_ids"][:B]  = chosen   (y_w)
    output["input_ids"][B:]  = rejected (y_l)
    same for labels, attention_mask, tasks, graphs

Layout (molpo.batch_division=3):
    [0:B]    = sft (y_w, used for L_SFT branch)
    [B:2B]   = chosen   (y_w, used for L_pref/anchor)
    [2B:3B]  = rejected (y_l)
    (sft slice and chosen slice may carry identical tokenization but separate indices.)

Sample format expected from MoleculeDataset (when MolPO data available):
    sample["target_text_chosen"]     str
    sample["target_text_rejected"]   str
    (or dual-column variant: target_text_chosen_{selfies,smiles})

Falls back to standard SFT (target_text only) — caller must filter MolPO-only batches.

Pre-filter: skip samples whose chosen or rejected answer length is 0 after tokenization
(MICCAI MolPO 의 logps/rejected NaN 미로깅 issue 의 collator-side 방어).
"""
import logging
from typing import List

import torch

from src.data.collator import (
    GRAPH_PATTERN,
    _build_graph_batch,
)

logger = logging.getLogger(__name__)


def _get_chosen_text(sample: dict, mol_token_type: str | None = None):
    """Extract chosen target text, supporting both MolPO and dual-column formats."""
    if "target_text_chosen" in sample:
        return sample["target_text_chosen"]
    if mol_token_type and f"target_text_chosen_{mol_token_type}" in sample:
        return sample[f"target_text_chosen_{mol_token_type}"]
    return None


def _get_rejected_text(sample: dict, mol_token_type: str | None = None):
    if "target_text_rejected" in sample:
        return sample["target_text_rejected"]
    if mol_token_type and f"target_text_rejected_{mol_token_type}" in sample:
        return sample[f"target_text_rejected_{mol_token_type}"]
    return None


class MolPOTrainCollator:
    """Build a 2B (or 3B) batch from chosen/rejected sample pairs.

    Args:
        tokenizer:           HF tokenizer
        mol_representation:  "string_only" | "string+graph" | "graph_only"
        max_length:          token length cap (right-padded with EOS)
        batch_division:      2 (chosen, rejected) | 3 (sft, chosen, rejected)
        mol_token_type:      "selfies" | "smiles" — for dual-column lookup
        require_pair:        if True, raise on samples missing chosen/rejected fields
                             if False, skip them silently (count logged)
    """

    def __init__(
        self,
        tokenizer,
        mol_representation: str = "string_only",
        max_length: int = 512,
        batch_division: int = 2,
        mol_token_type: str | None = None,
        require_pair: bool = True,
    ):
        assert batch_division in (2, 3), (
            f"molpo batch_division must be 2 or 3, got {batch_division}"
        )
        self.tokenizer = tokenizer
        self.mol_representation = mol_representation
        self.max_length = max_length
        self.batch_division = batch_division
        self.mol_token_type = mol_token_type
        self.require_pair = require_pair
        self.eos_token_id = tokenizer.eos_token_id

        # Logged once per dataloader lifetime (reset by datamodule when needed)
        self._n_filtered_no_pair = 0
        self._n_filtered_empty_answer = 0

    def _tokenize_one(self, prompt_text: str, target_text: str):
        """Tokenize prompt + target → (input_ids, labels, attention_mask, prompt_len, answer_len)."""
        if self.mol_representation == "string_only":
            prompt_text = GRAPH_PATTERN.sub("", prompt_text)

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)
        full_ids = prompt_ids + target_ids + [self.eos_token_id]
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]
            prompt_len = min(len(prompt_ids), self.max_length)
        else:
            prompt_len = len(prompt_ids)

        labels = [-100] * prompt_len + full_ids[prompt_len:]
        answer_len = len(full_ids) - prompt_len
        return full_ids, labels, prompt_len, answer_len

    def _pad_one(self, ids: list, labels: list):
        pad_len = self.max_length - len(ids)
        ids_pad = ids + [self.eos_token_id] * pad_len
        labels_pad = labels + [self.eos_token_id] * pad_len
        attn = [1] * self.max_length
        return ids_pad, labels_pad, attn

    def __call__(self, batch: List[dict]) -> dict:
        chosen_ids, chosen_labels, chosen_prompt_lens = [], [], []
        rejected_ids, rejected_labels, rejected_prompt_lens = [], [], []
        kept_samples = []  # samples that survived filtering — used for graphs
        tasks = []

        for sample in batch:
            chosen_text = _get_chosen_text(sample, self.mol_token_type)
            rejected_text = _get_rejected_text(sample, self.mol_token_type)
            if chosen_text is None or rejected_text is None:
                if self.require_pair:
                    raise ValueError(
                        "MolPOTrainCollator requires `target_text_chosen` and "
                        "`target_text_rejected` (or dual-column variant) in each sample. "
                        "Got keys: " + ", ".join(sample.keys())
                    )
                self._n_filtered_no_pair += 1
                continue

            prompt_text = sample["prompt_text"]

            c_ids, c_labels, c_plen, c_alen = self._tokenize_one(prompt_text, chosen_text)
            r_ids, r_labels, r_plen, r_alen = self._tokenize_one(prompt_text, rejected_text)

            # NaN guard: pre-filter samples with empty answer on either side
            if c_alen <= 0 or r_alen <= 0:
                self._n_filtered_empty_answer += 1
                continue

            chosen_ids.append(c_ids)
            chosen_labels.append(c_labels)
            chosen_prompt_lens.append(c_plen)
            rejected_ids.append(r_ids)
            rejected_labels.append(r_labels)
            rejected_prompt_lens.append(r_plen)
            kept_samples.append(sample)
            tasks.append(sample["task"])

        if not kept_samples:
            raise RuntimeError(
                "MolPOTrainCollator: all samples filtered. "
                f"no_pair={self._n_filtered_no_pair}, "
                f"empty_answer={self._n_filtered_empty_answer}, "
                f"input batch size={len(batch)}"
            )

        B = len(kept_samples)
        out_ids, out_labels, out_attn = [], [], []
        out_prompt_lens = []

        if self.batch_division == 3:
            # Slot 1: SFT (uses chosen tokens) — same as chosen but kept separate to allow
            # Loss SFT to be computed on this slice independently
            for ids, labs in zip(chosen_ids, chosen_labels):
                p_ids, p_labs, p_attn = self._pad_one(ids, labs)
                out_ids.append(p_ids); out_labels.append(p_labs); out_attn.append(p_attn)
            out_prompt_lens.extend(chosen_prompt_lens)
            tasks_full = list(tasks)  # tasks for sft slice
        else:
            tasks_full = []

        # Slot 2 (or 1 if div=2): chosen
        for ids, labs in zip(chosen_ids, chosen_labels):
            p_ids, p_labs, p_attn = self._pad_one(ids, labs)
            out_ids.append(p_ids); out_labels.append(p_labs); out_attn.append(p_attn)
        out_prompt_lens.extend(chosen_prompt_lens)
        tasks_full.extend(tasks)

        # Slot 3 (or 2): rejected
        for ids, labs in zip(rejected_ids, rejected_labels):
            p_ids, p_labs, p_attn = self._pad_one(ids, labs)
            out_ids.append(p_ids); out_labels.append(p_labs); out_attn.append(p_attn)
        out_prompt_lens.extend(rejected_prompt_lens)
        tasks_full.extend(tasks)

        out = {
            "input_ids": torch.tensor(out_ids, dtype=torch.long),
            "attention_mask": torch.tensor(out_attn, dtype=torch.long),
            "labels": torch.tensor(out_labels, dtype=torch.long),
            "prompt_lengths": torch.tensor(out_prompt_lens, dtype=torch.long),
            "tasks": tasks_full,
            "molpo_batch_size": B,            # B (number of pairs)
            "molpo_batch_division": self.batch_division,  # 2 or 3
        }

        # Graphs: replicate per slot (same prompt → same graph for chosen/rejected/sft)
        if self.mol_representation in ("string+graph", "graph_only"):
            graph_samples = kept_samples * self.batch_division  # replicate
            graphs = _build_graph_batch(graph_samples)
            if graphs is not None:
                out["graphs"] = graphs

        return out

    def reset_filter_counts(self):
        self._n_filtered_no_pair = 0
        self._n_filtered_empty_answer = 0
