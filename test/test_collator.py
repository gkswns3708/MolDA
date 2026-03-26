"""Tests for TrainCollator and EvalCollator in src/data/collator.py."""

import pytest
import torch

from src.data.collator import TrainCollator, EvalCollator, EOS_TOKEN_ID, PAD_TOKEN_ID


def _make_samples(n=3, with_graph=False):
    """Create synthetic samples mimicking dataset format."""
    samples = []
    for i in range(n):
        prompt = f"<INSTRUCTION>Predict the property.</INSTRUCTION>"
        if with_graph:
            prompt += " <GRAPH>graph data here</GRAPH>"
        prompt += f" Input molecule {i}."
        samples.append({
            "prompt_text": prompt,
            "target_text": f"<BOOLEAN> True </BOOLEAN>",
            "task": f"task_{i}",
            "input_mol_string": f"<mol>mol_{i}",
        })
    return samples


class TestTrainCollator:

    def test_output_shapes(self, real_tokenizer):
        max_length = 128
        collator = TrainCollator(real_tokenizer, max_length=max_length)
        batch = collator(_make_samples(3))
        assert batch["input_ids"].shape == (3, max_length)
        assert batch["labels"].shape == (3, max_length)
        assert batch["attention_mask"].shape == (3, max_length)
        assert batch["prompt_lengths"].shape == (3,)

    def test_right_padding_with_eos(self, real_tokenizer):
        collator = TrainCollator(real_tokenizer, max_length=256)
        batch = collator(_make_samples(1))
        ids = batch["input_ids"][0]
        attn = batch["attention_mask"][0]
        # Find where padding starts
        pad_start = attn.sum().item()
        if pad_start < len(ids):
            assert (ids[pad_start:] == EOS_TOKEN_ID).all()

    def test_labels_prompt_masked(self, real_tokenizer):
        collator = TrainCollator(real_tokenizer, max_length=256)
        batch = collator(_make_samples(1))
        prompt_len = batch["prompt_lengths"][0].item()
        labels = batch["labels"][0]
        assert (labels[:prompt_len] == -100).all()

    def test_labels_answer_has_token_ids(self, real_tokenizer):
        collator = TrainCollator(real_tokenizer, max_length=256)
        batch = collator(_make_samples(1))
        prompt_len = batch["prompt_lengths"][0].item()
        labels = batch["labels"][0]
        input_ids = batch["input_ids"][0]
        # Answer positions (non-padding) should match input_ids
        attn = batch["attention_mask"][0]
        answer_end = attn.sum().item()
        if prompt_len < answer_end:
            assert torch.equal(labels[prompt_len:answer_end], input_ids[prompt_len:answer_end])

    def test_graph_tag_removal(self, real_tokenizer):
        collator = TrainCollator(real_tokenizer, mol_representation="string_only", max_length=256)
        samples = _make_samples(1, with_graph=True)
        batch = collator(samples)
        # Decode and check no GRAPH tag in output
        decoded = real_tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False)
        assert "<GRAPH>" not in decoded
        assert "</GRAPH>" not in decoded

    def test_attention_mask_correct(self, real_tokenizer):
        collator = TrainCollator(real_tokenizer, max_length=256)
        batch = collator(_make_samples(2))
        for i in range(2):
            attn = batch["attention_mask"][i]
            ids = batch["input_ids"][i]
            real_len = attn.sum().item()
            # All 1s should be contiguous at the start (right-padded)
            assert (attn[:real_len] == 1).all()
            assert (attn[real_len:] == 0).all()

    def test_tasks_preserved(self, real_tokenizer):
        collator = TrainCollator(real_tokenizer, max_length=256)
        samples = _make_samples(3)
        batch = collator(samples)
        assert batch["tasks"] == [s["task"] for s in samples]

    def test_truncation(self, real_tokenizer):
        collator = TrainCollator(real_tokenizer, max_length=16)  # Very short
        samples = _make_samples(1)
        batch = collator(samples)
        assert batch["input_ids"].shape[1] == 16


class TestEvalCollator:

    def test_output_shapes(self, real_tokenizer):
        collator = EvalCollator(real_tokenizer, max_length=256)
        batch = collator(_make_samples(3))
        B = 3
        assert batch["prompt_input_ids"].shape[0] == B
        assert batch["prompt_attention_mask"].shape[0] == B
        assert len(batch["tasks"]) == B
        assert len(batch["target_texts"]) == B

    def test_left_padding_with_pad(self, real_tokenizer):
        collator = EvalCollator(real_tokenizer, max_length=256)
        # Create samples with different prompt lengths
        samples = [
            {"prompt_text": "Short prompt.", "target_text": "yes", "task": "t1", "input_mol_string": ""},
            {"prompt_text": "A much longer prompt with more words to make it longer.", "target_text": "no", "task": "t2", "input_mol_string": ""},
        ]
        batch = collator(samples)
        ids = batch["prompt_input_ids"]
        # Shorter prompt should have PAD at the beginning
        shorter_idx = 0 if ids[0].ne(PAD_TOKEN_ID).sum() < ids[1].ne(PAD_TOKEN_ID).sum() else 1
        assert ids[shorter_idx][0] == PAD_TOKEN_ID

    def test_attention_mask_correct(self, real_tokenizer):
        collator = EvalCollator(real_tokenizer, max_length=256)
        batch = collator(_make_samples(2))
        for i in range(2):
            attn = batch["prompt_attention_mask"][i]
            ids = batch["prompt_input_ids"][i]
            # Left-padded: 0s then 1s
            first_one = (attn == 1).nonzero(as_tuple=True)[0]
            if len(first_one) > 0:
                start = first_one[0].item()
                assert (attn[:start] == 0).all()
                assert (attn[start:] == 1).all()

    def test_preserves_target_texts(self, real_tokenizer):
        collator = EvalCollator(real_tokenizer, max_length=256)
        samples = _make_samples(2)
        batch = collator(samples)
        assert batch["target_texts"] == [s["target_text"] for s in samples]

    def test_graph_tag_removal(self, real_tokenizer):
        collator = EvalCollator(real_tokenizer, mol_representation="string_only", max_length=256)
        samples = _make_samples(1, with_graph=True)
        batch = collator(samples)
        decoded = real_tokenizer.decode(batch["prompt_input_ids"][0], skip_special_tokens=False)
        assert "<GRAPH>" not in decoded

    def test_preserves_tasks(self, real_tokenizer):
        collator = EvalCollator(real_tokenizer, max_length=256)
        samples = _make_samples(3)
        batch = collator(samples)
        assert batch["tasks"] == [s["task"] for s in samples]
