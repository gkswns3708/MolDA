"""Tests for TrainCollator and EvalCollator in src/data/collator.py."""

import pytest
import torch

from src.data.collator import TrainCollator, EvalCollator


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


def _real_len(sample, prompt_len, tokenizer, collator):
    """Compute actual content length (prompt + target + EOS) matching collator logic.

    LLaDA TrainCollator uses all-1 attention_mask (padding EOS is part of the answer),
    so attention_mask cannot be used to find sequence end. Reconstruct from inputs instead.
    """
    prompt_text = sample["prompt_text"]
    if collator.mol_representation == "string_only":
        from src.data.collator import GRAPH_PATTERN
        prompt_text = GRAPH_PATTERN.sub("", prompt_text)
    target_ids = tokenizer.encode(sample["target_text"], add_special_tokens=False)
    real_len = prompt_len + len(target_ids) + 1  # +1 for trailing EOS
    return min(real_len, collator.max_length)


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
        samples = _make_samples(1)
        batch = collator(samples)
        ids = batch["input_ids"][0]
        eos_id = real_tokenizer.eos_token_id
        prompt_len = batch["prompt_lengths"][0].item()
        pad_start = _real_len(samples[0], prompt_len, real_tokenizer, collator)
        if pad_start < len(ids):
            assert (ids[pad_start:] == eos_id).all()

    def test_eos_matches_tokenizer(self, real_tokenizer):
        """Regression: collator EOS must match tokenizer.eos_token_id."""
        collator = TrainCollator(real_tokenizer, max_length=256)
        assert collator.eos_token_id == real_tokenizer.eos_token_id

    def test_labels_prompt_masked(self, real_tokenizer):
        collator = TrainCollator(real_tokenizer, max_length=256)
        batch = collator(_make_samples(1))
        prompt_len = batch["prompt_lengths"][0].item()
        labels = batch["labels"][0]
        assert (labels[:prompt_len] == -100).all()

    def test_labels_answer_has_token_ids(self, real_tokenizer):
        collator = TrainCollator(real_tokenizer, max_length=256)
        samples = _make_samples(1)
        batch = collator(samples)
        prompt_len = batch["prompt_lengths"][0].item()
        labels = batch["labels"][0]
        input_ids = batch["input_ids"][0]
        answer_end = _real_len(samples[0], prompt_len, real_tokenizer, collator)
        if prompt_len < answer_end:
            assert torch.equal(labels[prompt_len:answer_end], input_ids[prompt_len:answer_end])

    def test_eos_appended_after_target(self, real_tokenizer):
        """EOS token must be appended immediately after target (before padding)."""
        collator = TrainCollator(real_tokenizer, max_length=256)
        samples = _make_samples(1)
        batch = collator(samples)
        ids = batch["input_ids"][0]
        eos_id = real_tokenizer.eos_token_id
        prompt_len = batch["prompt_lengths"][0].item()
        target_ids = real_tokenizer.encode(samples[0]["target_text"], add_special_tokens=False)
        # Position immediately after target content must be the appended EOS
        eos_pos = prompt_len + len(target_ids)
        assert ids[eos_pos].item() == eos_id

    def test_eos_in_labels(self, real_tokenizer):
        """EOS token appended after target should be trainable (labels != -100)."""
        collator = TrainCollator(real_tokenizer, max_length=256)
        samples = _make_samples(1)
        batch = collator(samples)
        labels = batch["labels"][0]
        prompt_len = batch["prompt_lengths"][0].item()
        target_ids = real_tokenizer.encode(samples[0]["target_text"], add_special_tokens=False)
        eos_pos = prompt_len + len(target_ids)
        assert labels[eos_pos].item() != -100

    def test_graph_tag_removal(self, real_tokenizer):
        collator = TrainCollator(real_tokenizer, mol_representation="string_only", max_length=256)
        samples = _make_samples(1, with_graph=True)
        batch = collator(samples)
        # Decode and check no GRAPH tag in output
        decoded = real_tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False)
        assert "<GRAPH>" not in decoded
        assert "</GRAPH>" not in decoded

    def test_attention_mask_is_all_ones(self, real_tokenizer):
        """LLaDA contract: attention_mask is all-1 for TrainCollator.

        Unlike AR models, LLaDA masked diffusion trains on padding EOS to learn
        sentence termination, so attention flows to every position including padding.
        """
        collator = TrainCollator(real_tokenizer, max_length=256)
        batch = collator(_make_samples(2))
        attn = batch["attention_mask"]
        assert attn.shape == (2, 256)
        assert (attn == 1).all()

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
        pad_id = collator.pad_token_id
        # Create samples with different prompt lengths
        samples = [
            {"prompt_text": "Short prompt.", "target_text": "yes", "task": "t1", "input_mol_string": ""},
            {"prompt_text": "A much longer prompt with more words to make it longer.", "target_text": "no", "task": "t2", "input_mol_string": ""},
        ]
        batch = collator(samples)
        ids = batch["prompt_input_ids"]
        # Shorter prompt should have PAD at the beginning
        shorter_idx = 0 if ids[0].ne(pad_id).sum() < ids[1].ne(pad_id).sum() else 1
        assert ids[shorter_idx][0] == pad_id

    def test_pad_matches_tokenizer(self, real_tokenizer):
        """Regression: collator PAD must derive from tokenizer."""
        collator = EvalCollator(real_tokenizer, max_length=256)
        expected = real_tokenizer.pad_token_id if real_tokenizer.pad_token_id is not None else real_tokenizer.eos_token_id
        assert collator.pad_token_id == expected

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


def _make_graph_samples(n=2):
    """Synthetic samples with x/edge_index/edge_attr fields (Stage 2 graph batching)."""
    samples = []
    for i in range(n):
        n_nodes = 3 + i
        samples.append({
            "prompt_text": "<INSTRUCTION>Use <mol> as input.</INSTRUCTION>",
            "target_text": "<BOOLEAN> True </BOOLEAN>",
            "task": f"task_{i}",
            "input_mol_string": f"mol_{i}",
            "x": [[0] * 9 for _ in range(n_nodes)],
            "edge_index": [[0, 1], [1, 0]],
            "edge_attr": [[0] * 3, [0] * 3],
            "additional_x": [[0] * 9],
            "additional_edge_index": [[0], [0]],
            "additional_edge_attr": [[0] * 3],
        })
    return samples


class TestGraphBatching:
    """string+graph 모드에서 collator가 `graphs`/`additional_graphs`를 emit 하는지 검증."""

    def test_train_collator_emits_graphs(self, real_tokenizer):
        collator = TrainCollator(real_tokenizer, mol_representation="string+graph", max_length=128)
        batch = collator(_make_graph_samples(2))
        assert "graphs" in batch
        assert "additional_graphs" in batch
        graphs = batch["graphs"]
        # PyG Batch: total nodes = sum of per-sample n_nodes
        assert graphs.x.shape[0] == 3 + 4
        assert graphs.batch.unique().numel() == 2

    def test_string_only_does_not_emit_graphs(self, real_tokenizer):
        collator = TrainCollator(real_tokenizer, mol_representation="string_only", max_length=128)
        batch = collator(_make_graph_samples(2))
        assert "graphs" not in batch
        assert "additional_graphs" not in batch

    def test_eval_collator_emits_graphs(self, real_tokenizer):
        collator = EvalCollator(real_tokenizer, mol_representation="string+graph", max_length=128)
        batch = collator(_make_graph_samples(2))
        assert "graphs" in batch
        assert "additional_graphs" in batch
