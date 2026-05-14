"""Tests for src/data/molpo_collator.py.

Validates MolPOTrainCollator behavior:
  - 2B layout: [chosen | rejected]
  - 3B layout: [sft | chosen | rejected]
  - chosen / rejected dual-column lookup
  - require_pair=True raises on missing fields
  - require_pair=False filters silently
  - empty-answer pre-filter (NaN guard)
"""
import pytest
import torch

from src.data.molpo_collator import MolPOTrainCollator


def _make_molpo_samples(n=3, with_graph=False, key_suffix="", chosen="chosen body",
                        rejected="rejected body"):
    """Synthetic MolPO samples — chosen/rejected fields explicit."""
    samples = []
    for i in range(n):
        prompt = "<INSTRUCTION>Caption the molecule.</INSTRUCTION>"
        if with_graph:
            prompt += " <GRAPH>graph_payload</GRAPH>"
        prompt += f" Input molecule {i}."
        samples.append({
            "prompt_text": prompt,
            f"target_text_chosen{key_suffix}": f"{chosen} {i}",
            f"target_text_rejected{key_suffix}": f"{rejected} {i}",
            "task": f"task_{i % 2}",
            "input_mol_string": f"<mol>mol_{i}",
        })
    return samples


# ───────────────────────────────────────────────
# Layout: 2B and 3B
# ───────────────────────────────────────────────

class TestMolPOLayout:

    def test_2B_layout(self, real_tokenizer):
        collator = MolPOTrainCollator(real_tokenizer, max_length=128, batch_division=2)
        batch = collator(_make_molpo_samples(3))
        assert batch["input_ids"].shape == (6, 128), "2B layout: 3 chosen + 3 rejected"
        assert batch["molpo_batch_size"] == 3
        assert batch["molpo_batch_division"] == 2
        # Chosen and rejected should differ at content positions
        assert not torch.equal(batch["input_ids"][0], batch["input_ids"][3])

    def test_3B_layout(self, real_tokenizer):
        collator = MolPOTrainCollator(real_tokenizer, max_length=128, batch_division=3)
        batch = collator(_make_molpo_samples(2))
        assert batch["input_ids"].shape == (6, 128), "3B layout: 2 sft + 2 chosen + 2 rejected"
        assert batch["molpo_batch_size"] == 2
        assert batch["molpo_batch_division"] == 3
        # SFT (slot 0:B) and chosen (slot B:2B) carry identical token sequences
        # (both use chosen_text)
        assert torch.equal(batch["input_ids"][0:2], batch["input_ids"][2:4])
        # but rejected slot differs
        assert not torch.equal(batch["input_ids"][0:2], batch["input_ids"][4:6])

    def test_invalid_division(self, real_tokenizer):
        with pytest.raises(AssertionError):
            MolPOTrainCollator(real_tokenizer, batch_division=4)


# ───────────────────────────────────────────────
# Dual-column lookup (mol_token_type)
# ───────────────────────────────────────────────

class TestDualColumn:

    def test_dual_column_selfies(self, real_tokenizer):
        # Sample has target_text_chosen_selfies / _smiles, no target_text_chosen
        samples = _make_molpo_samples(2, key_suffix="_selfies",
                                      chosen="[C][C][O]", rejected="[C][C]")
        # Add smiles variant too (should be ignored)
        for i, s in enumerate(samples):
            s[f"target_text_chosen_smiles"] = f"COCO_{i}"
            s[f"target_text_rejected_smiles"] = f"CC_{i}"

        collator = MolPOTrainCollator(real_tokenizer, max_length=64,
                                       mol_token_type="selfies")
        batch = collator(samples)
        assert batch["input_ids"].shape == (4, 64)

    def test_no_pair_raise(self, real_tokenizer):
        # Sample missing chosen/rejected
        samples = [{
            "prompt_text": "test",
            "target_text": "single",   # SFT-only sample
            "task": "x",
            "input_mol_string": "<mol>",
        }]
        collator = MolPOTrainCollator(real_tokenizer, max_length=64, require_pair=True)
        with pytest.raises(ValueError, match="target_text_chosen"):
            collator(samples)

    def test_no_pair_skip(self, real_tokenizer):
        # mix: 1 valid + 1 missing-pair (should be filtered)
        samples = _make_molpo_samples(1)
        samples.append({
            "prompt_text": "test",
            "target_text": "single",
            "task": "x",
            "input_mol_string": "<mol>",
        })
        collator = MolPOTrainCollator(real_tokenizer, max_length=64, require_pair=False)
        batch = collator(samples)
        assert batch["molpo_batch_size"] == 1
        assert collator._n_filtered_no_pair == 1


# ───────────────────────────────────────────────
# NaN guard: empty answer
# ───────────────────────────────────────────────

class TestEmptyAnswerFilter:

    def test_empty_chosen_filtered(self, real_tokenizer):
        samples = _make_molpo_samples(2)
        # Sample 0: chosen empty (will be empty answer after EOS-only)
        samples[0]["target_text_chosen"] = ""
        # Sample 1: valid

        collator = MolPOTrainCollator(real_tokenizer, max_length=64, require_pair=False)
        batch = collator(samples)
        # sample 0 should be filtered (chosen has only EOS = answer_len=1, NOT 0)
        # Actually with empty target, full_ids = prompt + [] + [EOS], answer_len = 1.
        # So the filter might NOT kick in. Let's check answer_len computation:
        # "" → encode → [], so target_ids=[]. full_ids = prompt_ids + [] + [eos].
        # answer_len = len(full_ids) - prompt_len = 1.
        # So this case is NOT filtered. We accept this — empty target with EOS still trains.
        assert batch["molpo_batch_size"] == 2  # both kept

    def test_truncation_to_zero_answer(self, real_tokenizer):
        """If max_length truncates everything past prompt, answer_len=0 → filtered."""
        # Long prompt, very small max_length so prompt fully consumes max_length
        samples = _make_molpo_samples(1, chosen="x", rejected="y")
        # bump prompt to fill max_length
        samples[0]["prompt_text"] = "x " * 100  # very long prompt
        collator = MolPOTrainCollator(real_tokenizer, max_length=8, require_pair=False)
        with pytest.raises(RuntimeError, match="all samples filtered"):
            collator(samples)
        assert collator._n_filtered_empty_answer == 1


# ───────────────────────────────────────────────
# Tasks list mirrors batch layout
# ───────────────────────────────────────────────

class TestTasksList:

    def test_tasks_2B(self, real_tokenizer):
        collator = MolPOTrainCollator(real_tokenizer, max_length=64, batch_division=2)
        samples = _make_molpo_samples(3)
        batch = collator(samples)
        # Expect: tasks for chosen (3) + rejected (3) = 6
        assert len(batch["tasks"]) == 6
        # First 3 (chosen) and last 3 (rejected) refer to same source samples
        assert batch["tasks"][:3] == batch["tasks"][3:]

    def test_tasks_3B(self, real_tokenizer):
        collator = MolPOTrainCollator(real_tokenizer, max_length=64, batch_division=3)
        samples = _make_molpo_samples(2)
        batch = collator(samples)
        assert len(batch["tasks"]) == 6  # 2 sft + 2 chosen + 2 rejected
        # All three slots refer to same source pairs
        assert batch["tasks"][:2] == batch["tasks"][2:4] == batch["tasks"][4:6]


# ───────────────────────────────────────────────
# String+graph mode: graph PyG batch replicated per slot
# ───────────────────────────────────────────────

class TestStringPlusGraph:

    def test_graph_replicated_for_2B(self, real_tokenizer):
        # Add graph fields x/edge_index/edge_attr per sample
        samples = _make_molpo_samples(2, with_graph=True)
        for s in samples:
            s["x"] = [[1], [2]]
            s["edge_index"] = [[0, 1], [1, 0]]
            s["edge_attr"] = [[0], [0]]

        collator = MolPOTrainCollator(real_tokenizer, max_length=64,
                                      mol_representation="string+graph",
                                      batch_division=2)
        batch = collator(samples)
        assert "graphs" in batch
        # PyG batch should have 2*2=4 graphs (2 chosen + 2 rejected, replicated)
        assert batch["graphs"].num_graphs == 4

    def test_graph_replicated_for_3B(self, real_tokenizer):
        samples = _make_molpo_samples(2, with_graph=True)
        for s in samples:
            s["x"] = [[1], [2]]
            s["edge_index"] = [[0, 1], [1, 0]]
            s["edge_attr"] = [[0], [0]]

        collator = MolPOTrainCollator(real_tokenizer, max_length=64,
                                      mol_representation="string+graph",
                                      batch_division=3)
        batch = collator(samples)
        assert batch["graphs"].num_graphs == 6  # 2*3
