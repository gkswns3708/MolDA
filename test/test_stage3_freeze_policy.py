"""Test Stage 3 freeze policy: LLM backbone 외 전부 trainable 검증.

사용자 결정 (2026-05-10):
  Stage 3 V-MolPO 에서 LLM base weights 만 frozen,
  나머지(LoRA, embed_new, lm_head, GNN, Q-Former, query_tokens, opt_proj, ln_graph) 는
  모두 trainable. ref_model 은 별개로 항상 frozen.

이 테스트는 실제 모델을 로드하지 않고 freeze policy 의 if/elif 분기 로직만
parameter 이름 패턴에 대해 시뮬레이션해서 검증.
"""
import pytest
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────
# 실제 PEFT-wrapped + GNN + QFormer 모델의 parameter 이름 샘플
# (수동으로 정리한 대표 케이스)
# ─────────────────────────────────────────────────────────────────

EXPECTED_TRAINABLE = [
    # LoRA adapters
    "llada._model.base_model.model.model.transformer.blocks.0.q_proj.lora_A.default.weight",
    "llada._model.base_model.model.model.transformer.blocks.0.q_proj.lora_B.default.weight",
    "llada._model.base_model.model.model.transformer.blocks.5.up_proj.lora_A.default.weight",
    "llada._model.base_model.model.model.transformer.blocks.31.gate_proj.lora_B.default.weight",

    # PEFT-wrapped wte / lm_head (modules_to_save.default = trainable copy)
    "llada._model.base_model.model.model.transformer.wte.modules_to_save.default.weight",
    "llada._model.base_model.model.lm_head.modules_to_save.default.weight",

    # GNN encoder (Stage 3 + tune_gnn=true)
    "gnn.graph_encoder_gine.atom_encoder.atom_embedding_list.0.weight",
    "gnn.graph_encoder_gine.gnns.0.edge_embedding1.weight",
    "gnn.graph_encoder_gine.x_embedding1.weight",
    "gnn.graph_encoder_tokengt.encoder.layer.0.attention.self.query.weight",
    "gnn.fusion.weight",

    # Q-Former bridge
    "qformer.Qformer.bert.encoder.layer.0.attention.self.query.weight",
    "qformer.Qformer.bert.embeddings.word_embeddings.weight",
    "qformer.Qformer.bert.embeddings.LayerNorm.weight",
    "qformer.query_tokens",
    "qformer.opt_proj.weight",
    "qformer.opt_proj.bias",
    "qformer.ln_graph.weight",
    "qformer.ln_graph.bias",
]

EXPECTED_FROZEN = [
    # LLaDA base weights (LoRA target 아님)
    "llada._model.base_model.model.model.transformer.blocks.0.q_proj.base_layer.weight",
    "llada._model.base_model.model.model.transformer.blocks.0.attn_norm.weight",
    "llada._model.base_model.model.model.transformer.blocks.0.ff_norm.weight",
    "llada._model.base_model.model.model.transformer.ln_f.weight",
    "llada._model.base_model.model.model.transformer.blocks.0.attn_out.weight",
    "llada._model.base_model.model.model.transformer.blocks.0.ff_out.weight",
    "llada._model.base_model.model.model.transformer.blocks.0.ff_proj.weight",

    # PEFT-wrapped 의 frozen 원본 (original_module — modules_to_save 의 사본 아님)
    "llada._model.base_model.model.model.transformer.wte.original_module.weight",
    "llada._model.base_model.model.lm_head.original_module.weight",

    # Reference model (V-MolPO) — 항상 frozen
    "ref_model.molda.llada._model.base_model.model.model.transformer.blocks.0.q_proj.lora_A.default.weight",
    "ref_model.molda.gnn.graph_encoder_gine.atom_encoder.atom_embedding_list.0.weight",
    "ref_model.molda.qformer.query_tokens",
]


# ─────────────────────────────────────────────────────────────────
# Stage 3 freeze logic (simulated — mirrors molda._apply_stage_freeze_policy stage 3 분기)
# ─────────────────────────────────────────────────────────────────

def _stage3_decide(name: str, tune_gnn: bool) -> bool:
    """Return whether parameter `name` should be trainable in Stage 3.
    Mirrors the if/elif chain in MolDA._apply_stage_freeze_policy.
    """
    if name.startswith("ref_model."):
        return False
    if name.startswith("gnn.") or name.startswith("qformer."):
        return tune_gnn
    if "lora_" in name or ".modules_to_save." in name:
        return True
    return False


# ─────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────

class TestStage3TuneGnnTrue:
    """기본값 (tune_gnn=true): LLM backbone 외 모두 trainable."""

    def test_all_expected_trainable_are_trainable(self):
        for name in EXPECTED_TRAINABLE:
            assert _stage3_decide(name, tune_gnn=True), (
                f"Expected trainable but got frozen: {name}"
            )

    def test_all_expected_frozen_are_frozen(self):
        for name in EXPECTED_FROZEN:
            assert not _stage3_decide(name, tune_gnn=True), (
                f"Expected frozen but got trainable: {name}"
            )

    def test_user_intent_summary(self):
        """LLM backbone 외 전부 trainable 라는 사용자 결정의 핵심 종류 cover."""
        # LoRA → trainable
        assert _stage3_decide("X.q_proj.lora_A.default.weight", tune_gnn=True)
        # embed_new (PEFT modules_to_save) → trainable
        assert _stage3_decide("X.wte.modules_to_save.default.weight", tune_gnn=True)
        # lm_head (PEFT modules_to_save) → trainable
        assert _stage3_decide("X.lm_head.modules_to_save.default.weight", tune_gnn=True)
        # GNN atom embeddings → trainable
        assert _stage3_decide("gnn.graph_encoder_gine.atom_encoder.atom_embedding_list.0.weight",
                              tune_gnn=True)
        # Q-Former → trainable
        assert _stage3_decide("qformer.Qformer.bert.encoder.layer.0.weight", tune_gnn=True)
        # query_tokens → trainable
        assert _stage3_decide("qformer.query_tokens", tune_gnn=True)
        # opt_proj → trainable
        assert _stage3_decide("qformer.opt_proj.weight", tune_gnn=True)
        # ln_graph → trainable
        assert _stage3_decide("qformer.ln_graph.weight", tune_gnn=True)
        # LLaDA base weights → frozen
        assert not _stage3_decide("X.attn_norm.weight", tune_gnn=True)
        assert not _stage3_decide("X.q_proj.base_layer.weight", tune_gnn=True)
        # ref_model → always frozen
        assert not _stage3_decide("ref_model.molda.qformer.query_tokens", tune_gnn=True)


class TestStage3TuneGnnFalse:
    """ablation (tune_gnn=false): GNN/Q-Former frozen, LoRA+embed 만 trainable."""

    def test_gnn_qformer_frozen_when_tune_gnn_false(self):
        for name in [
            "gnn.graph_encoder_gine.atom_encoder.atom_embedding_list.0.weight",
            "qformer.query_tokens",
            "qformer.opt_proj.weight",
            "qformer.ln_graph.weight",
        ]:
            assert not _stage3_decide(name, tune_gnn=False), (
                f"Expected frozen with tune_gnn=False: {name}"
            )

    def test_lora_still_trainable_when_tune_gnn_false(self):
        assert _stage3_decide("X.q_proj.lora_A.default.weight", tune_gnn=False)
        assert _stage3_decide("X.wte.modules_to_save.default.weight", tune_gnn=False)


# ─────────────────────────────────────────────────────────────────
# Boundary: substring 'embed' 가 더 이상 GNN 잡지 않음 (Review C1 fix 검증)
# ─────────────────────────────────────────────────────────────────

class TestC1FixSubstringRegression:
    """이전 버그: STAGE3_LORA_TRAINABLE_KEYS 의 'embed' 가
    GNN.atom_embedding 까지 매치 → tune_gnn=False 인데 학습됨.
    Fix: prefix-based dispatch 로 변경. tune_gnn=False 일 때 GNN/Q-Former frozen 보장.
    """

    def test_gnn_embedding_not_matched_by_old_substring(self):
        """과거 버그 케이스: 'gnn.graph_encoder_gine.atom_encoder.atom_embedding_list.0.weight'
        에 'embed' 가 들어있어도 tune_gnn=False 면 frozen 이어야 함."""
        name = "gnn.graph_encoder_gine.atom_encoder.atom_embedding_list.0.weight"
        assert not _stage3_decide(name, tune_gnn=False), (
            "Review C1 regression: GNN atom_embedding should be frozen with tune_gnn=False"
        )

    def test_qformer_embeddings_not_matched_by_old_substring(self):
        name = "qformer.Qformer.bert.embeddings.word_embeddings.weight"
        assert not _stage3_decide(name, tune_gnn=False), (
            "Review C1 regression: QFormer embeddings should be frozen with tune_gnn=False"
        )
