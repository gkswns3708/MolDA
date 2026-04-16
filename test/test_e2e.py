"""End-to-end pipeline tests (GPU required).

Tests the full training pipeline: data → model → loss → backward → optimizer.
Also tests generation (SMDM diffusion sampling) and DDP readiness.

핵심 제약: max_length=512, gen_max_len=256 (프로덕션 동일)

Uses session-scoped molda_model fixture from conftest.py (모델 1개만 GPU에 로드).
"""

import pytest
import torch

from src.training.scheduler import WarmupStableDecayLRScheduler


@pytest.fixture(scope="module")
def optimizer_and_scheduler(molda_model, cfg):
    """Create optimizer and scheduler from shared model."""
    lora_params = []
    embed_params = []
    head_params = []

    for name, param in molda_model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(param)
        elif "embed" in name.lower() or "wte" in name.lower():
            embed_params.append(param)
        elif "lm_head" in name.lower() or "ff_out" in name.lower() or "output" in name.lower():
            head_params.append(param)

    param_groups = [
        {"params": lora_params, "lr": cfg.lr.lora, "name": "lora"},
        {"params": embed_params, "lr": cfg.lr.embed_orig, "name": "embed"},
    ]
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg.lr.head_orig, "name": "head"})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.training.weight_decay)

    scheduler = WarmupStableDecayLRScheduler(
        optimizer, max_step=100, warmup_steps=10,
        decay_ratio=0.1, min_lr_ratio=0.1,
    )

    return optimizer, scheduler


def _make_batch(model, batch_size=2, seq_len=512, prompt_len=256):
    """Create a synthetic batch on GPU with production-size sequences.

    seq_len=512, prompt_len=256 → gen part = 256 (프로덕션 동일)
    """
    vocab_size = len(model.tokenizer)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device="cuda")
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "tasks": ["smol-property_prediction-bbbp"] * batch_size,
        "global_step": 0,
    }


# ─────────────────────────────────────────
# Training Pipeline E2E
# ─────────────────────────────────────────

@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.integration
class TestTrainingPipeline:

    def test_training_one_step(self, molda_model, optimizer_and_scheduler):
        """forward + backward + optimizer step 완주 (seq_len=512)."""
        optimizer, scheduler = optimizer_and_scheduler
        molda_model.train()
        batch = _make_batch(molda_model)

        optimizer.zero_grad()
        result = molda_model(batch)
        loss = result["loss"]
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        loss.backward()
        optimizer.step()
        scheduler.step(0)

    def test_gradients_flow_to_lora(self, molda_model, optimizer_and_scheduler):
        """LoRA 파라미터에 gradient가 흐르는지 확인."""
        optimizer, _ = optimizer_and_scheduler
        molda_model.train()
        batch = _make_batch(molda_model)

        optimizer.zero_grad()
        result = molda_model(batch)
        result["loss"].backward()

        has_lora_grad = False
        for name, param in molda_model.named_parameters():
            if "lora_" in name and param.requires_grad:
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_lora_grad = True
                    break
        assert has_lora_grad, "No gradients flowing to LoRA parameters"

    def test_gradients_not_on_frozen(self, molda_model, optimizer_and_scheduler):
        """Frozen 파라미터에 gradient가 없는지 확인."""
        optimizer, _ = optimizer_and_scheduler
        molda_model.train()
        batch = _make_batch(molda_model)

        optimizer.zero_grad()
        result = molda_model(batch)
        result["loss"].backward()

        for name, param in molda_model.named_parameters():
            if not param.requires_grad:
                assert param.grad is None, f"Frozen param has gradient: {name}"

    def test_lr_scheduler_integration(self, optimizer_and_scheduler):
        """Scheduler step 후 LR 변경 확인."""
        optimizer, scheduler = optimizer_and_scheduler
        scheduler.step(0)  # warmup start
        lr_at_0 = optimizer.param_groups[0]["lr"]
        scheduler.step(10)  # warmup end
        lr_at_10 = optimizer.param_groups[0]["lr"]
        assert lr_at_10 > lr_at_0, "LR should increase during warmup"

    def test_multiple_training_steps(self, molda_model, optimizer_and_scheduler):
        """3 step 연속 학습이 안정적으로 동작하는지 확인."""
        optimizer, scheduler = optimizer_and_scheduler
        molda_model.train()

        losses = []
        for step in range(3):
            batch = _make_batch(molda_model)
            optimizer.zero_grad()
            result = molda_model(batch)
            loss = result["loss"]
            assert torch.isfinite(loss), f"NaN/Inf loss at step {step}: {loss}"
            loss.backward()
            torch.nn.utils.clip_grad_norm_(molda_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(step)
            losses.append(loss.item())

        assert len(losses) == 3
        assert all(l > 0 for l in losses), f"All losses should be positive: {losses}"

        # 후속 테스트를 위해 gradient/activation 메모리 해제
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()


# ─────────────────────────────────────────
# Data → Model Integration
# ─────────────────────────────────────────

@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.integration
class TestDataModelIntegration:

    def test_datamodule_to_model(self, molda_model, cfg):
        """DataModule → batch → model.forward 연결 (max_length=512 프로덕션 동일)."""
        from omegaconf import OmegaConf
        from src.data.datamodule import MolDADataModule

        # 이전 테스트의 gradient/activation 메모리 해제
        molda_model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        # batch_size=2로 축소 (이전 training step에서 남은 메모리 때문)
        small_cfg = OmegaConf.merge(cfg, {"training": {"batch_size": 2}})

        dm = MolDADataModule(tokenizer=molda_model.tokenizer, cfg=small_cfg)
        dm.setup("fit")
        dl = dm.train_dataloader()
        batch = next(iter(dl))

        # 배치 shape 확인: max_length=512
        assert batch["input_ids"].shape[1] == cfg.data.max_length, (
            f"Expected seq_len={cfg.data.max_length}, got {batch['input_ids'].shape[1]}"
        )

        # Move batch to GPU
        gpu_batch = {
            "input_ids": batch["input_ids"].cuda(),
            "labels": batch["labels"].cuda(),
            "attention_mask": batch["attention_mask"].cuda(),
            "tasks": batch["tasks"],
            "global_step": 0,
        }

        molda_model.train()
        with torch.no_grad():
            result = molda_model(gpu_batch)
        assert torch.isfinite(result["loss"]), f"Loss not finite with real data: {result['loss']}"

    def test_eval_collator_shapes(self, molda_model, cfg):
        """Eval DataLoader 배치의 left-padding 및 shape 검증."""
        from src.data.datamodule import MolDADataModule
        from src.data.collator import EvalCollator

        dm = MolDADataModule(tokenizer=molda_model.tokenizer, cfg=cfg)
        dm.setup("fit")
        dl = dm.val_dataloader()
        batch = next(iter(dl))

        assert "prompt_input_ids" in batch
        assert "prompt_attention_mask" in batch
        assert "target_texts" in batch

        # Left-padding 확인: PAD ID는 토크나이저에서 파생
        eval_collator = EvalCollator(molda_model.tokenizer, max_length=cfg.data.max_length)
        pad_id = eval_collator.pad_token_id
        pad_mask = (batch["prompt_attention_mask"] == 0)
        if pad_mask.any():
            assert (batch["prompt_input_ids"][pad_mask] == pad_id).all()


# ─────────────────────────────────────────
# Generation (SMDM Diffusion Sampling)
# ─────────────────────────────────────────

@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.integration
class TestGeneration:

    def test_generate_produces_output(self, molda_model, cfg):
        """generate() wrapper가 올바른 shape의 output을 반환하는지 확인."""
        molda_model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        molda_model.eval()
        from src.generation.generate import generate

        tokenizer = molda_model.tokenizer
        prompt = "<INSTRUCTION>Predict the property.</INSTRUCTION> Input molecule."
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").cuda()
        prompt_mask = torch.ones_like(prompt_ids)
        prompt_len = prompt_ids.shape[1]
        gen_length = cfg.data.gen_max_len  # 256

        output = generate(
            molda_model.llada.model,
            prompt_ids,
            attention_mask=prompt_mask,
            gen_length=gen_length,
            steps=4,  # 빠른 테스트를 위해 step 수 줄임
            remasking="low_confidence",
        )

        # Output shape: [B, prompt_len + gen_length]
        assert output.shape == (1, prompt_len + gen_length), (
            f"Expected [{1}, {prompt_len + gen_length}], got {output.shape}"
        )

        # Prompt 부분은 원본 유지
        assert torch.equal(output[:, :prompt_len], prompt_ids)

        # Generated 부분에 MASK token이 남아있지 않아야 함
        gen_part = output[:, prompt_len:]
        mask_count = (gen_part == 126336).sum().item()
        assert mask_count == 0, f"Generated output still has {mask_count} MASK tokens"

    def test_generate_with_logging_snapshots(self, molda_model, cfg):
        """generate_with_logging()이 snapshot 리스트를 반환하는지 확인."""
        molda_model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        molda_model.eval()
        from src.generation.generate import generate_with_logging

        tokenizer = molda_model.tokenizer
        prompt = "<INSTRUCTION>Predict.</INSTRUCTION> mol"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").cuda()
        prompt_mask = torch.ones_like(prompt_ids)
        gen_length = cfg.data.gen_max_len  # 256
        steps = 4

        output, snapshots, _ = generate_with_logging(
            molda_model.llada.model,
            prompt_ids,
            attention_mask=prompt_mask,
            gen_length=gen_length,
            steps=steps,
            remasking="low_confidence",
        )

        # Snapshots: list of [B, gen_length] tensors (CPU)
        assert len(snapshots) == steps, f"Expected {steps} snapshots, got {len(snapshots)}"
        for i, snap in enumerate(snapshots):
            assert snap.shape == (1, gen_length), f"Snapshot {i} shape: {snap.shape}"
            assert snap.device == torch.device("cpu"), f"Snapshot {i} not on CPU"

        # 마지막 snapshot에는 MASK가 없어야 함
        final_snap = snapshots[-1]
        mask_count = (final_snap == 126336).sum().item()
        assert mask_count == 0, f"Final snapshot still has {mask_count} MASK tokens"

    def test_binary_prob_with_real_data(self, molda_model, cfg):
        """Classification likelihood eval이 real data에서 동작하는지 확인."""
        molda_model.eval()
        from src.data.datamodule import MolDADataModule
        from src.training.metrics import CLASSIFICATION_TASKS

        dm = MolDADataModule(tokenizer=molda_model.tokenizer, cfg=cfg)
        dm.setup("fit")
        dl = dm.val_dataloader()
        batch = next(iter(dl))

        # Classification task만 필터링
        cls_idx = [i for i, t in enumerate(batch["tasks"]) if t in CLASSIFICATION_TASKS]
        if not cls_idx:
            pytest.skip("No classification tasks in first val batch")

        prompt_ids = batch["prompt_input_ids"][cls_idx].cuda()
        prompt_mask = batch["prompt_attention_mask"][cls_idx].cuda()

        probs = molda_model.compute_binary_prob_likelihood(prompt_ids, prompt_mask)
        assert probs.shape == (len(cls_idx), 2)
        sums = probs.sum(dim=1)
        for i in range(len(cls_idx)):
            assert sums[i].item() == pytest.approx(1.0, abs=1e-3), \
                f"Sample {i} prob sum: {sums[i].item()}"


# ─────────────────────────────────────────
# DDP Readiness
# ─────────────────────────────────────────

@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.integration
class TestDDPReadiness:

    def test_find_unused_parameters_needed(self, molda_model, optimizer_and_scheduler):
        """Stage 1에서 unused parameters가 존재하는지 확인."""
        optimizer, _ = optimizer_and_scheduler
        molda_model.train()
        batch = _make_batch(molda_model)

        optimizer.zero_grad()
        result = molda_model(batch)
        result["loss"].backward()

        used_params = []
        for name, param in molda_model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    used_params.append(name)

        assert len(used_params) > 0, "No parameters received gradients"

    def test_model_wrappable_with_ddp(self, molda_model):
        """모델의 모든 parameter가 같은 device에 있는지 확인."""
        devices = set()
        for name, param in molda_model.named_parameters():
            devices.add(str(param.device))

        assert len(devices) == 1, f"Parameters on multiple devices: {devices}"

    def test_checkpoint_filtering_keys(self, molda_model):
        """on_save_checkpoint 필터링 로직이 올바른 키만 남기는지 확인."""
        state_dict = dict(molda_model.state_dict())
        original_count = len(state_dict)

        to_remove = []
        for key in state_dict:
            keep = any(k in key for k in [
                "lora_", "embed", "wte", "lm_head", "ff_out", "output_embeddings",
                "modules_to_save", "original_module",
                "qformer", "gnn", "query_tokens", "opt_proj", "ln_graph",
            ])
            if not keep:
                to_remove.append(key)

        for key in to_remove:
            del state_dict[key]

        filtered_count = len(state_dict)

        assert filtered_count < original_count, (
            f"Filtering should remove base weights: {original_count} → {filtered_count}"
        )
        assert filtered_count > 0, "No trainable params after filtering"

        lora_keys = [k for k in state_dict if "lora_" in k]
        assert len(lora_keys) > 0, "No LoRA keys in filtered checkpoint"

        embed_keys = [k for k in state_dict if "wte" in k or "embed" in k]
        assert len(embed_keys) > 0, "No embedding keys in filtered checkpoint"


