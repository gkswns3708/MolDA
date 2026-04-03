"""OptimizerMixin: optimizer/scheduler 설정 및 gradient scaling 로직.

분리 대상:
- configure_optimizers: 5 param groups + WSD scheduler
- _estimate_total_steps: scheduler duration 계산
- on_before_optimizer_step: new vocab gradient scaling + grad norm logging
- _is_output_head_param: output head 파라미터 판별 (모듈 레벨 헬퍼)
"""

import logging

import torch
from torch.optim import AdamW

from src.training.scheduler import WarmupStableDecayLRScheduler
from src.loggers.grad_logger import compute_grad_norms

logger = logging.getLogger(__name__)


def _is_output_head_param(name: str) -> bool:
    """Output head 파라미터인지 판별. Block-level ff_out은 제외.

    매칭 대상: transformer.ff_out (최종 output head)
    제외 대상: transformer.blocks.*.ff_out (block FFN down projection)
    """
    lower = name.lower()
    if "lm_head" in lower:
        return True
    if "ff_out" in lower:
        # block-level ff_out 제외: "blocks." 뒤에 ff_out이 오는 패턴
        return "blocks." not in lower
    return False


class OptimizerMixin:
    """Optimizer/scheduler 설정 및 gradient scaling을 담당하는 Mixin."""

    def configure_optimizers(self):
        """Param groups: LoRA / embed / head(optional) / other.

        embed_new/head_new LR은 gradient scaling으로 구현:
        - embed/head param group은 orig LR로 등록
        - on_before_optimizer_step에서 new vocab rows의 gradient에
          (lr_new / lr_orig) 비율을 곱해 effective LR을 높임
        - add_mol_dict=false(no_dict)이면 scaling 안 함
        """
        cfg = self.cfg
        lr = cfg.lr
        orig_vocab_size = cfg.model.original_vocab_size

        # Collect params by group
        lora_params = []
        embed_params = []
        head_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if "lora_" in name:
                lora_params.append(param)
            elif "embed" in name.lower() or "wte" in name.lower():
                embed_params.append(param)
            elif _is_output_head_param(name):
                head_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {"params": lora_params, "lr": lr.lora, "name": "lora"},
            {"params": embed_params, "lr": lr.embed_orig, "name": "embed"},
        ]

        # head group: weight_tying=True면 별도 ff_out이 없어 비어있을 수 있음
        if head_params:
            param_groups.append(
                {"params": head_params, "lr": lr.head_orig, "name": "head"}
            )

        # Only add other group if there are params (Stage 2+)
        if other_params and lr.other > 0:
            param_groups.append(
                {"params": other_params, "lr": lr.other, "name": "other"}
            )

        # --- Gradient scaling info for new vocab rows ---
        # add_mol_dict=false (no_dict)이면 new vocab이 없으므로 scaling 불필요
        use_new_scaling = cfg.tokenizer.get("add_mol_dict", False)

        if use_new_scaling and lr.embed_orig > 0:
            lr_ratio_embed = lr.embed_new / lr.embed_orig
        else:
            lr_ratio_embed = 1.0

        if use_new_scaling and lr.head_orig > 0 and head_params:
            lr_ratio_head = lr.head_new / lr.head_orig
        else:
            lr_ratio_head = 1.0

        self._embed_head_split_info = {
            "original_vocab_size": orig_vocab_size,
            "lr_ratio_embed": lr_ratio_embed,
            "lr_ratio_head": lr_ratio_head,
            "embed_params": embed_params,
            "head_params": head_params,
        }

        optimizer = AdamW(
            param_groups,
            betas=(0.9, 0.95),
            weight_decay=cfg.training.weight_decay,
        )

        # Estimate total steps
        total_steps = self._estimate_total_steps()

        scheduler = WarmupStableDecayLRScheduler(
            optimizer=optimizer,
            max_step=total_steps,
            warmup_steps=cfg.scheduler.warmup_steps,
            decay_ratio=cfg.scheduler.decay_ratio,
            min_lr_ratio=cfg.scheduler.min_lr_ratio,
        )

        # Store scheduler for manual step in training_step
        # (PL expects torch _LRScheduler; ours is custom)
        self._scheduler = scheduler

        return optimizer

    def _estimate_total_steps(self) -> int:
        """Estimate total training steps for scheduler."""
        cfg = self.cfg
        if cfg.training.max_steps > 0:
            return cfg.training.max_steps

        # Estimate from epochs (rough; trainer may override)
        try:
            n_samples = len(self.trainer.datamodule.train_dataset)
        except Exception:
            n_samples = 2100  # fallback for toy dataset

        num_devices = max(1, len(str(cfg.hardware.devices).split(",")))
        per_device_steps = n_samples // (cfg.training.batch_size * num_devices)
        accumulate = cfg.training.global_batch_size // (cfg.training.batch_size * num_devices)
        steps_per_epoch = max(1, per_device_steps // max(1, accumulate))
        return steps_per_epoch * cfg.training.max_epochs

    def on_before_optimizer_step(self, optimizer):
        """Apply gradient scaling for new vocab rows, then log grad norms."""
        # --- Gradient scaling: new vocab rows에 lr_ratio 곱하기 ---
        info = getattr(self, "_embed_head_split_info", None)
        if info is not None:
            orig_size = info["original_vocab_size"]
            ratio_embed = info["lr_ratio_embed"]
            ratio_head = info["lr_ratio_head"]

            if ratio_embed != 1.0:
                for param in info["embed_params"]:
                    if param.grad is not None and param.shape[0] > orig_size:
                        param.grad[orig_size:] *= ratio_embed

            if ratio_head != 1.0:
                for param in info["head_params"]:
                    if param.grad is not None and param.shape[0] > orig_size:
                        param.grad[orig_size:] *= ratio_head

        # --- Gradient norm 누적 (구간 평균) ---
        interval = self.cfg.logging.get("weight_norm_interval", 10)
        if self.global_step % interval != 0:
            return
        for name, norm in compute_grad_norms(optimizer).items():
            self._accumulate(f"train/grad_norm/{name}", norm, sync_dist=False)
