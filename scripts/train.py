"""
MolDA training entry point (Hydra).

Usage:
    cd /opt/11-MolDA/New_MolDA

    # Experiment configs (실제 실험):
    python scripts/train.py +experiment=selfies_dict trainer=stage1
    python scripts/train.py +experiment=selfies_nodict trainer=stage1
    python scripts/train.py +experiment=smiles_nodict trainer=stage1

    # Toy configs (디버깅/테스트):
    python scripts/train.py --config-name toy_SELFIES
    python scripts/train.py --config-name toy_SMILES

    # Override examples:
    python scripts/train.py +experiment=selfies_dict trainer=stage1 hardware.devices="'0,1'"
    python scripts/train.py +experiment=selfies_dict trainer=stage2 \
        pretrained_ckpt_path=./checkpoint/selfies_dict/stage1/last.ckpt
"""

import os
import sys
from datetime import timedelta
from pathlib import Path

# Old_MolDA에서 검증된 설정 (NCCL hang / OOM 방지)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.set_float32_matmul_precision("medium")

# Suppress DDP + V-MolPO 의 AccumulateGrad stream-mismatch UserWarning.
# 매 backward 마다 발생하여 tqdm progress 를 가림. 학습 정확도에 영향 없음.
# (DDP 가 grad node 를 stash 하여 발생하는 정상 동작)
if hasattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch"):
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

# PyTorch 2.6+: default weights_only=True blocks OmegaConf-pickled hparams.
# Our ckpts are our own trusted artifacts, so hard-force weights_only=False
# even when callers (Lightning) explicitly pass weights_only=True.
_orig_torch_load = torch.load
def _trusted_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _trusted_torch_load

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy

# Ensure project root is in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.training.trainer import MolDATrainer
from src.data.datamodule import MolDADataModule


def _load_dotenv(env_path: Path = Path(PROJECT_ROOT) / ".env"):
    """프로젝트 루트의 .env 파일에서 환경변수를 로드한다 (기존 값 덮어쓰지 않음)."""
    if not env_path.is_file():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("\"'")
            if key not in os.environ:
                os.environ[key] = value


_load_dotenv()


@hydra.main(config_path="../src/configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed)

    # Model
    model = MolDATrainer(cfg)

    # DataModule
    dm = MolDADataModule(tokenizer=model.tokenizer, cfg=cfg)

    # Callbacks
    log_dir = cfg.logging.dir
    callbacks = [
        ModelCheckpoint(
            dirpath=log_dir,
            filename="epoch={epoch}-step={step}",
            every_n_epochs=cfg.logging.save_every_n_epochs,
            save_top_k=-1,
            save_last=True,
            save_on_train_epoch_end=True,  # validation 안 도는 epoch에서도 저장 (중도 종료 시 손실 방지)
        ),
    ]

    # Strategy
    devices = cfg.hardware.devices
    if isinstance(devices, str):
        device_list = [int(d) for d in devices.split(",") if d.strip()]
    elif isinstance(devices, int):
        device_list = [devices]
    else:
        device_list = list(devices)

    if len(device_list) > 1:
        strategy = DDPStrategy(
            find_unused_parameters=cfg.hardware.find_unused_parameters,
            timeout=timedelta(minutes=90),
            start_method="spawn",
        )
    else:
        strategy = "auto"

    # Logger: CSVLogger (항상) + WandB (선택)
    csv_logger = CSVLogger(save_dir=".", name="lightning_logs")
    loggers = [csv_logger]
    wandb_logger = None
    if cfg.wandb.get("enabled", False):
        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity") or os.environ.get("WANDB_ENTITY"),
            name=cfg.wandb.get("run_name"),
            save_dir=log_dir,
            id=cfg.wandb.get("id"),
            resume="allow" if cfg.wandb.get("id") else None,
            tags=cfg.wandb.get("tags", []),
            group=cfg.wandb.get("group"),
            log_model=cfg.wandb.get("log_model", False),
        )
        loggers.append(wandb_logger)

    # Compute accumulate_grad_batches from global_batch_size
    num_devices = len(device_list)
    per_gpu_bs = cfg.training.batch_size
    global_bs = cfg.training.global_batch_size
    assert global_bs % (per_gpu_bs * num_devices) == 0, (
        f"global_batch_size({global_bs}) must be divisible by "
        f"batch_size({per_gpu_bs}) × devices({num_devices}) = {per_gpu_bs * num_devices}"
    )
    accumulate_grad_batches = global_bs // (per_gpu_bs * num_devices)

    # Trainer
    trainer = pl.Trainer(
        accelerator=cfg.hardware.accelerator,
        devices=device_list,
        precision=cfg.hardware.precision,
        strategy=strategy,
        max_epochs=cfg.training.max_epochs,
        max_steps=cfg.training.max_steps,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=cfg.training.gradient_clip_val,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        num_sanity_val_steps=cfg.validation.num_sanity_val_steps,
        val_check_interval=cfg.validation.val_check_interval,
        check_val_every_n_epoch=cfg.validation.check_val_every_n_epoch,
        limit_val_batches=cfg.validation.limit_val_batches,
        enable_progress_bar=True,
    )

    # WandB: gradient/param histogram 로깅 (opt-in)
    if wandb_logger is not None and cfg.wandb.get("watch_model", False):
        wandb_logger.watch(
            model,
            log="all",
            log_freq=cfg.wandb.get("log_freq", 10),
            log_graph=False,
        )

    # Run
    if cfg.mode == "ft":
        trainer.fit(model, dm, ckpt_path=cfg.ckpt_path)
    elif cfg.mode == "test":
        trainer.test(model, dm, ckpt_path=cfg.ckpt_path)

    # WandB: run 정상 종료
    if wandb_logger is not None:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
