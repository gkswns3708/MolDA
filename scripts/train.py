"""
MolDA training entry point (Hydra).

Usage:
    cd /opt/11-MolDA/New_MolDA
    python scripts/train.py --config-name toy

    # Override examples:
    python scripts/train.py --config-name toy hardware.devices="'0,1'"
    python scripts/train.py --config-name toy training.max_steps=100
"""

import sys
from datetime import timedelta
from pathlib import Path

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


@hydra.main(config_path="../src/configs", config_name="toy", version_base="1.3")
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
            filename="step-{step}",
            every_n_train_steps=cfg.logging.save_on_n_steps,
            save_top_k=cfg.logging.save_top_k_checkpoints,
            save_last=True,
        ),
        ModelCheckpoint(
            dirpath=log_dir,
            filename="val-{epoch:02d}-{step}",
            every_n_epochs=cfg.logging.save_every_n_epochs,
            save_top_k=-1, # config로 수정할 수 있음
            save_on_train_epoch_end=False,  # validation 끝에 저장
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
        )
    else:
        strategy = "auto"

    # Logger: CSVLogger (항상) + WandB (선택)
    csv_logger = CSVLogger(save_dir=".", name="lightning_logs")
    loggers = [csv_logger]
    if cfg.wandb.get("enabled", False):
        from pytorch_lightning.loggers import WandbLogger
        loggers.append(WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity"),
            name=cfg.wandb.get("run_name"),
            save_dir=log_dir,
        ))

    # Trainer
    trainer = pl.Trainer(
        accelerator=cfg.hardware.accelerator,
        devices=device_list,
        precision=cfg.hardware.precision,
        strategy=strategy,
        max_epochs=cfg.training.max_epochs,
        max_steps=cfg.training.max_steps,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
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

    # Run
    if cfg.mode == "ft":
        trainer.fit(model, dm, ckpt_path=cfg.ckpt_path)
    elif cfg.mode == "test":
        trainer.test(model, dm, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()
