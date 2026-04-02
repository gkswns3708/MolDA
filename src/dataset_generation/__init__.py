"""Dataset generation package for MolDA.

Provides the full pipeline: download → process → decontaminate → concat → map.
"""

from dataset_generation.dedup import run_decontamination_pipeline

__all__ = ["run_decontamination_pipeline"]
