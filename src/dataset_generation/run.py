#!/usr/bin/env python3
"""CLI entry point for the MolDA dataset generation pipeline.

Usage:
    # Single config
    python -m dataset_generation.run --config smiles

    # Multiple configs (SMILES + SELFIES 동시 생성)
    python -m dataset_generation.run --config smiles selfies

    # Toy mode: 100 samples per task for quick testing
    python -m dataset_generation.run --config smiles --toy 100

    # Multiple configs + toy mode
    python -m dataset_generation.run --config smiles selfies --toy 100

Pipeline:
    Step 1 : Download & Process Individual Tasks
    Step 2 : Cross-Source Decontamination (leakage removal + cross-source dedup)
    Step 3 : Concatenate & Map (prompt formatting → final dataset)
"""

import argparse
import multiprocessing as mp
import os
import time
import traceback

import datasets
import yaml
from tqdm import tqdm

from dataset_generation.generator import (
    ChEBIDataset,
    MoleculeNetDatasetDeepChem,
    MolInstructionDataset,
    SMolInstructDataset,
    dataset_to_arrow_dicts,
    get_dataset,
    prepare_data_instance,
)
from dataset_generation.dedup import run_decontamination_pipeline
from dataset_generation.utils import from_dict, get_num_workers, get_task_subtask_info


def run_single_config(config_name, config_dir, num_workers, toy_n):
    """단일 config에 대한 전체 파이프라인 실행.

    Step 1-2는 1회 실행 (canonical SMILES 기반).
    Step 3에서 SMILES/SELFIES 양쪽 컬럼을 동시에 생성.
    """

    arg_path = os.path.join(config_dir, config_name) + ".yaml"
    with open(arg_path, "r") as f:
        cfg = yaml.safe_load(f)

    _run_single_config_with_cfg(cfg, config_name, num_workers, toy_n)


def _run_single_config_with_cfg(cfg_dict, config_name, num_workers, toy_n):
    """실제 파이프라인 실행 (이미 파싱된 cfg dict 사용)."""
    cfg = from_dict(cfg_dict)

    raw_data_root = cfg.raw_data_root
    data_tag = cfg.data_tag
    if toy_n:
        data_tag = f"{data_tag}_toy{toy_n}"

    # 단일 출력 경로: dataset/Raw/{data_tag}/ (mol_representation 구분 없음)
    tag_root = os.path.join(raw_data_root, data_tag)
    os.makedirs(tag_root, exist_ok=True)

    start_time = time.time()
    task_subtask_dict, task_subtask_pairs = get_task_subtask_info(cfg.target_benchmarks)

    print(f"\n{'#' * 70}")
    print(f"# Config: {config_name} | Tag: {data_tag}")
    print(f"# Output: {tag_root}")
    print(f"{'#' * 70}")

    # =========================================================================
    # Step 1: Download & Process Individual Tasks
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Step 1] Download & Process Individual Tasks")
    print("=" * 70)

    for task_subtask_pair in tqdm(task_subtask_pairs, desc="Step 1: Processing tasks"):
        task_name = task_subtask_pair[0]
        subtask_idx = task_subtask_pair[1]

        check_train_path = os.path.join(tag_root, f"{task_name}_subtask-{subtask_idx}_train")
        if os.path.exists(check_train_path):
            continue

        try:
            new_dataset = get_dataset(
                task_name=task_name,
                raw_data_root=raw_data_root,
                toy_n=toy_n,
            )
        except Exception as e:
            print(f"[Error] Failed to get dataset for {task_name}: {e}")
            traceback.print_exc()
            continue

        subtasks = new_dataset[0]
        subtask_idx = task_subtask_pair[1]

        if subtask_idx == "multi_label_classification":
            pair_name = f"{task_name}/multi_label_classification"
        elif task_name in ["toxcast", "tox21", "qm9_additional_label", "hopv"]:
            pair_name = f"{task_name}/{subtasks[subtask_idx]}"
        else:
            pair_name = f"{task_name}/0"

        data_split = new_dataset[1:]

        dataset_cls = (
            SMolInstructDataset if "smol" in task_name else
            MoleculeNetDatasetDeepChem if task_name in ["toxcast", "tox21", "qm9_additional_label", "hopv"] else
            ChEBIDataset if task_name in ["chebi-20-mol2text", "chebi-20-text2mol"] else
            MolInstructionDataset
        )

        def _task_arg_for(cls):
            return task_name if cls is MolInstructionDataset else pair_name

        dataset_splits = {"train": data_split[0], "val": data_split[1], "test": data_split[2]}
        process_order = ["train", "val", "test"]
        common_features = None

        for split in process_order:
            if task_name in ["smol-name_conversion-i2s", "smol-name_conversion-s2i"] and split != "train":
                continue

            raw_data = dataset_splits[split]
            if raw_data is None or len(raw_data) == 0:
                continue

            ds = dataset_cls(
                data=raw_data,
                task_subtask_pair=_task_arg_for(dataset_cls),
                subtask_idx=subtask_idx,
                num_workers=num_workers,
            )

            list_dict_data = dataset_to_arrow_dicts(ds)
            save_path = os.path.join(tag_root, f"{task_name}_subtask-{subtask_idx}_{split}")

            if len(list_dict_data) > 0:
                print(f"  [{task_name}/{split}] Building HF Dataset ({len(list_dict_data)} samples)...")
                output_dataset = datasets.Dataset.from_list(list_dict_data)
                if common_features is None:
                    common_features = output_dataset.features
                print(f"  [{task_name}/{split}] Saving to disk...")
                output_dataset.save_to_disk(save_path)
            else:
                if common_features is not None:
                    empty_data = {key: [] for key in common_features}
                    output_dataset = datasets.Dataset.from_dict(empty_data, features=common_features)
                    output_dataset.save_to_disk(save_path)

    # =========================================================================
    # Step 2: Cross-Source Decontamination
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Step 2] Cross-Source Decontamination")
    print("=" * 70)

    task_split_datasets = {}
    for task_subtask_pair in task_subtask_pairs:
        task, subtask_idx = task_subtask_pair
        splits = {}
        for split in ["train", "val", "test"]:
            path = os.path.join(tag_root, f"{task}_subtask-{subtask_idx}_{split}")
            if os.path.exists(path):
                try:
                    ds = datasets.Dataset.load_from_disk(path)
                    if len(ds) > 0:
                        splits[split] = ds
                except Exception as e:
                    print(f"[Warning] Failed to load {path}: {e}")
        if splits:
            task_split_datasets[task] = splits

    task_split_datasets = run_decontamination_pipeline(task_split_datasets)

    import shutil
    for task, splits in task_split_datasets.items():
        subtask_idx = task_subtask_dict.get(task, [0])[0]
        for split_name, ds in splits.items():
            path = os.path.join(tag_root, f"{task}_subtask-{subtask_idx}_{split_name}")
            tmp_path = path + "_tmp"
            ds.save_to_disk(tmp_path)
            if os.path.exists(path):
                shutil.rmtree(path)
            os.rename(tmp_path, path)

    # =========================================================================
    # Step 3: Concatenate & Map
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Step 3] Concatenate & Map")
    print("=" * 70)

    trainsets, testsets, valsets = [], [], []
    for task_subtask_pair in task_subtask_pairs:
        task, subtask_idx = task_subtask_pair
        for split, target_list in [("train", trainsets), ("test", testsets), ("val", valsets)]:
            split_path = os.path.join(tag_root, f"{task}_subtask-{subtask_idx}_{split}")
            try:
                if os.path.exists(split_path):
                    ds = datasets.Dataset.load_from_disk(split_path)
                    if len(ds) > 0:
                        target_list.append(ds)
            except Exception as e:
                print(f"Error loading {split} path {split_path}: {e}")

    system_prompt = (
        "You are a helpful assistant for molecular chemistry, to address tasks including "
        "molecular property classification, molecular property regression, chemical reaction "
        "prediction, molecule captioning, molecule generation."
    )

    llm_model = cfg.llm_model

    map_kwargs = {
        "fn_kwargs": {
            "system_prompt": system_prompt,
            "llm_model_name": llm_model,
        },
        "num_proc": num_workers,
    }

    for split_name, split_sets in [("Train", trainsets), ("Test", testsets), ("Val", valsets)]:
        if not split_sets:
            print(f"Warning: No {split_name} datasets were concatenated.")
            continue

        concat_set = datasets.concatenate_datasets(split_sets)
        print(f"Mapping {split_name.upper()} dataset ({len(concat_set)} examples)...")

        mapped_set = concat_set.map(prepare_data_instance, **map_kwargs)

        # SELFIES 변환 실패 sample 제거
        before_count = len(mapped_set)
        mapped_set = mapped_set.filter(lambda x: x["_selfies_valid"])
        after_count = len(mapped_set)
        if before_count != after_count:
            print(f"  Filtered {before_count - after_count} SELFIES-failed samples "
                  f"({before_count} → {after_count})")
        mapped_set = mapped_set.remove_columns(["_selfies_valid"])

        save_name = os.path.join(tag_root, split_name)
        mapped_set.save_to_disk(save_name)
        print(f"Saved Final {split_name} Dataset: {save_name} (Size: {len(mapped_set)})")

    elapsed = (time.time() - start_time) / 60
    print(f"\n[{config_name}] Completed in {elapsed:.2f} minutes")


def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="MolDA Dataset Generation Pipeline")
    parser.add_argument("--config_dir", type=str, default="configs/download/")
    parser.add_argument("--config", type=str, nargs="+", default=["smiles"],
                        help="Config name(s). Multiple configs → sequential execution. "
                             "E.g., --config smiles selfies")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Parallel workers. 0 = auto (os.cpu_count())")
    parser.add_argument("--toy", type=int, default=None,
                        help="Toy mode: sample N examples per task per split for quick testing")
    args = parser.parse_args()

    num_workers = get_num_workers(args.num_workers)
    configs = args.config

    print(f"[Config] Configs: {configs} | Workers: {num_workers} | Toy: {args.toy or 'OFF'}")

    total_start = time.time()

    for i, config_name in enumerate(configs):
        print(f"\n{'=' * 70}")
        print(f"  Running config {i + 1}/{len(configs)}: {config_name}")
        print(f"{'=' * 70}")
        run_single_config(config_name, args.config_dir, num_workers, args.toy)

    total_elapsed = (time.time() - total_start) / 60
    if len(configs) > 1:
        print(f"\n[All Done] {len(configs)} configs completed in {total_elapsed:.2f} minutes total")


if __name__ == "__main__":
    main()
