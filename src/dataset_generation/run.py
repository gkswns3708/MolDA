#!/usr/bin/env python3
"""CLI entry point for the MolDA dataset generation pipeline.

Usage:
    # Single config, dedup ON (default)
    python -m dataset_generation.run --config smiles

    # dedup OFF (mirrors official Mol-LLM pipeline — concat + map only)
    python -m dataset_generation.run --config smiles --dedup off

    # Multiple configs
    python -m dataset_generation.run --config smiles selfies

    # Toy mode: 100 samples per task for quick testing
    python -m dataset_generation.run --config smiles --toy 100

Pipeline:
    Step 1 : Download & Process Individual Tasks     → Raw/{tag}/step1/
    Step 2 : Cross-Source Decontamination             → Raw/{tag}/step2/
             (skipped when --dedup off)
    Step 3 : Concatenate & Map (prompt formatting)    → Processed/{tag}/{Train,Val,Test}/

Directory layout:
    {raw_data_root}/{data_tag}/step1/{task}_subtask-{i}_{split}/
    {raw_data_root}/{data_tag}/step2/{task}_subtask-{i}_{split}/   (dedup on only)
    {raw_data_root}/{data_tag}/.step2_done                          (marker file)
    {processed_data_root}/{data_tag}/{Train,Val,Test}/
"""

import argparse
import multiprocessing as mp
import os
import time
import traceback

import datasets
import yaml
from tqdm import tqdm

# mmap 한계 회피: PyTorch 기본 file_descriptor 전략은 tensor마다 FD-mapped
# region을 만들어 vm.max_map_count(컨테이너 기본 65530, /proc/sys 변경 불가)에
# 금방 부딪힘. 모듈 top-level에 설정하여 ProcessPoolExecutor가 spawn으로 생성한
# worker가 module을 재import할 때도 동일하게 적용되도록 함.
try:
    import torch.multiprocessing as _torch_mp
    _torch_mp.set_sharing_strategy("file_system")
except ImportError:
    pass  # torch 없는 환경은 그대로 진행

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


STEP2_MARKER_FILENAME = ".step2_done"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_data_tag(base_tag, toy_n, dedup):
    """Apply toy / nodedup suffixes to keep outputs separated by mode."""
    tag = base_tag
    if toy_n:
        tag = f"{tag}_toy{toy_n}"
    if dedup == "off":
        tag = f"{tag}_nodedup"
    return tag


def step1_dir(raw_root, data_tag):
    return os.path.join(raw_root, data_tag, "step1")


def step2_dir(raw_root, data_tag):
    return os.path.join(raw_root, data_tag, "step2")


def step2_marker(raw_root, data_tag):
    return os.path.join(raw_root, data_tag, STEP2_MARKER_FILENAME)


def processed_dir(processed_root, data_tag):
    return os.path.join(processed_root, data_tag)


def task_arrow_name(task, subtask_idx, split):
    return f"{task}_subtask-{subtask_idx}_{split}"


# ---------------------------------------------------------------------------
# Step 1: Download & Process Individual Tasks
# ---------------------------------------------------------------------------

def run_step1(cfg, task_subtask_pairs, step1_root, num_workers, toy_n):
    """Step 1은 덮어쓰지 않는다 — 이미 Arrow가 있으면 skip."""
    print("\n" + "=" * 70)
    print(f"[Step 1] Download & Process Individual Tasks → {step1_root}")
    print("=" * 70)
    os.makedirs(step1_root, exist_ok=True)

    for task_subtask_pair in tqdm(task_subtask_pairs, desc="Step 1: Processing tasks"):
        task_name, subtask_idx = task_subtask_pair[0], task_subtask_pair[1]

        check_train_path = os.path.join(step1_root, task_arrow_name(task_name, subtask_idx, "train"))
        if os.path.exists(check_train_path):
            continue

        try:
            new_dataset = get_dataset(
                task_name=task_name,
                raw_data_root=cfg.raw_data_root,
                toy_n=toy_n,
            )
        except Exception as e:
            print(f"[Error] Failed to get dataset for {task_name}: {e}")
            traceback.print_exc()
            continue

        subtasks = new_dataset[0]

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
        common_features = None

        for split in ["train", "val", "test"]:
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
            save_path = os.path.join(step1_root, task_arrow_name(task_name, subtask_idx, split))

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


def load_step1_task_datasets(task_subtask_pairs, step1_root):
    """step1/에서 {task → {split → Dataset}} 구조로 로드."""
    out = {}
    for task, subtask_idx in task_subtask_pairs:
        splits = {}
        for split in ["train", "val", "test"]:
            path = os.path.join(step1_root, task_arrow_name(task, subtask_idx, split))
            if os.path.exists(path):
                try:
                    ds = datasets.Dataset.load_from_disk(path)
                    if len(ds) > 0:
                        splits[split] = ds
                except Exception as e:
                    print(f"[Warning] Failed to load {path}: {e}")
        if splits:
            out[task] = splits
    return out


# ---------------------------------------------------------------------------
# Step 2: Cross-Source Decontamination
# ---------------------------------------------------------------------------

def run_step2(task_subtask_pairs, step1_root, step2_root, tag_root):
    """step1/을 읽기전용 입력으로 받아 step2/에 decontam 결과를 저장.

    완료 시 `.step2_done` 마커 파일을 생성 — 다음 실행에서 Step 2 skip.
    """
    marker_path = os.path.join(tag_root, STEP2_MARKER_FILENAME)
    if os.path.exists(marker_path) and os.path.isdir(step2_root):
        print(f"\n[Step 2] Skipped — marker exists: {marker_path}")
        return

    print("\n" + "=" * 70)
    print(f"[Step 2] Cross-Source Decontamination → {step2_root}")
    print("=" * 70)
    os.makedirs(step2_root, exist_ok=True)

    task_split_datasets = load_step1_task_datasets(task_subtask_pairs, step1_root)
    task_split_datasets = run_decontamination_pipeline(task_split_datasets)

    # step2/에 저장 (atomic: tmp → rename)
    for task, splits in task_split_datasets.items():
        # subtask_idx 복원 (task_subtask_pairs에서 탐색)
        subtask_idx = next(
            (s for (t, s) in task_subtask_pairs if t == task), 0
        )
        for split_name, ds in splits.items():
            path = os.path.join(step2_root, task_arrow_name(task, subtask_idx, split_name))
            tmp_path = path + "_tmp"
            ds.save_to_disk(tmp_path)
            if os.path.exists(path):
                import shutil
                shutil.rmtree(path)
            os.rename(tmp_path, path)

    # 완료 마커 생성
    with open(marker_path, "w") as f:
        f.write("step2 completed\n")
    print(f"[Step 2] Marker written: {marker_path}")


# ---------------------------------------------------------------------------
# Step 3: Concatenate & Map
# ---------------------------------------------------------------------------

def run_step3(cfg, task_subtask_pairs, input_root, processed_root, num_workers):
    """input_root(step1/ or step2/)에서 per-task Arrow를 로드하고 concat + map + save."""
    print("\n" + "=" * 70)
    print(f"[Step 3] Concatenate & Map → {processed_root}")
    print(f"         input: {input_root}")
    print("=" * 70)
    os.makedirs(processed_root, exist_ok=True)

    trainsets, testsets, valsets = [], [], []
    for task, subtask_idx in task_subtask_pairs:
        for split, target_list in [("train", trainsets), ("test", testsets), ("val", valsets)]:
            split_path = os.path.join(input_root, task_arrow_name(task, subtask_idx, split))
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

    map_kwargs = {
        "fn_kwargs": {
            "system_prompt": system_prompt,
            "llm_model_name": cfg.llm_model,
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

        save_name = os.path.join(processed_root, split_name)
        mapped_set.save_to_disk(save_name)
        print(f"Saved Final {split_name} Dataset: {save_name} (Size: {len(mapped_set)})")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_single_config(config_name, config_dir, num_workers, toy_n, dedup):
    arg_path = os.path.join(config_dir, config_name) + ".yaml"
    with open(arg_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    _run_single_config_with_cfg(cfg_dict, config_name, num_workers, toy_n, dedup)


def _run_single_config_with_cfg(cfg_dict, config_name, num_workers, toy_n, dedup):
    cfg = from_dict(cfg_dict)

    raw_root = cfg.raw_data_root
    processed_root = getattr(
        cfg,
        "processed_data_root",
        os.path.join(os.path.dirname(raw_root.rstrip(os.sep)), "Processed"),
    )
    data_tag = resolve_data_tag(cfg.data_tag, toy_n, dedup)

    tag_root = os.path.join(raw_root, data_tag)
    s1_root = step1_dir(raw_root, data_tag)
    s2_root = step2_dir(raw_root, data_tag)
    p_root = processed_dir(processed_root, data_tag)

    os.makedirs(tag_root, exist_ok=True)

    start_time = time.time()
    _, task_subtask_pairs = get_task_subtask_info(cfg.target_benchmarks)

    print(f"\n{'#' * 70}")
    print(f"# Config: {config_name} | Tag: {data_tag} | Dedup: {dedup}")
    print(f"# Raw root:       {raw_root}")
    print(f"# Processed root: {processed_root}")
    print(f"{'#' * 70}")

    # Step 1
    run_step1(cfg, task_subtask_pairs, s1_root, num_workers, toy_n)

    # Step 2 (only when dedup on)
    if dedup == "on":
        run_step2(task_subtask_pairs, s1_root, s2_root, tag_root)
        step3_input = s2_root
    else:
        print("\n[Step 2] Skipped — --dedup off (replicating official Mol-LLM pipeline)")
        step3_input = s1_root

    # Step 3
    run_step3(cfg, task_subtask_pairs, step3_input, p_root, num_workers)

    elapsed = (time.time() - start_time) / 60
    print(f"\n[{config_name}] Completed in {elapsed:.2f} minutes")


def main():
    mp.set_start_method("spawn", force=True)
    # set_sharing_strategy는 모듈 top-level에서 이미 설정됨 (worker에도 적용되도록).

    parser = argparse.ArgumentParser(description="MolDA Dataset Generation Pipeline")
    parser.add_argument("--config_dir", type=str, default="configs/download/")
    parser.add_argument("--config", type=str, nargs="+", default=["smiles"],
                        help="Config name(s). Multiple configs → sequential execution.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Parallel workers. 0 = auto (os.cpu_count())")
    parser.add_argument("--toy", type=int, default=None,
                        help="Toy mode: sample N examples per task per split for quick testing")
    parser.add_argument("--dedup", type=str, choices=["on", "off"], default="on",
                        help="Cross-source decontamination. 'off' mirrors official Mol-LLM "
                             "(Step 2 skipped, data_tag suffixed with _nodedup)")
    args = parser.parse_args()

    num_workers = get_num_workers(args.num_workers)
    configs = args.config

    print(f"[Config] Configs: {configs} | Workers: {num_workers} | "
          f"Toy: {args.toy or 'OFF'} | Dedup: {args.dedup}")

    total_start = time.time()

    for i, config_name in enumerate(configs):
        print(f"\n{'=' * 70}")
        print(f"  Running config {i + 1}/{len(configs)}: {config_name}")
        print(f"{'=' * 70}")
        run_single_config(config_name, args.config_dir, num_workers, args.toy, args.dedup)

    total_elapsed = (time.time() - total_start) / 60
    if len(configs) > 1:
        print(f"\n[All Done] {len(configs)} configs completed in {total_elapsed:.2f} minutes total")


if __name__ == "__main__":
    main()
