"""Cross-source decontamination for the MolDA dataset pipeline.

Implements the decontamination strategy described in the Mol-LLM paper
(arXiv:2502.02810): "tasks present in both datasets, such as forward synthesis
and molecule captioning, are deduplicated to ensure that molecules included in
the test set of one dataset do not appear in the training set of the combined
dataset."

This is NOT a "strict chemistry split pipeline" — it is a
"source-compatible multi-source benchmark builder."
The goal is NOT "global dedup" but "cross-source decontamination."
Source priority is NOT "quality superiority" but "benchmark compatibility anchor."

Pipeline:
  1. build_eval_blacklist  — collect entity keys from the eval boundary (test + optionally val)
  2. remove_eval_leakage   — drop train(/val) samples found in eval blacklist
  3. dedup_within_family   — cross-source dedup within entity families (priority rule)

NOTE on reaction dedup:
  현재 decontamination은 reaction string의 canonical component normalization 수준의
  exact/near-exact matching이며, full reaction-role harmonization (agent/reagent 위치,
  atom mapping 차이 등)까지를 목표로 하지는 않는다.
"""

from collections import defaultdict
from dataclasses import dataclass, field

from tqdm import tqdm

from dataset_generation.utils import extract_entity_key

# ---------------------------------------------------------------------------
# Constants — Entity Family Abstraction
# ---------------------------------------------------------------------------

# Chemical entity family 기반 그룹핑.
# "특정 두 task가 겹친다"가 아니라 "이 family는 어떤 chemical key를 공유하는가"로 설계.
# 새 task 추가 시 해당 family에 붙이기만 하면 됨.
ENTITY_FAMILIES = {
    # Reaction Forward Family: 동일 forward reaction을 다루는 task들
    "forward_reaction_prediction": "REACTION_FORWARD_FAMILY",
    "smol-forward_synthesis": "REACTION_FORWARD_FAMILY",
    # Reaction Retro Family
    "retrosynthesis": "REACTION_RETRO_FAMILY",
    "smol-retrosynthesis": "REACTION_RETRO_FAMILY",
    # Mol2Text Family: 동일 molecule→description을 다루는 task들
    "chebi-20-mol2text": "MOL2TEXT_FAMILY",
    "smol-molecule_captioning": "MOL2TEXT_FAMILY",
    # Text2Mol Family: 동일 description→molecule을 다루는 task들
    "chebi-20-text2mol": "TEXT2MOL_FAMILY",
    "smol-molecule_generation": "TEXT2MOL_FAMILY",
}

# Benchmark compatibility anchor: 중복 시 제거할 task.
# 이것은 quality superiority rule이 아니라 benchmark compatibility rule이다.
# SMolInstruct 계열의 evaluation semantics를 중심으로 freeze하기 위한 anchor choice.
REMOVE_ON_CONFLICT = {
    "REACTION_FORWARD_FAMILY": "forward_reaction_prediction",
    "REACTION_RETRO_FAMILY": "retrosynthesis",
    "MOL2TEXT_FAMILY": "chebi-20-mol2text",
    "TEXT2MOL_FAMILY": "chebi-20-text2mol",
}

# Removal reason codes for audit trail
REMOVAL_REASONS = {
    "eval_blacklist": "train/val sample found in eval boundary",
    "within_family_dup": "cross-source duplicate within family (priority rule)",
    "invalid_key": "could not extract entity key (skipped, not removed)",
}


# ---------------------------------------------------------------------------
# Removal statistics
# ---------------------------------------------------------------------------

@dataclass
class RemovalStats:
    """사유별 제거 통계 추적."""
    counts: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    # counts[task_name][split_name][reason] = count

    def record(self, task_name, split_name, reason, count=1):
        self.counts[task_name][split_name][reason] += count

    def print_report(self, title="Removal Report"):
        print(f"\n=== [{title}] ===")
        print(f"{'Task':<45} {'Split':<8} {'Reason':<25} {'Count':>8}")
        print("-" * 90)
        total = 0
        for task_name in sorted(self.counts):
            for split_name in sorted(self.counts[task_name]):
                for reason in sorted(self.counts[task_name][split_name]):
                    count = self.counts[task_name][split_name][reason]
                    if count > 0:
                        print(f"{task_name:<45} {split_name:<8} {reason:<25} {count:>8}")
                        total += count
        print("-" * 90)
        print(f"{'TOTAL':<80} {total:>8}")


def _get_family(task_name):
    """Task name → entity family ID. family에 속하지 않으면 None."""
    return ENTITY_FAMILIES.get(task_name, None)


# ---------------------------------------------------------------------------
# Step 1: Eval blacklist
# ---------------------------------------------------------------------------

def build_eval_blacklist(task_split_datasets, include_validation=True):
    """Eval boundary(test + optionally validation)에서 entity key를 수집하여
    family별 blacklist 생성.

    test는 반드시 포함. validation은 include_validation 파라미터로 제어 (기본=True).
    validation을 포함하는 이유: HPO에 validation을 사용할 경우 이것도 held-out boundary.

    Args:
        task_split_datasets: {task_name: {"train": Dataset, "val": Dataset, "test": Dataset}}
        include_validation: True이면 val도 blacklist에 포함

    Returns:
        {family_or_task_id: set(canonical_entity_key)}
    """
    blacklist = defaultdict(set)
    eval_splits = ["test"]
    if include_validation:
        eval_splits.append("val")

    # 전체 eval 샘플 수 계산 (tqdm용)
    total_eval = sum(
        len(splits.get(s, []))
        for splits in task_split_datasets.values()
        for s in eval_splits
    )

    with tqdm(total=total_eval, desc="Step 1: Building eval blacklist") as pbar:
        for task_name, splits in task_split_datasets.items():
            family = _get_family(task_name)
            group_id = family if family is not None else task_name

            for split_name in eval_splits:
                ds = splits.get(split_name)
                if ds is None or len(ds) == 0:
                    continue
                mol_strings = ds["input_mol_string"]
                for mol_str in mol_strings:
                    key = extract_entity_key(mol_str)
                    if key is not None:
                        blacklist[group_id].add(key)
                    pbar.update(1)

    print("\n=== [Step 1] Eval Blacklist ===")
    print(f"  (include_validation={include_validation})")
    for group_id, keys in sorted(blacklist.items()):
        print(f"  {group_id}: {len(keys)} unique entity keys")

    return dict(blacklist)


# ---------------------------------------------------------------------------
# Step 2: Eval leakage removal
# ---------------------------------------------------------------------------

def remove_eval_leakage(task_split_datasets, eval_blacklist, include_validation=True):
    """train에서 eval blacklist entity를 제거.

    include_validation=True (기본): val은 eval boundary의 일부이므로 frozen → train만 필터.
    include_validation=False: val도 필터 대상 (test-only blacklist 기준).

    Returns:
        (task_split_datasets, RemovalStats)
    """
    stats = RemovalStats()
    print("\n=== [Step 2] Eval Leakage Removal ===")

    # val이 eval boundary에 포함되면 train만 필터, 아니면 train+val 둘 다 필터
    filter_splits = ["train"] if include_validation else ["train", "val"]

    # 전체 필터 대상 샘플 수 (tqdm용)
    total_filter = sum(
        len(splits.get(s, []))
        for splits in task_split_datasets.values()
        for s in filter_splits
    )

    with tqdm(total=total_filter, desc="Step 2: Checking eval leakage") as pbar:
        for task_name, splits in task_split_datasets.items():
            family = _get_family(task_name)
            group_id = family if family is not None else task_name

            bl = eval_blacklist.get(group_id, set())

            for split_name in filter_splits:
                ds = splits.get(split_name)
                if ds is None or len(ds) == 0:
                    continue

                before = len(ds)
                mol_strings = ds["input_mol_string"]

                keep_indices = []
                for i, mol_str in enumerate(mol_strings):
                    key = extract_entity_key(mol_str)
                    if not bl or key is None or key not in bl:
                        keep_indices.append(i)
                    pbar.update(1)

                removed = before - len(keep_indices)
                if removed > 0:
                    splits[split_name] = ds.select(keep_indices)
                    stats.record(task_name, split_name, "eval_blacklist", removed)
                    tqdm.write(f"  {task_name}/{split_name}: {removed} removed (eval leakage), "
                               f"{len(keep_indices)} remaining")

    return task_split_datasets, stats


# ---------------------------------------------------------------------------
# Step 3: Cross-source dedup within family
# ---------------------------------------------------------------------------

def dedup_within_family(task_split_datasets):
    """같은 entity family에 속하는 task들 간 cross-source 중복 제거.

    REMOVE_ON_CONFLICT에 지정된 task 쪽에서 중복 샘플을 제거한다.
    이것은 quality superiority가 아니라 benchmark compatibility anchor이다.

    split-aware 정책:
    - eval split(test, val)은 건드리지 않음
    - train 간 중복만 priority로 제거

    Returns:
        (task_split_datasets, RemovalStats)
    """
    stats = RemovalStats()
    print("\n=== [Step 3] Cross-Source Dedup Within Families ===")

    # family별로 task 목록 수집
    family_tasks = defaultdict(list)
    for task_name in task_split_datasets:
        family = _get_family(task_name)
        if family is not None:
            family_tasks[family].append(task_name)

    for family, tasks in family_tasks.items():
        if len(tasks) < 2:
            continue

        remove_task = REMOVE_ON_CONFLICT.get(family)
        if remove_task is None or remove_task not in tasks:
            continue

        # anchor task(유지할 task)의 train entity keys 수집
        anchor_tasks = [t for t in tasks if t != remove_task]
        anchor_keys = set()

        for at in anchor_tasks:
            train_ds = task_split_datasets[at].get("train")
            if train_ds is None or len(train_ds) == 0:
                continue
            for mol_str in tqdm(train_ds["input_mol_string"],
                                desc=f"  Step 3: Collecting anchor keys ({at})"):
                key = extract_entity_key(mol_str)
                if key is not None:
                    anchor_keys.add(key)

        if not anchor_keys:
            continue

        # remove_task의 train에서 anchor_keys에 있는 샘플 제거
        remove_splits = task_split_datasets.get(remove_task, {})
        train_ds = remove_splits.get("train")
        if train_ds is None or len(train_ds) == 0:
            continue

        before = len(train_ds)
        keep_indices = []
        for i, mol_str in enumerate(tqdm(train_ds["input_mol_string"],
                                          desc=f"  Step 3: Dedup {remove_task} train")):
            key = extract_entity_key(mol_str)
            if key is None or key not in anchor_keys:
                keep_indices.append(i)

        removed = before - len(keep_indices)
        if removed > 0:
            remove_splits["train"] = train_ds.select(keep_indices)
            stats.record(remove_task, "train", "within_family_dup", removed)
            print(f"  {family}: removed {removed} from '{remove_task}' train "
                  f"(overlapping with {anchor_tasks}), {len(keep_indices)} remaining")

    return task_split_datasets, stats


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_decontamination_pipeline(task_split_datasets, include_validation_in_blacklist=True):
    """Cross-source decontamination 전체 파이프라인.

    Args:
        task_split_datasets: {task_name: {"train": Dataset, "val": Dataset, "test": Dataset}}
            각 Dataset은 HuggingFace datasets.Dataset 객체.
            "input_mol_string" 컬럼 필수.
        include_validation_in_blacklist: True이면 validation도 eval boundary에 포함 (기본=True)

    Returns:
        Decontamination 처리된 task_split_datasets
    """
    print("\n" + "=" * 70)
    print("[Decontamination Pipeline] Cross-source decontamination")
    print("  This is a source-compatible benchmark builder,")
    print("  NOT a strict chemistry split pipeline.")
    print("=" * 70)

    # 전체 통계 (before)
    total_before = {}
    for task_name, splits in task_split_datasets.items():
        for split_name, ds in splits.items():
            if ds is not None:
                total_before[(task_name, split_name)] = len(ds)

    # Step 1: Eval blacklist
    eval_blacklist = build_eval_blacklist(
        task_split_datasets,
        include_validation=include_validation_in_blacklist,
    )

    # Step 2: Eval leakage removal
    task_split_datasets, leakage_stats = remove_eval_leakage(
        task_split_datasets, eval_blacklist,
        include_validation=include_validation_in_blacklist,
    )

    # Step 3: Cross-source dedup within families
    task_split_datasets, family_stats = dedup_within_family(task_split_datasets)

    # --- Contamination Audit Report ---
    print("\n" + "=" * 70)
    print("[Contamination Audit Report]")
    print("=" * 70)

    leakage_stats.print_report("Eval Leakage Removals")
    family_stats.print_report("Within-Family Dedup Removals")

    # Overall summary
    print(f"\n{'Task':<45} {'Split':<8} {'Before':>8} {'After':>8} {'Removed':>8}")
    print("-" * 80)
    total_removed = 0
    for (task_name, split_name), before in sorted(total_before.items()):
        ds = task_split_datasets.get(task_name, {}).get(split_name)
        after = len(ds) if ds is not None else 0
        removed = before - after
        total_removed += removed
        if removed > 0:
            print(f"{task_name:<45} {split_name:<8} {before:>8} {after:>8} {removed:>8}")
    print("-" * 80)
    print(f"{'TOTAL REMOVED':<63} {total_removed:>8}")
    print("=" * 70)

    return task_split_datasets
