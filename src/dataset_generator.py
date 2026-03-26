# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import re
import traceback
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import yaml
import deepchem as dc
import selfies as sf
import torch
import datasets
from datasets import load_dataset
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

import instructions_smol
import model.added_tokens as added_tokens
from benchmark_constants import (
    CLASSIFICATION_BENCHMARKS,
    MOL2TEXT_BENCHMARKS,
    REGRESSION_BENCHMARKS,
    REACTION_BENCHMARKS,
    TEXT2MOL_BENCHMARKS,
)

# -----------------------------------------------------------------------------
# [Helper Functions]
# -----------------------------------------------------------------------------

def clean_mol_string(mol_str):
    """분자 문자열에서 세미콜론(;)을 점(.)으로 변환 — 다중 분자 구분자 정규화"""
    return re.sub(r"\s*;\s*", ".", str(mol_str))


_DUMMY_GRAPH = None

def get_dummy_graph():
    """smiles2data('CC') 결과를 캐싱하여 반복 호출 방지"""
    global _DUMMY_GRAPH
    if _DUMMY_GRAPH is None:
        _DUMMY_GRAPH = smiles2data('CC')
    return _DUMMY_GRAPH


def wrap_label(label, task):
    if task in CLASSIFICATION_BENCHMARKS:
        label_tokens = added_tokens.BOOL
    elif task in REGRESSION_BENCHMARKS:
        label_tokens = added_tokens.FLOAT
    elif task in ["smol-name_conversion-s2f", "smol-name_conversion-i2f"]:
        label_tokens = added_tokens.MOLFORMULA
    elif task == "smol-name_conversion-s2i":
        label_tokens = added_tokens.IUPAC
    elif task in MOL2TEXT_BENCHMARKS:
        label_tokens = added_tokens.DESCRIPTION
    elif task == "smol-name_conversion-i2s":
        label_tokens = added_tokens.SMILES
    elif task in TEXT2MOL_BENCHMARKS + REACTION_BENCHMARKS:
        label_tokens = added_tokens.SMILES
    else:
        raise NotImplementedError(f"Task {task} is not implemented in wrap_label")

    if task in CLASSIFICATION_BENCHMARKS:
        if isinstance(label, str):
            if "true" in label.lower() or "yes" in label.lower():
                label = "True"
            elif "false" in label.lower() or "no" in label.lower():
                label = "False"
            else:
                label = "False"
            label = label_tokens[0] + " " + label + " " + label_tokens[1]
        elif isinstance(label, list):
            label_language = ", ".join(label)
            label_boolean = "True" * len(label)
            label = label_language + label_tokens[0] + " " + label_boolean + " " + label_tokens[1]
        else:
            label = "True" if label else "False"
            label = label_tokens[0] + " " + label + " " + label_tokens[1]
        return label
    elif task in REGRESSION_BENCHMARKS:
        if isinstance(label, float) or isinstance(label, int):
            label = "{:.10f}".format(float(label))
        else:
            try:
                label = format(float(label), ".10f")
            except Exception:
                label = str(label)

        if "-" not in label and "+" not in label:
            label = "+" + label
        label = label[:7]
        converted_label = "".join([f"<|{char}|>" for char in label])
        return label_tokens[0] + " " + converted_label + " " + label_tokens[1]

    elif task in REACTION_BENCHMARKS + MOL2TEXT_BENCHMARKS + TEXT2MOL_BENCHMARKS + ["smol-name_conversion-i2s"]:
        return label_tokens[0] + str(label) + label_tokens[1]
    else:
        return str(label)


def smiles2data(smiles):
    from ogb.utils import smiles2graph
    try:
        graph = smiles2graph(smiles)
        x = torch.from_numpy(graph["node_feat"])
        edge_index = torch.from_numpy(graph["edge_index"])
        edge_attr = torch.from_numpy(graph["edge_feat"])
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data
    except Exception as e:
        raise ValueError(f"Failed to convert SMILES to graph: {smiles}. Error: {e}")

def get_canonical_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol) # Canonical SMILES
        else:
            return None
    except Exception:
        return None

def selfies_to_smiles(selfies_str):
    """SELFIES 문자열을 SMILES로 변환. 실패 시 None 반환."""
    try:
        smiles = sf.decoder(selfies_str)
        if smiles is None:
            return None
        canonical = get_canonical_smiles(smiles)
        return canonical if canonical else smiles
    except Exception:
        return None

def from_dict(d):
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    return Struct(**d)

def get_task_subtask_info(target_benchmarks):
    task_subtask_dict = {}
    for task in target_benchmarks:
        if isinstance(task, str):
            task_subtask_dict[task] = [0]
        else:
            task_subtask_dict.update(task)
    task_subtask_pairs = [
        (task, subtask)
        for task, subtasks in task_subtask_dict.items()
        for subtask in subtasks
    ]
    return task_subtask_dict, task_subtask_pairs

def _get_num_workers(num_workers_arg):
    """num_workers CLI 인자를 실제 worker 수로 변환. 0이면 os.cpu_count() 사용."""
    if num_workers_arg <= 0:
        return min(os.cpu_count() or 1, 64)
    return min(num_workers_arg, 64)

# -----------------------------------------------------------------------------
# [Top-level functions for multiprocessing (pickle-safe)]
# -----------------------------------------------------------------------------

def _process_moleculenet_sample(args):
    """MoleculeNetDatasetDeepChem 샘플 처리 (ProcessPoolExecutor용)"""
    index, smiles, raw_output, subtask_idx, task, instruction_templates, label_full_name = args
    if smiles is None:
        return None

    try:
        instruction = np.random.choice(instruction_templates)
        canonical = get_canonical_smiles(smiles)
        if canonical is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        input_mol_string = canonical

        input_mol_string = added_tokens.SMILES[0] + " " + input_mol_string + " " + added_tokens.SMILES[1]

        if subtask_idx == "multi_label_classification":
            label = raw_output
            label = [label_full_name[i] for i in range(len(label)) if label[i]]
            if len(label) == 0:
                label = "No toxicity identified. " + wrap_label("False", task)
            else:
                label = wrap_label(label, task)
        else:
            label = raw_output
            label = wrap_label(label, task)

        graph = [smiles2data(smiles), get_dummy_graph()]
        return (graph, label, input_mol_string, instruction)
    except Exception:
        return None


def _process_molinstruction_sample(args):
    """MolInstructionDataset 샘플 처리 (ProcessPoolExecutor용)

    Mol-Instructions 데이터셋은 SMILES 형태로 input/output 제공.
    BACE 데이터셋은 SELFIES 형태 → SMILES로 변환 필요.
    """
    index, input_, label_, task, instruction_templates, is_bace = args
    try:
        instruction = np.random.choice(instruction_templates)

        if pd.isna(input_) or pd.isna(label_):
            raise ValueError("Input or Label is NaN")

        input_ = str(input_)
        label_ = str(label_)

        # BACE는 SELFIES로 제공되므로 SMILES로 변환
        if is_bace:
            smiles = selfies_to_smiles(input_)
            if smiles is None:
                raise ValueError(f"SELFIES→SMILES conversion failed: {input_}")
            input_ = smiles

        if task in REACTION_BENCHMARKS:
            if task in ["reagent_prediction"]:
                if ">>" in input_:
                    list_smiles = input_.split(">>")
                    input_mol_string = input_.replace(">>", f"{added_tokens.SMILES[1]}{added_tokens.REACTION_DIRECTION[0]}{added_tokens.SMILES[0]}")
                    graph = [smiles2data(s.strip()) for s in list_smiles]
                elif "|>>|" in input_:
                    list_smiles = input_.split("|>>|")
                    input_mol_string = input_
                    graph = [smiles2data(s.strip()) for s in list_smiles]
                else:
                    raise ValueError(f"Invalid reagent format: {input_}")
            else:
                input_mol_string = input_
                graph = [smiles2data(input_mol_string.strip()), get_dummy_graph()]
        else:
            input_mol_string = input_
            if get_canonical_smiles(input_mol_string.strip()) is None:
                raise ValueError("Invalid SMILES")
            graph = [smiles2data(input_mol_string.strip()), get_dummy_graph()]

        label = wrap_label(label_, task)
        input_mol_string = added_tokens.SMILES[0] + " " + input_mol_string + " " + added_tokens.SMILES[1]
        return (graph, label, input_mol_string, instruction, str(input_))
    except Exception:
        return None


def _process_smol_sample(args):
    """SMolInstructDataset 샘플 처리 (ProcessPoolExecutor용)

    SMolInstruct를 use_selfies=False로 로드하므로 SMILES 형태로 제공됨.
    """
    index, raw_input, raw_output, task, instruction_templates, data_row = args
    try:
        label = raw_output

        if task in ["smol-name_conversion-i2s", "smol-name_conversion-i2f"]:
            s_token, e_token = added_tokens.IUPAC
            description = s_token + str(raw_input) + e_token
            instruction = np.random.choice(instruction_templates).replace("<INPUT>", description)

            dummy = get_dummy_graph()
            graph = [dummy, dummy]

            input_mol_string = "<None>"
            label = clean_mol_string(str(label))

        elif task in ["smol-name_conversion-s2i", "smol-name_conversion-s2f"]:
            instruction = np.random.choice(instruction_templates)
            input_mol_string = clean_mol_string(str(raw_input))

            try:
                canonical = get_canonical_smiles(input_mol_string)
                if canonical is None:
                    canonical = "CC"
                graph = [smiles2data(canonical), get_dummy_graph()]
            except Exception:
                dummy = get_dummy_graph()
                graph = [dummy, dummy]

            label = str(label)

        elif task in TEXT2MOL_BENCHMARKS:
            s_token, e_token = added_tokens.DESCRIPTION
            description = s_token + raw_input + e_token
            instruction = np.random.choice(instruction_templates).replace("<INPUT>", description)

            dummy = get_dummy_graph()
            graph = [dummy, dummy]
            input_mol_string = "<None>"
            label = clean_mol_string(label)

        elif task in REACTION_BENCHMARKS:
            instruction = np.random.choice(instruction_templates)
            input_mol_string = clean_mol_string(raw_input)
            try:
                canonical = get_canonical_smiles(input_mol_string)
                if canonical is None:
                    raise ValueError("Decode failed")
                graph = [smiles2data(canonical), get_dummy_graph()]
            except Exception:
                raise ValueError(f"Reaction SMILES invalid: {input_mol_string}")

        elif task in ["smol-property_prediction-sider"]:
            instance_input = data_row["input"] if data_row else raw_input
            instruction = re.sub(r"\[.*\]", "<INPUT>", instance_input)
            input_mol_string = clean_mol_string(raw_input)
            canonical = get_canonical_smiles(input_mol_string)
            if canonical is None:
                raise ValueError(f"Invalid SMILES: {input_mol_string}")
            graph = [smiles2data(canonical), get_dummy_graph()]

        elif task in MOL2TEXT_BENCHMARKS + CLASSIFICATION_BENCHMARKS + REGRESSION_BENCHMARKS:
            instruction = np.random.choice(instruction_templates)
            input_mol_string = clean_mol_string(raw_input)
            canonical = get_canonical_smiles(input_mol_string)
            if canonical is None:
                raise ValueError(f"Invalid SMILES: {input_mol_string}")
            graph = [smiles2data(canonical), get_dummy_graph()]

        else:
            raise NotImplementedError

        label = wrap_label(label, task)
        input_mol_string = added_tokens.SMILES[0] + " " + input_mol_string + " " + added_tokens.SMILES[1]
        return (graph, label, input_mol_string, instruction)
    except Exception:
        return None


def _process_chebi_sample(args):
    """ChEBIDataset 샘플 처리 (ProcessPoolExecutor용)"""
    index, desc, mol_data, is_selfies, task, instruction_templates = args
    try:
        instruction = np.random.choice(instruction_templates)

        if pd.isna(desc) or pd.isna(mol_data):
            raise ValueError("NaN data detected")

        # SMILES/SELFIES 처리 → 최종적으로 SMILES 사용
        if is_selfies:
            smiles = selfies_to_smiles(mol_data)
            if smiles is None:
                raise ValueError(f"SELFIES→SMILES conversion failed")
        else:
            smiles = mol_data
            canonical = get_canonical_smiles(smiles)
            if canonical:
                smiles = canonical

        if smiles is None:
            raise ValueError("SMILES is None")

        if task in TEXT2MOL_BENCHMARKS:
            label = smiles  # 라벨도 SMILES
            description = added_tokens.DESCRIPTION[0] + str(desc) + added_tokens.DESCRIPTION[1]
            instruction = instruction.replace("<INPUT>", description)
            dummy = get_dummy_graph()
            graph = [dummy, dummy]
            input_mol_string = "<None>"
        elif task in MOL2TEXT_BENCHMARKS:
            label = str(desc)
            input_mol_string = smiles
            graph = [smiles2data(smiles), get_dummy_graph()]
        else:
            raise NotImplementedError(f"Unsupported ChEBI task: {task}")

        label = wrap_label(label, task)
        input_mol_string = added_tokens.SMILES[0] + " " + input_mol_string + " " + added_tokens.SMILES[1]
        return (graph, label, input_mol_string, instruction)
    except Exception:
        return None


def _parallel_process(process_fn, args_list, num_workers, desc="Processing"):
    """범용 병렬 처리 래퍼. num_workers=1이면 sequential 실행."""
    if num_workers <= 1:
        results = []
        for args in tqdm(args_list, desc=desc):
            results.append(process_fn(args))
        return results

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(process_fn, args_list, chunksize=max(1, len(args_list) // (num_workers * 4))),
            total=len(args_list),
            desc=desc
        ))
    return results


# -----------------------------------------------------------------------------
# [Dataset Classes]
# -----------------------------------------------------------------------------

class MoleculeNetDatasetDeepChem(Dataset):
    def __init__(self, data, task_subtask_pair, subtask_idx=0, prompt=None, num_workers=1):
        self.data = data
        self.subtask_idx = subtask_idx
        self.task_subtask_pair = task_subtask_pair
        self.num_workers = num_workers
        if "/" in task_subtask_pair:
            self.task, self.subtask = task_subtask_pair.split("/", 1)
        else:
            self.task = task_subtask_pair
            self.subtask = str(subtask_idx)

        if self.task in CLASSIFICATION_BENCHMARKS:
            self.instruction_templates = getattr(instructions_smol, self.task)
        elif self.task in REGRESSION_BENCHMARKS:
            if self.task in ["qm9_additional_label"]:
                subtask_full_name_dict = {
                    "mu": "dipole_moment", "alpha": "isotropic_polarizability", "r2": "electronic_spatial_extent",
                    "zpve": "zero_point_vibrational_energy", "cv": "heat_capacity_298K",
                    "u298": "internal_energy_298K", "h298": "enthalpy_298K", "g298": "free_energy_298K",
                }
                task = self.task.replace("_additional_label", "")
                subtask_full_name = subtask_full_name_dict[self.subtask]
                self.instruction_templates = getattr(instructions_smol, f"{task}_{subtask_full_name}")
            else:
                self.instruction_templates = getattr(instructions_smol, f"{self.task}_{self.subtask}".lower())
        else:
            raise NotImplementedError
        self.set_necessary_data()

    def set_label_fullname(self):
        self.label_full_name = None
        if self.task == "tox21":
            self.label_full_name = [
                "androgen receptor, full (AR, full)", "androgen receptor, LBD (AR, LBD)", "aryl hydrocarbon receptor (AhR)",
                "aromatase", "estrogen receptor alpha, full (ER, full)", "estrogen receptor alpha, LBD (ER, LBD)",
                "peroxisome proliferator-activated receptor gamma (PPAR-gamma)",
                "nuclear factor (erythroid-derived 2)-like 2/antioxidant responsive element (Nrf2/ARE)",
                "ATPase family AAA domain containing 5 (ATAD5)", "heat shock factor response element (HSE)",
                "mitochondrial membrane potential (MMP)", "tumor suppressor protein p53",
            ]

    def set_necessary_data(self):
        self.raw_inputs = self.data.X
        if self.subtask_idx == "multi_label_classification":
            self.set_label_fullname()
            self.raw_outputs = self.data.y
        else:
            self.raw_outputs = self.data.y[:, self.subtask_idx]

        smiles_list = []
        for mol in self.raw_inputs:
            try:
                s = Chem.MolToSmiles(mol)
                smiles_list.append(s)
            except Exception:
                smiles_list.append(None)

        label_full_name = self.label_full_name if hasattr(self, 'label_full_name') else None
        args_list = [
            (i, smiles_list[i], self.raw_outputs[i], self.subtask_idx, self.task,
             self.instruction_templates, label_full_name)
            for i in range(len(self.raw_inputs))
        ]

        results = _parallel_process(
            _process_moleculenet_sample, args_list, self.num_workers,
            desc=f"{self.task}-{self.subtask_idx}"
        )

        self.count_invalid_smiles = sum(1 for r in results if r is None)
        valid = [r for r in results if r is not None]
        self.graph_list = [r[0] for r in valid]
        self.label_list = [r[1] for r in valid]
        self.input_mol_string_list = [r[2] for r in valid]
        self.instruction_list = [r[3] for r in valid]

    def __len__(self): return len(self.label_list)
    def __getitem__(self, index):
        return self.graph_list[index], self.label_list[index], self.input_mol_string_list[index], self.task_subtask_pair, self.instruction_list[index]


class MolInstructionDataset(Dataset):
    def __init__(self, data, task_subtask_pair, num_workers=1, **kwargs):
        self.data = data
        self.task_subtask_pair = task_subtask_pair
        self.num_workers = num_workers
        if "/" in task_subtask_pair:
            self.task, self.subtask = task_subtask_pair.split("/", 1)
        else:
            self.task = task_subtask_pair
            self.subtask = "0"
        self.set_necessary_data()

    def set_necessary_data(self):
        is_bace = self.task == "bace"
        if is_bace:
            input_list = self.data["SELFIES"][:]  # BACE는 SELFIES → SMILES 변환 필요
            label_list = self.data["label"][:]
        else:
            input_list = self.data["input"][:]  # Mol-Instructions는 SMILES 제공
            label_list = self.data["output"][:]

        self.instruction_templates = getattr(instructions_smol, self.task)

        args_list = [
            (i, input_list[i], label_list[i], self.task, self.instruction_templates, is_bace)
            for i in range(len(input_list))
        ]

        results = _parallel_process(
            _process_molinstruction_sample, args_list, self.num_workers,
            desc=self.task
        )

        self.count_invalid_smiles = sum(1 for r in results if r is None)
        valid = [r for r in results if r is not None]
        self.graph_list = [r[0] for r in valid]
        self.label_list = [r[1] for r in valid]
        self.input_mol_string_list = [r[2] for r in valid]
        self.instruction_list = [r[3] for r in valid]
        self.input_list = [r[4] for r in valid]

    def __len__(self): return len(self.label_list)

    def __getitem__(self, index):
        return self.graph_list[index], self.label_list[index], self.input_mol_string_list[index], self.task, self.instruction_list[index]


class SMolInstructDataset(Dataset):
    def __init__(self, data, task_subtask_pair, num_workers=1, **kwargs):
        self.data = data
        self.task_subtask_pair = task_subtask_pair
        self.num_workers = num_workers
        if "/" in task_subtask_pair:
            self.task, self.subtask = task_subtask_pair.split("/", 1)
        else:
            self.task = task_subtask_pair
            self.subtask = "0"
        if "forward_synthesis" in self.task:
            self.instruction_templates = getattr(instructions_smol, "forward_reaction_prediction")
        else:
            self.instruction_templates = getattr(instructions_smol, self.task.replace("smol-", "").replace("-", "_"))
        self.set_necessary_data()

    def set_necessary_data(self):
        raw_inputs = self.data["raw_input"][:]
        raw_outputs = self.data["raw_output"][:]

        print(f"[{self.task}] Start Processing. Raw Data Size: {len(self.data)}")

        # sider의 경우 data_row가 필요하므로 별도 처리
        if self.task in ["smol-property_prediction-sider"]:
            args_list = [
                (i, raw_inputs[i], raw_outputs[i], self.task, self.instruction_templates, self.data[i])
                for i in range(len(self.data))
            ]
        else:
            args_list = [
                (i, raw_inputs[i], raw_outputs[i], self.task, self.instruction_templates, None)
                for i in range(len(self.data))
            ]

        results = _parallel_process(
            _process_smol_sample, args_list, self.num_workers,
            desc=self.task
        )

        self.count_invalid_smiles = sum(1 for r in results if r is None)
        valid = [r for r in results if r is not None]

        print(f"[{self.task}] Finished. Total: {len(self.data)} | Success: {len(valid)} | Failed: {self.count_invalid_smiles}")

        self.graph_list = [r[0] for r in valid]
        self.label_list = [r[1] for r in valid]
        self.input_mol_string_list = [r[2] for r in valid]
        self.instruction_list = [r[3] for r in valid]

    def __len__(self): return len(self.label_list)

    def __getitem__(self, index):
        return self.graph_list[index], self.label_list[index], self.input_mol_string_list[index], self.task_subtask_pair, self.instruction_list[index]


# =============================================================================
# [ChEBIDataset: liupf/ChEBI-20-MM 호환]
# =============================================================================
class ChEBIDataset(Dataset):
    def __init__(self, data, task_subtask_pair, num_workers=1, **kwargs):
        self.data = data
        self.task_subtask_pair = task_subtask_pair
        self.num_workers = num_workers
        if "/" in task_subtask_pair:
            self.task, self.subtask = task_subtask_pair.split("/", 1)
        else:
            self.task = task_subtask_pair
            self.subtask = "0"
        self.set_necessary_data()

    def set_necessary_data(self):
        # 1. 컬럼 확인
        if hasattr(self.data, "column_names"):
            cols = self.data.column_names
        elif hasattr(self.data, "columns"):
            cols = self.data.columns
        else:
            cols = []

        print(f"[Debug] Processing Task: {self.task} | Columns: {cols}")

        # 2. Description 컬럼 매핑
        if "description" in cols:
            description_list = self.data["description"]
        elif "text" in cols:
            description_list = self.data["text"]
        elif "caption" in cols:
            description_list = self.data["caption"]
        else:
            raise ValueError(f"Column 'description' not found. Available: {cols}")

        # 3. Molecule (SMILES) 컬럼 매핑
        is_selfies = False
        if "SMILES" in cols:
            mol_list = self.data["SMILES"]
        elif "smiles" in cols:
            mol_list = self.data["smiles"]
        elif "SELFIES" in cols:
            mol_list = self.data["SELFIES"]
            is_selfies = True
        else:
            raise ValueError(f"Molecule column (SMILES/smiles/SELFIES) not found. Available: {cols}")

        if "mol2text" in self.task:
            self.instruction_templates = getattr(instructions_smol, "molecule_captioning")
        elif "text2mol" in self.task:
            self.instruction_templates = getattr(instructions_smol, "molecule_generation")

        total_len = len(description_list)
        args_list = [
            (i, description_list[i], mol_list[i], is_selfies, self.task, self.instruction_templates)
            for i in range(total_len)
        ]

        results = _parallel_process(
            _process_chebi_sample, args_list, self.num_workers,
            desc=self.task
        )

        self.count_invalid_smiles = sum(1 for r in results if r is None)
        valid = [r for r in results if r is not None]

        print(f"[{self.task}] Finished. Valid: {len(valid)}, Invalid: {self.count_invalid_smiles}")

        self.graph_list = [r[0] for r in valid]
        self.label_list = [r[1] for r in valid]
        self.input_mol_string_list = [r[2] for r in valid]
        self.instruction_list = [r[3] for r in valid]

    def __len__(self): return len(self.label_list)

    def __getitem__(self, index):
        return self.graph_list[index], self.label_list[index], self.input_mol_string_list[index], self.task_subtask_pair, self.instruction_list[index]


# =============================================================================
# [get_dataset: 데이터셋 로딩]
# =============================================================================
def get_dataset(task_name, raw_data_root):
    if "chebi-20" in task_name:
        print(f"[Info] Loading Standard ChEBI-20 from 'liupf/ChEBI-20-MM' for {task_name}")
        try:
            dataset = load_dataset("liupf/ChEBI-20-MM")

            train_dataset = dataset["train"]
            valid_dataset = dataset["validation"]
            test_dataset = dataset["test"]

            tasks = [task_name]
            return tasks, train_dataset, valid_dataset, test_dataset

        except Exception as e:
            print(f"[Warning] Failed to load liupf/ChEBI-20-MM: {e}. Fallback to local CSV.")
            train_dataset = pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_chebi20_train.csv"))
            valid_dataset = pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_chebi20_valid.csv"))
            test_dataset = pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_chebi20_test.csv"))
            tasks = [task_name]
            return tasks, train_dataset, valid_dataset, test_dataset

    elif "smol" in task_name:
        # use_selfies=False → SMILES 형태로 데이터 수령
        smol_dataset = load_dataset(
            "osunlp/SMolInstruct",
            use_selfies=False,
            insert_core_tags=False,
            trust_remote_code=True,
        )
        _task = re.sub("smol-", "", task_name)

        train_dataset = smol_dataset["train"].filter(lambda x: x["task"] == _task)
        valid_dataset = smol_dataset["validation"].filter(lambda x: x["task"] == _task)
        test_dataset = smol_dataset["test"].filter(lambda x: x["task"] == _task)
        tasks = [task_name]

    elif task_name in ["toxcast", "tox21", "hopv", "qm9_additional_label"]:
        loading_fn = getattr(dc.molnet, f"load_{task_name}" if task_name != "qm9_additional_label" else "load_qm9")
        base_path = f"dataset/{task_name}"
        os.makedirs(base_path, exist_ok=True)
        tasks, datasets_, transformers = loading_fn(featurizer="Raw", splitter="scaffold", save_dir=base_path, data_dir=base_path, reload=True)
        train_dataset, valid_dataset, test_dataset = datasets_

    elif task_name == "bace":
        train_dataset = pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_bace_train.csv"))
        valid_dataset = pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_bace_valid.csv"))
        test_dataset = pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_bace_test.csv"))
        tasks = [task_name]

    elif task_name in ["reagent_prediction", "forward_reaction_prediction", "retrosynthesis", "qm9_homo", "qm9_lumo", "qm9_homo_lumo_gap"]:
        mol_instruction_dataset = load_dataset("zjunlp/Mol-Instructions", "Molecule-oriented Instructions", trust_remote_code=True)
        if "qm9_" in task_name:
            dataset = mol_instruction_dataset["property_prediction"]
            subtask_name = task_name.replace("qm9_", "")
            subtask_instruction_templates = getattr(instructions_smol, "filtering_template_" + subtask_name)
            dataset = dataset.filter(lambda x: x["instruction"] in subtask_instruction_templates)
        else:
            dataset = mol_instruction_dataset[task_name]

        train_dataset = dataset.filter(lambda x: "train" in x["metadata"])
        split = train_dataset.train_test_split(test_size=0.02, shuffle=True)
        train_dataset, valid_dataset = split["train"], split["test"]
        test_dataset = dataset.filter(lambda x: "test" in x["metadata"])
        tasks = [task_name]
    else:
        raise NotImplementedError(f"Task {task_name} not supported in get_dataset")

    return tasks, train_dataset, valid_dataset, test_dataset


def prepare_data_instance(
        data_instance,
        system_prompt,
        llm_model_name="mistral",
        mol_token="<mol>",
        num_query_tokens=32,
):
    input_mol_string = data_instance["input_mol_string"]
    input_mol_string = re.sub(r"<SMILES>\s*", "<SMILES> ", input_mol_string)
    input_mol_string = re.sub(r"\s*</SMILES>", " </SMILES>", input_mol_string)

    input_prompt = data_instance["instruction"]

    if "<INPUT>" in input_prompt:
        input_prompt = input_prompt.replace("<INPUT>", input_mol_string)

    graph_sequence = "<GRAPH> " + mol_token * num_query_tokens + " </GRAPH>"
    input_prompt += graph_sequence

    is_llada = "llada" in llm_model_name.lower() or "llama-3" in llm_model_name.lower()

    if is_llada:
        formatted_prompt_text = (
            "<|startoftext|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            + system_prompt + "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            + input_prompt + "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        formatted_target_text = data_instance["label"] + "<|eot_id|>"

    else:
        formatted_prompt_text = "<s>[INST] " + system_prompt + " \n\n" + input_prompt + " [/INST] "
        formatted_target_text = data_instance["label"] + " </s>"

    raw_task = data_instance["task_subtask_pair"]

    if "qm9_additional_label" in raw_task:
        convert_dict = {
            'qm9_additional_label/mu' : "qm9_dipole_moment",
            'qm9_additional_label/alpha' : "qm9_isotropic_polarizability",
            'qm9_additional_label/r2' : "qm9_electronic_spatial_extent",
            'qm9_additional_label/zpve' : "qm9_zero_point_vibrational_energy",
            'qm9_additional_label/cv' : "qm9_heat_capacity_298K",
            'qm9_additional_label/u298' : "qm9_internal_energy_298K",
            'qm9_additional_label/h298' : "qm9_enthalpy_298K",
            'qm9_additional_label/g298' : "qm9_free_energy_298K",
        }
        task = convert_dict.get(raw_task, raw_task)
    elif raw_task.endswith("/0"):
        task = raw_task[:-2]
    elif "/multi_label_classification" in raw_task:
         task = raw_task.split("/")[0]
    else:
        task = raw_task

    data = {
        "task": task,
        "x": data_instance["x"],
        "edge_index": data_instance["edge_index"],
        "edge_attr": data_instance["edge_attr"],
        "additional_x": data_instance["additional_x"],
        "additional_edge_index": data_instance["additional_edge_index"],
        "additional_edge_attr": data_instance["additional_edge_attr"],
        "prompt_text": formatted_prompt_text,
        "target_text": formatted_target_text,
        "input_mol_string": input_mol_string,
    }
    return data


def dataset_to_arrow_dicts(ds):
    """Dataset 객체를 HuggingFace Arrow 직렬화용 dict 리스트로 변환."""
    list_dict_data = []
    for i in range(len(ds)):
        try:
            graph, label, input_mol_string, task_pair_or_name, instruction = ds[i]
            if hasattr(instruction, "item"):
                instruction = instruction.item()
            instruction = str(instruction)

            if isinstance(graph, list) and len(graph) >= 2:
                g0, g1 = graph[0], graph[1]
            elif isinstance(graph, list) and len(graph) == 1:
                g0 = graph[0]
                g1 = get_dummy_graph()
            else:
                g0 = graph
                g1 = get_dummy_graph()

            list_dict_data.append({
                "x": g0.x, "edge_index": g0.edge_index, "edge_attr": g0.edge_attr,
                "label": label, "input_mol_string": input_mol_string,
                "task_subtask_pair": task_pair_or_name, "instruction": instruction,
                "additional_x": g1.x, "additional_edge_index": g1.edge_index, "additional_edge_attr": g1.edge_attr,
            })
        except Exception:
            continue
    return list_dict_data


# -----------------------------------------------------------------------------
# [Main Execution Block]
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="./configs/download/")
    parser.add_argument("--config", type=str, default="default_llada")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for parallel processing. 0 = os.cpu_count()")
    args = parser.parse_args()

    num_workers = _get_num_workers(args.num_workers)
    print(f"[Config] Using {num_workers} workers for parallel processing")

    arg_path = os.path.join(args.config_dir, args.config) + ".yaml"
    with open(arg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = from_dict(cfg)

    raw_data_root = cfg.raw_data_root
    data_tag = cfg.data_tag
    tag_root = os.path.join(raw_data_root, data_tag)
    if not os.path.exists(tag_root):
        os.makedirs(tag_root)
    start_time = time.time()
    task_subtask_dict, task_subtask_pairs = get_task_subtask_info(cfg.target_benchmarks)

    # --- Dataset Downloading & Saving (First Stage) ---
    for task_subtask_pair in tqdm(task_subtask_pairs, desc="Downloading task_subtask_pairs"):
        task_name = task_subtask_pair[0]
        subtask_idx = task_subtask_pair[1]

        # 이미 처리된 데이터셋(Train Split 기준)이 존재하면 건너뛰기
        check_train_path = os.path.join(tag_root, f"{task_name}_subtask-{subtask_idx}_train")
        if os.path.exists(check_train_path):
            continue
        try:
            new_dataset = get_dataset(task_name=task_name, raw_data_root=raw_data_root)
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

        def _task_arg_for(dataset_cls):
            return task_name if dataset_cls is MolInstructionDataset else pair_name

        dataset_cls = (
            SMolInstructDataset if "smol" in task_name else
            MoleculeNetDatasetDeepChem if task_name in ["toxcast", "tox21", "qm9_additional_label", "hopv"] else
            ChEBIDataset if task_name in ["chebi-20-mol2text", "chebi-20-text2mol"] else
            MolInstructionDataset
        )

        dataset_wrapper = dataset_cls

        dataset_splits = {
            "train": data_split[0],
            "val": data_split[1],
            "test": data_split[2]
        }

        process_order = ["train", "val", "test"]
        common_features = None

        for split in process_order:
            if task_name in ["smol-name_conversion-i2s", "smol-name_conversion-s2i"] and split != "train":
                continue

            raw_data = dataset_splits[split]

            if raw_data is None or len(raw_data) == 0:
                continue

            ds = dataset_wrapper(
                data=raw_data,
                task_subtask_pair=_task_arg_for(dataset_cls),
                subtask_idx=subtask_idx,
                num_workers=num_workers,
            )

            list_dict_data = dataset_to_arrow_dicts(ds)

            save_path = os.path.join(tag_root, f"{task_name}_subtask-{subtask_idx}_{split}")

            if len(list_dict_data) > 0:
                output_dataset = datasets.Dataset.from_list(list_dict_data)
                if common_features is None:
                    common_features = output_dataset.features
                output_dataset.save_to_disk(save_path)
            else:
                if common_features is not None:
                    output_dataset = datasets.Dataset.from_dict({}, features=common_features)
                    output_dataset.save_to_disk(save_path)

    # --- Concatenate & Mapping (Second Stage) ---
    print("\n--- Starting Concatenation and Mapping ---")
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

    system_prompt = "You are a helpful assistant for molecular chemistry, to address tasks including molecular property classification, molecular property regression, chemical reaction prediction, molecule captioning, molecule generation."

    llm_model = cfg.llm_model
    mol_representation = "string+graph"
    num_query_token = 32
    base_model = llm_model.replace("/", "-")
    tags = [base_model, mol_representation]
    if "graph" in mol_representation:
        tags += [f"q{num_query_token}"]
    processed_file_name = "_".join(tags)

    map_kwargs = {
        "fn_kwargs": {
            "system_prompt": system_prompt,
            "llm_model_name": llm_model
        },
        "num_proc": num_workers,
    }

    for split_name, split_sets in [("train", trainsets), ("test", testsets), ("validation", valsets)]:
        if not split_sets:
            print(f"Warning: No {split_name} datasets were concatenated.")
            continue

        concat_set = datasets.concatenate_datasets(split_sets)
        print(f"Mapping {split_name.upper()} dataset ({len(concat_set)} examples)...")

        mapped_set = concat_set.map(prepare_data_instance, **map_kwargs)
        save_name = os.path.join(tag_root, f"{processed_file_name}_{split_name}_{data_tag}")
        mapped_set.save_to_disk(save_name)
        print(f"Saved Final {split_name.capitalize()} Dataset: {save_name} (Size: {len(mapped_set)})")

    end_time = time.time()
    print(f"Total time: {(end_time - start_time) / 60:.2f} minutes")
