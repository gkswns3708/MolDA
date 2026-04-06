# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Dataset classes, processing functions, and data loading logic.

Moved from the monolithic dataset_generator.py into the dataset_generation package.
"""

import re
from concurrent.futures import ProcessPoolExecutor

import deepchem as dc
import numpy as np
import pandas as pd
from datasets import load_dataset
from rdkit import Chem
from torch.utils.data import Dataset
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
from dataset_generation.utils import (
    clean_mol_string,
    convert_mol_representation,
    get_canonical_smiles,
    get_dummy_graph,
    selfies_to_smiles,
    smiles2data,
)


# ---------------------------------------------------------------------------
# Label wrapping (re-export from utils for backward compat, but defined here
# because it depends on benchmark_constants and added_tokens)
# ---------------------------------------------------------------------------

def wrap_label(label, task):
    # Step 1에서는 항상 SMILES 태그 사용 (SELFIES 변환은 Step 3에서 수행)
    _mol_tokens = added_tokens.SMILES

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
        label_tokens = _mol_tokens
    elif task in TEXT2MOL_BENCHMARKS + REACTION_BENCHMARKS:
        label_tokens = _mol_tokens
    else:
        raise NotImplementedError(f"Task {task} is not implemented in wrap_label")

    if task in CLASSIFICATION_BENCHMARKS:
        if isinstance(label, str):
            if "true" in label.lower() or "yes" in label.lower():
                label = "True"
            elif "false" in label.lower() or "no" in label.lower():
                label = "False"
            else:
                raise ValueError(f"Unexpected classification label: {label!r} for task {task}")
            label = label_tokens[0] + " " + label + " " + label_tokens[1]
        elif isinstance(label, list):
            label_language = ", ".join(label)
            label_boolean = ", ".join(["True"] * len(label))
            label = label_language + " " + label_tokens[0] + " " + label_boolean + " " + label_tokens[1]
        else:
            label = "True" if label else "False"
            label = label_tokens[0] + " " + label + " " + label_tokens[1]
        return label
    elif task in REGRESSION_BENCHMARKS:
        if isinstance(label, (float, int)):
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


# ---------------------------------------------------------------------------
# Top-level processing functions (pickle-safe for multiprocessing)
# ---------------------------------------------------------------------------

def _process_moleculenet_sample(args):
    index, smiles, raw_output, subtask_idx, task, instruction_templates, label_full_name = args
    if smiles is None:
        return None
    try:
        instruction = np.random.choice(instruction_templates)
        canonical = get_canonical_smiles(smiles)
        if canonical is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        graph = [smiles2data(canonical), get_dummy_graph()]
        input_mol_string = added_tokens.SMILES[0] + " " + canonical + " " + added_tokens.SMILES[1]

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

        return (graph, label, input_mol_string, instruction)
    except Exception:
        return None


def _selfies_field_to_smiles(selfies_str):
    """SELFIES 문자열을 SMILES로 변환. >> 구분자도 처리."""
    if ">>" in selfies_str:
        parts = selfies_str.split(">>")
        converted = [selfies_to_smiles(p.strip()) for p in parts]
        if any(c is None for c in converted):
            return None
        return ">>".join(converted)
    smiles = selfies_to_smiles(selfies_str)
    return smiles


def _process_molinstruction_sample(args):
    index, input_, label_, task, instruction_templates, is_selfies_input = args
    try:
        instruction = np.random.choice(instruction_templates)
        if pd.isna(input_) or pd.isna(label_):
            raise ValueError("Input or Label is NaN")
        input_ = str(input_)
        label_ = str(label_)

        # Mol-Instructions 데이터는 SELFIES 포맷 → SMILES로 변환
        if is_selfies_input:
            input_ = _selfies_field_to_smiles(input_)
            if input_ is None:
                raise ValueError("Input SELFIES→SMILES conversion failed")
            # Reaction label도 SELFIES → SMILES 변환 필요
            if task in REACTION_BENCHMARKS:
                label_ = _selfies_field_to_smiles(label_)
                if label_ is None:
                    raise ValueError("Label SELFIES→SMILES conversion failed")

        if task in REACTION_BENCHMARKS:
            if task in ["reagent_prediction"]:
                if ">>" in input_:
                    list_smiles = input_.split(">>")
                    graph = [smiles2data(s.strip()) for s in list_smiles]
                    input_mol_string = input_.replace(
                        ">>",
                        f"{added_tokens.SMILES[1]}{added_tokens.REACTION_DIRECTION[0]}{added_tokens.SMILES[0]}",
                    )
                elif "|>>|" in input_:
                    list_smiles = input_.split("|>>|")
                    input_mol_string = input_
                    graph = [smiles2data(s.strip()) for s in list_smiles]
                else:
                    raise ValueError(f"Invalid reagent format: {input_}")
            else:
                # forward/retro reaction — 단일 input
                canonical = get_canonical_smiles(input_.strip())
                if canonical is None:
                    raise ValueError(f"Invalid reaction SMILES: {input_}")
                graph = [smiles2data(canonical), get_dummy_graph()]
                input_mol_string = input_
        else:
            # property prediction 등 단일 분자
            canonical = get_canonical_smiles(input_.strip())
            if canonical is None:
                raise ValueError("Invalid SMILES")
            graph = [smiles2data(canonical), get_dummy_graph()]
            input_mol_string = canonical

        label = wrap_label(label_, task)
        input_mol_string = added_tokens.SMILES[0] + " " + input_mol_string + " " + added_tokens.SMILES[1]
        return (graph, label, input_mol_string, instruction, str(input_))
    except Exception:
        return None


def _process_smol_sample(args):
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
                    raise ValueError(f"Invalid SMILES: {input_mol_string}")
                graph = [smiles2data(canonical), get_dummy_graph()]
                input_mol_string = canonical
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
                input_mol_string = canonical
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
            input_mol_string = canonical

        elif task in MOL2TEXT_BENCHMARKS + CLASSIFICATION_BENCHMARKS + REGRESSION_BENCHMARKS:
            instruction = np.random.choice(instruction_templates)
            input_mol_string = clean_mol_string(raw_input)
            canonical = get_canonical_smiles(input_mol_string)
            if canonical is None:
                raise ValueError(f"Invalid SMILES: {input_mol_string}")
            graph = [smiles2data(canonical), get_dummy_graph()]
            input_mol_string = canonical
        else:
            raise NotImplementedError

        label = wrap_label(label, task)
        input_mol_string = added_tokens.SMILES[0] + " " + input_mol_string + " " + added_tokens.SMILES[1]
        return (graph, label, input_mol_string, instruction)
    except Exception:
        return None


def _process_chebi_sample(args):
    index, desc, mol_data, is_selfies, task, instruction_templates = args
    try:
        instruction = np.random.choice(instruction_templates)
        if pd.isna(desc) or pd.isna(mol_data):
            raise ValueError("NaN data detected")

        # 항상 canonical SMILES를 확보
        if is_selfies:
            smiles = selfies_to_smiles(mol_data)
            if smiles is None:
                raise ValueError("SELFIES→SMILES conversion failed")
        else:
            smiles = mol_data
            canonical = get_canonical_smiles(smiles)
            if canonical:
                smiles = canonical

        if smiles is None:
            raise ValueError("SMILES is None")

        if task in TEXT2MOL_BENCHMARKS:
            label = smiles
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


# ---------------------------------------------------------------------------
# Parallel processing helper
# ---------------------------------------------------------------------------

def _parallel_process(process_fn, args_list, num_workers, desc="Processing"):
    if num_workers <= 1:
        results = []
        for args in tqdm(args_list, desc=desc):
            results.append(process_fn(args))
        return results

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(process_fn, args_list, chunksize=max(1, len(args_list) // (num_workers * 4))),
            total=len(args_list),
            desc=desc,
        ))
    return results


# ---------------------------------------------------------------------------
# Dataset Classes
# ---------------------------------------------------------------------------

class MoleculeNetDatasetDeepChem(Dataset):
    def __init__(self, data, task_subtask_pair, subtask_idx=0, prompt=None, num_workers=1, **kwargs):
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

        label_full_name = self.label_full_name if hasattr(self, "label_full_name") else None
        args_list = [
            (i, smiles_list[i], self.raw_outputs[i], self.subtask_idx, self.task,
             self.instruction_templates, label_full_name)
            for i in range(len(self.raw_inputs))
        ]

        results = _parallel_process(
            _process_moleculenet_sample, args_list, self.num_workers,
            desc=f"{self.task}-{self.subtask_idx}",
        )

        self.count_invalid_smiles = sum(1 for r in results if r is None)
        valid = [r for r in results if r is not None]
        self.graph_list = [r[0] for r in valid]
        self.label_list = [r[1] for r in valid]
        self.input_mol_string_list = [r[2] for r in valid]
        self.instruction_list = [r[3] for r in valid]

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        return (self.graph_list[index], self.label_list[index],
                self.input_mol_string_list[index], self.task_subtask_pair,
                self.instruction_list[index])


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
            input_list = self.data["SELFIES"][:]
            label_list = self.data["label"][:]
        else:
            input_list = self.data["input"][:]
            label_list = self.data["output"][:]
        self.instruction_templates = getattr(instructions_smol, self.task)

        # All Mol-Instructions data uses SELFIES format (bace: SELFIES column, others: input column)
        is_selfies_input = True

        args_list = [
            (i, input_list[i], label_list[i], self.task, self.instruction_templates, is_selfies_input)
            for i in range(len(input_list))
        ]

        results = _parallel_process(
            _process_molinstruction_sample, args_list, self.num_workers,
            desc=self.task,
        )

        self.count_invalid_smiles = sum(1 for r in results if r is None)
        valid = [r for r in results if r is not None]
        self.graph_list = [r[0] for r in valid]
        self.label_list = [r[1] for r in valid]
        self.input_mol_string_list = [r[2] for r in valid]
        self.instruction_list = [r[3] for r in valid]
        self.input_list = [r[4] for r in valid]

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        return (self.graph_list[index], self.label_list[index],
                self.input_mol_string_list[index], self.task,
                self.instruction_list[index])


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
            desc=self.task,
        )

        self.count_invalid_smiles = sum(1 for r in results if r is None)
        valid = [r for r in results if r is not None]
        print(f"[{self.task}] Finished. Total: {len(self.data)} | Success: {len(valid)} | Failed: {self.count_invalid_smiles}")

        self.graph_list = [r[0] for r in valid]
        self.label_list = [r[1] for r in valid]
        self.input_mol_string_list = [r[2] for r in valid]
        self.instruction_list = [r[3] for r in valid]

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        return (self.graph_list[index], self.label_list[index],
                self.input_mol_string_list[index], self.task_subtask_pair,
                self.instruction_list[index])


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
        if hasattr(self.data, "column_names"):
            cols = self.data.column_names
        elif hasattr(self.data, "columns"):
            cols = self.data.columns
        else:
            cols = []

        print(f"[Debug] Processing Task: {self.task} | Columns: {cols}")

        if "description" in cols:
            description_list = self.data["description"]
        elif "text" in cols:
            description_list = self.data["text"]
        elif "caption" in cols:
            description_list = self.data["caption"]
        else:
            raise ValueError(f"Column 'description' not found. Available: {cols}")

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
            desc=self.task,
        )

        self.count_invalid_smiles = sum(1 for r in results if r is None)
        valid = [r for r in results if r is not None]
        print(f"[{self.task}] Finished. Valid: {len(valid)}, Invalid: {self.count_invalid_smiles}")

        self.graph_list = [r[0] for r in valid]
        self.label_list = [r[1] for r in valid]
        self.input_mol_string_list = [r[2] for r in valid]
        self.instruction_list = [r[3] for r in valid]

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        return (self.graph_list[index], self.label_list[index],
                self.input_mol_string_list[index], self.task_subtask_pair,
                self.instruction_list[index])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_dataset(task_name, raw_data_root, toy_n=None):
    """소스별 데이터셋 로드. toy_n이 지정되면 각 split에서 N개만 sampling."""

    def _maybe_sample(ds, n):
        if n is None or ds is None:
            return ds
        if hasattr(ds, "select"):
            # HuggingFace Dataset
            indices = list(range(min(n, len(ds))))
            return ds.select(indices)
        elif hasattr(ds, "head"):
            # pandas DataFrame
            return ds.head(n)
        return ds

    import os

    if "chebi-20" in task_name:
        print(f"[Info] Loading Standard ChEBI-20 from 'liupf/ChEBI-20-MM' for {task_name}")
        try:
            dataset = load_dataset("liupf/ChEBI-20-MM")
            train_dataset = _maybe_sample(dataset["train"], toy_n)
            valid_dataset = _maybe_sample(dataset["validation"], toy_n)
            test_dataset = _maybe_sample(dataset["test"], toy_n)
            tasks = [task_name]
            return tasks, train_dataset, valid_dataset, test_dataset
        except Exception as e:
            print(f"[Warning] Failed to load liupf/ChEBI-20-MM: {e}. Fallback to local CSV.")
            train_dataset = _maybe_sample(pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_chebi20_train.csv")), toy_n)
            valid_dataset = _maybe_sample(pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_chebi20_valid.csv")), toy_n)
            test_dataset = _maybe_sample(pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_chebi20_test.csv")), toy_n)
            tasks = [task_name]
            return tasks, train_dataset, valid_dataset, test_dataset

    elif "smol" in task_name:
        smol_dataset = load_dataset(
            "osunlp/SMolInstruct",
            use_selfies=False,
            insert_core_tags=False,
            trust_remote_code=True,
        )
        _task = re.sub("smol-", "", task_name)
        train_dataset = _maybe_sample(smol_dataset["train"].filter(lambda x: x["task"] == _task), toy_n)
        valid_dataset = _maybe_sample(smol_dataset["validation"].filter(lambda x: x["task"] == _task), toy_n)
        test_dataset = _maybe_sample(smol_dataset["test"].filter(lambda x: x["task"] == _task), toy_n)
        tasks = [task_name]

    elif task_name in ["toxcast", "tox21", "hopv", "qm9_additional_label"]:
        loading_fn = getattr(dc.molnet, f"load_{task_name}" if task_name != "qm9_additional_label" else "load_qm9")
        base_path = f"dataset/{task_name}"
        os.makedirs(base_path, exist_ok=True)
        tasks, datasets_, transformers = loading_fn(
            featurizer="Raw", splitter="scaffold", save_dir=base_path, data_dir=base_path, reload=True,
        )
        train_dataset, valid_dataset, test_dataset = datasets_
        # DeepChem datasets don't support easy sampling; toy_n handled downstream

    elif task_name == "bace":
        train_dataset = _maybe_sample(pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_bace_train.csv")), toy_n)
        valid_dataset = _maybe_sample(pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_bace_valid.csv")), toy_n)
        test_dataset = _maybe_sample(pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_bace_test.csv")), toy_n)
        tasks = [task_name]

    elif task_name in ["reagent_prediction", "forward_reaction_prediction", "retrosynthesis",
                       "qm9_homo", "qm9_lumo", "qm9_homo_lumo_gap"]:
        mol_instruction_dataset = load_dataset(
            "zjunlp/Mol-Instructions", "Molecule-oriented Instructions", trust_remote_code=True,
        )
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

        train_dataset = _maybe_sample(train_dataset, toy_n)
        valid_dataset = _maybe_sample(valid_dataset, toy_n)
        test_dataset = _maybe_sample(test_dataset, toy_n)
        tasks = [task_name]
    else:
        raise NotImplementedError(f"Task {task_name} not supported in get_dataset")

    return tasks, train_dataset, valid_dataset, test_dataset


# ---------------------------------------------------------------------------
# Prompt formatting & Arrow serialization
# ---------------------------------------------------------------------------

def _convert_smiles_tags_to_selfies(tagged_string):
    """<SMILES>canonical_smiles</SMILES> → <SELFIES>selfies_string</SELFIES> 변환.

    <SMILES> 태그가 없는 문자열(예: <|bool|>, <|float|>)은 그대로 반환.
    <None> 값은 태그만 교체. reaction SMILES (>> 포함)도 처리.
    """
    def _replace(match):
        content = match.group(1).strip()
        if not content or content == "<None>":
            return "<SELFIES> " + content + " </SELFIES>"
        converted = convert_mol_representation(content, "selfies")
        if converted is None:
            converted = content  # fallback
        return "<SELFIES> " + converted + " </SELFIES>"

    return re.sub(r'<SMILES>\s*(.*?)\s*</SMILES>', _replace, tagged_string)


def prepare_data_instance(
    data_instance,
    system_prompt,
    llm_model_name="mistral",
    mol_token="<mol>",
    num_query_tokens=32,
    mol_representation="smiles",
):
    input_mol_string = data_instance["input_mol_string"]
    label = data_instance["label"]

    # Step 3: SELFIES 변환 (canonical SMILES → SELFIES)
    if mol_representation == "selfies":
        input_mol_string = _convert_smiles_tags_to_selfies(input_mol_string)
        label = _convert_smiles_tags_to_selfies(label)

    # 태그 spacing 정규화
    if mol_representation == "selfies":
        input_mol_string = re.sub(r"<SELFIES>\s*", "<SELFIES> ", input_mol_string)
        input_mol_string = re.sub(r"\s*</SELFIES>", " </SELFIES>", input_mol_string)
    else:
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
        formatted_target_text = label + "<|eot_id|>"
    else:
        formatted_prompt_text = "<s>[INST] " + system_prompt + " \n\n" + input_prompt + " [/INST] "
        formatted_target_text = label + " </s>"

    raw_task = data_instance["task_subtask_pair"]
    if "qm9_additional_label" in raw_task:
        convert_dict = {
            "qm9_additional_label/mu": "qm9_dipole_moment",
            "qm9_additional_label/alpha": "qm9_isotropic_polarizability",
            "qm9_additional_label/r2": "qm9_electronic_spatial_extent",
            "qm9_additional_label/zpve": "qm9_zero_point_vibrational_energy",
            "qm9_additional_label/cv": "qm9_heat_capacity_298K",
            "qm9_additional_label/u298": "qm9_internal_energy_298K",
            "qm9_additional_label/h298": "qm9_enthalpy_298K",
            "qm9_additional_label/g298": "qm9_free_energy_298K",
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
    for i in tqdm(range(len(ds)), desc="  Arrow 변환", disable=len(ds) < 1000):
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
