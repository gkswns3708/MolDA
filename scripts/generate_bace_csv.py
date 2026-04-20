#!/usr/bin/env python3
"""Generate BACE CSV files from DeepChem MoleculeNet.

Loads BACE dataset via DeepChem, converts to {SELFIES, label} CSV format.
Output: dataset/Raw/raw/BioT5_bace_{train,valid,test}.csv
"""

import os
import numpy as np
import pandas as pd
from rdkit import Chem
import selfies as sf
import deepchem as dc

RAW_DIR = "/opt/EMNLP_MolDA/New_MolDA/dataset/Raw/raw"
os.makedirs(RAW_DIR, exist_ok=True)


def to_bool_str(y):
    try:
        return "True" if float(y) > 0.0 else "False"
    except Exception:
        return "True" if str(y).strip().lower() in ("1", "true", "yes") else "False"


def dc_dataset_to_csv(dc_dataset, out_csv_path):
    rows = []
    ys = np.array(dc_dataset.y).reshape(-1)
    for mol, y in zip(dc_dataset.X, ys):
        if mol is None:
            continue
        smi = Chem.MolToSmiles(mol)
        if not smi:
            continue
        try:
            selfies_str = sf.encoder(smi)
        except Exception:
            continue
        rows.append({"SELFIES": selfies_str, "label": to_bool_str(y)})
    df = pd.DataFrame(rows, columns=["SELFIES", "label"])
    df.to_csv(out_csv_path, index=False)
    print(f"[saved] {out_csv_path} (rows={len(df)})")


def main():
    loader = dc.molnet.load_bace_classification
    tasks, datasets, transformers = loader(
        featurizer="Raw", splitter="scaffold", reload=True
    )
    train_dc, valid_dc, test_dc = datasets

    dc_dataset_to_csv(train_dc, os.path.join(RAW_DIR, "BioT5_bace_train.csv"))
    dc_dataset_to_csv(valid_dc, os.path.join(RAW_DIR, "BioT5_bace_valid.csv"))
    dc_dataset_to_csv(test_dc, os.path.join(RAW_DIR, "BioT5_bace_test.csv"))
    print("Done.")


if __name__ == "__main__":
    main()
