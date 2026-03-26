"""
MoleculeDataset: thin wrapper around HuggingFace Arrow dataset.

Usage:
    ds = MoleculeDataset("dataset/Train_toy100")
    sample = ds[0]  # dict with keys: x, edge_index, edge_attr, ..., prompt_text, target_text
"""

from datasets import load_from_disk
from torch.utils.data import Dataset


class MoleculeDataset(Dataset):

    def __init__(self, path: str):
        self.dataset = load_from_disk(path)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        return self.dataset[idx]
