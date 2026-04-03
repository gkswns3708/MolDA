"""
Task-specific evaluation metrics.

Task categorization is based on actual Train_toy100 dataset (21 tasks).
Old_MolDA help_funcs.py 참고하되, toy dataset에 실제 존재하는 task만 포함.
"""

import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Task categorization (from Train_toy100 actual tasks)
# ─────────────────────────────────────────────

CLASSIFICATION_TASKS = {
    "smol-property_prediction-bbbp",
    "smol-property_prediction-clintox",
    "smol-property_prediction-hiv",
    "smol-property_prediction-sider",
    "bace",
}

REGRESSION_TASKS = {
    "smol-property_prediction-esol",
    "smol-property_prediction-lipo",
    "qm9_homo",
    "qm9_lumo",
    "qm9_homo_lumo_gap",
}

REACTION_TASKS = {
    "forward_reaction_prediction",
    "smol-forward_synthesis",
    "retrosynthesis",
    "smol-retrosynthesis",
    "reagent_prediction",
}

TEXT2MOL_TASKS = {
    "chebi-20-text2mol",
    "smol-molecule_generation",
}

MOL2TEXT_TASKS = {
    "chebi-20-mol2text",
    "smol-molecule_captioning",
}

NAME_CONVERSION_TASKS = {
    "smol-name_conversion-i2s",
    "smol-name_conversion-s2i",
}

# Union of all generation-based tasks (everything except classification and name_conversion)
GENERATION_TASKS = REGRESSION_TASKS | REACTION_TASKS | TEXT2MOL_TASKS | MOL2TEXT_TASKS

ALL_TASKS = (CLASSIFICATION_TASKS | REGRESSION_TASKS | REACTION_TASKS
             | TEXT2MOL_TASKS | MOL2TEXT_TASKS | NAME_CONVERSION_TASKS)


def get_task_type(task: str) -> str:
    if task in CLASSIFICATION_TASKS:
        return "classification"
    elif task in REGRESSION_TASKS:
        return "regression"
    elif task in REACTION_TASKS or task in TEXT2MOL_TASKS:
        return "molecule"
    elif task in MOL2TEXT_TASKS:
        return "caption"
    elif task in NAME_CONVERSION_TASKS:
        return "name_conversion"
    else:
        logger.warning(f"Unknown task: {task}, treating as molecule")
        return "molecule"


# ─────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────

def _parse_tag(text: str, tag: str) -> Optional[str]:
    """Extract content between <TAG>...</TAG>."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def _parse_float_tag(text: str) -> Optional[float]:
    """Extract float from <FLOAT>...</FLOAT>."""
    content = _parse_tag(text, "FLOAT")
    if content is None:
        return None
    # Remove special number tokens like <|0|>, <|.|>, etc.
    cleaned = re.sub(r"<\|(.)\|>", r"\1", content).strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_boolean_tag(text: str) -> Optional[bool]:
    """Extract boolean from <BOOLEAN>...</BOOLEAN>."""
    content = _parse_tag(text, "BOOLEAN")
    if content is None:
        return None
    content = content.strip().lower()
    if content == "true":
        return True
    elif content == "false":
        return False
    return None


# ─────────────────────────────────────────────
# Classification: likelihood-based eval
# ─────────────────────────────────────────────

def classification_evaluate(probs: "torch.Tensor", label_texts: List[str],
                            task: str) -> Dict[str, float]:
    """Evaluate binary classification using likelihood probabilities.

    Args:
        probs: [N, 2] tensor of [P(False), P(True)]
        label_texts: list of ground truth strings e.g. "<BOOLEAN> True </BOOLEAN>"
        task: task name

    Returns:
        dict with accuracy, f1, roc_auc, failure_rate
    """
    import torch
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

    gt_labels = []
    valid_indices = []
    for i, text in enumerate(label_texts):
        parsed = _parse_boolean_tag(text)
        if parsed is not None:
            gt_labels.append(1 if parsed else 0)
            valid_indices.append(i)

    if not gt_labels:
        return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "roc_auc": 0.0, "failure_rate": 1.0}

    gt = np.array(gt_labels)
    prob_true = probs[valid_indices, 1].cpu().numpy()
    preds = (prob_true > 0.5).astype(int)

    results = {
        "accuracy": float(accuracy_score(gt, preds)),
        "f1": float(f1_score(gt, preds, zero_division=0)),
        "precision": float(precision_score(gt, preds, zero_division=0)),
        "recall": float(recall_score(gt, preds, zero_division=0)),
        "failure_rate": 1.0 - len(valid_indices) / len(label_texts),
    }

    # ROC AUC requires both classes present
    if len(set(gt)) > 1:
        results["roc_auc"] = float(roc_auc_score(gt, prob_true))
    else:
        results["roc_auc"] = 0.0

    return results


# ─────────────────────────────────────────────
# Regression: generation + float parse
# ─────────────────────────────────────────────

def regression_evaluate(pred_texts: List[str], label_texts: List[str],
                        task: str) -> Dict[str, float]:
    """Evaluate regression by parsing <FLOAT> tags and computing MAE/RMSE.

    Args:
        pred_texts: model-generated strings
        label_texts: ground truth strings

    Returns:
        dict with mae, rmse, failure_rate
    """
    preds, gts = [], []
    for p, g in zip(pred_texts, label_texts):
        pred_val = _parse_float_tag(p)
        gt_val = _parse_float_tag(g)
        if pred_val is not None and gt_val is not None:
            preds.append(pred_val)
            gts.append(gt_val)

    n_total = len(pred_texts)
    n_valid = len(preds)

    if n_valid == 0:
        return {"mae": float("inf"), "mse": float("inf"), "rmse": float("inf"), "failure_rate": 1.0}

    preds = np.array(preds)
    gts = np.array(gts)
    errors = np.abs(preds - gts)

    return {
        "mae": float(errors.mean()),
        "mse": float((errors ** 2).mean()),
        "rmse": float(np.sqrt((errors ** 2).mean())),
        "failure_rate": 1.0 - n_valid / n_total,
    }


# ─────────────────────────────────────────────
# Molecule: generation + SELFIES parse
# ─────────────────────────────────────────────

def molecule_evaluate(pred_texts: List[str], label_texts: List[str],
                      task: str, tokenizer=None) -> Dict[str, float]:
    """Evaluate molecule generation (reaction, text2mol) by parsing SELFIES.

    Args:
        pred_texts: model-generated strings containing <SELFIES>...</SELFIES>
        label_texts: ground truth strings
        tokenizer: HF tokenizer for BLEU tokenization (bleu_smiles, bleu_selfies)

    Returns:
        dict with validity_ratio, exact_match_ratio, fingerprint similarities,
        bleu_smiles, bleu_selfies, levenshtein_score, failure_rate
    """
    try:
        import selfies as sf
        from rdkit import Chem
        from rdkit.Chem import MACCSkeys, AllChem, DataStructs
    except ImportError:
        logger.warning("selfies/rdkit not installed, returning empty metrics")
        return {"validity_ratio": 0.0, "exact_match_ratio": 0.0,
                "maccs_fts": 0.0, "rdk_fts": 0.0, "morgan_fts": 0.0,
                "bleu_smiles": 0.0, "bleu_selfies": 0.0,
                "levenshtein_score": 0.0, "failure_rate": 1.0}

    n_total = len(pred_texts)
    n_valid_smiles = 0
    n_exact_match = 0
    n_parsed = 0
    levenshtein_scores = []
    maccs_scores = []
    rdk_scores = []
    morgan_scores = []
    # For BLEU: tokenized canonical SMILES/SELFIES pairs
    ref_smiles_list, pred_smiles_list = [], []
    ref_selfies_list, pred_selfies_list = [], []

    for pred, gt in zip(pred_texts, label_texts):
        pred_selfies = _parse_tag(pred, "SELFIES")
        gt_selfies = _parse_tag(gt, "SELFIES")

        if pred_selfies is None or gt_selfies is None:
            continue
        n_parsed += 1

        # SELFIES → SMILES
        try:
            pred_smiles = sf.decoder(pred_selfies)
            gt_smiles = sf.decoder(gt_selfies)
        except Exception:
            continue

        # Validity
        pred_mol = Chem.MolFromSmiles(pred_smiles)
        gt_mol = Chem.MolFromSmiles(gt_smiles)
        if pred_mol is not None:
            n_valid_smiles += 1
            pred_canonical = Chem.MolToSmiles(pred_mol)
            if gt_mol is not None:
                gt_canonical = Chem.MolToSmiles(gt_mol)
                if pred_canonical == gt_canonical:
                    n_exact_match += 1

                # BLEU tokenization (only when both are valid + canonical)
                if tokenizer is not None:
                    pred_canonical_selfies = sf.encoder(pred_canonical)
                    gt_canonical_selfies = sf.encoder(gt_canonical)
                    pred_smiles_list.append(tokenizer.tokenize(pred_canonical))
                    ref_smiles_list.append([tokenizer.tokenize(gt_canonical)])
                    pred_selfies_list.append(tokenizer.tokenize(pred_canonical_selfies))
                    ref_selfies_list.append([tokenizer.tokenize(gt_canonical_selfies)])

        # Fingerprint Tanimoto Similarities
        if pred_mol is not None and gt_mol is not None:
            try:
                # MACCS
                fp_pred = MACCSkeys.GenMACCSKeys(pred_mol)
                fp_gt = MACCSkeys.GenMACCSKeys(gt_mol)
                maccs_scores.append(DataStructs.TanimotoSimilarity(fp_pred, fp_gt))
                # RDK
                rdk_pred = Chem.RDKFingerprint(pred_mol)
                rdk_gt = Chem.RDKFingerprint(gt_mol)
                rdk_scores.append(DataStructs.TanimotoSimilarity(rdk_pred, rdk_gt))
                # Morgan (radius=2)
                morgan_pred = AllChem.GetMorganFingerprint(pred_mol, 2)
                morgan_gt = AllChem.GetMorganFingerprint(gt_mol, 2)
                morgan_scores.append(DataStructs.TanimotoSimilarity(morgan_pred, morgan_gt))
            except Exception:
                pass

        # Levenshtein on canonical SMILES
        if pred_smiles and gt_smiles:
            lev = _levenshtein_distance(pred_smiles, gt_smiles)
            max_len = max(len(pred_smiles), len(gt_smiles))
            levenshtein_scores.append(1.0 - lev / max_len if max_len > 0 else 1.0)

    # BLEU scores
    bleu_smiles, bleu_selfies = 0.0, 0.0
    if pred_smiles_list:
        try:
            from nltk.translate.bleu_score import corpus_bleu
            bleu_smiles = corpus_bleu(
                ref_smiles_list, pred_smiles_list,
                weights=(0.25, 0.25, 0.25, 0.25),
            ) * 100
            bleu_selfies = corpus_bleu(
                ref_selfies_list, pred_selfies_list,
                weights=(0.25, 0.25, 0.25, 0.25),
            ) * 100
        except ImportError:
            pass

    return {
        "validity_ratio": n_valid_smiles / n_parsed if n_parsed > 0 else 0.0,
        "exact_match_ratio": n_exact_match / n_parsed if n_parsed > 0 else 0.0,
        "levenshtein_score": float(np.mean(levenshtein_scores)) if levenshtein_scores else 0.0,
        "maccs_fts": float(np.mean(maccs_scores)) if maccs_scores else 0.0,
        "rdk_fts": float(np.mean(rdk_scores)) if rdk_scores else 0.0,
        "morgan_fts": float(np.mean(morgan_scores)) if morgan_scores else 0.0,
        "bleu_smiles": float(bleu_smiles),
        "bleu_selfies": float(bleu_selfies),
        "failure_rate": 1.0 - n_parsed / n_total if n_total > 0 else 1.0,
    }


def _levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


# ─────────────────────────────────────────────
# Caption: generation + text parse
# ─────────────────────────────────────────────

def caption_evaluate(pred_texts: List[str], label_texts: List[str],
                     task: str) -> Dict[str, float]:
    """Evaluate text generation (mol2text, captioning) with BLEU/ROUGE.

    Args:
        pred_texts: model-generated strings
        label_texts: ground truth strings

    Returns:
        dict with bleu2, bleu4, rouge1, rougeL, failure_rate
    """
    # Determine tag type based on task
    if "mol2text" in task or "captioning" in task:
        tag = "DESCRIPTION"
    elif "i2f" in task or "s2f" in task:
        tag = "MOLFORMULA"
    elif "s2i" in task or "i2s" in task:
        tag = "IUPAC"
    else:
        tag = "DESCRIPTION"

    preds_parsed, gts_parsed = [], []
    for pred, gt in zip(pred_texts, label_texts):
        p = _parse_tag(pred, tag)
        g = _parse_tag(gt, tag)
        if p is not None and g is not None:
            preds_parsed.append(p)
            gts_parsed.append(g)

    n_total = len(pred_texts)
    n_valid = len(preds_parsed)

    if n_valid == 0:
        return {"bleu2": 0.0, "bleu4": 0.0, "meteor": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "failure_rate": 1.0}

    # BLEU
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        smooth = SmoothingFunction().method1
        refs = [[g.split()] for g in gts_parsed]
        hyps = [p.split() for p in preds_parsed]
        bleu2 = corpus_bleu(refs, hyps, weights=(0.5, 0.5), smoothing_function=smooth)
        bleu4 = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    except ImportError:
        bleu2, bleu4 = 0.0, 0.0

    # METEOR
    try:
        from nltk.translate.meteor_score import meteor_score as nltk_meteor
        meteor_scores = [nltk_meteor([g.split()], p.split())
                         for p, g in zip(preds_parsed, gts_parsed)]
        meteor = float(np.mean(meteor_scores))
    except ImportError:
        meteor = 0.0

    # ROUGE
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1_scores, r2_scores, rl_scores = [], [], []
        for p, g in zip(preds_parsed, gts_parsed):
            scores = scorer.score(g, p)
            r1_scores.append(scores["rouge1"].fmeasure)
            r2_scores.append(scores["rouge2"].fmeasure)
            rl_scores.append(scores["rougeL"].fmeasure)
        rouge1 = float(np.mean(r1_scores))
        rouge2 = float(np.mean(r2_scores))
        rougeL = float(np.mean(rl_scores))
    except ImportError:
        rouge1, rouge2, rougeL = 0.0, 0.0, 0.0

    return {
        "bleu2": float(bleu2),
        "bleu4": float(bleu4),
        "meteor": meteor,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
        "failure_rate": 1.0 - n_valid / n_total,
    }


# ─────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────

def evaluate_by_task(task: str, **kwargs) -> Dict[str, float]:
    """Dispatch to the appropriate evaluation function based on task type."""
    task_type = get_task_type(task)
    if task_type == "classification":
        return classification_evaluate(**kwargs, task=task)
    elif task_type == "regression":
        return regression_evaluate(**kwargs, task=task)
    elif task_type == "molecule":
        return molecule_evaluate(**kwargs, task=task)
    elif task_type == "caption":
        return caption_evaluate(**kwargs, task=task)
    elif task_type == "name_conversion":
        return {}  # MVP: skip
    else:
        return {}
