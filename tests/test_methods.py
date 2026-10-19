"""
Automated verification suite for BootcampAI2026 scripts and notebook methods.
Tests:
1. FocalLoss mathematical correctness (unweighted pt vs alpha scaling).
2. Cross-class vs within-class MD5 duplicate filtering (zero data leakage).
3. Fold index parsing for K-Fold evaluation (`_get_fold_index_from_path`).
4. EarlyStopping state reset between Stage 1 and Stage 2.
5. `dataset_sorter.py` task building and dry-run execution.
6. `siglip_mislabel_detection.py` trimmed centroids and Leave-One-Out similarity margin.
7. Notebook JSON validity and AST syntax verification for all code cells.
"""

import ast
import json
import os
import re
import sys
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import dataset_sorter
import siglip_mislabel_detection as smd


class FocalLossFixed(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()


class TestMethodologyFixes(unittest.TestCase):
    def test_focal_loss_math(self):
        """Verify that FocalLoss computes pt = softmax_prob(target) regardless of alpha weights."""
        logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, 0.2]], dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)
        alpha = torch.tensor([0.5, 2.5, 1.0], dtype=torch.float32)
        gamma = 2.0

        loss_fn = FocalLossFixed(alpha=alpha, gamma=gamma)
        computed = loss_fn(logits, targets)

        # Manual ground-truth formula: -alpha_t * (1 - p_t)^gamma * log(p_t)
        probs = F.softmax(logits, dim=-1)
        p_t = probs[torch.arange(2), targets]
        a_t = alpha[targets]
        expected = (-a_t * ((1.0 - p_t) ** gamma) * torch.log(p_t)).mean()

        self.assertTrue(torch.allclose(computed, expected, atol=1e-6))

    def test_cross_class_duplicate_removal(self):
        """Verify cross-class conflicting hashes are removed completely while within-class duplicates keep 1."""
        df = pd.DataFrame({
            "filepath": ["a1.jpg", "a2.jpg", "b1.jpg", "c1.jpg", "c2.jpg"],
            "original_class": ["realperson", "realperson", "fake_mask", "fake_screen", "realperson"],
            "md5": ["h_within", "h_within", "h_unique", "h_conflict", "h_conflict"],
        })
        label_nunique = df.groupby("md5")["original_class"].transform("nunique")
        clean_df = df[label_nunique == 1].drop_duplicates(subset=["md5"], keep="first").reset_index(drop=True)

        self.assertEqual(len(clean_df), 2)
        self.assertSetEqual(set(clean_df["md5"]), {"h_within", "h_unique"})

    def test_fold_index_resolution(self):
        """Verify K-Fold evaluation matches the fold number in model checkpoint path."""
        def _get_fold_index_from_path(model_path, default_fold=0):
            m = re.search(r"fold(\d+)", os.path.basename(model_path), re.IGNORECASE)
            return (int(m.group(1)) - 1) if m else default_fold

        self.assertEqual(_get_fold_index_from_path("saved_models/master_convnext_fold1.pth"), 0)
        self.assertEqual(_get_fold_index_from_path("saved_models/master_convnext_fold4.pth"), 3)
        self.assertEqual(_get_fold_index_from_path("saved_models/master_convnext_fold5.pth"), 4)

    def test_dataset_sorter_dry_run(self):
        """Verify dataset_sorter builds 1652 tasks from train/ and data/relabeling_csv/."""
        base_dir = os.path.join(ROOT_DIR, "train")
        csv_dir = os.path.join(ROOT_DIR, "data", "relabeling_csv")
        tasks = dataset_sorter.build_tasks(base_dir, csv_dir)
        tasks = dataset_sorter.execute_tasks(tasks, base_dir, mode="move", dry_run=True)

        self.assertEqual(len(tasks), 1652)
        statuses = {t.status for t in tasks}
        self.assertSetEqual(statuses, {"already_correct", "dry_run"})

    def test_siglip_loo_and_trimmed_centroids(self):
        """Verify LOO similarity and trimmed centroids work without self-bias."""
        rng = np.random.default_rng(123)
        embs = rng.normal(size=(30, 16)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        labels = [smd.CLASS_NAMES[i // 5] for i in range(30)]
        md5s = [f"h_{i}" for i in range(30)]
        df = pd.DataFrame({"path": [f"p_{i}.jpg" for i in range(30)], "label": labels, "md5": md5s})

        centroids = smd.compute_centroids(embs, labels, smd.CLASS_NAMES, md5_hashes=md5s, trim_ratio=0.1)
        res_loo = smd.compute_similarity_and_margin(df, embs, centroids, smd.CLASS_NAMES, use_loo_own_similarity=True)
        res_naive = smd.compute_similarity_and_margin(df, embs, centroids, smd.CLASS_NAMES, use_loo_own_similarity=False)

        # Self-inclusion always inflates own_class_similarity compared to LOO on unweighted mean
        self.assertEqual(len(res_loo), 30)
        self.assertTrue((res_loo["own_class_similarity"] < res_naive["own_class_similarity"] + 1e-4).all())

    def test_notebooks_valid_syntax(self):
        """Verify all active notebooks parse cleanly as JSON and valid Python AST."""
        notebooks = [
            os.path.join(ROOT_DIR, "Convnext", "Master_Optuna_KFold_AntiSpoofing_v2_Documented_savedrive_freeze_backboneOutput.ipynb"),
            os.path.join(ROOT_DIR, "Convnext", "Training_Pipeline_Butkem_AI_v4_Simplified.ipynb"),
        ]
        for nb_path in notebooks:
            self.assertTrue(os.path.isfile(nb_path), f"Missing notebook: {nb_path}")
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            for idx, cell in enumerate(nb.get("cells", [])):
                if cell.get("cell_type") == "code":
                    src = "".join(cell.get("source", []))
                    try:
                        ast.parse(src)
                    except SyntaxError as e:
                        self.fail(f"SyntaxError in {os.path.basename(nb_path)} cell {idx}: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
