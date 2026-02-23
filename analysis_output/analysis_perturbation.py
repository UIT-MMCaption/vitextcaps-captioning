#!/usr/bin/env python3
"""
OCR Perturbation Robustness Test for HSTFG Paper
=================================================

Tests model robustness by applying perturbations to OCR features at
inference time and measuring performance degradation.

Perturbation types:
  1. Token Drop: Randomly drop X% of OCR tokens (10%, 30%, 50%)
  2. Shuffle Order: Randomly shuffle OCR token order
  3. Combined: Drop 30% + shuffle remaining

For each perturbation, runs get_predictions() with modified features
and computes CIDEr + THR + NF degradation.

Usage:
    python analysis_perturbation.py \
        --config configs/hstfg_captioner.yaml \
        --perturbations drop_10 drop_30 drop_50 shuffle combined \
        --output perturbation_results.json

    # Also test M4C
    python analysis_perturbation.py \
        --config configs/mmf_m4c_captioner.yaml \
        --perturbations drop_10 drop_30 drop_50 shuffle combined \
        --output perturbation_results_m4c.json
"""

import os
import sys
import json
import argparse
import random
import itertools
import time
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import get_config
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset
from builders.vocab_builder import build_vocab
from data_utils.utils import collate_fn
from torch.utils.data import DataLoader
import evaluation


# ================================================================
# Perturbation functions
# ================================================================

def apply_perturbation(items, perturbation_type, rng):
    """Apply perturbation to OCR features in a batch.

    Modifies items in-place by zeroing out dropped OCR token features.

    Args:
        items: batch Instance with ocr_* fields
        perturbation_type: str, one of drop_10, drop_30, drop_50, shuffle, combined
        rng: random.Random instance for reproducibility
    """
    B = items.ocr_det_features.shape[0]
    N_ocr = items.ocr_det_features.shape[1]

    if perturbation_type.startswith("drop_"):
        drop_rate = int(perturbation_type.split("_")[1]) / 100.0
        _drop_tokens(items, drop_rate, B, N_ocr, rng)

    elif perturbation_type == "shuffle":
        _shuffle_tokens(items, B, N_ocr, rng)

    elif perturbation_type == "combined":
        _drop_tokens(items, 0.3, B, N_ocr, rng)
        _shuffle_tokens(items, B, N_ocr, rng)

    return items


def _drop_tokens(items, drop_rate, B, N_ocr, rng):
    """Zero out random OCR tokens."""
    for b in range(B):
        n_drop = max(1, int(N_ocr * drop_rate))
        drop_indices = rng.sample(range(N_ocr), min(n_drop, N_ocr))
        for idx in drop_indices:
            items.ocr_det_features[b, idx] = 0.0
            items.ocr_rec_features[b, idx] = 0.0
            items.ocr_fasttext_features[b, idx] = 0.0
            items.ocr_boxes[b, idx] = 0.0
            if hasattr(items, 'ocr_scores') and items.ocr_scores is not None:
                if items.ocr_scores.dim() == 2:
                    items.ocr_scores[b, idx] = 0.0
                elif items.ocr_scores.dim() == 3:
                    items.ocr_scores[b, idx, :] = 0.0


def _shuffle_tokens(items, B, N_ocr, rng):
    """Randomly permute OCR token order within each sample."""
    for b in range(B):
        perm = list(range(N_ocr))
        rng.shuffle(perm)
        perm_t = torch.tensor(perm, device=items.ocr_det_features.device)

        items.ocr_det_features[b] = items.ocr_det_features[b][perm_t]
        items.ocr_rec_features[b] = items.ocr_rec_features[b][perm_t]
        items.ocr_fasttext_features[b] = items.ocr_fasttext_features[b][perm_t]
        items.ocr_boxes[b] = items.ocr_boxes[b][perm_t]

        if hasattr(items, 'ocr_scores') and items.ocr_scores is not None:
            if items.ocr_scores.dim() >= 2:
                items.ocr_scores[b] = items.ocr_scores[b][perm_t]

        # Shuffle ocr_tokens list
        if hasattr(items, 'ocr_tokens') and items.ocr_tokens is not None:
            if isinstance(items.ocr_tokens[b], list):
                items.ocr_tokens[b] = [items.ocr_tokens[b][i] for i in perm]


# ================================================================
# Inference with perturbation
# ================================================================

def run_perturbed_inference(config, perturbation_type, seed=42):
    """Run test inference with perturbed OCR features.

    Returns (overall_gens, overall_gts) dicts compatible with evaluation.
    """
    rng = random.Random(seed)

    # Build vocab and model
    vocab = build_vocab(config.DATASET)
    model = build_model(config.MODEL, vocab)
    device = torch.device(config.MODEL.DEVICE)

    # Load best checkpoint
    ckpt_path = os.path.join(
        config.TRAINING.CHECKPOINT_PATH,
        config.MODEL.NAME,
        "best_model.pth"
    )
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return None, None

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # Build test dataset
    test_dataset = build_dataset(
        config.DATASET.JSON_PATH.TEST,
        vocab,
        config.DATASET.DICT_DATASET,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    overall_gens = {}
    overall_gts = {}

    desc = f"Perturbation: {perturbation_type}"
    with tqdm(desc=desc, total=len(test_loader), unit='it') as pbar:
        for it, items in enumerate(test_loader):
            items = items.to(device)

            # Apply perturbation
            if perturbation_type != "none":
                apply_perturbation(items, perturbation_type, rng)

            with torch.no_grad():
                result = model(items)

            outs = result["scores"].argmax(dim=-1)
            answers_gt = items.answers
            answers_gen = vocab.decode_answer(
                outs.contiguous(), items.ocr_tokens, join_words=False
            )

            for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                key = f"{it}_{i}"
                overall_gens[key] = [gen_i]
                overall_gts[key] = [' '.join(gts_i)] if isinstance(gts_i, list) else [gts_i]

            pbar.update()

    return overall_gens, overall_gts


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OCR perturbation robustness test"
    )
    parser.add_argument("--config", required=True,
                        help="Model config YAML path")
    parser.add_argument("--perturbations", nargs="+",
                        default=["none", "drop_10", "drop_30", "drop_50",
                                 "shuffle", "combined"],
                        help="Perturbation types to test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="perturbation_results.json")
    args = parser.parse_args()

    config = get_config(args.config)
    model_name = config.MODEL.NAME
    print(f"Model: {model_name}")
    print(f"Perturbations: {args.perturbations}")

    all_results = {}

    for perturb in args.perturbations:
        print(f"\n{'=' * 50}")
        print(f"Running: {perturb}")
        print(f"{'=' * 50}")

        gens, gts = run_perturbed_inference(config, perturb, args.seed)
        if gens is None:
            continue

        # Compute caption metrics
        scores, _ = evaluation.compute_scores(gts, gens)
        print(f"  Scores: {scores}")

        all_results[perturb] = {
            "n_samples": len(gens),
            **{k: float(v) if isinstance(v, float) else v
               for k, v in scores.items()},
        }

    # Print comparison table
    print(f"\n{'=' * 70}")
    print(f"  Perturbation Robustness: {model_name}")
    print(f"{'=' * 70}")

    if "none" in all_results:
        base_cider = all_results["none"].get("CIDEr", 0)
    else:
        base_cider = None

    header = f"{'Perturbation':<16} {'CIDEr':>8} {'METEOR':>8} {'ROUGE':>8}"
    if base_cider is not None:
        header += f" {'Δ CIDEr':>10}"
    print(header)
    print("-" * 70)

    for perturb in args.perturbations:
        if perturb not in all_results:
            continue
        r = all_results[perturb]
        cider = r.get("CIDEr", 0)
        meteor = r.get("METEOR", 0)
        rouge = r.get("ROUGE", r.get("ROUGE_L", 0))
        row = f"{perturb:<16} {cider:>8.4f} {meteor:>8.4f} {rouge:>8.4f}"
        if base_cider is not None:
            delta = cider - base_cider
            row += f" {delta:>+10.4f}"
        print(row)

    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({"model": model_name, "results": all_results},
                  f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
