#!/usr/bin/env python3
"""
Fine-Grained Analysis for HSTFG Paper
======================================

Analyses:
  1. Caption Dependency on Scene Text (text-dependent vs text-independent)
  2. Impact of Text Density (sparse / medium / dense OCR tokens)
  3. OCR Confidence Sensitivity (low / medium / high confidence)
  4. Layout Complexity (single-line / multi-line / multi-block)
  5. Per-split CIDEr, METEOR, ROUGE, THR, NF for each model

Usage:
    python analysis_finegrained.py \
        --test_json ./features/vitextcaps_test.json \
        --ocr_dir ./features/swintextspotter \
        --results \
            saved_models/hstfg_vinvl_swimspotter/test_results.json \
            saved_models/mmf_m4c_vinvl_swimspotter/test_results.json \
            saved_models/m2_transformer/test_results.json \
        --names HSTFG M4C M2 \
        --output analysis_results.json
"""

import json
import os
import re
import sys
import argparse
import numpy as np
from collections import defaultdict

# ================================================================
# Try importing evaluation module (for CIDEr, METEOR, ROUGE)
# Falls back gracefully if not available
# ================================================================
try:
    import evaluation
    HAS_EVAL = True
except ImportError:
    HAS_EVAL = False
    print("[WARN] evaluation module not found. CIDEr/METEOR/ROUGE per-split disabled.")
    print("       Only THR/NF/OTR per-split will be computed.")


# ================================================================
# Vietnamese stopwords (shared with compute_thr_nf.py)
# ================================================================
STOPWORDS = {
    "và", "của", "có", "là", "được", "cho", "với", "trong", "trên", "dưới",
    "này", "đó", "những", "các", "một", "không", "rất", "cũng", "đã", "sẽ",
    "đang", "từ", "đến", "về", "tại", "ở", "bên", "như", "khi", "nếu", "vì",
    "do", "để", "mà", "nào", "gì", "ai", "đâu", "sao", "thì", "lại", "ra",
    "vào", "lên", "xuống", "qua", "sang", "theo", "bằng", "giữa", "sau",
    "trước", "ngoài", "nên", "hay", "hoặc", "nhưng", "tuy", "dù", "bị",
    "phải", "cần", "muốn", "biết", "thấy", "làm", "đi", "tới", "hơn",
    "nhất", "quá", "khá", "rồi", "vẫn", "còn", "chỉ", "mới", "đều",
    "toàn", "cả", "mọi", "bao", "nhiều", "ít", "nữa", "rằng",
    "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười",
    "trăm", "nghìn", "ngàn", "triệu", "tỷ",
    "hình", "ảnh", "bức", "tấm", "chiếc", "cái", "con", "người", "đồ",
    "màu", "đỏ", "xanh", "trắng", "đen", "vàng", "nâu", "hồng", "tím",
    "cam", "sáng", "tối", "lớn", "nhỏ", "cao", "thấp", "dài", "ngắn",
    "rộng", "hẹp", "phía", "cạnh", "gần", "xa", "quanh", "góc",
    "nền", "mặt", "phần", "đoạn", "dãy", "mép", "chính", "khác",
    "viết", "ghi", "đặt", "treo", "dán", "chữ", "số", "dòng",
    "bảng", "biển", "hiệu", "cửa", "hàng", "quán", "nhà", "phố",
    "đường", "xe", "máy", "ô", "tô", "buýt", "thực", "đơn", "giá",
    "tiền", "bán", "mua", "đồng", "vnd", "vn", "việt", "nam",
    "thể", "hiện", "nhìn", "ngồi", "đứng", "nằm", "chạy", "đọc",
    "trông", "mang", "cầm", "giữ", "đưa", "lấy", "đem", "bỏ",
    "đẹp", "xấu", "tốt", "mới", "cũ", "khác", "giống", "riêng",
    "chung", "đầy", "trống", "sạch", "bẩn", "nóng", "lạnh",
    "ăn", "uống", "ngon", "món", "cơm", "nước", "thức",
    "trang", "trí", "quảng", "cáo", "thông", "tin", "báo",
    "khu", "vực", "trung", "tâm", "thành", "thị", "xã",
    "logo", "menu", "the", "and", "of", "for", "in", "on", "at",
}


# ================================================================
# Utility functions (shared with compute_thr_nf.py)
# ================================================================

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'^[^\w]+|[^\w]+$', '', text, flags=re.UNICODE)
    return text


def levenshtein_distance(s1, s2):
    n, m = len(s1), len(s2)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def anls_score(s1, s2):
    if not s1 and not s2: return 1.0
    if not s1 or not s2: return 0.0
    max_len = max(len(s1), len(s2))
    if abs(len(s1) - len(s2)) / max_len > 0.7:
        return 0.0
    dist = levenshtein_distance(s1, s2)
    return 1.0 - dist / max_len


def is_text_candidate(word):
    if re.search(r'\d', word): return True
    if len(word) >= 2 and word.isupper(): return True
    if re.search(r'[A-Z]{2,}', word): return True
    if re.search(r'[@#&:/]', word): return True
    return False


def extract_numbers(text):
    raw_matches = re.findall(r'\d+(?:[.,]\d+)*[%kKhgm]?', text)
    results = []
    for r in raw_matches:
        norm = re.sub(r'[.,]', '', r.rstrip('%kKhgm'))
        if norm:
            results.append(norm)
    return results


# ================================================================
# OCR Feature Loading
# ================================================================

def load_ocr_data(image_id, ocr_dir):
    """Load full OCR data (texts, boxes, scores) for an image."""
    path = os.path.join(ocr_dir, f"{image_id}.npy")
    if not os.path.exists(path):
        return {"texts": [], "boxes": np.zeros((0, 4)), "scores": []}
    feat = np.load(path, allow_pickle=True)[()]
    texts = feat.get("texts", [])
    if len(texts) == 0:
        return {"texts": [], "boxes": np.zeros((0, 4)), "scores": []}

    boxes = feat.get("boxes", np.zeros((len(texts), 4)))
    if isinstance(boxes, np.ndarray) and boxes.shape[0] < 1:
        boxes = np.zeros((0, 4))

    scores = feat.get("scores", [0.0] * len(texts))
    if hasattr(scores, 'tolist'):
        scores = scores.tolist()

    return {"texts": texts, "boxes": boxes, "scores": scores}


# ================================================================
# Test Set Categorization
# ================================================================

def categorize_text_dependency(gt_captions, ocr_texts, threshold=0.5):
    """Determine if captions are text-dependent.

    A caption is text-dependent if its reference contains text-like spans
    (numbers, brand names, alphanumeric strings) supported by OCR tokens.
    """
    if not ocr_texts:
        return "text_independent"

    ocr_norm = [normalize_text(t) for t in ocr_texts if normalize_text(t)]
    if not ocr_norm:
        return "text_independent"

    for cap in gt_captions:
        for word in cap.split():
            w_norm = normalize_text(word)
            if not w_norm or w_norm in STOPWORDS:
                continue
            if not is_text_candidate(word):
                continue
            # Check if this text-candidate matches any OCR token
            for ocr_t in ocr_norm:
                if anls_score(w_norm, ocr_t) >= threshold:
                    return "text_dependent"

    return "text_independent"


def categorize_text_density(ocr_texts):
    """Categorize by number of valid OCR tokens."""
    valid = [t for t in ocr_texts if t.strip() and t != "no_token"]
    n = len(valid)
    if n <= 3:
        return "sparse"
    elif n <= 10:
        return "medium"
    else:
        return "dense"


def categorize_ocr_confidence(scores):
    """Categorize by average OCR confidence."""
    if not scores or len(scores) == 0:
        return "no_ocr"
    avg = np.mean([s for s in scores if s > 0])
    if np.isnan(avg) or avg == 0:
        return "no_ocr"
    if avg < 0.5:
        return "low"
    elif avg < 0.8:
        return "medium"
    else:
        return "high"


def categorize_layout_complexity(boxes):
    """Categorize text layout by clustering OCR bounding boxes.

    Groups boxes by y-coordinate proximity (center-y) using a simple
    threshold-based clustering. Then counts the number of distinct
    horizontal text lines and spatial blocks.
    """
    if not isinstance(boxes, np.ndarray) or boxes.shape[0] == 0:
        return "no_text"

    # Filter out zero boxes
    valid_mask = boxes.sum(axis=1) > 0
    boxes = boxes[valid_mask]
    if boxes.shape[0] == 0:
        return "no_text"

    # Compute center-y and box heights
    cy = (boxes[:, 1] + boxes[:, 3]) / 2  # center y
    h = np.maximum(boxes[:, 3] - boxes[:, 1], 1e-6)  # heights
    median_h = np.median(h)

    # Sort by center-y
    sorted_cy = np.sort(cy)

    # Cluster lines: gap > 0.6 * median_height = new line
    n_lines = 1
    for i in range(1, len(sorted_cy)):
        if sorted_cy[i] - sorted_cy[i - 1] > 0.6 * median_h:
            n_lines += 1

    # Further check for multi-block: cluster by x-coordinate gaps
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    w = np.maximum(boxes[:, 2] - boxes[:, 0], 1e-6)
    median_w = np.median(w)

    # Count x-clusters (blocks) within each line
    # Simple approach: count large x-gaps
    sorted_cx = np.sort(cx)
    n_x_gaps = 0
    for i in range(1, len(sorted_cx)):
        if sorted_cx[i] - sorted_cx[i - 1] > 3.0 * median_w:
            n_x_gaps += 1

    if n_lines <= 1 and n_x_gaps <= 0:
        return "single_line"
    elif n_lines <= 3 and n_x_gaps <= 1:
        return "multi_line"
    else:
        return "multi_block"


def build_categories(test_json_path, ocr_dir):
    """Categorize all test images across 4 dimensions.

    Returns: dict mapping image_id -> {text_dep, density, confidence, layout}
    """
    with open(test_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Build image_id -> GT captions mapping
    img_captions = defaultdict(list)
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        for answer in ann["answers"]:
            if isinstance(answer, list):
                answer = " ".join(answer)
            img_captions[img_id].append(answer)

    # Collect unique image IDs
    image_ids = list(img_captions.keys())
    print(f"Categorizing {len(image_ids)} unique test images...")

    categories = {}
    for i, img_id in enumerate(image_ids):
        ocr_data = load_ocr_data(img_id, ocr_dir)

        categories[img_id] = {
            "text_dep": categorize_text_dependency(
                img_captions[img_id], ocr_data["texts"]
            ),
            "density": categorize_text_density(ocr_data["texts"]),
            "confidence": categorize_ocr_confidence(ocr_data["scores"]),
            "layout": categorize_layout_complexity(ocr_data["boxes"]),
            "n_ocr_tokens": len([t for t in ocr_data["texts"]
                                 if t.strip() and t != "no_token"]),
            "avg_confidence": float(np.mean(ocr_data["scores"]))
                if ocr_data["scores"] else 0.0,
        }

        if (i + 1) % 2000 == 0:
            print(f"  Processed {i + 1}/{len(image_ids)} images")

    return categories


# ================================================================
# Parse model predictions
# ================================================================

def parse_results(results_path):
    """Parse test_results.json into list of dicts with gen/gt/image_id."""
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for batch in data["results"]:
        image_ids = batch["image_id"]
        gens = batch["gens"]
        gts = batch.get("gts", {})

        for key, gen_value in gens.items():
            idx = int(key.split("_")[-1])

            if isinstance(image_ids, list):
                img_id = image_ids[idx] if idx < len(image_ids) else image_ids[-1]
            else:
                img_id = image_ids

            if isinstance(gen_value, (list, tuple)):
                caption = str(gen_value[0])
            else:
                caption = str(gen_value)

            gt_value = gts.get(key, [])
            if isinstance(gt_value, list):
                gt_texts = [str(g) for g in gt_value]
            else:
                gt_texts = [str(gt_value)]

            samples.append({
                "key": key,
                "caption": caption,
                "gt": gt_texts,
                "image_id": img_id,
            })

    return samples


# ================================================================
# Per-Split Metric Computation
# ================================================================

def compute_caption_metrics(gts_dict, gens_dict):
    """Compute CIDEr, METEOR, ROUGE using the evaluation module."""
    if not HAS_EVAL:
        return {}
    try:
        scores, _ = evaluation.compute_scores(gts_dict, gens_dict)
        return scores
    except Exception as e:
        print(f"  [WARN] evaluation.compute_scores failed: {e}")
        return {}


def compute_thr_nf_for_samples(samples, ocr_dir, anls_threshold=0.5):
    """Compute THR, NF, OTR for a list of samples."""
    corpus = defaultdict(int)

    for sample in samples:
        caption = sample["caption"]
        img_id = sample["image_id"]

        ocr_data = load_ocr_data(img_id, ocr_dir)
        ocr_norm = [normalize_text(t) for t in ocr_data["texts"]
                    if normalize_text(t)]

        # THR
        words = caption.split()
        for w in words:
            w_norm = normalize_text(w)
            if not w_norm or w_norm in STOPWORDS:
                continue
            if not is_text_candidate(w):
                continue
            corpus["text_candidates"] += 1
            matched = any(anls_score(w_norm, o) >= anls_threshold
                         for o in ocr_norm)
            if not matched:
                corpus["hallucinated"] += 1

        # NF
        cap_nums = extract_numbers(caption)
        ocr_nums = set()
        for t in ocr_data["texts"]:
            ocr_nums.update(extract_numbers(t))

        for cn in cap_nums:
            corpus["nums_in_caption"] += 1
            if cn in ocr_nums or any(anls_score(cn, on) >= 0.8
                                      for on in ocr_nums):
                corpus["nums_matched"] += 1

        # OTR
        cap_words_norm = [normalize_text(w) for w in words]
        for ocr_t in ocr_norm:
            if not ocr_t:
                continue
            corpus["ocr_tokens"] += 1
            if any(anls_score(ocr_t, cw) >= anls_threshold
                   for cw in cap_words_norm):
                corpus["ocr_recalled"] += 1

    return {
        "THR": (corpus["hallucinated"] / corpus["text_candidates"]
                if corpus["text_candidates"] > 0 else 0.0),
        "NF": (corpus["nums_matched"] / corpus["nums_in_caption"]
               if corpus["nums_in_caption"] > 0 else 1.0),
        "OTR": (corpus["ocr_recalled"] / corpus["ocr_tokens"]
                if corpus["ocr_tokens"] > 0 else 0.0),
        "n_samples": len(samples),
        "text_candidates": corpus["text_candidates"],
        "hallucinated": corpus["hallucinated"],
        "nums_in_caption": corpus["nums_in_caption"],
        "nums_matched": corpus["nums_matched"],
    }


def evaluate_split(samples, ocr_dir):
    """Compute all metrics for a subset of samples."""
    results = {}

    # Caption quality metrics (CIDEr, METEOR, ROUGE)
    gts_dict = {}
    gens_dict = {}
    for s in samples:
        gts_dict[s["key"]] = s["gt"]
        gens_dict[s["key"]] = [s["caption"]]

    caption_scores = compute_caption_metrics(gts_dict, gens_dict)
    results.update(caption_scores)

    # THR / NF / OTR
    thr_nf = compute_thr_nf_for_samples(samples, ocr_dir)
    results.update(thr_nf)

    return results


# ================================================================
# Main Analysis Pipeline
# ================================================================

def run_analysis(test_json_path, ocr_dir, results_paths, model_names, output_path):
    """Run all fine-grained analyses and save results."""

    # Step 1: Categorize test images
    print("\n" + "=" * 70)
    print("STEP 1: Categorizing test images")
    print("=" * 70)
    categories = build_categories(test_json_path, ocr_dir)

    # Print category distribution
    dims = ["text_dep", "density", "confidence", "layout"]
    for dim in dims:
        counts = defaultdict(int)
        for img_id, cats in categories.items():
            counts[cats[dim]] += 1
        print(f"\n  {dim}:")
        for k, v in sorted(counts.items()):
            print(f"    {k}: {v} images ({100 * v / len(categories):.1f}%)")

    # Step 2: Per-model, per-split evaluation
    print("\n" + "=" * 70)
    print("STEP 2: Per-split evaluation for each model")
    print("=" * 70)

    all_results = {}

    for model_name, results_path in zip(model_names, results_paths):
        print(f"\n--- {model_name}: {results_path} ---")
        if not os.path.exists(results_path):
            print(f"  [SKIP] File not found: {results_path}")
            continue

        samples = parse_results(results_path)
        print(f"  Loaded {len(samples)} predictions")

        model_results = {"overall": {}, "splits": {}}

        # Overall metrics
        print(f"  Computing overall metrics...")
        model_results["overall"] = evaluate_split(samples, ocr_dir)

        # Per dimension splits
        for dim in dims:
            model_results["splits"][dim] = {}

            # Group samples by category
            split_samples = defaultdict(list)
            for s in samples:
                img_id = s["image_id"]
                if img_id in categories:
                    cat = categories[img_id][dim]
                    split_samples[cat].append(s)
                else:
                    split_samples["unknown"].append(s)

            for cat_name, cat_samples in sorted(split_samples.items()):
                if cat_name == "unknown" or len(cat_samples) < 10:
                    continue
                print(f"  {dim}/{cat_name}: {len(cat_samples)} samples...")
                split_result = evaluate_split(cat_samples, ocr_dir)
                model_results["splits"][dim][cat_name] = split_result

        all_results[model_name] = model_results

    # Step 3: Print summary tables
    print("\n" + "=" * 70)
    print("STEP 3: Summary Tables")
    print("=" * 70)

    # Table 1: Text Dependency
    print_split_table("Caption Dependency on Scene Text", all_results,
                      "text_dep", model_names)

    # Table 2: Text Density
    print_split_table("Impact of Text Density", all_results,
                      "density", model_names)

    # Table 3: OCR Confidence
    print_split_table("OCR Confidence Sensitivity", all_results,
                      "confidence", model_names)

    # Table 4: Layout Complexity
    print_split_table("Layout Complexity", all_results,
                      "layout", model_names)

    # Step 4: Save all results
    # Convert numpy types to native Python for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        return obj

    save_data = {
        "category_distribution": {},
        "model_results": make_serializable(all_results),
    }

    for dim in dims:
        counts = defaultdict(int)
        for img_id, cats in categories.items():
            counts[cats[dim]] += 1
        save_data["category_distribution"][dim] = dict(counts)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    return all_results


def print_split_table(title, all_results, dim, model_names):
    """Print a formatted table for one analysis dimension."""
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")

    # Collect all categories in this dimension
    cats = set()
    for name in model_names:
        if name in all_results and dim in all_results[name].get("splits", {}):
            cats.update(all_results[name]["splits"][dim].keys())
    cats = sorted(cats)

    if not cats:
        print("  (no data)")
        return

    # Header
    metrics_to_show = ["CIDEr", "METEOR", "THR", "NF"]
    header = f"{'Model':<16} {'Split':<16}"
    for m in metrics_to_show:
        header += f" {m:>8}"
    header += f" {'N':>6}"
    print(header)
    print("-" * 70)

    for name in model_names:
        if name not in all_results:
            continue
        splits = all_results[name].get("splits", {}).get(dim, {})

        for cat in cats:
            if cat not in splits:
                continue
            r = splits[cat]
            row = f"{name:<16} {cat:<16}"
            for m in metrics_to_show:
                val = r.get(m, 0.0)
                if isinstance(val, (list, tuple)):
                    val = val[0] if val else 0.0
                row += f" {val:>8.4f}"
            row += f" {r.get('n_samples', 0):>6}"
            print(row)
        print()


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-grained analysis for HSTFG paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test_json", required=True,
                        help="Path to vitextcaps_test.json")
    parser.add_argument("--ocr_dir", default="./features/swintextspotter",
                        help="Path to SwinTextSpotter .npy features")
    parser.add_argument("--results", nargs="+", required=True,
                        help="Path(s) to test_results.json")
    parser.add_argument("--names", nargs="+", required=True,
                        help="Model names (must match --results count)")
    parser.add_argument("--output", default="analysis_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    if len(args.names) != len(args.results):
        parser.error("--names count must match --results count")

    run_analysis(args.test_json, args.ocr_dir, args.results,
                 args.names, args.output)


if __name__ == "__main__":
    main()
