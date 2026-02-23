#!/usr/bin/env python3
"""
Compute Text Hallucination Rate (THR) and Numeric Faithfulness (NF)
for Vietnamese Text-aware Image Captioning evaluation.

Metrics:
    THR (Text Hallucination Rate): Proportion of text-candidate tokens in
        generated captions that are NOT supported by OCR evidence. Lower is better.
    NF (Numeric Faithfulness): Proportion of numeric expressions in generated
        captions that match OCR-detected numbers. Higher is better.
    OTR (OCR Text Recall): Proportion of OCR tokens that appear in the
        generated caption. Higher is better.

Usage:
    # Single model
    python compute_thr_nf.py --results saved_models/hstfg_vinvl_swimspotter/test_results.json

    # Compare all models
    python compute_thr_nf.py --compare \
        --results saved_models/hstfg_vinvl_swimspotter/test_results.json \
                  saved_models/mmf_m4c_vinvl_swimspotter/test_results.json \
                  saved_models/mmf_butd_vinvl/test_results.json \
        --names HSTFG_v2 M4C BUTD

    # Custom OCR path and threshold
    python compute_thr_nf.py --results saved_models/hstfg_vinvl_swimspotter/test_results.json \
        --ocr_dir ./features/swintextspotter --anls_threshold 0.5
"""

import json
import os
import re
import argparse
import numpy as np
from collections import defaultdict


# ================================================================
# Vietnamese stopwords: common function/descriptive words that
# should NOT be treated as text references from images.
# ================================================================
STOPWORDS = {
    # function words
    "và", "của", "có", "là", "được", "cho", "với", "trong", "trên", "dưới",
    "này", "đó", "những", "các", "một", "không", "rất", "cũng", "đã", "sẽ",
    "đang", "từ", "đến", "về", "tại", "ở", "bên", "như", "khi", "nếu", "vì",
    "do", "để", "mà", "nào", "gì", "ai", "đâu", "sao", "thì", "lại", "ra",
    "vào", "lên", "xuống", "qua", "sang", "theo", "bằng", "giữa", "sau",
    "trước", "ngoài", "nên", "hay", "hoặc", "nhưng", "tuy", "dù", "bị",
    "phải", "cần", "muốn", "biết", "thấy", "làm", "đi", "tới", "hơn",
    "nhất", "quá", "khá", "rồi", "vẫn", "còn", "chỉ", "mới", "đều",
    "toàn", "cả", "mọi", "bao", "nhiều", "ít", "nữa", "rằng",
    # numbers as words
    "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười",
    "trăm", "nghìn", "ngàn", "triệu", "tỷ",
    # common descriptive words in captions
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
# Core utility functions
# ================================================================

def levenshtein_distance(s1, s2):
    """Compute Levenshtein edit distance between two strings."""
    n, m = len(s1), len(s2)
    if n == 0:
        return m
    if m == 0:
        return n

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
    """Compute ANLS (Average Normalized Levenshtein Similarity)."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    # Quick reject: if lengths are very different, similarity is low
    if abs(len(s1) - len(s2)) / max_len > 0.7:
        return 0.0
    dist = levenshtein_distance(s1, s2)
    return 1.0 - dist / max_len


def normalize_text(text):
    """Lowercase and strip punctuation from edges."""
    text = text.lower().strip()
    text = re.sub(r'^[^\w]+|[^\w]+$', '', text, flags=re.UNICODE)
    return text


def is_text_candidate(word):
    """Check if a word looks like it could be text copied from an image.

    Detects: numbers, abbreviations, brand-like tokens, codes.
    """
    # Contains digits → likely price, phone, address, year
    if re.search(r'\d', word):
        return True
    # All uppercase and length >= 2 → abbreviation or brand
    if len(word) >= 2 and word.isupper():
        return True
    # Contains 2+ consecutive uppercase → acronym embedded in word
    if re.search(r'[A-Z]{2,}', word):
        return True
    # Contains special chars suggesting code/URL/email
    if re.search(r'[@#&:/]', word):
        return True
    return False


def extract_numbers(text):
    """Extract numeric expressions from text, return normalized forms."""
    # Match number patterns: 100, 100.000, 100,000, 50%, 12h, etc.
    raw_matches = re.findall(r'\d+(?:[.,]\d+)*[%kKhgm]?', text)

    results = []
    for r in raw_matches:
        # Normalize: strip trailing unit chars, remove separators
        norm = re.sub(r'[.,]', '', r.rstrip('%kKhgm'))
        if norm:
            results.append(norm)
    return results


def best_match(word, candidates, threshold):
    """Find best ANLS match for word against candidate list."""
    best = 0.0
    for c in candidates:
        s = anls_score(word, c)
        if s > best:
            best = s
        if best >= threshold:
            break  # good enough, early exit
    return best


# ================================================================
# Data loading
# ================================================================

def load_ocr_tokens(image_id, ocr_dir):
    """Load OCR text tokens from SwinTextSpotter .npy feature file."""
    path = os.path.join(ocr_dir, f"{image_id}.npy")
    if not os.path.exists(path):
        return []
    feat = np.load(path, allow_pickle=True)[()]
    texts = feat.get("texts", [])
    if len(texts) == 0:
        return []
    return [t for t in texts if t.strip() and t != "no_token"]


def parse_results(results_path):
    """Parse test_results.json into list of (gen_caption, gt_captions, image_id)."""
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for batch in data["results"]:
        image_ids = batch["image_id"]
        gens = batch["gens"]
        gts = batch.get("gts", {})

        for key, gen_value in gens.items():
            idx = int(key.split("_")[-1])

            # image_id
            if isinstance(image_ids, list):
                img_id = image_ids[idx]
            else:
                img_id = image_ids

            # generated caption
            if isinstance(gen_value, (list, tuple)):
                caption = str(gen_value[0])
            else:
                caption = str(gen_value)

            # ground truth caption(s)
            gt_value = gts.get(key, [])
            if isinstance(gt_value, list):
                gt_texts = [str(g) for g in gt_value]
            else:
                gt_texts = [str(gt_value)]

            samples.append({
                "caption": caption,
                "gt": gt_texts,
                "image_id": img_id,
            })

    return samples


# ================================================================
# Metric computation
# ================================================================

def compute_metrics(results_path, ocr_dir, anls_threshold=0.5):
    """Compute THR, NF, and OTR for one model.

    Returns dict of metric values.
    """
    samples = parse_results(results_path)

    # Per-sample accumulators
    per_thr = []
    per_nf = []
    per_otr = []

    # Corpus-level counters
    corpus = defaultdict(int)

    for sample in samples:
        caption = sample["caption"]
        gt_texts = sample["gt"]
        img_id = sample["image_id"]

        # Load OCR tokens
        ocr_raw = load_ocr_tokens(img_id, ocr_dir)
        ocr_norm = [normalize_text(t) for t in ocr_raw if normalize_text(t)]

        # Build GT word set for filtering
        gt_words = set()
        for gt in gt_texts:
            for w in gt.split():
                gt_words.add(normalize_text(w))

        # ──────────────────────────────────────────
        # THR: Text Hallucination Rate
        # ──────────────────────────────────────────
        words = caption.split()
        n_candidates = 0
        n_hallucinated = 0

        for w in words:
            w_raw = w  # keep original for is_text_candidate
            w_norm = normalize_text(w)

            if not w_norm or w_norm in STOPWORDS:
                continue

            # Only consider "text candidate" words:
            # words with digits, uppercase patterns, special chars
            if not is_text_candidate(w_raw):
                continue

            n_candidates += 1
            match = best_match(w_norm, ocr_norm, anls_threshold)
            if match < anls_threshold:
                n_hallucinated += 1

        corpus["text_candidates"] += n_candidates
        corpus["hallucinated"] += n_hallucinated

        if n_candidates > 0:
            per_thr.append(n_hallucinated / n_candidates)
        else:
            per_thr.append(0.0)

        # ──────────────────────────────────────────
        # NF: Numeric Faithfulness
        # ──────────────────────────────────────────
        cap_nums = extract_numbers(caption)
        ocr_nums = set()
        for t in ocr_raw:
            ocr_nums.update(extract_numbers(t))

        n_matched = 0
        for cn in cap_nums:
            if cn in ocr_nums:
                n_matched += 1
            else:
                # Fuzzy match for OCR recognition errors (e.g. 100000 vs 10000O)
                for on in ocr_nums:
                    if anls_score(cn, on) >= 0.8:
                        n_matched += 1
                        break

        corpus["nums_in_caption"] += len(cap_nums)
        corpus["nums_matched"] += n_matched

        if len(cap_nums) > 0:
            per_nf.append(n_matched / len(cap_nums))
        else:
            per_nf.append(1.0)  # no numbers → trivially faithful

        # ──────────────────────────────────────────
        # OTR: OCR Text Recall
        # ──────────────────────────────────────────
        cap_words_norm = [normalize_text(w) for w in words]
        n_recalled = 0

        for ocr_t in ocr_norm:
            if not ocr_t:
                continue
            found = any(anls_score(ocr_t, cw) >= anls_threshold
                        for cw in cap_words_norm)
            if found:
                n_recalled += 1

        corpus["ocr_tokens"] += len(ocr_norm)
        corpus["ocr_recalled"] += n_recalled

        if len(ocr_norm) > 0:
            per_otr.append(n_recalled / len(ocr_norm))
        else:
            per_otr.append(1.0)

    n = len(samples)
    return {
        # Sample-averaged metrics
        "THR": sum(per_thr) / n if n else 0.0,
        "NF":  sum(per_nf) / n if n else 0.0,
        "OTR": sum(per_otr) / n if n else 0.0,
        # Corpus-level metrics
        "THR_corpus": (corpus["hallucinated"] / corpus["text_candidates"]
                       if corpus["text_candidates"] > 0 else 0.0),
        "NF_corpus":  (corpus["nums_matched"] / corpus["nums_in_caption"]
                       if corpus["nums_in_caption"] > 0 else 1.0),
        "OTR_corpus": (corpus["ocr_recalled"] / corpus["ocr_tokens"]
                       if corpus["ocr_tokens"] > 0 else 0.0),
        # Counts
        "n_samples": n,
        "text_candidates": corpus["text_candidates"],
        "hallucinated": corpus["hallucinated"],
        "nums_in_caption": corpus["nums_in_caption"],
        "nums_matched": corpus["nums_matched"],
        "ocr_tokens": corpus["ocr_tokens"],
        "ocr_recalled": corpus["ocr_recalled"],
    }


# ================================================================
# Display
# ================================================================

def print_single(name, m):
    """Print metrics for one model."""
    print(f"\n{'=' * 55}")
    print(f"  {name}")
    print(f"  {m['n_samples']} samples")
    print(f"{'=' * 55}")

    print(f"\n  THR (Text Hallucination Rate)    ↓ lower is better")
    print(f"    Sample avg : {m['THR']:.4f}")
    print(f"    Corpus     : {m['THR_corpus']:.4f}")
    print(f"    ({m['hallucinated']} hallucinated / {m['text_candidates']} text candidates)")

    print(f"\n  NF  (Numeric Faithfulness)       ↑ higher is better")
    print(f"    Sample avg : {m['NF']:.4f}")
    print(f"    Corpus     : {m['NF_corpus']:.4f}")
    print(f"    ({m['nums_matched']} matched / {m['nums_in_caption']} numbers in captions)")

    print(f"\n  OTR (OCR Text Recall)            ↑ higher is better")
    print(f"    Sample avg : {m['OTR']:.4f}")
    print(f"    Corpus     : {m['OTR_corpus']:.4f}")
    print(f"    ({m['ocr_recalled']} recalled / {m['ocr_tokens']} OCR tokens)")
    print()


def print_comparison(all_results):
    """Print side-by-side comparison table."""
    names = list(all_results.keys())
    metrics_list = list(all_results.values())

    print(f"\n{'=' * 70}")
    print("  Model Comparison")
    print(f"{'=' * 70}")

    # Header
    header = f"{'Metric':<20}"
    for name in names:
        header += f"  {name:>12}"
    print(header)
    print("-" * 70)

    rows = [
        ("THR ↓",        "THR"),
        ("THR_corpus ↓", "THR_corpus"),
        ("NF ↑",         "NF"),
        ("NF_corpus ↑",  "NF_corpus"),
        ("OTR ↑",        "OTR"),
        ("OTR_corpus ↑", "OTR_corpus"),
    ]

    for label, key in rows:
        row = f"{label:<20}"
        for m in metrics_list:
            row += f"  {m[key]:>12.4f}"
        print(row)

    print("-" * 70)

    # Count rows
    count_rows = [
        ("text candidates", "text_candidates"),
        ("hallucinated",    "hallucinated"),
        ("nums in caption", "nums_in_caption"),
        ("nums matched",    "nums_matched"),
        ("OCR tokens",      "ocr_tokens"),
        ("OCR recalled",    "ocr_recalled"),
    ]
    for label, key in count_rows:
        row = f"{label:<20}"
        for m in metrics_list:
            row += f"  {m[key]:>12d}"
        print(row)
    print()


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute THR, NF, OTR metrics for ViTextCaps captioning models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--results", nargs="+", required=True,
                        help="Path(s) to test_results.json file(s)")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Model names for display (must match --results count)")
    parser.add_argument("--ocr_dir", default="./features/swintextspotter",
                        help="Path to SwinTextSpotter .npy feature directory")
    parser.add_argument("--anls_threshold", type=float, default=0.5,
                        help="ANLS threshold for fuzzy matching (default: 0.5)")
    parser.add_argument("--compare", action="store_true",
                        help="Print comparison table when multiple models given")
    parser.add_argument("--save", default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    # Validate
    if args.names and len(args.names) != len(args.results):
        parser.error("--names count must match --results count")

    names = args.names or [os.path.basename(os.path.dirname(r)) for r in args.results]

    print(f"OCR features: {args.ocr_dir}")
    print(f"ANLS threshold: {args.anls_threshold}")

    all_results = {}
    for name, rpath in zip(names, args.results):
        print(f"\nComputing metrics for: {rpath} ...")
        m = compute_metrics(rpath, args.ocr_dir, args.anls_threshold)
        all_results[name] = m

        if not args.compare or len(args.results) == 1:
            print_single(name, m)

    if args.compare and len(args.results) > 1:
        print_comparison(all_results)

    if args.save:
        # Convert to serializable
        save_data = {}
        for name, m in all_results.items():
            save_data[name] = {k: (float(v) if isinstance(v, float) else v)
                               for k, v in m.items()}
        with open(args.save, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {args.save}")


if __name__ == "__main__":
    main()
