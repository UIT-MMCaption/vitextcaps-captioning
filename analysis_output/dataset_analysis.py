#!/usr/bin/env python3
"""
ViTextCaps Dataset Analysis for HSTFG Paper
=============================================

Computes comprehensive dataset statistics for the paper's Dataset Analysis
section, inspired by OpenViVQA, LiGT, ViTextVQA, ViOCRVQA.

JSON format: {seq_key: {"image_id": str, "caption": str}, ...}
Multiple entries can share the same image_id (multiple captions per image).

Outputs:
  - dataset_stats.json: all statistics
  - dataset_stats_latex.tex: LaTeX table snippets
  - Matplotlib figures (PDF) for paper inclusion

Usage:
    python dataset_analysis.py --output_dir analysis_output
    python dataset_analysis.py --output_dir analysis_output --skip_figures
"""

import json
import os
import re
import argparse
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

# ================================================================
# Matplotlib setup (non-interactive backend for server)
# ================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Publication style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

COLORS = {'train': '#2196F3', 'dev': '#FF9800', 'test': '#4CAF50'}
SPLIT_LABELS = {'train': 'Train', 'dev': 'Dev', 'test': 'Test'}


# ================================================================
# Vietnamese stopwords
# ================================================================
STOPWORDS = {
    "va", "cua", "co", "la", "duoc", "cho", "voi", "trong", "tren", "duoi",
    "nay", "do", "nhung", "cac", "mot", "khong", "rat", "cung", "da", "se",
    "dang", "tu", "den", "ve", "tai", "o", "ben", "nhu", "khi", "neu", "vi",
    "de", "ma", "nao", "gi", "ai", "dau", "sao", "thi", "lai", "ra",
    "vao", "len", "xuong", "qua", "sang", "theo", "bang", "giua", "sau",
    "truoc", "ngoai", "nen", "hay", "hoac", "nhung", "tuy", "du", "bi",
    "phai", "can", "muon", "biet", "thay", "lam", "di", "toi", "hon",
    "nhat", "qua", "kha", "roi", "van", "con", "chi", "moi", "deu",
    "toan", "ca", "moi", "bao", "nhieu", "it", "nua", "rang",
    # Vietnamese with diacritics
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
    "nội", "dung", "gồm", "phong", "cảnh", "kèm",
}


# ================================================================
# Utility functions (reused from compute_thr_nf.py)
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
    return 1.0 - levenshtein_distance(s1, s2) / max_len


def is_text_candidate(word):
    if re.search(r'\d', word): return True
    if len(word) >= 2 and word.isupper(): return True
    if re.search(r'[A-Z]{2,}', word): return True
    if re.search(r'[@#&:/]', word): return True
    return False


# ================================================================
# Data loading & parsing
# ================================================================

def load_json_split(path):
    """Load JSON and parse into annotations list + unique image IDs.

    JSON format: {seq_key: {"image_id": str, "caption": str}, ...}

    Returns:
        annotations: list of {"image_id": str, "caption": str}
        unique_image_ids: sorted list of unique image IDs
        image_to_captions: dict mapping image_id -> [caption1, caption2, ...]
    """
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    annotations = []
    image_to_captions = defaultdict(list)

    for seq_key, entry in raw.items():
        img_id = str(entry["image_id"])
        caption = entry["caption"]
        annotations.append({"image_id": img_id, "caption": caption})
        image_to_captions[img_id].append(caption)

    unique_image_ids = sorted(image_to_captions.keys())

    return annotations, unique_image_ids, dict(image_to_captions)


def load_ocr_data(image_id, ocr_dir):
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


def load_visual_data(image_id, vis_dir):
    path = os.path.join(vis_dir, f"{image_id}.npy")
    if not os.path.exists(path):
        return {"n_regions": 0}
    feat = np.load(path, allow_pickle=True)[()]
    rf = feat.get("region_features", np.zeros((0, 1)))
    n = rf.shape[0] if hasattr(rf, 'shape') else 0
    return {"n_regions": n}


# ================================================================
# Layout complexity (reused from analysis_finegrained.py)
# ================================================================

def categorize_layout_complexity(boxes):
    if not isinstance(boxes, np.ndarray) or boxes.shape[0] == 0:
        return "no_text"
    valid_mask = boxes.sum(axis=1) > 0
    boxes = boxes[valid_mask]
    if boxes.shape[0] == 0:
        return "no_text"

    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    h = np.maximum(boxes[:, 3] - boxes[:, 1], 1e-6)
    median_h = np.median(h)

    sorted_cy = np.sort(cy)
    n_lines = 1
    for i in range(1, len(sorted_cy)):
        if sorted_cy[i] - sorted_cy[i - 1] > 0.6 * median_h:
            n_lines += 1

    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    w = np.maximum(boxes[:, 2] - boxes[:, 0], 1e-6)
    median_w = np.median(w)
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


# ================================================================
# 1. Basic Dataset Statistics
# ================================================================

def compute_basic_stats(annotations, unique_image_ids, image_to_captions):
    n_images = len(unique_image_ids)
    n_annotations = len(annotations)  # = n_captions (1 caption per annotation)

    cap_lens = []
    all_words = []

    for ann in annotations:
        caption = ann["caption"]
        words = caption.split()
        cap_lens.append(len(words))
        all_words.extend(words)

    vocab = set(w.lower() for w in all_words)

    # Captions per image distribution
    caps_per_image = [len(caps) for caps in image_to_captions.values()]

    return {
        "n_images": n_images,
        "n_annotations": n_annotations,
        "n_captions": n_annotations,
        "avg_captions_per_image": float(np.mean(caps_per_image)) if caps_per_image else 0,
        "median_captions_per_image": float(np.median(caps_per_image)) if caps_per_image else 0,
        "caption_length_mean": float(np.mean(cap_lens)) if cap_lens else 0,
        "caption_length_median": float(np.median(cap_lens)) if cap_lens else 0,
        "caption_length_std": float(np.std(cap_lens)) if cap_lens else 0,
        "caption_length_min": int(min(cap_lens)) if cap_lens else 0,
        "caption_length_max": int(max(cap_lens)) if cap_lens else 0,
        "vocabulary_size": len(vocab),
        "total_words": len(all_words),
        "caption_lengths": cap_lens,  # raw data for histograms
    }


# ================================================================
# 2. OCR Text Statistics
# ================================================================

def compute_ocr_stats(unique_image_ids, ocr_dir):
    n_tokens_list = []
    avg_conf_list = []
    all_conf = []

    for img_id in tqdm(unique_image_ids, desc="  OCR stats", leave=False):
        ocr = load_ocr_data(img_id, ocr_dir)
        valid_texts = [t for t in ocr["texts"] if t.strip() and t != "no_token"]
        n_tokens_list.append(len(valid_texts))

        valid_scores = [s for s in ocr["scores"] if isinstance(s, (int, float)) and s > 0]
        if valid_scores:
            avg_conf_list.append(float(np.mean(valid_scores)))
            all_conf.extend(valid_scores)

    n_arr = np.array(n_tokens_list) if n_tokens_list else np.array([0])

    # Text density buckets
    density = {
        "no_text": int((n_arr == 0).sum()),
        "sparse": int(((n_arr >= 1) & (n_arr <= 3)).sum()),
        "medium": int(((n_arr >= 4) & (n_arr <= 10)).sum()),
        "dense": int((n_arr >= 11).sum()),
    }
    n_total = len(n_arr)
    density_pct = {k: 100 * v / max(n_total, 1) for k, v in density.items()}

    # Confidence buckets
    conf_arr = np.array(avg_conf_list) if avg_conf_list else np.array([])
    confidence = {}
    if len(conf_arr) > 0:
        confidence = {
            "low": int((conf_arr < 0.5).sum()),
            "medium": int(((conf_arr >= 0.5) & (conf_arr < 0.8)).sum()),
            "high": int((conf_arr >= 0.8).sum()),
        }
    conf_total = len(conf_arr)
    confidence_pct = {k: 100 * v / max(conf_total, 1) for k, v in confidence.items()}

    return {
        "tokens_per_image_mean": float(n_arr.mean()),
        "tokens_per_image_median": float(np.median(n_arr)),
        "tokens_per_image_std": float(n_arr.std()),
        "tokens_per_image_max": int(n_arr.max()),
        "zero_ocr_pct": float(100 * (n_arr == 0).sum() / max(n_total, 1)),
        "avg_confidence": float(np.mean(all_conf)) if all_conf else 0.0,
        "avg_confidence_std": float(np.std(all_conf)) if all_conf else 0.0,
        "density": density,
        "density_pct": density_pct,
        "confidence": confidence,
        "confidence_pct": confidence_pct,
        "n_tokens_raw": n_tokens_list,  # raw data for histograms
        "avg_conf_raw": avg_conf_list,  # raw data for histograms
    }


# ================================================================
# 3. Caption-Text Dependency Analysis
# ================================================================

def compute_text_dependency(annotations, unique_image_ids, ocr_dir):
    # Build image_id -> normalized OCR tokens cache
    ocr_cache = {}
    for img_id in tqdm(unique_image_ids, desc="  Loading OCR for dependency", leave=False):
        ocr = load_ocr_data(img_id, ocr_dir)
        ocr_cache[img_id] = [normalize_text(t) for t in ocr["texts"]
                             if normalize_text(t)]

    text_dep_count = 0
    text_indep_count = 0
    text_types = Counter()  # numbers, brands, mixed

    for ann in tqdm(annotations, desc="  Text dependency", leave=False):
        img_id = ann["image_id"]
        caption = ann["caption"]
        ocr_norm = ocr_cache.get(img_id, [])
        is_dependent = False

        for word in caption.split():
            w_norm = normalize_text(word)
            if not w_norm or w_norm in STOPWORDS:
                continue
            if not is_text_candidate(word):
                continue
            for ocr_t in ocr_norm:
                if anls_score(w_norm, ocr_t) >= 0.5:
                    is_dependent = True
                    # Classify text type
                    if re.search(r'\d', word):
                        text_types["numeric"] += 1
                    elif word.isupper() and len(word) >= 2:
                        text_types["brand_abbrev"] += 1
                    else:
                        text_types["other"] += 1
                    break
            if is_dependent:
                break

        if is_dependent:
            text_dep_count += 1
        else:
            text_indep_count += 1

    total = text_dep_count + text_indep_count
    return {
        "text_dependent": text_dep_count,
        "text_independent": text_indep_count,
        "text_dependent_pct": 100 * text_dep_count / max(total, 1),
        "text_independent_pct": 100 * text_indep_count / max(total, 1),
        "text_types": dict(text_types.most_common(10)),
    }


# ================================================================
# 4. Layout Complexity
# ================================================================

def compute_layout_stats(unique_image_ids, ocr_dir):
    layout_counts = Counter()

    for img_id in tqdm(unique_image_ids, desc="  Layout analysis", leave=False):
        ocr = load_ocr_data(img_id, ocr_dir)
        layout = categorize_layout_complexity(ocr["boxes"])
        layout_counts[layout] += 1

    total = sum(layout_counts.values())
    layout_pct = {k: 100 * v / max(total, 1) for k, v in layout_counts.items()}

    return {
        "layout_counts": dict(layout_counts),
        "layout_pct": layout_pct,
    }


# ================================================================
# 5. Caption Characteristics
# ================================================================

def compute_caption_stats(annotations, image_to_captions):
    word_freq = Counter()
    all_words = []

    for ann in annotations:
        caption = ann["caption"]
        words = caption.lower().split()
        all_words.extend(words)
        for w in words:
            if w not in STOPWORDS and len(w) > 1:
                word_freq[w] += 1

    vocab = set(all_words)
    ttr = len(vocab) / max(len(all_words), 1)

    # Caption diversity: average pairwise dissimilarity for images with multiple captions
    diversity_scores = []
    for img_id, caps in image_to_captions.items():
        if len(caps) >= 2:
            word_sets = [set(c.lower().split()) for c in caps]
            pairwise = []
            for i in range(len(word_sets)):
                for j in range(i + 1, len(word_sets)):
                    inter = len(word_sets[i] & word_sets[j])
                    union = len(word_sets[i] | word_sets[j])
                    if union > 0:
                        pairwise.append(1.0 - inter / union)  # Jaccard distance
            if pairwise:
                diversity_scores.append(float(np.mean(pairwise)))

    return {
        "top_30_words": dict(word_freq.most_common(30)),
        "type_token_ratio": ttr,
        "total_unique_words": len(vocab),
        "total_words": len(all_words),
        "caption_diversity_mean": float(np.mean(diversity_scores)) if diversity_scores else 0.0,
        "n_images_multi_caption": len(diversity_scores),
    }


# ================================================================
# 6. Visual Region Statistics
# ================================================================

def compute_visual_stats(unique_image_ids, vis_dir):
    n_regions = []

    for img_id in tqdm(unique_image_ids, desc="  Visual stats", leave=False):
        vdata = load_visual_data(img_id, vis_dir)
        n_regions.append(vdata["n_regions"])

    r_arr = np.array(n_regions) if n_regions else np.array([0])
    return {
        "regions_per_image_mean": float(r_arr.mean()),
        "regions_per_image_median": float(np.median(r_arr)),
        "regions_per_image_std": float(r_arr.std()),
        "regions_per_image_max": int(r_arr.max()),
    }


# ================================================================
# 7. Related Dataset Comparison (static)
# ================================================================

RELATED_DATASETS = [
    {"name": "TextCaps", "lang": "English", "task": "Captioning",
     "images": 28408, "pairs": 142040, "has_ocr": True, "avg_ocr": 5.3},
    {"name": "OpenViVQA", "lang": "Vietnamese", "task": "Open-domain VQA",
     "images": 11199, "pairs": 37914, "has_ocr": False, "avg_ocr": None},
    {"name": "ViTextVQA", "lang": "Vietnamese", "task": "Text-VQA",
     "images": 16762, "pairs": 50342, "has_ocr": True, "avg_ocr": 8.2},
    {"name": "ViOCRVQA", "lang": "Vietnamese", "task": "OCR-VQA",
     "images": 28282, "pairs": 85217, "has_ocr": True, "avg_ocr": 12.4},
]


# ================================================================
# LaTeX Output
# ================================================================

def generate_latex(all_stats, output_path):
    lines = []
    lines.append("% === Auto-generated by dataset_analysis.py ===\n")

    # Table 1: Basic Statistics
    lines.append("% Table: Basic Dataset Statistics")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Statistics of the ViTextCaps dataset.}")
    lines.append("\\label{tab:dataset_stats}")
    lines.append("\\begin{tabular}{lrrr|r}")
    lines.append("\\toprule")
    lines.append("Statistic & Train & Dev & Test & Total \\\\")
    lines.append("\\midrule")

    total_imgs = sum(all_stats[s]["basic"]["n_images"] for s in ["train", "dev", "test"])
    total_caps = sum(all_stats[s]["basic"]["n_captions"] for s in ["train", "dev", "test"])
    total_words_count = sum(all_stats[s]["basic"]["total_words"] for s in ["train", "dev", "test"])

    rows = [
        ("\\#Images",
         [all_stats[s]["basic"]["n_images"] for s in ["train", "dev", "test"]] + [total_imgs],
         "{:,}"),
        ("\\#Captions",
         [all_stats[s]["basic"]["n_captions"] for s in ["train", "dev", "test"]] + [total_caps],
         "{:,}"),
        ("Avg. captions/image",
         [all_stats[s]["basic"]["avg_captions_per_image"] for s in ["train", "dev", "test"]] + [
             total_caps / max(total_imgs, 1)],
         "{:.2f}"),
        ("Avg. caption length",
         [all_stats[s]["basic"]["caption_length_mean"] for s in ["train", "dev", "test"]] + [
             np.mean([all_stats[s]["basic"]["caption_length_mean"] for s in ["train", "dev", "test"]])],
         "{:.1f}"),
        ("Max caption length",
         [all_stats[s]["basic"]["caption_length_max"] for s in ["train", "dev", "test"]] + [
             max(all_stats[s]["basic"]["caption_length_max"] for s in ["train", "dev", "test"])],
         "{:,}"),
        ("Vocabulary size",
         [all_stats[s]["basic"]["vocabulary_size"] for s in ["train", "dev", "test"]] + [
             sum(all_stats[s]["basic"]["vocabulary_size"] for s in ["train", "dev", "test"])],
         "{:,}"),
        ("Total words",
         [all_stats[s]["basic"]["total_words"] for s in ["train", "dev", "test"]] + [total_words_count],
         "{:,}"),
    ]

    for label, vals, fmt in rows:
        formatted = " & ".join(fmt.format(v) for v in vals)
        lines.append(f"{label} & {formatted} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}\n")

    # Table 2: OCR Statistics
    lines.append("% Table: OCR Text Statistics")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{OCR text statistics across dataset splits.}")
    lines.append("\\label{tab:ocr_stats}")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("Statistic & Train & Dev & Test \\\\")
    lines.append("\\midrule")

    ocr_rows = [
        ("Avg. OCR tokens/image",
         [all_stats[s]["ocr"]["tokens_per_image_mean"] for s in ["train", "dev", "test"]],
         "{:.1f}"),
        ("Median OCR tokens/image",
         [all_stats[s]["ocr"]["tokens_per_image_median"] for s in ["train", "dev", "test"]],
         "{:.0f}"),
        ("Max OCR tokens/image",
         [all_stats[s]["ocr"]["tokens_per_image_max"] for s in ["train", "dev", "test"]],
         "{:d}"),
        ("Images w/o OCR (\\%)",
         [all_stats[s]["ocr"]["zero_ocr_pct"] for s in ["train", "dev", "test"]],
         "{:.1f}"),
        ("Avg. OCR confidence",
         [all_stats[s]["ocr"]["avg_confidence"] for s in ["train", "dev", "test"]],
         "{:.3f}"),
    ]

    for label, vals, fmt in ocr_rows:
        formatted = " & ".join(fmt.format(v) for v in vals)
        lines.append(f"{label} & {formatted} \\\\")

    lines.append("\\midrule")
    lines.append("\\multicolumn{4}{l}{\\textit{Text Density Distribution (\\%)}} \\\\")
    for cat in ["no_text", "sparse", "medium", "dense"]:
        cat_label = {"no_text": "No text (0)", "sparse": "Sparse (1--3)",
                     "medium": "Medium (4--10)", "dense": "Dense (11+)"}[cat]
        vals = [all_stats[s]["ocr"]["density_pct"].get(cat, 0) for s in ["train", "dev", "test"]]
        formatted = " & ".join(f"{v:.1f}" for v in vals)
        lines.append(f"\\quad {cat_label} & {formatted} \\\\")

    lines.append("\\midrule")
    lines.append("\\multicolumn{4}{l}{\\textit{OCR Confidence Distribution (\\%)}} \\\\")
    for cat in ["low", "medium", "high"]:
        cat_label = {"low": "Low ($<$0.5)", "medium": "Medium (0.5--0.8)",
                     "high": "High ($\\geq$0.8)"}[cat]
        vals = [all_stats[s]["ocr"]["confidence_pct"].get(cat, 0) for s in ["train", "dev", "test"]]
        formatted = " & ".join(f"{v:.1f}" for v in vals)
        lines.append(f"\\quad {cat_label} & {formatted} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}\n")

    # Table 3: Text Dependency
    lines.append("% Table: Caption-Text Dependency")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Caption dependency on scene text.}")
    lines.append("\\label{tab:text_dependency}")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("Category & Train & Dev & Test \\\\")
    lines.append("\\midrule")
    for cat in ["text_dependent", "text_independent"]:
        label = "Text-dependent" if cat == "text_dependent" else "Text-independent"
        vals = [all_stats[s]["text_dep"][f"{cat}_pct"] for s in ["train", "dev", "test"]]
        formatted = " & ".join(f"{v:.1f}\\%" for v in vals)
        lines.append(f"{label} & {formatted} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}\n")

    # Table 4: Layout Complexity
    lines.append("% Table: Layout Complexity")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Distribution of text layout complexity.}")
    lines.append("\\label{tab:layout_complexity}")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("Layout Type & Train & Dev & Test \\\\")
    lines.append("\\midrule")
    for cat in ["no_text", "single_line", "multi_line", "multi_block"]:
        label = {"no_text": "No text", "single_line": "Single-line",
                 "multi_line": "Multi-line", "multi_block": "Multi-block"}[cat]
        vals = [all_stats[s]["layout"]["layout_pct"].get(cat, 0) for s in ["train", "dev", "test"]]
        formatted = " & ".join(f"{v:.1f}\\%" for v in vals)
        lines.append(f"{label} & {formatted} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}\n")

    # Table 5: Visual Region Stats
    lines.append("% Table: Visual Region Statistics")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Visual region statistics per image.}")
    lines.append("\\label{tab:visual_stats}")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("Statistic & Train & Dev & Test \\\\")
    lines.append("\\midrule")
    if all_stats["train"].get("visual"):
        vis_rows = [
            ("Avg. regions/image",
             [all_stats[s]["visual"]["regions_per_image_mean"] for s in ["train", "dev", "test"]],
             "{:.1f}"),
            ("Median regions/image",
             [all_stats[s]["visual"]["regions_per_image_median"] for s in ["train", "dev", "test"]],
             "{:.0f}"),
            ("Max regions/image",
             [all_stats[s]["visual"]["regions_per_image_max"] for s in ["train", "dev", "test"]],
             "{:d}"),
        ]
        for label, vals, fmt in vis_rows:
            formatted = " & ".join(fmt.format(v) for v in vals)
            lines.append(f"{label} & {formatted} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}\n")

    # Table 6: Comparison with Related Datasets
    lines.append("% Table: Comparison with Related Datasets")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Comparison of ViTextCaps with related Vietnamese multimodal datasets.}")
    lines.append("\\label{tab:dataset_comparison}")
    lines.append("\\begin{tabular}{llrrl}")
    lines.append("\\toprule")
    lines.append("Dataset & Task & \\#Images & \\#Pairs & OCR \\\\")
    lines.append("\\midrule")
    for ds in RELATED_DATASETS:
        ocr_str = f"\\checkmark ({ds['avg_ocr']:.1f})" if ds['has_ocr'] and ds['avg_ocr'] else ("\\checkmark" if ds['has_ocr'] else "---")
        lines.append(f"{ds['name']} & {ds['task']} & {ds['images']:,} & {ds['pairs']:,} & {ocr_str} \\\\")

    # Add ViTextCaps row
    total_imgs_val = sum(all_stats[s]["basic"]["n_images"] for s in ["train", "dev", "test"])
    total_caps_val = sum(all_stats[s]["basic"]["n_captions"] for s in ["train", "dev", "test"])
    avg_ocr_all = np.mean([all_stats[s]["ocr"]["tokens_per_image_mean"] for s in ["train", "dev", "test"]])
    lines.append(f"\\textbf{{ViTextCaps}} & \\textbf{{Captioning}} & \\textbf{{{total_imgs_val:,}}} & \\textbf{{{total_caps_val:,}}} & \\checkmark ({avg_ocr_all:.1f}) \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}\n")

    # Table 7: Caption Characteristics
    lines.append("% Table: Caption Characteristics")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Caption characteristics across splits.}")
    lines.append("\\label{tab:caption_chars}")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("Statistic & Train & Dev & Test \\\\")
    lines.append("\\midrule")
    cap_rows = [
        ("Type-Token Ratio",
         [all_stats[s]["caption"]["type_token_ratio"] for s in ["train", "dev", "test"]],
         "{:.4f}"),
        ("Unique words",
         [all_stats[s]["caption"]["total_unique_words"] for s in ["train", "dev", "test"]],
         "{:,}"),
        ("Caption diversity (Jaccard)",
         [all_stats[s]["caption"]["caption_diversity_mean"] for s in ["train", "dev", "test"]],
         "{:.3f}"),
    ]
    for label, vals, fmt in cap_rows:
        formatted = " & ".join(fmt.format(v) for v in vals)
        lines.append(f"{label} & {formatted} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}\n")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"LaTeX tables saved to: {output_path}")


# ================================================================
# Matplotlib Figures
# ================================================================

def generate_figures(all_stats, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Fig 1: Caption length distribution
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for split in ["train", "dev", "test"]:
        cap_lens = all_stats[split]["basic"]["caption_lengths"]
        if cap_lens:
            ax.hist(cap_lens, bins=range(0, max(cap_lens) + 2), alpha=0.6,
                    color=COLORS[split], label=SPLIT_LABELS[split], density=True)
    ax.set_xlabel("Caption Length (words)")
    ax.set_ylabel("Density")
    ax.set_title("Caption Length Distribution")
    ax.legend()
    ax.set_xlim(0, 60)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_caption_length_dist.pdf"))
    plt.close(fig)
    print("  Saved fig_caption_length_dist.pdf")

    # Fig 2: OCR token count distribution
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for split in ["train", "dev", "test"]:
        tokens = all_stats[split]["ocr"]["n_tokens_raw"]
        if tokens:
            ax.hist(tokens, bins=range(0, min(max(tokens) + 2, 52)), alpha=0.6,
                    color=COLORS[split], label=SPLIT_LABELS[split], density=True)
    ax.set_xlabel("Number of OCR Tokens per Image")
    ax.set_ylabel("Density")
    ax.set_title("OCR Token Count Distribution")
    ax.legend()
    ax.set_xlim(0, 50)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_ocr_token_dist.pdf"))
    plt.close(fig)
    print("  Saved fig_ocr_token_dist.pdf")

    # Fig 3: Text density grouped bar chart
    fig, ax = plt.subplots(figsize=(7, 3.5))
    categories = ["no_text", "sparse", "medium", "dense"]
    cat_labels = ["No text\n(0)", "Sparse\n(1-3)", "Medium\n(4-10)", "Dense\n(11+)"]
    x = np.arange(len(categories))
    width = 0.25
    for i, split in enumerate(["train", "dev", "test"]):
        vals = [all_stats[split]["ocr"]["density_pct"].get(c, 0) for c in categories]
        ax.bar(x + i * width, vals, width, color=COLORS[split],
               label=SPLIT_LABELS[split], edgecolor='white', linewidth=0.5)
    ax.set_xlabel("Text Density Category")
    ax.set_ylabel("Percentage of Images (%)")
    ax.set_title("Text Density Distribution")
    ax.set_xticks(x + width)
    ax.set_xticklabels(cat_labels)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_text_density_bar.pdf"))
    plt.close(fig)
    print("  Saved fig_text_density_bar.pdf")

    # Fig 4: OCR confidence distribution
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for split in ["train", "dev", "test"]:
        conf = all_stats[split]["ocr"]["avg_conf_raw"]
        if conf:
            ax.hist(conf, bins=50, alpha=0.6, color=COLORS[split],
                    label=SPLIT_LABELS[split], density=True, range=(0, 1))
    ax.set_xlabel("Average OCR Confidence Score")
    ax.set_ylabel("Density")
    ax.set_title("OCR Confidence Distribution")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_confidence_dist.pdf"))
    plt.close(fig)
    print("  Saved fig_confidence_dist.pdf")

    # Fig 5: Layout complexity grouped bar chart
    fig, ax = plt.subplots(figsize=(7, 3.5))
    layout_cats = ["no_text", "single_line", "multi_line", "multi_block"]
    layout_labels = ["No text", "Single-line", "Multi-line", "Multi-block"]
    x = np.arange(len(layout_cats))
    width = 0.25
    for i, split in enumerate(["train", "dev", "test"]):
        vals = [all_stats[split]["layout"]["layout_pct"].get(c, 0) for c in layout_cats]
        ax.bar(x + i * width, vals, width, color=COLORS[split],
               label=SPLIT_LABELS[split], edgecolor='white', linewidth=0.5)
    ax.set_xlabel("Layout Type")
    ax.set_ylabel("Percentage of Images (%)")
    ax.set_title("Text Layout Complexity Distribution")
    ax.set_xticks(x + width)
    ax.set_xticklabels(layout_labels)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_layout_complexity_bar.pdf"))
    plt.close(fig)
    print("  Saved fig_layout_complexity_bar.pdf")

    # Fig 6: Text dependency pie charts (3 side by side)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for idx, split in enumerate(["train", "dev", "test"]):
        dep = all_stats[split]["text_dep"]["text_dependent_pct"]
        indep = all_stats[split]["text_dep"]["text_independent_pct"]
        axes[idx].pie([dep, indep],
                      labels=["Text-dep.", "Text-indep."],
                      autopct='%1.1f%%',
                      colors=['#2196F3', '#E0E0E0'],
                      startangle=90)
        axes[idx].set_title(SPLIT_LABELS[split])
    plt.suptitle("Caption Dependency on Scene Text", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_text_dependency_pie.pdf"))
    plt.close(fig)
    print("  Saved fig_text_dependency_pie.pdf")

    print(f"All figures saved to: {output_dir}")


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="ViTextCaps Dataset Analysis")
    parser.add_argument("--data_dir", default="./features",
                        help="Directory containing JSON + feature files")
    parser.add_argument("--output_dir", default="analysis_output",
                        help="Output directory for results")
    parser.add_argument("--skip_figures", action="store_true",
                        help="Skip matplotlib figure generation")
    parser.add_argument("--skip_visual", action="store_true",
                        help="Skip visual region stats (slower)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ocr_dir = os.path.join(args.data_dir, "swintextspotter")
    vis_dir = os.path.join(args.data_dir, "vinvl_vinvl")

    all_stats = {}

    for split in ["train", "dev", "test"]:
        print(f"\n{'=' * 60}")
        print(f"  Processing: {split}")
        print(f"{'=' * 60}")

        json_path = os.path.join(args.data_dir, f"vitextcaps_{split}.json")
        annotations, unique_image_ids, image_to_captions = load_json_split(json_path)

        print(f"  {len(unique_image_ids)} unique images, "
              f"{len(annotations)} annotations/captions")

        stats = {}

        print("  [1/6] Basic statistics...")
        stats["basic"] = compute_basic_stats(annotations, unique_image_ids, image_to_captions)

        print("  [2/6] OCR text statistics...")
        stats["ocr"] = compute_ocr_stats(unique_image_ids, ocr_dir)

        print("  [3/6] Caption-text dependency...")
        stats["text_dep"] = compute_text_dependency(annotations, unique_image_ids, ocr_dir)

        print("  [4/6] Layout complexity...")
        stats["layout"] = compute_layout_stats(unique_image_ids, ocr_dir)

        print("  [5/6] Caption characteristics...")
        stats["caption"] = compute_caption_stats(annotations, image_to_captions)

        if not args.skip_visual:
            print("  [6/6] Visual region statistics...")
            stats["visual"] = compute_visual_stats(unique_image_ids, vis_dir)
        else:
            print("  [6/6] Visual stats skipped")
            stats["visual"] = {}

        all_stats[split] = stats

        # Print summary
        b = stats["basic"]
        o = stats["ocr"]
        t = stats["text_dep"]
        print(f"\n  Summary:")
        print(f"    Images: {b['n_images']:,} | Captions: {b['n_captions']:,}")
        print(f"    Avg captions/image: {b['avg_captions_per_image']:.2f}")
        print(f"    Avg caption length: {b['caption_length_mean']:.1f} words")
        print(f"    Vocabulary: {b['vocabulary_size']:,} unique words")
        print(f"    Avg OCR tokens/image: {o['tokens_per_image_mean']:.1f}")
        print(f"    Text-dependent: {t['text_dependent_pct']:.1f}%")

    # === Output ===
    print(f"\n{'=' * 60}")
    print("  Generating outputs")
    print(f"{'=' * 60}")

    # Remove raw arrays before JSON serialization
    json_stats = {}
    for split, stats in all_stats.items():
        json_stats[split] = {}
        for key, val in stats.items():
            if isinstance(val, dict):
                json_stats[split][key] = {
                    k: v for k, v in val.items()
                    if not k.endswith("_raw") and k != "caption_lengths"
                }
            else:
                json_stats[split][key] = val

    json_path = os.path.join(args.output_dir, "dataset_stats.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_stats, f, indent=2, ensure_ascii=False)
    print(f"JSON saved to: {json_path}")

    latex_path = os.path.join(args.output_dir, "dataset_stats_latex.tex")
    generate_latex(all_stats, latex_path)

    if not args.skip_figures:
        print("\nGenerating figures...")
        generate_figures(all_stats, args.output_dir)

    print("\n" + "=" * 60)
    print("  Dataset analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
