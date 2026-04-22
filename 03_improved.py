"""
Script 3: IMPROVED v2 - Category-Level Averaged UCA Descriptions
=================================================================
Key fix: Instead of using video-specific UCA sentences,
we now average ALL sentences from the same category across
the entire UCA dataset. This gives a much richer and more
general representation of each anomaly type.

Why this is better:
- Video-specific sentences describe one specific event
- Category-averaged sentences capture the general concept
- More sentences = more robust CLIP text embedding
"""

import os
import json
import numpy as np
import torch
import clip
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

# ─── CONFIG ───────────────────────────────────────────────────────────────────
FEATURE_DIR      = "features"
ANNOTATION_DIR   = "data/annotations"
RESULTS_DIR      = "results/improved"
DEVICE           = "cpu"
ALPHA            = 0.2
THRESHOLD_METHOD = "mean"
FIXED_THRESHOLD  = 0.81
MAX_SENTENCES_PER_CAT = 30   # use up to 30 sentences per category
# ──────────────────────────────────────────────────────────────────────────────

ANOMALY_CATEGORIES = ["abuse", "arrest", "assault", "burglary", "fighting"]
NORMAL_CATEGORY    = "normal"

# Keyword mapping to match UCA video names to categories
CATEGORY_KEYWORDS = {
    "abuse":    ["abuse"],
    "arrest":   ["arrest"],
    "assault":  ["assault"],
    "burglary": ["burglary"],
    "fighting": ["fighting"],
    "normal":   ["normal"]
}


def load_clip_model():
    print("[INFO] Loading CLIP model...")
    model, _ = clip.load("ViT-B/16", device=DEVICE)
    model.eval()
    return model


def load_uca_annotations():
    """Load all UCA JSON annotation files."""
    annotations = {}
    for split in ["Train", "Test", "Val"]:
        path = os.path.join(ANNOTATION_DIR, f"UCFCrime_{split}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            annotations.update(data)
            print(f"  Loaded {len(data)} entries from UCFCrime_{split}.json")
        else:
            print(f"  [WARNING] Not found: {path}")
    print(f"  Total: {len(annotations)} annotations")
    return annotations


def build_category_sentences(annotations):
    """
    KEY IMPROVEMENT: Group ALL UCA sentences by category.
    Instead of per-video sentences, we collect all sentences
    belonging to each category and average them.
    """
    category_sentences = defaultdict(list)

    for video_name, data in annotations.items():
        sentences = data.get("sentences", [])
        if not sentences:
            continue

        # Determine category from video name
        video_lower = video_name.lower()
        assigned = False
        for cat, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in video_lower:
                    category_sentences[cat].extend(sentences)
                    assigned = True
                    break
            if assigned:
                break

    # Report how many sentences per category
    print("\n[INFO] Category sentence counts from UCA:")
    for cat, sents in category_sentences.items():
        print(f"  {cat:12s}: {len(sents)} sentences")

    return category_sentences


def encode_sentences_averaged(model, sentences, max_sentences=30):
    """
    Encode multiple sentences and return their AVERAGED feature vector.
    More sentences = more robust representation.
    Randomly sample if too many.
    """
    if len(sentences) > max_sentences:
        # Sample evenly across the list for diversity
        indices = np.linspace(0, len(sentences)-1, max_sentences, dtype=int)
        sentences = [sentences[i] for i in indices]

    features = []
    with torch.no_grad():
        for sent in sentences:
            sent = sent.strip()[:200]
            if not sent:
                continue
            try:
                tokens = clip.tokenize([sent]).to(DEVICE)
                feat   = model.encode_text(tokens)
                feat   = feat / feat.norm(dim=-1, keepdim=True)
                features.append(feat.cpu().numpy())
            except Exception:
                continue

    if not features:
        return None

    # Average all features → single rich representation
    avg = np.mean(np.vstack(features), axis=0, keepdims=True)
    avg = avg / (np.linalg.norm(avg) + 1e-8)
    return avg


def compute_similarity(visual_features, text_feature):
    return (visual_features @ text_feature.T).squeeze()


def compute_anomaly_score(visual_features, anomaly_feat,
                           normal_feat, alpha=0.2, method="mean"):
    """Compute frame-level anomaly scores using fused similarity."""
    S_an = compute_similarity(visual_features, normal_feat)
    S_aa = compute_similarity(visual_features, anomaly_feat)

    eps = 1e-8
    S_an_norm = S_an / (np.linalg.norm(S_an) + eps)
    S_aa_norm = S_aa / (np.linalg.norm(S_aa) + eps)

    psi = alpha * S_aa_norm + (1 - alpha) * (1 - S_an_norm)

    if method == "fixed":
        theta = FIXED_THRESHOLD
    elif method == "mean":
        theta = np.mean(psi)
    elif method == "percentile80":
        theta = np.percentile(psi, 80)
    else:
        theta = np.mean(psi)

    pseudo_labels = (psi >= theta).astype(int)
    return pseudo_labels, psi, theta


def load_features():
    meta_path = os.path.join(FEATURE_DIR, "video_metadata.json")
    with open(meta_path, "r") as f:
        return json.load(f)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("IMPROVED v2: Category-Averaged UCA Descriptions")
    print("=" * 60)

    # Load CLIP
    model = load_clip_model()

    # Load UCA annotations
    print("\n[INFO] Loading UCA annotations...")
    annotations = load_uca_annotations()

    # Build category-level sentence pools
    category_sentences = build_category_sentences(annotations)

    # ── Pre-encode ALL category text features ─────────────────────
    print("\n[INFO] Encoding category-averaged text features...")
    category_text_feats = {}

    for cat in ANOMALY_CATEGORIES + [NORMAL_CATEGORY]:
        sents = category_sentences.get(cat, [])
        if sents:
            feat = encode_sentences_averaged(
                model, sents, max_sentences=MAX_SENTENCES_PER_CAT
            )
            if feat is not None:
                category_text_feats[cat] = feat
                print(f"  ✓ {cat:12s}: encoded from {min(len(sents), MAX_SENTENCES_PER_CAT)} sentences")
        else:
            # Fallback to simple label if no UCA sentences found
            with torch.no_grad():
                tokens = clip.tokenize([cat]).to(DEVICE)
                feat   = model.encode_text(tokens)
                feat   = feat / feat.norm(dim=-1, keepdim=True)
                category_text_feats[cat] = feat.cpu().numpy()
            print(f"  ! {cat:12s}: fallback to simple label (no UCA sentences)")

    # Normal text feature
    normal_feat = category_text_feats[NORMAL_CATEGORY]

    # Load video features
    metadata = load_features()
    print(f"\n[INFO] Processing {len(metadata)} videos...")

    all_scores  = []
    all_labels  = []
    results     = {}

    for video_name, info in tqdm(metadata.items(), desc="Processing"):
        category   = info["category"].lower()
        true_label = info["label"]
        feat_path  = info["feature_path"]

        if not os.path.exists(feat_path):
            continue

        visual_feats = np.load(feat_path).astype(np.float32)

        # Get the category-level text feature
        if category in category_text_feats:
            anomaly_feat = category_text_feats[category]
        else:
            anomaly_feat = category_text_feats.get(
                NORMAL_CATEGORY, list(category_text_feats.values())[0]
            )

        # Compute scores for ALL videos (normal and anomaly)
        pseudo_labels, psi, theta = compute_anomaly_score(
            visual_feats, anomaly_feat, normal_feat,
            alpha=ALPHA, method=THRESHOLD_METHOD
        )

        video_score = float(np.max(psi))
        all_scores.append(video_score)
        all_labels.append(true_label)

        results[video_name] = {
            "category":       category,
            "true_label":     true_label,
            "video_score":    video_score,
            "mean_score":     float(np.mean(psi)),
            "pseudo_labels":  pseudo_labels.tolist(),
            "threshold":      float(theta),
            "num_frames":     len(visual_feats),
            "anomaly_frames": int(np.sum(pseudo_labels)),
            "text_source":    "category_averaged_uca"
        }

    # ── Evaluate ──────────────────────────────────────────────────
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    print(f"\n[INFO] Score summary:")
    print(f"  Normal  videos mean score: "
          f"{np.mean(all_scores[all_labels==0]):.4f}")
    print(f"  Anomaly videos mean score: "
          f"{np.mean(all_scores[all_labels==1]):.4f}")
    print(f"  Score gap: "
          f"{np.mean(all_scores[all_labels==1]) - np.mean(all_scores[all_labels==0]):.4f}")

    auc = roc_auc_score(all_labels, all_scores)
    fpr, tpr, _ = roc_curve(all_labels, all_scores)

    print(f"\n{'='*40}")
    print(f"  IMPROVED v2 AUC: {auc:.4f} ({auc*100:.2f}%)")
    print(f"  Threshold: {THRESHOLD_METHOD}")
    print(f"{'='*40}")

    # ── Save ──────────────────────────────────────────────────────
    summary = {
        "model":              "Improved v2 (Category-Averaged UCA)",
        "threshold_method":   THRESHOLD_METHOD,
        "alpha":              ALPHA,
        "auc":                float(auc),
        "num_videos":         len(all_labels),
        "normal_videos":      int(np.sum(all_labels == 0)),
        "anomaly_videos":     int(np.sum(all_labels == 1)),
        "normal_mean_score":  float(np.mean(all_scores[all_labels==0])),
        "anomaly_mean_score": float(np.mean(all_scores[all_labels==1])),
        "score_gap":          float(np.mean(all_scores[all_labels==1]) -
                                    np.mean(all_scores[all_labels==0])),
        "sentences_per_cat":  MAX_SENTENCES_PER_CAT
    }

    with open(os.path.join(RESULTS_DIR, "improved_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "improved_per_video.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── ROC Plot ──────────────────────────────────────────────────
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="red", lw=2,
             label=f"Improved v2 ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "gray", linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Improved v2: ROC Curve\n(Category-Averaged UCA Descriptions)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "improved_roc.png"), dpi=150)
    plt.close()

    # ── Score Bar Chart ───────────────────────────────────────────
    vnames  = list(results.keys())
    vscores = [results[v]["video_score"] for v in vnames]
    vlabels = [results[v]["true_label"]  for v in vnames]
    colors  = ["red" if l == 1 else "green" for l in vlabels]

    plt.figure(figsize=(14, 5))
    plt.bar(range(len(vnames)), vscores, color=colors, alpha=0.7)
    plt.xlabel("Video Index")
    plt.ylabel("Anomaly Score")
    plt.title("Improved v2: Anomaly Scores per Video\n"
              "(Red=Anomaly, Green=Normal) — Category-Averaged UCA")
    plt.axhline(y=np.mean(vscores), color="black",
                linestyle="--", label="Mean score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "improved_scores.png"), dpi=150)
    plt.close()

    print(f"\n[SAVED] Plots → {RESULTS_DIR}/")
    print(f"[DONE]  Improved v2 AUC = {auc*100:.2f}%")
    return auc


if __name__ == "__main__":
    main()
