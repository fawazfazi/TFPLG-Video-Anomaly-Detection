"""
Script 2: BASELINE - Pseudo Label Generation with Simple Text Labels (FIXED)
=============================================================================
Fix: Compute anomaly scores for ALL videos (including normal ones)
using the same similarity matching mechanism.
This gives realistic AUC scores instead of artificial 1.0.
"""

import os
import json
import numpy as np
import torch
import clip
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────
FEATURE_DIR      = "features"
RESULTS_DIR      = "results/baseline"
DEVICE           = "cpu"
ALPHA            = 0.2
THRESHOLD_METHOD = "mean"   # "fixed", "mean", "percentile80"
FIXED_THRESHOLD  = 0.81
# ──────────────────────────────────────────────────────────────────────────────

SIMPLE_TEXT_LABELS = {
    "abuse":    "abuse",
    "arrest":   "arrest",
    "assault":  "assault",
    "burglary": "burglary",
    "fighting": "fighting",
    "normal":   "normal activity"
}

ANOMALY_CATEGORIES = ["abuse", "arrest", "assault", "burglary", "fighting"]
NORMAL_CATEGORY    = "normal"


def load_clip_model():
    print("[INFO] Loading CLIP model...")
    model, _ = clip.load("ViT-B/16", device=DEVICE)
    model.eval()
    return model


def encode_text_labels(model, labels_dict):
    text_features = {}
    with torch.no_grad():
        for key, text in labels_dict.items():
            tokens = clip.tokenize([text]).to(DEVICE)
            feat   = model.encode_text(tokens)
            feat   = feat / feat.norm(dim=-1, keepdim=True)
            text_features[key] = feat.cpu().numpy()
    return text_features


def compute_similarity(visual_features, text_feature):
    sim = visual_features @ text_feature.T
    return sim.squeeze()


def compute_anomaly_score(visual_features, text_features,
                           anomaly_cats, normal_key,
                           alpha=0.2, method="mean"):
    """
    Compute frame-level anomaly scores for ANY video (normal or anomaly).
    This is the KEY FIX — we no longer hardcode normal videos to zero.
    
    For normal videos: scores should naturally be LOW
    For anomaly videos: scores should naturally be HIGH
    This is what gives us a meaningful AUC.
    """
    T_normal = text_features[normal_key]

    # Similarity between frames and normal text
    S_an = compute_similarity(visual_features, T_normal)

    # Similarity between frames and ALL anomaly texts → take max
    S_aa_list = []
    for cat in anomaly_cats:
        if cat in text_features:
            S_aa = compute_similarity(visual_features, text_features[cat])
            S_aa_list.append(S_aa)

    S_aa = np.max(np.stack(S_aa_list, axis=0), axis=0)

    # Normalize
    eps = 1e-8
    S_an_norm = S_an / (np.linalg.norm(S_an) + eps)
    S_aa_norm = S_aa / (np.linalg.norm(S_aa) + eps)

    # Fuse (Equation 19 from paper)
    psi = alpha * S_aa_norm + (1 - alpha) * (1 - S_an_norm)

    # Threshold for pseudo labels
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


def load_features(feature_dir):
    meta_path = os.path.join(feature_dir, "video_metadata.json")
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    return metadata


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("BASELINE: Simple Text Labels (Fixed Version)")
    print("=" * 60)

    model        = load_clip_model()
    text_features = encode_text_labels(model, SIMPLE_TEXT_LABELS)

    print("\n[INFO] Text label features encoded:")
    for k, v in text_features.items():
        print(f"  '{k}' → {v.shape}")

    metadata = load_features(FEATURE_DIR)
    print(f"\n[INFO] Loaded metadata for {len(metadata)} videos")

    all_anomaly_scores = []
    all_true_labels    = []
    results_per_video  = {}

    print(f"\n[INFO] Computing scores for ALL videos (normal + anomaly)...")

    for video_name, info in tqdm(metadata.items(), desc="Processing"):
        category   = info["category"].lower()
        true_label = info["label"]
        feat_path  = info["feature_path"]

        if not os.path.exists(feat_path):
            print(f"  [SKIP] Feature not found: {feat_path}")
            continue

        visual_feats = np.load(feat_path).astype(np.float32)

        # ── KEY FIX: compute psi for ALL videos, not just anomaly ──
        pseudo_labels, psi, theta = compute_anomaly_score(
            visual_feats, text_features,
            ANOMALY_CATEGORIES, NORMAL_CATEGORY,
            alpha=ALPHA, method=THRESHOLD_METHOD
        )

        # Video-level score = max frame-level score
        video_score = float(np.max(psi))

        all_anomaly_scores.append(video_score)
        all_true_labels.append(true_label)

        results_per_video[video_name] = {
            "category":       category,
            "true_label":     true_label,
            "video_score":    video_score,
            "mean_score":     float(np.mean(psi)),
            "pseudo_labels":  pseudo_labels.tolist(),
            "threshold":      float(theta),
            "num_frames":     len(visual_feats),
            "anomaly_frames": int(np.sum(pseudo_labels))
        }

    # ── Evaluate ──────────────────────────────────────────────────
    all_scores = np.array(all_anomaly_scores)
    all_labels = np.array(all_true_labels)

    print(f"\n[INFO] Score summary:")
    print(f"  Normal videos  mean score: "
          f"{np.mean(all_scores[all_labels==0]):.4f}")
    print(f"  Anomaly videos mean score: "
          f"{np.mean(all_scores[all_labels==1]):.4f}")

    if len(np.unique(all_labels)) < 2:
        print("[WARNING] Need both normal and anomaly videos!")
        return

    auc = roc_auc_score(all_labels, all_scores)
    fpr, tpr, _ = roc_curve(all_labels, all_scores)

    print(f"\n{'='*40}")
    print(f"  BASELINE AUC: {auc:.4f} ({auc*100:.2f}%)")
    print(f"  Threshold:    {THRESHOLD_METHOD}")
    print(f"{'='*40}")

    # ── Save ──────────────────────────────────────────────────────
    summary = {
        "model":              "Baseline (Simple Text Labels) - Fixed",
        "threshold_method":   THRESHOLD_METHOD,
        "alpha":              ALPHA,
        "auc":                float(auc),
        "num_videos":         len(all_labels),
        "normal_videos":      int(np.sum(all_labels == 0)),
        "anomaly_videos":     int(np.sum(all_labels == 1)),
        "normal_mean_score":  float(np.mean(all_scores[all_labels==0])),
        "anomaly_mean_score": float(np.mean(all_scores[all_labels==1])),
        "text_labels_used":   SIMPLE_TEXT_LABELS
    }

    with open(os.path.join(RESULTS_DIR, "baseline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "baseline_per_video.json"), "w") as f:
        json.dump(results_per_video, f, indent=2)

    # ── ROC Plot ──────────────────────────────────────────────────
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2,
             label=f"Baseline ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "gray", linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Baseline: ROC Curve\n(Simple Text Labels - Fixed)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "baseline_roc.png"), dpi=150)
    plt.close()

    # ── Score Bar Chart ───────────────────────────────────────────
    video_names  = list(results_per_video.keys())
    video_scores = [results_per_video[v]["video_score"] for v in video_names]
    video_labels = [results_per_video[v]["true_label"]  for v in video_names]
    colors       = ["red" if l == 1 else "green" for l in video_labels]

    plt.figure(figsize=(14, 5))
    plt.bar(range(len(video_names)), video_scores, color=colors, alpha=0.7)
    plt.xlabel("Video Index")
    plt.ylabel("Anomaly Score (max psi)")
    plt.title("Baseline: Anomaly Scores per Video\n"
              "(Red=Anomaly, Green=Normal) — Fixed Version")
    plt.axhline(y=np.mean(video_scores), color="black",
                linestyle="--", label="Mean score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "baseline_scores.png"), dpi=150)
    plt.close()

    print(f"\n[SAVED] Plots → {RESULTS_DIR}/")
    print(f"[DONE]  Baseline AUC = {auc*100:.2f}%")
    return auc


if __name__ == "__main__":
    main()
