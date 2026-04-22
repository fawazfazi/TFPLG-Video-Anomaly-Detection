"""
Script: Try All Threshold Strategies
=====================================
Tests all combinations of:
- Models: baseline (simple labels) vs improved (rich UCA)
- Thresholds: fixed (0.81), mean, percentile80

Prints a full comparison table and saves the best results.
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
FEATURE_DIR    = "features"
ANNOTATION_DIR = "data/annotations"
RESULTS_DIR    = "results/threshold_search"
DEVICE         = "cpu"
ALPHA          = 0.2
MAX_SENTENCES  = 30
# ──────────────────────────────────────────────────────────────────────────────

ANOMALY_CATEGORIES = ["abuse", "arrest", "assault", "burglary", "fighting"]
NORMAL_CATEGORY    = "normal"
THRESHOLDS         = ["fixed", "mean", "percentile80"]

SIMPLE_LABELS = {
    "abuse": "abuse", "arrest": "arrest", "assault": "assault",
    "burglary": "burglary", "fighting": "fighting", "normal": "normal activity"
}

FALLBACK_RICH = {
    "abuse":    "a person is being physically abused or mistreated violently",
    "arrest":   "police officers are arresting and detaining a person",
    "assault":  "a person is physically attacking and assaulting another person",
    "burglary": "a person is breaking into a building to steal property",
    "fighting": "two or more people are physically fighting and hitting each other",
    "normal":   "people are walking and going about their daily activities normally"
}

CATEGORY_KEYWORDS = {
    "abuse": ["abuse"], "arrest": ["arrest"], "assault": ["assault"],
    "burglary": ["burglary"], "fighting": ["fighting"], "normal": ["normal"]
}


def load_clip():
    model, _ = clip.load("ViT-B/16", device=DEVICE)
    model.eval()
    return model


def encode_text(model, text):
    with torch.no_grad():
        tokens = clip.tokenize([text[:200]]).to(DEVICE)
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()


def encode_sentences_avg(model, sentences, max_n=30):
    if len(sentences) > max_n:
        idx = np.linspace(0, len(sentences)-1, max_n, dtype=int)
        sentences = [sentences[i] for i in idx]
    feats = []
    with torch.no_grad():
        for s in sentences:
            try:
                tokens = clip.tokenize([s[:200]]).to(DEVICE)
                f = model.encode_text(tokens)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu().numpy())
            except:
                continue
    if not feats:
        return None
    avg = np.mean(np.vstack(feats), axis=0, keepdims=True)
    return avg / (np.linalg.norm(avg) + 1e-8)


def load_uca():
    ann = {}
    for split in ["Train", "Test", "Val"]:
        p = os.path.join(ANNOTATION_DIR, f"UCFCrime_{split}.json")
        if os.path.exists(p):
            with open(p) as f:
                ann.update(json.load(f))
    return ann


def build_cat_feats_rich(model, ann):
    """Build category-averaged UCA features."""
    cat_sents = defaultdict(list)
    for vname, data in ann.items():
        vl = vname.lower()
        for cat, kws in CATEGORY_KEYWORDS.items():
            if any(k in vl for k in kws):
                cat_sents[cat].extend(data.get("sentences", []))
                break

    feats = {}
    for cat in ANOMALY_CATEGORIES + [NORMAL_CATEGORY]:
        sents = cat_sents.get(cat, [])
        if sents:
            f = encode_sentences_avg(model, sents, MAX_SENTENCES)
            feats[cat] = f if f is not None else encode_text(model, cat)
        else:
            feats[cat] = encode_text(model, FALLBACK_RICH.get(cat, cat))
    return feats


def build_cat_feats_simple(model):
    """Build simple one-word text features."""
    return {k: encode_text(model, v) for k, v in SIMPLE_LABELS.items()}


def sim(visual, text):
    return (visual @ text.T).squeeze()


def compute_psi(visual, anomaly_feat, normal_feat, alpha=0.2):
    S_an = sim(visual, normal_feat)
    S_aa = sim(visual, anomaly_feat)
    eps  = 1e-8
    S_an_n = S_an / (np.linalg.norm(S_an) + eps)
    S_aa_n = S_aa / (np.linalg.norm(S_aa) + eps)
    return alpha * S_aa_n + (1 - alpha) * (1 - S_an_n)


def get_threshold(psi, method):
    if method == "fixed":       return 0.81
    elif method == "mean":      return np.mean(psi)
    elif method == "percentile80": return np.percentile(psi, 80)
    return np.mean(psi)


def run_experiment(model, cat_feats, metadata, threshold_method, label):
    """Run one full experiment and return AUC + per-video results."""
    normal_feat = cat_feats[NORMAL_CATEGORY]
    scores, labels, per_video = [], [], {}

    for vname, info in metadata.items():
        cat        = info["category"].lower()
        true_label = info["label"]
        feat_path  = info["feature_path"]

        if not os.path.exists(feat_path):
            continue

        visual = np.load(feat_path).astype(np.float32)

        # Get anomaly text feature for this category
        if cat in cat_feats:
            anomaly_feat = cat_feats[cat]
        else:
            # Use most similar available category
            anomaly_feat = cat_feats.get(
                ANOMALY_CATEGORIES[0], list(cat_feats.values())[0]
            )

        psi   = compute_psi(visual, anomaly_feat, normal_feat, ALPHA)
        theta = get_threshold(psi, threshold_method)
        score = float(np.max(psi))

        scores.append(score)
        labels.append(true_label)
        per_video[vname] = {
            "category": cat, "true_label": true_label,
            "video_score": score, "mean_score": float(np.mean(psi)),
            "threshold": float(theta),
            "pseudo_labels": (psi >= theta).astype(int).tolist()
        }

    scores = np.array(scores)
    labels = np.array(labels)

    if len(np.unique(labels)) < 2:
        return None, None, None

    auc          = roc_auc_score(labels, scores)
    score_gap    = float(np.mean(scores[labels==1]) - np.mean(scores[labels==0]))
    fpr, tpr, _  = roc_curve(labels, scores)

    return auc, score_gap, (fpr, tpr, scores, labels, per_video)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 65)
    print("THRESHOLD SEARCH: All Combinations")
    print("=" * 65)

    model    = load_clip()
    ann      = load_uca()
    metadata = json.load(open(os.path.join(FEATURE_DIR, "video_metadata.json")))

    print("\n[INFO] Building text features...")
    simple_feats = build_cat_feats_simple(model)
    rich_feats   = build_cat_feats_rich(model, ann)
    print("  ✓ Simple labels ready")
    print("  ✓ Rich UCA labels ready")

    # ── Run all 6 combinations ─────────────────────────────────────────────
    results = {}
    print(f"\n{'─'*65}")
    print(f"{'Model':<35} {'Threshold':<15} {'AUC':>8} {'Gap':>10}")
    print(f"{'─'*65}")

    experiments = [
        ("Baseline (Simple Labels)", simple_feats),
        ("Improved (Rich UCA)",      rich_feats),
    ]

    for model_name, feats in experiments:
        for thr in THRESHOLDS:
            auc, gap, extras = run_experiment(
                model, feats, metadata, thr, model_name
            )
            if auc is None:
                continue

            key = f"{model_name}_{thr}"
            results[key] = {
                "model": model_name, "threshold": thr,
                "auc": auc, "score_gap": gap,
                "extras": extras
            }
            print(f"  {model_name:<33} {thr:<15} {auc*100:>7.2f}%  {gap:>+.4f}")

    print(f"{'─'*65}")

    # ── Find best result ───────────────────────────────────────────────────
    best_key = max(results, key=lambda k: results[k]["auc"])
    best     = results[best_key]
    print(f"\n{'★'*65}")
    print(f"  BEST: {best['model']}")
    print(f"        Threshold: {best['threshold']}")
    print(f"        AUC:       {best['auc']*100:.2f}%")
    print(f"        Score Gap: {best['score_gap']:+.4f}")
    print(f"{'★'*65}")

    # ── Find best baseline and best improved ──────────────────────────────
    baseline_results = {k: v for k, v in results.items() if "Baseline" in k}
    improved_results = {k: v for k, v in results.items() if "Improved" in k}

    best_baseline_key = max(baseline_results, key=lambda k: baseline_results[k]["auc"])
    best_improved_key = max(improved_results, key=lambda k: improved_results[k]["auc"])

    best_baseline = baseline_results[best_baseline_key]
    best_improved = improved_results[best_improved_key]

    print(f"\n[BEST BASELINE] {best_baseline['threshold']} → AUC = {best_baseline['auc']*100:.2f}%")
    print(f"[BEST IMPROVED] {best_improved['threshold']} → AUC = {best_improved['auc']*100:.2f}%")
    diff = best_improved['auc'] - best_baseline['auc']
    arrow = "↑ IMPROVEMENT" if diff > 0 else "↓ No improvement"
    print(f"[DIFFERENCE]    {diff*100:+.2f}% {arrow}")

    # ── Save best results ──────────────────────────────────────────────────
    fpr_b, tpr_b, _, labels_b, pvid_b = best_baseline["extras"]
    fpr_i, tpr_i, _, labels_i, pvid_i = best_improved["extras"]

    # ROC comparison plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_b, tpr_b, color="blue", lw=2,
             label=f"Best Baseline [{best_baseline['threshold']}] (AUC={best_baseline['auc']:.4f})")
    plt.plot(fpr_i, tpr_i, color="red",  lw=2,
             label=f"Best Improved [{best_improved['threshold']}] (AUC={best_improved['auc']:.4f})")
    plt.plot([0,1],[0,1],"gray",linestyle="--",label="Random")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Best Models: ROC Curve Comparison")
    plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "best_roc_comparison.png"), dpi=150)
    plt.close()

    # AUC bar chart — all 6 combinations
    labels_plot = [f"{r['model'].split('(')[0].strip()}\n[{r['threshold']}]"
                   for r in results.values()]
    aucs_plot   = [r['auc']*100 for r in results.values()]
    colors_plot = ["steelblue" if "Baseline" in r['model'] else "tomato"
                   for r in results.values()]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(labels_plot)), aucs_plot,
                   color=colors_plot, alpha=0.8, edgecolor="black")
    for bar, val in zip(bars, aucs_plot):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    plt.xticks(range(len(labels_plot)), labels_plot, fontsize=9)
    plt.ylabel("AUC Score (%)"); plt.ylim(0, 105)
    plt.title("AUC Scores: All Threshold Strategies\n(Blue=Baseline, Red=Improved)")
    plt.grid(axis="y", alpha=0.3)
    import matplotlib.patches as mpatches
    plt.legend(handles=[
        mpatches.Patch(color="steelblue", label="Baseline (Simple Labels)"),
        mpatches.Patch(color="tomato",    label="Improved (Rich UCA)")
    ])
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "all_combinations_auc.png"), dpi=150)
    plt.close()

    # Final summary JSON
    summary = {
        "all_results": {
            k: {"model": v["model"], "threshold": v["threshold"],
                "auc": v["auc"], "score_gap": v["score_gap"]}
            for k, v in results.items()
        },
        "best_baseline": {
            "model": best_baseline["model"],
            "threshold": best_baseline["threshold"],
            "auc": best_baseline["auc"]
        },
        "best_improved": {
            "model": best_improved["model"],
            "threshold": best_improved["threshold"],
            "auc": best_improved["auc"]
        },
        "improvement": diff,
        "improvement_pct": f"{diff*100:+.2f}%"
    }

    with open(os.path.join(RESULTS_DIR, "threshold_search_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save best per-video results
    with open(os.path.join(RESULTS_DIR, "best_baseline_per_video.json"), "w") as f:
        json.dump(pvid_b, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "best_improved_per_video.json"), "w") as f:
        json.dump(pvid_i, f, indent=2)

    print(f"\n[SAVED] Plots and results → {RESULTS_DIR}/")
    print(f"[DONE]  Run complete!")


if __name__ == "__main__":
    main()
