"""
Script 4: Compare Baseline vs Improved — Generate Report Figures
=================================================================
- Loads results from both baseline and improved models
- Creates comparison plots for your report
- Prints final summary table
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = "results"


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("COMPARISON: Baseline vs Improved")
    print("=" * 60)

    # ── Load summaries ────────────────────────────────────────────
    baseline_summary = load_results("results/baseline/baseline_summary.json")
    improved_summary = load_results("results/improved/improved_summary.json")

    baseline_per_vid = load_results("results/baseline/baseline_per_video.json")
    improved_per_vid = load_results("results/improved/improved_per_video.json")

    base_auc = baseline_summary["auc"]
    impr_auc = improved_summary["auc"]
    diff     = impr_auc - base_auc

    # ── Print Summary Table ───────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"{'Model':<35} {'AUC':>10}")
    print(f"{'─'*50}")
    print(f"{'Baseline (Simple labels)':<35} {base_auc*100:>9.2f}%")
    print(f"{'Improved (Rich UCA descriptions)':<35} {impr_auc*100:>9.2f}%")
    print(f"{'─'*50}")
    arrow = "↑" if diff > 0 else "↓"
    print(f"{'Difference':<35} {arrow}{abs(diff)*100:>8.2f}%")
    print(f"{'─'*50}\n")

    os.makedirs(os.path.join(RESULTS_DIR, "comparison"), exist_ok=True)

    # ── Plot 1: AUC Bar Chart ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["Baseline\n(Simple Labels)", "Improved\n(Rich UCA Descriptions)"],
        [base_auc * 100, impr_auc * 100],
        color=["steelblue", "tomato"],
        width=0.5,
        edgecolor="black"
    )
    for bar, val in zip(bars, [base_auc, impr_auc]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val*100:.2f}%",
                ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_ylim(0, 105)
    ax.set_ylabel("AUC Score (%)", fontsize=12)
    ax.set_title("Baseline vs Improved Model\nAUC Comparison", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/comparison/auc_comparison.png", dpi=150)
    plt.close()
    print("[SAVED] AUC comparison → results/comparison/auc_comparison.png")

    # ── Plot 2: ROC Curves Together ───────────────────────────────
    # Recompute ROC curves from per-video scores
    from sklearn.metrics import roc_curve, roc_auc_score

    def get_scores_labels(per_vid):
        scores = [v["video_score"] for v in per_vid.values()]
        labels = [v["true_label"]  for v in per_vid.values()]
        return np.array(scores), np.array(labels)

    b_scores, b_labels = get_scores_labels(baseline_per_vid)
    i_scores, i_labels = get_scores_labels(improved_per_vid)

    b_fpr, b_tpr, _ = roc_curve(b_labels, b_scores)
    i_fpr, i_tpr, _ = roc_curve(i_labels, i_scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(b_fpr, b_tpr, color="steelblue", lw=2,
            label=f"Baseline (AUC = {base_auc:.4f})")
    ax.plot(i_fpr, i_tpr, color="tomato",    lw=2,
            label=f"Improved (AUC = {impr_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve Comparison\nBaseline vs Improved", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/comparison/roc_comparison.png", dpi=150)
    plt.close()
    print("[SAVED] ROC comparison → results/comparison/roc_comparison.png")

    # ── Plot 3: Per-Video Score Comparison ────────────────────────
    common_videos = [v for v in baseline_per_vid if v in improved_per_vid]
    b_vid_scores  = [baseline_per_vid[v]["video_score"] for v in common_videos]
    i_vid_scores  = [improved_per_vid[v]["video_score"] for v in common_videos]
    vid_labels    = [baseline_per_vid[v]["true_label"]  for v in common_videos]

    x = np.arange(len(common_videos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 5))
    bars1 = ax.bar(x - width/2, b_vid_scores, width,
                   label="Baseline", color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + width/2, i_vid_scores, width,
                   label="Improved", color="tomato",    alpha=0.8)

    # Add background color for anomaly vs normal
    for i, lbl in enumerate(vid_labels):
        color = "lightyellow" if lbl == 1 else "lightcyan"
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.2, color=color, zorder=0)

    ax.set_xlabel("Video", fontsize=11)
    ax.set_ylabel("Anomaly Score", fontsize=11)
    ax.set_title("Per-Video Anomaly Scores: Baseline vs Improved\n"
                 "(Yellow background = Anomaly video, Cyan = Normal video)",
                 fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [v[:15] for v in common_videos], rotation=45, ha="right", fontsize=7
    )
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/comparison/per_video_scores.png", dpi=150)
    plt.close()
    print("[SAVED] Per-video scores → results/comparison/per_video_scores.png")

    # ── Plot 4: Category-wise AUC ─────────────────────────────────
    categories = list(set(v["category"] for v in baseline_per_vid.values()
                          if v["category"] != "normal"))

    cat_base_scores, cat_impr_scores = [], []
    for cat in categories:
        b_s = [baseline_per_vid[v]["video_score"]
               for v in baseline_per_vid
               if baseline_per_vid[v]["category"] == cat]
        i_s = [improved_per_vid[v]["video_score"]
               for v in improved_per_vid
               if improved_per_vid[v]["category"] == cat]
        cat_base_scores.append(np.mean(b_s) if b_s else 0)
        cat_impr_scores.append(np.mean(i_s) if i_s else 0)

    x2 = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x2 - width/2, cat_base_scores, width,
           label="Baseline", color="steelblue", alpha=0.8)
    ax.bar(x2 + width/2, cat_impr_scores, width,
           label="Improved", color="tomato",    alpha=0.8)
    ax.set_xlabel("Anomaly Category", fontsize=12)
    ax.set_ylabel("Mean Anomaly Score", fontsize=12)
    ax.set_title("Category-wise Mean Anomaly Score\nBaseline vs Improved",
                 fontsize=13)
    ax.set_xticks(x2)
    ax.set_xticklabels([c.capitalize() for c in categories], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/comparison/category_scores.png", dpi=150)
    plt.close()
    print("[SAVED] Category scores → results/comparison/category_scores.png")

    # ── Save Final Summary ────────────────────────────────────────
    final = {
        "baseline_auc":   base_auc,
        "improved_auc":   impr_auc,
        "improvement":    diff,
        "improvement_pct": f"{diff*100:+.2f}%"
    }
    with open("results/comparison/final_summary.json", "w") as f:
        json.dump(final, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Baseline AUC : {base_auc*100:.2f}%")
    print(f"  Improved AUC : {impr_auc*100:.2f}%")
    print(f"  Improvement  : {diff*100:+.2f}%")
    print(f"{'='*50}")
    print("\n[DONE] All comparison plots saved to results/comparison/")
    print("[DONE] Use these figures directly in your report!")


if __name__ == "__main__":
    main()
