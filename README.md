# TFPLG-Video-Anomaly-Detection

## Training-Free VLM-Based Pseudo Label Generation for Video Anomaly Detection
### Investigating the Impact of Text Description Quality Using UCA Annotations

> **MSc Computing / Data Science Project**  
> University of Roehampton, 2025  
> Student: Fawaz Fazi

---

## 📋 Project Overview

This project implements and evaluates a simplified version of the **Training-Free Pseudo Label Generation (TFPLG)** framework proposed by Abdalla and Javed (2025) for Weakly Supervised Video Anomaly Detection (WSVAD).

The central research question is:

> *Does replacing simple one-word text labels with rich, contextual sentence descriptions from the UCF-Crime Annotation (UCA) dataset improve pseudo label quality and anomaly detection performance?*

### Key Findings
- **Baseline AUC**: 68.25% (using simple text labels like `"fighting"`, `"abuse"`)
- **Improved AUC**: 63.49% (using category-averaged UCA sentence descriptions)
- The improvement did not yield higher AUC on this 16-video dataset, but frame-level score gap increased marginally (0.0067 → 0.0072)
- Primary bottleneck identified as **dataset scale**, not text description quality

---

## 🗂️ Repository Structure

```
TFPLG-Video-Anomaly-Detection/
├── 00_setup.py               # Install packages and create folder structure
├── 01_extract_features.py    # Extract CLIP ViT-B/16 features from videos
├── 02_baseline.py            # Baseline model — simple text labels
├── 03_improved.py            # Improved model — rich UCA descriptions
├── 04_compare.py             # Compare both models, generate plots
├── 05_threshold_search.py    # Test all threshold strategies
└── README.md
```

---

## 📦 Datasets

### 1. UCF-Crime Mini Dataset (Visual Input)
- **Description**: A subset of the UCF-Crime dataset containing CCTV surveillance videos across 6 anomaly categories
- **Categories**: Abuse, Arrest, Assault, Burglary, Fighting, Normal
- **Size**: ~959 MB (42 videos, 7 per category)
- **Download**: [Kaggle — UCF Crime Mini Dataset](https://www.kaggle.com/datasets/shashiprakash204/ucfcrimeminidataset)
- **Used for**: Extracting visual CLIP features (Script 01)

### 2. UCA (UCF-Crime Annotation) Dataset (Text Input)
- **Description**: Rich sentence-level annotations for 1,854 UCF-Crime videos with 23,542 human-written sentences averaging 20 words each
- **Size**: ~11 MB (JSON files only — videos not required)
- **Download**: [Kaggle — UCA UCF Crime Annotation Dataset](https://www.kaggle.com/datasets/vigneshwar472/ucaucf-crime-annotation-dataset)
- **Files needed**:
  - `UCFCrime_Train.json`
  - `UCFCrime_Test.json`
  - `UCFCrime_Val.json`
- **Used for**: Rich text descriptions in the improved model (Script 03)

### 3. CLIP Model (Pre-trained)
- **Model**: ViT-B/16 from OpenAI CLIP
- **Downloaded automatically** when running the scripts
- **Source**: [OpenAI CLIP GitHub](https://github.com/openai/CLIP)

---

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Windows / Mac / Linux
- CPU only (no GPU required)
- ~2 GB free disk space (for mini dataset + features)

### Step 1 — Clone the Repository
```bash
git clone https://github.com/fawazfazi/TFPLG-Video-Anomaly-Detection.git
cd TFPLG-Video-Anomaly-Detection
```

### Step 2 — Run Setup Script
```bash
python 00_setup.py
```
This installs all required packages and creates the folder structure automatically.

### Step 3 — Place Dataset Files

After downloading the datasets, organise them as follows:

```
TFPLG-Video-Anomaly-Detection/
├── data/
│   ├── videos/
│   │   ├── Abuse/          ← .mp4 files from mini dataset
│   │   ├── Arrest/
│   │   ├── Assault/
│   │   ├── Burglary/
│   │   ├── Fighting/
│   │   └── normal/
│   └── annotations/
│       ├── UCFCrime_Train.json   ← from UCA dataset
│       ├── UCFCrime_Test.json
│       └── UCFCrime_Val.json
```

---

## 🚀 Running the Experiments

Run the scripts in order:

### Script 1 — Extract CLIP Features
```bash
python 01_extract_features.py
```
- Extracts frames from all videos (1 frame per 10 frames)
- Computes CLIP ViT-B/16 visual features
- Saves features as `.npy` files in `features/` folder
- ⏱️ Takes ~20-30 minutes on CPU

### Script 2 — Baseline Model
```bash
python 02_baseline.py
```
- Uses simple one-word text labels (`"fighting"`, `"abuse"`, etc.)
- Computes pseudo labels using CLIP similarity matching
- Evaluates with AUC score and generates ROC curve
- Results saved to `results/baseline/`

### Script 3 — Improved Model
```bash
python 03_improved.py
```
- Uses category-averaged rich UCA sentence descriptions
- Same pipeline as baseline — only text input differs
- Results saved to `results/improved/`

### Script 4 — Compare Results
```bash
python 04_compare.py
```
- Generates side-by-side comparison of baseline vs improved
- Outputs: AUC bar chart, ROC curves, per-video scores, category scores
- Results saved to `results/comparison/`

### Script 5 — Threshold Sensitivity Analysis
```bash
python 05_threshold_search.py
```
- Tests all 6 combinations (2 models × 3 threshold strategies)
- Threshold strategies: Fixed (0.81), Mean, 80th Percentile
- Results saved to `results/threshold_search/`

---

## 📊 Results Summary

| Model | Threshold | AUC | Score Gap |
|-------|-----------|-----|-----------|
| Baseline (Simple Labels) | Fixed / Mean / P80 | **68.25%** | +0.0074 |
| Improved (Rich UCA) | Fixed / Mean / P80 | 63.49% | +0.0072 |

> **Note**: All three threshold methods produce identical AUC scores within each model on this 16-video dataset, confirming that threshold selection does not affect video-level ranking when the dataset is small.

### Result Plots

After running all scripts, the following plots are saved in `results/`:

```
results/
├── baseline/
│   ├── baseline_roc.png          # ROC curve
│   └── baseline_scores.png       # Per-video anomaly scores
├── improved/
│   ├── improved_roc.png
│   └── improved_scores.png
├── comparison/
│   ├── auc_comparison.png        # Side-by-side AUC bar chart
│   ├── roc_comparison.png        # Overlaid ROC curves
│   ├── per_video_scores.png      # Per-video comparison
│   └── category_scores.png       # Category-level scores
└── threshold_search/
    └── all_combinations_auc.png  # All 6 combinations
```

---

## 🔬 Methodology

### How Pseudo Labels Are Generated

The core algorithm follows the TFPLG framework (Abdalla & Javed, 2025):

**1. Feature Extraction**
```
Video → Extract frames (1 per 10) → CLIP ViT-B/16 → 512-dim feature vectors
```

**2. Similarity Computation**
```
S_an = X · T_normal^T       (frame-to-normal similarity)
S_aa = X · T_anomaly^T      (frame-to-anomaly similarity)
```

**3. Normalisation**
```
S̃_an = S_an / (||S_an|| + ε)
S̃_aa = S_aa / (||S_aa|| + ε)
```

**4. Fusion (Equation 19 from paper)**
```
ψ = α · S̃_aa + (1 - α) · (1 - S̃_an)     where α = 0.2
```

**5. Thresholding**
```
γ_j = 1 if ψ_j ≥ θ, else 0
```

**6. Video-level score**
```
video_score = max(ψ)
```

### Key Difference Between Models

| | Baseline | Improved |
|---|---|---|
| Text input | `"fighting"` (1 word) | 30 averaged UCA sentences |
| Source | Hardcoded labels | UCFCrime_Train/Test/Val.json |
| Text encoding | Single CLIP embedding | Average of 30 CLIP embeddings |
| Everything else | Identical | Identical |

---

## 📚 References

1. M. Abdalla and S. Javed, *"Training-Free VLM-Based Pseudo Label Generation for Video Anomaly Detection"*, IEEE Access, vol. 13, pp. 92155–92167, 2025.

2. W. Sultani, C. Chen, and M. Shah, *"Real-world Anomaly Detection in Surveillance Videos"*, CVPR, 2018.

3. T. Yuan et al., *"Towards Surveillance Video-and-Language Understanding: New Dataset, Baselines, and Challenges"*, CVPR, 2024.

4. A. Radford et al., *"Learning Transferable Visual Models from Natural Language Supervision"*, ICML, 2021.

5. P. Wu et al., *"VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection"*, AAAI, 2024.

---

## ⚙️ Requirements

All packages are installed automatically by `00_setup.py`. Manual installation:

```bash
pip install torch torchvision
pip install opencv-python
pip install numpy scikit-learn matplotlib tqdm Pillow
pip install git+https://github.com/openai/CLIP.git
```

---

## 📄 Licence

This project is for academic research purposes only. The UCF-Crime and UCA datasets are used under academic fair use. The CLIP model is used under OpenAI's open-source research licence.

---

## 🔗 Links

- 📄 **Paper**: [Abdalla & Javed (2025) — IEEE Access](https://doi.org/10.1109/ACCESS.2025.3573594)
- 📊 **Mini Dataset**: [Kaggle — UCF Crime Mini Dataset](https://www.kaggle.com/datasets/shashiprakash204/ucfcrimeminidataset)
- 📝 **UCA Annotations**: [Kaggle — UCA Dataset](https://www.kaggle.com/datasets/vigneshwar472/ucaucf-crime-annotation-dataset)
- 🤖 **CLIP Model**: [OpenAI CLIP](https://github.com/openai/CLIP)
- 📦 **Original TFPLG Code**: [MoshiraAbdalla/TFPLG_VAD](https://github.com/MoshiraAbdalla/TFPLG_VAD)
