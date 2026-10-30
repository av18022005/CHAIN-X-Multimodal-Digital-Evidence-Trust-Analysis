# CHAIN-X: A Unified Multimodal Framework for Chain-of-Custody Evidence Trust Analysis

Developed as part of the course on "Forensic Analytics" at Vellore Institute of Technology, Chennai.
by "Arista Vania", "Tannya Pasricha"

Subject Code: CSE4074
Semester: Fall 2026-27
Professor: Dr. S. Johanan Joysingh

---

> **Project explanation** [`docs/PROJECT_EXPLAINER.md`](docs/PROJECT_EXPLAINER.md)
> **Research paper:** `docs/CHAIN-X_IEEE_Research_Paper.docx`

## What is CHAIN-X?

Digital evidence today is a package: images, video, text reports, file metadata and multi-party custody logs. Existing forensic tools check each of these in isolation. CHAIN-X asks a different question: **how trustworthy is all of the evidence together, across the entire chain of custody?**

It analyzes images, videos, reports, metadata and custody records together, detects cross-modal inconsistencies (for example a report stating one timestamp while the EXIF data records another) and custody-graph anomalies (unauthorized nodes, timing gaps), and produces two scores with an explanation of what drove them:

- **Evidence Health Index (EHI)**: weighted integrity score minus a cross-modal contradiction penalty, normalized to [0, 1].
- **Custody Risk Score (CRS)**: graph-aware risk probability, expressed as a percentage.

Explanations use SHAP attribution and are shown on a Streamlit dashboard.

![CHAIN-X architecture](docs/figures/fig_arch.png)

## Project status (honest summary)

| Component | Status |
|---|---|
| Image localization study (ELA + SRM, ResNet-18 U-Net, CASIA v2.0) | Implemented and evaluated (see `notebooks/`) |
| Evidence-survival experiments (JPEG, resize, blur, repeated transforms) | Implemented and evaluated |
| Image features for fusion (ELA, frozen EfficientNet-B0, RF/XGBoost) | Code in `src/image/` (update this row with your actual run results) |
| Metadata, NLP, graph engines, cross-modal check | Update with your actual status |
| Fusion (attention-gated MLP), EHI, CRS, SHAP, dashboard | Update with your actual status |
| Synthetic custody benchmark (2,000 cases) | Planned |
| Physical screen recapture, external dataset (IMD2020) | Not run; no results claimed |

## Key results so far (image engine, CASIA v2.0, 500-image source-disjoint test set)

| Metric | Value |
|---|---|
| ROC-AUC / accuracy | 0.9315 / 84.6% |
| Recall / precision / F1 (manipulated) | 0.892 / 0.817 / 0.853 |
| Mean IoU / **median IoU** | 0.337 / **0.023** |
| Mean Dice, original vs. after JPEG Q80 | 0.318 vs. 0.0004 |

Caveats stated in the paper: a trivial "TIFF means manipulated" rule scores 83.9% on the same data, localization is weak for most images, performance collapses under recompression, and the three-class classifier did not train successfully (macro-F1 0.249).

![Evidence survival](docs/figures/fig_survival.png)

## Repository structure

```
CHAIN-X-Multimodal-Digital-Evidence-Trust-Analysis/
├── README.md                  <- this file
├── docs/
│   ├── PROJECT_EXPLAINER.md   <- project explanation (parallel to the PPT)
│   ├── CHAIN-X_IEEE_Research_Paper.docx
│   └── figures/
├── notebooks/                 <- Kaggle notebooks (image localization + survival)
├── configs/  data/  experiments/  features/  models/  scripts/  src/
├── app_tier1.py  app_tier2.py <- dashboards
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Datasets are not stored in the repository. See `scripts/DOWNLOAD_INSTRUCTIONS.md` for CASIA v2.0 and COVERAGE.

## How to reproduce

1. Get CASIA v2.0 onto disk (`scripts/DOWNLOAD_INSTRUCTIONS.md`), then run `scripts/build_master_csv.py` to create `data/master_index.csv`.
2. Image feature pipeline: `src/image/feature_extractor.py` (GPU, Kaggle/Colab), `src/image/ela.py` (CPU), `src/image/classifier.py`.
3. Localization and evidence-survival study: open the notebook in `notebooks/` on Kaggle with a GPU (Tesla T4 was used; about 60 s per epoch). Enable internet if you want the ImageNet-pretrained encoder; the main reported run had none and trained from random initialization.
4. Dashboards: `streamlit run app_tier1.py` (and `app_tier2.py`).

## Technology stack

Python 3.10+, PyTorch, OpenCV, timm, Pillow, Hugging Face Transformers (DeBERTa-v3 / Legal-BERT), spaCy, exifread, hachoir, XGBoost / scikit-learn, NetworkX, SHAP, SQLite, Streamlit. Runs on a 16 GB workstation (RTX 3050/4050) or Google Colab / Kaggle.

## Limitations

CHAIN-X is forensic decision support. A model output does not by itself establish authenticity or forgery, identify a responsible person, or determine legal admissibility. Hash verification shows a file is unchanged since it was recorded, not that its content is genuine.

## Team

Arista Vania (23MIA1036), Tannya Pasricha (23MIA1130)
