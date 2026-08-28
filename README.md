# CHAIN-X

Multimodal digital-evidence integrity analysis: image tampering + metadata
anomalies + forensic-report (NLP) consistency + provenance/custody graph
analysis, fused into an Evidence Health Index (EHI) and Custody Risk Score
(CRS), explained with SHAP.

## Guiding rules (do not violate these while building)

1. **No training of big models from scratch.** EfficientNet-B0 and
   DeBERTa-v3-small are used **frozen**, as feature extractors only.
2. **GPU only for feature extraction** (image CNN + text embeddings).
   Everything else (ELA, metadata, graph, XGBoost, SHAP, dashboard) runs on
   CPU on your laptop.
3. **Checkpoint everything.** Feature extraction writes `.npy` files in
   small batches (100 samples) so a dropped Kaggle session never loses more
   than a batch.
4. **Real datasets first.** CASIA2 (train/val/internal-test), COVERAGE
   (external test only, never touched during training), NIST CFReDS /
   CASE-Corpora (provenance), Dundee digital-forensics corpus (NLP).
   Synthetic data is used ONLY as small controlled perturbations
   (~100-200 cases) to stress-test contradiction detection — never as the
   training foundation.
5. **One phase at a time.** Don't move to the next phase until the current
   one runs end-to-end on a small sample and produces a checked-in result.
6. **Master CSV is the spine.** Every module reads/writes rows keyed by
   `case_id` in `data/master_index.csv`. Modalities can be missing per row
   — the fusion model must tolerate that (see Phase 7).

## Phase map

| Phase | What | Status |
|---|---|---|
| 0 | Repo structure, env, config | ✅ this delivery |
| 1 | Dataset acquisition + master CSV | ✅ this delivery (scripts + instructions) |
| 2 | Image analyzer (ELA + EfficientNet + RF/XGB) | ✅ this delivery (code, needs your data) |
| 3 | Metadata analyzer | ⏭ next |
| 4 | NLP analyzer | ⏭ next |
| 5 | Cross-modal consistency | ⏭ next |
| 6 | Provenance graph | ⏭ next |
| 7 | Fusion (EHI/CRS) | ⏭ next |
| 8 | SHAP explainability | ⏭ next |
| 9 | Streamlit dashboard | ⏭ next |
| 10 | Ablation + external validation | ⏭ next |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Order of operations for Phase 1-2 (do this now)

1. Read `scripts/DOWNLOAD_INSTRUCTIONS.md` and get CASIA2 + COVERAGE onto disk.
2. Run `scripts/build_master_csv.py` to generate `data/master_index.csv`.
3. Run `src/image/feature_extractor.py` (GPU on Kaggle/Colab) to extract
   EfficientNet embeddings in checkpointed batches.
4. Run `src/image/ela.py` locally (CPU) to compute ELA/noise features.
5. Run `src/image/classifier.py` to train the RF/XGBoost tampering
   classifier and print accuracy/F1/ROC-AUC.

Come back after step 5 works and we'll do Phase 3 (metadata).
