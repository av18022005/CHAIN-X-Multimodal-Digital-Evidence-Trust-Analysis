"""
TIER 1 STREAMLIT DASHBOARD: Evidence Health Index (Research Paper Version)
Real data only — image + metadata → EHI with SHAP explanation.

Usage:
    streamlit run app_tier1.py

Dependencies:
    pip install streamlit shap pandas numpy pillow opencv-python-headless scikit-learn
"""

import os
import pickle
import json
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import shap

# Add repo root to path so we can import from src/
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from src.image.ela import extract_all_features as extract_ela_features
from src.metadata.extractor import extract_all_metadata
from src.metadata.anomaly import analyze as analyze_metadata

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

st.set_page_config(page_title="Evidence Health Index - Research Demo", layout="wide")

# Model paths
MODEL_RF_PATH = "models/rf_final_locked_in.pkl"
MODEL_FUSION_PATH = "models/fusion_v1.pkl"
SHAP_BG_PATH = "/kaggle/working/shap_background_combined.pkl"
SHAP_COLS_PATH = "/kaggle/working/shap_feature_columns.pkl"
NORM_STATS_PATH = "/kaggle/working/normalization_stats.json"
DASHBOARD_METRICS_PATH = "/kaggle/working/features/dashboard_metrics_tier1.csv"
# NOTE: if you run this outside the Kaggle notebook environment, copy these
# four files next to the repo and update the paths above accordingly.

# Model performance metrics (from Phase 2 & 7)
MODEL_METRICS = {
    "rf_classifier": {
        "name": "Image Classifier (Random Forest)",
        "accuracy": 0.789,
        "precision": 0.795,  # macro avg
        "recall": 0.79,
        "f1": 0.806,
        "auc": 0.861,
        "split": "internal_test (CASIA)",
        "n_samples": 465,
        "threshold": 0.447,
    },
    "fusion_model": {
        "name": "Fusion Model (Logistic Regression)",
        "f1": 0.813,
        "auc": 0.871,
        "baseline_f1": 0.806,
        "baseline_auc": 0.861,
        "improvement_f1_percent": 0.87,
        "improvement_auc_percent": 1.16,
        "split": "internal_test (CASIA)",
        "n_samples": 465,
    },
}

# Dataset statistics for context
DATASET_STATS = {
    "total_cases": 3302,
    "authentic": 1605,
    "tampered": 1697,
    "cross_modal_agreement_mean": 0.5678,
    "cross_modal_agreement_std": 0.1551,
    "confidence_score_mean": 0.8885,
    "confidence_score_std": 0.2002,
    "case_priority_score_mean": 0.4814,
    "case_priority_score_std": 0.4551,
}

# ELA feature names for SHAP — order MUST match training order
ELA_FEATURE_NAMES = [
    "ela_mean", "ela_std", "ela_max", "ela_p95", "ela_high_energy_ratio",
    "laplacian_var", "local_noise_std_mean", "local_noise_std_std", "edge_density",
    "patch_ela_mean_of_means", "patch_ela_std_of_means", "patch_ela_max_zscore",
    "patch_ela_max_minus_median", "patch_edge_std_of_means",
]

METADATA_FEATURE_NAMES = [
    "flag_exif_missing", "flag_editing_software", "flag_implausible_timestamp",
    "flag_no_camera_info", "metadata_anomaly_score", "exif_present",
    "camera_make", "camera_model", "software", "datetime_original",
]

# ============================================================================
# LOAD MODELS & DATA (cached for performance)
# ============================================================================

@st.cache_resource
def load_models():
    """Load RF classifier, fusion model, and SHAP background."""
    with open(MODEL_RF_PATH, "rb") as f:
        rf_data = pickle.load(f)
    rf_model = rf_data["model"]

    with open(MODEL_FUSION_PATH, "rb") as f:
        fusion_data = pickle.load(f)
    fusion_model = fusion_data["model"]
    fusion_threshold = fusion_data["threshold"]

    with open(SHAP_BG_PATH, "rb") as f:
        shap_bg = pickle.load(f)  # shape (n, 24): 14 ELA cols + 10 metadata cols, in that order

    with open(SHAP_COLS_PATH, "rb") as f:
        shap_cols = pickle.load(f)

    with open(NORM_STATS_PATH, "r") as f:
        norm_stats = json.load(f)

    return rf_model, fusion_model, fusion_threshold, shap_bg, shap_cols, norm_stats


@st.cache_data
def load_dataset_sample():
    """Load dashboard metrics for context."""
    try:
        return pd.read_csv(DASHBOARD_METRICS_PATH)
    except Exception:
        return None


# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def process_uploaded_image(image_file):
    """Extract ELA features and metadata from uploaded image."""
    temp_path = f"/tmp/upload_{image_file.name}"
    with open(temp_path, "wb") as f:
        f.write(image_file.getbuffer())

    # Extract ELA features
    ela_result = extract_ela_features(temp_path)
    ela_features = {col: ela_result[col] for col in ELA_FEATURE_NAMES}

    # Extract metadata
    metadata_raw = extract_all_metadata(temp_path)
    metadata_analyzed = analyze_metadata(metadata_raw)

    metadata_features = {}
    for col in METADATA_FEATURE_NAMES:
        if col in metadata_analyzed:
            metadata_features[col] = metadata_analyzed[col]
        elif col in metadata_raw:
            metadata_features[col] = metadata_raw[col]
        else:
            metadata_features[col] = None

    # Load image for display from the temp file (avoids stream-position issues)
    img = Image.open(temp_path)
    img.load()

    return {
        "image": img,
        "ela_features": ela_features,
        "metadata_features": metadata_features,
        "metadata_raw": metadata_raw,
        "temp_path": temp_path,
    }


def cleanup_temp_file(temp_path):
    try:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
    except OSError:
        pass


# ============================================================================
# PREDICTION & SCORING
# ============================================================================

def predict_image_tampering(rf_model, ela_features):
    """Predict image tampering probability using the 14 ELA features."""
    X = np.array([[ela_features[col] for col in ELA_FEATURE_NAMES]])
    proba = rf_model.predict_proba(X)[0]
    tamper_prob = proba[1]  # probability of tampering class
    return tamper_prob


def score_metadata_anomaly(metadata_features):
    """Extract metadata anomaly score."""
    return metadata_features.get("metadata_anomaly_score", 0.5)


def compute_evidence_health_index(rf_model, fusion_model, fusion_threshold,
                                   ela_features, metadata_features, norm_stats):
    """Compute all validated metrics from real data."""
    image_tamper_prob = predict_image_tampering(rf_model, ela_features)
    metadata_anomaly_score = score_metadata_anomaly(metadata_features)

    # Fusion EHI
    X_fusion = np.array([[image_tamper_prob, metadata_anomaly_score]])
    fusion_prob = fusion_model.predict_proba(X_fusion)[0][1]

    # Cross-modal agreement (z-score based)
    image_z = (image_tamper_prob - norm_stats["image_tamper_prob"]["mean"]) / norm_stats["image_tamper_prob"]["std"]
    metadata_z = (metadata_anomaly_score - norm_stats["metadata_anomaly_score"]["mean"]) / norm_stats["metadata_anomaly_score"]["std"]
    disagreement = abs(image_z - metadata_z)
    cross_modal_agreement = 1.0 / (1.0 + disagreement)

    # Confidence (distance from 0.5)
    confidence = abs(fusion_prob - 0.5) * 2

    # Case priority (same as EHI)
    case_priority = fusion_prob

    return {
        "image_tamper_prob": image_tamper_prob,
        "metadata_anomaly_score": metadata_anomaly_score,
        "evidence_health_index": fusion_prob,
        "cross_modal_agreement_score": cross_modal_agreement,
        "confidence_score": confidence,
        "case_priority_score": case_priority,
        "fusion_threshold": fusion_threshold,
    }


# ============================================================================
# EXPLANATIONS
# Two separate, correctly-scoped explanations rather than one mismatched one:
#   1. SHAP over the 14 ELA features -> explains the image classifier
#   2. Coefficient x value breakdown over the 2 fusion inputs -> explains EHI
# ============================================================================

@st.cache_resource
def build_shap_explainer(_rf_model, _shap_bg):
    """
    Build a SHAP TreeExplainer for the RF image classifier.
    _shap_bg is the combined (24-col) background; the RF model only ever
    saw the first 14 (ELA) columns, so we slice those out and subsample
    for speed, then use them as the interventional background.
    """
    ela_background = np.asarray(_shap_bg)[:, :len(ELA_FEATURE_NAMES)]
    if ela_background.shape[0] > 100:
        rng = np.random.default_rng(42)
        idx = rng.choice(ela_background.shape[0], size=100, replace=False)
        ela_background = ela_background[idx]
    return shap.TreeExplainer(_rf_model, data=ela_background, feature_perturbation="interventional")


def compute_shap_values(explainer, ela_features):
    """Compute SHAP values for the 14 ELA features only (matches rf_model's training input)."""
    X = np.array([[ela_features[col] for col in ELA_FEATURE_NAMES]])
    shap_values = explainer.shap_values(X)
    return shap_values


def compute_fusion_contribution(fusion_model, image_tamper_prob, metadata_anomaly_score):
    """
    Exact (not approximated) contribution of each of the fusion model's two
    inputs to its logit, via coefficient x value. Returns None if the fusion
    model isn't a linear model with .coef_.
    """
    if not hasattr(fusion_model, "coef_"):
        return None
    coefs = fusion_model.coef_[0]
    contrib_image = float(coefs[0] * image_tamper_prob)
    contrib_metadata = float(coefs[1] * metadata_anomaly_score)
    return {
        "Image Signal": contrib_image,
        "Metadata Signal": contrib_metadata,
    }


# ============================================================================
# UI LAYOUT
# ============================================================================

def main():
    st.title("🔍 Evidence Health Index Dashboard")
    st.subheader("Research Paper Demo — Real Data Analysis")

    with st.spinner("Loading models..."):
        rf_model, fusion_model, fusion_threshold, shap_bg, shap_cols, norm_stats = load_models()
        dataset_df = load_dataset_sample()

    tab_upload, tab_explained, tab_metrics, tab_dataset = st.tabs([
        "📤 Upload & Analyze",
        "📊 Understanding the Metrics",
        "🎯 Model Performance",
        "📈 Dataset Overview",
    ])

    # ========================================================================
    # TAB 1: UPLOAD & ANALYZE
    # ========================================================================
    with tab_upload:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Upload Image")
            uploaded_file = st.file_uploader("Choose an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            with st.spinner("Processing image..."):
                result = process_uploaded_image(uploaded_file)
                img = result["image"]
                ela_features = result["ela_features"]
                metadata_features = result["metadata_features"]
                metadata_raw = result["metadata_raw"]
                temp_path = result["temp_path"]

            scores = compute_evidence_health_index(
                rf_model, fusion_model, fusion_threshold,
                ela_features, metadata_features, norm_stats
            )

            with col1:
                st.image(img, caption="Uploaded Image", use_container_width=True)
                st.write(f"**Dimensions:** {img.width} × {img.height} px")
                st.write(f"**Format:** {img.format or 'Unknown'}")

            with col2:
                st.subheader("📋 Analysis Results")

                ehi = scores["evidence_health_index"]
                threshold = scores["fusion_threshold"]
                is_flagged = ehi > threshold

                verdict = "🔴 FLAGGED" if is_flagged else "🟢 AUTHENTIC"
                st.markdown(f"### {verdict}")

                col_meter, col_val = st.columns([3, 1])
                with col_meter:
                    st.progress(float(ehi), text="Evidence Health Index")
                with col_val:
                    st.metric("EHI", f"{ehi:.3f}", f"Threshold: {threshold:.3f}")

                st.divider()
                st.write("**Confidence & Alignment Metrics:**")

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric(
                        "Confidence",
                        f"{scores['confidence_score']:.3f}",
                        help="How far is the model from uncertain (0.5)? 1.0 = very confident.",
                    )
                with m2:
                    st.metric(
                        "Cross-Modal Agreement",
                        f"{scores['cross_modal_agreement_score']:.3f}",
                        help="Do image and metadata anomaly scores align? 1.0 = perfect agreement.",
                    )
                with m3:
                    st.metric(
                        "Case Priority",
                        f"{scores['case_priority_score']:.3f}",
                        help="Triage score: higher = more suspicious.",
                    )

                st.divider()

            st.subheader("📸 Image Signal")
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.write("**ELA-based Tampering Probability:**")
                st.metric("Image Tamper Prob", f"{scores['image_tamper_prob']:.3f}")
            with img_col2:
                st.write("**ELA Feature Values (Sample):**")
                st.json({
                    "ela_max": ela_features["ela_max"],
                    "ela_std": ela_features["ela_std"],
                    "laplacian_var": ela_features["laplacian_var"],
                })

            st.subheader("🏷️ Metadata Signal")
            meta_col1, meta_col2 = st.columns(2)
            with meta_col1:
                st.write("**Metadata Anomaly Score:**")
                st.metric("Metadata Anomaly", f"{scores['metadata_anomaly_score']:.3f}")
            with meta_col2:
                st.write("**Metadata Flags & Info:**")
                st.json({
                    "EXIF Present": metadata_features.get("exif_present", False),
                    "Editing Software Detected": metadata_features.get("flag_editing_software", False),
                    "Missing Camera Info": metadata_features.get("flag_no_camera_info", False),
                    "Software": metadata_raw.get("Software", "N/A"),
                })
                st.caption(
                    "Metadata fields are shown for context only — the image classifier "
                    "below was trained on ELA features alone, not on metadata."
                )

            st.divider()

            # ---- Explanation: image classifier (SHAP, 14 ELA features) ----
            st.subheader("🧠 Image Classifier — SHAP Feature Importance")
            with st.spinner("Computing SHAP explanation..."):
                explainer = build_shap_explainer(rf_model, shap_bg)
                shap_values = compute_shap_values(explainer, ela_features)

            if isinstance(shap_values, list):
                shap_vals = shap_values[1]  # tampering class
            elif shap_values.ndim == 3:
                shap_vals = shap_values[:, :, 1]
            else:
                shap_vals = shap_values

            shap_df = pd.DataFrame({
                "Feature": ELA_FEATURE_NAMES,
                "SHAP Value": np.abs(shap_vals[0]).tolist(),
            }).sort_values("SHAP Value", ascending=True)

            st.bar_chart(shap_df.set_index("Feature")["SHAP Value"])
            st.caption("Larger bars = stronger influence on the image tampering prediction.")

            # ---- Explanation: fusion layer (image vs metadata contribution) ----
            st.subheader("⚖️ Fusion Layer — Image vs Metadata Contribution")
            fusion_contrib = compute_fusion_contribution(
                fusion_model, scores["image_tamper_prob"], scores["metadata_anomaly_score"]
            )
            if fusion_contrib is not None:
                contrib_df = pd.DataFrame({
                    "Signal": list(fusion_contrib.keys()),
                    "Contribution to EHI logit": list(fusion_contrib.values()),
                })
                st.bar_chart(contrib_df.set_index("Signal")["Contribution to EHI logit"])
                st.caption(
                    "Exact coefficient × value contribution of each input to the fusion "
                    "model's decision — not an approximation."
                )
            else:
                st.info("Fusion model does not expose linear coefficients; contribution breakdown unavailable.")

            cleanup_temp_file(temp_path)

        else:
            st.info("👆 Upload an image to analyze it.")

    # ========================================================================
    # TAB 2: UNDERSTANDING THE METRICS
    # ========================================================================
    with tab_explained:
        st.subheader("What do these metrics mean?")
        st.write("""
        ## Evidence Health Index (EHI)

        The **EHI** is a fusion score (0–1) combining two independent real-data signals:
        - **Image tampering probability** from ELA (Error Level Analysis) features
        - **Metadata anomaly score** from EXIF/file properties

        **Verdict:** EHI > 0.387 = FLAGGED for review; otherwise AUTHENTIC.

        ---

        ## Confidence Score

        How **certain** is the model about its verdict?
        - 0 = completely uncertain (right at the threshold)
        - 1 = completely confident (far from threshold)

        **Use case:** Help prioritize cases — high confidence + high EHI = definitely suspicious.

        ---

        ## Cross-Modal Agreement Score

        Do the **image** and **metadata** signals point in the same direction?
        - 1.0 = perfect agreement (both flag or both pass)
        - 0 = complete disagreement (one flags, other passes)

        **Why it matters:** If both signal tampering independently, it's more credible.

        ---

        ## Case Priority Score

        Triage ranking: higher = more suspicious, worth investigating first.
        Same as EHI (0–1 scale).

        ---

        ## How they're computed

        1. **ELA Features** (14 numbers): statistical properties of error-level residuals
        2. **Metadata Flags** (10 numbers): presence of EXIF, editing software, anomalies
        3. **Image Classifier** (Random Forest): trained on the 14 ELA features, 81% accuracy
        4. **Fusion Model** (Logistic Regression): combines image + metadata scores into EHI
        5. **SHAP Explanation**: per-image breakdown of which ELA features mattered most,
           plus an exact contribution breakdown for the fusion step
        """)

    # ========================================================================
    # TAB 3: MODEL PERFORMANCE
    # ========================================================================
    with tab_metrics:
        st.subheader("Model Performance Metrics")
        st.write("All metrics are from **internal test set** (CASIA, n=465) — held-out validation data the model never saw during training.")

        st.markdown("### 1️⃣ Image Classifier (Random Forest on ELA Features)")
        rf_metrics = MODEL_METRICS["rf_classifier"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{rf_metrics['accuracy']:.1%}")
        col2.metric("Precision", f"{rf_metrics['precision']:.1%}")
        col3.metric("Recall", f"{rf_metrics['recall']:.1%}")
        col4.metric("F1 Score", f"{rf_metrics['f1']:.3f}")

        col1b, col2b = st.columns(2)
        col1b.metric("ROC-AUC", f"{rf_metrics['auc']:.3f}")
        col2b.metric("Decision Threshold", f"{rf_metrics['threshold']:.3f}")

        st.caption(f"Model trained/evaluated on {rf_metrics['n_samples']} internal test cases.")

        st.markdown("### 2️⃣ Fusion Model (Image + Metadata → EHI)")
        fusion_metrics = MODEL_METRICS["fusion_model"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("F1 Score", f"{fusion_metrics['f1']:.3f}")
        col2.metric("ROC-AUC", f"{fusion_metrics['auc']:.3f}")
        col3.metric("Baseline (Image Only)", f"F1: {fusion_metrics['baseline_f1']:.3f}")
        col4.metric("Improvement", f"+{fusion_metrics['improvement_f1_percent']:.2f}% (F1)")

        st.caption("Fusion adds metadata signal to improve detection. Modest improvement is expected & honest.")

        st.markdown("### Fusion vs. Baseline (Image Only)")
        comparison_df = pd.DataFrame({
            "Model": ["Image Only (Baseline)", "Fusion (Image + Metadata)"],
            "F1": [fusion_metrics["baseline_f1"], fusion_metrics["f1"]],
            "ROC-AUC": [fusion_metrics["baseline_auc"], fusion_metrics["auc"]],
        })
        st.dataframe(comparison_df, use_container_width=True)

    # ========================================================================
    # TAB 4: DATASET OVERVIEW
    # ========================================================================
    with tab_dataset:
        st.subheader("Dataset Statistics (3302 cases)")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Cases", f"{DATASET_STATS['total_cases']:,}")
        col2.metric("Authentic", f"{DATASET_STATS['authentic']:,}")
        col3.metric("Tampered", f"{DATASET_STATS['tampered']:,}")

        st.markdown("### Metric Distributions (Over Full Dataset)")
        dist_col1, dist_col2, dist_col3 = st.columns(3)
        with dist_col1:
            st.metric(
                "Cross-Modal Agreement",
                f"{DATASET_STATS['cross_modal_agreement_mean']:.3f} ± {DATASET_STATS['cross_modal_agreement_std']:.3f}",
                help="Mean ± Std over all 3302 cases",
            )
        with dist_col2:
            st.metric(
                "Confidence Score",
                f"{DATASET_STATS['confidence_score_mean']:.3f} ± {DATASET_STATS['confidence_score_std']:.3f}",
                help="Mean ± Std over all 3302 cases",
            )
        with dist_col3:
            st.metric(
                "Case Priority (EHI)",
                f"{DATASET_STATS['case_priority_score_mean']:.3f} ± {DATASET_STATS['case_priority_score_std']:.3f}",
                help="Mean ± Std over all 3302 cases",
            )

        if dataset_df is not None:
            st.markdown("### Sample Cases from Dataset")
            st.dataframe(dataset_df.head(10), use_container_width=True, height=400)
            st.caption(f"Showing first 10 of {len(dataset_df)} cases")

        st.markdown("### Data Splits")
        st.write("""
        - **Train:** 2170 cases (used only for model training)
        - **Validation:** 465 cases (used for threshold tuning)
        - **Internal Test:** 465 cases (CASIA dataset, held-out evaluation)
        - **External Test:** 202 cases (COVERAGE dataset, different camera/compression)

        *Note: Train accuracy is 100% (memorized), so we report validation/test instead.*
        """)

    st.divider()
    st.markdown("""
    ---
    **Research Paper Version** | Real Data Only | No Synthetic Components in Verdict

    Built with Streamlit + SHAP + scikit-learn
    """)


if __name__ == "__main__":
    main()
