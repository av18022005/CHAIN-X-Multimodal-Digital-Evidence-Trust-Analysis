"""
TIER 1 STREAMLIT DASHBOARD: Evidence Health Index (Research Paper Version)
Real data only — image + metadata → EHI with SHAP explanation.
Visually redesigned version with Plotly charts and custom styling.
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
import plotly.graph_objects as go
import plotly.express as px

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from src.image.ela import extract_all_features as extract_ela_features
from src.metadata.extractor import extract_all_metadata
from src.metadata.anomaly import analyze as analyze_metadata

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(page_title="Evidence Health Index - Research Demo", layout="wide", page_icon="🔍")

MODEL_RF_PATH = "models/rf_final_locked_in.pkl"
MODEL_FUSION_PATH = "models/fusion_v1.pkl"
SHAP_BG_PATH = "/kaggle/working/shap_background_combined.pkl"
SHAP_COLS_PATH = "/kaggle/working/shap_feature_columns.pkl"
NORM_STATS_PATH = "/kaggle/working/normalization_stats.json"
DASHBOARD_METRICS_PATH = "/kaggle/working/features/dashboard_metrics_tier1.csv"

MODEL_METRICS = {
    "rf_classifier": {
        "name": "Image Classifier (Random Forest)",
        "accuracy": 0.789, "precision": 0.795, "recall": 0.79,
        "f1": 0.806, "auc": 0.861, "split": "internal_test (CASIA)",
        "n_samples": 465, "threshold": 0.447,
    },
    "fusion_model": {
        "name": "Fusion Model (Logistic Regression)",
        "f1": 0.813, "auc": 0.871, "baseline_f1": 0.806, "baseline_auc": 0.861,
        "improvement_f1_percent": 0.87, "improvement_auc_percent": 1.16,
        "split": "internal_test (CASIA)", "n_samples": 465,
    },
}

DATASET_STATS = {
    "total_cases": 3302, "authentic": 1605, "tampered": 1697,
    "cross_modal_agreement_mean": 0.5678, "cross_modal_agreement_std": 0.1551,
    "confidence_score_mean": 0.8885, "confidence_score_std": 0.2002,
    "case_priority_score_mean": 0.4814, "case_priority_score_std": 0.4551,
}

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
# CUSTOM CSS
# ============================================================================

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .verdict-card {
        padding: 1.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .verdict-flagged {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
    }
    .verdict-authentic {
        background: linear-gradient(135deg, #38ef7d 0%, #11998e 100%);
        color: white;
    }
    .verdict-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
    }
    .verdict-subtitle {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.3rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #4361ee;
        margin-bottom: 0.5rem;
    }
    .metric-card-label {
        font-size: 0.8rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .section-divider {
        border-top: 1px solid #e9ecef;
        margin: 1.5rem 0;
    }
    .info-pill {
        display: inline-block;
        background: #e7f0ff;
        color: #3a56d4;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.8rem;
        margin-right: 0.4rem;
    }
</style>
"""

# ============================================================================
# LOAD MODELS & DATA
# ============================================================================

@st.cache_resource
def load_models():
    with open(MODEL_RF_PATH, "rb") as f:
        rf_data = pickle.load(f)
    rf_model = rf_data["model"]

    with open(MODEL_FUSION_PATH, "rb") as f:
        fusion_data = pickle.load(f)
    fusion_model = fusion_data["model"]
    fusion_threshold = fusion_data["threshold"]

    with open(SHAP_BG_PATH, "rb") as f:
        shap_bg = pickle.load(f)

    with open(SHAP_COLS_PATH, "rb") as f:
        shap_cols = pickle.load(f)

    with open(NORM_STATS_PATH, "r") as f:
        norm_stats = json.load(f)

    return rf_model, fusion_model, fusion_threshold, shap_bg, shap_cols, norm_stats


@st.cache_data
def load_dataset_sample():
    try:
        return pd.read_csv(DASHBOARD_METRICS_PATH)
    except Exception:
        return None


# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def process_uploaded_image(image_file):
    temp_path = f"/tmp/upload_{image_file.name}"
    with open(temp_path, "wb") as f:
        f.write(image_file.getbuffer())

    ela_result = extract_ela_features(temp_path)
    ela_features = {col: ela_result[col] for col in ELA_FEATURE_NAMES}

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

    img = Image.open(temp_path)
    img.load()

    return {
        "image": img, "ela_features": ela_features,
        "metadata_features": metadata_features, "metadata_raw": metadata_raw,
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
    X = np.array([[ela_features[col] for col in ELA_FEATURE_NAMES]])
    proba = rf_model.predict_proba(X)[0]
    return proba[1]


def score_metadata_anomaly(metadata_features):
    return metadata_features.get("metadata_anomaly_score", 0.5)


def compute_evidence_health_index(rf_model, fusion_model, fusion_threshold,
                                   ela_features, metadata_features, norm_stats):
    image_tamper_prob = predict_image_tampering(rf_model, ela_features)
    metadata_anomaly_score = score_metadata_anomaly(metadata_features)

    X_fusion = np.array([[image_tamper_prob, metadata_anomaly_score]])
    fusion_prob = fusion_model.predict_proba(X_fusion)[0][1]

    image_z = (image_tamper_prob - norm_stats["image_tamper_prob"]["mean"]) / norm_stats["image_tamper_prob"]["std"]
    metadata_z = (metadata_anomaly_score - norm_stats["metadata_anomaly_score"]["mean"]) / norm_stats["metadata_anomaly_score"]["std"]
    disagreement = abs(image_z - metadata_z)
    cross_modal_agreement = 1.0 / (1.0 + disagreement)

    confidence = abs(fusion_prob - 0.5) * 2
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
# ============================================================================

@st.cache_resource
def build_shap_explainer(_rf_model, _shap_bg):
    # FIX: explicit float64 cast — the combined background pickle has mixed
    # dtypes (ELA floats + metadata strings), so slicing alone still leaves
    # an object-dtype array, which SHAP's C extension refuses to cast.
    ela_background = np.asarray(_shap_bg)[:, :len(ELA_FEATURE_NAMES)].astype(np.float64)
    if ela_background.shape[0] > 100:
        rng = np.random.default_rng(42)
        idx = rng.choice(ela_background.shape[0], size=100, replace=False)
        ela_background = ela_background[idx]
    return shap.TreeExplainer(_rf_model, data=ela_background, feature_perturbation="interventional")


def compute_shap_values(explainer, ela_features):
    X = np.array([[ela_features[col] for col in ELA_FEATURE_NAMES]])
    return explainer.shap_values(X)


def compute_fusion_contribution(fusion_model, image_tamper_prob, metadata_anomaly_score):
    if not hasattr(fusion_model, "coef_"):
        return None
    coefs = fusion_model.coef_[0]
    return {
        "Image Signal": float(coefs[0] * image_tamper_prob),
        "Metadata Signal": float(coefs[1] * metadata_anomaly_score),
    }


# ============================================================================
# VISUAL HELPERS
# ============================================================================

def render_gauge(value, threshold, title="Evidence Health Index"):
    color = "#ee5a6f" if value > threshold else "#11998e"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"valueformat": ".3f", "font": {"size": 40}},
        title={"text": title, "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "white",
            "steps": [
                {"range": [0, threshold], "color": "#d4f5e9"},
                {"range": [threshold, 1], "color": "#ffe0e3"},
            ],
            "threshold": {
                "line": {"color": "#1a1a2e", "width": 3},
                "thickness": 0.8,
                "value": threshold,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=10))
    return fig


def metric_card(label, value, icon=""):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">{icon} {label}</div>
            <div class="metric-card-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)


def render_shap_bar(shap_df):
    colors = ["#ee5a6f" if v > 0 else "#4361ee" for v in shap_df["Raw SHAP"]]
    fig = go.Figure(go.Bar(
        x=shap_df["SHAP Value"], y=shap_df["Feature"],
        orientation="h", marker_color=colors,
        text=[f"{v:.3f}" for v in shap_df["SHAP Value"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Feature contributions to tampering prediction",
        xaxis_title="Absolute SHAP value", height=450,
        margin=dict(l=10, r=40, t=50, b=10),
        plot_bgcolor="white",
    )
    return fig


def render_fusion_bar(contrib_df):
    colors = ["#ee5a6f" if v > 0 else "#11998e" for v in contrib_df["Contribution to EHI logit"]]
    fig = go.Figure(go.Bar(
        x=contrib_df["Signal"], y=contrib_df["Contribution to EHI logit"],
        marker_color=colors,
        text=[f"{v:.3f}" for v in contrib_df["Contribution to EHI logit"]],
        textposition="outside",
    ))
    fig.update_layout(height=350, plot_bgcolor="white", margin=dict(l=10, r=10, t=30, b=10))
    return fig


# ============================================================================
# UI LAYOUT
# ============================================================================

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<p class="main-header">🔍 Evidence Health Index Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Research Paper Demo · Real Data Analysis · Image + Metadata Fusion</p>', unsafe_allow_html=True)

    with st.spinner("Loading models..."):
        rf_model, fusion_model, fusion_threshold, shap_bg, shap_cols, norm_stats = load_models()
        dataset_df = load_dataset_sample()

    tab_upload, tab_explained, tab_metrics, tab_dataset = st.tabs([
        "📤  Upload & Analyze", "📊  Understanding the Metrics",
        "🎯  Model Performance", "📈  Dataset Overview",
    ])

    # ------------------------------------------------------------------
    # TAB 1: UPLOAD & ANALYZE
    # ------------------------------------------------------------------
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
                st.markdown(
                    f'<span class="info-pill">{img.width}×{img.height}px</span>'
                    f'<span class="info-pill">{img.format or "Unknown"}</span>',
                    unsafe_allow_html=True,
                )

            ehi = scores["evidence_health_index"]
            threshold = scores["fusion_threshold"]
            is_flagged = ehi > threshold

            with col2:
                verdict_class = "verdict-flagged" if is_flagged else "verdict-authentic"
                verdict_text = "🔴 FLAGGED FOR REVIEW" if is_flagged else "🟢 LIKELY AUTHENTIC"
                st.markdown(f"""
                    <div class="verdict-card {verdict_class}">
                        <p class="verdict-title">{verdict_text}</p>
                        <p class="verdict-subtitle">EHI {ehi:.3f} vs threshold {threshold:.3f}</p>
                    </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(render_gauge(ehi, threshold), use_container_width=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.write("**Confidence & Alignment Metrics**")
            m1, m2, m3 = st.columns(3)
            with m1:
                metric_card("Confidence", f"{scores['confidence_score']:.3f}", "🎯")
            with m2:
                metric_card("Cross-Modal Agreement", f"{scores['cross_modal_agreement_score']:.3f}", "🔗")
            with m3:
                metric_card("Case Priority", f"{scores['case_priority_score']:.3f}", "🚨")

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            sig_col1, sig_col2 = st.columns(2)
            with sig_col1:
                st.subheader("📸 Image Signal")
                metric_card("ELA Tamper Probability", f"{scores['image_tamper_prob']:.3f}")
                ela_chart_df = pd.DataFrame({
                    "Feature": ["ela_max", "ela_std", "laplacian_var"],
                    "Value": [ela_features["ela_max"], ela_features["ela_std"], ela_features["laplacian_var"]],
                })
                fig_ela = px.bar(ela_chart_df, x="Feature", y="Value", color="Feature",
                                  color_discrete_sequence=px.colors.qualitative.Set2)
                fig_ela.update_layout(showlegend=False, height=280, plot_bgcolor="white",
                                       margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig_ela, use_container_width=True)

            with sig_col2:
                st.subheader("🏷️ Metadata Signal")
                metric_card("Metadata Anomaly Score", f"{scores['metadata_anomaly_score']:.3f}")
                flags = {
                    "EXIF Present": bool(metadata_features.get("exif_present", False)),
                    "Editing Software": bool(metadata_features.get("flag_editing_software", False)),
                    "Missing Camera Info": bool(metadata_features.get("flag_no_camera_info", False)),
                }
                flags_df = pd.DataFrame({"Flag": list(flags.keys()), "Status": [int(v) for v in flags.values()]})
                fig_flags = px.bar(flags_df, x="Flag", y="Status", color="Flag",
                                    color_discrete_sequence=["#4361ee", "#ee5a6f", "#f9a826"])
                fig_flags.update_layout(showlegend=False, height=280, plot_bgcolor="white",
                                         yaxis=dict(range=[0, 1.2], tickvals=[0, 1], ticktext=["No", "Yes"]),
                                         margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig_flags, use_container_width=True)
                st.caption("Metadata shown for context only — the image classifier below is ELA-only.")

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            st.subheader("🧠 Image Classifier — SHAP Feature Importance")
            with st.spinner("Computing SHAP explanation..."):
                explainer = build_shap_explainer(rf_model, shap_bg)
                shap_values = compute_shap_values(explainer, ela_features)

            if isinstance(shap_values, list):
                shap_vals = shap_values[1]
            elif shap_values.ndim == 3:
                shap_vals = shap_values[:, :, 1]
            else:
                shap_vals = shap_values

            shap_df = pd.DataFrame({
                "Feature": ELA_FEATURE_NAMES,
                "Raw SHAP": shap_vals[0].tolist(),
                "SHAP Value": np.abs(shap_vals[0]).tolist(),
            }).sort_values("SHAP Value", ascending=True)

            st.plotly_chart(render_shap_bar(shap_df), use_container_width=True)
            st.caption("🔴 Red = pushes toward TAMPERED · 🔵 Blue = pushes toward AUTHENTIC")

            st.subheader("⚖️ Fusion Layer — Image vs Metadata Contribution")
            fusion_contrib = compute_fusion_contribution(
                fusion_model, scores["image_tamper_prob"], scores["metadata_anomaly_score"]
            )
            if fusion_contrib is not None:
                contrib_df = pd.DataFrame({
                    "Signal": list(fusion_contrib.keys()),
                    "Contribution to EHI logit": list(fusion_contrib.values()),
                })
                st.plotly_chart(render_fusion_bar(contrib_df), use_container_width=True)
                st.caption("Exact coefficient × value contribution — not an approximation.")
            else:
                st.info("Fusion model does not expose linear coefficients.")

            cleanup_temp_file(temp_path)

        else:
            st.info("👆 Upload an image to analyze it.")

    # ------------------------------------------------------------------
    # TAB 2: UNDERSTANDING THE METRICS
    # ------------------------------------------------------------------
    with tab_explained:
        st.subheader("What do these metrics mean?")
        st.write("""
        ## Evidence Health Index (EHI)
        The **EHI** is a fusion score (0–1) combining two independent real-data signals:
        image tampering probability (ELA) and metadata anomaly score.
        **Verdict:** EHI > 0.387 → FLAGGED; otherwise AUTHENTIC.

        ## Confidence Score
        Distance of the fusion probability from 0.5 (the uncertain midpoint).

        ## Cross-Modal Agreement Score
        Do the image and metadata signals point the same direction?

        ## Case Priority Score
        Triage rank — same scale as EHI, higher = investigate first.
        """)

    # ------------------------------------------------------------------
    # TAB 3: MODEL PERFORMANCE
    # ------------------------------------------------------------------
    with tab_metrics:
        st.subheader("Model Performance Metrics")
        st.caption("All metrics from internal test set (CASIA, n=465) — held out during training.")

        rf_metrics = MODEL_METRICS["rf_classifier"]
        fusion_metrics = MODEL_METRICS["fusion_model"]

        col1, col2, col3, col4 = st.columns(4)
        with col1: metric_card("Accuracy", f"{rf_metrics['accuracy']:.1%}")
        with col2: metric_card("Precision", f"{rf_metrics['precision']:.1%}")
        with col3: metric_card("Recall", f"{rf_metrics['recall']:.1%}")
        with col4: metric_card("F1 Score", f"{rf_metrics['f1']:.3f}")

        comp_df = pd.DataFrame({
            "Model": ["Image Only (Baseline)", "Fusion (Image+Metadata)"],
            "F1": [fusion_metrics["baseline_f1"], fusion_metrics["f1"]],
            "ROC-AUC": [fusion_metrics["baseline_auc"], fusion_metrics["auc"]],
        })
        comp_long = comp_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig_comp = px.bar(comp_long, x="Metric", y="Score", color="Model", barmode="group",
                           color_discrete_sequence=["#4361ee", "#11998e"], text="Score")
        fig_comp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_comp.update_layout(height=380, plot_bgcolor="white", yaxis_range=[0, 1])
        st.plotly_chart(fig_comp, use_container_width=True)
        st.caption("Fusion adds a modest, honest improvement over image-only baseline.")

    # ------------------------------------------------------------------
    # TAB 4: DATASET OVERVIEW
    # ------------------------------------------------------------------
    with tab_dataset:
        st.subheader("Dataset Statistics")

        pie_df = pd.DataFrame({
            "Label": ["Authentic", "Tampered"],
            "Count": [DATASET_STATS["authentic"], DATASET_STATS["tampered"]],
        })
        fig_pie = px.pie(pie_df, names="Label", values="Count", hole=0.5,
                          color="Label", color_discrete_map={"Authentic": "#11998e", "Tampered": "#ee5a6f"})
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

        dist_df = pd.DataFrame({
            "Metric": ["Cross-Modal Agreement", "Confidence Score", "Case Priority"],
            "Mean": [DATASET_STATS["cross_modal_agreement_mean"], DATASET_STATS["confidence_score_mean"], DATASET_STATS["case_priority_score_mean"]],
            "Std": [DATASET_STATS["cross_modal_agreement_std"], DATASET_STATS["confidence_score_std"], DATASET_STATS["case_priority_score_std"]],
        })
        fig_dist = go.Figure(go.Bar(
            x=dist_df["Metric"], y=dist_df["Mean"],
            error_y=dict(type="data", array=dist_df["Std"]),
            marker_color=["#4361ee", "#f9a826", "#ee5a6f"],
        ))
        fig_dist.update_layout(title="Metric distributions (mean ± std, n=3302)", height=380, plot_bgcolor="white")
        st.plotly_chart(fig_dist, use_container_width=True)

        if dataset_df is not None:
            st.markdown("### Sample Cases")
            st.dataframe(dataset_df.head(10), use_container_width=True, height=350)

        st.markdown("""
        ### Data Splits
        - **Train:** 2170 cases · **Validation:** 465 · **Internal Test:** 465 (CASIA) · **External Test:** 202 (COVERAGE)
        """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption("Research Paper Version · Real Data Only · Built with Streamlit + SHAP + Plotly")


if __name__ == "__main__":
    main()
