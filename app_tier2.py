"""
TIER 2 STREAMLIT DASHBOARD: Evidence Health Index — Proof-of-Concept Demo
CHAIN-X — Multimodal Digital Evidence Trust Analysis

Shows: "what if we had custody + NLP report data?" — one fixed authentic
case and one fixed tampered case (both with real EXIF ground truth, and
now chosen so the model's own verdict agrees with the label), plus an
"Upload Your Own" tab that runs the real Tier 1 pipeline live. Interactive
toggles inject synthetic custody anomalies / report contradictions and
watch the two synthetic-only trust metrics respond live.

IMPORTANT: per project rules, this file is removed before paper submission.
Only app_tier1.py (real data only) goes in the paper.

Usage:
    streamlit run app_tier2.py

Dependencies:
    pip install streamlit shap pandas numpy pillow opencv-python-headless
    scikit-learn plotly networkx
"""

import os
import sys
import json
import pickle
import random
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

# Reuse your REAL scoring/analysis logic directly — only the anomaly
# *injection choice* is made interactive here; the scoring math is
# untouched and identical to what produced your paper's numbers.
from src.custody.graph_analyzer import build_case_graph, analyze_case_graph
from src.nlp.consistency import score_row as consistency_score_row

# Same real Tier 1 pipeline, reused so the "Upload Your Own" tab produces
# genuinely computed image_tamper_prob / metadata_anomaly_score / EHI
# instead of anything synthetic.
from src.image.ela import extract_all_features as extract_ela_features
from src.metadata.extractor import extract_all_metadata
from src.metadata.anomaly import analyze as analyze_metadata

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(page_title="ChainX — Tier 2 POC Demo", layout="wide", page_icon="🧪")

MASTER_CSV = "/kaggle/working/data/master_index.csv"
METADATA_CSV = "/kaggle/working/features/metadata_features.csv"
EHI_CSV = "/kaggle/working/features/evidence_health_index.csv"

# Tier 1 models — needed here only for the "Upload Your Own" tab, so a
# fresh upload gets a real image_tamper_prob / metadata_anomaly_score / EHI
# instead of nothing.
MODEL_RF_PATH = "models/rf_final_locked_in.pkl"
MODEL_FUSION_PATH = "models/fusion_v1.pkl"
NORM_STATS_PATH = "/kaggle/working/normalization_stats.json"

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

CUSTODY_ACTORS = [
    "Officer R. Malik", "Officer S. Chen", "Evidence Custodian A. Diaz",
    "Forensic Analyst K. Osei", "Forensic Analyst T. Lindqvist",
    "Lab Technician M. Fischer", "Case Reviewer P. Novak",
]
CUSTODY_ROLES_SEQUENCE = [
    ("acquisition", "Field Investigator"),
    ("intake_storage", "Evidence Custodian"),
    ("lab_handoff", "Forensic Analyst"),
    ("analysis", "Forensic Analyst"),
    ("report_filed", "Case Reviewer"),
]
FORCED_ANOMALY_EVENT_IDX = 2  # lab_handoff — deterministic placement for demo reproducibility
# NOTE: this index (and therefore which role gets flagged) is intentionally
# fixed regardless of case — see the "About This Demo" tab. The custody
# risk score for a given anomaly TYPE is also fixed by graph_analyzer.py's
# rule-based scoring, not derived per-case — this is why "missing_custodian"
# gives the same Custody Risk Score / flagged role on every case you try.
# That's expected behavior, not a bug in this file.

CAMERA_POOL = [
    "Canon EOS 5D", "Canon EOS 5D Mark II", "Canon PowerShot G12",
    "Nikon D90", "Nikon D7000", "Nikon COOLPIX P510",
    "Sony DSC-W800", "Sony Alpha a6000",
    "Apple iPhone 6", "Apple iPhone 11", "Samsung Galaxy S9",
    "Olympus E-M10", "Fujifilm FinePix S1",
]
REPORT_TEMPLATE = "Camera used: {camera}. Date of capture: {date}."

# Real, validated detection performance from your Phase 6 / Phase 4 runs
CUSTODY_DETECTION_METRICS = {
    "precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0,
    "n_cases": 3302, "flagged_pct": 30.7,
}
NLP_DETECTION_METRICS = {
    "precision": 1.0, "recall": 0.991, "f1": 0.995, "accuracy": 0.996,
    "n_evaluated": 278,
}

# ============================================================================
# CUSTOM CSS (same visual language as Tier 1)
# ============================================================================

CUSTOM_CSS = """
<style>
    .hero-header { padding: 1.2rem 0 0.5rem 0; }
    .project-badge {
        display: inline-block;
        background: linear-gradient(135deg, #f9a826 0%, #e85d04 100%);
        color: white; font-weight: 800; letter-spacing: 2px; font-size: 0.85rem;
        padding: 0.3rem 1rem; border-radius: 20px; margin-bottom: 0.8rem;
    }
    .main-header { font-size: 3rem; font-weight: 800; color: #1a1a2e; margin-bottom: 0; line-height: 1.1; }
    .sub-header { font-size: 1.1rem; color: #6c757d; margin-top: 0.4rem; margin-bottom: 1.5rem; }
    .poc-banner {
        background: #fff3cd; color: #856404; border-radius: 10px;
        padding: 0.7rem 1.2rem; font-size: 0.9rem; margin-bottom: 1.5rem;
        border-left: 4px solid #f9a826;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 12px; padding: 1rem 1.2rem;
        border-left: 4px solid #4361ee; margin-bottom: 0.5rem;
    }
    .metric-card-label { font-size: 0.8rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card-value { font-size: 1.6rem; font-weight: 700; color: #1a1a2e; }
    .synthetic-card { border-left: 4px solid #f9a826 !important; }
    .section-divider { border-top: 1px solid #e9ecef; margin: 1.5rem 0; }
    .info-pill {
        display: inline-block; background: #e7f0ff; color: #3a56d4; border-radius: 20px;
        padding: 0.2rem 0.8rem; font-size: 0.8rem; margin-right: 0.4rem;
    }
    .synthetic-pill {
        display: inline-block; background: #fff0d9; color: #b3610a; border-radius: 20px;
        padding: 0.2rem 0.8rem; font-size: 0.75rem; margin-left: 0.5rem; font-weight: 700;
    }
    .verdict-card {
        padding: 1.2rem 1.8rem; border-radius: 16px; text-align: center; margin-bottom: 1rem;
    }
    .verdict-flagged { background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); color: white; }
    .verdict-authentic { background: linear-gradient(135deg, #38ef7d 0%, #11998e 100%); color: white; }
    .verdict-title { font-size: 1.5rem; font-weight: 800; margin: 0; }
    .verdict-subtitle { font-size: 0.85rem; opacity: 0.9; margin-top: 0.3rem; }
    .report-box {
        background: #f8f9fa; border-radius: 10px; padding: 1rem 1.2rem;
        font-family: Georgia, serif; font-style: italic; color: #333;
        border-left: 3px solid #4361ee; margin-bottom: 0.8rem;
    }
</style>
"""

# ============================================================================
# CASE SELECTION — find one authentic + one tampered case with real EXIF
# ============================================================================

@st.cache_data
def select_demo_cases():
    """Pick an authentic and a tampered case that both have real EXIF ground
    truth (camera_model + datetime_original present), since contradiction
    injection is only meaningful with real ground truth to contradict.

    Among the eligible cases, prefer ones where the model's own verdict
    AGREES with the ground-truth label — otherwise the fixed demo can
    accidentally showcase a misclassification (an authentic case that gets
    flagged, or a tampered case scored as authentic), which is confusing to
    present. Falls back to the full candidate pool if no agreeing case
    exists for a given label.
    """
    master = pd.read_csv(MASTER_CSV)
    meta = pd.read_csv(METADATA_CSV)
    ehi = pd.read_csv(EHI_CSV)

    merged = master.merge(meta, on="case_id", how="left")

    # Drop metadata version of metadata_anomaly_score so it doesn't silently
    # rename to _x/_y suffix when we merge in the EHI version
    merged = merged.drop(columns=["metadata_anomaly_score"], errors="ignore")

    merged = merged.merge(
        ehi[["case_id", "image_tamper_prob", "metadata_anomaly_score", "evidence_health_index"]],
        on="case_id", how="left"
    )

    has_real_exif = merged["camera_model"].notna() & merged["datetime_original"].notna() & \
                    (merged["camera_model"].astype(str).str.strip() != "") & \
                    (merged["datetime_original"].astype(str).str.strip() != "")

    candidates = merged[has_real_exif]

    # Handle both label naming conventions, and both string ("authentic"/
    # "tampered") and numeric (0/1) label encodings.
    label_col = None
    if "label" in candidates.columns:
        label_col = "label"
    elif "true_label" in candidates.columns:
        label_col = "true_label"

    if label_col:
        label_series = candidates[label_col]
        if label_series.dtype == object:
            normalized = label_series.astype(str).str.lower()
            auth_candidates = candidates[normalized.isin(["authentic", "au", "0"])]
            tamp_candidates = candidates[normalized.isin(["tampered", "tp", "1"])]
        else:
            auth_candidates = candidates[label_series == 0]
            tamp_candidates = candidates[label_series == 1]
    else:
        auth_candidates = candidates.iloc[:len(candidates) // 2]
        tamp_candidates = candidates.iloc[len(candidates) // 2:]

    FUSION_THRESHOLD = 0.387  # keep in sync with the threshold used in render_case_panel

    # Prefer cases the model gets right; among those, the most confidently
    # correct one makes for the cleanest demo.
    auth_agree = auth_candidates[auth_candidates["evidence_health_index"] < FUSION_THRESHOLD]
    tamp_agree = tamp_candidates[tamp_candidates["evidence_health_index"] > FUSION_THRESHOLD]

    auth_pool = auth_agree if len(auth_agree) > 0 else auth_candidates
    tamp_pool = tamp_agree if len(tamp_agree) > 0 else tamp_candidates

    if len(auth_pool) > 0:
        auth_case = auth_pool.sort_values("evidence_health_index").iloc[0]
    else:
        auth_case = candidates.iloc[0]

    if len(tamp_pool) > 0:
        tamp_case = tamp_pool.sort_values("evidence_health_index", ascending=False).iloc[0]
    else:
        tamp_case = candidates.iloc[1] if len(candidates) > 1 else candidates.iloc[0]

    return auth_case, tamp_case


def clean_exif_date(raw: str) -> str:
    """EXIF dates look like '2015:06:12 14:30:00' -> normalize to YYYY-MM-DD."""
    if not isinstance(raw, str) or not raw.strip():
        return ""
    date_part = raw.strip().split(" ")[0].replace(":", "-", 2)
    try:
        datetime.strptime(date_part, "%Y-%m-%d")
        return date_part
    except ValueError:
        return ""


# ============================================================================
# "UPLOAD YOUR OWN" — runs the real Tier 1 pipeline on a fresh image
# ============================================================================

@st.cache_resource
def load_tier1_models():
    with open(MODEL_RF_PATH, "rb") as f:
        rf_data = pickle.load(f)
    rf_model = rf_data["model"]

    with open(MODEL_FUSION_PATH, "rb") as f:
        fusion_data = pickle.load(f)
    fusion_model = fusion_data["model"]
    fusion_threshold = fusion_data["threshold"]

    with open(NORM_STATS_PATH, "r") as f:
        norm_stats = json.load(f)

    return rf_model, fusion_model, fusion_threshold, norm_stats


def process_uploaded_image(image_file):
    """Same extraction as Tier 1: save to a temp path, pull ELA + metadata."""
    temp_path = f"/tmp/tier2_upload_{image_file.name}"
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

    return {
        "ela_features": ela_features,
        "metadata_features": metadata_features,
        "metadata_raw": metadata_raw,
        "temp_path": temp_path,
    }


def predict_tier1_scores(rf_model, fusion_model, ela_features, metadata_features):
    """Real image_tamper_prob / metadata_anomaly_score / EHI — identical math to app_tier1.py."""
    X_img = np.array([[ela_features[c] for c in ELA_FEATURE_NAMES]])
    image_tamper_prob = rf_model.predict_proba(X_img)[0][1]

    metadata_anomaly_score = metadata_features.get("metadata_anomaly_score", 0.5)

    X_fusion = np.array([[image_tamper_prob, metadata_anomaly_score]])
    ehi = fusion_model.predict_proba(X_fusion)[0][1]

    return float(image_tamper_prob), float(metadata_anomaly_score), float(ehi)


def cleanup_temp_file(temp_path):
    try:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
    except OSError:
        pass


# ============================================================================
# FORCED SYNTHETIC CUSTODY GENERATION
# ============================================================================

def fake_hash(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def generate_case_graph_forced(case_id: str, anomaly_choice: str, seed: int = 42) -> pd.DataFrame:
    """anomaly_choice: 'none' | 'hash_mismatch' | 'timestamp_violation' | 'missing_custodian'"""
    rng = random.Random(f"{case_id}-{seed}")
    base_time = datetime(2018, 1, 1) + timedelta(days=rng.randint(0, 365 * 5))
    real_hash = fake_hash(f"{case_id}-original")

    nodes = []
    event_timestamps = {}
    current_time = base_time
    current_hash = real_hash
    parent_event_id = None

    for idx, (action, role) in enumerate(CUSTODY_ROLES_SEQUENCE):
        event_id = f"{case_id}_ev{idx}"
        actor = rng.choice(CUSTODY_ACTORS)
        is_anomalous_event = (idx == FORCED_ANOMALY_EVENT_IDX) and (anomaly_choice != "none")

        current_time = current_time + timedelta(hours=rng.randint(2, 72))
        this_timestamp = current_time
        this_hash = current_hash
        this_actor = actor
        anomaly_here = "none"

        if is_anomalous_event:
            if anomaly_choice == "hash_mismatch":
                this_hash = fake_hash(f"{case_id}-tampered-{idx}")
                current_hash = this_hash
                anomaly_here = "hash_mismatch"
            elif anomaly_choice == "timestamp_violation":
                parent_timestamp = event_timestamps.get(parent_event_id, base_time)
                this_timestamp = parent_timestamp - timedelta(hours=rng.randint(1, 100))
                anomaly_here = "timestamp_violation"
            elif anomaly_choice == "missing_custodian":
                this_actor = None
                anomaly_here = "missing_custodian"

        nodes.append({
            "case_id": case_id, "event_id": event_id, "parent_event_id": parent_event_id,
            "actor": this_actor, "role": role, "action": action,
            "timestamp": this_timestamp.isoformat(), "sha256_hash": this_hash,
            "anomaly_injected": anomaly_here,
        })
        event_timestamps[event_id] = this_timestamp
        parent_event_id = event_id

    return pd.DataFrame(nodes)


def compute_custody_metrics(case_id: str, anomaly_choice: str):
    nodes_df = generate_case_graph_forced(case_id, anomaly_choice)
    g = build_case_graph(nodes_df)
    metrics = analyze_case_graph(g)
    metrics["custody_integrity_score"] = round(1.0 - metrics["custody_risk_score"], 4)
    return nodes_df, metrics


# ============================================================================
# FORCED SYNTHETIC REPORT GENERATION
# ============================================================================

def random_wrong_date(real_date_str: str, seed_rng: random.Random) -> str:
    try:
        base = datetime.strptime(real_date_str[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        base = datetime(2015, 1, 1)
    offset_days = seed_rng.choice([-1, 1]) * seed_rng.randint(30, 1000)
    return (base + timedelta(days=offset_days)).strftime("%Y-%m-%d")


def generate_report_forced(case_id: str, real_camera: str, real_date: str, contradiction_choice: str, seed: int = 42):
    """contradiction_choice: 'none' | 'camera' | 'date' | 'both'"""
    rng = random.Random(f"{case_id}-report-{seed}")

    claimed_camera = real_camera
    claimed_date = real_date

    if contradiction_choice in ("camera", "both"):
        distractors = [c for c in CAMERA_POOL if c.lower() != real_camera.lower()]
        claimed_camera = rng.choice(distractors)
    if contradiction_choice in ("date", "both"):
        claimed_date = random_wrong_date(real_date, rng)

    report_text = REPORT_TEMPLATE.format(camera=claimed_camera, date=claimed_date)
    return {
        "report_text": report_text,
        "claimed_camera": claimed_camera,
        "claimed_date": claimed_date,
        "contradiction_choice": contradiction_choice,
    }


def compute_report_metrics(case_id: str, real_camera: str, real_date: str, contradiction_choice: str):
    report = generate_report_forced(case_id, real_camera, real_date, contradiction_choice)
    score = consistency_score_row(report["claimed_camera"], report["claimed_date"], real_camera, real_date)
    score["report_evidence_consistency"] = score["consistency_score"]
    return report, score


# ============================================================================
# VISUAL HELPERS
# ============================================================================

def metric_card(label, value, icon="", synthetic=False):
    cls = "metric-card synthetic-card" if synthetic else "metric-card"
    tag = '<span class="synthetic-pill">SYNTHETIC</span>' if synthetic else ""
    st.markdown(f"""
        <div class="{cls}">
            <div class="metric-card-label">{icon} {label}{tag}</div>
            <div class="metric-card-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)


def render_custody_timeline(nodes_df):
    fig = go.Figure()
    x = list(range(len(nodes_df)))
    colors = ["#ee5a6f" if a != "none" else "#11998e" for a in nodes_df["anomaly_injected"]]

    fig.add_trace(go.Scatter(
        x=x, y=[1] * len(x), mode="markers+text",
        marker=dict(size=26, color=colors),
        text=nodes_df["role"], textposition="top center",
        showlegend=False,
    ))
    for i in range(len(x) - 1):
        fig.add_annotation(x=x[i + 1], y=1, ax=x[i], ay=1, xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True, arrowhead=3, arrowcolor="#adb5bd")
    fig.update_layout(
        height=180, plot_bgcolor="white",
        xaxis=dict(visible=False, range=[-0.5, len(x) - 0.5]),
        yaxis=dict(visible=False, range=[0, 2]),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def render_case_panel(case_row, case_label, case_key):
    case_id = case_row["case_id"]
    image_path = case_row["image_path"]
    real_camera = str(case_row["camera_model"]).strip()
    real_date = clean_exif_date(str(case_row["datetime_original"]))
    image_tamper_prob = case_row["image_tamper_prob"]
    metadata_anomaly_score = case_row["metadata_anomaly_score"]
    ehi = case_row["evidence_health_index"]

    col_img, col_scores = st.columns([1, 1])

    with col_img:
        try:
            img = Image.open(image_path)
            st.image(img, caption=f"{case_label}: {case_id}", use_container_width=True)
        except Exception:
            st.warning(f"Could not load image at {image_path}")
        st.markdown(
            f'<span class="info-pill">Camera: {real_camera}</span>'
            f'<span class="info-pill">Date: {real_date}</span>',
            unsafe_allow_html=True,
        )

    with col_scores:
        is_flagged = ehi > 0.387
        verdict_class = "verdict-flagged" if is_flagged else "verdict-authentic"
        verdict_text = "🔴 FLAGGED" if is_flagged else "🟢 AUTHENTIC"
        st.markdown(f"""
            <div class="verdict-card {verdict_class}">
                <p class="verdict-title">{verdict_text} (Tier 1 real-data verdict)</p>
                <p class="verdict-subtitle">EHI {ehi:.3f} vs threshold 0.387</p>
            </div>
        """, unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        with m1:
            metric_card("Image Tamper Prob", f"{image_tamper_prob:.3f}", "📸")
        with m2:
            metric_card("Metadata Anomaly", f"{metadata_anomaly_score:.3f}", "🏷️")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ---- Interactive synthetic layer ----
    st.markdown("#### 🔗 Synthetic Chain of Custody")
    custody_col1, custody_col2 = st.columns([1, 2])
    with custody_col1:
        anomaly_choice = st.selectbox(
            "Inject custody anomaly:",
            ["none", "hash_mismatch", "timestamp_violation", "missing_custodian"],
            key=f"custody_{case_key}",
        )
    nodes_df, custody_metrics = compute_custody_metrics(case_id, anomaly_choice)

    with custody_col2:
        st.plotly_chart(render_custody_timeline(nodes_df), use_container_width=True, key=f"custody_timeline_{case_key}")

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        metric_card("Custody Risk Score", f"{custody_metrics['custody_risk_score']:.3f}", "⚠️")
    with cc2:
        metric_card("Custody Integrity Score", f"{custody_metrics['custody_integrity_score']:.3f}", "🛡️", synthetic=True)
    with cc3:
        flag_text = "YES" if custody_metrics["flagged_custody_anomaly"] else "NO"
        metric_card("Flagged Anomalous?", flag_text, "🚩")

    if anomaly_choice != "none":
        st.caption(
            "This anomaly's score and flagged role are fixed by anomaly TYPE (rule-based scoring, "
            "same on every case) — not derived from this specific case. See the About tab."
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ---- Interactive synthetic report ----
    st.markdown("#### 📝 Synthetic Forensic Report")
    report_col1, report_col2 = st.columns([1, 2])
    with report_col1:
        contradiction_choice = st.selectbox(
            "Inject report contradiction:",
            ["none", "camera", "date", "both"],
            key=f"report_{case_key}",
        )
    report, consistency_score = compute_report_metrics(case_id, real_camera, real_date, contradiction_choice)

    with report_col2:
        st.markdown(f'<div class="report-box">"{report["report_text"]}"</div>', unsafe_allow_html=True)
        st.caption(f"Ground truth: {real_camera} · {real_date}")

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        metric_card("Camera Match", "✅ Yes" if consistency_score["camera_match"] else "❌ No", "📷")
    with rc2:
        metric_card("Date Match", "✅ Yes" if consistency_score["date_match"] else "❌ No", "📅")
    with rc3:
        metric_card("Report-Evidence Consistency", f"{consistency_score['report_evidence_consistency']:.3f}", "🧾", synthetic=True)

    return {
        "image_tamper_prob": image_tamper_prob,
        "metadata_anomaly_score": metadata_anomaly_score,
        "ehi": ehi,
        "custody_risk_score": custody_metrics["custody_risk_score"],
        "custody_integrity_score": custody_metrics["custody_integrity_score"],
        "report_evidence_consistency": consistency_score["report_evidence_consistency"],
    }


# ============================================================================
# UI LAYOUT
# ============================================================================

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown("""
        <div class="hero-header">
            <p class="project-badge">CHAIN-X · TIER 2</p>
            <p class="main-header">🧪 Proof-of-Concept Demo</p>
            <p class="sub-header">Image + Metadata + Synthetic Custody + Synthetic NLP Reports</p>
        </div>
        <div class="poc-banner">
            ⚠️ <b>This dashboard uses synthetic custody and report data</b> to demonstrate what the
            system would show if real chain-of-custody and forensic-report data were available.
            It is a proof-of-concept only and is <b>not included in the research paper's reported results</b>.
        </div>
    """, unsafe_allow_html=True)

    with st.spinner("Selecting demo cases..."):
        auth_case, tamp_case = select_demo_cases()

    tab_authentic, tab_tampered, tab_upload, tab_compare, tab_about = st.tabs([
        "🟢  Authentic Case", "🔴  Tampered Case", "📤  Upload Your Own",
        "⚖️  Side-by-Side", "ℹ️  About This Demo",
    ])

    with tab_authentic:
        auth_results = render_case_panel(auth_case, "Authentic Case", "authentic")

    with tab_tampered:
        tamp_results = render_case_panel(tamp_case, "Tampered Case", "tampered")

    with tab_upload:
        st.subheader("Upload an image to run the full pipeline")
        st.caption(
            "Image tamper probability, metadata anomaly score and EHI below are computed live from "
            "your upload using the real Tier 1 models. The chain of custody and forensic report are "
            "still generated synthetically — an uploaded image has no real case file to draw either from."
        )
        uploaded_file = st.file_uploader(
            "Choose an image (JPG, PNG)", type=["jpg", "jpeg", "png"], key="tier2_uploader"
        )
        if uploaded_file:
            with st.spinner("Running the Tier 1 pipeline on your upload..."):
                rf_model, fusion_model, fusion_threshold, norm_stats = load_tier1_models()
                proc = process_uploaded_image(uploaded_file)
                image_tamper_prob, metadata_anomaly_score, ehi = predict_tier1_scores(
                    rf_model, fusion_model, proc["ela_features"], proc["metadata_features"]
                )

            camera_model = proc["metadata_raw"].get("Make") or proc["metadata_raw"].get("Model") or "Unknown"
            datetime_original = proc["metadata_raw"].get("DateTimeOriginal", "")
            upload_case_id = f"upload_{uploaded_file.name}"

            fake_case_row = pd.Series({
                "case_id": upload_case_id,
                "image_path": proc["temp_path"],
                "camera_model": camera_model,
                "datetime_original": datetime_original,
                "image_tamper_prob": image_tamper_prob,
                "metadata_anomaly_score": metadata_anomaly_score,
                "evidence_health_index": ehi,
            })

            render_case_panel(fake_case_row, "Uploaded Case", "uploaded")
            cleanup_temp_file(proc["temp_path"])
        else:
            st.info("👆 Upload an image to see the full pipeline — real detection + synthetic custody/report — run on it.")

    with tab_compare:
        st.subheader("Side-by-Side: All 5 Trust Metrics")
        compare_df = pd.DataFrame({
            "Metric": ["Image Tamper Prob", "Metadata Anomaly", "EHI (Tier 1)",
                       "Custody Integrity (synthetic)", "Report Consistency (synthetic)"],
            "Authentic Case": [
                auth_results["image_tamper_prob"], auth_results["metadata_anomaly_score"],
                auth_results["ehi"], auth_results["custody_integrity_score"],
                auth_results["report_evidence_consistency"],
            ],
            "Tampered Case": [
                tamp_results["image_tamper_prob"], tamp_results["metadata_anomaly_score"],
                tamp_results["ehi"], tamp_results["custody_integrity_score"],
                tamp_results["report_evidence_consistency"],
            ],
        })
        compare_long = compare_df.melt(id_vars="Metric", var_name="Case", value_name="Score")
        fig_compare = px.bar(compare_long, x="Metric", y="Score", color="Case", barmode="group",
                              color_discrete_sequence=["#11998e", "#ee5a6f"], text="Score")
        fig_compare.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_compare.update_layout(height=420, plot_bgcolor="white", yaxis_range=[0, 1.05],
                                   xaxis_tickangle=-15)
        st.plotly_chart(fig_compare, use_container_width=True, key="side_by_side_comparison")
        st.caption("Adjust the custody/report dropdowns in the case tabs above, then revisit this "
                   "chart — it updates live with your chosen anomaly/contradiction injections.")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("Detection Performance (validated on full 3302-case dataset)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔗 Custody Anomaly Detection**")
            m1, m2, m3, m4 = st.columns(4)
            with m1: metric_card("Precision", f"{CUSTODY_DETECTION_METRICS['precision']:.3f}")
            with m2: metric_card("Recall", f"{CUSTODY_DETECTION_METRICS['recall']:.3f}")
            with m3: metric_card("F1", f"{CUSTODY_DETECTION_METRICS['f1']:.3f}")
            with m4: metric_card("Accuracy", f"{CUSTODY_DETECTION_METRICS['accuracy']:.3f}")
            st.caption(f"n={CUSTODY_DETECTION_METRICS['n_cases']} synthetic cases, "
                       f"{CUSTODY_DETECTION_METRICS['flagged_pct']}% flagged")
        with col2:
            st.markdown("**📝 Report Contradiction Detection**")
            m1, m2, m3, m4 = st.columns(4)
            with m1: metric_card("Precision", f"{NLP_DETECTION_METRICS['precision']:.3f}")
            with m2: metric_card("Recall", f"{NLP_DETECTION_METRICS['recall']:.3f}")
            with m3: metric_card("F1", f"{NLP_DETECTION_METRICS['f1']:.3f}")
            with m4: metric_card("Accuracy", f"{NLP_DETECTION_METRICS['accuracy']:.3f}")
            st.caption(f"n={NLP_DETECTION_METRICS['n_evaluated']} cases with real EXIF ground truth")

    with tab_about:
        st.markdown("""
            <div class="poc-banner" style="margin-bottom: 1.5rem;">
                This entire dashboard is scoped as evaluation-only proof-of-concept material
                and will be removed from the repository before paper submission, per project scope.
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        ### What's real vs. synthetic here?

        | Signal | Source | Status |
        |---|---|---|
        | Image tamper probability | RF classifier on real ELA features | **Real** |
        | Metadata anomaly score | Real EXIF analysis | **Real** |
        | Evidence Health Index (EHI) | Fusion of the two real signals above | **Real** (same as Tier 1) |
        | Chain of custody | Simulated per Phase 6 methodology | **Synthetic** — CASIA has no real custody metadata |
        | Forensic report text | Simulated per Phase 4 methodology | **Synthetic** — CASIA has no real case reports |
        | Custody Integrity Score | `1 − custody_risk_score` | **Synthetic-derived** |
        | Report-Evidence Consistency | Camera/date claim vs. real EXIF | **Synthetic-derived** |

        ### Why simulate custody and reports at all?

        Real forensic evidence has a chain of custody and human-authored reports — CASIA2 (and most
        public tampering-detection datasets) has neither. This demo shows how the same trust-scoring
        framework would extend to a real deployment where that data exists, and validates that the
        detection logic itself (custody anomaly detection, report-contradiction detection) works
        correctly against controlled, injected ground truth — precision/recall of 1.0 across most
        anomaly types, as shown in the Side-by-Side tab.

        ### How the interactive toggles work

        - **Custody anomaly dropdown**: regenerates a synthetic 5-event custody chain for the selected
          case, optionally injecting a hash break, timestamp violation, or missing custodian at a fixed
          point in the chain, then scores it with the *exact same* `graph_analyzer.py` logic used to
          validate detection across all 3302 cases. Because the injection point and the rule-based
          scoring are fixed **per anomaly type**, the resulting risk score and flagged role are the same
          regardless of which case you're viewing — this demonstrates the detection logic working
          correctly against known, injected ground truth, not case-specific severity.
        - **Report contradiction dropdown**: regenerates a synthetic forensic report claiming a camera
          and date, optionally corrupting one or both against the case's real EXIF ground truth, then
          scores it with the *exact same* `consistency.py` logic from Phase 4.

        ### Upload Your Own tab

        Runs the real, trained Tier 1 models (`rf_final_locked_in.pkl` + `fusion_v1.pkl`) on whatever
        image you upload — the image tamper probability, metadata anomaly score and EHI shown there are
        genuinely computed, not synthetic. The chain of custody and forensic report are generated fresh
        for the upload, since it has no real case file to draw either from.
        """)


if __name__ == "__main__":
    main()
