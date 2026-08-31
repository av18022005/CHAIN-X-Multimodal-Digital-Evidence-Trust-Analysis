"""
Metadata anomaly scoring: converts raw extracted metadata (from
extractor.py) into a small set of interpretable forensic flags, and a
single weighted anomaly score in [0, 1].

Design principle: every flag here is individually explainable ("EXIF
missing", "editing software detected", "implausible timestamp") so that
Phase 8 (SHAP) and the dashboard can show *why* a case was flagged, not
just a black-box number.
"""
from datetime import datetime

# Weights are deliberately simple and interpretable (sum to 1.0), not
# learned — this is a rule-based forensic heuristic, not a trained model.
# Phase 7 (fusion) is where a learned model combines this score with the
# image/NLP/graph scores.
WEIGHTS = {
    "exif_missing": 0.40,
    "editing_software": 0.35,
    "implausible_timestamp": 0.15,
    "no_camera_info": 0.10,
}


def check_implausible_timestamp(datetime_original: str) -> bool:
    """Flags EXIF timestamps that are missing, malformed, in the future,
    or absurdly old (e.g. 0000:00:00 or before digital cameras existed)."""
    if not datetime_original or datetime_original.strip() == "":
        return False  # missing is already captured by exif_missing; don't double-count
    try:
        # EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
        dt = datetime.strptime(datetime_original, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return True  # malformed timestamp string is itself an anomaly

    now = datetime.now()
    earliest_plausible = datetime(1990, 1, 1)  # before consumer digital cameras
    return dt > now or dt < earliest_plausible


def compute_anomaly_flags(meta: dict) -> dict:
    exif_missing = not meta.get("exif_present", False)
    editing_software = bool(meta.get("editing_software_detected", False))
    implausible_ts = check_implausible_timestamp(meta.get("datetime_original", ""))
    no_camera_info = exif_missing is False and not meta.get("camera_make", "") and not meta.get("camera_model", "")

    return {
        "flag_exif_missing": exif_missing,
        "flag_editing_software": editing_software,
        "flag_implausible_timestamp": implausible_ts,
        "flag_no_camera_info": no_camera_info,
    }


def compute_anomaly_score(flags: dict) -> float:
    score = 0.0
    score += WEIGHTS["exif_missing"] * flags["flag_exif_missing"]
    score += WEIGHTS["editing_software"] * flags["flag_editing_software"]
    score += WEIGHTS["implausible_timestamp"] * flags["flag_implausible_timestamp"]
    score += WEIGHTS["no_camera_info"] * flags["flag_no_camera_info"]
    return round(score, 4)


def analyze(meta: dict) -> dict:
    flags = compute_anomaly_flags(meta)
    score = compute_anomaly_score(flags)
    return {**flags, "metadata_anomaly_score": score}
