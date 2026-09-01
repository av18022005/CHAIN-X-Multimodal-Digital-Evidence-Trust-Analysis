"""
src/nlp/report_generator.py
Generates controlled synthetic report sentences from real Phase 3 metadata
(metadata_features.csv), for evaluation of the NLP consistency checker.

This is EVALUATION-ONLY synthetic data (per project rules) -- not used
to train any model. It exists purely to test whether consistency.py
catches deliberately injected contradictions.

For rows with real camera_model + datetime_original (EXIF present),
~65% of generated reports state the TRUE values, ~35% deliberately
state a WRONG camera and/or date (a "contradiction").

For rows with no real EXIF (the majority in CASIA2), there's nothing
true to contradict -- a plausible-sounding claim is fabricated instead,
and ground_truth_available=False so evaluation correctly excludes these
rows from precision/recall (they can only ever score "unverifiable",
never a true/false contradiction).

Output columns:
    case_id, report_text, is_contradiction_injected, ground_truth_available
"""

import argparse
import random
from datetime import datetime, timedelta

import pandas as pd

CONTRADICTION_RATE = 0.35

# Fallback / distractor camera pool -- used both to fabricate claims for
# rows with no real EXIF, and to pick a "wrong" camera when injecting a
# contradiction for rows that DO have a real camera_model.
CAMERA_POOL = [
    "Canon EOS 5D", "Canon EOS 5D Mark II", "Canon PowerShot G12",
    "Nikon D90", "Nikon D7000", "Nikon COOLPIX P510",
    "Sony DSC-W800", "Sony Alpha a6000",
    "Apple iPhone 6", "Apple iPhone 11", "Samsung Galaxy S9",
    "Olympus E-M10", "Fujifilm FinePix S1",
]

REPORT_TEMPLATES = [
    "The image was acquired using a {camera} on {date}.",
    "Evidence photo captured with {camera} on {date}.",
    "Camera used: {camera}. Date of capture: {date}.",
]


def random_wrong_date(real_date_str: str) -> str:
    """Shift a real date by a random 30-1000 day offset (either direction)
    to produce a plausible-but-wrong date."""
    try:
        base = datetime.strptime(real_date_str[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        base = datetime(2015, 1, 1)
    offset_days = random.choice([-1, 1]) * random.randint(30, 1000)
    return (base + timedelta(days=offset_days)).strftime("%Y-%m-%d")


def fabricate_date() -> str:
    base = datetime(2010, 1, 1)
    offset_days = random.randint(0, 365 * 12)
    return (base + timedelta(days=offset_days)).strftime("%Y-%m-%d")


def clean_date(raw: str) -> str:
    """EXIF dates look like '2015:06:12 14:30:00' -- normalize to YYYY-MM-DD."""
    if not isinstance(raw, str) or not raw.strip():
        return ""
    s = raw.strip()
    date_part = s.split(" ")[0].replace(":", "-", 2)
    try:
        datetime.strptime(date_part, "%Y-%m-%d")
        return date_part
    except ValueError:
        return ""


def generate_report_for_row(row) -> dict:
    real_camera = str(row.get("camera_model", "") or "").strip()
    real_date = clean_date(row.get("datetime_original", ""))
    has_ground_truth = bool(real_camera) and bool(real_date)

    template = random.choice(REPORT_TEMPLATES)

    if has_ground_truth:
        inject_contradiction = random.random() < CONTRADICTION_RATE
        camera_corrupted = False
        date_corrupted = False
        if inject_contradiction:
            # Corrupt camera, date, or both (roughly evenly)
            corrupt_what = random.choice(["camera", "date", "both"])
            claimed_camera = real_camera
            claimed_date = real_date
            if corrupt_what in ("camera", "both"):
                distractors = [c for c in CAMERA_POOL if c.lower() != real_camera.lower()]
                claimed_camera = random.choice(distractors)
                camera_corrupted = True
            if corrupt_what in ("date", "both"):
                claimed_date = random_wrong_date(real_date)
                date_corrupted = True
        else:
            claimed_camera = real_camera
            claimed_date = real_date

        report_text = template.format(camera=claimed_camera, date=claimed_date)
        return {
            "report_text": report_text,
            "is_contradiction_injected": inject_contradiction,
            "camera_contradiction_injected": camera_corrupted,
            "date_contradiction_injected": date_corrupted,
            "ground_truth_available": True,
        }

    # No real EXIF to compare against -- fabricate a claim.
    claimed_camera = random.choice(CAMERA_POOL)
    claimed_date = fabricate_date()
    report_text = template.format(camera=claimed_camera, date=claimed_date)
    return {
        "report_text": report_text,
        "is_contradiction_injected": False,  # not meaningful here
        "camera_contradiction_injected": False,
        "date_contradiction_injected": False,
        "ground_truth_available": False,
    }


def generate_reports(metadata_csv: str, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    meta = pd.read_csv(metadata_csv)

    records = []
    for _, row in meta.iterrows():
        gen = generate_report_for_row(row)
        gen["case_id"] = row["case_id"]
        records.append(gen)

    out = pd.DataFrame(records)[[
        "case_id", "report_text", "is_contradiction_injected",
        "camera_contradiction_injected", "date_contradiction_injected",
        "ground_truth_available",
    ]]
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic evidence reports for Phase 4 evaluation")
    parser.add_argument("--metadata-csv", required=True, help="Phase 3 metadata_features.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    reports = generate_reports(args.metadata_csv, args.seed)
    reports.to_csv(args.out, index=False)

    n_with_gt = reports["ground_truth_available"].sum()
    n_injected = reports["is_contradiction_injected"].sum()
    print(f"Generated {len(reports)} reports -> {args.out}")
    print(f"  Rows with real ground truth (real EXIF): {n_with_gt} ({n_with_gt/len(reports)*100:.1f}%)")
    print(f"  Contradictions injected (of those with ground truth): {n_injected} "
          f"({n_injected/n_with_gt*100:.1f}% of GT rows)" if n_with_gt else "  N/A")


if __name__ == "__main__":
    main()
