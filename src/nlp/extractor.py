"""
src/nlp/extractor.py
Extracts the claimed camera model and claimed acquisition date from
template-generated report text (see report_generator.py).

Report format example:
"The image was acquired using a Canon EOS 5D on 2015-06-12."

Since reports are template-generated (not free-form), simple regex is
sufficient and more reliable than an NLP model here.
"""

import re
import argparse
import pandas as pd

# Matches "using <camera model> on <date>"
CAMERA_DATE_RE = re.compile(
    r"using\s+(?P<camera>.+?)\s+on\s+(?P<date>\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)

# Fallback patterns in case template phrasing varies slightly
CAMERA_ONLY_RE = re.compile(r"using\s+(?P<camera>.+?)(?:\.|,|$)", re.IGNORECASE)
DATE_ONLY_RE = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})")


def extract_claims(report_text: str) -> dict:
    """Extract claimed camera and date from a single report string.

    Returns dict with keys: claimed_camera, claimed_date
    (None if not found).
    """
    if not isinstance(report_text, str) or not report_text.strip():
        return {"claimed_camera": None, "claimed_date": None}

    match = CAMERA_DATE_RE.search(report_text)
    if match:
        return {
            "claimed_camera": match.group("camera").strip(),
            "claimed_date": match.group("date").strip(),
        }

    # Fallback: try camera and date separately
    camera_match = CAMERA_ONLY_RE.search(report_text)
    date_match = DATE_ONLY_RE.search(report_text)
    return {
        "claimed_camera": camera_match.group("camera").strip() if camera_match else None,
        "claimed_date": date_match.group("date").strip() if date_match else None,
    }


def extract_from_csv(reports_csv: str, text_col: str = "report_text") -> pd.DataFrame:
    """Batch-extract claims from a CSV of generated reports.

    Expects a column with the report text (default 'report_text') and
    a 'filename' or 'image_id' column to key back to master_index.csv.
    """
    df = pd.read_csv(reports_csv)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in {reports_csv}. "
                          f"Available columns: {list(df.columns)}")

    extracted = df[text_col].apply(extract_claims).apply(pd.Series)
    return pd.concat([df, extracted], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Extract claimed camera/date from report text")
    parser.add_argument("--reports-csv", required=True, help="CSV produced by report_generator.py")
    parser.add_argument("--text-col", default="report_text")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    result = extract_from_csv(args.reports_csv, args.text_col)
    result.to_csv(args.out, index=False)

    n_missing_camera = result["claimed_camera"].isna().sum()
    n_missing_date = result["claimed_date"].isna().sum()
    print(f"Extracted {len(result)} rows -> {args.out}")
    print(f"  Missing camera: {n_missing_camera} | Missing date: {n_missing_date}")


if __name__ == "__main__":
    main()
