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

# Each of report_generator.py's REPORT_TEMPLATES needs its own pattern --
# a single generic regex can't reliably span "using a X on DATE" AND
# "Camera used: X. Date of capture: DATE" at once, since the connecting
# words differ. Tried in order; first match wins.
TEMPLATE_PATTERNS = [
    # "The image was acquired using a {camera} on {date}."
    re.compile(r"using\s+a\s+(?P<camera>.+?)\s+on\s+(?P<date>\d{4}-\d{2}-\d{2})", re.IGNORECASE),
    # "Evidence photo captured with {camera} on {date}."
    re.compile(r"captured\s+with\s+(?P<camera>.+?)\s+on\s+(?P<date>\d{4}-\d{2}-\d{2})", re.IGNORECASE),
    # "Camera used: {camera}. Date of capture: {date}."
    re.compile(r"camera\s+used:\s*(?P<camera>.+?)\.\s*date\s+of\s+capture:\s*(?P<date>\d{4}-\d{2}-\d{2})", re.IGNORECASE),
    # Generic fallback: any "using <camera> on <date>" without the "a"
    re.compile(r"using\s+(?P<camera>.+?)\s+on\s+(?P<date>\d{4}-\d{2}-\d{2})", re.IGNORECASE),
]

# Last-resort fallbacks if none of the template patterns match at all
# (e.g. free-form or unexpected phrasing) -- extract camera and date independently.
CAMERA_ONLY_RE = re.compile(
    r"(?:using\s+a|using|captured\s+with|camera\s+used:)\s+(?P<camera>.+?)(?:\.|,|\s+on\s+\d{4}|$)",
    re.IGNORECASE,
)
DATE_ONLY_RE = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})")


def extract_claims(report_text: str) -> dict:
    """Extract claimed camera and date from a single report string.

    Returns dict with keys: claimed_camera, claimed_date
    (None if not found).
    """
    if not isinstance(report_text, str) or not report_text.strip():
        return {"claimed_camera": None, "claimed_date": None}

    for pattern in TEMPLATE_PATTERNS:
        match = pattern.search(report_text)
        if match:
            return {
                "claimed_camera": match.group("camera").strip(),
                "claimed_date": match.group("date").strip(),
            }

    # None of the known templates matched -- fall back to independent extraction
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
