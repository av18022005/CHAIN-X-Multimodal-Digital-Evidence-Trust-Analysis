"""
src/custody/graph_generator.py
Phase 6, step 1: generates a synthetic chain-of-custody GRAPH per case_id.

CASIA has no real custody metadata (it's a research dataset of individual
images, not real forensic case files), so -- same principle as Phase 4's
synthetic reports -- this is EVALUATION-ONLY synthetic data. It exists to
test whether the custody-graph analyzer can catch injected integrity
violations, not to train anything.

Each case's custody graph models a realistic evidence lifecycle:
    Acquisition -> Storage -> Lab handoff -> Analysis -> Report
with a chance of BRANCHING (evidence copied to a second analyst/lab),
producing a real DAG rather than a flat linear chain.

A genuine evidence file's cryptographic hash should stay IDENTICAL across
every custody event -- that's the entire point of hashing evidence. Three
anomaly types are injected (~30% of cases, roughly evenly split):
    - hash_mismatch:      the hash changes between two custody events
                           (the file was altered while in someone's custody)
    - timestamp_violation: a later event is timestamped BEFORE its parent
                           (impossible in a real chain -- clock tampering
                           or a fabricated record)
    - missing_custodian:   an event has no actor recorded (a real chain
                           requires every handoff to name who received it)

Output (two CSVs, joined by case_id + event_id):
    custody_nodes.csv: case_id, event_id, parent_event_id, actor, role,
                        action, timestamp, sha256_hash, anomaly_injected
    custody_cases.csv: case_id, case_has_injected_anomaly (ground truth)

Usage:
    python -m src.custody.graph_generator \
        --master-csv /kaggle/working/data/master_index.csv \
        --out-nodes features/custody_nodes.csv \
        --out-cases features/custody_cases.csv
"""
import argparse
import hashlib
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ANOMALY_RATE = 0.30
BRANCH_PROBABILITY = 0.30  # chance evidence is copied to a second custodian

ACTORS = [
    "Officer R. Malik", "Officer S. Chen", "Evidence Custodian A. Diaz",
    "Forensic Analyst K. Osei", "Forensic Analyst T. Lindqvist",
    "Lab Technician M. Fischer", "Case Reviewer P. Novak",
]

ROLES_SEQUENCE = [
    ("acquisition", "Field Investigator"),
    ("intake_storage", "Evidence Custodian"),
    ("lab_handoff", "Forensic Analyst"),
    ("analysis", "Forensic Analyst"),
    ("report_filed", "Case Reviewer"),
]


def fake_hash(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def generate_case_graph(case_id: str, rng: random.Random) -> list[dict]:
    """Returns a list of node dicts for one case's custody graph."""
    base_time = datetime(2018, 1, 1) + timedelta(days=rng.randint(0, 365 * 5))
    real_hash = fake_hash(f"{case_id}-original")

    case_will_have_anomaly = rng.random() < ANOMALY_RATE
    # Pick which single event in the chain carries the injected anomaly
    n_main_events = len(ROLES_SEQUENCE)
    anomaly_event_idx = rng.randint(1, n_main_events - 1) if case_will_have_anomaly else -1
    anomaly_type = rng.choice(["hash_mismatch", "timestamp_violation", "missing_custodian"]) \
        if case_will_have_anomaly else None

    nodes = []
    current_time = base_time
    current_hash = real_hash
    parent_event_id = None

    for idx, (action, role) in enumerate(ROLES_SEQUENCE):
        event_id = f"{case_id}_ev{idx}"
        actor = rng.choice(ACTORS)
        is_anomalous_event = (idx == anomaly_event_idx)

        # Advance time normally (a realistic gap between custody events)
        current_time = current_time + timedelta(hours=rng.randint(2, 72))
        this_timestamp = current_time
        this_hash = current_hash
        this_actor = actor
        anomaly_here = "none"

        if is_anomalous_event:
            if anomaly_type == "hash_mismatch":
                this_hash = fake_hash(f"{case_id}-tampered-{idx}")
                current_hash = this_hash  # tampering persists downstream, as it would in reality
                anomaly_here = "hash_mismatch"
            elif anomaly_type == "timestamp_violation":
                this_timestamp = current_time - timedelta(hours=rng.randint(48, 200))
                anomaly_here = "timestamp_violation"
            elif anomaly_type == "missing_custodian":
                this_actor = None
                anomaly_here = "missing_custodian"

        nodes.append({
            "case_id": case_id,
            "event_id": event_id,
            "parent_event_id": parent_event_id,
            "actor": this_actor,
            "role": role,
            "action": action,
            "timestamp": this_timestamp.isoformat(),
            "sha256_hash": this_hash,
            "anomaly_injected": anomaly_here,
        })
        parent_event_id = event_id

        # Occasionally branch: evidence copied to a second custodian mid-chain
        if 0 < idx < n_main_events - 1 and rng.random() < BRANCH_PROBABILITY:
            branch_time = this_timestamp + timedelta(hours=rng.randint(1, 24))
            branch_event_id = f"{case_id}_ev{idx}_copy"
            nodes.append({
                "case_id": case_id,
                "event_id": branch_event_id,
                "parent_event_id": event_id,
                "actor": rng.choice(ACTORS),
                "role": "Secondary Analyst",
                "action": "copy_for_review",
                "timestamp": branch_time.isoformat(),
                "sha256_hash": this_hash,  # a legitimate copy keeps the same hash
                "anomaly_injected": "none",
            })

    return nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-csv", type=Path, required=True)
    ap.add_argument("--out-nodes", type=Path, required=True)
    ap.add_argument("--out-cases", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    master = pd.read_csv(args.master_csv)

    all_nodes = []
    case_rows = []
    for case_id in master["case_id"]:
        case_nodes = generate_case_graph(case_id, rng)
        all_nodes.extend(case_nodes)
        has_anomaly = any(n["anomaly_injected"] != "none" for n in case_nodes)
        case_rows.append({"case_id": case_id, "case_has_injected_anomaly": has_anomaly})

    nodes_df = pd.DataFrame(all_nodes)
    cases_df = pd.DataFrame(case_rows)

    args.out_nodes.parent.mkdir(parents=True, exist_ok=True)
    args.out_cases.parent.mkdir(parents=True, exist_ok=True)
    nodes_df.to_csv(args.out_nodes, index=False)
    cases_df.to_csv(args.out_cases, index=False)

    print(f"Generated custody graphs for {len(cases_df)} cases -> {args.out_nodes}, {args.out_cases}")
    print(f"Total events (incl. branches): {len(nodes_df)}")
    print(f"Cases with an injected anomaly: {cases_df['case_has_injected_anomaly'].sum()} "
          f"({cases_df['case_has_injected_anomaly'].mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
