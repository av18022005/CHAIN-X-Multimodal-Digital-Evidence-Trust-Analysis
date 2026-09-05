"""
src/custody/graph_analyzer.py
Phase 6, step 2: rebuilds each case's custody chain as a real networkx
DiGraph, walks every edge to detect integrity violations, and computes a
per-case Custody Risk Score. Then evaluates detection against the ground
truth injected by graph_generator.py.

Detection logic per edge (parent -> child):
    - hash_break:          child's hash != parent's hash
                            (evidence was altered while in custody)
    - timestamp_violation: child's timestamp <= parent's timestamp
                            (impossible in a real, honest chain)
    - missing_custodian:   child event has no actor recorded

custody_risk_score (per case) = weighted fraction of edges/nodes with an
issue, weighted by severity: a hash break is the most serious integrity
violation (evidence was actually altered), so it's weighted higher than
a missing-actor bookkeeping gap.

Usage:
    python -m src.custody.graph_analyzer \
        --nodes-csv features/custody_nodes.csv \
        --cases-csv features/custody_cases.csv \
        --out features/custody_risk_scores.csv
"""
import argparse
from pathlib import Path

import networkx as nx
import pandas as pd

# Relative severity weights -- a hash break means the evidence file itself
# was altered, which is strictly worse than a bookkeeping gap.
WEIGHT_HASH_BREAK = 0.6
WEIGHT_TIMESTAMP_VIOLATION = 0.25
WEIGHT_MISSING_CUSTODIAN = 0.15


def build_case_graph(case_nodes: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    for _, row in case_nodes.iterrows():
        g.add_node(row["event_id"], actor=row["actor"], role=row["role"],
                   action=row["action"], timestamp=pd.Timestamp(row["timestamp"]),
                   sha256_hash=row["sha256_hash"])
        if pd.notna(row["parent_event_id"]):
            g.add_edge(row["parent_event_id"], row["event_id"])
    return g


def analyze_case_graph(g: nx.DiGraph) -> dict:
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()

    hash_breaks = 0
    timestamp_violations = 0
    for parent, child in g.edges():
        if g.nodes[parent]["sha256_hash"] != g.nodes[child]["sha256_hash"]:
            hash_breaks += 1
        if g.nodes[child]["timestamp"] <= g.nodes[parent]["timestamp"]:
            timestamp_violations += 1

    missing_custodian = sum(
        1 for _, data in g.nodes(data=True)
        if data["actor"] is None or (isinstance(data["actor"], float) and pd.isna(data["actor"]))
    )

    branch_count = sum(1 for _, out_deg in g.out_degree() if out_deg > 1)

    edge_denom = max(n_edges, 1)
    node_denom = max(n_nodes, 1)
    custody_risk_score = (
        WEIGHT_HASH_BREAK * (hash_breaks / edge_denom)
        + WEIGHT_TIMESTAMP_VIOLATION * (timestamp_violations / edge_denom)
        + WEIGHT_MISSING_CUSTODIAN * (missing_custodian / node_denom)
    )

    # Flag on ANY detected issue, not a blended-score threshold. A single
    # missing custodian record or one hash break is a real, independently
    # meaningful chain-of-custody break -- diluting it into an averaged
    # score (and thresholding that) let real issues slip through purely
    # because the chain had many otherwise-clean events.
    flagged = (hash_breaks > 0) or (timestamp_violations > 0) or (missing_custodian > 0)

    return {
        "n_events": n_nodes,
        "n_handoffs": n_edges,
        "n_branches": branch_count,
        "hash_breaks": hash_breaks,
        "timestamp_violations": timestamp_violations,
        "missing_custodian": missing_custodian,
        "custody_risk_score": round(custody_risk_score, 4),
        "flagged_custody_anomaly": flagged,
    }


def compute_binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true, y_pred = y_true.astype(bool), y_pred.astype(bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "accuracy": round(accuracy, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes-csv", type=Path, required=True)
    ap.add_argument("--cases-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    nodes = pd.read_csv(args.nodes_csv)
    cases = pd.read_csv(args.cases_csv)

    results = []
    for case_id, case_nodes in nodes.groupby("case_id"):
        g = build_case_graph(case_nodes)
        metrics = analyze_case_graph(g)
        metrics["case_id"] = case_id
        results.append(metrics)

    results_df = pd.DataFrame(results)
    merged = results_df.merge(cases, on="case_id", how="left")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"Analyzed {len(merged)} custody graphs -> {args.out}")
    print(f"Flagged as anomalous: {merged['flagged_custody_anomaly'].sum()} "
          f"({merged['flagged_custody_anomaly'].mean()*100:.1f}%)")

    metrics = compute_binary_metrics(
        merged["case_has_injected_anomaly"], merged["flagged_custody_anomaly"]
    )
    print("\n--- Evaluation vs. injected ground truth ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n--- Detection breakdown by anomaly type ---")
    for anomaly_type in ["hash_mismatch", "timestamp_violation", "missing_custodian"]:
        type_cases = nodes[nodes["anomaly_injected"] == anomaly_type]["case_id"].unique()
        if len(type_cases) == 0:
            continue
        subset = merged[merged["case_id"].isin(type_cases)]
        caught = subset["flagged_custody_anomaly"].sum()
        print(f"  {anomaly_type}: {caught}/{len(subset)} caught ({caught/len(subset)*100:.1f}%)")


if __name__ == "__main__":
    main()
