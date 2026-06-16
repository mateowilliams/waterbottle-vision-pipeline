"""Production model monitoring and drift detection.

Reads from PostgreSQL (predictions / labels tables) and builds a report with:
  - prediction volume and a daily time series
  - predicted-class distribution
  - confidence stats (mean, low-confidence rate, histogram)
  - real performance (accuracy and per class) wherever labels exist
  - drift vs. a reference (baseline) using PSI (Population Stability Index)

PSI compares two distributions (reference vs. production):
  PSI = Σ (a_i - e_i) * ln(a_i / e_i)
Common interpretation:
  < 0.10  -> no relevant drift
  0.10-0.25 -> moderate drift (watch)
  > 0.25  -> significant drift (consider retraining)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import psycopg

NUM_BINS = 10  # confidence histogram bins over [0, 1]


# --------------------------------------------------------------------------
# Statistical metrics
# --------------------------------------------------------------------------
def population_stability_index(
    expected: List[float], actual: List[float], eps: float = 1e-6
) -> float:
    """PSI between two distributions given as bin-aligned proportions."""
    total = 0.0
    for e, a in zip(expected, actual):
        e = max(e, eps)
        a = max(a, eps)
        total += (a - e) * math.log(a / e)
    return round(total, 4)


def psi_severity(value: float) -> str:
    if value < 0.10:
        return "no relevant drift"
    if value < 0.25:
        return "moderate drift"
    return "significant drift"


def confidence_histogram(values: List[float], num_bins: int = NUM_BINS) -> List[float]:
    """Histogram of confidences over [0,1] returned as per-bin proportions."""
    counts = [0] * num_bins
    for v in values:
        idx = min(int(v * num_bins), num_bins - 1)
        if idx < 0:
            idx = 0
        counts[idx] += 1
    n = len(values)
    if n == 0:
        return [0.0] * num_bins
    return [c / n for c in counts]


# --------------------------------------------------------------------------
# Database queries
# --------------------------------------------------------------------------
def _where(model_version: Optional[str]):
    if model_version:
        return "WHERE model_version = %s", [model_version]
    return "", []


def build_report(
    conn: psycopg.Connection,
    reference: Optional[Dict[str, Any]] = None,
    low_conf_threshold: float = 0.60,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    where, params = _where(model_version)
    report: Dict[str, Any] = {
        "model_version_filter": model_version,
        "low_conf_threshold": low_conf_threshold,
    }

    with conn.cursor() as cur:
        # --- summary ---
        cur.execute(
            f"SELECT count(*), min(created_at), max(created_at) FROM predictions {where};",
            params,
        )
        total, first_ts, last_ts = cur.fetchone()
        report["summary"] = {
            "total_predictions": int(total or 0),
            "first_prediction_at": first_ts.isoformat() if first_ts else None,
            "last_prediction_at": last_ts.isoformat() if last_ts else None,
        }

        # --- models seen ---
        cur.execute(
            "SELECT model_version, count(*) FROM predictions "
            "GROUP BY model_version ORDER BY count(*) DESC;"
        )
        report["models_seen"] = [{"model_version": m, "n": int(n)} for m, n in cur.fetchall()]

        # --- predicted-class distribution ---
        cur.execute(
            f"SELECT pred_label, count(*) FROM predictions {where} GROUP BY pred_label;",
            params,
        )
        class_counts = {label: int(n) for label, n in cur.fetchall()}
        denom = sum(class_counts.values()) or 1
        report["class_distribution"] = {
            "counts": class_counts,
            "proportions": {k: round(v / denom, 4) for k, v in class_counts.items()},
        }

        # --- confidence stats ---
        cur.execute(f"SELECT confidence FROM predictions {where};", params)
        confidences = [float(r[0]) for r in cur.fetchall()]
        n_conf = len(confidences)
        low_conf = sum(1 for c in confidences if c < low_conf_threshold)
        live_hist = confidence_histogram(confidences)
        report["confidence"] = {
            "mean": round(sum(confidences) / n_conf, 4) if n_conf else None,
            "min": round(min(confidences), 4) if n_conf else None,
            "max": round(max(confidences), 4) if n_conf else None,
            "low_confidence_rate": round(low_conf / n_conf, 4) if n_conf else None,
            "histogram": live_hist,
        }

        # --- daily time series ---
        cur.execute(
            f"""
            SELECT to_char(date_trunc('day', created_at), 'YYYY-MM-DD') AS d,
                   count(*), avg(confidence)
            FROM predictions {where}
            GROUP BY d ORDER BY d;
            """,
            params,
        )
        report["volume_daily"] = [
            {"date": d, "count": int(c), "avg_confidence": round(float(ac), 4)}
            for d, c, ac in cur.fetchall()
        ]

        # --- real performance (where labels exist) ---
        lwhere, lparams = ("WHERE p.model_version = %s", [model_version]) if model_version else ("", [])
        cur.execute(
            f"""
            SELECT count(*),
                   avg(CASE WHEN p.pred_label = l.true_label THEN 1.0 ELSE 0.0 END)
            FROM predictions p
            JOIN labels l ON l.image_id = p.image_id
            {lwhere};
            """,
            lparams,
        )
        n_labeled, acc = cur.fetchone()
        n_labeled = int(n_labeled or 0)
        per_class = []
        if n_labeled:
            cur.execute(
                f"""
                SELECT l.true_label, count(*),
                       sum(CASE WHEN p.pred_label = l.true_label THEN 1 ELSE 0 END)
                FROM predictions p
                JOIN labels l ON l.image_id = p.image_id
                {lwhere}
                GROUP BY l.true_label ORDER BY l.true_label;
                """,
                lparams,
            )
            for label, n, correct in cur.fetchall():
                per_class.append(
                    {
                        "label": label,
                        "n": int(n),
                        "correct": int(correct),
                        "recall": round(int(correct) / int(n), 4) if n else None,
                    }
                )
        report["performance"] = {
            "n_labeled": n_labeled,
            "accuracy": round(float(acc), 4) if acc is not None else None,
            "per_class": per_class,
        }

    # --- drift vs. reference ---
    report["drift"] = _drift_section(reference, report)
    return report


def _drift_section(reference: Optional[Dict[str, Any]], report: Dict[str, Any]) -> Dict[str, Any]:
    if not reference:
        return {"available": False, "reason": "No reference_stats.json (baseline) loaded."}

    out: Dict[str, Any] = {"available": True, "reference_model_version": reference.get("model_version")}

    # Class-distribution drift.
    ref_classes = reference.get("class_distribution", {})  # label -> proportion
    if ref_classes:
        labels = sorted(set(ref_classes) | set(report["class_distribution"]["proportions"]))
        expected = [ref_classes.get(lbl, 0.0) for lbl in labels]
        actual = [report["class_distribution"]["proportions"].get(lbl, 0.0) for lbl in labels]
        value = population_stability_index(expected, actual)
        out["class_distribution_psi"] = {
            "value": value,
            "severity": psi_severity(value),
            "labels": labels,
            "reference": expected,
            "live": actual,
        }

    # Confidence drift.
    ref_hist = reference.get("confidence_hist")
    if ref_hist:
        value = population_stability_index(ref_hist, report["confidence"]["histogram"])
        out["confidence_psi"] = {"value": value, "severity": psi_severity(value)}

    return out
