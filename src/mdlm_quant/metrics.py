"""Statistical summaries, paired tests, bootstrap intervals, and correlations."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np


PRIMARY_METRICS = (
    "masked_cross_entropy",
    "masked_token_accuracy",
    "sequence_reconstruction_accuracy",
    "reference_token_agreement",
    "kl_divergence_from_reference",
    "final_logit_mse",
)


def _mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    return float(np.mean(arr)) if arr.size else 0.0


def _std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    return float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def bootstrap_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    """Return a bootstrap 95% confidence interval for the mean."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        draw = rng.choice(arr, size=arr.size, replace=True)
        means[i] = np.mean(draw)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_evaluation(rows: list[dict[str, float | int | str]], bootstrap_samples: int, seed: int) -> list[dict[str, float | int | str]]:
    """Summarize evaluation rows by method and metric."""

    grouped: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    out: list[dict[str, float | int | str]] = []
    for method, method_rows in grouped.items():
        for metric in PRIMARY_METRICS:
            values = [float(row[metric]) for row in method_rows]
            low, high = bootstrap_ci(values, bootstrap_samples, seed + len(out))
            out.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": _mean(values),
                    "std": _std(values),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "n": len(values),
                    "average_bit_width": _mean(float(row["average_bit_width"]) for row in method_rows),
                    "total_bit_budget": int(round(_mean(float(row["total_bit_budget"]) for row in method_rows))),
                    "predicted_gronwall_objective_p2": _mean(float(row["predicted_gronwall_objective_p2"]) for row in method_rows),
                }
            )
    return out


def _normal_two_sided_p(z: float) -> float:
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def paired_tests(
    rows: list[dict[str, float | int | str]],
    reference_method: str,
    metrics: tuple[str, ...] = ("final_logit_mse", "masked_token_accuracy"),
) -> list[dict[str, float | int | str]]:
    """Compute paired normal-approximation tests against a reference method."""

    by_key: dict[tuple[str, int, int], dict[str, float | int | str]] = {}
    methods: set[str] = set()
    for row in rows:
        method = str(row["method"])
        methods.add(method)
        by_key[(method, int(row["seed"]), int(row["sample_index"]))] = row
    out: list[dict[str, float | int | str]] = []
    for method in sorted(methods):
        if method == reference_method:
            continue
        for metric in metrics:
            diffs: list[float] = []
            for key, ref_row in by_key.items():
                if key[0] != reference_method:
                    continue
                other = by_key.get((method, key[1], key[2]))
                if other is not None:
                    diffs.append(float(other[metric]) - float(ref_row[metric]))
            arr = np.asarray(diffs, dtype=np.float64)
            if arr.size < 2:
                z = 0.0
                p_value = float("nan")
                mean_diff = float(np.mean(arr)) if arr.size else 0.0
                status = "insufficient_pairs"
            else:
                mean_diff = float(np.mean(arr))
                std = float(np.std(arr, ddof=1))
                if std == 0.0:
                    if mean_diff == 0.0:
                        z = 0.0
                        p_value = 1.0
                        status = "no_difference"
                    else:
                        z = math.copysign(float("inf"), mean_diff)
                        p_value = float("nan")
                        status = "constant_nonzero_difference_normal_test_undefined"
                else:
                    stderr = std / math.sqrt(arr.size)
                    z = mean_diff / stderr
                    p_value = _normal_two_sided_p(z)
                    status = "normal_approximation"
            out.append(
                {
                    "reference_method": reference_method,
                    "method": method,
                    "metric": metric,
                    "mean_paired_difference": mean_diff,
                    "z_statistic_normal_approx": z,
                    "p_value_normal_approx": p_value,
                    "n_pairs": int(arr.size),
                    "test_status": status,
                }
            )
    return out


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def correlation_analysis(summary_rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    """Correlate predicted objectives with observed errors across methods."""

    error_rows = [row for row in summary_rows if row["metric"] == "final_logit_mse" and row["method"] != "unquantized_reference"]
    predicted = np.asarray([float(row["predicted_gronwall_objective_p2"]) for row in error_rows], dtype=np.float64)
    observed = np.asarray([float(row["mean"]) for row in error_rows], dtype=np.float64)
    pearson = _pearson(predicted, observed)
    spearman = _pearson(_rankdata(predicted), _rankdata(observed))
    return [
        {
            "x": "predicted_gronwall_objective_p2",
            "y": "final_logit_mse_mean",
            "correlation": "pearson",
            "value": pearson,
            "n_methods": len(error_rows),
            "scope": "quantized_methods_only",
            "units_note": "activation-calibrated score compared to observed logit MSE by correlation, not as a certified numeric upper bound",
        },
        {
            "x": "predicted_gronwall_objective_p2",
            "y": "final_logit_mse_mean",
            "correlation": "spearman",
            "value": spearman,
            "n_methods": len(error_rows),
            "scope": "quantized_methods_only",
            "units_note": "activation-calibrated score compared to observed logit MSE by correlation, not as a certified numeric upper bound",
        },
    ]
