"""Verify generated quick/research outputs are CSV-backed and provenance-linked."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


EXPECTED_CSVS = {
    "training_log.csv",
    "quantization_mse.csv",
    "quantization_fits.csv",
    "amplification_raw.csv",
    "amplification_summary.csv",
    "gronwall_factors.csv",
    "continuous_allocation.csv",
    "schedules.csv",
    "evaluation_raw.csv",
    "evaluation_summary.csv",
    "paired_tests.csv",
    "correlations.csv",
    "conclusion_provenance.csv",
}

EXPECTED_FIGURES = {
    "figure_01_calculus_function_derivative.png",
    "figure_02_tangent_line.png",
    "figure_03_continuous_gronwall.png",
    "figure_04_per_timestep_C_L_G.png",
    "figure_05_predicted_vs_observed.png",
    "figure_06_accuracy_vs_average_bits.png",
    "figure_07_final_error_comparison.png",
    "figure_08_bit_allocations.png",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify_output_root(root: Path) -> None:
    csv_dir = root / "csv"
    fig_dir = root / "figures"
    missing_csvs = sorted(name for name in EXPECTED_CSVS if not (csv_dir / name).exists())
    missing_figs = sorted(name for name in EXPECTED_FIGURES if not (fig_dir / name).exists())
    if missing_csvs:
        raise AssertionError(f"missing CSV outputs: {missing_csvs}")
    if missing_figs:
        raise AssertionError(f"missing figure outputs: {missing_figs}")

    schedules = _read_csv(csv_dir / "schedules.csv")
    budgets: dict[str, int] = defaultdict(int)
    for row in schedules:
        budgets[row["method"]] += int(row["bits"])
    mixed = {method: budget for method, budget in budgets.items() if method.endswith("budget6")}
    if mixed and len(set(mixed.values())) != 1:
        raise AssertionError(f"mixed precision budgets differ: {mixed}")

    eval_rows = _read_csv(csv_dir / "evaluation_raw.csv")
    stale_columns = {"fp16_token_agreement", "kl_divergence_from_fp16", "predicted_gronwall_bound"}
    if eval_rows and stale_columns.intersection(eval_rows[0]):
        raise AssertionError(f"stale/misleading columns remain: {stale_columns.intersection(eval_rows[0])}")

    conclusions = _read_csv(csv_dir / "conclusion_provenance.csv")
    if not conclusions:
        raise AssertionError("conclusion_provenance.csv is empty")
    allowed_status = {"supported", "rejected", "inconclusive"}
    for row in conclusions:
        if row["status"] not in allowed_status:
            raise AssertionError(f"invalid conclusion status: {row}")
        if not row["evidence"].strip():
            raise AssertionError(f"empty conclusion evidence: {row}")
        for source in row["source_csv"].split(";"):
            if not (csv_dir / source).exists():
                raise AssertionError(f"conclusion source CSV missing: {source}")

    report = root / "research_summary.md"
    if not report.exists():
        raise AssertionError("research_summary.md is missing")
    report_text = report.read_text(encoding="utf-8")
    for row in conclusions:
        if row["hypothesis"] not in report_text or row["status"] not in report_text:
            raise AssertionError(f"report does not mention conclusion row: {row}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify generated experiment outputs.")
    parser.add_argument("--output-root", default="outputs/quick")
    args = parser.parse_args()
    verify_output_root(Path(args.output_root))
    print(f"Verified outputs under {args.output_root}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        raise
