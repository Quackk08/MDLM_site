"""End-to-end experiment pipeline."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

import numpy as np
import torch

from .calibration import run_calibration
from .config import ExperimentConfig, get_config, load_config
from .data import TextSplits, make_initial_mask, tokenize_splits
from .generation import build_reference_trajectories, evaluate_schedule_on_trajectories
from .metrics import correlation_analysis, paired_tests, summarize_evaluation
from .model import MaskedDiffusionTransformer
from .plotting import generate_all_figures
from .reporting import build_conclusion_rows, generate_summary_report
from .schedules import (
    Schedule,
    build_baseline_schedules,
    choose_empirical_oracle,
    make_schedule,
    schedules_to_rows,
)
from .tokenizer import ByteTokenizer
from .training import train_model
from .utils import ensure_dir, package_versions, parameter_count, resolve_device, set_seed, wall_time, write_csv, write_json


def _masked_eval_tokens(tokens: torch.Tensor, tokenizer: ByteTokenizer, eval_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    subset = tokens[:eval_size].clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return make_initial_mask(subset, tokenizer, mask_ratio=0.5, generator=generator)


def _build_trajectories_for_seeds(
    model: MaskedDiffusionTransformer,
    tokens: torch.Tensor,
    tokenizer: ByteTokenizer,
    config: ExperimentConfig,
    device: torch.device,
    seeds: tuple[int, ...],
) -> list:
    trajectories = []
    for seed in seeds:
        masked, labels = _masked_eval_tokens(tokens, tokenizer, config.data.eval_size, seed)
        trajectories.extend(build_reference_trajectories(model, masked, labels, tokenizer, config.model.timesteps, seed, device))
    return trajectories


def _score_candidate(
    model: MaskedDiffusionTransformer,
    trajectories: list,
    tokenizer: ByteTokenizer,
    device: torch.device,
    bits: dict[int, int],
) -> float:
    rows = evaluate_schedule_on_trajectories(
        model,
        trajectories,
        tokenizer,
        device,
        method="candidate",
        bit_schedule=bits,
        predicted_bound=0.0,
        avg_bits=sum(bits.values()) / len(bits),
        budget=sum(bits.values()),
    )
    return float(np.mean([float(row["final_logit_mse"]) for row in rows])) if rows else float("inf")


def _evaluate_all(
    model: MaskedDiffusionTransformer,
    schedules: list[Schedule],
    trajectories: list,
    tokenizer: ByteTokenizer,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for schedule in schedules:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = wall_time()
        bit_schedule = None if schedule.method == "unquantized_reference" else schedule.bits
        method_rows = evaluate_schedule_on_trajectories(
            model,
            trajectories,
            tokenizer,
            device,
            method=schedule.method,
            bit_schedule=bit_schedule,
            predicted_bound=schedule.predicted_theoretical,
            avg_bits=schedule.average_bits,
            budget=schedule.budget,
        )
        elapsed = wall_time() - start
        peak_memory = float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0
        for row in method_rows:
            row["wall_clock_latency_seconds_per_sample"] = elapsed / max(1, len(method_rows))
            row["peak_memory_bytes_reference_only"] = peak_memory
            row["actual_reference_dtype"] = "float32_unquantized_reference"
        rows.extend(method_rows)
    return rows


def run_experiment(config: ExperimentConfig) -> dict[str, Path]:
    """Run training, calibration, schedule search, evaluation, plotting, and reporting."""

    set_seed(config.train.seed)
    device = resolve_device(config.device)
    out_root = ensure_dir(config.output.root)
    csv_dir = ensure_dir(out_root / "csv")
    fig_dir = ensure_dir(out_root / "figures")
    config.save(out_root / "config_resolved.json")

    tokenizer = ByteTokenizer()
    splits = tokenize_splits(config.data, tokenizer, config.model.max_seq_len, config.train.seed)
    model = MaskedDiffusionTransformer(config.model, tokenizer.vocab_size, tokenizer.pad_token_id)
    run_start = wall_time()
    train_model(model, splits, tokenizer, config, device)

    calibration = run_calibration(model, splits, tokenizer, config, device)
    write_csv(csv_dir / "quantization_mse.csv", calibration.quant_rows)
    write_csv(csv_dir / "quantization_fits.csv", calibration.fit_rows)
    write_csv(csv_dir / "amplification_raw.csv", calibration.amplification_raw_rows)
    write_csv(csv_dir / "amplification_summary.csv", calibration.amplification_summary_rows)
    write_csv(csv_dir / "gronwall_factors.csv", calibration.gronwall_rows)

    schedules, continuous, clipped = build_baseline_schedules(
        config.model.timesteps,
        tuple(config.calibration.allowed_bits),
        config.calibration.target_avg_bits,
        calibration.g,
        calibration.theoretical_c,
        calibration.fitted_c,
        calibration.fitted_p,
    )
    gronwall_base = next(schedule for schedule in schedules if schedule.method == "gronwall_weighted_budget6")
    oracle_tokens = splits.validation[: max(1, min(config.data.eval_size, splits.validation.size(0)))]
    oracle_trajectories = _build_trajectories_for_seeds(model, oracle_tokens, tokenizer, config, device, (config.calibration.eval_seeds[0],))
    oracle = choose_empirical_oracle(
        gronwall_base,
        tuple(config.calibration.allowed_bits),
        config.calibration.oracle_max_candidates,
        lambda bits: _score_candidate(model, oracle_trajectories, tokenizer, device, bits),
        calibration.g,
        calibration.theoretical_c,
        calibration.fitted_c,
        calibration.fitted_p,
    )
    schedules.append(oracle)
    reference = Schedule(
        method="unquantized_reference",
        bits={step: 16 for step in range(1, config.model.timesteps + 1)},
        budget=16 * config.model.timesteps,
        average_bits=16.0,
        predicted_theoretical=0.0,
        predicted_fitted=0.0,
    )
    schedules = [reference] + schedules
    write_csv(csv_dir / "schedules.csv", schedules_to_rows(schedules))
    write_csv(
        csv_dir / "continuous_allocation.csv",
        [
            {
                "step": step,
                "continuous_bits": continuous[step],
                "clipped_continuous_bits": clipped[step],
                "A_k": calibration.g[step] * calibration.theoretical_c[step],
            }
            for step in sorted(continuous)
        ],
    )

    test_tokens = splits.test[: max(1, min(config.data.eval_size, splits.test.size(0)))]
    test_trajectories = _build_trajectories_for_seeds(model, test_tokens, tokenizer, config, device, tuple(config.calibration.eval_seeds))
    eval_rows = _evaluate_all(model, schedules, test_trajectories, tokenizer, device)
    write_csv(csv_dir / "evaluation_raw.csv", eval_rows)
    summary_rows = summarize_evaluation(eval_rows, config.calibration.bootstrap_samples, config.train.seed)
    write_csv(csv_dir / "evaluation_summary.csv", summary_rows)
    pair_rows = paired_tests(eval_rows, "gronwall_weighted_budget6")
    write_csv(csv_dir / "paired_tests.csv", pair_rows)
    corr_rows = correlation_analysis(summary_rows)
    write_csv(csv_dir / "correlations.csv", corr_rows)
    conclusion_rows = build_conclusion_rows(summary_rows, corr_rows, pair_rows)
    write_csv(csv_dir / "conclusion_provenance.csv", conclusion_rows)

    figure_paths = generate_all_figures(fig_dir, calibration.fit_rows, calibration.amplification_summary_rows, calibration.gronwall_rows, summary_rows, schedules)
    materially_differs = any(float(row["materially_differs_from_2"]) > 0.5 for row in calibration.fit_rows)
    runtime = wall_time() - run_start
    metadata = {
        "mode": config.mode,
        "dataset_source": splits.source_name,
        "device": str(device),
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "parameter_count": parameter_count(model),
        "train_size": int(splits.train.size(0)),
        "calibration_size": int(splits.calibration.size(0)),
        "validation_size": int(splits.validation.size(0)),
        "test_size": int(splits.test.size(0)),
        "runtime_seconds": runtime,
        "fake_quantization_note": "Fake quantization simulates activation rounding; it does not provide real low-bit storage or hardware acceleration.",
    }
    write_json(csv_dir / "metadata.json", metadata)
    report_path = out_root / "research_summary.md"
    generate_summary_report(
        report_path,
        config.mode,
        splits.source_name,
        str(device),
        parameter_count(model),
        runtime,
        summary_rows,
        corr_rows,
        pair_rows,
        conclusion_rows,
        figure_paths,
        materially_differs,
    )
    return {"output_root": out_root, "summary": report_path}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run the Grönwall mixed-precision MDLM project.")
    parser.add_argument("--mode", choices=["quick", "research"], default="quick")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config path.")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu or cuda.")
    parser.add_argument("--output-root", type=str, default=None, help="Override output directory.")
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    config = load_config(args.config) if args.config else get_config(args.mode)
    if args.device is not None:
        config.device = args.device
    if args.output_root is not None:
        config.output.root = args.output_root
    if args.no_resume:
        config.train.resume = False
    paths = run_experiment(config)
    print(f"Completed {config.mode} run. Outputs: {paths['output_root']}")


if __name__ == "__main__":
    main()
