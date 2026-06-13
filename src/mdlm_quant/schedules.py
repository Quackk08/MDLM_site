"""Bit allocation algorithms and baselines."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Schedule:
    """A timestep-wise activation bit schedule."""

    method: str
    bits: dict[int, int]
    budget: int
    average_bits: float
    predicted_theoretical: float
    predicted_fitted: float


def predicted_objective(bits: dict[int, int], g: dict[int, float], c: dict[int, float], p: dict[int, float] | None = None) -> float:
    """Compute sum_k G_k C_k 2^{-p b_k}, defaulting to theoretical p=2."""

    total = 0.0
    for step, bit in bits.items():
        exponent = p[step] if p is not None else 2.0
        total += float(g.get(step, 1.0)) * float(c.get(step, 0.0)) * (2.0 ** (-float(exponent) * int(bit)))
    return float(total)


def continuous_lagrange_solution(a: dict[int, float], budget: float) -> dict[int, float]:
    """Solve the continuous Lagrange-multiplier allocation by bisection."""

    steps = sorted(a)
    low, high = -100.0, 100.0
    for _ in range(100):
        mid = (low + high) / 2.0
        values = {}
        for step in steps:
            coeff = max(float(a[step]), 1e-30)
            values[step] = 0.5 * math.log2((2.0 * math.log(2.0) * coeff) / math.exp(mid))
        total = sum(values.values())
        if total > budget:
            low = mid
        else:
            high = mid
    lambda_log = (low + high) / 2.0
    return {
        step: 0.5 * math.log2((2.0 * math.log(2.0) * max(float(a[step]), 1e-30)) / math.exp(lambda_log))
        for step in steps
    }


def clip_continuous(bits: dict[int, float], low: int, high: int) -> dict[int, float]:
    """Clip a continuous allocation to valid bit bounds."""

    return {step: min(float(high), max(float(low), value)) for step, value in bits.items()}


def _allocation_is_better(
    candidate: tuple[float, float, list[int]],
    incumbent: tuple[float, float, list[int]] | None,
) -> bool:
    """Compare allocations by objective first, then tie distance."""

    if incumbent is None:
        return True
    if candidate[0] < incumbent[0]:
        return True
    if candidate[0] == incumbent[0] and candidate[1] < incumbent[1]:
        return True
    return False


def exact_discrete_allocation(
    weights: dict[int, float],
    allowed_bits: tuple[int, ...],
    budget: int,
    p: float = 2.0,
    tie_target: dict[int, float] | None = None,
) -> dict[int, int]:
    """Find the exact-budget discrete schedule minimizing weighted quantization objective."""

    steps = sorted(weights)
    inf = float("inf")
    dp: dict[tuple[int, int], tuple[float, float, list[int]]] = {(0, 0): (0.0, 0.0, [])}
    for i, step in enumerate(steps, start=1):
        for used in range(0, budget + 1):
            best: tuple[float, float, list[int]] | None = None
            for bit in allowed_bits:
                previous = dp.get((i - 1, used - bit))
                if previous is None:
                    continue
                objective = float(weights.get(step, 0.0)) * (2.0 ** (-p * bit))
                tie = 0.0
                if tie_target is not None:
                    tie = (float(bit) - float(tie_target.get(step, bit))) ** 2
                candidate = (previous[0] + objective, previous[1] + tie, previous[2] + [int(bit)])
                if _allocation_is_better(candidate, best):
                    best = candidate
            if best is not None:
                dp[(i, used)] = best
    result = dp.get((len(steps), budget))
    if result is None:
        raise ValueError(f"no exact allocation for budget={budget} and bits={allowed_bits}")
    return {step: bit for step, bit in zip(steps, result[2])}


def project_to_budget(target: dict[int, float], allowed_bits: tuple[int, ...], budget: int) -> dict[int, int]:
    """Project a continuous target to exact discrete bits by minimizing squared distance."""

    return exact_discrete_allocation({step: 0.0 for step in target}, allowed_bits, budget, p=2.0, tie_target=target)


def linear_schedule(timesteps: int, low: int, high: int, budget: int, allowed_bits: tuple[int, ...], reverse: bool) -> dict[int, int]:
    """Create an exact-budget low-to-high or high-to-low schedule."""

    values = np.linspace(high if reverse else low, low if reverse else high, timesteps)
    target = {step: float(values[step - 1]) for step in range(1, timesteps + 1)}
    return project_to_budget(target, allowed_bits, budget)


def uniform_schedule(timesteps: int, bit: int) -> dict[int, int]:
    """Create a uniform schedule."""

    return {step: int(bit) for step in range(1, timesteps + 1)}


def make_schedule(
    method: str,
    bits: dict[int, int],
    g: dict[int, float],
    theoretical_c: dict[int, float],
    fitted_c: dict[int, float],
    fitted_p: dict[int, float],
) -> Schedule:
    """Build a Schedule with objective values."""

    budget = int(sum(bits.values()))
    avg = budget / len(bits)
    return Schedule(
        method=method,
        bits=dict(bits),
        budget=budget,
        average_bits=avg,
        predicted_theoretical=predicted_objective(bits, g, theoretical_c, None),
        predicted_fitted=predicted_objective(bits, g, fitted_c, fitted_p),
    )


def build_baseline_schedules(
    timesteps: int,
    allowed_bits: tuple[int, ...],
    target_avg_bits: int,
    g: dict[int, float],
    theoretical_c: dict[int, float],
    fitted_c: dict[int, float],
    fitted_p: dict[int, float],
) -> tuple[list[Schedule], dict[int, float], dict[int, float]]:
    """Build uniform and analytic mixed-precision schedules."""

    target_budget = int(timesteps * target_avg_bits)
    schedules: list[Schedule] = []
    for bit in (8, 6, 4):
        schedules.append(make_schedule(f"uniform_{bit}bit", uniform_schedule(timesteps, bit), g, theoretical_c, fitted_c, fitted_p))
    schedules.append(
        make_schedule(
            "low_to_high_linear_budget6",
            linear_schedule(timesteps, min(allowed_bits), max(allowed_bits), target_budget, allowed_bits, reverse=False),
            g,
            theoretical_c,
            fitted_c,
            fitted_p,
        )
    )
    schedules.append(
        make_schedule(
            "high_to_low_linear_budget6",
            linear_schedule(timesteps, min(allowed_bits), max(allowed_bits), target_budget, allowed_bits, reverse=True),
            g,
            theoretical_c,
            fitted_c,
            fitted_p,
        )
    )
    local_weights = {step: theoretical_c[step] for step in range(1, timesteps + 1)}
    local_bits = exact_discrete_allocation(local_weights, allowed_bits, target_budget, p=2.0)
    schedules.append(make_schedule("local_error_only_budget6", local_bits, g, theoretical_c, fitted_c, fitted_p))
    gronwall_weights = {step: g[step] * theoretical_c[step] for step in range(1, timesteps + 1)}
    continuous = continuous_lagrange_solution(gronwall_weights, target_budget)
    clipped = clip_continuous(continuous, min(allowed_bits), max(allowed_bits))
    gronwall_bits = exact_discrete_allocation(gronwall_weights, allowed_bits, target_budget, p=2.0, tie_target=clipped)
    schedules.append(make_schedule("gronwall_weighted_budget6", gronwall_bits, g, theoretical_c, fitted_c, fitted_p))
    return schedules, continuous, clipped


def enumerate_candidate_schedules(
    base: dict[int, int],
    allowed_bits: tuple[int, ...],
    budget: int,
    max_candidates: int,
) -> list[dict[int, int]]:
    """Generate a bounded set of exact-budget neighbors for empirical oracle search."""

    steps = sorted(base)
    candidates: list[dict[int, int]] = [dict(base)]
    for i, j in itertools.permutations(steps, 2):
        for up in allowed_bits:
            for down in allowed_bits:
                proposal = dict(base)
                proposal[i] = int(up)
                proposal[j] = int(down)
                if sum(proposal.values()) == budget:
                    candidates.append(proposal)
    unique: list[dict[int, int]] = []
    seen: set[tuple[int, ...]] = set()
    for cand in candidates:
        key = tuple(cand[step] for step in steps)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
        if len(unique) >= max_candidates:
            break
    return unique


def choose_empirical_oracle(
    base: Schedule,
    allowed_bits: tuple[int, ...],
    max_candidates: int,
    score_fn: Callable[[dict[int, int]], float],
    g: dict[int, float],
    theoretical_c: dict[int, float],
    fitted_c: dict[int, float],
    fitted_p: dict[int, float],
) -> Schedule:
    """Pick the best exact-budget schedule from a bounded local search."""

    best_bits = dict(base.bits)
    best_score = score_fn(best_bits)
    for candidate in enumerate_candidate_schedules(base.bits, allowed_bits, base.budget, max_candidates):
        score = score_fn(candidate)
        if score < best_score:
            best_bits = dict(candidate)
            best_score = score
    return make_schedule("empirical_oracle_budget6", best_bits, g, theoretical_c, fitted_c, fitted_p)

def schedules_to_rows(schedules: list[Schedule]) -> list[dict[str, float | int | str]]:
    """Flatten schedules for CSV output."""

    rows: list[dict[str, float | int | str]] = []
    for schedule in schedules:
        for step, bit in sorted(schedule.bits.items()):
            rows.append(
                {
                    "method": schedule.method,
                    "step": step,
                    "bits": bit,
                    "total_bit_budget": schedule.budget,
                    "average_bit_width": schedule.average_bits,
                    "predicted_gronwall_objective_p2": schedule.predicted_theoretical,
                    "predicted_gronwall_objective_fitted_p": schedule.predicted_fitted,
                }
            )
    return rows
