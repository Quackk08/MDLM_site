"""Figure generation for calculus and experimental results."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from .schedules import Schedule


def _configure_fonts() -> None:
    """Prefer a Korean-capable font when one is available."""

    candidates = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/NanumGothic.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


_configure_fonts()


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_calculus_function(out_dir: Path, a: float = 1.0) -> Path:
    """Plot f(b)=A*2^(-2b) and its derivative."""

    b = np.linspace(3, 8, 200)
    f = a * 2.0 ** (-2.0 * b)
    derivative = -2.0 * np.log(2.0) * f
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(b, f, label=r"$f(b)=A2^{-2b}$")
    ax.plot(b, derivative, label=r"$f'(b)=-2\ln(2)A2^{-2b}$")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("비트 폭에 따른 이론적 오차 함수와 도함수")
    ax.set_xlabel("비트 폭 b")
    ax.set_ylabel("값")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = out_dir / "figure_01_calculus_function_derivative.png"
    _save(fig, path)
    return path


def plot_tangent(out_dir: Path, a: float = 1.0, b0: float = 6.0) -> Path:
    """Plot a tangent line at a selected bit width."""

    b = np.linspace(3, 8, 200)
    f = a * 2.0 ** (-2.0 * b)
    f0 = a * 2.0 ** (-2.0 * b0)
    slope = -2.0 * np.log(2.0) * f0
    tangent = f0 + slope * (b - b0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(b, f, label=r"$f(b)$")
    ax.plot(b, tangent, "--", label=fr"$b_0={b0}$에서의 접선")
    ax.scatter([b0], [f0], color="black", zorder=5, label="접점")
    ax.set_title("이론적 오차 함수의 접선 근사")
    ax.set_xlabel("비트 폭 b")
    ax.set_ylabel("오차")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = out_dir / "figure_02_tangent_line.png"
    _save(fig, path)
    return path


def plot_continuous_gronwall(out_dir: Path) -> Path:
    """Plot an illustrative continuous Gronwall curve and derivative."""

    t = np.linspace(0, 1, 200)
    l_const = 1.7
    eps = 0.08
    e = eps * (np.exp(l_const * t) - 1.0) / l_const
    derivative = l_const * e + eps
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, e, label=r"$e(t)$")
    ax.plot(t, derivative, label=r"$e'(t)=Le(t)+\epsilon$")
    ax.set_title("연속 Grönwall 형태의 오차 성장 예시")
    ax.set_xlabel("정규화된 시간 t")
    ax.set_ylabel("오차 크기")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = out_dir / "figure_03_continuous_gronwall.png"
    _save(fig, path)
    return path


def plot_per_timestep(
    out_dir: Path,
    fit_rows: list[dict[str, float | int]],
    amp_rows: list[dict[str, float | int]],
    g_rows: list[dict[str, float | int]],
) -> Path:
    """Plot C_k, L_k, and G_k by timestep."""

    steps = [int(row["step"]) for row in fit_rows]
    c_vals = [float(row["C_theoretical_p2"]) for row in fit_rows]
    l_vals = [float(row["L_mean"]) for row in amp_rows]
    g_vals = [float(row["G"]) for row in g_rows]
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(steps, c_vals, marker="o", label=r"$C_k$")
    axes[1].plot(steps, l_vals, marker="o", color="tab:orange", label=r"$L_k$")
    axes[2].plot(steps, g_vals, marker="o", color="tab:green", label=r"$G_k$")
    labels = ["양자화 계수 C_k", "국소 증폭 L_k", "누적 증폭 G_k"]
    for ax, label in zip(axes, labels):
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[2].set_xlabel("denoising timestep k")
    axes[0].set_title("시간 단계별 보정 계수")
    path = out_dir / "figure_04_per_timestep_C_L_G.png"
    _save(fig, path)
    return path


def plot_predicted_vs_observed(out_dir: Path, summary_rows: list[dict[str, float | int | str]]) -> Path:
    """Scatter predicted Gronwall objective against observed final logit MSE."""

    rows = [row for row in summary_rows if row["metric"] == "final_logit_mse" and row["method"] != "unquantized_reference"]
    x = [float(row["predicted_gronwall_objective_p2"]) for row in rows]
    y = [float(row["mean"]) for row in rows]
    labels = [str(row["method"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y)
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_title("예측 Grönwall 목적식과 관측 최종 오차")
    ax.set_xlabel("예측 Grönwall 목적식 (p=2)")
    ax.set_ylabel("관측 final logit MSE")
    ax.grid(True, alpha=0.3)
    path = out_dir / "figure_05_predicted_vs_observed.png"
    _save(fig, path)
    return path


def plot_accuracy_vs_bits(out_dir: Path, summary_rows: list[dict[str, float | int | str]]) -> Path:
    """Plot accuracy versus average bit width."""

    rows = [row for row in summary_rows if row["metric"] == "masked_token_accuracy"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in rows:
        ax.scatter(float(row["average_bit_width"]), float(row["mean"]), label=str(row["method"]))
    ax.set_title("평균 비트 폭과 masked-token 정확도")
    ax.set_xlabel("평균 activation bit width")
    ax.set_ylabel("masked-token accuracy")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    path = out_dir / "figure_06_accuracy_vs_average_bits.png"
    _save(fig, path)
    return path


def plot_final_error_comparison(out_dir: Path, summary_rows: list[dict[str, float | int | str]]) -> Path:
    """Bar chart of final logit MSE across methods."""

    rows = [row for row in summary_rows if row["metric"] == "final_logit_mse"]
    labels = [str(row["method"]) for row in rows]
    values = [float(row["mean"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(rows)), values)
    ax.set_xticks(range(len(rows)), labels, rotation=35, ha="right")
    ax.set_title("방법별 최종 logit MSE 비교")
    ax.set_xlabel("방법")
    ax.set_ylabel("final logit MSE")
    ax.grid(True, axis="y", alpha=0.3)
    path = out_dir / "figure_07_final_error_comparison.png"
    _save(fig, path)
    return path


def plot_bit_allocations(out_dir: Path, schedules: list[Schedule]) -> Path:
    """Plot per-timestep bit allocations for mixed-precision methods."""

    mixed = [s for s in schedules if "budget6" in s.method]
    fig, ax = plt.subplots(figsize=(9, 5))
    for schedule in mixed:
        steps = sorted(schedule.bits)
        ax.plot(steps, [schedule.bits[step] for step in steps], marker="o", label=schedule.method)
    ax.set_title("혼합 정밀도 방법의 timestep별 bit allocation")
    ax.set_xlabel("denoising timestep k")
    ax.set_ylabel("activation bit width")
    ax.set_yticks([4, 6, 8])
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    path = out_dir / "figure_08_bit_allocations.png"
    _save(fig, path)
    return path


def generate_all_figures(
    out_dir: Path,
    fit_rows: list[dict[str, float | int]],
    amp_rows: list[dict[str, float | int]],
    g_rows: list[dict[str, float | int]],
    summary_rows: list[dict[str, float | int | str]],
    schedules: list[Schedule],
) -> list[Path]:
    """Generate every required figure."""

    return [
        plot_calculus_function(out_dir),
        plot_tangent(out_dir),
        plot_continuous_gronwall(out_dir),
        plot_per_timestep(out_dir, fit_rows, amp_rows, g_rows),
        plot_predicted_vs_observed(out_dir, summary_rows),
        plot_accuracy_vs_bits(out_dir, summary_rows),
        plot_final_error_comparison(out_dir, summary_rows),
        plot_bit_allocations(out_dir, schedules),
    ]
