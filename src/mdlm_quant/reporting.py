"""Final Markdown research summary generation."""

from __future__ import annotations

import math
from pathlib import Path


def _summary_value(rows: list[dict[str, float | int | str]], method: str, metric: str) -> float | None:
    for row in rows:
        if row["method"] == method and row["metric"] == metric:
            return float(row["mean"])
    return None


def _paired_row(rows: list[dict[str, float | int | str]], method: str, metric: str) -> dict[str, float | int | str] | None:
    for row in rows:
        if row["method"] == method and row["metric"] == metric:
            return row
    return None


def _is_significant_or_degenerate(row: dict[str, float | int | str] | None) -> bool:
    if row is None:
        return False
    status = str(row.get("test_status", ""))
    p_value = float(row.get("p_value_normal_approx", float("nan")))
    return (math.isfinite(p_value) and p_value < 0.05) or status == "constant_nonzero_difference_normal_test_undefined"


def build_conclusion_rows(
    summary_rows: list[dict[str, float | int | str]],
    corr_rows: list[dict[str, float | int | str]],
    paired_rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    """Build hypothesis decisions from computed CSV-backed rows."""

    rows: list[dict[str, float | int | str]] = []
    uniform4 = _summary_value(summary_rows, "uniform_4bit", "final_logit_mse")
    uniform6 = _summary_value(summary_rows, "uniform_6bit", "final_logit_mse")
    uniform8 = _summary_value(summary_rows, "uniform_8bit", "final_logit_mse")
    if uniform4 is not None and uniform6 is not None and uniform8 is not None:
        status = "supported" if uniform4 >= uniform6 >= uniform8 else "inconclusive"
        rows.append(
            {
                "hypothesis": "H1",
                "status": status,
                "claim": "낮은 bit width는 더 큰 observed final logit MSE를 낸다.",
                "evidence": f"uniform_4bit={uniform4:.6g}, uniform_6bit={uniform6:.6g}, uniform_8bit={uniform8:.6g}",
                "source_csv": "evaluation_summary.csv",
            }
        )

    h2_pair = _paired_row(paired_rows, "uniform_6bit", "final_logit_mse")
    gr = _summary_value(summary_rows, "gronwall_weighted_budget6", "final_logit_mse")
    u6 = _summary_value(summary_rows, "uniform_6bit", "final_logit_mse")
    if h2_pair is not None and gr is not None and u6 is not None:
        diff = float(h2_pair["mean_paired_difference"])
        decisive = _is_significant_or_degenerate(h2_pair)
        if decisive and diff > 0.0:
            status = "supported"
        elif decisive and diff < 0.0:
            status = "rejected"
        else:
            status = "inconclusive"
        rows.append(
            {
                "hypothesis": "H2",
                "status": status,
                "claim": "같은 총 bit budget에서 Grönwall-weighted schedule은 uniform 6-bit보다 낮은 final logit MSE를 낸다.",
                "evidence": (
                    f"gronwall={gr:.6g}, uniform_6bit={u6:.6g}, "
                    f"paired_diff_uniform_minus_gronwall={diff:.6g}, "
                    f"test_status={h2_pair['test_status']}, p={h2_pair['p_value_normal_approx']}"
                ),
                "source_csv": "evaluation_summary.csv;paired_tests.csv",
            }
        )

    spearman = None
    n_methods = 0
    for row in corr_rows:
        if row["correlation"] == "spearman":
            spearman = float(row["value"])
            n_methods = int(row["n_methods"])
            break
    if spearman is not None:
        if n_methods >= 5 and spearman > 0.5:
            status = "supported"
        elif n_methods >= 5 and spearman <= 0.0:
            status = "rejected"
        else:
            status = "inconclusive"
        rows.append(
            {
                "hypothesis": "H3",
                "status": status,
                "claim": "예측 Grönwall 목적식과 observed final logit MSE는 양의 순위상관을 가진다.",
                "evidence": f"spearman={spearman:.6g}, n_methods={n_methods}",
                "source_csv": "correlations.csv",
            }
        )
    return rows


def generate_summary_report(
    path: Path,
    mode: str,
    dataset_source: str,
    device_name: str,
    parameter_count: int,
    runtime_seconds: float,
    summary_rows: list[dict[str, float | int | str]],
    corr_rows: list[dict[str, float | int | str]],
    paired_rows: list[dict[str, float | int | str]],
    conclusion_rows: list[dict[str, float | int | str]],
    figure_paths: list[Path],
    materially_differs: bool,
) -> None:
    """Write the final Markdown research summary."""

    lines = [
        "# Discrete Grönwall Mixed-Precision MDLM 결과 요약",
        "",
        f"- 실행 모드: `{mode}`",
        f"- 데이터 소스: `{dataset_source}`",
        f"- 실제 실행 장치: `{device_name}`",
        f"- 학습 가능 파라미터 수: `{parameter_count}`",
        f"- 총 실행 시간(초): `{runtime_seconds:.2f}`",
        "",
        "## 해석상 주의",
        "",
        "- 이 프로젝트의 low-bit 결과는 activation fake quantization simulation입니다.",
        "- fake quantization은 실제 INT4/INT6 저장 공간 절감이나 하드웨어 가속을 제공하지 않습니다.",
        "- 기준 방법은 `unquantized_reference`이며, quick CPU 실행에서는 FP16 기준선이 아니라 float32 unquantized 기준선입니다.",
        "- 오차 recurrence는 discrete token 선택 이전의 continuous hidden/logit representation에서 성립한다고 가정했습니다.",
        "- 저장된 `predicted_gronwall_objective_p2`는 activation MSE calibration에서 온 목적식입니다. observed logit MSE와는 상관 분석으로만 비교하며, 인증된 동일-단위 upper bound로 해석하지 않습니다.",
        "- Grönwall 식은 이론적 upper-bound 형태이며, 관측 오차에 억지로 맞춘 회귀식이 아닙니다.",
    ]
    if materially_differs:
        lines.append("- 일부 timestep에서 fitted exponent `p_k`가 2와 실질적으로 달라서, theoretical `p=2` 분석과 empirical fitted-`p` 분석을 모두 저장했습니다.")

    lines.extend(["", "## 가설 판정", ""])
    for row in conclusion_rows:
        lines.append(f"- {row['hypothesis']}: **{row['status']}**. {row['claim']} 근거: {row['evidence']} (`{row['source_csv']}`)")

    lines.extend(["", "## 주요 평균 지표", ""])
    for row in summary_rows:
        if row["metric"] in {"masked_token_accuracy", "final_logit_mse"}:
            lines.append(
                f"- `{row['method']}` / `{row['metric']}`: mean={float(row['mean']):.6g}, "
                f"95% bootstrap CI=[{float(row['bootstrap_ci_low']):.6g}, {float(row['bootstrap_ci_high']):.6g}]"
            )

    lines.extend(["", "## 상관 분석", ""])
    for row in corr_rows:
        lines.append(f"- {row['correlation']}: {float(row['value']):.4f} (n={row['n_methods']}, scope={row.get('scope', 'all')})")

    lines.extend(["", "## paired test 주의", ""])
    for row in paired_rows:
        if row["metric"] == "final_logit_mse":
            lines.append(
                f"- `{row['method']}` vs `gronwall_weighted_budget6`: "
                f"diff={float(row['mean_paired_difference']):.6g}, status={row['test_status']}, p={row['p_value_normal_approx']}"
            )

    lines.extend(["", "## 생성된 그림", ""])
    for fig in figure_paths:
        lines.append(f"- `{fig.as_posix()}`")

    lines.extend(
        [
            "",
            "## 남은 방법론적 제한",
            "",
            "- quick mode는 smoke-test용 작은 corpus와 매우 짧은 학습만 수행하므로 연구 결론으로 일반화할 수 없습니다.",
            "- calibration의 `L_k`는 finite perturbation으로 얻은 local sensitivity estimate이며, 전체 denoising 동역학의 엄밀한 Lipschitz 상수라고 보장할 수 없습니다.",
            "- observed evaluation은 공정 비교를 위해 reference trajectory를 고정하므로, quantized prediction이 이후 discrete state를 바꾸는 자유 생성 오차 전파를 완전히 측정하지 않습니다.",
            "- latency와 peak memory는 reference-only metric이며 fake quantization의 실제 low-bit kernel 성능을 뜻하지 않습니다.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
