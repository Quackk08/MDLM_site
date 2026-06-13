# Discrete Grönwall Mixed-Precision MDLM 결과 요약

- 실행 모드: `quick`
- 데이터 소스: `bundled_smoke_corpus`
- 실제 실행 장치: `cpu`
- 학습 가능 파라미터 수: `954048`
- 총 실행 시간(초): `67.81`

## 해석상 주의

- 이 프로젝트의 low-bit 결과는 activation fake quantization simulation입니다.
- fake quantization은 실제 INT4/INT6 저장 공간 절감이나 하드웨어 가속을 제공하지 않습니다.
- 기준 방법은 `unquantized_reference`이며, quick CPU 실행에서는 FP16 기준선이 아니라 float32 unquantized 기준선입니다.
- 오차 recurrence는 discrete token 선택 이전의 continuous hidden/logit representation에서 성립한다고 가정했습니다.
- 저장된 `predicted_gronwall_objective_p2`는 activation MSE calibration에서 온 목적식입니다. observed logit MSE와는 상관 분석으로만 비교하며, 인증된 동일-단위 upper bound로 해석하지 않습니다.
- Grönwall 식은 이론적 upper-bound 형태이며, 관측 오차에 억지로 맞춘 회귀식이 아닙니다.

## 가설 판정

- H1: **supported**. 낮은 bit width는 더 큰 observed final logit MSE를 낸다. 근거: uniform_4bit=0.000874985, uniform_6bit=4.03696e-05, uniform_8bit=2.3853e-06 (`evaluation_summary.csv`)
- H2: **rejected**. 같은 총 bit budget에서 Grönwall-weighted schedule은 uniform 6-bit보다 낮은 final logit MSE를 낸다. 근거: gronwall=0.000314268, uniform_6bit=4.03696e-05, paired_diff_uniform_minus_gronwall=-0.000273898, test_status=normal_approximation, p=0.0 (`evaluation_summary.csv;paired_tests.csv`)
- H3: **inconclusive**. 예측 Grönwall 목적식과 observed final logit MSE는 양의 순위상관을 가진다. 근거: spearman=0.421687, n_methods=8 (`correlations.csv`)

## 주요 평균 지표

- `unquantized_reference` / `masked_token_accuracy`: mean=0.2, 95% bootstrap CI=[0.181431, 0.220968]
- `unquantized_reference` / `final_logit_mse`: mean=0, 95% bootstrap CI=[0, 0]
- `uniform_8bit` / `masked_token_accuracy`: mean=0.2, 95% bootstrap CI=[0.182218, 0.21373]
- `uniform_8bit` / `final_logit_mse`: mean=2.3853e-06, 95% bootstrap CI=[2.36761e-06, 2.40267e-06]
- `uniform_6bit` / `masked_token_accuracy`: mean=0.2, 95% bootstrap CI=[0.182258, 0.216935]
- `uniform_6bit` / `final_logit_mse`: mean=4.03696e-05, 95% bootstrap CI=[4.00557e-05, 4.06831e-05]
- `uniform_4bit` / `masked_token_accuracy`: mean=0.2, 95% bootstrap CI=[0.183851, 0.220161]
- `uniform_4bit` / `final_logit_mse`: mean=0.000874985, 95% bootstrap CI=[0.00086863, 0.00088097]
- `low_to_high_linear_budget6` / `masked_token_accuracy`: mean=0.2, 95% bootstrap CI=[0.182238, 0.216169]
- `low_to_high_linear_budget6` / `final_logit_mse`: mean=0.000221437, 95% bootstrap CI=[0.000218298, 0.000224376]
- `high_to_low_linear_budget6` / `masked_token_accuracy`: mean=0.2, 95% bootstrap CI=[0.183871, 0.216169]
- `high_to_low_linear_budget6` / `final_logit_mse`: mean=0.000252581, 95% bootstrap CI=[0.000248637, 0.000256299]
- `local_error_only_budget6` / `masked_token_accuracy`: mean=0.2, 95% bootstrap CI=[0.183851, 0.216129]
- `local_error_only_budget6` / `final_logit_mse`: mean=4.03696e-05, 95% bootstrap CI=[4.01015e-05, 4.06558e-05]
- `gronwall_weighted_budget6` / `masked_token_accuracy`: mean=0.2, 95% bootstrap CI=[0.181411, 0.214536]
- `gronwall_weighted_budget6` / `final_logit_mse`: mean=0.000314268, 95% bootstrap CI=[0.00031132, 0.000317531]
- `empirical_oracle_budget6` / `masked_token_accuracy`: mean=0.2, 95% bootstrap CI=[0.183065, 0.220181]
- `empirical_oracle_budget6` / `final_logit_mse`: mean=0.000218845, 95% bootstrap CI=[0.000216398, 0.000220998]

## 상관 분석

- pearson: 0.6962 (n=8, scope=quantized_methods_only)
- spearman: 0.4217 (n=8, scope=quantized_methods_only)

## paired test 주의

- `empirical_oracle_budget6` vs `gronwall_weighted_budget6`: diff=-9.54234e-05, status=normal_approximation, p=0.0
- `high_to_low_linear_budget6` vs `gronwall_weighted_budget6`: diff=-6.16873e-05, status=normal_approximation, p=5.304587738453634e-182
- `local_error_only_budget6` vs `gronwall_weighted_budget6`: diff=-0.000273898, status=normal_approximation, p=0.0
- `low_to_high_linear_budget6` vs `gronwall_weighted_budget6`: diff=-9.28315e-05, status=normal_approximation, p=0.0
- `uniform_4bit` vs `gronwall_weighted_budget6`: diff=0.000560717, status=normal_approximation, p=0.0
- `uniform_6bit` vs `gronwall_weighted_budget6`: diff=-0.000273898, status=normal_approximation, p=0.0
- `uniform_8bit` vs `gronwall_weighted_budget6`: diff=-0.000311883, status=normal_approximation, p=0.0
- `unquantized_reference` vs `gronwall_weighted_budget6`: diff=-0.000314268, status=normal_approximation, p=0.0

## 생성된 그림

- `outputs/quick/figures/figure_01_calculus_function_derivative.png`
- `outputs/quick/figures/figure_02_tangent_line.png`
- `outputs/quick/figures/figure_03_continuous_gronwall.png`
- `outputs/quick/figures/figure_04_per_timestep_C_L_G.png`
- `outputs/quick/figures/figure_05_predicted_vs_observed.png`
- `outputs/quick/figures/figure_06_accuracy_vs_average_bits.png`
- `outputs/quick/figures/figure_07_final_error_comparison.png`
- `outputs/quick/figures/figure_08_bit_allocations.png`

## 남은 방법론적 제한

- quick mode는 smoke-test용 작은 corpus와 매우 짧은 학습만 수행하므로 연구 결론으로 일반화할 수 없습니다.
- calibration의 `L_k`는 finite perturbation으로 얻은 local sensitivity estimate이며, 전체 denoising 동역학의 엄밀한 Lipschitz 상수라고 보장할 수 없습니다.
- observed evaluation은 공정 비교를 위해 reference trajectory를 고정하므로, quantized prediction이 이후 discrete state를 바꾸는 자유 생성 오차 전파를 완전히 측정하지 않습니다.
- latency와 peak memory는 reference-only metric이며 fake quantization의 실제 low-bit kernel 성능을 뜻하지 않습니다.
