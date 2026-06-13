# Discrete Grönwall Mixed-Precision MDLM 결과 요약

- 실행 모드: `research`
- 데이터 소스: `roneneldan/TinyStories`
- 실제 실행 장치: `cpu`
- 학습 가능 파라미터 수: `3245824`
- 총 실행 시간(초): `2694.60`

## 해석상 주의

- 이 프로젝트의 low-bit 결과는 activation fake quantization simulation입니다.
- fake quantization은 실제 INT4/INT6 저장 공간 절감이나 하드웨어 가속을 제공하지 않습니다.
- 기준 방법은 `unquantized_reference`이며, quick CPU 실행에서는 FP16 기준선이 아니라 float32 unquantized 기준선입니다.
- 오차 recurrence는 discrete token 선택 이전의 continuous hidden/logit representation에서 성립한다고 가정했습니다.
- 저장된 `predicted_gronwall_objective_p2`는 activation MSE calibration에서 온 목적식입니다. observed logit MSE와는 상관 분석으로만 비교하며, 인증된 동일-단위 upper bound로 해석하지 않습니다.
- Grönwall 식은 이론적 upper-bound 형태이며, 관측 오차에 억지로 맞춘 회귀식이 아닙니다.

## 가설 판정

- H1: **supported**. 낮은 bit width는 더 큰 observed final logit MSE를 낸다. 근거: uniform_4bit=0.103941, uniform_6bit=0.00356833, uniform_8bit=0.000206489 (`evaluation_summary.csv`)
- H2: **rejected**. 같은 총 bit budget에서 Grönwall-weighted schedule은 uniform 6-bit보다 낮은 final logit MSE를 낸다. 근거: gronwall=0.0636727, uniform_6bit=0.00356833, paired_diff_uniform_minus_gronwall=-0.0601044, test_status=normal_approximation, p=0.0 (`evaluation_summary.csv;paired_tests.csv`)
- H3: **inconclusive**. 예측 Grönwall 목적식과 observed final logit MSE는 양의 순위상관을 가진다. 근거: spearman=0.277108, n_methods=8 (`correlations.csv`)

## 주요 평균 지표

- `unquantized_reference` / `masked_token_accuracy`: mean=0.536038, 95% bootstrap CI=[0.516176, 0.557361]
- `unquantized_reference` / `final_logit_mse`: mean=0, 95% bootstrap CI=[0, 0]
- `uniform_8bit` / `masked_token_accuracy`: mean=0.53629, 95% bootstrap CI=[0.516328, 0.559124]
- `uniform_8bit` / `final_logit_mse`: mean=0.000206489, 95% bootstrap CI=[0.000196974, 0.000215743]
- `uniform_6bit` / `masked_token_accuracy`: mean=0.535786, 95% bootstrap CI=[0.515718, 0.556907]
- `uniform_6bit` / `final_logit_mse`: mean=0.00356833, 95% bootstrap CI=[0.00342595, 0.003733]
- `uniform_4bit` / `masked_token_accuracy`: mean=0.529637, 95% bootstrap CI=[0.510936, 0.550556]
- `uniform_4bit` / `final_logit_mse`: mean=0.103941, 95% bootstrap CI=[0.100829, 0.107235]
- `low_to_high_linear_budget6` / `masked_token_accuracy`: mean=0.535937, 95% bootstrap CI=[0.516079, 0.557461]
- `low_to_high_linear_budget6` / `final_logit_mse`: mean=0.0180502, 95% bootstrap CI=[0.0171963, 0.0190873]
- `high_to_low_linear_budget6` / `masked_token_accuracy`: mean=0.535181, 95% bootstrap CI=[0.51285, 0.555192]
- `high_to_low_linear_budget6` / `final_logit_mse`: mean=0.0322771, 95% bootstrap CI=[0.0310089, 0.0336109]
- `local_error_only_budget6` / `masked_token_accuracy`: mean=0.535786, 95% bootstrap CI=[0.516023, 0.555546]
- `local_error_only_budget6` / `final_logit_mse`: mean=0.00356833, 95% bootstrap CI=[0.00341331, 0.0037132]
- `gronwall_weighted_budget6` / `masked_token_accuracy`: mean=0.531956, 95% bootstrap CI=[0.513406, 0.551817]
- `gronwall_weighted_budget6` / `final_logit_mse`: mean=0.0636727, 95% bootstrap CI=[0.0614735, 0.0660787]
- `empirical_oracle_budget6` / `masked_token_accuracy`: mean=0.53256, 95% bootstrap CI=[0.513248, 0.553131]
- `empirical_oracle_budget6` / `final_logit_mse`: mean=0.0535113, 95% bootstrap CI=[0.051632, 0.0554731]

## 상관 분석

- pearson: 0.4293 (n=8, scope=quantized_methods_only)
- spearman: 0.2771 (n=8, scope=quantized_methods_only)

## paired test 주의

- `empirical_oracle_budget6` vs `gronwall_weighted_budget6`: diff=-0.0101615, status=normal_approximation, p=2.8630506148620807e-127
- `high_to_low_linear_budget6` vs `gronwall_weighted_budget6`: diff=-0.0313957, status=normal_approximation, p=8.639720021773788e-276
- `local_error_only_budget6` vs `gronwall_weighted_budget6`: diff=-0.0601044, status=normal_approximation, p=0.0
- `low_to_high_linear_budget6` vs `gronwall_weighted_budget6`: diff=-0.0456225, status=normal_approximation, p=3.86e-321
- `uniform_4bit` vs `gronwall_weighted_budget6`: diff=0.0402678, status=normal_approximation, p=0.0
- `uniform_6bit` vs `gronwall_weighted_budget6`: diff=-0.0601044, status=normal_approximation, p=0.0
- `uniform_8bit` vs `gronwall_weighted_budget6`: diff=-0.0634663, status=normal_approximation, p=0.0
- `unquantized_reference` vs `gronwall_weighted_budget6`: diff=-0.0636727, status=normal_approximation, p=0.0

## 생성된 그림

- `outputs/research/figures/figure_01_calculus_function_derivative.png`
- `outputs/research/figures/figure_02_tangent_line.png`
- `outputs/research/figures/figure_03_continuous_gronwall.png`
- `outputs/research/figures/figure_04_per_timestep_C_L_G.png`
- `outputs/research/figures/figure_05_predicted_vs_observed.png`
- `outputs/research/figures/figure_06_accuracy_vs_average_bits.png`
- `outputs/research/figures/figure_07_final_error_comparison.png`
- `outputs/research/figures/figure_08_bit_allocations.png`

## 남은 방법론적 제한

- quick mode는 smoke-test용 작은 corpus와 매우 짧은 학습만 수행하므로 연구 결론으로 일반화할 수 없습니다.
- calibration의 `L_k`는 finite perturbation으로 얻은 local sensitivity estimate이며, 전체 denoising 동역학의 엄밀한 Lipschitz 상수라고 보장할 수 없습니다.
- observed evaluation은 공정 비교를 위해 reference trajectory를 고정하므로, quantized prediction이 이후 discrete state를 바꾸는 자유 생성 오차 전파를 완전히 측정하지 않습니다.
- latency와 peak memory는 reference-only metric이며 fake quantization의 실제 low-bit kernel 성능을 뜻하지 않습니다.
