# Discrete Grönwall 기반 timestep-wise mixed-precision MDLM

이 저장소는 작은 masked diffusion language model(MDLM)을 학습하고, denoising timestep별 activation fake quantization bit allocation을 비교하는 재현 가능한 학부 연구 프로젝트입니다.

## 핵심 주의

- 이 프로젝트의 low-bit 결과는 **activation fake quantization simulation**입니다.
- fake quantization은 실제 INT4/INT6 저장 공간 절감이나 하드웨어 가속을 제공하지 않습니다.
- Grönwall recurrence는 discrete token 선택 이전의 continuous hidden/logit representation에서 작동한다고 가정합니다.
- quick mode는 smoke test용 작은 내장 corpus를 사용합니다. 실제 TinyStories subset 연구 실행은 `research` mode에서 수행합니다.

## 설치

```bash
pip install -r requirements.txt
```

## 빠른 실행

```bash
python scripts/run_experiment.py --mode quick --device cpu --no-resume
```

출력은 기본적으로 `outputs/quick/`에 저장됩니다.

## 연구 실행

Colab GPU 런타임에서는 다음 명령을 사용할 수 있습니다.

```bash
python scripts/run_experiment.py --mode research --device cuda
```

`research` mode는 Hugging Face `roneneldan/TinyStories` subset을 불러옵니다. 네트워크와 `datasets` 패키지가 필요합니다.

## 주요 산출물

- `src/mdlm_quant/`: tokenizer, model, quantization, calibration, schedule, evaluation, plotting 로직
- `tests/`: CPU-compatible unit tests
- `notebooks/gronwall_mixed_precision_mdlm.ipynb`: 한국어 설명과 수식이 포함된 Colab notebook
- `outputs/<mode>/csv/`: 모든 결과 표 CSV
- `outputs/<mode>/figures/`: 모든 요구 그래프 PNG
- `outputs/<mode>/research_summary.md`: 자동 생성 최종 연구 요약

## 테스트

```bash
pytest
```

테스트는 fake quantizer의 range와 monotonic error, hand-computed discrete Grönwall factor, exact budget allocation, model shape, deterministic inference, sensitivity finite-value 조건을 확인합니다.

## 비교 방법

구현된 방법은 다음과 같습니다.

- unquantized reference (`unquantized_reference`; CPU quick run에서는 FP16 기준선이 아님)
- uniform 8-bit, 6-bit, 4-bit activation fake quantization
- low-to-high / high-to-low linear mixed schedules
- local-error-only exact-budget schedule
- Grönwall-weighted exact-budget schedule
- bounded-search empirical oracle schedule

공정 비교를 주장하는 mixed-precision 방법은 모두 같은 총 bit budget(`timesteps * 6`)을 사용합니다.
