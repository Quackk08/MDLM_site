# AGENTS.md

## Project objective

Build a reproducible, Google Colab-ready research project that trains a small masked diffusion language model and evaluates timestep-wise mixed-precision activation quantization using a discrete Grönwall error bound.

The project is for a calculus research assessment at approximately first- to second-year undergraduate level. Mathematical derivations, experimental validity, and reproducibility are more important than model scale.

## Core rules

* Use Python and PyTorch.
* Keep the core model small enough for a normal Google Colab GPU runtime.
* Provide `quick` and `research` configurations.
* The notebook must run top-to-bottom without manually editing Python source files.
* Put reusable logic in `src/`; keep the notebook as orchestration, explanation, and visualization.
* Use deterministic seeds where possible.
* Do not silently catch errors or silently switch to CPU.
* Report the actual device, package versions, parameter count, dataset size, and runtime configuration.
* Do not claim real INT4 speed or memory savings from fake quantization.
* Label fake-quantized results as simulation results.
* Do not invent experimental results. All tables and figures must be generated from saved CSV data.
* Add type hints and concise docstrings.
* All mathematical notation in notebook Markdown cells must be in LaTeX.
* Notebook explanations and chart labels should be written in Korean. Code identifiers should be in English.

## Mathematical requirements

Implement and document:

1. The recurrence
   `e_k <= L_k * e_(k-1) + epsilon_k`.
2. The discrete Grönwall bound
   `e_T <= sum_k epsilon_k * product_(j=k+1)^T L_j`.
3. The quantization error model
   `epsilon_k(b) = C_k * 2^(-2b)`.
4. The constrained objective
   `min sum_k A_k * 2^(-2b_k)` subject to `sum_k b_k = B`.
5. The continuous optimum
   `b_k = 0.5 * log2((2 * ln(2) * A_k) / lambda)`.
6. An exact discrete bit allocation method for allowed bits `{4, 6, 8}` under an exact total-bit budget.

## Model requirements

* Implement a small encoder-style Transformer masked diffusion language model.
* Default research config:

  * 4 Transformer layers
  * hidden size 256
  * 4 attention heads
  * feed-forward size 1024
  * sequence length 64
  * byte-level tokenizer
  * 12 denoising timesteps
* Default quick config:

  * 2 layers
  * hidden size 192
  * 8 timesteps
  * reduced dataset and training steps
* Train only on masked token positions.
* Implement deterministic argmax denoising for quantitative evaluation.
* Implement optional temperature sampling only for qualitative examples.

## Quantization requirements

* Implement symmetric per-tensor activation fake quantization.
* Allow a different bit width at each denoising timestep.
* Support FP16 baseline and fake 3-, 4-, 6-, and 8-bit activations.
* Quantize activations after each Transformer block.
* Keep model weights unchanged in the core experiment.
* Record local activation MSE at every timestep.

## Sensitivity requirements

* Estimate `C_k` from quantization error across multiple bit widths.
* Estimate local amplification `L_k` using finite perturbations of hidden states.
* Store mean, median, and 95th-percentile estimates.
* Calculate cumulative amplification in log space to prevent numerical overflow.
* Compare predicted Grönwall error with observed final error.

## Experimental baselines

Implement:

* FP16
* uniform 8-bit
* uniform 6-bit
* uniform 4-bit
* low-to-high linear schedule
* high-to-low linear schedule
* local-error-only schedule
* Grönwall-weighted schedule
* empirical oracle or bounded-search schedule on a small calibration subset

All compared schedules must use the same total bit budget when the experiment claims a fair comparison.

## Evaluation

Report:

* masked cross-entropy
* masked-token accuracy
* sequence reconstruction accuracy
* FP16 token agreement
* KL divergence from FP16 logits
* actual final hidden/logit MSE
* predicted Grönwall bound
* Pearson and Spearman correlations
* average bit width
* theoretical activation memory ratio
* wall-clock latency and peak memory as reference-only metrics

Use paired samples, at least five evaluation seeds, mean, standard deviation, and bootstrap 95% confidence intervals.

## Required visualizations

Generate and save:

1. `f(b) = A * 2^(-2b)` and its derivative.
2. A tangent line at a selected bit width.
3. Continuous Grönwall error `e(t)` and `e'(t)`.
4. Per-timestep `C_k`, `L_k`, and `G_k`.
5. Predicted versus observed error scatter plot.
6. Accuracy versus average bit width.
7. Final error comparison across methods.
8. Per-timestep bit allocation for each mixed-precision method.

## Tests

Add CPU-compatible tests for:

* fake quantizer output ranges and monotonic error behavior
* discrete Grönwall calculation on hand-computed examples
* exact budget preservation in bit allocation
* model input/output shapes
* deterministic inference under a fixed seed
* no NaN or infinity in sensitivity calculations

Run all tests before finishing.

## Deliverables

* A Colab notebook that runs end-to-end.
* Modular Python source files.
* Unit tests.
* A Korean README with setup and experiment instructions.
* CSV files for every result table.
* PNG files for every figure.
* A final automatically generated Markdown research summary.
