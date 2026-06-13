import math

from mdlm_quant.metrics import correlation_analysis, paired_tests


def test_paired_tests_marks_constant_nonzero_difference_as_degenerate():
    rows = []
    for sample_index in range(3):
        rows.append({"method": "ref", "seed": 1, "sample_index": sample_index, "final_logit_mse": 1.0, "masked_token_accuracy": 0.5})
        rows.append({"method": "other", "seed": 1, "sample_index": sample_index, "final_logit_mse": 2.0, "masked_token_accuracy": 0.4})
    out = paired_tests(rows, "ref", metrics=("final_logit_mse",))
    assert out[0]["test_status"] == "constant_nonzero_difference_normal_test_undefined"
    assert math.isnan(float(out[0]["p_value_normal_approx"]))
    assert math.isinf(float(out[0]["z_statistic_normal_approx"]))


def test_correlation_excludes_unquantized_reference_anchor():
    summary_rows = [
        {"method": "unquantized_reference", "metric": "final_logit_mse", "mean": 0.0, "predicted_gronwall_objective_p2": 0.0},
        {"method": "a", "metric": "final_logit_mse", "mean": 2.0, "predicted_gronwall_objective_p2": 1.0},
        {"method": "b", "metric": "final_logit_mse", "mean": 1.0, "predicted_gronwall_objective_p2": 2.0},
    ]
    out = correlation_analysis(summary_rows)
    assert out[0]["n_methods"] == 2
    assert out[0]["scope"] == "quantized_methods_only"
