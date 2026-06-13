from mdlm_quant.calibration import cumulative_gronwall


def test_discrete_gronwall_factors_hand_computed():
    g, rows = cumulative_gronwall({1: 2.0, 2: 3.0, 3: 4.0}, timesteps=3)
    assert abs(g[1] - 12.0) < 1e-9
    assert abs(g[2] - 4.0) < 1e-9
    assert abs(g[3] - 1.0) < 1e-9
    assert len(rows) == 3

