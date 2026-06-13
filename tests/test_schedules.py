from mdlm_quant.schedules import exact_discrete_allocation


def test_exact_discrete_allocation_preserves_budget():
    weights = {step: float(step) for step in range(1, 9)}
    schedule = exact_discrete_allocation(weights, (4, 6, 8), budget=48)
    assert set(schedule) == set(weights)
    assert sum(schedule.values()) == 48
    assert all(bit in {4, 6, 8} for bit in schedule.values())


def test_tie_target_does_not_override_objective():
    weights = {1: 1e-16, 2: 0.0}
    schedule = exact_discrete_allocation(weights, (4, 8), budget=12, tie_target={1: 4.0, 2: 8.0})
    assert schedule == {1: 8, 2: 4}
