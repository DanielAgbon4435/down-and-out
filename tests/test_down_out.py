import numpy as np

from src.down_out import down_out_call, hedge_errors, monte_carlo_price, vanilla_call


def test_price_is_zero_on_the_barrier():
    assert down_out_call(90.0, 100.0, 90.0, 1.0, 0.02, 0.0, 0.2) == 0.0


def test_a_far_barrier_is_the_vanilla_call():
    vanilla = vanilla_call(100.0, 100.0, 1.0, 0.02, 0.0, 0.2)
    knocked = down_out_call(100.0, 100.0, 1.0, 1.0, 0.02, 0.0, 0.2)
    assert abs(knocked - vanilla) < 1e-8


def test_the_barrier_makes_the_call_cheaper():
    vanilla = vanilla_call(100.0, 100.0, 1.0, 0.02, 0.0, 0.2)
    knocked = down_out_call(100.0, 100.0, 90.0, 1.0, 0.02, 0.0, 0.2)
    assert 0 < knocked < vanilla


def test_monte_carlo_matches_the_closed_form():
    closed = down_out_call(100.0, 100.0, 90.0, 1.0, 0.02, 0.0, 0.2)
    mc = monte_carlo_price(100.0, 100.0, 90.0, 1.0, 0.02, 0.0, 0.2, n_steps=100, n_paths=40_000, seed=3)
    assert abs(mc - closed) < 0.15


def test_wrong_volatility_widens_the_hedge_error():
    right = hedge_errors(100, 100, 90, 1.0, 0.02, 0.0, 0.20, 0.20, n_days=64, n_sub=2, n_paths=4_000, seed=4)
    wrong = hedge_errors(100, 100, 90, 1.0, 0.02, 0.0, 0.20, 0.30, n_days=64, n_sub=2, n_paths=4_000, seed=4)
    assert abs(right.mean()) < 0.40
    assert abs(wrong.mean()) < 0.50
    assert wrong.std(ddof=1) > 1.5 * right.std(ddof=1)
