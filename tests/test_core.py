import numpy as np
import pandas as pd

from src.metrics import beta, drawdown_series, historical_var, log_returns
from src.monte_carlo import simulate_gbm, summarize_simulation
from src.valuation import dcf_equity_value, dcf_sensitivity_table


def test_log_returns_length():
    prices = pd.Series([100.0, 101.0, 103.0, 102.0])
    assert len(log_returns(prices)) == 3


def test_beta_identical_series_is_one():
    idx = pd.date_range('2026-01-01', periods=5)
    r = pd.Series([0.01, -0.02, 0.015, 0.0, 0.02], index=idx)
    assert np.isclose(beta(r, r), 1.0)


def test_drawdown():
    prices = pd.Series([100.0, 110.0, 99.0, 121.0])
    assert np.isclose(drawdown_series(prices).iloc[2], -0.10)


def test_historical_var_nonnegative():
    returns = pd.Series([-0.05, -0.02, 0.01, 0.02, 0.03])
    assert historical_var(returns, 0.95) >= 0


def test_dcf_returns_per_share():
    result = dcf_equity_value(100.0, 20.0, 10.0, 10.0, 0.05, 0.10, 0.02, 5)
    assert result['intrinsic_value_per_share'] > 0


def test_dcf_rejects_invalid_terminal_growth():
    try:
        dcf_equity_value(100.0, 0.0, 0.0, 10.0, wacc=0.03, terminal_growth=0.03)
    except ValueError:
        pass
    else:
        raise AssertionError('Expected ValueError')


def test_sensitivity_shape():
    table = dcf_sensitivity_table(100.0, 20.0, 10.0, 10.0, points=5)
    assert table.shape == (5, 5)


def test_monte_carlo_reproducible():
    idx = pd.date_range('2024-01-01', periods=300, freq='B')
    prices = pd.Series(100 * np.exp(np.linspace(0, 0.3, 300)), index=idx)
    p1 = simulate_gbm(prices, days=10, simulations=50, seed=7)
    p2 = simulate_gbm(prices, days=10, simulations=50, seed=7)
    assert np.allclose(p1, p2)
    assert p1.shape == (11, 50)


def test_simulation_summary_fields():
    paths = np.array([[100.0, 100.0, 100.0], [90.0, 100.0, 110.0]])
    result = summarize_simulation(paths)
    assert result['terminal_median'] == 100.0
    assert 0 <= result['probability_terminal_below_initial'] <= 1
