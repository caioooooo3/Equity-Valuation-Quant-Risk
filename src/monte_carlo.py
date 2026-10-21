from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import log_returns


def simulate_gbm(
    prices: pd.Series,
    days: int = 252,
    simulations: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """Simulate price paths using historical daily log-return moments."""
    if days < 1:
        raise ValueError('days must be at least 1.')
    if simulations < 1:
        raise ValueError('simulations must be at least 1.')

    returns = log_returns(prices)
    if len(returns) < 2:
        raise ValueError('At least two return observations are required.')

    daily_mu = float(returns.mean())
    daily_sigma = float(returns.std(ddof=1))
    start_price = float(prices.dropna().iloc[-1])

    rng = np.random.default_rng(seed)
    shocks = rng.normal(
        loc=daily_mu,
        scale=daily_sigma,
        size=(days, simulations),
    )
    log_paths = np.vstack([np.zeros(simulations), np.cumsum(shocks, axis=0)])
    return start_price * np.exp(log_paths)


def summarize_simulation(paths: np.ndarray) -> dict[str, float]:
    if paths.ndim != 2 or paths.shape[0] < 2:
        raise ValueError('paths must be a 2D matrix with at least two time rows.')

    initial = float(paths[0, 0])
    terminal = paths[-1, :]
    q05, q50, q95 = np.quantile(terminal, [0.05, 0.50, 0.95])
    var_95_abs = max(0.0, initial - float(q05))

    return {
        'initial_price': initial,
        'terminal_p05': float(q05),
        'terminal_median': float(q50),
        'terminal_p95': float(q95),
        'probability_terminal_below_initial': float(np.mean(terminal < initial)),
        'simulation_var_95_abs': float(var_95_abs),
        'simulation_var_95_pct': float(var_95_abs / initial) if initial else float('nan'),
    }
