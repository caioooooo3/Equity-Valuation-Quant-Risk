from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def log_returns(prices: pd.Series) -> pd.Series:
    prices = prices.astype(float).dropna()
    if (prices <= 0).any():
        raise ValueError('Prices must be strictly positive.')
    result = np.log(prices / prices.shift(1)).dropna()
    result.name = 'log_return'
    return result


def annualized_return_from_log_returns(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(np.expm1(returns.mean() * periods)) if not returns.empty else float('nan')


def annualized_volatility(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods)) if not returns.empty else float('nan')


def beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([asset_returns.rename('asset'), benchmark_returns.rename('benchmark')], axis=1, join='inner').dropna()
    if len(aligned) < 2:
        return float('nan')
    benchmark_var = aligned['benchmark'].var(ddof=1)
    if benchmark_var == 0 or np.isnan(benchmark_var):
        return float('nan')
    return float(aligned['asset'].cov(aligned['benchmark']) / benchmark_var)


def sharpe_ratio(annual_return: float, annual_volatility: float, risk_free_rate: float = 0.0) -> float:
    if annual_volatility <= 0 or np.isnan(annual_volatility):
        return float('nan')
    return float((annual_return - risk_free_rate) / annual_volatility)


def drawdown_series(prices: pd.Series) -> pd.Series:
    prices = prices.astype(float).dropna()
    dd = prices / prices.cummax() - 1.0
    dd.name = 'drawdown'
    return dd


def max_drawdown(prices: pd.Series) -> float:
    dd = drawdown_series(prices)
    return float(dd.min()) if not dd.empty else float('nan')


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    if not 0 < confidence < 1:
        raise ValueError('confidence must be between 0 and 1.')
    q = float(returns.quantile(1.0 - confidence))
    return max(0.0, -q)


def compute_market_metrics(prices: pd.Series, benchmark_prices: pd.Series, risk_free_rate: float = 0.0) -> dict[str, float]:
    asset_r = log_returns(prices)
    benchmark_r = log_returns(benchmark_prices)
    ann_return = annualized_return_from_log_returns(asset_r)
    ann_vol = annualized_volatility(asset_r)
    return {
        'annualized_return': ann_return,
        'annualized_volatility': ann_vol,
        'beta': beta(asset_r, benchmark_r),
        'sharpe_ratio': sharpe_ratio(ann_return, ann_vol, risk_free_rate),
        'max_drawdown': max_drawdown(prices),
        'historical_daily_var_95': historical_var(asset_r, 0.95),
    }
