from __future__ import annotations

import numpy as np
import pandas as pd


def dcf_equity_value(
    free_cash_flow: float,
    cash: float,
    debt: float,
    shares_outstanding: float,
    forecast_growth: float = 0.06,
    wacc: float = 0.09,
    terminal_growth: float = 0.025,
    forecast_years: int = 5,
) -> dict[str, float]:
    """Simplified DCF based on FCF growth and Gordon terminal value."""
    if forecast_years < 1:
        raise ValueError('forecast_years must be at least 1.')
    if wacc <= terminal_growth:
        raise ValueError('WACC must be greater than terminal growth.')
    if shares_outstanding <= 0:
        raise ValueError('shares_outstanding must be positive.')

    pv_fcfs = 0.0
    projected_fcf = float(free_cash_flow)
    for year in range(1, forecast_years + 1):
        projected_fcf *= 1.0 + forecast_growth
        pv_fcfs += projected_fcf / ((1.0 + wacc) ** year)

    terminal_value = projected_fcf * (1.0 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value / ((1.0 + wacc) ** forecast_years)
    enterprise_value = pv_fcfs + pv_terminal
    equity_value = enterprise_value + cash - debt
    intrinsic_value_per_share = equity_value / shares_outstanding

    return {
        'pv_explicit_fcfs': float(pv_fcfs),
        'pv_terminal_value': float(pv_terminal),
        'enterprise_value': float(enterprise_value),
        'equity_value': float(equity_value),
        'intrinsic_value_per_share': float(intrinsic_value_per_share),
    }


def dcf_sensitivity_table(
    free_cash_flow: float, cash: float, debt: float, shares_outstanding: float,
    base_growth: float = 0.06, base_wacc: float = 0.09,
    terminal_growth: float = 0.025, forecast_years: int = 5,
    growth_spread: float = 0.02, wacc_spread: float = 0.02, points: int = 5,
) -> pd.DataFrame:
    growth_rates = np.linspace(base_growth - growth_spread, base_growth + growth_spread, points)
    wacc_rates = np.linspace(base_wacc - wacc_spread, base_wacc + wacc_spread, points)
    values = []
    for current_wacc in wacc_rates:
        row = []
        for growth in growth_rates:
            try:
                result = dcf_equity_value(
                    free_cash_flow, cash, debt, shares_outstanding,
                    float(growth), float(current_wacc), terminal_growth, forecast_years
                )
                row.append(result['intrinsic_value_per_share'])
            except ValueError:
                row.append(float('nan'))
        values.append(row)
    table = pd.DataFrame(values, index=[f'{x:.1%}' for x in wacc_rates], columns=[f'{x:.1%}' for x in growth_rates])
    table.index.name = 'WACC'
    table.columns.name = 'FCF growth'
    return table
