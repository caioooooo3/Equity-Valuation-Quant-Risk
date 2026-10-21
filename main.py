from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import extract_latest_free_cash_flow, get_cashflow_statement, get_company_snapshot, get_price_history
from src.metrics import compute_market_metrics
from src.monte_carlo import simulate_gbm, summarize_simulation
from src.reporting import save_dcf_sensitivity_heatmap, save_drawdown_chart, save_monte_carlo_paths, save_price_chart, save_terminal_histogram
from src.valuation import dcf_equity_value, dcf_sensitivity_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Equity valuation and quantitative risk analysis.')
    parser.add_argument('--ticker', required=True)
    parser.add_argument('--benchmark', default='^GSPC')
    parser.add_argument('--period', default='5y')
    parser.add_argument('--risk-free', type=float, default=0.05)
    parser.add_argument('--forecast-growth', type=float, default=0.06)
    parser.add_argument('--wacc', type=float, default=0.09)
    parser.add_argument('--terminal-growth', type=float, default=0.025)
    parser.add_argument('--forecast-years', type=int, default=5)
    parser.add_argument('--mc-sims', type=int, default=10_000)
    parser.add_argument('--mc-days', type=int, default=252)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fcf', type=float, default=None)
    parser.add_argument('--cash', type=float, default=None)
    parser.add_argument('--debt', type=float, default=None)
    parser.add_argument('--shares', type=float, default=None)
    parser.add_argument('--output-dir', default='outputs')
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return None if np.isnan(value) or np.isinf(value) else value
    return value


def resolve_dcf_inputs(args: argparse.Namespace, snapshot: dict) -> dict[str, float]:
    if args.fcf is not None:
        fcf = args.fcf
    else:
        fcf = extract_latest_free_cash_flow(get_cashflow_statement(args.ticker))

    cash = args.cash if args.cash is not None else snapshot.get('totalCash')
    debt = args.debt if args.debt is not None else snapshot.get('totalDebt')
    shares = args.shares if args.shares is not None else snapshot.get('sharesOutstanding')

    missing = [name for name, value in {'cash': cash, 'debt': debt, 'shares': shares}.items() if value is None]
    if missing:
        raise ValueError('Missing DCF inputs: ' + ', '.join(missing) + '. Supply them using CLI overrides.')

    return {
        'free_cash_flow': float(fcf),
        'cash': float(cash),
        'debt': float(debt),
        'shares_outstanding': float(shares),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir) / args.ticker.replace('^', '').replace('.', '_')
    out.mkdir(parents=True, exist_ok=True)

    print(f'Downloading {args.ticker} and benchmark {args.benchmark}...')
    prices = get_price_history(args.ticker, args.period)
    benchmark_prices = get_price_history(args.benchmark, args.period)
    snapshot = get_company_snapshot(args.ticker)

    metrics = compute_market_metrics(prices, benchmark_prices, args.risk_free)
    paths = simulate_gbm(prices, days=args.mc_days, simulations=args.mc_sims, seed=args.seed)
    mc_summary = summarize_simulation(paths)

    dcf_result = None
    dcf_inputs = None
    sensitivity = None
    dcf_error = None

    try:
        dcf_inputs = resolve_dcf_inputs(args, snapshot)
        dcf_result = dcf_equity_value(
            **dcf_inputs,
            forecast_growth=args.forecast_growth,
            wacc=args.wacc,
            terminal_growth=args.terminal_growth,
            forecast_years=args.forecast_years,
        )
        sensitivity = dcf_sensitivity_table(
            **dcf_inputs,
            base_growth=args.forecast_growth,
            base_wacc=args.wacc,
            terminal_growth=args.terminal_growth,
            forecast_years=args.forecast_years,
        )
    except Exception as exc:
        dcf_error = str(exc)

    pd.DataFrame([{'metric': k, 'value': v} for k, v in metrics.items()]).to_csv(out / 'metrics.csv', index=False)
    if sensitivity is not None:
        sensitivity.to_csv(out / 'dcf_sensitivity.csv')

    save_price_chart(prices, out / 'price_history.png')
    save_drawdown_chart(prices, out / 'drawdown.png')
    save_monte_carlo_paths(paths, out / 'monte_carlo_paths.png')
    save_terminal_histogram(paths, out / 'monte_carlo_terminal_distribution.png')
    if sensitivity is not None:
        save_dcf_sensitivity_heatmap(sensitivity, out / 'dcf_sensitivity_heatmap.png')

    summary = {
        'ticker': args.ticker,
        'benchmark': args.benchmark,
        'period': args.period,
        'snapshot': snapshot,
        'market_metrics': metrics,
        'monte_carlo': mc_summary,
        'dcf_assumptions': {
            'forecast_growth': args.forecast_growth,
            'wacc': args.wacc,
            'terminal_growth': args.terminal_growth,
            'forecast_years': args.forecast_years,
        },
        'dcf_inputs': dcf_inputs,
        'dcf_result': dcf_result,
        'dcf_error': dcf_error,
    }
    with (out / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(json_safe(summary), f, indent=2, ensure_ascii=False)

    print('\n=== Market metrics ===')
    for key, value in metrics.items():
        print(f'{key:28s}: {value:.4f}')

    print('\n=== Monte Carlo ===')
    for key, value in mc_summary.items():
        print(f'{key:38s}: {value:.4f}')

    print('\n=== DCF ===')
    if dcf_result is not None:
        for key, value in dcf_result.items():
            print(f'{key:28s}: {value:,.2f}')
    else:
        print('DCF was skipped:', dcf_error)

    print(f'\nOutputs saved to: {out.resolve()}')


if __name__ == '__main__':
    main()
