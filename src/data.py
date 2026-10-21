from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf


def get_price_history(ticker: str, period: str = '5y') -> pd.Series:
    """Return adjusted closing prices as a clean Series."""
    data = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if data.empty or 'Close' not in data.columns:
        raise ValueError(f'No price history returned for {ticker!r}.')
    prices = data['Close'].dropna().astype(float).copy()
    prices.name = ticker.upper()
    if len(prices) < 30:
        raise ValueError(f'Only {len(prices)} observations returned for {ticker!r}.')
    return prices


def get_company_snapshot(ticker: str) -> dict[str, Any]:
    """Fetch a compact set of company/fundamental fields."""
    obj = yf.Ticker(ticker)
    try:
        info = obj.info or {}
    except Exception:
        info = {}
    return {
        'symbol': ticker.upper(),
        'longName': info.get('longName'),
        'currency': info.get('currency'),
        'sector': info.get('sector'),
        'industry': info.get('industry'),
        'marketCap': info.get('marketCap'),
        'enterpriseValue': info.get('enterpriseValue'),
        'trailingPE': info.get('trailingPE'),
        'forwardPE': info.get('forwardPE'),
        'priceToBook': info.get('priceToBook'),
        'enterpriseToEbitda': info.get('enterpriseToEbitda'),
        'totalCash': info.get('totalCash'),
        'totalDebt': info.get('totalDebt'),
        'sharesOutstanding': info.get('sharesOutstanding'),
    }


def get_cashflow_statement(ticker: str) -> pd.DataFrame:
    """Return annual cash-flow statement when available."""
    try:
        statement = yf.Ticker(ticker).cashflow
    except Exception as exc:
        raise ValueError(f'Could not retrieve cash-flow statement: {exc}') from exc
    if statement is None or statement.empty:
        raise ValueError(f'No annual cash-flow statement returned for {ticker!r}.')
    return statement


def extract_latest_free_cash_flow(statement: pd.DataFrame) -> float:
    """Extract the most recent Free Cash Flow value."""
    for label in statement.index:
        normalized = str(label).strip().lower().replace('_', ' ').replace(' ', '')
        if normalized == 'freecashflow':
            values = pd.to_numeric(statement.loc[label], errors='coerce').dropna()
            if not values.empty:
                return float(values.iloc[0])
    raise ValueError('Free Cash Flow was not found. Supply it manually with --fcf.')
