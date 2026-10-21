# Methodology

## 1. Historical prices

Adjusted closing prices are requested through `yfinance`. Returns are calculated from the historical series selected by the user.

## 2. Log returns

`r_t = ln(P_t / P_{t-1})`

Log returns are used because they aggregate naturally across time and work well in the simulation workflow.

## 3. Annualized metrics

Annualized return is obtained from the average daily log return over 252 trading days. Volatility is annualized with the square-root-of-time convention.

## 4. Beta

`beta = Cov(asset, benchmark) / Var(benchmark)`

This is a historical sample estimate and depends on the benchmark and period selected.

## 5. Sharpe ratio

`Sharpe = (annualized return - risk-free rate) / annualized volatility`

## 6. Drawdown

Drawdown compares each price with the historical running peak. Maximum drawdown is the most negative observation.

## 7. Historical VaR

The reported daily 95% historical VaR is based on the empirical 5th percentile of daily log returns. It is an educational risk statistic, not a production or regulatory risk model.

## 8. DCF

The simplified DCF projects Free Cash Flow using one explicit growth rate, discounts the flows by WACC and applies a Gordon Growth terminal value.

`TV = FCF_n * (1 + g_terminal) / (WACC - g_terminal)`

Enterprise value is the present value of explicit FCFs plus the present value of terminal value. Equity value adds cash and subtracts debt. Per-share value divides equity value by shares outstanding.

### Important limitation

A professional valuation normally forecasts operating drivers, taxes, reinvestment, working capital and capital structure in greater detail. This model is intentionally compact and assumption-sensitive.

## 9. Monte Carlo

The simulation estimates historical daily mean and volatility from log returns and generates independent normal shocks. Simulated prices are built cumulatively from these shocks.

This resembles a simplified GBM-style process.

### Important limitations

- constant mean and volatility;
- normally distributed log returns;
- independent shocks;
- no jumps;
- no regime changes;
- no volatility clustering.

## 10. Reproducibility

A configurable random seed makes simulations reproducible. Unit tests use synthetic data so core numerical logic can be validated without internet access.
