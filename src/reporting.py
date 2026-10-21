from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import drawdown_series


def save_price_chart(prices: pd.Series, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(prices.index, prices.values)
    ax.set_title(f"{prices.name or 'Asset'} — Adjusted Price History")
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_drawdown_chart(prices: pd.Series, output_path: Path) -> None:
    dd = drawdown_series(prices)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dd.index, dd.values)
    ax.set_title(f"{prices.name or 'Asset'} — Drawdown")
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_monte_carlo_paths(paths: np.ndarray, output_path: Path, max_paths: int = 100) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    n = min(max_paths, paths.shape[1])
    ax.plot(paths[:, :n], alpha=0.25)
    ax.set_title(f'Monte Carlo Price Paths — first {n} simulations')
    ax.set_xlabel('Trading day')
    ax.set_ylabel('Simulated price')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_terminal_histogram(paths: np.ndarray, output_path: Path) -> None:
    terminal = paths[-1, :]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(terminal, bins=60)
    ax.axvline(np.quantile(terminal, 0.05), linestyle='--', label='5th percentile')
    ax.axvline(np.median(terminal), linestyle='--', label='Median')
    ax.set_title('Monte Carlo Terminal Price Distribution')
    ax.set_xlabel('Terminal price')
    ax.set_ylabel('Frequency')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_dcf_sensitivity_heatmap(table: pd.DataFrame, output_path: Path) -> None:
    values = table.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(values, aspect='auto')
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_xticklabels(table.columns)
    ax.set_yticklabels(table.index)
    ax.set_xlabel(table.columns.name or 'FCF growth')
    ax.set_ylabel(table.index.name or 'WACC')
    ax.set_title('DCF Intrinsic Value per Share — Sensitivity')
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            label = 'NA' if np.isnan(value) else f'{value:,.2f}'
            ax.text(j, i, label, ha='center', va='center', fontsize=8)
    fig.colorbar(image, ax=ax, label='Value per share')
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
