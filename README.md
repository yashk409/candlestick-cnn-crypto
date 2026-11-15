# Candlestick-CNN — Re-Imaging Price Trends, on Equities and Crypto

A replication of the Jiang–Kelly–Xiu "(Re-)Imaging Price Trends" CNN, run twice: once on
US equities, and once on crypto as an out-of-distribution transfer test. Same architecture,
same scientific harness; only the data distribution changes.

## What is actually in here

- A **2×2 pre-registered config grid** (image window {5, 20} × forecast horizon {5, 20}),
  frozen before any run and treated as the search family it is.
- **Multiple-testing correction applied and reported**: 2 of the 4 crypto configs survive
  Bonferroni (Sharpe **1.47**, t = 3.45 and **1.09**, t = 2.54), both also clearing
  Benjamini–Hochberg at FDR 5%, with stationary block-bootstrap Sharpe confidence intervals.
- **Survivorship bias measured rather than assumed** (−0.001 and −0.213 for the two winners).
  Crypto inverts the usual problem: Binance retains delisted coins and their full history,
  so the point-in-time universe can be scored instead of merely bounded.
- **24/7 corrections** an equities harness gets wrong — annualisation by 365 (BTCUSDT prints
  on 100.0% of calendar days; using 252 understates every Sharpe by ~1.20×), an exactly
  H-day rebalance calendar, and gap samples dropped and counted.
- A **device-adaptive training pipeline**: one codebase across Apple M4 (`mps`) and
  Kaggle 2×T4 (`cuda`), ensemble members trained concurrently over a memory-mapped image store.

The write-up and figures are in `FINAL/` (`crypto_cnn_paper.pdf`, `crypto_cnn_short.pdf`,
`figures/`).

## Data

Not included — regenerate it. Crypto bars come from **Binance public klines**
(`data-api.binance.vision`, no API key required). The equities panel is built from a
licensed CRSP extract which cannot be redistributed; supply your own.

## Layout

| Path | What |
|---|---|
| `01_Data_Prep.ipynb`, `notebook_data_prep.ipynb` | universe construction and image generation |
| `Re_Imaging_Price_Trends_*.ipynb` | training and evaluation |
| `mac_dataset.py` | memory-mapped image-store dataset |
| `FINAL/` | paper, short-form write-up, figures |
