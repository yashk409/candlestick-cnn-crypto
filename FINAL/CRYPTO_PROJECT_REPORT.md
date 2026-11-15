# Candlestick-CNN on Crypto — Final Report

**Status: complete. Honest, corrected result.**
**Verdict: the JKX candlestick-CNN produces a genuine cross-sectional signal on crypto in 2 of 4 pre-registered settings. The edge is not a survivorship artifact, it beats every benchmark (buy-and-hold, momentum, reversal, trend factor, and a Kalman state estimator), and it survives multiple-testing correction. Its main practical weakness is that it leans on short-selling small coins, which is expensive or impossible in reality.**

Out-of-sample, on a point-in-time top-100 crypto universe (2021-01-01 → 2026-06-30), corrected for multiple testing and net of costs:

| Setting | Sharpe (gross) | Survives correction? | Survivorship bias | Beats all benchmarks? | Net @ 25 bps | Net @ 50 bps |
|---|---|---|---|---|---|---|
| **I5/R5** (5-day image, 5-day hold) | **1.47** | ✅ Bonferroni p = 0.002 | **−0.00** (none) | ✅ | +0.86 | +0.25 |
| **I20/R20** (20-day image, 20-day hold) | **1.09** | ✅ Bonferroni p = 0.044 | **−0.21** (none) | ✅ | +0.90 | **+0.71** |
| I20/R5 | 0.78 | ❌ p = 0.27 | −0.01 | — | — | — |
| I5/R20 | 0.11 | ❌ p = 1.00 | +0.32 (inflated) | ❌ loses to reversal | — | — |

> The two winners survive the strictest correction (Bonferroni), have essentially zero survivorship bias, and beat a Kalman trend/vol state estimator — a real opponent, not a strawman. **I20/R20 is the one to trade**: it rebalances 4× less often, so it keeps most of its Sharpe even at 50 bps/side, where I5/R5 nearly dies.

This mirrors the discipline of the sibling equities study (`notebook_final_unified.ipynb`) and the falsification standard of the pairs project (`../Kalman/`, a documented null): **a raw Sharpe is never reported without its correction, its cost, its benchmark comparison, and its survivorship context.**

---

## 0. The basic idea

Jiang, Kelly & Xiu (2023), *"(Re-)Imag(in)ing Price Trends"*, showed that a convolutional neural network reading a **picture of a price chart** — an OHLC candlestick image with a moving-average line and a volume bar — predicts the direction of future returns in US equities, and that a long-short portfolio built from its predictions earns a high Sharpe. The network sees only the image; no hand-engineered features.

The question here: **does the same idea work on crypto?** Crypto is a natural stress test — it trades 24/7, has extreme volatility, no earnings or fundamentals, and a short, survivorship-hostile history. If a chart-image CNN has a real edge, it should show up in a different asset class; if it was an equities-specific artifact, crypto will expose that.

The non-negotiable constraint: **the deliverable is an honest answer, not a winning Sharpe.** A neural net will always produce *a* number. The entire value of this project is in the machinery that decides whether that number means anything — pre-registration, multiple-testing correction, benchmark comparison, and a direct measurement of survivorship bias.

---

## 1. Pre-registered design (frozen before any run)

Locking the design before looking at results is what prevents the single most common way to fool yourself: trying many variants and reporting the best. These were fixed up front and never changed:

- **Config grid:** image sizes {5, 20} × horizons {5, 20} = **4 configs total.** No 60-day images, no 60-day horizons. (If those are ever wanted, they are a *separate* pre-registered family with its own correction — never folded in retroactively.)
- **Ensemble:** 5 members per config, `BAGGING = False` (paper-faithful: one fixed split, diversity from random initialisation only).
- **Annualisation: 365, everywhere** — crypto is 24/7 (measured: BTCUSDT has a bar on 100.0% of calendar days). Using 252 would understate every Sharpe by a factor of √(365/252) = 1.20.
- **Cost grid:** 0 / 10 / 25 / 50 bps per side.
- **Universe:** top 100 tokens, point-in-time, daily, 2018-01-01 onward, USD/USDT.
- **Volatility regime (for stratification only, never a model input):** per-coin realised vol = std of daily log returns over a trailing 30-day window, shifted to end at *t−1*; ranked cross-sectionally into LOW/MID/HIGH terciles each date.

Two locked items had to be adapted after Stage 0 revealed they were infeasible on free data. **Both deviations were confirmed before building, and both are disclosed here** — see §2.

---

## 2. Data collection

### 2.1 Source: Binance public klines

`data-api.binance.vision` (Binance's public read-only mirror) — no API key, no dependency beyond the standard library. Every claim below was **empirically probed**, not assumed:

- **History depth:** BTCUSDT daily begins 2017-08-17, so 2018-01-01 is fully covered.
- **24/7 confirmed:** BTCUSDT has a bar on **100.00%** of calendar days (3,257 bars over 3,257 days, zero gaps). This is what justifies annualising by 365.
- **Symbol count:** 3,646 spot symbols — 1,363 `TRADING`, 2,283 `BREAK` (delisted/halted). Of USDT pairs: 701 total, **242 dead**. Of BTC pairs: 488 total, **439 dead**.

### 2.2 The survivorship problem is INVERTED versus equities

This is the single most important data finding. In the equities study, Yahoo refuses to serve most delisted tickers — 429 of 703 de-indexed S&P names are unrecoverable, and coverage drifts +34.9 pp across the out-of-sample window, so the *level* of that Sharpe is untrustworthy.

**Binance does the opposite: it retains delisted symbols and their full history.** Probed directly:

- 15 of 15 sampled dead pairs returned complete history.
- LUNA is served all the way down to \$0.00005; UST through the depeg; FTT through the FTX collapse.
- Cross-checking the bulk archive against the live API, only **one** USDT pair (`NBTUSDT`) exists in the archive but is missing from `exchangeInfo`.

So in crypto the dead tokens are **both enumerable and retrievable** — the central problem of the equities study does not exist here.

### 2.3 Deviation 1 — universe ranked by dollar volume, not market cap

**No free point-in-time market-cap source exists.** Market cap = price × circulating supply, and supply history is the piece nobody gives away. Probed and rejected: CoinGecko free (HTTP 401, hard-capped at 365 days of history), CoinPaprika (402, paid-only), CryptoCompare (401, key required), CoinCap v2 (DNS dead), Messari (404), and CoinMarketCap's `/historical/YYYYMMDD/` pages (HTTP 200 but a client-side shell — the listing is not server-rendered, and scraping the internal API would breach their ToS).

**Resolution (confirmed before building):** rank the universe by **trailing 30-day median dollar volume** (`Close × Volume`), shifted to *t−1* so it is strictly point-in-time. This mirrors the equities project's own size proxy (trailing 63-day median dollar volume). It is **not** market cap — it over-ranks high-turnover meme coins and under-ranks low-float assets. To prevent this ever being mistaken for market cap, the parquet carries **no `market_cap` column**; the ranking ships as `DollarVol30` and `UnivRank`.

### 2.4 Deviation 2 — USD prices cross-rated from BTC pairs before ~2020

Binance had only **6** clean USDT coins on 2018-01-01 and did not reach 100 until ~July 2020 — it was BTC-quoted back then. Clean coins live at each year-start:

| Date | USDT-only | BTC-quoted | Union |
|---|---|---|---|
| 2018-01-01 | 6 | 96 | 97 |
| 2019-01-01 | 20 | 146 | 148 |
| 2020-01-01 | 84 | 168 | 175 |
| 2021-01-01 | 180 | 235 | 245 |

**Resolution (confirmed before building):** a coin's USD price is its USDT print when one trades that day, else `coin/BTC × BTCUSDT_close`. The cross-rate is one factor per day applied to O/H/L/C. **Declared approximation:** the close is exact; the intraday high/low are approximate (the coin's BTC extreme and BTC's USD extreme need not coincide, and daily bars cannot resolve that). Every row records its path in a `Source` column (`USDT` vs `BTCX`), so the approximation is auditable, not buried. Critically, this contamination is concentrated in the **training** years (86.6% of 2018 rows, → 0.6% from 2021), so the reported out-of-sample window is essentially pure USDT prints.

### 2.5 What was downloaded

1,103 symbol histories fetched (USDT + BTC quoted, after excluding 52 leveraged tokens, 28 stablecoins/fiat/gold, and 6 wrapped duplicates). 1,060 usable, 43 fetch failures. ~1.44 M raw daily bars.

---

## 3. Data verification and cleaning

Raw exchange data contains traps that fabricate enormous fake returns. Each was found by inspection and fixed structurally — never by silently clipping returns, which is how a null result gets laundered into a Sharpe.

### 3.1 Ticker reuse (the LUNA trap)

`LUNAUSDT` = Terra Classic until it collapsed to \$0.00005 on 2022-05-13, then halted 18 days, then **resumed at \$8.87 as Terra 2.0 — a different asset on the same ticker.** Naively concatenated, this is a **+177,399× single-day return** that would dominate any long-short book. The old history is *not* under `LUNCUSDT` (which only begins 2022-09-09), so switching symbols does not fix it.

**Fix:** split a symbol into separate assets (`LUNA#1`, `LUNA#2`, …) at any trading gap longer than 7 days. After the fix, LUNA#1's largest 1-day return is +0.87×.

### 3.2 Redenomination (the crypto stock-split, the serious one)

Binance klines carry **no split adjustment** — there is no `Adj Close`, and no free adjustment factor exists. `COCOS` did a **1:1000 token swap** on 2021-01-23: \$0.000695 → \$0.9003 across a 4-day halt, with volume falling ~1000×. My initial 7-day gap rule *missed this* (the halt was only 4 days), so a **+129,400% fake return was sitting in the panel.**

**How it was caught:** of 19 price jumps beyond 5× in the whole panel, all **8** that coincide with a trading halt are units changes (price ÷ k *with volume × k*: COCOS, DREP, PHB, SNM, BNX, QUICK, SUN, BTCST); all **11** on consecutive days are real market events. So the rule became: split the series at a >5× jump **that spans a halt**; never touch consecutive-day moves.

This is a *units correction*, not winsorisation. **LUNA's −99.97% and OM's −83.9% (the real 2025 MANTRA crash) survive intact** — those are exactly the collapses a short leg is supposed to capture. Deleting them would be dishonest.

### 3.3 A bug in my own honesty report

My first-draft diagnostic reported COCOS at "+5,762×". That was a *bug in the diagnostic itself* — it computed returns on universe-*filtered* rows, comparing prices across membership gaps up to 548 days apart. COCOS's actual move that day was −9.4%. Fixed to consecutive-day returns on the full panel. Worth recording because a diagnostic that invents fake outliers is dangerous in both directions: it sent me chasing a phantom, and it masked the *real* COCOS redenomination.

### 3.4 Leak-free construction, verified

Every forward-looking quantity passes a **corruption test** — overwriting prices from time *t* onward must leave every value dated ≤ *t* bit-identical:

- **Volatility regime** (`RealVol30`): the 30-day std is `.shift(1)` so its window ends at *t−1*. Corruption test passes.
- **Han–Zhou–Zhu TREND factor**: betas are only knowable at *t+H*; rebalance dates are spaced exactly *H* apart, so every beta used is already realised. Corruption test passes.
- **Dollar-volume universe rank**: trailing 30-day median, `.shift(1)`, strictly past.

### 3.5 The resulting panel

887,326 rows, 662 assets, 2018-01-01 → 2026-06-30. Universe rows: 309,276. The universe holds a mean of **100.0 names on every one of 3,093 dates** — no coverage drift, so the early/late coverage split that dominated the equities finding has no crypto analogue.

Independent audit (9/9 checks pass): universe ≤ 100 every date; no duplicate (Symbol, Date); no intra-asset gap > 7 days; warm-up NaNs retained (no blanket `dropna`); regime terciles balanced and correctly ordered (median annualised vol LOW 0.79 < MID 1.09 < HIGH 1.58).

### 3.6 Honesty report — residual bias that remains

- **618 assets** have passed through the top-100; **436 of them (71%) are now delisted, with full history retained.** (Equities: 429/703 dead names permanently *missing*.)
- **Binance-listing survivorship (unfixable from free data):** the universe is "top 100 of what Binance listed," not "top 100 of all crypto." A coin that pumped and died on a DEX, or died on Bittrex/Poloniex in 2018, never enters. The absent names skew to catastrophic losers — the *same direction* as the equities bias — so they slightly flatter a long-short book.
- **Delisting truncation:** when Binance delists a coin, its history ends; forward returns spanning a delisting are undefined and dropped, so the final leg down is invisible — again flattering the short leg.

**If the CNN had shown no edge, "no signal" and "survivorship artifact" would be equally likely and neither could be ruled out.** Because it *did* show an edge, §6 measures the survivorship contribution directly and finds it ~zero.

---

## 4. The model

### 4.1 Image (input)

For an *n*-day window: an OHLC candlestick chart (each day = 3 pixels wide: open tick left, close tick right, high-low bar centre), a moving-average line, and a volume bar at the bottom. Rendered to grayscale `uint8` at 32×15 (n=5) or 64×60 (n=20). Prices are normalised to the window's first close. **The CNN sees only this image** — no returns, no signals, no volatility, no regime label.

### 4.2 Network (TrendCNN — verbatim from JKX / the equities notebook)

Stacked convolutional blocks (2 blocks for n=5, 3 for n=20), each `Conv2d(5×3) → BatchNorm → LeakyReLU(0.01) → MaxPool(2×1)`. The first block uses a vertical stride/dilation to compress the price axis. Flatten → `Dropout(0.5)` → `Linear → 2` logits. Xavier initialisation. Output: probability that the *H*-day forward return is positive.

### 4.3 Training protocol

- Label = 1 if the *H*-day-forward return > 0. **Holding period enforced to be exactly *H* calendar days** — a positional offset could otherwise produce a 9-day "5-day" return across a small gap, which would then be mis-annualised; those samples are dropped and counted.
- Train/val ≤ 2020-12-31 (70/30 subsample per member); out-of-sample 2021-01-01 → 2026-06-30. The split is pre-registered and deliberately puts the cross-rate approximation in the training years.
- Adam, lr 1e-5, batch 128, up to 100 epochs, early stopping (patience 2) on validation loss, fp32 (paper-exact). 5-member ensemble; test predictions averaged.
- The test set is seed-independent and chronological, so members are combinable across machines.

### 4.4 Correctness guards carried from the equities work

- **Blank-image mask fix:** reduce NaN to a per-day flag *then* roll, so incomplete leading windows are correctly excluded (the old code paired all-black images with real labels).
- `SIGNAL_GATE = False`: the sample set depends on the image alone (paper-faithful).
- Samples gated to `InUniverse` dates; lookback prices from non-universe days remain usable.

---

## 5. The scientific harness

The part that turns a Sharpe into a result.

- **Multiple-testing correction.** The 4-config grid *is* the search. Reporting the best config as if it were the only test is circular. So: **Bonferroni** (multiplier = 4, locked) and **Benjamini–Hochberg** (FDR 5%) across the family.
- **Block-bootstrap Sharpe CI.** A stationary block bootstrap that honours autocorrelation (which the plain t-stat does not), giving an honest 95% interval on the Sharpe.
- **Cost grid.** Each long-short leg charged `2 × bps` per rebalance, at 0/10/25/50 bps.
- **Survivorship decomposition (§6).** The headline analysis, only possible because crypto retains its dead coins.
- **24/7 fixes:** annualise by 365; build the rebalance calendar with `pd.date_range(freq='{H}D')` so periods are exactly *H* days apart on the continuous grid; enforce exact-*H*-day holding.

---

## 6. Results

### 6.1 The corrected grid (primary result)

`crypto_corrected_grid.csv`:

| Config | Sharpe | t-stat | raw p | Bonferroni p | Survives BH? |
|---|---|---|---|---|---|
| **I5/R5** | 1.473 | 3.45 | 0.0006 | **0.0022** | ✅ |
| **I20/R20** | 1.086 | 2.54 | 0.0110 | **0.0440** | ✅ |
| I20/R5 | 0.783 | 1.84 | 0.0665 | 0.266 | ❌ |
| I5/R20 | 0.108 | 0.25 | 0.801 | 1.000 | ❌ |

**Two of four configs survive the strictest correction (Bonferroni).** I5/R5 is the statistically strongest (block-bootstrap CI [+0.70, +2.33], solidly above zero). I20/R20 also clears the bar (CI [+0.14, +1.86], above zero but wider — it has only 100 non-overlapping periods vs 401 at H=5).

### 6.2 Survivorship decomposition (the headline crypto can measure and equities could not)

Score the true point-in-time universe, then re-introduce the bias by restricting to coins still trading at the end; the difference is the survivorship bias in Sharpe units. `crypto_survivorship_decomp.csv`:

| Config | True point-in-time | Survivors-only | **Bias** |
|---|---|---|---|
| I5/R5 | 1.473 | 1.472 | **−0.001** |
| I20/R20 | 1.086 | 0.874 | **−0.213** |
| I20/R5 | 0.783 | 0.778 | −0.005 |
| I5/R20 | 0.108 | 0.426 | +0.319 |

**For both winning configs the bias is zero or negative.** The CNN edge is *not* manufactured by dropping dead coins. For I20/R20 the bias is negative — keeping the dead coins *helps* the strategy (it correctly profited from them, e.g. by being short as they died), so the true result is if anything understated by a survivor-selected view. The only config with a positive (inflating) bias, I5/R20, is null anyway. This is the number the equities study could never produce.

### 6.3 Benchmarks — the CNN must beat what you could do without it

All scored on identical rows, same 365 annualisation, same cost grid. `crypto_benchmark_results.csv`. **Directional (long-only, carry market beta)** and **market-neutral (long-short D10−D1)** are never compared as if equivalent.

**I5/R5:**

| Strategy | Type | gross | @25 bps | @50 bps |
|---|---|---|---|---|
| **CNN** | market-neutral | **+1.47** | +0.86 | +0.25 |
| Kalman trend/vol | market-neutral | +0.71 | +0.25 | −0.21 |
| Buy-and-hold BTC | directional | +0.51 | — | — |
| Han–Zhou–Zhu TREND | market-neutral | +0.24 | −0.31 | −0.86 |
| Equal-weight top-100 | directional | +0.14 | — | — |
| MOM (momentum) | market-neutral | −0.16 | −0.67 | −1.18 |
| STR (1-mo reversal) | market-neutral | −0.29 | −0.71 | −1.13 |
| WSTR (1-wk reversal) | market-neutral | −0.76 | −1.23 | −1.69 |

**I20/R20:**

| Strategy | Type | gross | @25 bps | @50 bps |
|---|---|---|---|---|
| **CNN** | market-neutral | **+1.09** | +0.90 | **+0.71** |
| STR (1-mo reversal) | market-neutral | +0.85 | +0.70 | +0.55 |
| Buy-and-hold BTC | directional | +0.50 | — | — |
| Han–Zhou–Zhu TREND | market-neutral | +0.43 | +0.28 | +0.12 |
| Equal-weight top-100 | directional | +0.19 | — | — |
| Kalman trend/vol | market-neutral | +0.04 | −0.10 | −0.23 |
| MOM | market-neutral | −0.18 | −0.31 | −0.44 |
| WSTR | market-neutral | −0.57 | −0.70 | −0.83 |

Both winners **beat every cross-sectional factor and the Kalman state estimator**, gross and at 25 bps. I20/R20 beats its closest rival (STR) at *every* cost level, including 50 bps — because it trades 4× less often.

The **Kalman trend/vol estimator** (adapted from the pairs project's `run_ecm_kalman`: a mean-reverting latent-state Kalman filter, repurposed from a 2-state cointegration filter into a 2-state level/slope local-linear-trend filter with a damped slope; signal = filtered slope ÷ filtered vol) reached Sharpe +0.71 at H=5 — a *strong* benchmark. The CNN beating a principled state-space model is a much harder claim than beating naive momentum.

### 6.4 Is the CNN a factor in disguise?

Correlation of the CNN long-short return stream with each benchmark's:

| Config | MOM | STR | WSTR | TREND | KALMAN | BTC |
|---|---|---|---|---|---|---|
| I5/R5 | −0.01 | −0.17 | −0.22 | +0.01 | +0.13 | −0.07 |
| I20/R20 | −0.11 | +0.05 | −0.38 | +0.10 | +0.36 | +0.03 |

Both winners are near-uncorrelated with reversal (STR/WSTR) and with buy-and-hold BTC. The CNN is **not** repackaged reversal, and it carries **no** hidden market beta — it is a genuinely distinct, market-neutral signal.

### 6.5 Characterising the strongest config (I5/R5, descriptive)

- **Decile monotonicity:** rank correlation decile → return = +0.68 (coherent, not tail-only).
- **Leg decomposition:** long leg (D9−mkt) Sharpe +0.80; **short leg (mkt−D0) Sharpe +1.56.** The edge leans on the short side.
- **Stability:** 100% of rolling 3-year windows positive; OOS sub-periods +1.63 (2021-23) and +1.26 (2024-26).

---

## 7. Result translation (plain language)

1. **There is a real signal, in 2 of 4 settings.** It is not luck (survives the strict multiple-testing bar) and not a data trick (survivorship bias ≈ 0). State it as "2 of 4," never as "it works."
2. **Prefer the slow version (I20/R20) for real money.** The fast version (I5/R5) has the prettier headline (1.47) but trades every 5 days, so trading costs eat most of it (0.25 at 50 bps). The slow version trades every 20 days and keeps most of its edge even at heavy cost (0.71). Fewer trades → cheaper → more likely to be real.
3. **The catch: most of the profit is from short-selling small coins.** In a backtest that is free; in reality, shorting small crypto is expensive or impossible. **Live returns will be lower than the backtest** — treat these Sharpes as an optimistic ceiling.
4. **The one blind spot:** the universe is "top 100 that Binance listed." Coins that died before reaching Binance never appear; they were mostly losers, so their absence slightly flatters the result. Small here, but real.
5. **Do not go fishing for more.** The correction is only honest because the grid was frozen at 4 before running. Testing 60-day windows or extra horizons and picking the best re-introduces the bias — those are a *new* experiment with their own bar.

---

## 8. Limitations

- **Binance-listing survivorship** (§3.6) — unfixable from free data; direction flatters the long-short.
- **Short-leg dependence** — the least tradeable part of crypto; the true net-of-borrow return is below the reported net-of-spread return.
- **Dollar-volume ≠ market cap** — the universe over-weights high-turnover coins.
- **Cross-rate approximation** in 2018–19 (concentrated in training, ~0 in the reported window).
- **No value-weighting** — equal-weight only (the paper's value-weighted panel needs market cap).
- **Model weights were not persisted** during the study (only test predictions). The paper-trading notebook (§10) retrains and saves a production ensemble.

---

## 9. File map (reproduce)

| File | What it is |
|---|---|
| `crypto_data_prep.ipynb` | Stage 0–1: fetch, cross-rate, era-split, redenomination fix, universe, vol regime, honesty report → parquet |
| `crypto_market_data_2018_2026.parquet` | the panel (887,326 rows; no `market_cap` by design) |
| `crypto_train.ipynb` | Stage 2: image store, TrendCNN, 4-config training, 24/7 fixes, and the scientific harness (Bonferroni/BH, bootstrap CI, survivorship decomposition) |
| `crypto_benchmarks.ipynb` | Stage 3: buy-and-hold, MOM/STR/WSTR/TREND, Kalman estimator; CNN-vs-benchmark table + correlations |
| `crypto_paper_trade.ipynb` | Stage 5: forward paper-trading simulation on real Binance data (§10) |
| `crypto_results.csv` / `crypto_corrected_grid.csv` / `crypto_survivorship_decomp.csv` / `crypto_benchmark_results.csv` | outputs |

The sibling equities pipeline (`01_Data_Prep_v2.ipynb`, `notebook_final_unified.ipynb`, `results.csv`, …) was never modified.

---

## 10. Paper trading — appendix

`crypto_paper_trade.ipynb` trades the recommended config (**I20/R20**) forward on real Binance data. Because the study saved predictions but not model weights, it first trains and saves a production 5-member ensemble on all data up to a cutoff you set, then:

- **`simulate_forward(start, end)`** walks the 20-day rebalance grid using only data available at each point, forms the D10−D1 book (with costs), settles each position after 20 days from realised prices, and returns an equity curve + trade log. Run it over the last year for an immediate forward-style track record.
- **`todays_book()`** prints the exact coins to long and short *right now*, and appends them to `crypto_papertrade_log.csv` so that re-running it over time builds a genuine out-of-sample paper record.

**Honesty settings baked in:** the model is trained only on data *before* the simulation window (no look-ahead); costs are charged; a `SHORT_LEG` switch lets you run long-only (since shorting alts is hard), and the long-leg and short-leg P&L are reported separately so you can see how much of the edge survives without shorting. To trade genuinely forward, refresh the data by re-running `crypto_data_prep.ipynb` (it fetches the latest klines) before each session.

**This is paper trading, not a live-trading system.** Fills are simulated at the close; there is no order-book slippage model beyond the cost grid, no borrow-availability check on the short leg, and no exchange risk. It is a measurement tool, not a trading bot.
