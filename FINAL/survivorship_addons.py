# =============================================================================
# SURVIVORSHIP ADD-ONS
#
# CELL A + B  -> paste into 01_Data_Prep_v2.ipynb  (A after §3 download, B after §6)
# CELL C      -> paste into notebook_final_unified.ipynb (after the main run)
#                It RE-SCORES the cached predictions. No retraining. Costs seconds.
# =============================================================================


# ============================== CELL A: retry failed tickers =================
# Paste AFTER the §3 download cell. Then RE-RUN §4, §5, §6 so the new rows flow through.

def symbol_variants(t):
    """Yahoo's symbology for old / odd tickers differs from the index files."""
    v = {t}
    v.add(t.replace('-', '.'))            # BRK-B -> BRK.B
    v.add(t.replace('.', '-'))            # BF.B  -> BF-B
    if len(t) > 1 and t.endswith('Q'):    # bankruptcy suffix: AAMRQ -> AAMR
        v.add(t[:-1])
    for suf in ('.A', '.B', '-A', '-B'):
        if t.endswith(suf):
            v.add(t[:-2])
    return [x for x in v if x and x != t]

print(f"retrying {len(missing):,} failed tickers with symbol variants...")
recovered_frames, still_missing = [], []

for t in tqdm(missing, desc="retry"):
    got = None
    for cand in symbol_variants(t):
        try:
            d = yf.download(cand, start=START_DATE, end=END_DATE,
                            progress=False, auto_adjust=False)
        except Exception:
            continue
        if d is None or d.empty:
            continue
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        if 'Adj Close' not in d.columns:
            d['Adj Close'] = d['Close']
        d = d.dropna(subset=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
        if len(d) < MIN_HISTORY:
            continue
        d['Ticker'] = t                   # KEEP the index-file symbol so membership joins still work
        got = d.reset_index()[['Date','Ticker','Open','High','Low','Close','Adj Close','Volume']]
        break
    time.sleep(0.3)
    if got is not None:
        recovered_frames.append(got)
    else:
        still_missing.append(t)

if recovered_frames:
    extra = pd.concat(recovered_frames, ignore_index=True)
    raw_df = pd.concat([raw_df, extra], ignore_index=True)
    print(f"\nrecovered {len(recovered_frames):,} tickers ({len(extra):,} rows) via symbol variants")
else:
    print("\nno additional tickers recovered")
print(f"still unavailable: {len(still_missing):,}")
missing = still_missing
print("\n>>> NOW RE-RUN cells §4 (adjust_ohlc), §5 (signals), §6 (InIndex) <<<")


# ============================== CELL B: coverage by year ====================
# Paste AFTER §6. This is the honest denominator: how much of the TRUE index we actually have.

def coverage_report(final_df, intervals, oos_start=2005):
    got = set(final_df['Ticker'].unique())
    rows = []
    for y in sorted(final_df['Date'].dt.year.unique()):
        d = pd.Timestamp(f'{y}-07-01')                       # mid-year snapshot
        live   = intervals[(intervals['start'] <= d) & (intervals['end'] > d)]['ticker']
        true_n = live.nunique()
        have_n = live[live.isin(got)].nunique()
        rows.append({'year': y, 'true_members': true_n, 'available': have_n,
                     'coverage_%': 100.0 * have_n / max(true_n, 1)})
    cov = pd.DataFrame(rows)

    print("=" * 60)
    print("COVERAGE BY YEAR — available in-index names vs true membership")
    print("=" * 60)
    print(cov.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    oos = cov[cov['year'] >= oos_start]
    drift = oos['coverage_%'].iloc[-1] - oos['coverage_%'].iloc[0]
    print(f"\nOOS window ({oos_start}+):")
    print(f"  mean coverage : {oos['coverage_%'].mean():.1f}%")
    print(f"  range         : {oos['coverage_%'].min():.0f}% -> {oos['coverage_%'].max():.0f}%")
    print(f"  min names/yr  : {int(oos['available'].min())} "
          f"(>=100 gives usable deciles; >=200 is comfortable)")
    print(f"  DRIFT         : {drift:+.1f} pp across the OOS window")
    print("\n  The drift is the real issue: late years are near-complete while early years")
    print("  are survivor-selected, so performance is NOT measured on a constant universe.")
    print("\n  Note the direction of the bias is genuinely ambiguous for a LONG-SHORT book:")
    print("  most index departures are acquisitions (winners, takeover premium), not")
    print("  bankruptcies (losers). So the honest claim is 'the universe is incomplete and")
    print("  time-varying, so the LEVEL of the Sharpe is not trustworthy' -- NOT 'the Sharpe")
    print("  is inflated'. The sub-period test (Cell C) is what settles whether it matters.")
    return cov

cov = coverage_report(final_df, intervals)


# ============================== CELL C: sub-period robustness ================
# Paste into notebook_final_unified.ipynb AFTER the main run.
# Re-scores the CACHED predictions on coverage sub-periods. NO retraining.

def subperiod_report(splits=None):
    if splits is None:
        splits = {
            'full  2005-2026'          : ('2005-01-01', '2026-12-31'),
            'early 2005-2014 (low cov)': ('2005-01-01', '2014-12-31'),
            'late  2015-2026 (high cov)': ('2015-01-01', '2026-12-31'),
        }
    out = []
    for n_days in IMAGE_SIZES:
        store = load_store(n_days)
        if store is None:
            continue
        for horizon in HORIZONS:
            cfg = f"I{n_days}/R{horizon}"
            # rebuild eval_base exactly as the main loop did
            rows, label, base = label_for_horizon(store, df_market, horizon)
            te = test_positions(base)
            if len(te) == 0:
                continue
            eval_base = base.iloc[te].reset_index(drop=True)

            # pull the CACHED member predictions -- no training
            preds = []
            for i in range(ENSEMBLE_SIZE):
                p = pred_path(n_days, horizon, 1000 + i)
                if os.path.exists(p):
                    preds.append(np.load(p))
            if not preds or {len(p) for p in preds} != {len(te)}:
                log(f"{cfg}: cached predictions missing/stale, skipping.")
                continue
            eval_base['CNN_Prob'] = np.mean(preds, axis=0)
            eval_base = add_trend_factor(eval_base, horizon)

            for name, (a, b) in splits.items():
                sub = eval_base[(eval_base['Date'] >= a) & (eval_base['Date'] <= b)]
                if sub.empty:
                    continue
                for s in ['CNN_Prob', 'MOM', 'STR', 'WSTR', 'TREND']:
                    perf = portfolio_performance(s, sub, horizon)
                    out.append({'Config': cfg, 'Period': name, 'Strategy': s, **perf})
        del store
        free_memory()

    df = pd.DataFrame(out)
    df.to_csv(os.path.join(OUT_DIR, 'subperiod_results.csv'), index=False)

    print("\n" + "=" * 74)
    print("SUB-PERIOD ROBUSTNESS — does the CNN edge survive where coverage is HIGH?")
    print("=" * 74)
    cnn = df[df.Strategy == 'CNN_Prob'].pivot(index='Config', columns='Period',
                                              values='AnnSharpe')
    display(cnn.round(3))
    print("\nHOW TO READ THIS")
    print("  edge in BOTH halves  -> survivorship is not driving it. Say so, with this table.")
    print("  edge ONLY in 'early' -> it is likely an artifact of the survivor-selected universe.")
    print("  edge ONLY in 'late'  -> not survivorship; more likely a regime/period effect.")
    return df

sub = subperiod_report()
