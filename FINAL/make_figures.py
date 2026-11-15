"""Generate all figures for the crypto candlestick-CNN paper, from the real result files
and cached predictions. Writes vector PDFs into ./figures/. Run with the env that has
matplotlib (Anaconda). Read-only w.r.t. all study artifacts."""
import os, sys, json, io, contextlib
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

FIG = './figures'; os.makedirs(FIG, exist_ok=True)

# ---------------- Okabe-Ito colorblind-safe palette (lightness-distinct -> grayscale-safe) --------
C_CNN   = '#0072B2'   # blue  — the CNN, held constant across every figure (color follows entity)
C_ALT   = '#E69F00'   # orange — the second CNN config when shown beside the first
C_MUTE  = '#767676'   # gray  — benchmarks / reference
C_POS   = '#009E73'   # green — positive / significant
C_NEG   = '#D55E00'   # vermillion — negative
C_LIGHT = '#9ecae1'   # light blue — secondary bar

mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#DDDDDD', 'grid.linewidth': 0.5,
    'axes.axisbelow': True, 'figure.dpi': 140, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
def save(fig, name):
    p = os.path.join(FIG, name); fig.savefig(p); plt.close(fig); print(f"  wrote {p}")

# =============================================================================
# Load the study's machinery + cached predictions (reuse crypto_benchmarks.ipynb)
# =============================================================================
NS = {}
nb = json.load(open('crypto_benchmarks.ipynb'))
cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
for src in cells[:3]:                       # config + machinery + kalman signal
    with contextlib.redirect_stdout(io.StringIO()):
        exec(src, NS)
df_market   = NS['df_market']; kf_sig = NS['kf_sig']
rebuild     = NS['rebuild_eval_base']; _ls = NS['_ls_series']; _sh = NS['_sharpe']
rebal_grid  = NS['rebalance_grid']; ANN = NS['ANNUALIZER']
CFG = [(5,5),(5,20),(20,5),(20,20)]
print("machinery + kalman loaded; rebuilding eval_base per config from cached preds...")
EB = {}
for n,h in CFG:
    eb = rebuild(n,h)
    if eb is not None:
        EB[(n,h)] = eb.merge(kf_sig, on=['Symbol','Date'], how='left')
print(f"  rebuilt: {['I%d/R%d'%(n,h) for (n,h) in EB]}")

grid   = pd.read_csv('crypto_corrected_grid.csv').set_index('Config')
surv   = pd.read_csv('crypto_survivorship_decomp.csv').set_index('Config')
bench  = pd.read_csv('crypto_benchmark_results.csv')
def cfgname(n,h): return f"I{n}/R{h}"

# =============================================================================
# FIG 1 — Data honesty: dead-coin retention + cross-rate concentration
# =============================================================================
uni = df_market[df_market.InUniverse].copy()
last_seen = df_market.groupby('Symbol')['Date'].max()
alive_cut = df_market['Date'].max() - pd.Timedelta(days=30)
alive = set(last_seen[last_seen >= alive_cut].index)
uni['dead'] = ~uni['Symbol'].isin(alive)
uni['yr'] = uni['Date'].dt.year
by_yr = uni.groupby('yr').agg(dead=('dead', lambda s: s.groupby(uni.loc[s.index,'Date']).mean().mean()),
                              btcx=('Source', lambda s: (s=='BTCX').mean()))
yrs = by_yr.index.values
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
# Panel A: share of the traded universe that is (as of today) delisted
ax[0].bar(yrs, by_yr['dead']*100, color=C_CNN, width=0.7)
ax[0].set_title('(a) Universe now-delisted share'); ax[0].set_ylabel('% of universe names')
ax[0].set_ylim(0, 100); ax[0].xaxis.set_major_locator(MultipleLocator(2))
# Panel B: cross-rate fraction (approximation concentration)
ax[1].bar(yrs, by_yr['btcx']*100, color=C_MUTE, width=0.7)
ax[1].axvspan(2020.5, 2026.5, color=C_POS, alpha=0.08)
ax[1].set_title('(b) BTC cross-rated rows'); ax[1].set_ylabel('% of universe rows')
ax[1].set_ylim(0, 100); ax[1].xaxis.set_major_locator(MultipleLocator(2))
ax[1].text(2023.4, 55, 'reported\nOOS window', color=C_POS, fontsize=7.5, ha='center')
save(fig, 'fig_data.pdf')

# =============================================================================
# FIG 2 — Corrected grid: Sharpe by config, colored by BH significance
# =============================================================================
g = grid.loc[[cfgname(n,h) for n,h in CFG]]
fig, ax = plt.subplots(figsize=(5.2, 2.4))
cols = [C_CNN if r else C_MUTE for r in g['BH_reject']]
y = np.arange(len(g))[::-1]
ax.barh(y, g['Sharpe'], color=cols, height=0.62)
for yi, (cfg, row) in zip(y, g.iterrows()):
    star = r'$\ast$' if row['BH_reject'] else ''
    ax.text(row['Sharpe']+0.03, yi, f"p={row['bonferroni_p']:.3f} {star}", va='center', fontsize=7.5)
ax.set_yticks(y); ax.set_yticklabels(g.index)
ax.set_xlabel('Annualized Sharpe (out-of-sample, before costs)'); ax.set_xlim(0, 2.15)
ax.axvline(0, color='#333', lw=0.6)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_CNN, label='survives correction (Bonferroni)'),
                   Patch(color=C_MUTE, label='not significant')],
          loc='center right', bbox_to_anchor=(1.0, 0.60), frameon=False, fontsize=7.5)
save(fig, 'fig_grid.pdf')

# =============================================================================
# FIG 3 — Survivorship decomposition: true point-in-time vs survivors-only
# =============================================================================
s = surv.loc[[cfgname(n,h) for n,h in CFG]]
fig, ax = plt.subplots(figsize=(5.4, 2.6))
x = np.arange(len(s)); w = 0.38
ax.bar(x-w/2, s['Sharpe_true_PIT'],        w, color=C_CNN,  label='true point-in-time')
ax.bar(x+w/2, s['Sharpe_survivors_only'],  w, color=C_ALT,  label='survivors-only (bias re-introduced)')
for xi, (cfg, row) in zip(x, s.iterrows()):
    ax.annotate(f"bias {row['bias']:+.2f}", (xi, max(row['Sharpe_true_PIT'], row['Sharpe_survivors_only'])+0.03),
                ha='center', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(s.index); ax.set_ylabel('Annualized Sharpe')
ax.axhline(0, color='#333', lw=0.6); ax.legend(frameon=False, loc='upper right')
save(fig, 'fig_survivorship.pdf')

# =============================================================================
# FIG 4 — Benchmark comparison (the two winning configs), CNN highlighted
# =============================================================================
order = ['CNN_Prob','KALMAN','BTC_BH','TREND','EW_TOP100','STR','MOM','WSTR']
label = {'CNN_Prob':'CNN','KALMAN':'Kalman trend/vol','BTC_BH':'buy & hold BTC','TREND':'HZZ trend',
         'EW_TOP100':'EW top-100','STR':'reversal (1m)','MOM':'momentum','WSTR':'reversal (1w)'}
fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.0), sharex=True)
for k,(cfg) in enumerate(['I5/R5','I20/R20']):
    sub = bench[bench.Config==cfg].set_index('Strategy')
    sub = sub.loc[[o for o in order if o in sub.index]]
    yy = np.arange(len(sub))[::-1]
    for yi,(st,row) in zip(yy, sub.iterrows()):
        col = C_CNN if st=='CNN_Prob' else C_MUTE
        hatch = '////' if row['Type']=='directional' else None
        ax[k].barh(yi, row['Sharpe_0bps'], color=col, height=0.66, hatch=hatch, edgecolor='white', lw=0.4)
    ax[k].set_yticks(yy); ax[k].set_yticklabels([label[s] for s in sub.index] if k==0 else [])
    ax[k].set_title(cfg); ax[k].axvline(0, color='#333', lw=0.6)
    ax[k].set_xlabel('Sharpe (gross)')
ax[0].legend(handles=[Patch(color=C_CNN, label='CNN'), Patch(color=C_MUTE, label='benchmark'),
                      Patch(facecolor='white', edgecolor=C_MUTE, hatch='////', label='directional')],
             loc='lower right', frameon=False, fontsize=7)
save(fig, 'fig_benchmarks.pdf')

# =============================================================================
# FIG 5 — Cost sensitivity: the winners vs their closest competitor
# =============================================================================
costs = [0,10,25,50]
def row_sh(cfg, strat):
    r = bench[(bench.Config==cfg)&(bench.Strategy==strat)].iloc[0]
    return [r[f'Sharpe_{c}bps'] for c in costs]
fig, ax = plt.subplots(figsize=(5.2, 2.8))
ax.plot(costs, row_sh('I5/R5','CNN_Prob'),  '-o', color=C_CNN, lw=2, ms=5, label='CNN I5/R5')
ax.plot(costs, row_sh('I20/R20','CNN_Prob'),'-s', color=C_ALT, lw=2, ms=5, label='CNN I20/R20')
ax.plot(costs, row_sh('I5/R5','KALMAN'),    '--^', color=C_MUTE, lw=1.5, ms=4, label='Kalman (I5/R5 best benchmark)')
ax.plot(costs, row_sh('I20/R20','STR'),     ':v', color=C_MUTE, lw=1.5, ms=4, label='reversal (I20/R20 best benchmark)')
ax.axhline(0, color='#333', lw=0.6)
ax.set_xlabel('transaction cost (bps per side)'); ax.set_ylabel('Annualized Sharpe')
ax.set_xticks(costs); ax.legend(frameon=False, fontsize=7.5, loc='upper right')
save(fig, 'fig_costs.pdf')

# =============================================================================
# FIG 6 — Cumulative net-of-cost equity of the D10-D1 book vs buy&hold BTC
# =============================================================================
fig, ax = plt.subplots(figsize=(6.2, 2.9))
def equity(series):
    return (1+series.dropna()).cumprod()
for (n,h),col,lab,style in [((5,5),C_CNN,'CNN I5/R5 (net 25bps)','-'),
                            ((20,20),C_ALT,'CNN I20/R20 (net 25bps)','-')]:
    eb = EB[(n,h)]; r = _ls(eb,'CNN_Prob',h,cost_bps=25)
    e = equity(r); ax.plot(e.index, e.values, style, color=col, lw=1.8, label=lab)
# buy&hold BTC (directional, gross) on the H=20 grid for reference
eb = EB[(20,20)]; reb = rebal_grid(eb['Date'].values,20)
btc = eb[eb.Symbol=='BTC'].set_index('Date')['Fwd_Ret']; btc=btc[~btc.index.duplicated()].reindex(reb).dropna()
e = equity(btc); ax.plot(e.index, e.values, '-', color=C_MUTE, lw=1.4, label='buy & hold BTC (gross)')
ax.axhline(1, color='#333', lw=0.6)
ax.set_ylabel('growth of \$1 (log scale)'); ax.set_yscale('log')
ax.legend(frameon=False, fontsize=7.5, loc='upper left')
save(fig, 'fig_equity.pdf')

# =============================================================================
# FIG 7 — Decile monotonicity for the strongest config (I5/R5)
# =============================================================================
eb = EB[(5,5)]; H=5
d = eb.dropna(subset=['CNN_Prob','Fwd_Ret'])
d = d[d['Date'].isin(rebal_grid(eb['Date'].values,H))]
means = np.zeros(10); c=0
for _,gp in d.groupby('Date'):
    if len(gp)<20: continue
    dec = pd.qcut(gp['CNN_Prob'].rank(method='first'),10,labels=False)
    for k in range(10): means[k]+=gp.loc[dec==k,'Fwd_Ret'].mean()
    c+=1
means/=c
rho = np.corrcoef(np.arange(10), means)[0,1]
fig, ax = plt.subplots(figsize=(5.0, 2.6))
colors = [C_NEG]+[C_MUTE]*8+[C_POS]
ax.bar(np.arange(10), means*1e4, color=colors, width=0.7)
ax.axhline(0, color='#333', lw=0.6)
ax.set_xlabel('CNN probability decile (1 = lowest, 10 = highest)'); ax.set_ylabel('mean 5-day return (bps)')
ax.set_xticks(range(10)); ax.set_xticklabels(range(1,11))
ax.text(0.03,0.92, f'rank corr = {rho:+.2f}', transform=ax.transAxes, fontsize=8)
save(fig, 'fig_deciles.pdf')

# =============================================================================
# FIG 8 — CNN vs benchmark return-stream correlation (heatmap)
# =============================================================================
benchmarks = ['MOM','STR','WSTR','TREND','KALMAN','BTC_BH']
blab = {'MOM':'momentum','STR':'reversal 1m','WSTR':'reversal 1w','TREND':'HZZ trend',
        'KALMAN':'Kalman','BTC_BH':'buy&hold BTC'}
M = np.full((2,len(benchmarks)), np.nan)
for i,(n,h) in enumerate([(5,5),(20,20)]):
    eb=EB[(n,h)]; cnn=_ls(eb,'CNN_Prob',h); reb=rebal_grid(eb['Date'].values,h)
    for j,b in enumerate(benchmarks):
        if b=='BTC_BH':
            s=eb[eb.Symbol=='BTC'].set_index('Date')['Fwd_Ret']; s=s[~s.index.duplicated()].reindex(reb)
        else:
            s=_ls(eb,b,h)
        jn=pd.concat([cnn,s],axis=1,join='inner').dropna()
        M[i,j]=jn.iloc[:,0].corr(jn.iloc[:,1]) if len(jn)>5 else np.nan
fig, ax = plt.subplots(figsize=(5.6, 1.9))
im = ax.imshow(M, cmap='RdBu', vmin=-0.6, vmax=0.6, aspect='auto')
ax.set_xticks(range(len(benchmarks))); ax.set_xticklabels([blab[b] for b in benchmarks], rotation=30, ha='right')
ax.set_yticks([0,1]); ax.set_yticklabels(['CNN I5/R5','CNN I20/R20'])
for i in range(2):
    for j in range(len(benchmarks)):
        ax.text(j,i,f"{M[i,j]:+.2f}",ha='center',va='center',fontsize=7.5,
                color='white' if abs(M[i,j])>0.35 else '#222')
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label='correlation')
save(fig, 'fig_correlation.pdf')

print("\nALL FIGURES DONE ->", FIG)
