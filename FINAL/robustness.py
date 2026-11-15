"""Robustness battery for the crypto candlestick-CNN paper (Tier 1 + Tier 2).
Reuses crypto_benchmarks.ipynb machinery + cached predictions. Read-only w.r.t. study artifacts.
Writes: crypto_robustness.csv, figures/fig_ablation.pdf, figures/fig_netcost.pdf.

Analyses
  A. Linear / tree ablation      -- is the DEEP net needed? (logistic + HistGBM on same pixels)
  B. Corwin-Schultz spread costs -- endogenous per-coin effective spread (Corwin & Schultz 2012, JF)
  C. Deflated Sharpe Ratio       -- Bailey & Lopez de Prado (2014), corrects trials + non-normality
  D. Return moments + drawdown
  E. Delisting-return fix        -- assign terminal return instead of dropping (return-level survivorship)
  F. Nuisance sensitivity        -- decile count, min_names, top-50 vs top-100
  G. Seed / ensemble stability
  H. Tradeable variants          -- long-only, and long-D10 / short-BTC (BTC is shortable; alts are not)
"""
import os, json, io, contextlib, time
import numpy as np, pandas as pd
from math import erf, sqrt, log, exp
from scipy.stats import norm, skew, kurtosis
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

C_CNN, C_ALT, C_MUTE, C_POS, C_NEG = '#0072B2','#E69F00','#767676','#009E73','#D55E00'
mpl.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,'grid.color':'#DDDDDD','grid.linewidth':0.5,
    'axes.axisbelow':True,'figure.dpi':140,'savefig.bbox':'tight'})
os.makedirs('figures', exist_ok=True)

# ---- load the benchmark machinery (config + machinery + kalman) ----
NS = {}
nb = json.load(open('crypto_benchmarks.ipynb'))
cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
for src in cells[:3]:
    with contextlib.redirect_stdout(io.StringIO()):
        exec(src, NS)
df_market = NS['df_market']; kf_sig = NS['kf_sig']
load_store = NS['load_store']; label_for_horizon = NS['label_for_horizon']
test_positions = NS['test_positions']; rebalance_grid = NS['rebalance_grid']
add_trend = NS['add_trend_factor']; _ls = NS['_ls_series']; _sh = NS['_sharpe']
pred_path = NS['pred_path']; ENSEMBLE_SIZE = NS['ENSEMBLE_SIZE']; ANN = NS['ANNUALIZER']
TRAIN_END = '2020-12-31'
WIN = [(5,5),(20,20)]; ALL = [(5,5),(5,20),(20,5),(20,20)]
out = []
print("machinery loaded.\n")

def ls_series_cnn(n,h,cost=0.0,sig='CNN_Prob'):
    eb = EB[(n,h)]; return _ls(eb, sig, h, cost_bps=cost)

# rebuild eval_base per config (CNN prob + kalman + UnivRank merged)
def rebuild(n,h):
    store = load_store(n, df_market)
    rows, base = label_for_horizon(store, df_market, h)
    te = test_positions(base); eb = base.iloc[te].reset_index(drop=True)
    ps = [np.load(pred_path(n,h,1000+i)) for i in range(ENSEMBLE_SIZE)
          if os.path.exists(pred_path(n,h,1000+i))]
    eb['CNN_Prob'] = np.mean(ps, axis=0)
    eb = add_trend(eb, h)
    ur = df_market[['Symbol','Date','UnivRank']]
    eb = eb.merge(ur, on=['Symbol','Date'], how='left')
    return eb, store, rows, base, te
EB = {}; STORE={}; ROWS={}; BASE={}; TE={}
for n,h in ALL:
    eb,store,rows,base,te = rebuild(n,h)
    EB[(n,h)]=eb.merge(kf_sig,on=['Symbol','Date'],how='left')
    STORE[(n,h)]=store; ROWS[(n,h)]=rows; BASE[(n,h)]=base; TE[(n,h)]=te

# =====================================================================
# A. ABLATION -- is the DEEP net needed?
# =====================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
TRAIN_CAP = 120_000
print("="*70); print("A. ABLATION: CNN vs logistic vs gradient-boosted trees (same pixels)"); print("="*70)
abl = []
for n,h in WIN:
    store=STORE[(n,h)]; rows=ROWS[(n,h)]; base=BASE[(n,h)]; te=TE[(n,h)]
    y = (base['Fwd_Ret'].values > 0).astype(int)
    tv = np.where(base['Date'].values <= np.datetime64(TRAIN_END))[0]
    if len(tv) > TRAIN_CAP:
        tv = np.random.default_rng(0).choice(tv, TRAIN_CAP, replace=False)
    mm = np.memmap(store['dat'], dtype=np.uint8, mode='r', shape=store['shape'])
    Xtv = mm[rows[tv]].reshape(len(tv),-1).astype(np.float32)/255.0
    Xte = mm[rows[te]].reshape(len(te),-1).astype(np.float32)/255.0
    ytv = y[tv]
    eb = EB[(n,h)].copy()
    t0=time.time()
    lr = LogisticRegression(max_iter=200, C=1.0).fit(Xtv, ytv)
    eb['LIN'] = lr.predict_proba(Xte)[:,1]
    gb = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
                                        max_depth=4, random_state=0).fit(Xtv, ytv)
    eb['GBM'] = gb.predict_proba(Xte)[:,1]
    cnn_sh = _sh(_ls(eb,'CNN_Prob',h), h)
    lin_sh = _sh(_ls(eb,'LIN',h), h)
    gbm_sh = _sh(_ls(eb,'GBM',h), h)
    print(f"  I{n}/R{h}:  CNN {cnn_sh:+.2f} | logistic {lin_sh:+.2f} | GBM {gbm_sh:+.2f}   "
          f"(train {len(tv):,}, {time.time()-t0:.0f}s)")
    abl.append({'Config':f'I{n}/R{h}','CNN':cnn_sh,'Logistic':lin_sh,'GBM':gbm_sh})
    del mm, Xtv, Xte
abl = pd.DataFrame(abl)

# =====================================================================
# B. CORWIN-SCHULTZ (2012, JF) endogenous spread -> spread-aware net Sharpe
# =====================================================================
print("\n"+"="*70); print("B. Corwin-Schultz effective spread -> data-driven transaction costs"); print("="*70)
def corwin_schultz(g):
    # per-symbol proportional effective spread from consecutive daily high/low
    H = g['High'].values.astype(float); L = g['Low'].values.astype(float)
    n = len(g); S = np.full(n, np.nan)
    if n < 2: return pd.Series(S, index=g.index)
    hl = np.log(H/np.maximum(L,1e-12))**2
    for t in range(1,n):
        Ht=max(H[t],H[t-1]); Lt=min(L[t],L[t-1])
        beta = hl[t]+hl[t-1]; gamma = np.log(Ht/max(Lt,1e-12))**2
        den = 3-2*np.sqrt(2)
        alpha = (np.sqrt(2*beta)-np.sqrt(beta))/den - np.sqrt(gamma/den)
        s = 2*(np.exp(alpha)-1)/(1+np.exp(alpha))
        S[t] = max(s, 0.0)               # Corwin-Schultz: set negative estimates to 0
    return pd.Series(S, index=g.index)
cs = df_market.sort_values(['Symbol','Date']).groupby('Symbol', group_keys=False).apply(corwin_schultz)
df_market['CS_SPREAD'] = cs
csmed = df_market.loc[df_market.InUniverse,'CS_SPREAD'].median()
print(f"  median Corwin-Schultz effective spread across the universe: {csmed*1e4:.0f} bps "
      f"(round-trip); by-decile spreads applied per coin.")
sp_lookup = df_market.set_index(['Symbol','Date'])['CS_SPREAD']

def ls_series_cscost(n,h,sig='CNN_Prob',min_names=20):
    eb = EB[(n,h)].copy()
    eb['SP'] = sp_lookup.reindex(pd.MultiIndex.from_arrays([eb['Symbol'],eb['Date']])).values
    d = eb.dropna(subset=[sig,'Fwd_Ret'])
    reb = rebalance_grid(eb['Date'].values,h); d=d[d['Date'].isin(reb)]; res={}
    for dt,g in d.groupby('Date'):
        if len(g)<min_names: continue
        dec = pd.qcut(g[sig].rank(method='first'),10,labels=False)
        lo=g[dec==9]; sh=g[dec==0]
        gross = lo['Fwd_Ret'].mean() - sh['Fwd_Ret'].mean()
        # round-trip spread cost = mean effective spread over the coins actually traded (both legs)
        cost = pd.concat([lo['SP'],sh['SP']]).mean()
        res[dt] = gross - (cost if np.isfinite(cost) else 0.0)
    return pd.Series(res).sort_index()

for n,h in ALL:
    net = _sh(ls_series_cscost(n,h), h); gross=_sh(_ls(EB[(n,h)],'CNN_Prob',h),h)
    print(f"  I{n}/R{h}: gross {gross:+.2f} -> Corwin-Schultz net {net:+.2f}")
    out.append({'Config':f'I{n}/R{h}','metric':'CNN_gross_Sharpe','value':round(gross,3)})
    out.append({'Config':f'I{n}/R{h}','metric':'CNN_CorwinSchultz_net_Sharpe','value':round(net,3)})

# =====================================================================
# C. DEFLATED SHARPE RATIO (Bailey & Lopez de Prado 2014)
# =====================================================================
print("\n"+"="*70); print("C. Deflated Sharpe Ratio (trials + skew + kurtosis + length)"); print("="*70)
GAMMA = 0.5772156649
def psr(sr, sr_star, r):                 # probabilistic Sharpe (per-period sr)
    n=len(r); sk=skew(r); ku=kurtosis(r, fisher=False)
    num=(sr-sr_star)*np.sqrt(n-1)
    den=np.sqrt(1 - sk*sr + (ku-1)/4.0*sr**2)
    return float(norm.cdf(num/den))
def deflated_sharpe(series_by_cfg, cfg, N_trials):
    # per-period Sharpe of each trial -> variance across trials; expected max under N
    perp = {c: s.dropna().mean()/s.dropna().std(ddof=1) for c,s in series_by_cfg.items()}
    v = np.var(list(perp.values()), ddof=1)
    e_max = np.sqrt(v)*((1-GAMMA)*norm.ppf(1-1.0/N_trials)+GAMMA*norm.ppf(1-1.0/(N_trials*np.e)))
    r = series_by_cfg[cfg].dropna().values
    return psr(perp[cfg], e_max, r), perp[cfg], e_max
series_all = {f'I{n}/R{h}': _ls(EB[(n,h)],'CNN_Prob',h) for n,h in ALL}
for n,h in WIN:
    cfg=f'I{n}/R{h}'
    dsr,srp,emax = deflated_sharpe(series_all, cfg, N_trials=len(ALL))
    print(f"  {cfg}: per-period SR {srp:+.3f} | expected-max under {len(ALL)} trials {emax:+.3f} | "
          f"Deflated Sharpe (prob. real) = {dsr:.3f}")
    out.append({'Config':cfg,'metric':'DeflatedSharpe_prob','value':round(dsr,3)})

# =====================================================================
# D. RETURN MOMENTS + DRAWDOWN
# =====================================================================
print("\n"+"="*70); print("D. Return moments and drawdown (net 25bps flat, for comparability)"); print("="*70)
for n,h in WIN:
    r=_ls(EB[(n,h)],'CNN_Prob',h,cost_bps=25).dropna()
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min()
    down=r[r<0].std(ddof=1); sortino=r.mean()/down*np.sqrt(ANN/h) if down>0 else np.nan
    print(f"  I{n}/R{h}: skew {skew(r):+.2f} | ex-kurt {kurtosis(r):+.2f} | maxDD {dd:+.1%} | "
          f"Sortino {sortino:+.2f}")
    out += [{'Config':f'I{n}/R{h}','metric':m,'value':round(v,3)} for m,v in
            [('skew',skew(r)),('excess_kurtosis',kurtosis(r)),('max_drawdown',dd),('Sortino',sortino)]]

# =====================================================================
# E. DELISTING-RETURN FIX (assign terminal return instead of dropping)
# =====================================================================
print("\n"+"="*70); print("E. Delisting-return treatment (return-level survivorship)"); print("="*70)
last_close = df_market.sort_values('Date').groupby('Symbol').last()['Close']
last_date  = df_market.groupby('Symbol')['Date'].max()
close_lu = df_market.set_index(['Symbol','Date'])['Close']
def ls_with_delist(n,h,min_names=20):
    base=BASE[(n,h)]; te=TE[(n,h)]; eb=EB[(n,h)].copy()
    # recompute Fwd_Ret: if t+h beyond the coin's last bar, use terminal (last close) return
    fr=[]
    for sym,dt in zip(eb['Symbol'],eb['Date']):
        t1=dt+pd.Timedelta(days=h)
        try: p0=close_lu.loc[(sym,dt)]
        except KeyError: fr.append(np.nan); continue
        key=(sym,t1)
        if key in close_lu.index: fr.append(close_lu.loc[key]/p0-1)
        elif t1 > last_date[sym]:                          # delisted within the window
            fr.append(last_close[sym]/p0-1)                #   -> realize the terminal return
        else: fr.append(np.nan)
    eb['Fwd_Ret']=fr
    return _sh(_ls(eb,'CNN_Prob',h,min_names=min_names), h)
for n,h in WIN:
    base_sh=_sh(_ls(EB[(n,h)],'CNN_Prob',h),h); fix=ls_with_delist(n,h)
    print(f"  I{n}/R{h}: drop-delisting {base_sh:+.2f} -> terminal-return {fix:+.2f}")
    out.append({'Config':f'I{n}/R{h}','metric':'CNN_delisting_terminal_Sharpe','value':round(fix,3)})

# =====================================================================
# F. NUISANCE SENSITIVITY (decile count, min_names, top-50)
# =====================================================================
print("\n"+"="*70); print("F. Nuisance-parameter sensitivity (winners)"); print("="*70)
def ls_qcount(eb,sig,h,q,min_names,univ=None):
    d=eb.dropna(subset=[sig,'Fwd_Ret'])
    if univ is not None: d=d[d['UnivRank']<=univ]
    reb=rebalance_grid(eb['Date'].values,h); d=d[d['Date'].isin(reb)]; res={}
    for dt,g in d.groupby('Date'):
        if len(g)<min_names: continue
        bins=pd.qcut(g[sig].rank(method='first'),q,labels=False)
        res[dt]=g.loc[bins==q-1,'Fwd_Ret'].mean()-g.loc[bins==0,'Fwd_Ret'].mean()
    return _sh(pd.Series(res),h)
for n,h in WIN:
    eb=EB[(n,h)]; cfg=f'I{n}/R{h}'
    row={'quintiles':ls_qcount(eb,'CNN_Prob',h,5,20),
         'deciles':ls_qcount(eb,'CNN_Prob',h,10,20),
         'vigintiles':ls_qcount(eb,'CNN_Prob',h,20,20),
         'min_names=30':ls_qcount(eb,'CNN_Prob',h,10,30),
         'top50':ls_qcount(eb,'CNN_Prob',h,10,15,univ=50)}
    print(f"  {cfg}: "+" | ".join(f"{k} {v:+.2f}" for k,v in row.items()))
    out+=[{'Config':cfg,'metric':f'nuisance_{k}','value':round(v,3)} for k,v in row.items()]

# =====================================================================
# G. SEED / ENSEMBLE STABILITY
# =====================================================================
print("\n"+"="*70); print("G. Per-member seed stability (winners)"); print("="*70)
for n,h in WIN:
    eb=EB[(n,h)].copy(); shs=[]
    for i in range(ENSEMBLE_SIZE):
        p=pred_path(n,h,1000+i)
        if not os.path.exists(p): continue
        eb['m']=np.load(p); shs.append(_sh(_ls(eb,'m',h),h))
    shs=np.array(shs)
    print(f"  I{n}/R{h}: members {np.round(shs,2)} | mean {shs.mean():+.2f} sd {shs.std(ddof=1):.2f} "
          f"| ensemble {_sh(_ls(EB[(n,h)],'CNN_Prob',h),h):+.2f}")
    out.append({'Config':f'I{n}/R{h}','metric':'member_Sharpe_sd','value':round(shs.std(ddof=1),3)})

# =====================================================================
# H. TRADEABLE VARIANTS (long-only; long D10 / short BTC)
# =====================================================================
print("\n"+"="*70); print("H. Tradeable variants -- avoid shorting illiquid alts"); print("="*70)
def variant_series(n,h,kind,min_names=20):
    eb=EB[(n,h)]; d=eb.dropna(subset=['CNN_Prob','Fwd_Ret'])
    reb=rebalance_grid(eb['Date'].values,h); d=d[d['Date'].isin(reb)]
    btc=eb[eb.Symbol=='BTC'].set_index('Date')['Fwd_Ret']; btc=btc[~btc.index.duplicated()]
    res={}
    for dt,g in d.groupby('Date'):
        if len(g)<min_names: continue
        dec=pd.qcut(g['CNN_Prob'].rank(method='first'),10,labels=False)
        long=g.loc[dec==9,'Fwd_Ret'].mean()
        if kind=='long_only': res[dt]=long
        elif kind=='long_vs_btc':
            if dt in btc.index: res[dt]=long-btc.loc[dt]
    return _sh(pd.Series(res),h)
for n,h in WIN:
    lo=variant_series(n,h,'long_only'); lb=variant_series(n,h,'long_vs_btc')
    print(f"  I{n}/R{h}: long-only (directional) {lo:+.2f} | long-D10 / short-BTC (tradeable neutral) {lb:+.2f}")
    out+=[{'Config':f'I{n}/R{h}','metric':'long_only_Sharpe','value':round(lo,3)},
          {'Config':f'I{n}/R{h}','metric':'long_vs_BTC_Sharpe','value':round(lb,3)}]

pd.DataFrame(out).to_csv('crypto_robustness.csv', index=False)
print("\nsaved -> crypto_robustness.csv")

# =====================================================================
# FIGURES
# =====================================================================
# Ablation
fig,ax=plt.subplots(figsize=(5.0,2.6)); x=np.arange(len(abl)); w=0.26
ax.bar(x-w, abl['CNN'], w, color=C_CNN, label='CNN')
ax.bar(x,   abl['GBM'], w, color=C_ALT, label='gradient-boosted trees')
ax.bar(x+w, abl['Logistic'], w, color=C_MUTE, label='logistic (linear)')
ax.set_xticks(x); ax.set_xticklabels(abl['Config']); ax.set_ylabel('Annualized Sharpe (gross)')
ax.axhline(0,color='#333',lw=0.6); ax.legend(frameon=False,fontsize=7.5)
fig.savefig('figures/fig_ablation.pdf'); plt.close(fig); print("  wrote figures/fig_ablation.pdf")

# Net-of-cost: flat 25 vs Corwin-Schultz, winners
fig,ax=plt.subplots(figsize=(5.0,2.6)); x=np.arange(len(WIN)); w=0.26
g=[_sh(_ls(EB[(n,h)],'CNN_Prob',h),h) for n,h in WIN]
f25=[_sh(_ls(EB[(n,h)],'CNN_Prob',h,cost_bps=25),h) for n,h in WIN]
cs_=[_sh(ls_series_cscost(n,h),h) for n,h in WIN]
ax.bar(x-w,g,w,color=C_MUTE,label='gross'); ax.bar(x,f25,w,color=C_ALT,label='flat 25 bps')
ax.bar(x+w,cs_,w,color=C_CNN,label='Corwin-Schultz (endogenous)')
ax.set_xticks(x); ax.set_xticklabels([f'I{n}/R{h}' for n,h in WIN]); ax.set_ylabel('Annualized Sharpe')
ax.axhline(0,color='#333',lw=0.6); ax.legend(frameon=False,fontsize=7.5)
fig.savefig('figures/fig_netcost.pdf'); plt.close(fig); print("  wrote figures/fig_netcost.pdf")
print("\nDONE.")
