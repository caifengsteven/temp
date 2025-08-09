import numpy as np
import pandas as pd


def simulate_backtest(returns: pd.DataFrame,
                      weights: pd.DataFrame,
                      tx_cost_bps: float = 2.0,
                      attn_cost: pd.Series = None):
    # Align
    rets = returns.loc[weights.index]
    if attn_cost is not None:
        attn_cost = attn_cost.reindex(weights.index).fillna(0.0)
    # Compute portfolio return with transaction costs and optional attention costs
    w_prev = None
    port_rets = []
    for i, (dt, w) in enumerate(weights.iterrows()):
        r = rets.loc[dt].values
        gross = np.dot(w.values, r)
        if w_prev is None:
            tc = 0.0
        else:
            turnover = np.abs(w.values - w_prev).sum()
            tc = (tx_cost_bps * 1e-4) * turnover
        ac = float(attn_cost.loc[dt]) if attn_cost is not None else 0.0
        net = gross - tc - ac
        port_rets.append(net)
        w_prev = w.values
    port_rets = pd.Series(port_rets, index=weights.index, name='port_ret')
    eq_curve = (1 + port_rets).cumprod()
    ann = 252 / np.diff(weights.index.to_numpy('datetime64[D]')).mean().astype(float)
    ann_ret = eq_curve.iloc[-1] ** (ann / len(eq_curve)) - 1
    ann_vol = port_rets.std() * np.sqrt(ann)
    sharpe = ann_ret / (ann_vol + 1e-12)
    dd = (eq_curve / eq_curve.cummax() - 1).min()
    summary = dict(ann_ret=float(ann_ret), ann_vol=float(ann_vol), sharpe=float(sharpe), max_dd=float(dd))
    return port_rets, eq_curve, summary

