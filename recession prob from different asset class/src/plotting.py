from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_probs(P: pd.DataFrame, nber: pd.Series, title: str, outpath: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    P.plot(ax=ax)
    # Shade recessions
    nber = nber.reindex(P.index).fillna(0)
    rec = nber == 1
    if rec.any():
        starts = (rec & ~rec.shift(1).fillna(False))
        ends = (~rec & rec.shift(1).fillna(False))
        start_dates = list(P.index[starts])
        end_dates = list(P.index[ends])
        if len(end_dates) < len(start_dates):
            end_dates.append(P.index[-1])
        for s, e in zip(start_dates, end_dates):
            ax.axvspan(s, e, color='gray', alpha=0.2)
    ax.set_title(title)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

