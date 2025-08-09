from pathlib import Path
import pandas as pd
import numpy as np


def load_price_csvs(data_dir: str, tickers):
    data_dir = Path(data_dir)
    frames = []
    for t in tickers:
        p = data_dir / f"{t}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing CSV for {t}: {p}")
        df = pd.read_csv(p)
        # Expect Date, Adj Close
        cols = {c.lower(): c for c in df.columns}
        date_col = cols.get('date') or cols.get('timestamp') or list(df.columns)[0]
        px_col = cols.get('adj close') or cols.get('adj_close') or cols.get('close') or list(df.columns)[1]
        df = df.rename(columns={date_col: 'Date', px_col: t})[['Date', t]]
        df['Date'] = pd.to_datetime(df['Date'])
        frames.append(df.set_index('Date'))
    px = pd.concat(frames, axis=1).sort_index().ffill().dropna()
    return px


def compute_returns(px: pd.DataFrame, freq='D'):
    # Use log returns for stability
    rets = np.log(px / px.shift(1)).dropna()
    if freq and freq != 'D':
        # Resample and sum log returns at lower frequency (e.g., W-FRI)
        rets = rets.resample(freq).sum().dropna()
    return rets

