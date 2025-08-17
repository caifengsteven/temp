import os
import pandas as pd
import numpy as np
from src.data_loader import load_config, load_all_series, align_monthly
from src.features import engineer_features
from src.labels import make_labels
from src.models_minimal import MinimalEnsemble
from src.evaluate import evaluate_probs
from src.plotting import plot_probs


def main():
    os.makedirs("outputs", exist_ok=True)
    cfg = load_config()
    raw = load_all_series(cfg)
    df = align_monthly(raw, start_year=cfg.get("start_year", 1990))

    # Features
    X = engineer_features(df)
    # Labels
    Y = make_labels(df["nber_recession"], cfg["horizons"], cfg.get("label_mode", "any"))

    # Train minimal ensemble
    model = MinimalEnsemble(cfg["horizons"]).fit(X, Y, start_train=f"{cfg.get('start_year',1990)+5}-01-01")
    model.update_weights_by_recent_performance(X, Y, window=48)

    # Predict in-sample probabilities
    P = model.predict_proba(X)

    # Evaluate
    metrics = evaluate_probs(P, Y)
    metrics.to_csv("outputs/minimal_metrics.csv")

    # Plot
    plot_probs(P, df["nber_recession"], title="Minimal Ensemble Recession Probabilities", outpath="outputs/minimal_probs.png")
    P.to_csv("outputs/minimal_probs.csv")

    print("Saved outputs/minimal_metrics.csv, outputs/minimal_probs.csv, outputs/minimal_probs.png")


if __name__ == "__main__":
    main()

