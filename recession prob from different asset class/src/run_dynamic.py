import os
import pandas as pd
from src.data_loader import load_config, load_all_series, align_monthly
from src.features import engineer_features
from src.labels import make_labels
from src.models_dynamic import DynamicEnsemble
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

    # Fit dynamic online model
    dyn = DynamicEnsemble(cfg["horizons"], calibr_window=60)
    dyn.fit_online(X, Y, warmup=120)
    P = dyn.predict_proba(X)

    # Evaluate & save
    metrics = evaluate_probs(P, Y)
    metrics.to_csv("outputs/dynamic_metrics.csv")

    plot_probs(P, df["nber_recession"], title="Dynamic Online Recession Probabilities", outpath="outputs/dynamic_probs.png")
    P.to_csv("outputs/dynamic_probs.csv")

    print("Saved outputs/dynamic_metrics.csv, outputs/dynamic_probs.csv, outputs/dynamic_probs.png")


if __name__ == "__main__":
    main()

