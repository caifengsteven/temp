"""Butterfly relative-value fair-value model (Model 3).

Implements the JPMorgan US Rates Strategy "level-neutral, curve-neutral"
butterfly framework. For each Treasury butterfly the spread is regressed on
two structural drivers:

    1. The **level of rates** (the belly yield, in percent).
    2. The **wing slope** of the curve (``long_wing - short_wing``, in basis
       points).

The residual after netting out these two drivers reveals whether a given
sector of the curve is *truly* rich or cheap, independent of where the
overall level and slope of the curve are trading. Per JPM: "we regress the
butterfly on the body of the butterfly (5yr yields) and the slope of the
wings (2s/10s curve)".

The module is independently runnable (``python -m src.model_butterfly``) and
importable by downstream notebooks / scripts.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.base_model import FairValueModel
from src.data import get_model3_data


# ---------------------------------------------------------------------------
# Butterfly registry
# ---------------------------------------------------------------------------
# Each entry maps a butterfly name to:
#   - fly_col:    pre-computed butterfly spread column (basis points)
#   - belly:      belly yield column (percent)              -> level factor
#   - short_wing: short-wing yield column (percent)         -> used to build slope
#   - long_wing:  long-wing yield column (percent)          -> used to build slope
BUTTERFLIES: dict[str, dict[str, str]] = {
    "2s/3s/5s": {
        "fly_col": "fly_2s_3s_5s",
        "belly": "yield_3y",
        "short_wing": "yield_2y",
        "long_wing": "yield_5y",
    },
    "2s/5s/10s": {
        "fly_col": "fly_2s_5s_10s",
        "belly": "yield_5y",
        "short_wing": "yield_2y",
        "long_wing": "yield_10y",
    },
    "5s/7s/10s": {
        "fly_col": "fly_5s_7s_10s",
        "belly": "yield_7y",
        "short_wing": "yield_5y",
        "long_wing": "yield_10y",
    },
    "5s/10s/30s": {
        "fly_col": "fly_5s_10s_30s",
        "belly": "yield_10y",
        "short_wing": "yield_5y",
        "long_wing": "yield_30y",
    },
    "2s/10s/30s": {
        "fly_col": "fly_2s_10s_30s",
        "belly": "yield_10y",
        "short_wing": "yield_2y",
        "long_wing": "yield_30y",
    },
}

# Absolute residual (in basis points) below which a sector is deemed FAIR.
FAIR_THRESHOLD_BP: float = 1.0


def _classify_signal(residual_bp: float, threshold: float = FAIR_THRESHOLD_BP) -> str:
    """Classify a residual as ``CHEAP`` / ``RICH`` / ``FAIR``.

    Positive residual -> actual > fair value -> **CHEAP**.
    Negative residual -> actual < fair value -> **RICH**.
    ``|residual| < threshold`` -> **FAIR** (takes precedence over sign).
    """
    if abs(residual_bp) < threshold:
        return "FAIR"
    return "CHEAP" if residual_bp > 0.0 else "RICH"


class ButterflyModel:
    """Runs level-neutral, curve-neutral butterfly regressions across the Treasury curve.

    For each butterfly, regresses the spread on:

      1. The belly yield (level of rates).
      2. The wing slope (curve shape).

    The residual after this regression reveals whether a sector is rich or
    cheap after adjusting for rate level and curve slope.
    """

    def __init__(self) -> None:
        self.models: dict[str, FairValueModel] = {}
        self._data: Optional[pd.DataFrame] = None
        self._registry: dict[str, dict[str, str]] = {}
        # butterfly name -> constructed wing-slope column name
        self._slope_cols: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        data: pd.DataFrame,
        butterflies: Optional[dict[str, dict[str, str]]] = None,
    ) -> "ButterflyModel":
        """Fit a :class:`FairValueModel` for each butterfly in the registry.

        For each butterfly:

          - Construct ``wing_slope = (long_wing - short_wing) * 100``  [bp]
            (yields are stored in percent, so the ``* 100`` converts to bp).
          - Fit ``FairValueModel`` on ``fly_col ~ [belly_yield, wing_slope]``.

        Each fitted model is stored in ``self.models`` keyed by butterfly
        name.

        Parameters
        ----------
        data : pd.DataFrame
            Model-3 dataset (as returned by :func:`src.data.get_model3_data`).
        butterflies : dict, optional
            Custom butterfly registry. Defaults to the module-level
            :data:`BUTTERFLIES`.

        Returns
        -------
        ButterflyModel
            ``self`` (for chaining).
        """
        registry = dict(BUTTERFLIES) if butterflies is None else dict(butterflies)

        # Work on an augmented copy so constructed slope columns persist for
        # downstream plotting / factor decomposition.
        augmented = data.copy()
        self.models.clear()
        self._slope_cols.clear()
        self._registry = registry

        for name, comp in registry.items():
            fly_col = comp["fly_col"]
            belly = comp["belly"]
            short_wing = comp["short_wing"]
            long_wing = comp["long_wing"]

            missing = [
                c
                for c in (fly_col, belly, short_wing, long_wing)
                if c not in augmented.columns
            ]
            if missing:
                raise ValueError(
                    f"Butterfly '{name}' references missing columns: {missing}. "
                    f"Available columns: {list(augmented.columns)}."
                )

            # Construct the wing slope in basis points (yields are in percent).
            slope_col = f"{name.replace('/', '_')}_wing_slope"
            augmented[slope_col] = (
                augmented[long_wing] - augmented[short_wing]
            ) * 100.0
            self._slope_cols[name] = slope_col

            model = FairValueModel(name=f"{name} Fly")
            model.fit(augmented, y_col=fly_col, x_cols=[belly, slope_col])
            self.models[name] = model

        self._data = augmented
        return self

    # ------------------------------------------------------------------
    # Accessors / guards
    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        """Raise an informative error if the model has not been fitted yet."""
        if not self.models or self._data is None:
            raise RuntimeError(
                "ButterflyModel is not fitted yet. Call .fit(data) first."
            )

    def get_model(self, name: str) -> FairValueModel:
        """Return the :class:`FairValueModel` for a specific butterfly.

        Parameters
        ----------
        name : str
            Butterfly name (e.g. ``"2s/5s/10s"``); must be a key of the
            registry used at fit time.

        Raises
        ------
        KeyError
            If ``name`` is not among the fitted butterflies.
        """
        if name not in self.models:
            raise KeyError(
                f"No fitted model for butterfly '{name}'. "
                f"Fitted butterflies: {list(self.models)}."
            )
        return self.models[name]

    # ------------------------------------------------------------------
    # Cross-sectional summaries
    # ------------------------------------------------------------------
    def summary_table(self) -> pd.DataFrame:
        """Cross-sectional summary of ALL butterfly regressions.

        Replicates JPM's Figure 50. Columns::

            ["Butterfly", "Level Beta", "Level t-stat", "Curve Beta",
             "Curve t-stat", "R²", "Current Residual (bp)", "Signal"]

        where:

        - **Level Beta** / **Level t-stat** : coefficient / t-stat on the
          belly yield (level of rates).
        - **Curve Beta** / **Curve t-stat** : coefficient / t-stat on the
          wing slope (curve shape).
        - **Signal** : ``CHEAP`` (residual > 0), ``RICH`` (residual < 0) or
          ``FAIR`` (``|residual| < 1 bp``).

        Sorted by R² descending.
        """
        self._check_fitted()
        rows: list[dict[str, float | str]] = []

        for name, model in self.models.items():
            comp = self._registry[name]
            belly = comp["belly"]
            slope_col = self._slope_cols[name]
            coef = model.coefficient_table()

            level_beta = float(coef.loc[belly, "Coefficient"])
            level_t = float(coef.loc[belly, "t-stat"])
            curve_beta = float(coef.loc[slope_col, "Coefficient"])
            curve_t = float(coef.loc[slope_col, "t-stat"])
            resid_latest = float(model.residual_latest)

            rows.append(
                {
                    "Butterfly": name,
                    "Level Beta": level_beta,
                    "Level t-stat": level_t,
                    "Curve Beta": curve_beta,
                    "Curve t-stat": curve_t,
                    "R²": float(model.rsquared),
                    "Current Residual (bp)": resid_latest,
                    "Signal": _classify_signal(resid_latest),
                }
            )

        columns = [
            "Butterfly",
            "Level Beta",
            "Level t-stat",
            "Curve Beta",
            "Curve t-stat",
            "R²",
            "Current Residual (bp)",
            "Signal",
        ]
        df = pd.DataFrame(rows, columns=columns)
        df = df.sort_values("R²", ascending=False).reset_index(drop=True)
        return df

    def rich_cheap_report(self) -> pd.DataFrame:
        """Quick rich/cheap ranking across all butterflies.

        Columns::

            ["Butterfly", "Actual (bp)", "Fair Value (bp)",
             "Residual (bp)", "Signal", "Z-score"]

        ``Z-score`` = latest residual / in-sample residual std (sample std,
        ``ddof=1``).

        Sorted by residual descending (cheapest first).
        """
        self._check_fitted()
        rows: list[dict[str, float | str]] = []

        for name, model in self.models.items():
            resid_series = model.residual()
            resid_std = float(resid_series.std(ddof=1))
            actual = float(model.actual_latest)
            fair = float(model.fair_value_latest)
            resid_latest = float(model.residual_latest)
            z_score = resid_latest / resid_std if resid_std > 0.0 else float("nan")

            rows.append(
                {
                    "Butterfly": name,
                    "Actual (bp)": actual,
                    "Fair Value (bp)": fair,
                    "Residual (bp)": resid_latest,
                    "Signal": _classify_signal(resid_latest),
                    "Z-score": z_score,
                }
            )

        columns = [
            "Butterfly",
            "Actual (bp)",
            "Fair Value (bp)",
            "Residual (bp)",
            "Signal",
            "Z-score",
        ]
        df = pd.DataFrame(rows, columns=columns)
        df = df.sort_values("Residual (bp)", ascending=False).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def plot_butterfly(
        self,
        name: str,
        figsize: tuple[float, float] = (12.0, 8.0),
        fontsize: int = 9,
    ) -> plt.Figure:
        """Four-panel chart for a single butterfly.

        Panels:

        - **Top-left**     : Actual butterfly vs Fair Value over time.
        - **Top-right**    : Residual over time (green=cheap / red=rich).
        - **Bottom-left**  : Scatter of actual vs predicted.
        - **Bottom-right** : Factor contributions bar chart (const / level /
          curve), evaluated at the latest observation.

        Parameters
        ----------
        name : str
            Butterfly name (e.g. ``"2s/5s/10s"``).
        figsize, fontsize : optional
            Matplotlib figure styling.
        """
        self._check_fitted()
        assert self._data is not None  # narrows Optional for the type checker

        model = self.get_model(name)
        comp = self._registry[name]
        belly = comp["belly"]
        slope_col = self._slope_cols[name]

        # Reconstruct the actual series from public API only:
        #   residual = actual - fair  ->  actual = fair + residual
        fair_value = model.predict()
        resid = model.residual()
        actual_series = (fair_value + resid).rename(model.y_col)

        # Coefficients / latest factor values for the decomposition panel.
        params = model.params
        const = float(params.loc["const"])
        coef_table = model.coefficient_table()
        level_beta = float(coef_table.loc[belly, "Coefficient"])
        curve_beta = float(coef_table.loc[slope_col, "Coefficient"])
        belly_latest = float(self._data[belly].iloc[-1])
        slope_latest = float(self._data[slope_col].iloc[-1])
        level_contrib = level_beta * belly_latest
        curve_contrib = curve_beta * slope_latest

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        ax_tl, ax_tr = axes[0]
        ax_bl, ax_br = axes[1]

        # --- Top-left: actual vs fair value -------------------------------
        ax_tl.plot(
            actual_series.index,
            actual_series.values,
            label="Actual",
            linewidth=1.2,
            color="#1f77b4",
        )
        ax_tl.plot(
            fair_value.index,
            fair_value.values,
            label="Fair Value",
            linewidth=1.2,
            color="#d62728",
            alpha=0.85,
        )
        ax_tl.set_title("Actual vs Fair Value", fontsize=fontsize + 2)
        ax_tl.set_xlabel("Date", fontsize=fontsize)
        ax_tl.set_ylabel("Butterfly (bp)", fontsize=fontsize)
        ax_tl.legend(loc="best", fontsize=fontsize, frameon=True)
        ax_tl.grid(True, linestyle="--", alpha=0.4)
        ax_tl.tick_params(labelsize=fontsize)

        # --- Top-right: residual with green/red shading ------------------
        resid_vals = resid.values
        ax_tr.plot(resid.index, resid_vals, color="black", linewidth=0.9)
        ax_tr.fill_between(
            resid.index,
            resid_vals,
            0.0,
            where=(resid_vals >= 0.0),
            interpolate=True,
            color="green",
            alpha=0.3,
            label="Cheap (actual > fair)",
        )
        ax_tr.fill_between(
            resid.index,
            resid_vals,
            0.0,
            where=(resid_vals < 0.0),
            interpolate=True,
            color="red",
            alpha=0.3,
            label="Rich (actual < fair)",
        )
        ax_tr.axhline(0.0, color="black", linewidth=0.8)
        ax_tr.set_title("Residual (positive = cheap)", fontsize=fontsize + 2)
        ax_tr.set_xlabel("Date", fontsize=fontsize)
        ax_tr.set_ylabel("Residual (bp)", fontsize=fontsize)
        ax_tr.legend(loc="best", fontsize=fontsize, frameon=True)
        ax_tr.grid(True, linestyle="--", alpha=0.4)
        ax_tr.tick_params(labelsize=fontsize)

        # --- Bottom-left: scatter actual vs predicted --------------------
        actual_np = actual_series.values
        predicted_np = fair_value.values
        ax_bl.scatter(
            actual_np,
            predicted_np,
            s=12,
            alpha=0.5,
            color="#1f77b4",
            edgecolors="none",
        )
        lo = float(min(np.nanmin(actual_np), np.nanmin(predicted_np)))
        hi = float(max(np.nanmax(actual_np), np.nanmax(predicted_np)))
        ax_bl.plot(
            [lo, hi],
            [lo, hi],
            color="#d62728",
            linestyle="--",
            linewidth=1.0,
            label="45° line",
        )
        ax_bl.set_xlabel("Actual (bp)", fontsize=fontsize)
        ax_bl.set_ylabel("Predicted (bp)", fontsize=fontsize)
        ax_bl.set_title("Actual vs Predicted", fontsize=fontsize + 2)
        ax_bl.legend(loc="best", fontsize=fontsize, frameon=True)
        ax_bl.grid(True, linestyle="--", alpha=0.4)
        ax_bl.tick_params(labelsize=fontsize)

        # --- Bottom-right: factor contributions --------------------------
        contribs = [const, level_contrib, curve_contrib]
        labels = ["Const", f"Level\n({belly})", f"Curve\n({slope_col})"]
        colors = ["#7f7f7f", "#1f77b4", "#ff7f0e"]
        positions = list(range(len(contribs)))
        bars = ax_br.bar(
            positions,
            contribs,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )
        ax_br.axhline(0.0, color="black", linewidth=0.8)
        ax_br.set_xticks(positions)
        ax_br.set_xticklabels(labels, fontsize=fontsize)
        ax_br.set_ylabel("Contribution to Fair Value (bp)", fontsize=fontsize)
        ax_br.set_title("Factor Contributions (latest)", fontsize=fontsize + 2)
        ax_br.grid(True, linestyle="--", alpha=0.4, axis="y")
        ax_br.tick_params(labelsize=fontsize)

        peak = max((abs(c) for c in contribs), default=1.0)
        if peak == 0.0:
            peak = 1.0
        for bar, value in zip(bars, contribs):
            offset = peak * 0.02
            va = "bottom" if value >= 0.0 else "top"
            y = value + (offset if value >= 0.0 else -offset)
            ax_br.text(
                bar.get_x() + bar.get_width() / 2.0,
                y,
                f"{value:.2f}",
                ha="center",
                va=va,
                fontsize=fontsize,
            )

        fig.suptitle(
            f"{name} Butterfly: Level & Curve Neutral Analysis",
            fontsize=fontsize + 3,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        return fig

    def plot_all_residuals(
        self,
        figsize: tuple[float, float] = (12.0, 6.0),
        fontsize: int = 9,
    ) -> plt.Figure:
        """Multi-line chart of residuals for ALL butterflies over time.

        Each butterfly gets its own colored line; a horizontal line marks
        zero. Positive residual = cheap, negative = rich.
        """
        self._check_fitted()

        fig, ax = plt.subplots(figsize=figsize)
        for name, model in self.models.items():
            resid = model.residual()
            ax.plot(
                resid.index,
                resid.values,
                linewidth=1.0,
                label=name,
            )

        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.set_title(
            "Treasury Butterflies: Rich/Cheap Residuals (Level & Curve Neutral)",
            fontsize=fontsize + 2,
        )
        ax.set_xlabel("Date", fontsize=fontsize)
        ax.set_ylabel("Residual (bp)", fontsize=fontsize)
        ax.legend(loc="best", fontsize=fontsize, frameon=True, ncol=2)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=fontsize)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        n = len(self.models)
        return f"ButterflyModel({n} butterflies fitted)"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def run_butterfly_model(
    start_date: str = "2020-01-01",
    end_date: str = "2025-07-01",
    use_bloomberg: bool = True,
) -> ButterflyModel:
    """Load model-3 data and fit the :class:`ButterflyModel`.

    Parameters
    ----------
    start_date, end_date : str
        Inclusive date bounds (``YYYY-MM-DD``).
    use_bloomberg : bool
        If ``True``, attempt to load Bloomberg data via ``xbbg`` (the data
        layer transparently falls back to synthetic data on any failure).
        If ``False``, use the synthetic generator directly.

    Returns
    -------
    ButterflyModel
        The fitted model.
    """
    data = get_model3_data(
        start_date=start_date,
        end_date=end_date,
        use_bloomberg=use_bloomberg,
    )
    model = ButterflyModel()
    model.fit(data)
    return model


# ---------------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda v: f"{v:0.4f}")

    fitted = run_butterfly_model(use_bloomberg=False)
    print(fitted)
    print("\n=== Summary Table (Figure 50 replication) ===")
    print(fitted.summary_table())
    print("\n=== Rich / Cheap Report ===")
    print(fitted.rich_cheap_report())

    # Plots (use a non-interactive-safe call; plt.show is a no-op on
    # headless backends).
    _fig_summary = fitted.plot_butterfly("2s/5s/10s")
    _fig_residuals = fitted.plot_all_residuals()
    plt.show()
