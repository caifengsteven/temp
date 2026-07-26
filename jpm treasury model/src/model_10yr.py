"""JPMorgan 10-year Treasury Fair-Value Model.

Replicates JPMorgan US Rates Strategy's fair-value regression for the 10-year
US Treasury yield. The model explains roughly 97.5% of the variation in the
10yr yield using five macro / rates drivers:

1. **1Yx1Y OIS** (``ois_1y1y``) - medium-term Fed policy expectations.
2. **5Yx5Y breakeven** (``be_5y5y``) - medium-term inflation expectations.
3. **JPM FRI** (``jpm_fri``) - JPM US Growth Forecast Revision Index.
4. **Fed BS/GDP** (``fed_bs_gdp``) - Fed balance sheet as a share of US GDP.
5. **Trade dummy** (``trade_dummy``) - trade-policy regime dummy (1 from Apr 2025).

Per JPM: *"10-year Treasury yields tend to decline by ~10bp for every 1%
increase in the size of the Fed's balance sheet as its share of US GDP,
holding other factors constant."*

This module is a thin wrapper around :class:`src.base_model.FairValueModel`
that pins the JPM variable selection and adds domain-specific analytics:
factor decompositions (the bar chart in JPM Figure 45) and rich / cheap
signals.

Public API
----------
- :func:`run_10yr_model`      - fit and return the model.
- :func:`factor_contributions` - decompose the latest fair value by driver.
- :func:`plot_factor_contributions` - horizontal bar chart of contributions.
- :func:`fair_value_report`   - latest rich / cheap snapshot as a dict.
- :data:`X_COLS_10YR`         - canonical JPM explanatory variables.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.base_model import FairValueModel
from src.data import get_model1_data

# ---------------------------------------------------------------------------
# Variable selection (per JPM methodology)
# ---------------------------------------------------------------------------
Y_COL_10YR: str = "yield_10y"
"""Dependent variable: 10-year Treasury yield, in percent (e.g. 4.25)."""

X_COLS_10YR: list[str] = [
    "ois_1y1y",
    "be_5y5y",
    "jpm_fri",
    "fed_bs_gdp",
    "trade_dummy",
]
"""JPM explanatory variables for the 10yr fair-value regression."""

MODEL_NAME_10YR: str = "10-year Treasury Fair-Value Model"
"""Human-readable model name used in titles and repr."""

MODEL_DESCRIPTION_10YR: str = (
    "OLS regression of the 10y Treasury yield on 1y1y OIS, 5y5y breakeven, "
    "JPM FRI, Fed BS/GDP and a trade-policy regime dummy. Replicates the "
    "JPMorgan US Rates Strategy 10-year fair-value framework."
)

# Residual (in bp) beyond which the 10yr is flagged CHEAP / RICH rather than FAIR.
_SIGNAL_THRESHOLD_BP: float = 5.0


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def run_10yr_model(
    start_date: str = "2020-01-01",
    end_date: str = "2025-07-01",
    use_bloomberg: bool = True,
) -> FairValueModel:
    """Load data, fit the 10yr fair-value OLS, and return the model.

    Parameters
    ----------
    start_date, end_date : str
        Inclusive date bounds (``YYYY-MM-DD``) for the training window.
    use_bloomberg : bool, default True
        If ``True``, attempt to pull live data from Bloomberg via ``xbbg``;
        on any failure (or if a proprietary series such as ``jpm_fri`` is
        required) the data layer transparently falls back to a realistic
        synthetic generator. Pass ``False`` to force synthetic data.

    Returns
    -------
    FairValueModel
        A fitted model with ``y_col="yield_10y"`` and the five JPM drivers
        as explanatory variables.
    """
    data = get_model1_data(
        start_date=start_date,
        end_date=end_date,
        use_bloomberg=use_bloomberg,
    )
    model = FairValueModel(name=MODEL_NAME_10YR, description=MODEL_DESCRIPTION_10YR)
    model.fit(data, y_col=Y_COL_10YR, x_cols=X_COLS_10YR)
    return model


# ---------------------------------------------------------------------------
# Factor decomposition (JPM Figure 45 style)
# ---------------------------------------------------------------------------
def factor_contributions(model: FairValueModel) -> pd.DataFrame:
    """Decompose the latest 10yr fair value into per-factor contributions.

    For each explanatory variable ``x_i`` with coefficient ``beta_i`` and
    latest value ``v_i`` the contribution to fair value is
    ``beta_i * v_i``. The regression constant contributes its level
    directly. Contributions are expressed both in basis points (yields are
    in percent, so multiply by 100) and as a share of total fair value.

    This replicates the bar chart in JPM's Figure 45.

    Parameters
    ----------
    model : FairValueModel
        A fitted 10yr fair-value model (as produced by :func:`run_10yr_model`).

    Returns
    -------
    pd.DataFrame
        Columns ``["Factor", "Coefficient", "Current Value",
        "Contribution (bp)", "Contribution (%)"]``. The first row is the
        regression constant; the remaining rows are the JPM drivers in
        ``X_COLS_10YR`` order.
    """
    model._check_fitted()
    assert model._data is not None and model._result is not None

    params = model.params
    latest_row = model._data.iloc[-1]
    fair_value = model.fair_value_latest  # yield-percent units

    factors: list[str] = []
    coefficients: list[float] = []
    current_values: list[float] = []
    contributions_bp: list[float] = []

    for name, coeff in params.items():
        if name == "const":
            # The constant's "current value" is implicitly 1.0, so its
            # contribution equals the coefficient itself.
            current_value = 1.0
        else:
            current_value = float(latest_row[name])
        raw_contribution = float(coeff) * current_value  # yield-percent units
        factors.append(str(name))
        coefficients.append(float(coeff))
        current_values.append(current_value)
        contributions_bp.append(raw_contribution * 100.0)

    table = pd.DataFrame(
        {
            "Factor": factors,
            "Coefficient": coefficients,
            "Current Value": current_values,
            "Contribution (bp)": contributions_bp,
        }
    )
    # Contribution (%) = share of total fair value. Convert bp back to
    # yield-percent to match fair_value's units.
    table["Contribution (%)"] = (
        (table["Contribution (bp)"] / 100.0) / fair_value * 100.0
    )
    return table


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_factor_contributions(
    model: FairValueModel,
    figsize: tuple[float, float] = (8.0, 5.0),
    fontsize: int = 9,
) -> plt.Figure:
    """Horizontal bar chart of each driver's contribution (bp) to fair value.

    Positive contributions are drawn in blue, negative in red. Bars are
    sorted by absolute contribution with the largest at the top. The
    regression constant is excluded - it is the baseline level, not a macro
    driver - which matches the JPM Figure 45 presentation.

    Parameters
    ----------
    model : FairValueModel
        A fitted 10yr fair-value model.
    figsize : tuple[float, float], default (8.0, 5.0)
        Matplotlib figure size in inches.
    fontsize : int, default 9
        Base font size for labels and ticks.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure (caller is responsible for ``plt.show()``).
    """
    contrib = factor_contributions(model)
    # Exclude the constant - it is the baseline, not a macro factor.
    plot_df = contrib[contrib["Factor"] != "const"].copy()
    plot_df = plot_df.sort_values("Contribution (bp)", key=abs, ascending=True)

    colors = ["#d62728" if v < 0.0 else "#1f77b4" for v in plot_df["Contribution (bp)"]]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(
        plot_df["Factor"],
        plot_df["Contribution (bp)"],
        color=colors,
        edgecolor="none",
    )
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title(
        "10-year Treasury: Factor Contributions to Fair Value",
        fontsize=fontsize + 2,
    )
    ax.set_xlabel("Contribution (bp)", fontsize=fontsize)
    ax.set_ylabel("Factor", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Rich / cheap report
# ---------------------------------------------------------------------------
def fair_value_report(model: FairValueModel) -> dict[str, object]:
    """Return a summary dict of the latest 10yr rich / cheap snapshot.

    Parameters
    ----------
    model : FairValueModel
        A fitted 10yr fair-value model.

    Returns
    -------
    dict
        Keys:

        - ``actual_yield`` (float): latest 10yr yield (percent).
        - ``fair_value_yield`` (float): latest model fair value (percent).
        - ``residual_bp`` (float): ``actual - fair``, in basis points.
          Positive means CHEAP (yield trades above fair value).
        - ``signal`` (str): ``"CHEAP"`` / ``"RICH"`` / ``"FAIR"``.
        - ``rsquared`` (float): in-sample R-squared.
        - ``n_observations`` (int): training sample size.
        - ``date`` (str): latest training date (``YYYY-MM-DD``).
    """
    model._check_fitted()
    assert model._data is not None

    actual = model.actual_latest
    fair = model.fair_value_latest
    residual_bp = (actual - fair) * 100.0

    if residual_bp > _SIGNAL_THRESHOLD_BP:
        signal = "CHEAP"
    elif residual_bp < -_SIGNAL_THRESHOLD_BP:
        signal = "RICH"
    else:
        signal = "FAIR"

    latest_date = model._data.index[-1]

    return {
        "actual_yield": actual,
        "fair_value_yield": fair,
        "residual_bp": residual_bp,
        "signal": signal,
        "rsquared": model.rsquared,
        "n_observations": int(len(model._data)),
        "date": latest_date.strftime("%Y-%m-%d"),
    }


# ---------------------------------------------------------------------------
# CLI / demo entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Use synthetic data so the demo is fully runnable without Bloomberg.
    model = run_10yr_model(use_bloomberg=False)

    sep = "=" * 72
    sub = "-" * 72
    print(sep)
    print(model.name)
    print(sep)
    print(model.summary())

    print("\n" + sub)
    print("Coefficient Table")
    print(sub)
    print(model.coefficient_table().to_string())

    print("\n" + sub)
    print("Factor Contributions (latest row)")
    print(sub)
    print(factor_contributions(model).to_string(index=False))

    print("\n" + sub)
    print("Fair-Value Report")
    print(sub)
    report = fair_value_report(model)
    for key, value in report.items():
        if isinstance(value, float):
            print(f"  {key:>18}: {value:.4f}")
        else:
            print(f"  {key:>18}: {value}")

    # Plots
    _fig1 = model.plot_actual_vs_fair()
    _fig2 = model.plot_residual()
    _fig3 = plot_factor_contributions(model)
    plt.show()
