"""Base fair-value model for Treasury / rates analysis.

Wraps :mod:`statsmodels` OLS with rich, finance-domain convenience methods:
residual-based rich/cheap interpretation, professional plots, and a clean
coefficient table. The three downstream model files (10yr, curve, butterfly)
all subclass or compose this base class.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


class FairValueModel:
    """Base OLS fair-value model for Treasury/rates analysis.

    A *fair-value* model regresses an observed market quantity (e.g. the 10y
    yield) on a set of macro / rates drivers. The fitted value is interpreted
    as the model-implied *fair value*; deviations (residuals) signal that the
    asset is *rich* (negative residual) or *cheap* (positive residual).

    Parameters
    ----------
    name : str
        Human-readable model name (used in plot titles / repr).
    description : str, optional
        Free-form description of the model's purpose.
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name: str = name
        self.description: str = description
        self._result: Optional[sm.regression.linear_model.RegressionResultsWrapper] = (
            None
        )
        self._data: Optional[pd.DataFrame] = None
        self.y_col: str = ""
        self.x_cols: list[str] = []
        self._is_fit: bool = False

    # ------------------------------------------------------------------
    # Fitting / prediction
    # ------------------------------------------------------------------
    def fit(
        self,
        data: pd.DataFrame,
        y_col: str,
        x_cols: list[str],
    ) -> "FairValueModel":
        """Fit the OLS regression (a constant is added automatically).

        Parameters
        ----------
        data : pd.DataFrame
            Training data containing ``y_col`` and every column in ``x_cols``.
        y_col : str
            Name of the dependent variable (the market quantity being modeled).
        x_cols : list[str]
            Names of the explanatory variables.

        Returns
        -------
        FairValueModel
            ``self`` (for chaining).
        """
        required = [y_col, *x_cols]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(
                f"Columns missing from training data: {missing}. "
                f"Available columns: {list(data.columns)}."
            )

        self.y_col = y_col
        self.x_cols = list(x_cols)
        self._data = data[[y_col, *x_cols]].copy()

        X = sm.add_constant(self._data[self.x_cols], has_constant="add")
        y = self._data[y_col]
        self._result = sm.OLS(y, X).fit()
        self._is_fit = True
        return self

    def _check_fitted(self) -> None:
        """Raise an informative error if the model has not been fitted yet."""
        if not self._is_fit or self._result is None or self._data is None:
            raise RuntimeError(
                f"FairValueModel '{self.name}' is not fitted yet. "
                "Call .fit(data, y_col, x_cols) before using this method/property."
            )

    def predict(self, data: Optional[pd.DataFrame] = None) -> pd.Series:
        """Predict fair value.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Out-of-sample data. If ``None``, in-sample predictions over the
            training data are returned.

        Returns
        -------
        pd.Series
            Predicted fair values aligned to ``data``'s index.
        """
        self._check_fitted()
        assert self._result is not None and self.x_cols
        frame = self._data if data is None else data
        X = sm.add_constant(frame[self.x_cols], has_constant="add")
        values = self._result.predict(X)
        return pd.Series(np.asarray(values), index=frame.index, name="fair_value")

    def residual(self, data: Optional[pd.DataFrame] = None) -> pd.Series:
        """Return ``actual - fair_value``.

        Positive residual  -> the asset is **CHEAP** (actual yield > fair value).
        Negative residual -> the asset is **RICH**  (actual yield < fair value).
        """
        self._check_fitted()
        assert self._data is not None and self.y_col
        frame = self._data if data is None else data
        actual = frame[self.y_col]
        predicted = self.predict(data)
        return (actual - predicted).rename("residual")

    # ------------------------------------------------------------------
    # Latest-value convenience properties
    # ------------------------------------------------------------------
    @property
    def fair_value_latest(self) -> float:
        """Latest fair-value prediction (last row of the training data)."""
        self._check_fitted()
        return float(self.predict().iloc[-1])

    @property
    def actual_latest(self) -> float:
        """Latest actual value of the dependent variable."""
        self._check_fitted()
        assert self._data is not None
        return float(self._data[self.y_col].iloc[-1])

    @property
    def residual_latest(self) -> float:
        """Latest residual (positive = cheap, negative = rich)."""
        return self.actual_latest - self.fair_value_latest

    # ------------------------------------------------------------------
    # statsmodels pass-throughs
    # ------------------------------------------------------------------
    @property
    def rsquared(self) -> float:
        """Model R-squared."""
        self._check_fitted()
        assert self._result is not None
        return float(self._result.rsquared)

    @property
    def params(self) -> pd.Series:
        """Regression coefficients including the constant."""
        self._check_fitted()
        assert self._result is not None
        return self._result.params

    @property
    def bse(self) -> pd.Series:
        """Standard errors of the coefficients."""
        self._check_fitted()
        assert self._result is not None
        return self._result.bse

    def summary(self) -> str:
        """Return the full statsmodels summary as a string."""
        self._check_fitted()
        assert self._result is not None
        return str(self._result.summary())

    def coefficient_table(self) -> pd.DataFrame:
        """Return a clean coefficient summary.

        Columns: ``["Coefficient", "Std Error", "t-stat", "p-value", "Significant"]``.
        ``Significant`` is ``True`` when ``p-value < 0.05``.
        """
        self._check_fitted()
        assert self._result is not None
        return pd.DataFrame(
            {
                "Coefficient": self._result.params,
                "Std Error": self._result.bse,
                "t-stat": self._result.tvalues,
                "p-value": self._result.pvalues,
                "Significant": self._result.pvalues < 0.05,
            }
        )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def plot_actual_vs_fair(
        self,
        figsize: tuple[float, float] = (12.0, 5.0),
        fontsize: int = 9,
    ) -> plt.Figure:
        """Dual-line chart of actual vs fair-value over time."""
        self._check_fitted()
        assert self._data is not None
        predicted = self.predict()
        actual = self._data[self.y_col]

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(
            actual.index, actual.values, label="Actual", linewidth=1.2, color="#1f77b4"
        )
        ax.plot(
            predicted.index,
            predicted.values,
            label="Fair Value",
            linewidth=1.2,
            color="#d62728",
            alpha=0.85,
        )
        ax.set_title(f"{self.name}: Actual vs Fair Value", fontsize=fontsize + 2)
        ax.set_xlabel("Date", fontsize=fontsize)
        ax.set_ylabel(self.y_col, fontsize=fontsize)
        ax.legend(loc="best", fontsize=fontsize, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=fontsize)
        fig.tight_layout()
        return fig

    def plot_residual(
        self,
        figsize: tuple[float, float] = (12.0, 4.0),
        fontsize: int = 9,
    ) -> plt.Figure:
        """Residual time-series chart with green (cheap) / red (rich) shading."""
        self._check_fitted()
        resid = self.residual()
        values = resid.values

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(resid.index, values, color="black", linewidth=0.9)
        ax.fill_between(
            resid.index,
            values,
            0.0,
            where=(values >= 0.0),
            interpolate=True,
            color="green",
            alpha=0.3,
            label="Cheap (actual > fair)",
        )
        ax.fill_between(
            resid.index,
            values,
            0.0,
            where=(values < 0.0),
            interpolate=True,
            color="red",
            alpha=0.3,
            label="Rich (actual < fair)",
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"{self.name}: Residual (positive = cheap)", fontsize=fontsize + 2)
        ax.set_xlabel("Date", fontsize=fontsize)
        ax.set_ylabel("Residual", fontsize=fontsize)
        ax.legend(loc="best", fontsize=fontsize, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=fontsize)
        fig.tight_layout()
        return fig

    def plot_scatter(
        self,
        figsize: tuple[float, float] = (6.0, 6.0),
        fontsize: int = 9,
    ) -> plt.Figure:
        """Scatter plot of actual vs predicted with a 45-degree reference line."""
        self._check_fitted()
        assert self._data is not None
        actual = self._data[self.y_col].values
        predicted = self.predict().values

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(
            actual, predicted, s=12, alpha=0.5, color="#1f77b4", edgecolors="none"
        )

        lo = float(min(np.nanmin(actual), np.nanmin(predicted)))
        hi = float(max(np.nanmax(actual), np.nanmax(predicted)))
        ax.plot(
            [lo, hi],
            [lo, hi],
            color="#d62728",
            linestyle="--",
            linewidth=1.0,
            label="45° line",
        )

        ax.set_xlabel("Actual", fontsize=fontsize)
        ax.set_ylabel("Predicted", fontsize=fontsize)
        ax.set_title(f"{self.name}: Actual vs Predicted", fontsize=fontsize + 2)
        ax.legend(loc="best", fontsize=fontsize, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=fontsize)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        status = "fitted" if self._is_fit else "not fitted"
        r2 = f", R²={self.rsquared:.4f}" if self._is_fit else ""
        return f"FairValueModel('{self.name}' [{status}{r2}])"
