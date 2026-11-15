"""Per-market feature engineering, correlation pruning and standardisation."""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler

from .preprocess import PreprocessingLog

FEATURE_GROUPS: dict[str, list[str]] = {
    "level": ["mean_modal", "median_modal", "min_price", "max_price"],
    "dispersion": ["std_modal", "cv", "range_ratio", "mean_spread"],
    "shape": ["skewness", "kurtosis"],
    "dynamics": ["seasonal_amp", "trend_slope", "volatility", "share_above_regional"],
    "supply": ["avg_arrivals"],
}

FEATURE_DOC = {
    "mean_modal": "Mean weekly modal price (Rs/quintal)",
    "median_modal": "Median weekly modal price",
    "min_price": "Lowest weekly modal price",
    "max_price": "Highest weekly modal price",
    "std_modal": "Std of weekly modal price",
    "cv": "Coefficient of variation = std / mean",
    "range_ratio": "(max - min) / median of weekly modal price",
    "mean_spread": "Mean daily (Max-Min)/Modal intraday spread",
    "skewness": "Skewness of weekly price distribution",
    "kurtosis": "Excess kurtosis of weekly price distribution",
    "seasonal_amp": "(max - min) of calendar-month mean price / overall mean",
    "trend_slope": "Linear trend per year / mean price (relative drift)",
    "volatility": "Mean absolute week-on-week % return",
    "share_above_regional": "Share of weeks priced above the cross-market weekly median",
    "avg_arrivals": "Mean reported daily arrivals (supply volume)",
}

PRETTY = {
    "mean_modal": "price level", "median_modal": "median price", "min_price": "floor price",
    "max_price": "peak price", "std_modal": "price std", "cv": "relative variability",
    "range_ratio": "price range", "mean_spread": "intraday spread", "skewness": "price skew",
    "kurtosis": "tail heaviness", "seasonal_amp": "seasonality", "trend_slope": "price trend",
    "volatility": "weekly volatility", "share_above_regional": "premium vs region",
    "avg_arrivals": "arrival volume",
}


def compute_market_features(weekly: pd.DataFrame, rows: pd.DataFrame, coverage: pd.Series) -> pd.DataFrame:
    """Aggregate each market's price history into a feature vector (one row per market)."""
    regional = weekly.median(axis=1)
    spread = rows.groupby("market_id")["spread"].mean()
    arrivals = rows.groupby("market_id")["Arrivals"].mean() if rows["Arrivals"].notna().mean() > 0.5 else None
    out = {}
    warnings.filterwarnings("ignore", message="Precision loss", category=RuntimeWarning)  # near-constant series
    for m in weekly.columns:
        s = weekly[m].dropna()
        mean = s.mean()
        t = (s.index - s.index[0]).days.values / 365.25
        slope = np.polyfit(t, s.values, 1)[0] / mean
        monthly = s.groupby(s.index.month).mean()
        rec = {
            "mean_modal": mean, "median_modal": s.median(), "min_price": s.min(), "max_price": s.max(),
            "std_modal": s.std(), "cv": s.std() / mean, "range_ratio": (s.max() - s.min()) / s.median(),
            "mean_spread": spread.get(m, np.nan), "skewness": skew(s), "kurtosis": kurtosis(s),
            "seasonal_amp": (monthly.max() - monthly.min()) / mean, "trend_slope": slope,
            "volatility": weekly[m].pct_change(fill_method=None).abs().mean(),
            "share_above_regional": (weekly[m] > regional)[weekly[m].notna()].mean(),
        }
        if arrivals is not None:
            rec["avg_arrivals"] = arrivals.get(m, np.nan)
        out[m] = rec
    feat = pd.DataFrame(out).T.astype(float)
    feat["coverage"] = coverage.reindex(feat.index)       # diagnostic only, never clustered
    return feat


def drop_correlated(feat: pd.DataFrame, threshold: float) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Iteratively drop one feature of each pair with |r| > threshold (the one more correlated overall)."""
    corr = feat.corr()
    ac = corr.abs()
    cols, dropped = list(feat.columns), []
    while len(cols) > 1:
        sub = ac.loc[cols, cols].to_numpy(copy=True)
        np.fill_diagonal(sub, 0.0)
        if sub.max() <= threshold:
            break
        i, j = np.unravel_index(sub.argmax(), sub.shape)
        a, b = cols[i], cols[j]
        drop, keep = (a, b) if sub[i].mean() >= sub[j].mean() else (b, a)
        dropped.append({"dropped": drop, "kept_partner": keep, "abs_r": round(float(sub[i, j]), 3)})
        cols.remove(drop)
    return cols, pd.DataFrame(dropped, columns=["dropped", "kept_partner", "abs_r"]), corr


def prepare_matrix(feat: pd.DataFrame, cols: list[str], log: PreprocessingLog):
    """Median-impute remaining gaps then standardise (scaler fitted once and returned)."""
    X = feat[cols].copy()
    const = [c for c in cols if X[c].nunique(dropna=True) <= 1]
    if const:
        log.add("Scaling", f"Dropped constant feature(s) {const}: zero variance cannot be standardised")
        cols = [c for c in cols if c not in const]
        X = X[cols]
    if len(cols) < 2:
        raise ValueError("Fewer than 2 informative features remain: the markets in this file barely differ. "
                         "Use a file with more markets or a longer price history.")
    n_na = int(X.isna().sum().sum())
    X = X.fillna(X.median())
    log.add("Missing values", f"Feature-level median imputation applied to {n_na} remaining NaN cells")
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=cols)
    assert not Xs.isna().any().any(), "NaNs remain after preprocessing"
    assert np.allclose(Xs.mean(), 0, atol=1e-8) and np.allclose(Xs.std(ddof=0), 1, atol=1e-8), "scaling failed"
    log.add("Scaling", f"StandardScaler fitted once on {Xs.shape[0]} markets x {Xs.shape[1]} features "
                       "(unsupervised, so no label leakage); mean~0, std~1 asserted")
    return Xs, scaler, X
