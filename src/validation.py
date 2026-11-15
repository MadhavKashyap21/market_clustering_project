"""Model selection, cluster profiling, feature contributions and stability analysis."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.metrics import adjusted_rand_score

from .clustering import fit_model
from .features import PRETTY


def _mm(s: pd.Series) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0.0


def elbow_k(grid: pd.DataFrame, distance: str) -> int:
    """Chord (kneedle-style) elbow: K with max distance to the line joining first and last point."""
    g = grid[grid["distance"] == distance].sort_values("k")
    x, y = _mm(g["k"].astype(float)).values, _mm(g["cost"]).values
    d = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0])
    return int(g["k"].values[d.argmax()])


def select_configuration(grid: pd.DataFrame, cfg: dict[str, Any]):
    """Documented decision rule.

    1. Eligible: K >= min_k_for_selection and smallest cluster >= min_cluster_frac of markets.
    2. composite = mean of min-max normalised [silhouette, -Davies-Bouldin, PCA-2D silhouette].
    3. Highest composite wins; Manhattan must beat the preferred (Euclidean) best by > tie_tolerance.
    4. Runner-up = next best different (distance, K).
    """
    g = grid.copy()
    g["is_elbow"] = [k == elbow_k(grid, d) for d, k in zip(g["distance"], g["k"])]
    elig = g[(g["k"] >= cfg["min_k_for_selection"]) & (g["min_cluster_frac"] >= cfg["min_cluster_frac"])].copy()
    if elig.empty:
        elig = g[g["k"] >= cfg["min_k_for_selection"]].copy()
    elig["composite"] = (_mm(elig["silhouette"]) + _mm(-elig["davies_bouldin"]) + _mm(elig["pca2_silhouette"])) / 3
    ranked = elig.sort_values("composite", ascending=False).reset_index(drop=True)
    best = ranked.iloc[0]
    pref = cfg["prefer_distance"]
    pref_rows = ranked[ranked["distance"] == pref]
    if best["distance"] != pref and not pref_rows.empty:   # preferred distance may have no eligible K (tiny clusters)
        if best["composite"] - pref_rows.iloc[0]["composite"] <= cfg["tie_tolerance"]:
            best = pref_rows.iloc[0]
    same = (ranked["distance"] == best["distance"]) & (ranked["k"] == best["k"])
    others = ranked[~same]
    if others.empty:   # small files: fall back to the best remaining configuration, ignoring the size rule
        pool = g[~((g["distance"] == best["distance"]) & (g["k"] == best["k"]))]
        others = pool.assign(composite=np.nan).sort_values("silhouette", ascending=False)
    runner = others.iloc[0] if len(others) else best
    return ranked, best, runner


def label_clusters(centroids_z: pd.DataFrame) -> dict[int, str]:
    names = {}
    for c, row in centroids_z.iterrows():
        top = row.abs().sort_values(ascending=False).head(2)
        parts = [f"{'high' if row[f] > 0 else 'low'} {PRETTY.get(f, f)}" for f in top.index if abs(row[f]) >= 0.5]
        names[c] = (", ".join(parts) if parts else "near-average profile") + " (provisional)"
    return names


def feature_contributions(X: pd.DataFrame, labels: np.ndarray, centroids_z: pd.DataFrame) -> pd.DataFrame:
    F, p = f_classif(X.values, labels)
    grand = X.mean()
    ssb = sum(((X[labels == c].mean() - grand) ** 2) * (labels == c).sum() for c in np.unique(labels))
    sst = ((X - grand) ** 2).sum()
    out = pd.DataFrame({"anova_F": F, "p_value": p, "eta_squared": (ssb / sst).values,
                        "centroid_range": (centroids_z.max() - centroids_z.min()).values}, index=X.columns)
    return out.sort_values("eta_squared", ascending=False)


def seed_stability(X: np.ndarray, ref: np.ndarray, k: int, dist: str, n_seeds: int, n_init: int) -> pd.DataFrame:
    from .clustering import davies_bouldin, METRIC
    from sklearn.metrics import silhouette_score
    rows = []
    for s in range(n_seeds):
        r = fit_model(X, k, dist, seed=1000 + s, n_init=n_init)
        rows.append({"seed": 1000 + s, "ari_vs_final": adjusted_rand_score(ref, r.labels),
                     "silhouette": silhouette_score(X, r.labels, metric="euclidean" if dist == "euclidean" else "manhattan"),
                     "davies_bouldin": davies_bouldin(X, r.labels, r.centers, METRIC[dist]), "cost": r.cost})
    return pd.DataFrame(rows)


def bootstrap_stability(X: np.ndarray, ref: np.ndarray, k: int, dist: str, n_boot: int, frac: float,
                        seed: int, n_init: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_boot):
        idx = rng.choice(len(X), size=int(frac * len(X)), replace=False)
        r = fit_model(X[idx], k, dist, seed=seed + b, n_init=n_init)
        rows.append({"boot": b, "ari_vs_final": adjusted_rand_score(ref[idx], r.labels)})
    return pd.DataFrame(rows)


def verdict(score: float) -> str:
    return "stable" if score >= 0.8 else "moderately stable" if score >= 0.5 else "unstable"
