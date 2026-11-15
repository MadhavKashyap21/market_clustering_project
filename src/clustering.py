"""K-Means (Euclidean), K-Medians (Manhattan), metrics grid and Ward hierarchical clustering."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

METRIC = {"euclidean": "euclidean", "manhattan": "cityblock"}


@dataclass
class FitResult:
    labels: np.ndarray
    centers: np.ndarray
    cost: float          # WCSS for Euclidean, total L1 cost for Manhattan


def _kpp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centers = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = cdist(X, np.array(centers), "cityblock").min(axis=1) ** 2
        p = d2 / d2.sum() if d2.sum() > 0 else None
        centers.append(X[rng.choice(len(X), p=p)])
    return np.array(centers)


def kmedians(X: np.ndarray, k: int, seed: int, n_init: int = 20, max_iter: int = 300) -> FitResult:
    """K-Medians: L1 (Manhattan) assignment + coordinate-wise median update.

    Used for the Manhattan variant because the mean is NOT the minimiser of summed L1 distance;
    the median is. Plain KMeans with a Manhattan metric would therefore be inconsistent.
    """
    rng = np.random.default_rng(seed)
    best: FitResult | None = None
    for _ in range(n_init):
        C = _kpp_init(X, k, rng)
        for _ in range(max_iter):
            D = cdist(X, C, "cityblock")
            lab = D.argmin(axis=1)
            newC = C.copy()
            for j in range(k):
                pts = X[lab == j]
                newC[j] = np.median(pts, axis=0) if len(pts) else X[D.min(axis=1).argmax()]
            if np.abs(newC - C).sum() < 1e-9:
                C = newC
                break
            C = newC
        D = cdist(X, C, "cityblock")
        res = FitResult(D.argmin(axis=1), C, float(D.min(axis=1).sum()))
        if best is None or res.cost < best.cost:
            best = res
    assert best is not None
    return best


def fit_model(X: np.ndarray, k: int, distance: str, seed: int, n_init: int) -> FitResult:
    if distance == "euclidean":
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)
        return FitResult(km.labels_, km.cluster_centers_, float(km.inertia_))
    if distance == "manhattan":
        return kmedians(X, k, seed, n_init)
    raise ValueError(distance)


def davies_bouldin(X: np.ndarray, labels: np.ndarray, centers: np.ndarray, metric: str) -> float:
    """Davies-Bouldin index with a matching metric (sklearn's version is Euclidean only)."""
    ks = np.unique(labels)
    assert len(ks) == len(centers), "empty cluster"
    S = np.array([cdist(X[labels == j], centers[[j]], metric).mean() for j in ks])
    M = cdist(centers, centers, metric)
    np.fill_diagonal(M, np.inf)
    R = (S[:, None] + S[None, :]) / M
    np.fill_diagonal(R, -np.inf)
    return float(R.max(axis=1).mean())


def run_grid(X: np.ndarray, P2: np.ndarray, ks: range, distances: list[str], seed: int, n_init: int):
    """Fit every (distance, K); return tidy metrics frame and the fitted models."""
    rows, fits = [], {}
    for dist in distances:
        for k in ks:
            r = fit_model(X, k, dist, seed, n_init)
            fits[(dist, k)] = r
            sizes = np.bincount(r.labels, minlength=k)
            rows.append({
                "distance": dist, "k": k, "cost": r.cost,
                "silhouette": silhouette_score(X, r.labels, metric=METRIC[dist] if dist == "euclidean" else "manhattan"),
                "davies_bouldin": davies_bouldin(X, r.labels, r.centers, METRIC[dist]),
                "pca2_silhouette": silhouette_score(P2, r.labels),
                "min_cluster_size": int(sizes.min()), "min_cluster_frac": sizes.min() / len(X),
            })
    return pd.DataFrame(rows), fits


def ward_hierarchy(X: np.ndarray, k: int):
    """Ward linkage, cut at k clusters, cophenetic correlation of the dendrogram."""
    Z = linkage(X, method="ward")
    labels = fcluster(Z, t=k, criterion="maxclust") - 1
    coph, _ = cophenet(Z, pdist(X))
    return Z, labels, float(coph)
