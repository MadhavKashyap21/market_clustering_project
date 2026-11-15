"""All figures: pastel palette, titled, axis-labelled, saved at 300 dpi."""
from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette

from .features import PRETTY

PAL = ["#7FB3D5", "#F0A6A0", "#86D3B3", "#B4A7E0", "#F3CF7A", "#9ED2DE", "#E7B7D8", "#BFD99A", "#F4B98A", "#A5B4C8"]
INK, GRID, MUTED = "#3B4252", "#E9E3DA", "#8A8F9C"
DIST_COL = {"euclidean": PAL[0], "manhattan": PAL[1]}
PASTEL_DIV = LinearSegmentedColormap.from_list("pastel_div", ["#E8837F", "#F6E3DF", "#FFFFFF", "#DCEBF5", "#6FA3CC"])
PASTEL_SEQ = LinearSegmentedColormap.from_list("pastel_seq", ["#FFFFFF", "#DCEBF5", "#7FB3D5"])

sns.set_theme(style="ticks", rc={
    "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF", "axes.edgecolor": "#D8D1C7", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left", "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlepad": 12, "axes.labelsize": 10.5, "legend.frameon": False,
    "figure.max_open_warning": 0, "savefig.facecolor": "#FFFFFF"})


def _save(fig: plt.Figure, path: str | Path | None) -> plt.Figure:
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def _short(name: str) -> str:
    return str(name).split(" | ")[-1]


# ----------------------------------------------------------------------------- data overview
def plot_state_counts(df: pd.DataFrame, path=None):
    s = df.groupby("State")["market_id"].nunique().sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(6.5, 0.38 * len(s) + 1.6))
    ax.barh(s.index, s.values, color=PAL[0], height=0.65)
    for y, v in enumerate(s.values):
        ax.text(v, y, f"  {v}", va="center", fontsize=9, color=INK)
    ax.set(title="Markets reporting, by state", xlabel="Number of markets", ylabel="")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_price_hist(df: pd.DataFrame, path=None):
    x = df["Modal_Price"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.hist(x, bins=45, color=PAL[2], edgecolor="white")
    ax.axvline(x.median(), color=PAL[1], ls="--", lw=2, label=f"Median Rs {x.median():,.0f}")
    ax.set(title="Modal prices after cleaning", xlabel="Rs per quintal", ylabel="Number of reports")
    ax.legend()
    return _save(fig, path)


def plot_reports_per_month(df: pd.DataFrame, path=None):
    s = df["Arrival_Date"].dt.to_period("M").value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.bar(range(len(s)), s.values, color=PAL[4], width=0.7)
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels([p.strftime("%b %y") for p in s.index], rotation=45, ha="right")
    ax.set(title="Price reports per month", xlabel="", ylabel="Number of reports")
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def plot_coverage(stats: pd.DataFrame, threshold: float, path=None):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.hist(stats["coverage"], bins=28, color=PAL[9], edgecolor="white")
    ax.axvspan(threshold, 1.0, color=PAL[2], alpha=0.25)
    ax.axvline(threshold, color=PAL[1], lw=2)
    kept, dropped = int(stats["keep"].sum()), int((~stats["keep"]).sum())
    ax.text(0.98, 0.93, f"kept {kept}  |  dropped {dropped}", transform=ax.transAxes, ha="right", va="top", fontsize=10)
    ax.set(title="How often each market reported", xlabel="Share of calendar days with a report", ylabel="Number of markets", xlim=(0, 1))
    return _save(fig, path)


def plot_cleaning(stats: dict, path=None):
    items = {k: v for k, v in stats.items() if not k.startswith("_") and v > 0}
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * max(len(items), 2) + 1.6))
    if items:
        s = pd.Series(items).sort_values()
        ax.barh(s.index, s.values, color=PAL[1], height=0.65)
        for y, v in enumerate(s.values):
            ax.text(v, y, f"  {v:,}", va="center", fontsize=9)
        ax.set(xlabel="Number of values or rows", ylabel="")
        ax.grid(axis="y", visible=False)
    else:
        ax.text(0.5, 0.5, "No repairs were needed", ha="center", va="center")
        ax.axis("off")
    ax.set_title("What the cleaning step changed")
    return _save(fig, path)


def plot_missingness(rel: pd.DataFrame, path=None):
    fig, ax = plt.subplots(figsize=(6.5, 0.36 * len(rel) + 1.6))
    ax.barh(rel["column"], rel["pct_missing"], color=PAL[3], height=0.65)
    ax.set(title="Missing values per raw column", xlabel="% missing", ylabel="")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_feature_distributions(feat: pd.DataFrame, cols: list[str], path=None):
    ncols = 4
    nrows = ceil(len(cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows), squeeze=False)
    for i, (ax, c) in enumerate(zip(axes.ravel(), cols)):
        ax.hist(feat[c].dropna(), bins=20, color=PAL[i % 10], edgecolor="white")
        ax.set_title(PRETTY.get(c, c), fontsize=10.5)
        ax.set_yticks([])
        ax.grid(False)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.suptitle("How each market feature is distributed", x=0.01, ha="left", fontweight="bold", fontsize=13)
    return _save(fig, path)


def plot_correlation(corr: pd.DataFrame, threshold: float, path=None):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, cmap=PASTEL_DIV, vmin=-1, vmax=1, annot=True, fmt=".1f", annot_kws={"size": 7}, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set(title=f"Feature correlation matrix (pruned where |r| > {threshold})")
    ax.grid(False)
    return _save(fig, path)


# ----------------------------------------------------------------------------- choosing K
def plot_elbow(grid: pd.DataFrame, chosen=None, path=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for ax, d, lab in [(axes[0], "euclidean", "WCSS (inertia)"), (axes[1], "manhattan", "Total L1 cost")]:
        g = grid[grid["distance"] == d]
        ax.plot(g["k"], g["cost"], "o-", color=DIST_COL[d], lw=2.5, ms=7, mec="white")
        ax.set(title=f"Elbow curve: {d.title()}", xlabel="Number of clusters K", ylabel=lab)
        if chosen is not None:
            ax.axvline(chosen, ls="--", color=MUTED, lw=1)
    return _save(fig, path)


def plot_metric(grid: pd.DataFrame, col: str, title: str, chosen=None, path=None):
    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    for d in ["euclidean", "manhattan"]:
        g = grid[grid["distance"] == d]
        ax.plot(g["k"], g[col], "o-", color=DIST_COL[d], lw=2.5, ms=7, mec="white", label=d.title())
    if chosen is not None:
        ax.axvline(chosen, ls="--", color=MUTED, lw=1)
    ax.set(title=title, xlabel="Number of clusters K", ylabel=col.replace("_", " ").title())
    ax.legend()
    return _save(fig, path)


def plot_metric_curves(grid: pd.DataFrame, chosen=None, path=None):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    for ax, d, lab in [(axes[0], "euclidean", "WCSS (inertia)"), (axes[1], "manhattan", "Total L1 cost")]:
        g = grid[grid["distance"] == d]
        ax.plot(g["k"], g["cost"], "o-", color=DIST_COL[d], lw=2.5, ms=7, mec="white")
        ax.set(title=f"Elbow curve - {d.title()}", xlabel="Number of clusters K", ylabel=lab)
    for ax, col, ttl in [(axes[2], "silhouette", "Silhouette score (higher = better)"),
                         (axes[3], "davies_bouldin", "Davies-Bouldin index (lower = better)")]:
        for d in ["euclidean", "manhattan"]:
            g = grid[grid["distance"] == d]
            ax.plot(g["k"], g[col], "o-", color=DIST_COL[d], lw=2.5, ms=7, mec="white", label=d.title())
        ax.set(title=ttl, xlabel="Number of clusters K", ylabel=col.replace("_", " ").title())
        ax.legend()
    if chosen is not None:
        for ax in axes:
            ax.axvline(chosen, ls="--", color=MUTED, lw=1)
    return _save(fig, path)


def plot_composite(ranked: pd.DataFrame, path=None):
    r = ranked.head(10).iloc[::-1]
    labels = [f"{d.title()}  K={int(k)}" for d, k in zip(r["distance"], r["k"])]
    fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(r) + 1.6))
    bars = ax.barh(labels, r["composite"], color=[DIST_COL[d] for d in r["distance"]], height=0.65)
    bars[-1].set_edgecolor(INK)
    bars[-1].set_linewidth(1.6)
    for y, v in enumerate(r["composite"]):
        ax.text(v, y, f"  {v:.2f}", va="center", fontsize=9)
    ax.set(title="Selection score (best configuration outlined)", xlabel="Composite score (0 to 1)", ylabel="")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


# ----------------------------------------------------------------------------- PCA and hierarchy
def plot_scree(evr: np.ndarray, path=None):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = np.arange(1, len(evr) + 1)
    ax.bar(x, evr, color=PAL[0], label="Per component", width=0.7)
    ax.plot(x, np.cumsum(evr), "o-", color=PAL[1], lw=2.5, ms=7, mec="white", label="Cumulative")
    ax.set(title="PCA scree plot", xlabel="Principal component", ylabel="Explained variance ratio")
    ax.legend()
    return _save(fig, path)


def plot_pca_scatter(P: np.ndarray, index, evr: np.ndarray, panels: list, n_outliers: int = 3, path=None):
    """panels: list of (title, labels, centers_in_PC_space)."""
    ncols = min(len(panels), 2)
    nrows = ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.8 * nrows), squeeze=False)
    for ax, (title, labels, cen) in zip(axes.ravel(), panels):
        for c in np.unique(labels):
            m = labels == c
            ax.scatter(P[m, 0], P[m, 1], s=48, alpha=0.9, color=PAL[c % 10], edgecolor="white", linewidth=0.6,
                       label=f"Cluster {c} (n={m.sum()})")
        ax.scatter(cen[:, 0], cen[:, 1], marker="X", s=210, c=INK, edgecolor="white", label="Centroid")
        dist = np.linalg.norm(P[:, :2] - cen[labels, :2], axis=1)
        for i in np.argsort(dist)[-n_outliers:]:
            ax.annotate(_short(index[i]), (P[i, 0], P[i, 1]), fontsize=7.5, xytext=(4, 4), textcoords="offset points")
        ax.set(title=title, xlabel=f"PC1 ({evr[0]:.1%} var)", ylabel=f"PC2 ({evr[1]:.1%} var)")
        ax.legend(fontsize=8)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    return _save(fig, path)


def plot_loadings(pca, cols: list[str], n: int = 3, path=None):
    L = pd.DataFrame(pca.components_[:n].T, index=[PRETTY.get(c, c) for c in cols], columns=[f"PC{i + 1}" for i in range(n)])
    fig, ax = plt.subplots(figsize=(5.2, 0.45 * len(cols) + 2))
    sns.heatmap(L, cmap=PASTEL_DIV, center=0, annot=True, fmt=".2f", ax=ax, linewidths=0.5, linecolor="white")
    ax.set(title="PCA loadings", xlabel="", ylabel="")
    ax.grid(False)
    return _save(fig, path), L


def plot_dendrogram(Z: np.ndarray, k: int, path=None):
    set_link_color_palette(PAL)
    fig, ax = plt.subplots(figsize=(11, 5))
    cut = (Z[-k, 2] + Z[-(k - 1), 2]) / 2
    dendrogram(Z, truncate_mode="lastp", p=30, ax=ax, color_threshold=cut, leaf_rotation=90, leaf_font_size=8,
               show_contracted=True, above_threshold_color="#B8B2A7")
    ax.axhline(cut, ls="--", color=INK, lw=1, label=f"Cut for K={k}")
    ax.set(title="Ward linkage dendrogram (truncated to 30 leaves)", xlabel="Markets (merged groups show size)",
           ylabel="Ward distance")
    ax.legend()
    return _save(fig, path)


def plot_crosstab(ct: pd.DataFrame, path=None):
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    sns.heatmap(ct, annot=True, fmt="d", cmap=PASTEL_SEQ, ax=ax, linewidths=1, linecolor="white", cbar=False)
    ax.set(title="Chosen partition vs Ward clusters")
    ax.grid(False)
    return _save(fig, path)


# ----------------------------------------------------------------------------- profiles
def plot_cluster_sizes(sizes: pd.Series, path=None):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar([f"C{c}" for c in sizes.index], sizes.values, color=[PAL[c % 10] for c in sizes.index], width=0.65)
    for x, v in enumerate(sizes.values):
        ax.text(x, v, f"{v}", ha="center", va="bottom", fontsize=10)
    ax.set(title="Markets per cluster", xlabel="", ylabel="Number of markets")
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def plot_centroid_heatmap(centroids_z: pd.DataFrame, names: dict, path=None):
    fig, ax = plt.subplots(figsize=(11, 0.9 * len(centroids_z) + 2.5))
    T = centroids_z.copy()
    T.columns = [PRETTY.get(c, c) for c in T.columns]
    T.index = [f"C{c}: {names[c]}" for c in T.index]
    sns.heatmap(T, cmap=PASTEL_DIV, center=0, annot=True, fmt=".2f", annot_kws={"size": 8}, ax=ax, linewidths=0.5, linecolor="white")
    ax.set(title="Standardised cluster centroids (z-scores)", xlabel="", ylabel="")
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    return _save(fig, path)


def plot_radar(cz: pd.DataFrame, path=None, n_features: int = 8):
    cols = list((cz.max() - cz.min()).sort_values(ascending=False).head(n_features).index)
    ang = np.linspace(0, 2 * np.pi, len(cols), endpoint=False)
    fig, ax = plt.subplots(figsize=(6.2, 6.2), subplot_kw={"projection": "polar"})
    for c in cz.index:
        v = cz.loc[c, cols].values
        ax.plot(np.append(ang, ang[0]), np.append(v, v[0]), color=PAL[c % 10], lw=2.5, label=f"C{c}")
        ax.fill(np.append(ang, ang[0]), np.append(v, v[0]), color=PAL[c % 10], alpha=0.18)
    ax.set_xticks(ang)
    ax.set_xticklabels([PRETTY.get(c, c) for c in cols], fontsize=9)
    ax.set_ylim(min(-1.5, float(cz[cols].min().min())), max(1.5, float(cz[cols].max().max())))
    ax.grid(color=GRID)
    ax.set_title("Cluster fingerprints (z-scores, most distinguishing features)", loc="left", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.1))
    return _save(fig, path)


def plot_trends(weekly: pd.DataFrame, labels: np.ndarray, index, path=None):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    labels = np.asarray(labels)
    for c in np.unique(labels):
        members = [m for m, l in zip(index, labels) if l == c]
        s = weekly[members].median(axis=1)
        ax.plot(s.index, s.values, color=PAL[c % 10], lw=2.6, label=f"C{c} (n={len(members)})")
    ax.set(title="Median weekly modal price by cluster", xlabel="", ylabel="Rs per quintal")
    ax.legend(ncol=3)
    fig.autofmt_xdate()
    return _save(fig, path)


def plot_state_composition(ct: pd.DataFrame, path=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = PAL[: ct.shape[1]] if ct.shape[1] <= 10 else sns.blend_palette(PAL, ct.shape[1])
    ct.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.7, edgecolor="white")
    ax.set(title="Which states each cluster draws from (market counts)", xlabel="Cluster", ylabel="Number of markets")
    ax.legend(title="State", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(axis="x", visible=False)
    plt.setp(ax.get_xticklabels(), rotation=0)
    return _save(fig, path)


def plot_contrib(contrib: pd.DataFrame, path=None):
    s = contrib["eta_squared"].head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(s) + 1.6))
    ax.barh([PRETTY.get(i, i) for i in s.index], s.values, color=PAL[3], height=0.65)
    for y, v in enumerate(s.values):
        ax.text(v, y, f"  {v:.2f}", va="center", fontsize=9)
    ax.set(title="Which features separate the clusters", xlabel="Share of variance explained by cluster (eta squared)", ylabel="")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


# ----------------------------------------------------------------------------- stability
def plot_components(components: dict, path=None):
    s = pd.Series(components).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(s) + 1.8))
    ax.axvspan(0, 0.5, color=PAL[1], alpha=0.14)
    ax.axvspan(0.5, 0.8, color=PAL[4], alpha=0.16)
    ax.axvspan(0.8, 1.0, color=PAL[2], alpha=0.2)
    ax.barh(s.index, s.values.clip(min=0), color=PAL[0], height=0.55)
    for y, v in enumerate(s.values):
        ax.text(max(v, 0), y, f"  {v:.2f}", va="center", fontsize=10)
    ax.set_ylim(-0.95, len(s) - 0.4)
    for x, t in [(0.25, "unstable"), (0.65, "moderate"), (0.9, "stable")]:
        ax.text(x, -0.78, t, ha="center", fontsize=9, color=MUTED)
    ax.set(title="How well the clusters survive changes (mean ARI)", xlabel="Adjusted Rand Index", ylabel="", xlim=(0, 1))
    ax.grid(False)
    return _save(fig, path)


def plot_stability(seed_df: pd.DataFrame, boot_df: pd.DataFrame, sens: pd.DataFrame, path=None):
    data = pd.concat([
        pd.DataFrame({"test": "Random seeds", "ARI": seed_df["ari_vs_final"]}),
        pd.DataFrame({"test": "Bootstrap 80%", "ARI": boot_df["ari_vs_final"]}),
        pd.DataFrame({"test": sens["scenario"], "ARI": sens["ari_vs_final"]}),
    ])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.boxplot(data=data[data["test"].isin(["Random seeds", "Bootstrap 80%"])], x="test", y="ARI", ax=ax, color=PAL[0],
                width=0.45, linecolor=INK, linewidth=1)
    sns.stripplot(data=data[~data["test"].isin(["Random seeds", "Bootstrap 80%"])], x="test", y="ARI", ax=ax,
                  color=PAL[1], s=11, edgecolor="white", linewidth=1)
    ax.axhline(0.8, ls="--", color=PAL[2], lw=1.5)
    ax.axhline(0.5, ls="--", color=PAL[4], lw=1.5)
    ax.set(title="Stability: agreement with the final solution", xlabel="", ylim=(-0.05, 1.05), ylabel="Adjusted Rand Index")
    plt.setp(ax.get_xticklabels(), rotation=12)
    return _save(fig, path)


# ----------------------------------------------------------------------------- farmer advisor
def plot_net_prices(ranking: pd.DataFrame, n: int = 15, path=None):
    r = ranking.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(r) + 1.8))
    y = np.arange(len(r))
    ax.hlines(y, r["net_price_per_q"], r["recent_price"], color="#D8D1C7", lw=2)
    ax.scatter(r["recent_price"], y, s=70, color="white", edgecolor=PAL[1], linewidth=2, zorder=3, label="Listed price")
    ax.scatter(r["net_price_per_q"], y, s=70, color=PAL[2], edgecolor="white", linewidth=1, zorder=3, label="After transport and fee")
    ax.set_yticks(y)
    ax.set_yticklabels([_short(i) for i in r.index], fontsize=9)
    ax.set(title="Listed price vs what a farmer keeps", xlabel="Rs per quintal")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)
