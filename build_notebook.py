"""Builds notebook.ipynb (then execute with: jupyter nbconvert --execute --inplace notebook.ipynb)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
M = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s.strip()))
C = lambda s: nb.cells.append(nbf.v4.new_code_cell(s.strip()))

M("""# Regional Market Price Behaviour Segmentation for Comparative Pattern Analysis
**Units II, III & IV - Partition Clustering, Hierarchical Comparison, PCA Visualisation**

Unit of analysis: **markets (mandis)**, each summarised by a feature vector built from its price history.
`RANDOM_STATE = 42` everywhere. The data file is set in `config.yaml`; if it is absent a clearly labelled **synthetic** dataset
is generated so this notebook still runs - synthetic results are for demonstration only.""")
C("""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.metrics import adjusted_rand_score
from src import preprocess as pp, features as ft, clustering as cl, validation as va, visualise as vz, pipeline as pl

cfg = pp.load_config("config.yaml")
OUT, FIG = Path(cfg["output_dir"]), Path(cfg["output_dir"]) / "figures"
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
log = pp.PreprocessingLog()
""")
M("## Objective 1 - Data preparation\nLoad, filter to one commodity, review variables, repair impossible values, handle missing data.")
C("""
c = pl.stage_clean(cfg, log)
df = c["df"]
print("\\nShape:", df.shape, "| date range:", df.Arrival_Date.min().date(), "->", df.Arrival_Date.max().date())
display(df.groupby("State")["market_id"].nunique().rename("markets per state").to_frame().T)
""")
C("""
display(c["relevance"])                       # variable relevance review (keep / drop + justification)
vz.plot_missingness(c["relevance"], FIG / "06_missingness.png"); plt.show()
""")
M("""### Feature engineering, correlation pruning, standardisation
Prices are aggregated to **market-week medians** (this collapses variety/grade mixing). Markets with <60% day coverage are dropped, short gaps
are time-interpolated, outliers are winsorised (clipped, never silently deleted). `State` is a post-hoc label only (not a feature).""")
C("""
f = pl.stage_features(df, cfg, log)
X = f["X"]
print("\\n", len(f["dropped"]), "markets dropped (first 15 shown):"); display(f["dropped"].head(15))
display(f["feat"].describe().T.round(3))
vz.plot_feature_distributions(f["feat"], f["initial"], FIG / "07_feature_distributions.png"); plt.show()
display(f["corr_dropped"])
vz.plot_correlation(f["corr"].loc[f["initial"], f["initial"]], cfg["preprocessing"]["corr_threshold"], FIG / "08_correlation.png"); plt.show()
""")
M("## Objective 2 - Clustering workflow\nK-Means (Euclidean) and K-Medians (Manhattan: L1 assignment + median update) for K = 2..10; WCSS/L1 cost, Silhouette, Davies-Bouldin with matching metric.")
C("""
g = pl.stage_grid(X, cfg)
grid = g["grid"]
display(grid.round(4))
""")
C("""
ranked, best, runner = va.select_configuration(grid, cfg)
dist, k = best["distance"], int(best["k"])
vz.plot_metric_curves(grid, k, FIG / "12_metric_curves_all.png"); plt.show()
""")
M("### PCA - variance explained and loadings")
C("""
pca, P = g["pca"], g["P"]
vz.plot_scree(pca.explained_variance_ratio_, FIG / "14_pca_scree.png"); plt.show()
_, loadings = vz.plot_loadings(pca, list(X.columns), path=FIG / "15_pca_loadings.png"); plt.show()
""")
M("""### Decision rule and final configuration
Eligible: K >= 3 and smallest cluster >= 5% of markets. Score = mean of min-max-normalised **Silhouette, -Davies-Bouldin, PCA-2D silhouette**.
Euclidean is preferred unless Manhattan wins by more than the tie tolerance. Elbow K is flagged for reference. Runner-up is the next-best different configuration.""")
C("""
display(ranked.head(8).round(4))
print(f"CHOSEN: {dist} K={k}   |   RUNNER-UP: {runner['distance']} K={int(runner['k'])}")
fit = g["fits"][(dist, k)]
panels = []
for d in ["euclidean", "manhattan"]:
    for _, r in ranked[ranked.distance == d].head(2).iterrows():
        m = g["fits"][(d, int(r.k))]
        panels.append((f"{d.title()} K={int(r.k)}", m.labels, pca.transform(m.centers)[:, :2]))
vz.plot_pca_scatter(P, X.index, pca.explained_variance_ratio_, panels, path=FIG / "16_pca_clusters.png"); plt.show()
""")
M("### Hierarchical comparison (Unit III) - Ward linkage vs the chosen partition")
C("""
Z, hl, coph = cl.ward_hierarchy(X.values, k)
ari_h = adjusted_rand_score(fit.labels, hl)
print(f"Cophenetic correlation (Ward): {coph:.3f} | ARI Ward vs {dist} K={k}: {ari_h:.3f}")
display(pd.crosstab(pd.Series(fit.labels, name="Partition"), pd.Series(hl, name="Ward")))
vz.plot_dendrogram(Z, k, FIG / "17_dendrogram.png"); plt.show()
""")
M("## Objective 3 - Interpretation and responsible reporting\nProfiles, feature contributions, stability, limitations. **Cluster names are provisional, auto-generated descriptions, not real-world categories.**")
C("""
it = pl.stage_interpret(X, f, fit)
display(pd.DataFrame({"name": it["names"], "n_markets": it["sizes"]}))
print("Mean of original-scale features per cluster:"); display(it["mean"].round(3).T)
print("Median:"); display(it["median"].round(3).T)
vz.plot_centroid_heatmap(it["centroids_z"], it["names"], FIG / "21_centroid_heatmap.png"); plt.show()
vz.plot_cluster_sizes(it["sizes"], FIG / "19_cluster_sizes.png"); plt.show()
vz.plot_radar(it["centroids_z"], FIG / "20_cluster_radar.png"); plt.show()
vz.plot_trends(f["weekly"], it["labels"], X.index, FIG / "22_price_trends.png"); plt.show()
vz.plot_state_composition(it["state_ct"], FIG / "23_state_composition.png"); plt.show()
""")
C("""
print("Feature contributions (ANOVA F, eta-squared effect size, centroid range) - plus PCA loadings above")
display(it["contrib"].round(4))
vz.plot_contrib(it["contrib"], FIG / "24_feature_drivers.png"); plt.show()
""")
M("""### Stability analysis
(a) 20 random single-start seeds, (b) 50 bootstrap 80% subsamples, (c) Euclidean vs Manhattan labels, (d) sensitivity: no winsorising / without shape features / without dynamics features.
Verdict from the mean ARI: >= 0.8 stable, 0.5-0.8 moderately stable, < 0.5 unstable.""")
C("""
st = pl.stage_stability(df, cfg, f, dist, k, fit.labels)
display(pd.Series(st["components"], name="mean ARI").round(3).to_frame())
display(st["sens"].round(3))
print(f"Overall mean ARI = {st['overall']:.3f}  ->  {st['verdict'].upper()}")
vz.plot_stability(st["seeds"], st["boot"], st["sens"], FIG / "26_stability.png"); plt.show()
""")
M("""### Farmer-facing layer (SIH26132) - where to sell after transport
Per-market recent price (4-week median), momentum and volatility are combined with distance to a farmer location.
**Net price = price x (1 - fee) - transport rate x road km.** Transport rate and fee are configurable *assumptions*. Coordinates are *approximate* place centroids matched from State/District/Market names (or your own `data/mandi_coordinates.csv`).""")
C("""
from src import advisor as adv
coords = adv.market_coordinates(X.index, cfg, c["synthetic"], df)
if coords is None:
    print("No data/mandi_coordinates.csv - distance-based ranking skipped (see README). Showing price behaviour by cluster instead.")
    snap = adv.build_snapshot(f["weekly"], f["feat"], it["labels"], it["names"], X.index)
    snap.to_csv(OUT / "market_snapshot.csv")
    display(snap.groupby("cluster_name").agg(markets=("State", "size"), median_recent_price=("recent_price", "median"),
            avg_momentum_pct=("momentum_pct", "mean"), avg_volatility=("volatility", "mean")).round(3))
else:
    snap = adv.build_snapshot(f["weekly"], f["feat"], it["labels"], it["names"], X.index).join(coords)
    a = cfg["advisor"]
    print(f"Coordinates for {int(snap['lat'].notna().sum())} of {len(snap)} markets:", snap["coord_source"].value_counts().to_dict())
    ranking, csum = adv.rank_markets(snap, a["farmer_lat"], a["farmer_lon"], a["quantity_quintal"],
                                     a["transport_rate_per_quintal_km"], a["road_factor"], a["market_fee_pct"], a["max_km"])
    snap.to_csv(OUT / "market_snapshot.csv"); ranking.to_csv(OUT / "advisor_demo_ranking.csv"); csum.to_csv(OUT / "advisor_demo_clusters.csv", index=False)
    display(csum.round(2))
    display(ranking[["State", "cluster", "recent_price", "road_km", "net_price_per_q", "rank_net", "rank_listed"]].head(8).round(1))
""")
M("""### Limitations and possible bias
- **Reporting gaps / uneven coverage:** thin reporters are dropped or interpolated; clusters describe well-reporting mandis and may under-represent remote ones.
- **Variety and grade mixing, arrival effects:** medians hide quality and supply differences that can look like price behaviour.
- **Short window:** about 12 months of one commodity (wheat), so seasonality, trend and harvest-season arrival surges cannot be separated from one-off events.
- **Urban vs remote reporting bias:** better-connected mandis report more consistently.
- **K-Means assumptions:** spherical, similar-variance clusters; sensitive to scaling and outliers (hence standardising and winsorising).
- **PCA / metrics:** PCA 2-D plots lose variance; silhouette and Davies-Bouldin are heuristics and the two distances' scores are not strictly comparable.

### Responsible-use statement
These clusters are **exploratory, data-driven groupings** for one commodity and one time window. They are **not permanent market categories** and must not be used for
pricing decisions, policy targeting, or labelling regions without domain validation.""")
C("""
import json
assign = pd.DataFrame({"market_id": X.index, "State": it["states"].values, "cluster": it["labels"],
                       "cluster_name": [it["names"][x] for x in it["labels"]], "ward_cluster": hl,
                       "PC1": P[:, 0], "PC2": P[:, 1]}).join(f["imputed"], on="market_id")
for name, tbl in {"metrics_grid": grid, "selection_ranking": ranked, "cluster_assignments": assign,
                  "feature_contributions": it["contrib"]}.items():
    tbl.to_csv(OUT / f"{name}.csv", index=(name == "feature_contributions"))
log.add("Model selection", f"Chosen: {dist} K={k}; runner-up {runner['distance']} K={int(runner['k'])}")
log.save(OUT / "preprocessing_log.md")

print("=" * 70, "\\nRESULTS SUMMARY", "\\n" + "=" * 70)
print("Data           :", "SYNTHETIC (demonstration only)" if c["synthetic"] else "real Agmarknet data")
print(f"Chosen config  : {dist} K={k}  (runner-up: {runner['distance']} K={int(runner['k'])})")
print(f"Metrics        : silhouette={best['silhouette']:.3f}, Davies-Bouldin={best['davies_bouldin']:.3f}, cost={best['cost']:.1f}")
print(f"Stability      : {st['verdict']} (mean ARI {st['overall']:.3f})")
print("Top 3 features :", ", ".join(it["contrib"].index[:3]))
print("Key limitations:"); [print("  -", s) for s in pl.LIMITATIONS[:3]]
""")
C("""
print("Folder tree:")
for p in sorted(Path(".").rglob("*")):
    if p.is_file() and not any(x in p.parts for x in (".ipynb_checkpoints", "__pycache__", ".venv")): print("  ", p)
print(\"\"\"
Run it yourself:
  python -m pip install -r requirements.txt
  python -m streamlit run app.py   # upload app (recommended)
  python run_pipeline.py           # same pipeline from the command line
  jupyter nbconvert --execute --to notebook --inplace notebook.ipynb
\"\"\")
""")
nbf.write(nb, "notebook.ipynb")
