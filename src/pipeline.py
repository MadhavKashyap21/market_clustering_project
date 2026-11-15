"""Stage functions shared by the notebook, the command line and the Streamlit app."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from . import advisor as adv
from . import clustering as cl
from . import features as ft
from . import preprocess as pp
from . import validation as va

ROOT = Path(__file__).resolve().parent.parent

LIMITATIONS = [
    "Reporting gaps and uneven coverage: markets that report rarely are dropped or interpolated, so the "
    "clusters describe well-reporting mandis and may miss remote or low-volume ones.",
    "Variety/grade mixing and arrival-quantity effects: prices are medians across varieties and grades, "
    "so quality or supply differences can masquerade as 'price behaviour'.",
    "Method assumptions: K-Means favours compact, similar-sized spherical groups and is sensitive to scaling "
    "and outliers; PCA keeps only part of the variance; silhouette/Davies-Bouldin are heuristics, not truth.",
    "Short window: about 12 months of one commodity, so seasonality, trend and harvest-season arrival surges "
    "cannot be separated from one-off events.",
]


def stage_clean(cfg: dict[str, Any], log: pp.PreprocessingLog, source=None) -> dict[str, Any]:
    raw, synthetic = pp.load_data(cfg, log, source)
    df, rel = pp.clean_raw(raw, cfg, log)
    return {"raw": raw, "df": df, "relevance": rel, "synthetic": synthetic}


def stage_features(df: pd.DataFrame, cfg: dict[str, Any], log: pp.PreprocessingLog,
                   winsorise: bool | None = None, drop_groups: tuple[str, ...] = ()) -> dict[str, Any]:
    panel = pp.build_weekly_panel(df, cfg, log, winsorise)
    feat = ft.compute_market_features(panel["weekly"], panel["rows"], panel["stats"]["coverage"])
    initial = [c for c in feat.columns if c != "coverage"
               and not any(c in ft.FEATURE_GROUPS[g] for g in drop_groups) and feat[c].nunique(dropna=True) > 1]
    log.add("Feature engineering", f"{len(initial)} candidate features (State and coverage excluded from clustering): "
            + "; ".join(f"{c} = {ft.FEATURE_DOC[c]}" for c in initial))
    if drop_groups:
        log.add("Feature engineering", f"Feature group(s) removed for this run: {', '.join(drop_groups)}")
    log.add("Encoding", "State is kept only as a post-hoc interpretation label; no one-hot encoding was needed "
                        "because every clustering feature is numeric.")
    kept, dropped, corr = ft.drop_correlated(feat[initial], cfg["preprocessing"]["corr_threshold"])
    for _, r in dropped.iterrows():
        log.add("Correlation pruning", f"Dropped '{r['dropped']}' (|r|={r['abs_r']} with '{r['kept_partner']}')")
    X, scaler, imputed = ft.prepare_matrix(feat, kept, log)
    return {**panel, "feat": feat, "initial": initial, "kept": kept, "corr_dropped": dropped, "corr": corr,
            "X": X, "scaler": scaler, "imputed": imputed}


def stage_grid(X: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    pca = PCA(random_state=cfg["random_state"]).fit(X.values)
    P = pca.transform(X.values)
    ks = range(cfg["k_min"], cfg["k_max"] + 1)
    if not 2 <= cfg["k_min"] <= cfg["k_max"] < len(X):
        raise ValueError(f"K range {cfg['k_min']}..{cfg['k_max']} is invalid for only {len(X)} markets; "
                         "lower K max or the coverage threshold")
    grid, fits = cl.run_grid(X.values, P[:, :2], ks, ["euclidean", "manhattan"], cfg["random_state"], cfg["n_init"])
    return {"grid": grid, "fits": fits, "pca": pca, "P": P}


def stage_interpret(X: pd.DataFrame, feats: dict[str, Any], fit: cl.FitResult) -> dict[str, Any]:
    labels = fit.labels
    cz = X.groupby(labels).mean()
    names = va.label_clusters(cz)
    orig = feats["imputed"].join(feats["feat"][["coverage"]])
    sizes = pd.Series(labels).value_counts().sort_index().rename("n_markets")
    states = pd.Series(X.index.str.split(" \\| ").str[0], index=X.index)
    return {"labels": labels, "centroids_z": cz, "names": names, "sizes": sizes,
            "mean": orig.groupby(labels).mean(), "median": orig.groupby(labels).median(),
            "state_ct": pd.crosstab(states.values, labels).T.rename_axis("cluster"), "states": states,
            "contrib": va.feature_contributions(X, labels, cz)}


def stage_stability(df: pd.DataFrame, cfg: dict[str, Any], base: dict[str, Any], dist: str, k: int,
                    ref: np.ndarray) -> dict[str, Any]:
    s, Xv = cfg["stability"], base["X"].values
    seeds = va.seed_stability(Xv, ref, k, dist, s["n_seeds"], s["seed_n_init"])
    boot = va.bootstrap_stability(Xv, ref, k, dist, s["n_bootstrap"], s["bootstrap_frac"], cfg["random_state"])
    other = "manhattan" if dist == "euclidean" else "euclidean"
    alt = cl.fit_model(Xv, k, other, cfg["random_state"], cfg["n_init"])
    dist_ari = adjusted_rand_score(ref, alt.labels)
    rows = []
    quiet = pp.PreprocessingLog(silent=True)
    for name, kw in {"No winsorising": dict(winsorise=False), "Without shape group": dict(drop_groups=("shape",)),
                     "Without dynamics group": dict(drop_groups=("dynamics",))}.items():
        sc = stage_features(df, cfg, quiet, **kw)
        r = cl.fit_model(sc["X"].values, k, dist, cfg["random_state"], cfg["n_init"])
        common = base["X"].index.intersection(sc["X"].index)
        a = pd.Series(ref, index=base["X"].index).loc[common].values
        b = pd.Series(r.labels, index=sc["X"].index).loc[common].values
        rows.append({"scenario": name, "n_features": sc["X"].shape[1], "ari_vs_final": adjusted_rand_score(a, b)})
    sens = pd.DataFrame(rows)
    comp = {"Random seeds": seeds["ari_vs_final"].mean(), "Bootstrap 80%": boot["ari_vs_final"].mean(),
            f"{dist.title()} vs {other.title()}": dist_ari, "Sensitivity scenarios": sens["ari_vs_final"].mean()}
    overall = float(np.mean(list(comp.values())))
    return {"seeds": seeds, "boot": boot, "dist_ari": dist_ari, "sens": sens, "components": comp,
            "overall": overall, "verdict": va.verdict(overall)}


def stage_advisor(c: dict, f: dict, it: dict, cfg: dict[str, Any]) -> dict[str, Any]:
    X = f["X"]
    coords = adv.market_coordinates(X.index, cfg, c["synthetic"], c["df"])
    snap = adv.build_snapshot(f["weekly"], f["feat"], it["labels"], it["names"], X.index)
    out = {"snap": snap, "coords": coords, "ranking": None, "clusters": None, "origin": None}
    if coords is None:
        return out
    snap = snap.join(coords)
    a = cfg["advisor"]
    o_lat, o_lon = a["farmer_lat"], a["farmer_lon"]
    fm = a.get("farmer_market")
    if fm and fm in snap.index and pd.notna(snap.loc[fm, "lat"]):
        o_lat, o_lon = float(snap.loc[fm, "lat"]), float(snap.loc[fm, "lon"])
    ranking, csum = adv.rank_markets(snap, o_lat, o_lon, a["quantity_quintal"], a["transport_rate_per_quintal_km"],
                                     a["road_factor"], a["market_fee_pct"], a["max_km"])
    out.update(snap=snap, ranking=ranking, clusters=csum, origin=fm or f"{o_lat:.3f},{o_lon:.3f}")
    return out


def auto_settings(df: pd.DataFrame, cfg: dict[str, Any], log: pp.PreprocessingLog) -> list[str]:
    """Adapt coverage, minimum rows and the K range to what this file can support (mutates cfg).

    Coverage is lowered step by step from 60% until at least 30 markets qualify; if the file never reaches 30, the lowest
    step is used. Minimum rows follow the coverage (about half the days a market must report, capped at 100). K max is
    about one cluster per 5 markets (2 to 10) and every cluster must hold at least 2 markets.
    """
    span = (df["Arrival_Date"].max() - df["Arrival_Date"].min()).days + 1
    g = df.groupby("market_id").agg(n_obs=("Modal_Price", "size"), days=("Arrival_Date", "nunique"))
    cov = g["days"] / span
    n_total, target = len(g), 30
    options = []
    for t in [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02]:
        mo = int(min(100, max(3, round(t * span * 0.5))))
        options.append((t, mo, int(((cov >= t) & (g["n_obs"] >= mo) & (g["days"] >= 4)).sum())))
    max_n = max(o[2] for o in options)
    # strictest threshold that reaches the target; if the file can never reach it, the strictest that keeps every usable market
    t, mo, n = next(o for o in options if o[2] >= min(target, max_n))
    if n < 6:
        raise ValueError(
            f"This file has {n_total} market(s) for the selected commodity and only {n} of them have enough reports "
            f"to compare. Clustering compares markets, so it needs at least 6 markets with several weeks of prices each. "
            f"Upload a state-wide or country-wide export with many markets (a file for a single market cannot be clustered).")
    k_max = int(min(10, max(3, n // 5)))
    cfg["preprocessing"].update(min_coverage=t, min_obs=mo)
    cfg.update(k_min=2, k_max=k_max, min_k_for_selection=min(3, k_max) if n >= 12 else 2, min_cluster_frac=max(0.05, 2 / n))
    reasons = [
        f"File has {n_total} markets over {span} days.",
        f"Coverage threshold set to {t:.0%} and minimum rows to {mo}, which keeps {n} markets"
        + ("" if n >= target else f" (fewer than the {target} preferred, so this file has few usable markets: treat results with extra caution)") + ".",
        f"K tested: 2 to {k_max}; every cluster must contain at least {max(2, int(np.ceil(0.05 * n)))} markets.",
    ]
    for line in reasons:
        log.add("Auto-adjust", line)
    return reasons


def analyze(cfg: dict[str, Any], log: pp.PreprocessingLog, source=None, progress=None) -> dict[str, Any]:
    """Run every objective on the given data (bytes / path / DataFrame / None) and return all results in memory."""
    step = progress or (lambda msg, frac: None)
    step("Objective 1: loading and cleaning data", 0.03)
    c = stage_clean(cfg, log, source)
    auto = auto_settings(c["df"], cfg, log) if cfg.get("auto_adjust") else None
    step("Objective 1: engineering market features", 0.12)
    f = stage_features(c["df"], cfg, log)
    step("Objective 2: fitting K-Means and K-Medians for every K (slowest step)", 0.2)
    g = stage_grid(f["X"], cfg)
    ranked, best, runner = va.select_configuration(g["grid"], cfg)
    dist, k = best["distance"], int(best["k"])
    fit = g["fits"][(dist, k)]
    step("Objective 3: profiling clusters", 0.5)
    it = stage_interpret(f["X"], f, fit)
    Z, hl, coph = cl.ward_hierarchy(f["X"].values, k)
    hier = {"Z": Z, "labels": hl, "cophenetic": coph, "ari": adjusted_rand_score(fit.labels, hl),
            "crosstab": pd.crosstab(pd.Series(fit.labels, name="Partition"), pd.Series(hl, name="Ward"))}
    step("Objective 3: stability analysis (seeds, bootstrap, sensitivity)", 0.55)
    st = stage_stability(c["df"], cfg, f, dist, k, fit.labels)
    step("Farmer advisor", 0.93)
    advisor = stage_advisor(c, f, it, cfg)
    res = {"cfg": cfg, "log": log, "c": c, "f": f, "g": g, "ranked": ranked, "best": best, "runner": runner,
           "dist": dist, "k": k, "fit": fit, "it": it, "hier": hier, "st": st, "advisor": advisor, "auto": auto}
    res["summary"] = build_summary(res)
    log.add("Model selection", f"Chosen: {dist} K={k} (composite={best['composite']:.3f}); runner-up {res['summary']['runner_up']}")
    step("Done", 1.0)
    return res


def build_summary(res: dict[str, Any]) -> dict[str, Any]:
    cfg, c, f, it, st, best, runner = res["cfg"], res["c"], res["f"], res["it"], res["st"], res["best"], res["runner"]
    pca = res["g"]["pca"]
    adv_demo = None
    if res["advisor"]["ranking"] is not None and len(res["advisor"]["ranking"]):
        rk = res["advisor"]["ranking"]
        adv_demo = {"best_net_market": rk.index[0], "best_listed_market": rk["recent_price"].idxmax(),
                    "markets_with_coordinates": int(res["advisor"]["snap"]["lat"].notna().sum()),
                    "origin": res["advisor"]["origin"]}
    return {
        "data": "SYNTHETIC (demonstration only)" if c["synthetic"] else "real Agmarknet data",
        "commodity": cfg["commodity"], "n_rows_clean": int(len(c["df"])), "n_markets_before_filter": int(len(f["stats"])),
        "n_markets": int(len(f["X"])), "n_features": int(f["X"].shape[1]),
        "chosen_distance": res["dist"], "chosen_k": res["k"],
        "silhouette": round(float(best["silhouette"]), 4), "davies_bouldin": round(float(best["davies_bouldin"]), 4),
        "cost": round(float(best["cost"]), 3), "runner_up": f"{runner['distance']} K={int(runner['k'])}",
        "cluster_sizes": {int(a): int(b) for a, b in it["sizes"].items()},
        "cluster_names": {int(a): b for a, b in it["names"].items()},
        "ward_vs_kmeans_ARI": round(float(res["hier"]["ari"]), 4), "cophenetic_corr": round(res["hier"]["cophenetic"], 4),
        "stability_components": {a: round(float(b), 4) for a, b in st["components"].items()},
        "stability_overall_ARI": round(st["overall"], 4), "stability_verdict": st["verdict"],
        "top3_features": list(it["contrib"].index[:3]), "limitations": LIMITATIONS[:3],
        "pca_var_first2": round(float(pca.explained_variance_ratio_[:2].sum()), 4), "advisor_demo": adv_demo,
    }


def make_figures(res: dict[str, Any]) -> dict[str, Any]:
    """Build every figure (matplotlib Figure objects, not yet saved). Keys double as file names."""
    from . import visualise as vz
    cfg, f, g, it, X, k = res["cfg"], res["f"], res["g"], res["it"], res["f"]["X"], res["k"]
    c, pca, P, grid = res["c"], g["pca"], g["P"], g["grid"]
    panels = []
    for d in ["euclidean", "manhattan"]:
        for _, r in res["ranked"][res["ranked"]["distance"] == d].head(2).iterrows():
            m = g["fits"][(d, int(r["k"]))]
            panels.append((f"{d.title()} K={int(r['k'])}", m.labels, pca.transform(m.centers)[:, :2]))
    figs = {
        "01_state_counts": vz.plot_state_counts(c["df"]),
        "02_price_hist": vz.plot_price_hist(c["df"]),
        "03_reports_per_month": vz.plot_reports_per_month(c["df"]),
        "04_cleaning": vz.plot_cleaning(res["log"].stats),
        "05_coverage": vz.plot_coverage(f["stats"], cfg["preprocessing"]["min_coverage"]),
        "06_missingness": vz.plot_missingness(c["relevance"]),
        "07_feature_distributions": vz.plot_feature_distributions(f["feat"], f["initial"]),
        "08_correlation": vz.plot_correlation(f["corr"].loc[f["initial"], f["initial"]], cfg["preprocessing"]["corr_threshold"]),
        "09_elbow": vz.plot_elbow(grid, k),
        "10_silhouette": vz.plot_metric(grid, "silhouette", "Silhouette score (higher is better)", k),
        "11_davies_bouldin": vz.plot_metric(grid, "davies_bouldin", "Davies-Bouldin index (lower is better)", k),
        "12_metric_curves_all": vz.plot_metric_curves(grid, k),
        "13_selection_score": vz.plot_composite(res["ranked"]),
        "14_pca_scree": vz.plot_scree(pca.explained_variance_ratio_),
        "15_pca_loadings": vz.plot_loadings(pca, list(X.columns))[0],
        "16_pca_clusters": vz.plot_pca_scatter(P, X.index, pca.explained_variance_ratio_, panels),
        "17_dendrogram": vz.plot_dendrogram(res["hier"]["Z"], k),
        "18_ward_crosstab": vz.plot_crosstab(res["hier"]["crosstab"]),
        "19_cluster_sizes": vz.plot_cluster_sizes(it["sizes"]),
        "20_cluster_radar": vz.plot_radar(it["centroids_z"]),
        "21_centroid_heatmap": vz.plot_centroid_heatmap(it["centroids_z"], it["names"]),
        "22_price_trends": vz.plot_trends(f["weekly"], it["labels"], X.index),
        "23_state_composition": vz.plot_state_composition(it["state_ct"]),
        "24_feature_drivers": vz.plot_contrib(it["contrib"]),
        "25_stability_components": vz.plot_components(res["st"]["components"]),
        "26_stability": vz.plot_stability(res["st"]["seeds"], res["st"]["boot"], res["st"]["sens"]),
    }
    rk = res["advisor"]["ranking"]
    if rk is not None and len(rk):
        figs["27_net_prices"] = vz.plot_net_prices(rk)
    return figs


def save_outputs(res: dict[str, Any], out: str | Path, figs: dict[str, Any] | None = None) -> None:
    """Write every table, figure (300 dpi), the preprocessing log and results_summary.json."""
    import matplotlib.pyplot as plt
    out = Path(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    f, g, it, st, c, X = res["f"], res["g"], res["it"], res["st"], res["c"], res["f"]["X"]
    g["grid"].to_csv(out / "metrics_grid.csv", index=False)
    res["ranked"].to_csv(out / "selection_ranking.csv", index=False)
    c["relevance"].to_csv(out / "variable_relevance.csv", index=False)
    f["feat"].to_csv(out / "market_features.csv")
    f["corr_dropped"].to_csv(out / "correlation_pruning.csv", index=False)
    f["dropped"].to_csv(out / "dropped_markets.csv")
    it["mean"].to_csv(out / "cluster_profile_mean.csv")
    it["median"].to_csv(out / "cluster_profile_median.csv")
    it["contrib"].to_csv(out / "feature_contributions.csv")
    st["seeds"].to_csv(out / "stability_seeds.csv", index=False)
    st["boot"].to_csv(out / "stability_bootstrap.csv", index=False)
    st["sens"].to_csv(out / "stability_sensitivity.csv", index=False)
    res["hier"]["crosstab"].to_csv(out / "ward_vs_partition_crosstab.csv")
    assign = pd.DataFrame({"market_id": X.index, "State": it["states"].values, "cluster": it["labels"],
                           "cluster_name": [it["names"][x] for x in it["labels"]],
                           "ward_cluster": res["hier"]["labels"], "PC1": g["P"][:, 0], "PC2": g["P"][:, 1]}
                          ).join(f["imputed"], on="market_id")
    assign.to_csv(out / "cluster_assignments.csv", index=False)
    a = res["advisor"]
    a["snap"].to_csv(out / "market_snapshot.csv")
    if a["coords"] is None:
        a["snap"][["State"]].assign(lat=np.nan, lon=np.nan).rename_axis("market_id").to_csv(out / "mandi_coordinates_template.csv")
    else:
        a["coords"].to_csv(out / "market_coordinates_used.csv")
        if a["ranking"] is not None:
            a["ranking"].to_csv(out / "advisor_demo_ranking.csv")
            a["clusters"].to_csv(out / "advisor_demo_clusters.csv", index=False)
    figs = figs if figs is not None else make_figures(res)
    for name, fig in figs.items():
        fig.savefig(out / "figures" / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    (out / "results_summary.json").write_text(json.dumps(res["summary"], indent=2), encoding="utf-8")
    res["log"].save(out / "preprocessing_log.md")


def run_all(config_path: str = "config.yaml") -> dict[str, Any]:
    """Command-line entry: analyse the file in config.yaml (or the synthetic fallback) and write outputs/."""
    matplotlib.use("Agg")
    cfg = pp.load_config(config_path)
    log = pp.PreprocessingLog()
    res = analyze(cfg, log)
    save_outputs(res, cfg["output_dir"])
    print(json.dumps(res["summary"], indent=2))
    return res["summary"]
