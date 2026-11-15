"""Upload an Agmarknet price CSV and get every objective and project output.

Run:  python -m streamlit run app.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

import altair as alt
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

if get_script_run_ctx() is None:   # started as a plain script (e.g. Code Runner / "python app.py")
    print("\nThis is a Streamlit app, so it cannot run as a plain script.\n"
          "Open a terminal in this folder and run:\n\n    python -m streamlit run app.py\n")
    sys.exit(0)

matplotlib.use("Agg")
from src import features as ft  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import preprocess as pp  # noqa: E402
from src import visualise as vz  # noqa: E402
from src.advisor import rank_markets  # noqa: E402

ROOT = Path(__file__).resolve().parent
EXAMPLE = ROOT / "data" / "agmarknet_wheat.csv.gz"
REQUIRED = {"State", "Market", "Arrival_Date", "Modal_Price"}
PAL = vz.PAL

st.set_page_config(page_title="Market Price Behaviour Segmentation", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&family=Lora:wght@500;600&display=swap');
html, body, .stApp, [class*="st-"] { font-family: 'Nunito', 'Segoe UI', sans-serif; }
h1, h2, h3, .sec, .hero h1 { font-family: 'Lora', Georgia, serif !important; color: #3B4252; letter-spacing: 0; }
.stApp { background: #FAF7F2; }
section[data-testid="stSidebar"] { background: #F1EBF3; border-right: 1px solid #E5DCE8; }
div[data-baseweb="tab-list"] { gap: 6px; flex-wrap: wrap; border-bottom: none; }
button[data-baseweb="tab"] { background: #F2ECE3; border-radius: 999px; padding: 6px 16px; color: #5B6475; height: auto; }
button[data-baseweb="tab"][aria-selected="true"] { background: #CFE0F1; color: #2F3B52; font-weight: 700; }
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none; }
.hero { display: flex; align-items: center; justify-content: space-between; gap: 24px; background: #F6EEE2;
        border-radius: 22px; padding: 22px 30px; margin-bottom: 18px; }
.hero h1 { font-size: 2.1rem; margin: 0 0 6px 0; padding: 0; line-height: 1.2; }
.hero p { margin: 0; color: #5B6475; max-width: 640px; }
.hero svg { flex: 0 0 auto; max-width: 340px; width: 100%; height: auto; }
.sec { margin: 26px 0 4px 0; font-size: 1.35rem; }
.sub { color: #7A8090; margin: 0 0 10px 0; }
.note { background: #EDF4EE; border-left: 4px solid #9FCDB4; border-radius: 10px; padding: 11px 16px; margin: 6px 0 18px 0; color: #3B4252; }
.note.warm { background: #FBF0E4; border-left-color: #F0C29B; }
.note.rose { background: #FBEDEC; border-left-color: #EBAAA6; }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 6px 0 18px 0; }
.stat { border-radius: 18px; padding: 16px 20px; }
.stat .v { font-family: 'Lora', Georgia, serif; font-size: 1.6rem; color: #2F3B52; line-height: 1.2; }
.stat .l { font-weight: 700; color: #3B4252; margin-top: 4px; }
.stat .h { color: #6B7283; font-size: 0.85rem; }
[data-testid="stImage"] img { border-radius: 14px; box-shadow: 0 2px 12px rgba(90, 75, 60, 0.10); background: #fff; }
@media (max-width: 900px) { .stats { grid-template-columns: repeat(2, 1fr); } .hero { flex-direction: column; } }
</style>
""", unsafe_allow_html=True)

HERO_SVG = """<svg viewBox="0 0 360 170" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Wheat field and market stall">
<circle cx="296" cy="40" r="22" fill="#F6D9A8"/>
<path d="M0 128 Q90 84 180 118 T360 104 V170 H0Z" fill="#D5E6D8"/>
<path d="M0 150 Q110 112 220 142 T360 134 V170 H0Z" fill="#BCD9C9"/>
<g stroke="#D9B96A" stroke-width="3" stroke-linecap="round">
<line x1="34" y1="156" x2="34" y2="104"/><line x1="56" y1="158" x2="56" y2="96"/><line x1="78" y1="156" x2="78" y2="108"/><line x1="100" y1="158" x2="100" y2="100"/></g>
<g fill="#EACB84"><ellipse cx="34" cy="98" rx="5" ry="12"/><ellipse cx="56" cy="90" rx="5" ry="12"/><ellipse cx="78" cy="102" rx="5" ry="12"/><ellipse cx="100" cy="94" rx="5" ry="12"/></g>
<rect x="196" y="98" width="110" height="50" rx="4" fill="#F1E3CF"/>
<path d="M188 98 L314 98 L304 70 L198 70 Z" fill="#FFFFFF"/>
<path d="M198 70 L216 70 L212 98 L188 98 Z M234 70 L252 70 L250 98 L228 98 Z M270 70 L288 70 L290 98 L268 98 Z" fill="#F3B9B4"/>
<rect x="206" y="116" width="26" height="20" rx="3" fill="#B4A7E0"/><rect x="240" y="116" width="26" height="20" rx="3" fill="#F3CF7A"/><rect x="274" y="116" width="22" height="20" rx="3" fill="#86D3B3"/>
</svg>"""


def head(title, sub=""):
    st.markdown(f"<h3 class='sec'>{title}</h3>" + (f"<p class='sub'>{sub}</p>" if sub else ""), unsafe_allow_html=True)


def note(text, tone=""):
    st.markdown(f"<div class='note {tone}'>{text}</div>", unsafe_allow_html=True)


def stat_cards(items):
    cards = "".join(f"<div class='stat' style='background:{c}'><div class='v'>{v}</div><div class='l'>{l}</div>"
                    f"<div class='h'>{h}</div></div>" for v, l, h, c in items)
    st.markdown(f"<div class='stats'>{cards}</div>", unsafe_allow_html=True)


def cluster_scale():
    return alt.Scale(range=PAL)


st.markdown(f"""<div class="hero"><div><h1>Regional Market Price Behaviour Segmentation</h1>
<p>Upload an Agmarknet daily price file. The app cleans it, groups markets that behave alike, checks how far the groups
can be trusted and shows what a farmer would keep after transport. Groups are exploratory, not fixed market categories.</p></div>
{HERO_SVG}</div>""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Reading file...")
def read_upload(data: bytes):
    return pp.read_agmarknet(data)


@st.cache_data(show_spinner=False)
def file_profile(data: bytes, commodity: str):
    raw_, _ = read_upload(data)
    d = raw_
    if commodity:
        d = raw_[raw_["Commodity"].astype(str).str.lower().str.contains(commodity.lower(), na=False, regex=False)]
    dates = pd.to_datetime(d["Arrival_Date"], dayfirst=True, errors="coerce")
    n_markets = (d["State"].astype(str).str.strip() + " | " + d["Market"].astype(str).str.strip()).nunique()
    return {"rows": len(d), "markets": int(n_markets), "first": dates.min(), "last": dates.max()}


# ------------------------------------------------------------------ sidebar: data + settings
with st.sidebar:
    st.header("1. Data")
    up = st.file_uploader("Agmarknet CSV (.csv or .csv.gz)", type=["csv", "gz"])
    use_example = False
    if up is None and EXAMPLE.exists():
        use_example = st.checkbox("Use the bundled wheat example file", value=False)
    data = up.getvalue() if up is not None else (EXAMPLE.read_bytes() if use_example else None)

if data is None:
    note("Upload a CSV in the sidebar to start. Expected columns (names are matched automatically): State, District, "
         "Market, Commodity, Variety, Grade, Min/Max/Modal Price, Arrival Quantity, Arrival Date. "
         "A title row above the header and Indian number grouping such as 3,00,02,800.00 are handled.", "warm")
    st.stop()

try:
    raw, mapping = read_upload(data)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not read this file as CSV: {exc}")
    st.stop()
missing = REQUIRED - set(raw.columns)
if missing:
    st.error(f"Missing required columns: {sorted(missing)}. Columns found: {list(raw.columns)}")
    st.stop()

with st.sidebar:
    if "Commodity" in raw.columns and raw["Commodity"].notna().any():
        counts = raw["Commodity"].value_counts()
        commodity = st.selectbox("Commodity to analyse", list(counts.index),
                                 format_func=lambda c: f"{c} ({counts[c]:,} rows)")
    else:
        commodity = ""
    prof = file_profile(data, commodity)
    st.caption(f"{prof['rows']:,} rows | {prof['markets']} market(s) | "
               f"{prof['first']:%d %b %Y} to {prof['last']:%d %b %Y}" if prof["rows"] else "No rows for this commodity")
    if prof["markets"] < 6:
        st.warning(f"This commodity has only {prof['markets']} market(s). The project clusters markets, so it needs at least 6 "
                   "with several weeks of prices. Upload a state-wide or country-wide export instead.")
    st.header("2. Settings")
    auto = st.checkbox("Auto-adjust settings to this file (recommended)", True,
                       help="Chooses the coverage threshold, minimum rows and K range from what the file can support.")
    min_cov, (k_lo, k_hi), min_obs, min_frac = 0.6, (2, 10), 100, 0.05
    if not auto:
        min_cov = st.slider("Min reporting coverage (share of days)", 0.02, 0.9, 0.6, 0.02)
        k_lo, k_hi = st.slider("Cluster counts K to test", 2, 12, (2, 10))
        min_obs = st.number_input("Min rows per market", 1, 1000, 100)
        min_frac = st.slider("Smallest allowed cluster (% of markets)", 0, 30, 5) / 100
    else:
        st.caption("Coverage, minimum rows and K range are chosen automatically and reported in the Overview tab.")
    with st.expander("Advanced"):
        winsor = st.checkbox("Winsorise price outliers", True)
        corr_thr = st.slider("Correlation pruning threshold", 0.7, 0.99, 0.9, 0.01)
        n_seeds = st.slider("Stability: random seeds", 5, 30, 20)
        n_boot = st.slider("Stability: bootstrap samples", 10, 100, 50)
        qty = st.number_input("Advisor: quantity (quintals)", 1.0, 10000.0, 50.0)
        rate = st.number_input("Advisor: transport Rs per quintal-km (assumption)", 0.0, 5.0, 0.20)
        fee = st.number_input("Advisor: mandi fee % (assumption)", 0.0, 10.0, 1.0)
    run = st.button("Run analysis", type="primary")

settings = dict(auto=auto, commodity=commodity, min_cov=min_cov, k=(k_lo, k_hi), min_obs=min_obs, winsor=winsor, corr=corr_thr,
                min_frac=min_frac, seeds=n_seeds, boot=n_boot, qty=qty, rate=rate, fee=fee)
key = (hash(data), str(settings))
if run:
    cfg = pp.load_config(ROOT / "config.yaml")
    cfg.update(commodity=commodity or "", k_min=k_lo, k_max=k_hi, min_cluster_frac=min_frac, auto_adjust=auto)
    cfg["preprocessing"].update(min_coverage=min_cov, min_obs=int(min_obs), winsorise=winsor, corr_threshold=corr_thr)
    cfg["stability"].update(n_seeds=n_seeds, n_bootstrap=n_boot)
    cfg["advisor"].update(quantity_quintal=qty, transport_rate_per_quintal_km=rate, market_fee_pct=fee)
    bar = st.progress(0.0, text="Starting...")
    try:
        res = pl.analyze(cfg, pp.PreprocessingLog(silent=True), source=raw, progress=lambda m, f: bar.progress(f, text=m))
        bar.progress(0.97, text="Drawing charts...")
        st.session_state.update(res=res, figs=pl.make_figures(res), key=key)
        st.session_state.pop("zip", None)
        bar.empty()
    except Exception as exc:  # noqa: BLE001
        bar.empty()
        st.error(f"Analysis stopped: {exc}")
        st.stop()

if "res" not in st.session_state:
    note("Pick the commodity and settings, then press <b>Run analysis</b> in the sidebar. Every step gets its own tab.", "warm")
    st.stop()
if st.session_state.get("key") != key:
    note("The file or settings changed since the last run. Press <b>Run analysis</b> to refresh. Showing the previous results.", "rose")

res, figs = st.session_state["res"], st.session_state["figs"]
S, f, g, it, hier, stb, c = res["summary"], res["f"], res["g"], res["it"], res["hier"], res["st"], res["c"]
X, labels = f["X"], it["labels"]
cl_str = pd.Series([f"C{x}" for x in labels], index=X.index)
if S["data"].startswith("SYNTHETIC"):
    note("Synthetic data in use: demonstration only.", "rose")

TAB_NAMES = ["Overview", "Data", "Cleaning", "Features", "Correlation", "Choosing K", "PCA", "Hierarchy", "Profiles",
             "States and map", "Drivers", "Stability", "Farmer advisor", "Download"]
tabs = dict(zip(TAB_NAMES, st.tabs(TAB_NAMES)))

# ------------------------------------------------------------------ overview
with tabs["Overview"]:
    stat_cards([
        (f"{S['chosen_distance'].title()} K={S['chosen_k']}", "Chosen grouping", f"runner-up {S['runner_up']}", "#DCEBF5"),
        (f"{S['silhouette']:.2f}", "Silhouette", f"Davies-Bouldin {S['davies_bouldin']:.2f}", "#FBE3DF"),
        (f"{S['n_markets']} / {S['n_markets_before_filter']}", "Markets analysed", "met the reporting rule", "#E1F1E8"),
        (S["stability_verdict"], "Stability", f"mean ARI {S['stability_overall_ARI']:.2f}", "#EAE5F6"),
    ])
    if res.get("auto"):
        note("<b>Settings chosen for this file.</b> " + " ".join(res["auto"]), "warm")
    head("What the analysis found")
    weak = S["silhouette"] < 0.25
    note(f"{S['commodity'] or 'All rows'}: {S['n_rows_clean']:,} cleaned rows. {S['n_markets']} of {S['n_markets_before_filter']} markets "
         f"reported often enough, and {S['n_features']} features remained after removing near-duplicates. "
         f"The selected grouping is <b>{S['chosen_distance']} with K={S['chosen_k']}</b> (clusters of {', '.join(str(v) for v in S['cluster_sizes'].values())} markets). "
         f"Separation is {'weak, so read the groups as a description only' if weak else 'moderate to clear'}: silhouette {S['silhouette']:.2f}. "
         f"Across seeds, resamples and preprocessing choices the grouping is <b>{S['stability_verdict']}</b>. "
         f"Features that separate the groups most: {', '.join(ft.PRETTY.get(x, x) for x in S['top3_features'])}.")
    l, r = st.columns([1, 1.4])
    with l:
        st.pyplot(figs["19_cluster_sizes"])
    with r:
        head("Provisional group labels", "Written automatically from each group's most unusual features.")
        st.dataframe(pd.DataFrame({"label": it["names"], "markets": it["sizes"]}))
    head("Limitations")
    for line in pl.LIMITATIONS:
        st.markdown(f"- {line}")
    note("<b>Responsible use.</b> These groups are exploratory, data-driven and tied to one commodity and one time window. They are not "
         "permanent market categories and should not drive pricing, policy targeting or labelling of regions without local validation.", "rose")

# ------------------------------------------------------------------ data
with tabs["Data"]:
    head("The file at a glance")
    note("Before any cleaning: how many markets there are, when they reported and what the prices look like. "
         "This is the raw material every later step works from.")
    a, b = st.columns(2)
    with a:
        st.pyplot(figs["01_state_counts"])
    with b:
        st.pyplot(figs["02_price_hist"])
    st.pyplot(figs["03_reports_per_month"])
    head("Which columns are used", "Every column gets a keep or drop decision with a reason.")
    st.dataframe(c["relevance"])
    with st.expander("Preview of the uploaded rows"):
        st.dataframe(raw.head(30))

# ------------------------------------------------------------------ cleaning
with tabs["Cleaning"]:
    head("Cleaning and filtering")
    note("Impossible values are repaired or blanked, never silently deleted. Markets that report too rarely are removed, because "
         "their price behaviour cannot be measured reliably.")
    a, b = st.columns(2)
    with a:
        st.pyplot(figs["04_cleaning"])
    with b:
        st.pyplot(figs["05_coverage"])
    st.pyplot(figs["06_missingness"])
    head("Every preprocessing decision")
    with st.expander("Read the full log", expanded=False):
        st.markdown(res["log"].to_markdown("Preprocessing log").replace("# Preprocessing log", ""))
    head("Markets removed for low reporting")
    st.caption(f"{len(f['dropped'])} markets dropped")
    st.dataframe(f["dropped"])

# ------------------------------------------------------------------ features
with tabs["Features"]:
    head("What describes a market")
    note("Each market's price history becomes one row of numbers: how high prices are, how much they swing, how seasonal and how "
         "steady the trend is. Clustering compares these rows, not the raw price lists.")
    st.pyplot(figs["07_feature_distributions"])
    head("Feature definitions")
    st.dataframe(pd.DataFrame({"feature": f["initial"], "meaning": [ft.FEATURE_DOC[x] for x in f["initial"]]}))
    head("Feature table", "One row per market, original units.")
    st.dataframe(f["feat"].round(3))

# ------------------------------------------------------------------ correlation
with tabs["Correlation"]:
    head("Correlation pruning and scaling")
    note("Features that say almost the same thing would count twice in the distance calculation, so one of each near-duplicate pair is "
         "dropped. The rest are standardised so that no feature wins just because its numbers are larger.")
    st.pyplot(figs["08_correlation"])
    head("Features removed")
    st.dataframe(f["corr_dropped"])
    head("Standardised matrix used for clustering")
    st.dataframe(X.round(3))

# ------------------------------------------------------------------ choosing K
with tabs["Choosing K"]:
    head("How many groups, and which distance")
    note("Every cluster count is tested with two ways of measuring distance. Lower elbow cost, higher silhouette and lower "
         "Davies-Bouldin all mean tighter, better-separated groups.")
    st.pyplot(figs["09_elbow"])
    a, b = st.columns(2)
    with a:
        st.pyplot(figs["10_silhouette"])
    with b:
        st.pyplot(figs["11_davies_bouldin"])
    head("The decision rule", "Eligible: K of at least 3 (or 2 for tiny files) and no tiny clusters. Score = average of normalised silhouette, "
         "inverse Davies-Bouldin and 2-D PCA silhouette. Euclidean is preferred unless Manhattan wins clearly.")
    a, b = st.columns([1, 1.2])
    with a:
        st.pyplot(figs["13_selection_score"])
    with b:
        st.dataframe(res["ranked"].head(10).round(3))
    note(f"<b>Chosen: {S['chosen_distance']} K={S['chosen_k']}.</b> Runner-up: {S['runner_up']}.")
    with st.expander("All metrics for every distance and K"):
        st.dataframe(g["grid"].round(4))

# ------------------------------------------------------------------ PCA
with tabs["PCA"]:
    head("Seeing the groups in two dimensions")
    note("PCA squeezes all features into two axes so the groups can be drawn. The two axes keep "
         f"<b>{S['pca_var_first2']:.0%}</b> of the information, so overlaps in the picture can be exaggerated. Hover over a point to see the market.")
    options = {f"{d.title()} K={int(k)}": (d, int(k)) for d, k in zip(res["ranked"]["distance"].head(6), res["ranked"]["k"].head(6))}
    choice = st.selectbox("Grouping to draw", list(options))
    labs = g["fits"][options[choice]].labels
    pdf = pd.DataFrame({"PC1": g["P"][:, 0], "PC2": g["P"][:, 1], "market": [m.split(" | ")[-1] for m in X.index],
                        "state": [m.split(" | ")[0] for m in X.index], "cluster": [f"C{x}" for x in labs]})
    ev = g["pca"].explained_variance_ratio_
    sc = alt.Chart(pdf).mark_circle(size=90, opacity=0.85, stroke="white", strokeWidth=0.8).encode(
        x=alt.X("PC1:Q", title=f"PC1 ({ev[0]:.0%})"), y=alt.Y("PC2:Q", title=f"PC2 ({ev[1]:.0%})"),
        color=alt.Color("cluster:N", scale=cluster_scale(), legend=alt.Legend(title="Group")),
        tooltip=["market", "state", "cluster"]).properties(height=430, width="container").interactive()
    st.altair_chart(sc)
    a, b = st.columns(2)
    with a:
        st.pyplot(figs["14_pca_scree"])
    with b:
        st.pyplot(figs["15_pca_loadings"])
    with st.expander("Static comparison of the best configurations (also saved in the download)"):
        st.pyplot(figs["16_pca_clusters"])

# ------------------------------------------------------------------ hierarchy
with tabs["Hierarchy"]:
    head("A second opinion: hierarchical clustering")
    note("Ward linkage builds the groups bottom-up instead of by repeated averaging. If it largely agrees with the chosen grouping, "
         f"the structure is not an accident of one method. Agreement here: <b>ARI {hier['ari']:.2f}</b> (1 = identical). "
         f"Cophenetic correlation <b>{hier['cophenetic']:.2f}</b> says how faithfully the tree keeps the original distances.")
    st.pyplot(figs["17_dendrogram"])
    a, b = st.columns([1, 1.5])
    with a:
        st.pyplot(figs["18_ward_crosstab"])
    with b:
        st.dataframe(hier["crosstab"])

# ------------------------------------------------------------------ profiles
with tabs["Profiles"]:
    head("What each group looks like")
    note("Labels are generated from the features where a group stands furthest from average. They describe the data, "
         "they are not official categories.")
    a, b = st.columns(2)
    with a:
        st.pyplot(figs["20_cluster_radar"])
    with b:
        st.pyplot(figs["22_price_trends"])
    st.pyplot(figs["21_centroid_heatmap"])
    head("Explore one feature at a time")
    feat_choice = st.selectbox("Feature", list(f["imputed"].columns), format_func=lambda x: ft.PRETTY.get(x, x))
    bx = f["imputed"][[feat_choice]].assign(group=cl_str)
    box = alt.Chart(bx).mark_boxplot(extent="min-max", size=46).encode(
        x=alt.X("group:N", title=""), y=alt.Y(f"{feat_choice}:Q", title=ft.PRETTY.get(feat_choice, feat_choice)),
        color=alt.Color("group:N", scale=cluster_scale(), legend=None)).properties(height=320, width="container")
    st.altair_chart(box)
    a, b = st.columns(2)
    with a:
        head("Average of each feature")
        st.dataframe(it["mean"].round(3).T)
    with b:
        head("Median of each feature")
        st.dataframe(it["median"].round(3).T)

# ------------------------------------------------------------------ states and map
with tabs["States and map"]:
    head("Where the groups are")
    note("If a group is mostly one state, regional conditions are probably part of the explanation. Map positions come from matching "
         "market and district names to place names, so they are approximate.")
    snap = res["advisor"]["snap"]
    if "lat" in snap and snap["lat"].notna().any():
        mp = snap.dropna(subset=["lat", "lon"]).copy()
        mp["color"] = [PAL[int(x) % 10] for x in mp["cluster"]]
        st.map(mp, latitude="lat", longitude="lon", color="color", size=12000)
        st.caption("Colours match the group colours in the charts: " + ", ".join(f"C{c_}" for c_ in sorted(mp["cluster"].unique())))
    else:
        note("No market positions are available for this file. Add data/mandi_coordinates.csv (market_id, lat, lon) to see the map.", "warm")
    st.pyplot(figs["23_state_composition"])
    st.dataframe(it["state_ct"])
    head("Assignments")
    st.dataframe(pd.DataFrame({"group": cl_str, "label": [it["names"][x] for x in labels], "state": it["states"].values}))

# ------------------------------------------------------------------ drivers
with tabs["Drivers"]:
    head("What separates the groups")
    note("Eta squared is the share of a feature's variation explained by group membership. Close to 1 means the feature almost "
         "defines the groups; close to 0 means it barely matters.")
    a, b = st.columns([1.2, 1])
    with a:
        st.pyplot(figs["24_feature_drivers"])
    with b:
        st.dataframe(it["contrib"].round(3))

# ------------------------------------------------------------------ stability
with tabs["Stability"]:
    head("Can the groups be trusted")
    note(f"The grouping is re-computed with different random starts, resampled markets, the other distance and altered preprocessing. "
         f"The Adjusted Rand Index measures how much each new grouping agrees with the final one. Overall: "
         f"<b>{S['stability_verdict']}</b> ({S['stability_overall_ARI']:.2f}). Above 0.8 is stable, 0.5 to 0.8 moderate, below 0.5 unstable.")
    a, b = st.columns(2)
    with a:
        st.pyplot(figs["25_stability_components"])
    with b:
        st.pyplot(figs["26_stability"])
    st.dataframe(stb["sens"].round(3))

# ------------------------------------------------------------------ advisor
with tabs["Farmer advisor"]:
    head("Where would a farmer keep the most")
    adv_res = res["advisor"]
    snap = adv_res["snap"]
    if "lat" not in snap or snap["lat"].isna().all():
        note("No market positions are available, so transport costs cannot be worked out. Price behaviour by group instead:", "warm")
        st.dataframe(snap.groupby("cluster_name").agg(markets=("State", "size"), median_recent_price=("recent_price", "median"),
                     avg_momentum_pct=("momentum_pct", "mean"), avg_volatility=("volatility", "mean")).round(3))
    else:
        note("Price kept per quintal = listed price x (1 - fee) - transport rate x road km. Rate and fee are assumptions you can change in "
             "the sidebar. Positions are approximate, and the price is the median of the last four weeks.")
        xy = snap.dropna(subset=["lat", "lon"])
        origin = st.selectbox("Farmer's location", ["Custom latitude/longitude"] + sorted(xy.index))
        if origin == "Custom latitude/longitude":
            c1, c2 = st.columns(2)
            lat = c1.number_input("Latitude", value=float(xy["lat"].median()), format="%.4f")
            lon = c2.number_input("Longitude", value=float(xy["lon"].median()), format="%.4f")
        else:
            lat, lon = float(xy.loc[origin, "lat"]), float(xy.loc[origin, "lon"])
        km = st.number_input("Max road km", 10.0, 3000.0, 600.0)
        adv_cfg = res["cfg"]["advisor"]
        rank, csum = rank_markets(snap, lat, lon, adv_cfg["quantity_quintal"], adv_cfg["transport_rate_per_quintal_km"], 1.3,
                                  adv_cfg["market_fee_pct"], km)
        if rank.empty:
            st.warning("No market within reach.")
        else:
            a, b = st.columns([1.2, 1])
            with a:
                st.pyplot(vz.plot_net_prices(rank))
            with b:
                head("Best group of markets", "Needs at least 3 reachable markets to be a recommendation.")
                st.dataframe(csum.round(2))
            st.dataframe(rank[["State", "cluster_name", "recent_price", "road_km", "net_price_per_q", "net_total",
                               "momentum_pct", "rank_listed"]].round(1))
        st.caption("Decision support only, not advice.")

# ------------------------------------------------------------------ download
with tabs["Download"]:
    head("Take everything with you")
    note("One zip with every table, all charts at 300 dpi, the preprocessing log and the results summary.")
    if st.button("Build download package"):
        with tempfile.TemporaryDirectory() as tmp:
            pl.save_outputs(res, Path(tmp) / "outputs", pl.make_figures(res))
            zpath = shutil.make_archive(str(Path(tmp) / "results"), "zip", Path(tmp) / "outputs")
            st.session_state["zip"] = Path(zpath).read_bytes()
    if "zip" in st.session_state:
        st.download_button("Download results.zip", st.session_state["zip"], file_name="market_clustering_results.zip")
