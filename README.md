# Regional Market Price Behaviour Segmentation for Comparative Pattern Analysis

Unsupervised-learning class project based on SIH26132 (Units II-IV): K-Means / K-Medians partition clustering, Ward hierarchical
comparison, PCA visualisation and quantitative validation of **markets (mandis)** using Agmarknet price history for one commodity.

> **Data:** Agmarknet daily price export for **Wheat** (7 Nov 2025 to 3 Nov 2026, 269,406 rows, 1,379 markets, 16 states/UTs),
> supplied as `data/agmarknet_wheat.csv.gz`. If that file is missing, a labelled SYNTHETIC fallback is generated instead.

## Setup
```
python -m pip install -r requirements.txt
```
(`requirements-tested.txt` has the exact versions the project was last tested with.)

## Run (upload app, recommended)
```
python -m streamlit run app.py
```
Upload an Agmarknet CSV (`.csv` or `.csv.gz`) in the sidebar, pick the commodity, adjust settings if you like, and click **Run analysis**.
The app then derives everything: Objective 1 (variable review, cleaning log, features, correlation pruning, scaling),
Objective 2 (K-Means and K-Medians for every K, WCSS/L1, Silhouette, Davies-Bouldin, PCA, Ward comparison, documented selection),
Objective 3 (cluster profiles, feature contributions, stability, limitations, responsible-use statement) and the farmer advisor.
**One tab per step:** Overview, Data, Cleaning, Features, Correlation, Choosing K, PCA, Hierarchy, Profiles, States and map, Drivers, Stability,
Farmer advisor and Download. Each tab opens with a plain-language note on what the charts show. The PCA scatter and the feature box plots are
interactive (hover, zoom, pick a feature), and the map shows markets coloured by group. The look (pastel palette, fonts) lives in
`.streamlit/config.toml` and the style block at the top of `app.py`; chart colours are in `src/visualise.py`.

**Auto-adjust (on by default):** the app inspects the file and chooses the coverage threshold, minimum rows per market and K range
that the data can support, then reports what it chose in the Summary tab. Untick it to set them by hand. Clustering compares markets, so a file
needs at least 6 markets with several weeks of prices; a single-market file (for example 131 rows of one market) cannot be clustered and the app says so.
The **Download** tab packages all tables, 300-dpi figures, the preprocessing log and the results summary as one zip.

Start it with `python -m streamlit run app.py` in a terminal opened in this folder. Running `app.py` with the Code Runner button or
`python app.py` only prints this instruction (this folder's `.vscode/settings.json` also makes the Code Runner button launch it through Streamlit).

Other ways to run the same pipeline:
```
python run_pipeline.py                                                       # reads data_path in config.yaml, writes outputs/
python -m jupyter nbconvert --execute --to notebook --inplace notebook.ipynb  # narrated notebook version
```

## Use other data
Save another Agmarknet export in `data/` (the title row above the header, Indian digit grouping such as `3,00,02,800.00`, and
`State/UT` / `Arrival Quantity` style names are handled automatically), then set `data_path` and `commodity` in `config.yaml`
(or simply upload it in the app). `min_coverage: 0.60` keeps markets that reported on at least 60% of calendar days; lower it for
sparse commodities and say why.

## Layout
```
config.yaml  .streamlit/config.toml  requirements.txt  requirements-tested.txt  app.py  notebook.ipynb  run_pipeline.py  build_notebook.py
src/  preprocess.py  features.py  clustering.py  validation.py  visualise.py  advisor.py  geocode.py  pipeline.py
data/agmarknet_wheat.csv.gz   docs/  (SIH26132 submission text and idea PPT)
outputs/  figures/ (27 pastel charts, 300 dpi)  metrics_grid.csv  selection_ranking.csv  cluster_assignments.csv  preprocessing_log.md  ...
```

## Method summary
| Step | Choice |
|---|---|
| Unit | Market (State + Market name); State used only to interpret clusters |
| Cleaning | Zero prices -> NaN; Min>Max swapped; Modal/Min/Max entry errors (more than 5x from the median or Modal) -> NaN; missing Modal = (Min+Max)/2; Modal clipped into [Min, Max] |
| Panel | Median modal price per market-week; markets with <60% day coverage dropped; gaps <=4 weeks time-interpolated |
| Outliers | Winsorised per market at Tukey fences (k=3), clipped not deleted |
| Features | 15 candidates (level, dispersion, shape, dynamics, supply); pruned at \|r\|>0.9; median-imputed; `StandardScaler` |
| Euclidean | `sklearn.KMeans`, K=2..10, `n_init=20`, seed 42 |
| Manhattan | Custom **K-Medians** (L1 assignment + median update), because the mean does not minimise summed L1 distance |
| Metrics | WCSS / L1 cost, Silhouette and Davies-Bouldin with the matching metric, plus Silhouette on the first 2 PCs |
| Selection | K>=3, smallest cluster >=5%; mean of min-max normalised [Silhouette, -DB, PCA-2D silhouette]; Euclidean preferred within 0.02 |
| Hierarchy | Ward linkage, cut at chosen K; ARI vs partition; cophenetic correlation |
| Stability | 20 seeds (single-start), 50 bootstrap 80% subsamples, Euclidean vs Manhattan, 3 sensitivity scenarios; mean ARI >=0.8 stable, 0.5-0.8 moderate, <0.5 unstable |

## Findings (real wheat data)
- 269,406 rows; 426 of 1,379 markets kept (>= 60% day coverage). Cleaning log: 9 impossible Modal values, 12 Max and 143 Min entry errors (for example a Max of 3,00,02,800 against a Modal of 2,600) set to NaN; 156 weekly values winsorised; 11 features after correlation pruning; first two PCs explain 60.2% of variance.
- **Cluster structure is weak.** Silhouette peaks at K=2 (Manhattan 0.317) and falls as K grows. Euclidean K-Means at K >= 3 isolates micro-clusters of 1 to 5 outlier markets (the PCA plot labels APMC Mumbai, Solapur and Sironj as outliers), so no Euclidean K >= 3 passes the 5% minimum-size rule.
- **Selected: Manhattan (K-Medians), K=3**: silhouette 0.227, Davies-Bouldin 1.484, clusters of 129 / 148 / 149 markets. Runner-up: Manhattan K=4. K=2 scores best on silhouette but is excluded by the K >= 3 rule for interpretability.
- Provisional profiles: C0 = higher peak prices and above-regional-median prices in 74% of weeks (65 Madhya Pradesh, 35 Gujarat and 16 Maharashtra markets); C1 = low seasonality and trend (114 of 118 Uttar Pradesh markets); C2 = lower floor prices, mostly Madhya Pradesh and Rajasthan.
- Ward (K=3) vs the chosen partition: ARI 0.37, cophenetic correlation 0.39.
- Stability: seeds 0.76, bootstrap 0.77, Manhattan vs Euclidean 0.40, sensitivity 0.69; overall mean ARI 0.66, **moderately stable**. Different distance choices disagree most.
- Top differentiating features (eta-squared): max_price 0.61, seasonal_amp 0.60, trend_slope 0.53.
- Farmer advisor: the export has no coordinates, so `src/geocode.py` derives approximate ones from State/District/Market names with an offline gazetteer (GeoNames via `geonamescache`): 404 of 426 markets matched (303 by market name, 101 by district name; 22 unmatched are left out of distance ranking). Supply `data/mandi_coordinates.csv` (`market_id,lat,lon`) to override. In the demo (example farmer location, 50 quintals, assumed Rs 0.20 per quintal-km and 1% fee) the best net market and the best listed market coincide.

## Limitations and responsible use
Reporting gaps and uneven coverage (953 of 1,379 markets dropped); variety/grade mixing and arrival effects; one commodity and only about 12 months, so seasonality, trend and harvest-season surges are confounded;
K-Means assumes compact similar-variance clusters and is sensitive to scaling/outliers; PCA discards variance; Silhouette/DB are heuristics, and the two distances' scores are not strictly comparable.
Clusters are **exploratory and provisional**, not permanent categories, and must not drive pricing, policy targeting or regional labelling without domain validation.

## SIH26132 farmer layer (class project; all states)
`src/advisor.py` ranks markets by **net price = price x (1 - fee) - transport rate x road km** and recommends the best cluster of markets.
Distances use approximate coordinates matched from names (`src/geocode.py`) unless `data/mandi_coordinates.csv` (`market_id,lat,lon`, `market_id` = `State | Market`) is supplied. In the dashboard you can start from any market in the CSV. Clusters with fewer than 3 reachable markets are not recommended.
Transport rate and fee in `config.yaml` are assumptions. Submission text and problem mapping: `docs/SIH26132_submission.md`.
