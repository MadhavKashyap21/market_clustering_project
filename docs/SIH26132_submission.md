# SIH26132 - Strengthening market linkages and price discovery for farmers
Class project based on SIH26132 | Software | Agriculture, FoodTech & Rural Development

> Data: real Agmarknet wheat export (7 Nov 2025 to 3 Nov 2026; 426 well-reporting markets analysed). Cluster structure is weak, so groupings are exploratory.

## Abstract (about 190 words)
Farmers, especially smallholders, often cannot see how prices behave across nearby mandis and may sell at a poor time or place.
We turn open Agmarknet mandi price history into a decision-support layer. First, each market is summarised by an interpretable behaviour
profile (price level, volatility, seasonality, trend, intraday spread, premium over the regional median) and markets are grouped with
K-Means / K-Medians, validated with WCSS, Silhouette and Davies-Bouldin, visualised with PCA and cross-checked with Ward hierarchical
clustering. Stability is tested across seeds, bootstrap samples, distance choice and preprocessing choices so groupings are not over-claimed.
Second, a farmer-facing advisor (given market coordinates) combines each market's recent price and momentum with distance to the farmer to compute a
transport-adjusted net price per quintal, and recommends the cluster of markets with the best net outcome and the best market inside it.
This replaces "highest listed price" with "best net realisation". Clusters are exploratory and provisional, not permanent market categories.
The pipeline is reproducible, config-driven and can be re-run for any commodity or state, with a Streamlit app where a user uploads any Agmarknet CSV and every analysis output is regenerated.

## Problem mapping
| What SIH26132 asks | How this project addresses it | Where |
|---|---|---|
| Limited visibility of current and expected prices across nearby markets | Recent price, 4-week momentum, volatility per market; cluster behaviour profiles | `outputs/market_snapshot.csv`, notebook |
| Transparent price discovery | Reproducible metrics, documented preprocessing log, stability verdict | `outputs/preprocessing_log.md`, `stability_*.csv` |
| Weak bargaining power / selling at the wrong time | Momentum and seasonality signals shown next to net price | `src/advisor.py`, features |
| Efficient linkage from farm gate to suitable buyers | Net price after transport and fee; best cluster and best market | `advisor_demo_*.csv`, Streamlit "Farmer advisor" tab |
| Fragmented information on quality, logistics, buyers | NOT covered: no buyer credentials, quality specs, storage or payment data | See gaps below |

## Honest gaps (say these to the jury before they ask)
- Cluster structure on wheat is weak (silhouette 0.23 at K=3; overall stability 0.66, moderately stable). Euclidean K-Means isolates outlier markets, so K-Medians was selected.
- Only about 12 months of one commodity; 953 of 1,379 markets were dropped for low reporting coverage.
- The export has no coordinates, so market locations are approximate place centroids matched from State/District/Market names (404 of 426 markets; district-level for about a quarter). Transport rate (Rs 0.20 per quintal-km) and mandi fee (1%) are assumptions, not measured values.
- No forecasting model: "expected price" is a recent-median and momentum signal, not a forecast.
- No buyer/processor/FPO layer, quality grading or payment-reliability data; the farmer UI is a dashboard prototype, not a mobile/regional-language app.
- Scope is national (16 states/UTs in the export, 10 with markets that passed the coverage filter); results are not specific to any one state.
