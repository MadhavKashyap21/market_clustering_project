"""Farmer-facing layer for SIH26132: transport-adjusted net price and cluster-level selling guidance."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_STATE_CENTROIDS = {"Maharashtra": (19.7, 75.7), "Karnataka": (15.3, 75.7), "Madhya Pradesh": (23.5, 78.0),
                    "Gujarat": (22.3, 71.7), "Rajasthan": (26.6, 73.8), "Uttar Pradesh": (26.8, 80.9),
                    "West Bengal": (23.0, 87.9), "Punjab": (31.0, 75.3)}


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def market_coordinates(index: pd.Index, cfg: dict[str, Any], synthetic: bool, df: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """Synthetic run: PSEUDO coordinates. Real run: data/mandi_coordinates.csv if present (market_id, lat, lon),
    otherwise APPROXIMATE coordinates derived from State/District/Market names (src/geocode.py)."""
    if synthetic:
        rng = np.random.default_rng(cfg["random_state"])
        st = index.str.split(" \\| ").str[0]
        base = np.array([_STATE_CENTROIDS.get(s, (22.0, 78.0)) for s in st])
        xy = base + rng.uniform(-1.2, 1.2, base.shape)
        return pd.DataFrame(xy, index=index, columns=["lat", "lon"])
    try:
        user = pd.read_csv(cfg["advisor"]["coordinates_path"]).set_index("market_id")[["lat", "lon"]].reindex(index)
        return user.assign(coord_source="user file")
    except FileNotFoundError:
        pass
    if df is None:
        return None
    from .geocode import derive_coordinates
    meta = df.groupby("market_id").agg(State=("State", "first"), Market=("Market", "first"), District=("District", "first"))
    return derive_coordinates(meta.reindex(index))


def build_snapshot(weekly: pd.DataFrame, feat: pd.DataFrame, labels, names: dict, index: pd.Index) -> pd.DataFrame:
    recent, prev = weekly.tail(4).median(), weekly.iloc[-8:-4].median()
    snap = pd.DataFrame({"State": index.str.split(" \\| ").str[0], "cluster": labels,
                         "cluster_name": [names[c] for c in labels]}, index=index)
    snap["recent_price"] = recent.reindex(index)
    snap["momentum_pct"] = ((recent / prev - 1) * 100).reindex(index)
    snap["volatility"] = feat["volatility"].reindex(index)
    return snap


def rank_markets(snap: pd.DataFrame, lat: float, lon: float, qty: float, rate: float,
                 road_factor: float, fee_pct: float, max_km: float):
    """Net price per quintal = price*(1-fee) - rate*road_km. Returns (market ranking, cluster summary)."""
    d = snap.dropna(subset=["lat", "lon", "recent_price"]).copy()
    d["road_km"] = haversine_km(lat, lon, d["lat"], d["lon"]) * road_factor
    d = d[d["road_km"] <= max_km]
    d["transport_per_q"] = rate * d["road_km"]
    d["net_price_per_q"] = d["recent_price"] * (1 - fee_pct / 100) - d["transport_per_q"]
    d["net_total"] = d["net_price_per_q"] * qty
    d = d.sort_values("net_price_per_q", ascending=False)
    d["rank_net"] = range(1, len(d) + 1)
    d["rank_listed"] = d["recent_price"].rank(ascending=False, method="first").astype(int)
    cs = d.groupby(["cluster", "cluster_name"]).agg(
        markets_in_reach=("net_price_per_q", "size"), median_net_price=("net_price_per_q", "median"),
        best_net_price=("net_price_per_q", "max"), avg_volatility=("volatility", "mean"),
        avg_momentum_pct=("momentum_pct", "mean"), best_market=("net_price_per_q", lambda s: s.idxmax()))
    cs["enough_markets"] = cs["markets_in_reach"] >= 3          # a cluster median over 1-2 markets is not a recommendation
    return d, cs.sort_values(["enough_markets", "median_net_price"], ascending=False).reset_index()
