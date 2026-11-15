"""Approximate market coordinates from State / District / Market names using an offline gazetteer (GeoNames via geonamescache).

Matching is by name inside the same state: market name first, then district name. Coordinates are place centroids,
so they are APPROXIMATE (district-level for district matches). Markets with no match stay NaN. Override with data/mandi_coordinates.csv.
"""
from __future__ import annotations

import re

import pandas as pd

# GeoNames admin1 codes for India, verified against anchor cities (e.g. Nashik -> Maharashtra).
ADMIN1 = {"Maharashtra": "16", "Madhya Pradesh": "35", "Uttar Pradesh": "36", "Rajasthan": "24", "Gujarat": "09",
          "Karnataka": "19", "Punjab": "23", "West Bengal": "28", "Chattisgarh": "37", "Uttarakhand": "39",
          "Bihar": "34", "Telangana": "40", "NCT of Delhi": "07", "Haryana": "10"}
_NOISE = r"\b(apmc|mandi|krishi upaj|krishi|upaj|market|yard|sub|agricultural produce market committee|aprmc|samiti|bazar)\b"


def _clean(s: str) -> str:
    s = re.sub(r"\(.*?\)", "", str(s))
    return re.sub(r"[^a-z]", "", re.sub(_NOISE, " ", s.lower(), flags=re.I))


def derive_coordinates(markets: pd.DataFrame) -> pd.DataFrame | None:
    """markets: index market_id, columns State, Market, District. Returns lat, lon, coord_source (or None if no gazetteer)."""
    try:
        import geonamescache
    except ImportError:
        return None
    cities = [c for c in geonamescache.GeonamesCache(min_city_population=1000).get_cities().values() if c["countrycode"] == "IN"]
    idx: dict[tuple[str, str], tuple[float, float, int]] = {}
    for c in cities:
        for nm in [c["name"]] + [a for a in c["alternatenames"] if a and a.isascii()]:
            key = (c["admin1code"], _clean(nm))
            if key[1] and (key not in idx or c["population"] > idx[key][2]):
                idx[key] = (c["latitude"], c["longitude"], c["population"])
    rows = []
    for mid, r in markets.iterrows():
        adm, hit, src = ADMIN1.get(str(r["State"]).strip()), None, None
        if adm:
            for col, label in (("Market", "market-name"), ("District", "district-name")):
                hit = idx.get((adm, _clean(r[col])))
                if hit:
                    src = label
                    break
        rows.append({"market_id": mid, "lat": hit[0] if hit else float("nan"), "lon": hit[1] if hit else float("nan"),
                     "coord_source": src})
    return pd.DataFrame(rows).set_index("market_id")
