"""Data loading, synthetic fallback generation, cleaning and weekly panel construction."""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PRICE_COLS = ["Min_Price", "Max_Price", "Modal_Price"]


class PreprocessingLog:
    """Collects every preprocessing decision and writes it to Markdown."""

    def __init__(self, silent: bool = False) -> None:
        self.entries: list[tuple[str, str]] = []
        self.stats: dict[str, int] = {}          # numeric counts behind the log lines (used for charts)
        self.silent = silent

    def add(self, section: str, text: str) -> None:
        self.entries.append((section, text))
        if not self.silent:
            print(f"[{section}] {text}")

    def to_markdown(self, title: str = "Preprocessing Log") -> str:
        lines, current = [f"# {title}"], None
        for section, text in self.entries:
            if section != current:
                lines += ["", f"## {section}", ""]
                current = section
            lines.append(f"- {text}")
        return "\n".join(lines) + "\n"

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_markdown(), encoding="utf-8")


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- synthetic
_STATES = ["Maharashtra", "Karnataka", "Madhya Pradesh", "Gujarat",
           "Rajasthan", "Uttar Pradesh", "West Bengal", "Punjab"]
_ARCHETYPES = {
    "stable": dict(base=1500, vol=0.020, seas=0.10, trend=0.02, cov=(0.80, 0.98)),
    "volatile": dict(base=2500, vol=0.060, seas=0.30, trend=0.08, cov=(0.70, 0.95)),
    "seasonal": dict(base=1900, vol=0.030, seas=0.45, trend=0.03, cov=(0.75, 0.95)),
    "thin": dict(base=2100, vol=0.090, seas=0.15, trend=0.00, cov=(0.45, 0.75)),
}


def generate_synthetic(cfg: dict[str, Any], seed: int) -> pd.DataFrame:
    """Synthetic mandi price history with the Agmarknet schema (DEMONSTRATION ONLY)."""
    rng = np.random.default_rng(seed)
    s = cfg["synthetic"]
    dates = pd.date_range(s["start"], s["end"], freq="D")
    n = len(dates)
    years = (dates - dates[0]).days.values / 365.25
    doy = dates.dayofyear.values / 365.25
    nat = np.zeros(n)
    for t in range(1, n):
        nat[t] = 0.97 * nat[t - 1] + 0.02 * rng.standard_normal()

    names = list(_ARCHETYPES)
    state_probs = {st: rng.dirichlet(np.ones(4) * 1.5) for st in _STATES}
    rows: list[pd.DataFrame] = []
    n_markets = int(s["n_markets"])
    for i in range(n_markets):
        state = _STATES[i % len(_STATES)]
        sparse = i >= n_markets - 3                      # 3 deliberately sparse reporters
        arch = _ARCHETYPES[names[rng.choice(4, p=state_probs[state])]]
        jit = np.exp(rng.normal(0, 0.10))
        noise = np.zeros(n)
        for t in range(1, n):
            noise[t] = 0.9 * noise[t - 1] + arch["vol"] * 0.6 * rng.standard_normal()
        phase = 0.5 + rng.normal(0, 0.3)
        price = (arch["base"] * jit * (1 + arch["trend"] * years)
                 * (1 + arch["seas"] * np.sin(2 * np.pi * doy + phase))
                 * np.exp(0.5 * nat) * np.exp(noise))
        p_obs = 0.15 if sparse else rng.uniform(*arch["cov"])
        keep = rng.random(n) < p_obs
        arr_scale = rng.uniform(80, 900)
        for vname, vmult, vprob in [("Local", 1.0, 1.0), ("Hybrid", 1.05, 0.4)]:
            sel = keep & (rng.random(n) < vprob)
            modal = price[sel] * vmult * (1 + rng.normal(0, 0.01, sel.sum()))
            rows.append(pd.DataFrame({
                "State": state, "District": f"{state[:3]}-District-{i % 5}",
                "Market": f"{state[:3].upper()}-Mandi-{i:02d}", "Commodity": "Onion",
                "Variety": vname, "Grade": "FAQ",
                "Arrival_Date": dates[sel].strftime("%d/%m/%Y"),
                "Min_Price": modal * (1 - rng.uniform(0.03, 0.12, sel.sum())),
                "Max_Price": modal * (1 + rng.uniform(0.03, 0.12, sel.sum())),
                "Modal_Price": modal,
                "Arrivals": arr_scale * rng.lognormal(0, 0.4, sel.sum()),
            }))
    df = pd.concat(rows, ignore_index=True)

    m = len(df)                                           # inject realistic data-quality issues
    df.loc[rng.random(m) < 0.003, "Modal_Price"] = np.nan
    swap = rng.random(m) < 0.003
    df.loc[swap, ["Min_Price", "Max_Price"]] = df.loc[swap, ["Max_Price", "Min_Price"]].values
    df.loc[rng.random(m) < 0.001, "Min_Price"] = 0.0
    spike = rng.random(m) < 0.002
    df.loc[spike, PRICE_COLS] = df.loc[spike, PRICE_COLS].values * rng.uniform(4, 6, (spike.sum(), 1))
    df.loc[rng.random(m) < 0.05, "Arrivals"] = np.nan
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- loading
def _canonical(col: str) -> str | None:
    key = re.sub(r"[^a-z]", "", col.lower().replace("x0020", ""))
    if key.startswith("arrivaldate") or key == "date":
        return "Arrival_Date"
    special = {"commoditygroup": "Commodity_Group", "priceunit": "Price_Unit", "arrivalunit": "Arrival_Unit"}
    if key in special:
        return special[key]
    if key.startswith("arrival"):
        return "Arrivals"
    for prefix, name in [("minprice", "Min_Price"), ("maxprice", "Max_Price"),
                         ("modalprice", "Modal_Price"), ("state", "State"),
                         ("district", "District"), ("market", "Market"),
                         ("commodity", "Commodity"), ("variety", "Variety"),
                         ("grade", "Grade")]:
        if key.startswith(prefix):
            return name
    return None


def read_agmarknet(source) -> tuple[pd.DataFrame, dict]:
    """Read an Agmarknet export from a path or raw bytes (csv or csv.gz). Detects the real header row
    (exports often have a title line first) and renames columns to the canonical schema."""
    import io
    if isinstance(source, (bytes, bytearray)):
        comp = "gzip" if bytes(source[:2]) == b"\x1f\x8b" else None
        opener = lambda: io.BytesIO(bytes(source))
    else:
        comp, opener = "infer", lambda: source
    kw = dict(dtype=str, compression=comp, encoding_errors="replace")
    head = pd.read_csv(opener(), header=None, nrows=6, **kw)
    hdr = next((i for i, row in head.iterrows() if any("modal" in str(v).lower() for v in row)), 0)
    df = pd.read_csv(opener(), header=hdr, low_memory=False, **kw)
    mapping = {c: _canonical(c) for c in df.columns if _canonical(c)}
    return df.rename(columns=mapping), mapping


def load_data(cfg: dict[str, Any], log: PreprocessingLog, source=None) -> tuple[pd.DataFrame, bool]:
    """source: raw bytes, a path, or a DataFrame from read_agmarknet. If None, use cfg['data_path'] when it exists,
    otherwise generate the labelled SYNTHETIC fallback."""
    if source is None and Path(cfg["data_path"]).exists():
        source = str(cfg["data_path"])
    if isinstance(source, pd.DataFrame):
        df, synthetic = source.copy(), False
        log.add("Loading", "Using the uploaded data (columns already normalised)")
    elif source is not None:
        df, mapping = read_agmarknet(source)
        what = "uploaded file" if isinstance(source, (bytes, bytearray)) else source
        log.add("Loading", f"Loaded {what}; renamed columns {mapping}")
        synthetic = False
    else:
        warnings.warn("DATA FILE NOT FOUND - using SYNTHETIC data. Results are for demonstration only.")
        print("=" * 78 + "\nWARNING: SYNTHETIC DATA IN USE. Results are for demonstration only.\n" + "=" * 78)
        df = generate_synthetic(cfg, cfg["random_state"])
        log.add("Loading", f"{cfg['data_path']} not found -> generated SYNTHETIC data (demonstration only), "
                           f"{cfg['synthetic']['n_markets']} markets, seed={cfg['random_state']}")
        synthetic = True
    required = {"State", "Market", "Arrival_Date", "Modal_Price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)}. Columns found: {list(df.columns)}")
    for optional in ["Commodity", "District", "Variety", "Grade", "Min_Price", "Max_Price"]:
        if optional not in df.columns:
            df[optional] = np.nan
    return df, synthetic


def relevance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Keep/drop review of every column with a justification."""
    rows = []
    for c in df.columns:
        nun, miss = df[c].nunique(dropna=True), df[c].isna().mean() * 100
        if nun <= 1:
            dec, why = "Drop", "Constant after commodity filter - carries no information"
        elif c == "State":
            dec, why = "Keep (label only)", "Used post hoc to interpret clusters; NOT a clustering feature"
        elif c == "Market":
            dec, why = "Keep (unit ID)", "Unit of analysis; an identifier, never a feature"
        elif c == "District":
            dec, why = "Drop", "Nested inside Market/State; near-unique ID adds no behavioural signal"
        elif c in ("Variety", "Grade"):
            dec, why = "Keep (aggregated)", "Median across varieties/grades per market-week; mixing noted as a limitation"
        elif c == "Arrival_Date":
            dec, why = "Keep", "Time index for weekly series, seasonality, trend"
        elif c == "Modal_Price":
            dec, why = "Keep", "Primary price signal (most frequent traded price)"
        elif c in ("Min_Price", "Max_Price"):
            dec, why = "Keep", "Used for intraday spread (Max-Min)/Modal and data repair"
        elif c == "Arrivals":
            dec, why = ("Keep", "Supply-side context") if miss < 50 else ("Drop", f"{miss:.0f}% missing")
        else:
            dec, why = "Drop", "Not relevant to price behaviour"
        rows.append({"column": c, "n_unique": nun, "pct_missing": round(miss, 2),
                     "decision": dec, "justification": why})
    return pd.DataFrame(rows)


def clean_raw(df: pd.DataFrame, cfg: dict[str, Any], log: PreprocessingLog) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter commodity, parse dates, repair impossible values. Returns (clean_df, relevance_table)."""
    d = df.copy()
    log.add("Loading", f"Raw shape {d.shape}; dtypes: " + ", ".join(f"{c}={t}" for c, t in d.dtypes.astype(str).items()))
    com = str(cfg["commodity"]).lower()
    if d["Commodity"].notna().any() and com:
        d = d[d["Commodity"].astype(str).str.lower().str.contains(com, na=False, regex=False)].copy()
    log.add("Filtering", f"Filtered to commodity '{cfg['commodity']}': {len(d):,} rows")
    log.stats["_raw_rows"], log.stats["_commodity_rows"] = len(df), len(d)
    if d.empty:
        raise ValueError(f"No rows for commodity '{cfg['commodity']}'")
    rel = relevance_table(d)
    miss = d.isna().mean().mul(100).round(2)
    log.add("Missing values", "Percent missing per column: " + ", ".join(f"{c}={v}%" for c, v in miss.items()))

    d["Arrival_Date"] = pd.to_datetime(d["Arrival_Date"], dayfirst=True, errors="coerce")
    bad = int(d["Arrival_Date"].isna().sum())
    d = d.dropna(subset=["Arrival_Date"])
    log.stats["Unreadable dates removed"] = bad
    log.add("Parsing", f"Parsed dates (dayfirst); dropped {bad} unparseable rows. Range "
                       f"{d['Arrival_Date'].min().date()} to {d['Arrival_Date'].max().date()}")
    for c in PRICE_COLS + ["Arrivals"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", "", regex=False), errors="coerce")  # Indian digit grouping
    for c in PRICE_COLS:
        nonpos = int((d[c] <= 0).sum())
        d.loc[d[c] <= 0, c] = np.nan
        log.stats["Zero or negative prices blanked"] = log.stats.get("Zero or negative prices blanked", 0) + nonpos
        log.add("Impossible values", f"{c}: {nonpos} zero/negative values set to NaN")
    swap = (d["Min_Price"] > d["Max_Price"]).fillna(False)
    d.loc[swap, ["Min_Price", "Max_Price"]] = d.loc[swap, ["Max_Price", "Min_Price"]].values
    log.stats["Min and Max swapped"] = int(swap.sum())
    log.add("Impossible values", f"{int(swap.sum())} rows with Min_Price > Max_Price: values swapped")
    lim = cfg["preprocessing"].get("price_ratio_limit", 5)
    med = d["Modal_Price"].median()
    bad_modal = (d["Modal_Price"] < med / lim) | (d["Modal_Price"] > med * lim)
    d.loc[bad_modal, "Modal_Price"] = np.nan
    log.stats["Modal price entry errors blanked"] = int(bad_modal.sum())
    log.add("Impossible values", f"{int(bad_modal.sum())} Modal_Price values outside [median/{lim}, median x {lim}] "
                                 f"(median {med:,.0f}) treated as entry errors -> NaN")
    bad_max = d["Max_Price"] > lim * d["Modal_Price"]
    bad_min = d["Min_Price"] < d["Modal_Price"] / lim
    d.loc[bad_max, "Max_Price"] = np.nan
    d.loc[bad_min, "Min_Price"] = np.nan
    log.stats["Max price entry errors blanked"], log.stats["Min price entry errors blanked"] = int(bad_max.sum()), int(bad_min.sum())
    log.add("Impossible values", f"{int(bad_max.sum())} Max_Price values > {lim}x Modal and {int(bad_min.sum())} Min_Price values "
                                 f"< Modal/{lim} treated as entry errors -> NaN (row kept; Modal is the primary signal)")
    mid = d["Modal_Price"].isna() & d["Min_Price"].notna() & d["Max_Price"].notna()
    d.loc[mid, "Modal_Price"] = (d.loc[mid, "Min_Price"] + d.loc[mid, "Max_Price"]) / 2
    log.stats["Modal filled from Min and Max"] = int(mid.sum())
    log.add("Missing values", f"{int(mid.sum())} missing Modal_Price rows filled with (Min+Max)/2")
    before = len(d)
    d = d.dropna(subset=["Modal_Price"])
    log.stats["Rows dropped (no modal price)"] = before - len(d)
    log.add("Missing values", f"Dropped {before - len(d)} rows whose Modal_Price is still missing")
    both = d["Min_Price"].notna() & d["Max_Price"].notna()
    out = both & ((d["Modal_Price"] < d["Min_Price"]) | (d["Modal_Price"] > d["Max_Price"]))
    d.loc[out, "Modal_Price"] = np.clip(d.loc[out, "Modal_Price"], d.loc[out, "Min_Price"], d.loc[out, "Max_Price"])
    log.stats["Modal clipped into Min-Max range"] = int(out.sum())
    log.add("Impossible values", f"{int(out.sum())} rows with Modal outside [Min, Max]: clipped into range")
    d["spread"] = (d["Max_Price"] - d["Min_Price"]) / d["Modal_Price"]
    n0 = len(d)
    d = d.drop_duplicates()
    log.stats["Duplicate rows removed"] = n0 - len(d)
    log.add("Cleaning", f"Dropped {n0 - len(d)} exact duplicate rows; {len(d):,} rows remain")
    d["market_id"] = d["State"].astype(str).str.strip() + " | " + d["Market"].astype(str).str.strip()
    return d.reset_index(drop=True), rel


def build_weekly_panel(df: pd.DataFrame, cfg: dict[str, Any], log: PreprocessingLog,
                       winsorise: bool | None = None) -> dict[str, Any]:
    """Aggregate to a market x week panel, drop thin markets, interpolate short gaps, winsorise."""
    p = cfg["preprocessing"]
    winsorise = p["winsorise"] if winsorise is None else winsorise
    span_days = (df["Arrival_Date"].max() - df["Arrival_Date"].min()).days + 1
    stats = pd.DataFrame({
        "n_obs": df.groupby("market_id").size(),
        "coverage": df.groupby("market_id")["Arrival_Date"].nunique() / span_days,
    })
    stats["keep"] = (stats["coverage"] >= p["min_coverage"]) & (stats["n_obs"] >= p["min_obs"])
    if not stats["keep"].any():
        raise ValueError(f"No market reported on >= {p['min_coverage']:.0%} of days with >= {p['min_obs']} rows "
                         f"(best coverage {stats['coverage'].max():.0%}). Lower the coverage threshold.")
    dropped = stats[~stats["keep"]].copy()
    dropped["reason"] = np.where(dropped["coverage"] < p["min_coverage"],
                                 f"coverage < {p['min_coverage']:.0%}", f"n_obs < {p['min_obs']}")
    log.stats["Markets dropped (low coverage)"] = int((~stats["keep"]).sum())
    log.add("Market filtering", f"{int(stats['keep'].sum())} of {len(stats)} markets kept "
            f"(coverage >= {p['min_coverage']:.0%} of calendar days and >= {p['min_obs']} rows); "
            f"dropped: {', '.join(dropped.index) if len(dropped) else 'none'}")
    d = df[df["market_id"].isin(stats.index[stats["keep"]])].copy()
    d["week"] = d["Arrival_Date"].dt.to_period("W-SUN").dt.start_time
    weekly = d.groupby(["week", "market_id"])["Modal_Price"].median().unstack("market_id")
    weekly = weekly.reindex(pd.date_range(weekly.index.min(), weekly.index.max(), freq="W-MON"))
    thin = weekly.notna().sum() < 4                       # trend / seasonality need at least 4 weekly points
    if thin.any():
        log.add("Market filtering", f"{int(thin.sum())} more market(s) dropped: fewer than 4 weekly price points")
        weekly = weekly.loc[:, ~thin]
        d = d[d["market_id"].isin(weekly.columns)]
    if weekly.shape[1] == 0:
        raise ValueError("No market has at least 4 weeks of price data after filtering. Use a file with a longer history.")
    log.add("Aggregation", f"Median Modal_Price per market-week (collapses variety/grade mixing): "
                           f"{weekly.shape[0]} weeks x {weekly.shape[1]} markets")
    n_clipped = 0
    if winsorise:
        q1, q3 = weekly.quantile(0.25), weekly.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - p["iqr_multiplier"] * iqr, q3 + p["iqr_multiplier"] * iqr
        n_clipped = int(((weekly < lo) | (weekly > hi)).sum().sum())
        weekly = weekly.clip(lower=lo, upper=hi, axis=1)
        log.stats["Weekly outliers clipped"] = n_clipped
        log.add("Outliers", f"Winsorised per market at Tukey fences (k={p['iqr_multiplier']}): "
                            f"{n_clipped} weekly values clipped (clipped, not deleted)")
    else:
        log.add("Outliers", "Winsorising DISABLED for this run (sensitivity test)")
    n_gap = int(weekly.isna().sum().sum())
    weekly = weekly.interpolate(method="time", limit=p["max_gap_weeks"], limit_area="inside")
    log.stats["Weekly gaps interpolated"] = n_gap - int(weekly.isna().sum().sum())
    log.add("Missing values", f"Time-interpolated gaps up to {p['max_gap_weeks']} weeks inside each series: "
            f"{n_gap - int(weekly.isna().sum().sum())} of {n_gap} missing weeks filled; rest left NaN (nan-aware features)")
    return {"weekly": weekly, "stats": stats, "dropped": dropped, "rows": d, "n_clipped": n_clipped}
