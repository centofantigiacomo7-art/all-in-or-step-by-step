"""
Data loading, cleaning and provenance for "All In or Step by Step?".

Every transformation is written to a log so that a number in the app can be
traced back to the observed prices (guideline 4).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
FX_FILE = "USDCHF.csv"          # CHF per 1 USD
SARON_FILE = "SARON.csv"        # optional: SNB data portal, monthly, % per year

ASSETS = {
    "World equities (MSCI World, in CHF)": "ETF_MONDE.csv",
    "Swiss equities (SPI)": "CHSPI.csv",
}


@dataclass
class MarketData:
    asset: str
    prices: pd.Series                 # month-end prices in CHF, estimation window
    all_prices: pd.Series             # month-end prices in CHF, full history (backtest)
    returns: pd.Series                # monthly simple returns
    savings_rate: pd.Series | None    # monthly rate (decimal), None if no SARON file
    log: list[str] = field(default_factory=list)
    sources: dict = field(default_factory=dict)


def load_price_csv(path: Path, log: list[str]) -> pd.Series:
    """Read a Yahoo-style CSV (Date, ..., Adj Close / Close)."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.name}")
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() in ("date", "datetime")), None)
    if date_col is None:
        raise ValueError(f"{path.name}: no 'Date' column.")
    price_col = next((c for c in ("Adj Close", "Adj_Close", "Close", "NAV", "Price") if c in df.columns), None)
    if price_col is None:
        raise ValueError(f"{path.name}: no price column (Adj Close / Close).")
    s = pd.Series(pd.to_numeric(df[price_col], errors="coerce").values,
                  index=pd.DatetimeIndex(pd.to_datetime(df[date_col], utc=True, errors="coerce")).tz_localize(None),
                  name=path.stem).sort_index()
    n0 = len(s)
    s = s[s.index.notna()].dropna()
    s = s[s > 0]
    s = s[~s.index.duplicated(keep="last")]
    log.append(f"{path.name}: {n0} rows read, column '{price_col}', {n0 - len(s)} rows removed "
               f"(empty, ≤ 0 or duplicates), period {s.index.min():%Y-%m-%d} → {s.index.max():%Y-%m-%d}.")
    return s


def load_saron(path: Path, log: list[str]) -> pd.Series | None:
    """Optional SNB export: a date column and a rate in % per year. Returns a monthly decimal rate."""
    if not path.exists():
        log.append("No SARON.csv: the savings rate is a constant assumption chosen in the app.")
        return None
    raw = path.read_text(encoding="utf-8-sig").splitlines()
    sep = ";" if raw and raw[0].count(";") >= raw[0].count(",") else ","
    start = next((i for i, line in enumerate(raw) if line.lower().startswith(("date", '"date'))), 0)
    df = pd.read_csv(path, sep=sep, skiprows=start)
    date_col = df.columns[0]
    val_col = df.columns[-1]
    # SNB exports can hold several series in one file (a dimension column such as D0): keep SARON only
    for c in df.columns[1:-1]:
        labels = df[c].astype(str)
        if labels.str.contains("SARON", case=False).any():
            n_all = len(df)
            df = df[labels.str.contains("SARON", case=False)]
            log.append(f"SARON.csv: column '{c}' holds several series; kept the {len(df)} SARON rows of {n_all}.")
            break
    rate = pd.Series(pd.to_numeric(df[val_col], errors="coerce").values,
                     index=pd.to_datetime(df[date_col].astype(str), errors="coerce")).dropna()
    rate = rate[rate.index.notna()].sort_index()
    if rate.empty:
        raise ValueError("SARON.csv: no usable rows (expected a date column and a rate in % per year).")
    rate.index = rate.index + pd.offsets.MonthEnd(0)
    monthly = (1 + rate / 100) ** (1 / 12) - 1
    log.append(f"SARON.csv: {len(rate)} monthly rates, {rate.index.min():%Y-%m} → {rate.index.max():%Y-%m}; "
               "converted from % per year to a monthly rate: (1 + R/100)^(1/12) − 1.")
    return monthly


def load_market(asset: str, start: str | None = None, data_dir: Path = DATA_DIR) -> MarketData:
    if asset not in ASSETS:
        raise ValueError(f"Unknown asset: {asset}")
    log: list[str] = []
    sources = {}
    sp = data_dir / "sources.json"
    if sp.exists():
        sources = json.loads(sp.read_text())
    fname = ASSETS[asset]
    p = load_price_csv(data_dir / fname, log)
    if sources.get(fname, {}).get("currency") == "USD":
        fx = load_price_csv(data_dir / FX_FILE, log)
        p = (p * fx.reindex(p.index, method="ffill")).dropna()
        log.append(f"{fname} converted from USD to CHF: price_CHF = price_USD × (CHF per USD) on the same day.")
    monthly = p.resample("ME").last().dropna()
    if p.index.max() < monthly.index[-1] - pd.Timedelta(days=5):
        log.append(f"Last month ({monthly.index[-1]:%Y-%m}) is incomplete (data until "
                   f"{p.index.max():%Y-%m-%d}): removed.")
        monthly = monthly.iloc[:-1]
    full = monthly.copy()
    if start:
        monthly = monthly[monthly.index >= pd.Timestamp(start) - pd.offsets.MonthEnd(1)]
    rets = monthly.pct_change().dropna()
    if len(rets) < 36:
        raise ValueError(f"Only {len(rets)} months of data: at least 36 are needed.")
    log.append(f"Month-end prices → simple returns P_t / P_(t−1) − 1: {len(rets)} months, "
               f"{rets.index.min():%Y-%m} → {rets.index.max():%Y-%m}.")
    saron = load_saron(data_dir / SARON_FILE, log)
    used = {k: v for k, v in sources.items() if k in (fname, FX_FILE, SARON_FILE)}
    return MarketData(asset, monthly, full, rets, saron, log, used)


def moments(returns: pd.Series) -> tuple[float, float]:
    """g = E[1+R], h = E[(1+R)^2], estimated by sample averages."""
    x = 1 + returns.to_numpy()
    return float(x.mean()), float((x ** 2).mean())
