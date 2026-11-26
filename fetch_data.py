"""
Download the price series once and save them with their provenance.

Run on your own computer (needs internet):
    pip install yfinance pandas
    python fetch_data.py

It writes data/CHSPI.csv, data/ETF_MONDE.csv, data/USDCHF.csv and
data/sources.json (provider, ticker, retrieval date, period, fields).
The app never downloads anything: it only reads these saved files, so
every result can be reproduced from the same observations.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

SERIES = {
    "CHSPI.csv": ["CHSPI.SW"],                     # iShares Core SPI ETF (CH), CHF
    "ETF_MONDE.csv": ["SWDA.SW", "CSWD.SW", "WRDUSW.SW"],  # world equities (first ticker that works)
    "USDCHF.csv": ["CHF=X"],                       # exchange rate: CHF per 1 USD
}

sources = {}
for fname, candidates in SERIES.items():
    for ticker in candidates:
        df = yf.download(ticker, start="2010-01-01", auto_adjust=False, progress=False)
        if df is None or df.empty:
            print(f"  {ticker}: no data, trying next…")
            continue
        if hasattr(df.columns, "levels"):            # yfinance >= 0.2 returns a MultiIndex
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.to_csv(OUT / fname, index=False)
        try:
            currency = yf.Ticker(ticker).fast_info.get("currency", "?")
        except Exception:
            currency = "?"
        sources[fname] = {
            "provider": "Yahoo Finance (via yfinance)",
            "ticker": ticker,
            "url": f"https://finance.yahoo.com/quote/{ticker}/history",
            "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "first_date": str(df["Date"].min())[:10],
            "last_date": str(df["Date"].max())[:10],
            "rows": int(len(df)),
            "currency": currency,
            "fields_used": "Adj Close (adjusted for dividends and splits)" if fname != "USDCHF.csv" else "Close (CHF per 1 USD)",
        }
        print(f"OK {fname} <- {ticker} ({currency}), {len(df)} rows")
        break
    else:
        print(f"FAILED for {fname}: edit the ticker list in fetch_data.py")

(OUT / "sources.json").write_text(json.dumps(sources, indent=2, ensure_ascii=False))
print("Provenance saved in data/sources.json")
