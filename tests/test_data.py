"""Data pipeline tests on synthetic Yahoo-style files (test fixtures, not real data)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import ASSETS, load_market, moments  # noqa: E402

WORLD = "World equities (MSCI World, in CHF)"


def _write(d: Path, name: str, prices, start="2015-01-01"):
    idx = pd.bdate_range(start, periods=len(prices))
    pd.DataFrame({"Date": idx, "Close": prices, "Adj Close": prices}).to_csv(d / name, index=False)


def test_usd_asset_is_converted_to_chf(tmp_path):
    n = 1200
    _write(tmp_path, ASSETS[WORLD], np.full(n, 100.0))
    _write(tmp_path, "USDCHF.csv", np.linspace(1.0, 0.8, n))       # the franc strengthens
    (tmp_path / "sources.json").write_text(json.dumps({ASSETS[WORLD]: {"currency": "USD"}}))
    d = load_market(WORLD, data_dir=tmp_path)
    assert (d.returns < 0).all()                                    # flat in USD, falling in CHF


def test_moments_by_hand():
    g, h = moments(pd.Series([0.1, -0.1]))
    assert g == pytest.approx(1.0)
    assert h == pytest.approx((1.1 ** 2 + 0.9 ** 2) / 2)


def test_saron_file_is_parsed(tmp_path):
    _write(tmp_path, ASSETS[WORLD], 100 * np.cumprod(np.full(1200, 1.0003)))
    (tmp_path / "SARON.csv").write_text("Date;Value\n2020-01;-0.70\n2020-02;-0.72\n2023-06;1.70\n")
    d = load_market(WORLD, data_dir=tmp_path)
    assert d.savings_rate.iloc[-1] == pytest.approx(1.017 ** (1 / 12) - 1)
    assert len(d.savings_rate) == 3


def test_too_short_history(tmp_path):
    _write(tmp_path, ASSETS[WORLD], np.linspace(100, 110, 300))
    with pytest.raises(ValueError):
        load_market(WORLD, data_dir=tmp_path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_market(WORLD, data_dir=tmp_path)


def test_snb_export_with_several_series_keeps_saron(tmp_path):
    _write(tmp_path, ASSETS[WORLD], 100 * np.cumprod(np.full(1200, 1.0003)))
    (tmp_path / "SARON.csv").write_text('"CubeId";"zimoma"\n"PublishingDate";"2026-11-02"\n\n'
                                        '"Date";"D0";"Value"\n"2023-06";"SARON";"1.70"\n"2023-06";"1TGT";"1.40"\n'
                                        '"2023-07";"SARON";"1.72"\n"2023-07";"1TGT";"1.45"\n')
    d = load_market(WORLD, data_dir=tmp_path)
    assert len(d.savings_rate) == 2
    assert d.savings_rate.iloc[0] == pytest.approx(1.017 ** (1 / 12) - 1)
