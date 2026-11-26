"""The app runs end to end on the real data and handles invalid inputs (guideline 5)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
HAS_DATA = (Path(APP).parent / "data" / "ETF_MONDE.csv").exists()
pytestmark = pytest.mark.skipif(not HAS_DATA, reason="market data not downloaded")


def widget(at, label):
    for w in [*at.number_input, *at.selectbox, *at.slider, *at.radio]:
        if w.label.startswith(label):
            return w
    raise KeyError(label)


def run(**changes):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    for label, value in changes.items():
        widget(at, label).set_value(value)
    at.run()
    return at


def verdict(at):
    return next(m.value for m in at.markdown if "class='verdict" in m.value)


def test_default_run_recommends_all_now():
    at = run()
    assert not at.exception
    assert "verdict-now" in verdict(at) and "all now." in verdict(at)


def test_cautious_short_horizon_recommends_spreading():
    at = run(**{"Attitude to risk": "Cautious", "Horizon": 1})
    assert not at.exception
    assert "verdict-step" in verdict(at) and "twelve months." in verdict(at)


def test_spread_longer_than_horizon_is_refused():
    at = run(**{"Horizon": 1, "Spread over": 24})
    assert any("cannot spread" in e.value for e in at.error)


def test_zero_amount_is_refused():
    at = run(**{"Amount received": 0})
    assert any("amount" in e.value.lower() for e in at.error)


def test_bootstrap_and_fee_run_and_keep_the_verdict():
    at = run(**{"Simulated futures": "Historical bootstrap", "Fee per purchase": 9.0})
    assert not at.exception
    assert "verdict-now" in verdict(at) and "all now." in verdict(at)


def test_fee_larger_than_instalment_is_refused():
    at = run(**{"Amount received": 1000, "Fee per purchase": 100.0})
    assert any("larger than one instalment" in e.value for e in at.error)
