"""Validation (guideline 6): hand calculation, independent checks, edge cases."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import dblquad, quad
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import (backtest, ce_exact, ce_taylor, check_definiteness, cov_matrix,  # noqa: E402
                   expected_wealth, gamma_bounds, mean_cont, merton_share,
                   newton_gamma_threshold, second_moment_cont, simulate, to_continuous,
                   variance)

# Hand example (VALIDATION.md): W = 12'000, T = 2, r = 0, g = 1.01, sigma = 4 %
W, T, R0, G = 12_000, 2, 0.0, 1.01
H = G ** 2 + 0.04 ** 2   # 1.0217


def test_hand_example_expectations():
    assert expected_wealth(W, 1, T, R0, G) == pytest.approx(12_241.20)
    assert expected_wealth(W, 2, T, R0, G) == pytest.approx(12_180.60)


def test_hand_example_covariance_matrix():
    C = cov_matrix(2, T, G, H)
    assert C[0, 0] == pytest.approx(1.0217 ** 2 - 1.01 ** 4)      # 0.0032669
    assert C[0, 1] == pytest.approx(1.01 * 1.0217 - 1.01 ** 3)     # 0.0016160
    assert C[1, 1] == pytest.approx(1.0217 - 1.01 ** 2)            # 0.0016000
    assert C[0, 1] == pytest.approx(C[1, 0])


def test_hand_example_variances():
    assert variance(W, 1, T, R0, G, H) == pytest.approx(470_430.72, rel=1e-9)
    assert variance(W, 2, T, R0, G, H) == pytest.approx(291_559.68, rel=1e-9)


def test_covariance_matrix_is_positive_definite():
    rep = check_definiteness(cov_matrix(12, 60, 1.008, 1.008 ** 2 + 0.043 ** 2))
    assert rep.ok and all(m > 0 for m in rep.leading_minors)


@pytest.mark.parametrize("N,T", [(1, 12), (6, 24), (12, 60)])
def test_closed_forms_match_monte_carlo(N, T):
    g, h, r = 1.008, 1.008 ** 2 + 0.043 ** 2, 0.0004
    ls, dca = simulate(10_000, N, T, r, g, h, n_paths=400_000, seed=11)
    assert dca.mean() == pytest.approx(expected_wealth(10_000, N, T, r, g), rel=0.005)
    assert dca.var() == pytest.approx(variance(10_000, N, T, r, g, h), rel=0.03)


def test_lump_sum_special_case():
    assert expected_wealth(W, 1, 24, 0.001, 1.008) == pytest.approx(W * 1.008 ** 24)


def test_zero_volatility_gives_zero_variance():
    assert variance(W, 6, 24, 0.001, 1.008, 1.008 ** 2) == pytest.approx(0, abs=1e-6)


def test_equal_return_and_cash_rate_give_equal_expectations():
    g = 1.004
    vals = [expected_wealth(W, N, 24, g - 1, g) for N in (1, 3, 6, 12)]
    assert np.allclose(vals, vals[0])


def test_continuous_mean_matches_numerical_integral():
    mu, s2, rc = to_continuous(1.008, 1.008 ** 2 + 0.043 ** 2, 0.0004)
    tau, TT = 12.0, 60.0
    num = W / tau * quad(lambda s: np.exp(rc * s + mu * (TT - s)), 0, tau)[0]
    assert mean_cont(W, tau, TT, mu, rc) == pytest.approx(num, rel=1e-10)


def test_continuous_second_moment_matches_double_integral():
    mu, s2, rc = to_continuous(1.008, 1.008 ** 2 + 0.043 ** 2, 0.0004)
    tau, TT = 12.0, 60.0
    inner = lambda u, s: np.exp(rc * (s + u) + mu * (u - s) + (2 * mu + s2) * (TT - u))
    num = 2 * (W / tau) ** 2 * dblquad(inner, 0, tau, lambda s: s, lambda s: tau)[0]
    assert second_moment_cont(W, tau, TT, mu, s2, rc) == pytest.approx(num, rel=1e-8)


def test_continuous_limit_close_to_monthly_model():
    g, h, r = 1.008, 1.008 ** 2 + 0.043 ** 2, 0.0004
    mu, s2, rc = to_continuous(g, h, r)
    # N monthly tranches are spread over N − 1 months (first one today): compare with τ = N − 1
    disc = expected_wealth(W, 12, 60, r, g)
    cont = mean_cont(W, 11.0, 60, mu, rc)
    assert cont == pytest.approx(disc, rel=0.001)


def test_taylor_ce_close_to_exact_on_short_horizon():
    ls, _ = simulate(W, 1, 12, 0.0004, 1.008, 1.008 ** 2 + 0.043 ** 2, n_paths=200_000, seed=2)
    assert ce_taylor(ls.mean(), ls.var(), 4) == pytest.approx(ce_exact(ls, 4), rel=0.005)


def test_merton_share_is_where_derivative_vanishes():
    g, h, r, gm = 1.008, 1.008 ** 2 + 0.043 ** 2, 0.0004, 5.0
    mu, s2, rc = to_continuous(g, h, r)
    x = merton_share(g, h, r, gm)
    deriv = (mu - rc) - gm * x * s2
    assert deriv == pytest.approx(0, abs=1e-12)


def test_newton_threshold_matches_bisection_and_lies_between_bounds():
    g, h, r = 1.0081, 1.0081 ** 2 + 0.0429 ** 2, 0.005 / 12
    ls, dca = simulate(20_000, 12, 60, r, g, h, n_paths=50_000, seed=3)
    lo, hi = gamma_bounds(g, h, r)
    res = newton_gamma_threshold(ls, dca, start=hi)
    f = lambda gm: ce_exact(ls, gm) - ce_exact(dca, gm)
    ref = brentq(f, 1.0, 20.0)
    assert res.gamma == pytest.approx(ref, abs=1e-4)
    assert lo < res.gamma < hi


def test_backtest_by_hand():
    # prices 100 → 50 → 100, N = 2, no interest:
    # DCA buys half at 100 and half at 50 → value = 0.5·100/100 + 0.5·100/50 = 1.5 → +50 %
    p = pd.Series([100.0, 50.0, 100.0] + [100.0] * 12,
                  index=pd.date_range("2020-01-31", periods=15, freq="ME"))
    bt = backtest(p, 2, 0.0)
    assert bt.iloc[0, 0] == pytest.approx(0.5)
    assert bt.iloc[1, 0] == pytest.approx(0.5 * 1 + 0.5 * 50 / 100 - 1)


@pytest.mark.parametrize("args", [(-5, 3, 12), (1000, 0, 12), (1000, 13, 12)])
def test_invalid_inputs(args):
    w, n, t = args
    with pytest.raises(ValueError):
        expected_wealth(w, n, t, 0.0, 1.01)


def test_negative_gamma_rejected():
    with pytest.raises(ValueError):
        ce_taylor(100, 10, -1)


def test_fee_per_purchase_by_hand():
    # 12'000 in 2 instalments, fee 10 per purchase: invested amounts 5'990 and 5'990
    assert expected_wealth(W, 2, T, R0, G, fee=10) == pytest.approx(5_990 * 1.01 ** 2 + 5_990 * 1.01)
    # with fees, spreading out costs more: N fees instead of one
    gap_no_fee = expected_wealth(W, 1, 24, 0, 1.008) - expected_wealth(W, 12, 24, 0, 1.008)
    gap_fee = expected_wealth(W, 1, 24, 0, 1.008, fee=9) - expected_wealth(W, 12, 24, 0, 1.008, fee=9)
    assert gap_fee > gap_no_fee


def test_fee_larger_than_instalment_rejected():
    with pytest.raises(ValueError):
        expected_wealth(1_000, 10, 24, 0.0, 1.01, fee=150)


def test_bootstrap_mean_matches_closed_form():
    from model import bootstrap_growth, wealth
    rng = np.random.default_rng(5)
    hist = rng.normal(0.008, 0.043, 140)          # stand-in for 140 monthly returns
    g, h = 1 + hist.mean(), np.mean((1 + hist) ** 2)
    G = bootstrap_growth(hist, 24, n_paths=300_000, seed=9)
    assert wealth(G, W, 6, 0.0004).mean() == pytest.approx(expected_wealth(W, 6, 24, 0.0004, g), rel=0.003)
    assert wealth(G, W, 6, 0.0004).var() == pytest.approx(variance(W, 6, 24, 0.0004, g, h), rel=0.03)


def test_backtest_with_fee_by_hand():
    p = pd.Series([100.0, 50.0, 100.0] + [100.0] * 12, index=pd.date_range("2020-01-31", periods=15, freq="ME"))
    bt = backtest(p, 2, 0.0, W=1_000, fee=10)
    # DCA: 490 at 100 and 490 at 50 → 490 + 980 = 1'470 ; lump sum: 990 → ratio − 1
    assert bt.iloc[0, 0] == pytest.approx(1_470 / 990 - 1)


def test_no_tipping_point_is_reported_instead_of_cycling():
    from model import bootstrap_growth, wealth
    # constant +0.8 % per month, no risk: investing now beats waiting for every γ
    G = bootstrap_growth(np.full(140, 0.008), 120, n_paths=2_000, seed=1)
    res = newton_gamma_threshold(wealth(G, 20_000, 1, 0.0004), wealth(G, 20_000, 12, 0.0004))
    assert res.gamma is None and res.winner_if_none == "all now"


def test_tail_factor_lognormal_matches_simulation():
    from model import simulate_growth, tail_ce_factor
    g, h = 1.008, 1.008 ** 2 + 0.043 ** 2
    B = simulate_growth(36, g, h, n_paths=400_000, seed=4)[:, 0]
    for gm in (1.0, 4.0, 8.0):
        assert tail_ce_factor(gm, 36, g, h) == pytest.approx(ce_exact(B, gm), rel=0.01)


def test_tail_factor_bootstrap_matches_simulation():
    from model import bootstrap_growth, tail_ce_factor
    hist = np.random.default_rng(0).normal(0.008, 0.045, 140)
    B = bootstrap_growth(hist, 24, n_paths=300_000, seed=5)[:, 0]
    assert tail_ce_factor(4.0, 24, returns=hist) == pytest.approx(ce_exact(B, 4.0), rel=0.01)


def test_ce_factorises_over_the_buying_period():
    """CE of 10-year wealth = CE of wealth after the buying period × CE of the common tail."""
    from model import simulate_growth, tail_ce_factor, wealth
    g, h, r, W, N, T = 1.0084, 1.0187, 0.0004, 20_000, 12, 120
    G = simulate_growth(T, g, h, n_paths=200_000, seed=6)
    F = G[:, :N] / G[:, [N]]                       # growth up to month N only, same paths
    for n in (1, N):
        full = ce_exact(wealth(G, W, n, r), 4.0)
        split = ce_exact(wealth(F, W, n, r), 4.0) * tail_ce_factor(4.0, T - N, g, h)
        assert split == pytest.approx(full, rel=0.03)


def test_tipping_point_does_not_depend_on_the_horizon():
    from model import simulate_growth, wealth
    g, h, r = 1.0084, 1.0187, 0.0004
    F = simulate_growth(12, g, h, n_paths=100_000, seed=7)
    x_ls, x_dca = wealth(F, 20_000, 1, r), wealth(F, 20_000, 12, r)
    base = newton_gamma_threshold(x_ls, x_dca).gamma
    for k in (0.5, 3.0):                           # multiplying both plans by any common factor
        assert newton_gamma_threshold(k * x_ls, k * x_dca).gamma == pytest.approx(base, abs=1e-6)
