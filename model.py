"""
All In or Step by Step? — mathematical core.

A windfall W is invested in N equal monthly tranches (N = 1 is the lump sum).
Tranche k (k = 0..N-1) waits k months on a savings account (monthly rate r),
then is invested in an equity ETF until the horizon T (months).

Monthly gross returns 1+R_t are i.i.d. with
    g = E[1+R],   h = E[(1+R)^2] = g^2 + sigma^2.

Terminal wealth:  W_T = sum_k a_k G_k,  a_k = (W/N)(1+r)^k,  G_k = prod_{t=k+1..T} (1+R_t)
Expectation:      E[W_T] = sum_k a_k g^(T-k)
Variance:         Var[W_T] = a' C a,  C_jk = g^|k-j| h^(T-max(j,k)) - g^(2T-j-k)
Certainty equivalent (2nd-order Taylor of a CRRA utility around E[W_T]):
                  CE ~ E[W_T] - (gamma/2) Var[W_T] / E[W_T]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RISK_PROFILES = {"Cautious": 10.0, "Balanced": 4.0, "Bold": 1.5}   # CRRA relative risk aversion gamma


# --------------------------------------------------------------------------
# 0. Input checks
# --------------------------------------------------------------------------
def check_inputs(W: float, N: int, T: int, g: float, h: float, r: float, fee: float = 0.0) -> None:
    if W <= 0:
        raise ValueError("The amount must be positive.")
    if not (isinstance(N, (int, np.integer)) and N >= 1):
        raise ValueError("The number of instalments N must be a whole number ≥ 1.")
    if T < N:
        raise ValueError("The horizon must be at least as long as the investment period (T ≥ N).")
    if g <= 0 or h < g * g - 1e-15:
        raise ValueError("Invalid return moments: we need g > 0 and h ≥ g² (variance ≥ 0).")
    if r <= -1:
        raise ValueError("The savings rate must be above −100%.")
    if fee < 0 or fee >= W / N:
        raise ValueError("The fee per purchase must be ≥ 0 and smaller than one instalment.")


# --------------------------------------------------------------------------
# 1. Discrete model (closed forms)
# --------------------------------------------------------------------------
def tranche_weights(W: float, N: int, r: float, fee: float = 0.0) -> np.ndarray:
    """a_k = (W/N)(1+r)^k − F : amount of tranche k actually invested (after a fixed fee F per purchase)."""
    return W / N * (1 + r) ** np.arange(N) - fee


def expected_wealth(W, N, T, r, g, fee=0.0) -> float:
    check_inputs(W, N, T, g, g * g, r, fee)
    k = np.arange(N)
    return float(np.sum(tranche_weights(W, N, r, fee) * g ** (T - k)))


def cov_matrix(N, T, g, h) -> np.ndarray:
    """C_jk = Cov(G_j, G_k): tranches share the months in which both are invested."""
    j = np.arange(N)[:, None]
    k = np.arange(N)[None, :]
    return g ** np.abs(k - j) * h ** (T - np.maximum(j, k)) - g ** (2 * T - j - k)


def variance(W, N, T, r, g, h, fee=0.0) -> float:
    check_inputs(W, N, T, g, h, r, fee)
    a = tranche_weights(W, N, r, fee)
    return float(a @ cov_matrix(N, T, g, h) @ a)


@dataclass
class Definiteness:
    leading_minors: list[float]
    min_eigenvalue: float
    ok: bool


def check_definiteness(C: np.ndarray, tol: float = 1e-12) -> Definiteness:
    """Sylvester's criterion (leading principal minors) + smallest eigenvalue."""
    n = C.shape[0]
    minors = [float(np.linalg.det(C[:m, :m])) for m in range(1, min(n, 6) + 1)]
    eig = float(np.linalg.eigvalsh(C).min())
    return Definiteness(minors, eig, eig > -tol * max(1.0, abs(C).max()))


def ce_taylor(E: float, V: float, gamma: float) -> float:
    """CE ≈ E − (A/2)·Var with absolute risk aversion A = γ/E (CRRA, Taylor order 2)."""
    if gamma < 0:
        raise ValueError("Risk aversion γ cannot be negative.")
    return E - 0.5 * gamma * V / E


def strategy_table(W, T, r, g, h, gamma, n_max=24, fee=0.0) -> pd.DataFrame:
    """E, sd and CE for N = 1..n_max (only N ≤ T and instalments larger than the fee)."""
    rows = []
    for N in range(1, min(n_max, T) + 1):
        if fee >= W / N:
            break
        E = expected_wealth(W, N, T, r, g, fee)
        V = variance(W, N, T, r, g, h, fee)
        rows.append({"N": N, "E": E, "sd": np.sqrt(max(V, 0)), "CE": ce_taylor(E, V, gamma)})
    return pd.DataFrame(rows).set_index("N")


def optimal_n(W, T, r, g, h, gamma, n_max=24, fee=0.0) -> int:
    t = strategy_table(W, T, r, g, h, gamma, n_max, fee)
    return int(t["CE"].idxmax())


# --------------------------------------------------------------------------
# 2. Continuous-time version (integrals) and the risk-aversion threshold
# --------------------------------------------------------------------------
def to_continuous(g: float, h: float, r: float) -> tuple[float, float, float]:
    """Lognormal returns with the same first two moments, per month:
    E[S_t/S_0] = e^(mu t), Var log = s2 t, cash grows at e^(rc t)."""
    mu = np.log(g)
    s2 = np.log(h / g ** 2)
    rc = np.log(1 + r)
    return float(mu), float(s2), float(rc)


def _expm1_ratio(x: float, t: float) -> float:
    """(e^(x t) − 1) / x, stable when x → 0."""
    return t if abs(x) < 1e-12 else float(np.expm1(x * t) / x)


def mean_cont(W, tau, T, mu, rc) -> float:
    """E[W_T] = (W/τ) ∫_0^τ e^(r s) e^(μ (T−s)) ds = W e^(μT) (1 − e^(−(μ−r)τ)) / ((μ−r) τ)."""
    if tau <= 1e-9:
        return W * np.exp(mu * T)
    return W / tau * np.exp(mu * T) * _expm1_ratio(rc - mu, tau)


def second_moment_cont(W, tau, T, mu, s2, rc) -> float:
    """E[W_T^2] = 2 (W/τ)^2 ∫_0^τ ∫_s^τ e^(r(s+u)) e^(μ(u−s)) e^((2μ+σ²)(T−u)) du ds  (closed form)."""
    if tau <= 1e-9:
        return W ** 2 * np.exp((2 * mu + s2) * T)
    k = rc - mu - s2
    a = rc - mu
    inner = (np.exp(k * tau) * _expm1_ratio(a, tau) - _expm1_ratio(a + k, tau))
    inner = inner / k if abs(k) > 1e-12 else None
    if inner is None:  # k = 0 limit, integrate numerically
        from scipy.integrate import quad
        inner = quad(lambda s: np.exp(a * s) * (tau - s), 0, tau)[0]
    return 2 * (W / tau) ** 2 * np.exp((2 * mu + s2) * T) * inner


def ce_cont(W, tau, T, mu, s2, rc, gamma) -> float:
    E = mean_cont(W, tau, T, mu, rc)
    V = second_moment_cont(W, tau, T, mu, s2, rc) - E ** 2
    return ce_taylor(E, V, gamma)


def merton_share(g: float, h: float, r: float, gamma: float) -> float:
    """Best permanent share of wealth in equities for a CRRA investor:
    maximise x(μ − r) − (γ/2) x² σ²  →  derivative = 0  →  x* = (μ − r) / (γ σ²)."""
    mu, s2, rc = to_continuous(g, h, r)
    if gamma <= 0 or s2 <= 0:
        return float("inf")
    return (mu - rc) / (gamma * s2)


def gamma_bounds(g: float, h: float, r: float) -> tuple[float, float]:
    """Lower bound: γ_low = (μ − r)/σ² (x* = 100 %): below it, any delay lowers utility.
    Upper bound: γ_high = 2(μ − r)/σ²: above it, each month in equities is worth less than
    a month in cash (μ − γσ²/2 < r). The exact break-even γ* lies in between."""
    mu, s2, rc = to_continuous(g, h, r)
    if s2 <= 0:
        return float("inf"), float("inf")
    return (mu - rc) / s2, 2 * (mu - rc) / s2


# --------------------------------------------------------------------------
# 3. Simulation (exact CE, probability that DCA wins)
# --------------------------------------------------------------------------
def simulate_growth(T, g, h, n_paths=20000, seed=1, shock_first_month=0.0) -> np.ndarray:
    """G[:, k] = growth of 1 CHF invested at the start of month k until T (lognormal, same moments g, h)."""
    s2 = np.log(h / g ** 2)
    m = np.log(g) - s2 / 2
    rng = np.random.default_rng(seed)
    logret = rng.normal(m, np.sqrt(s2), size=(n_paths, T))
    if shock_first_month:
        logret[:, 0] = np.log1p(shock_first_month)
    return np.exp(np.cumsum(logret[:, ::-1], axis=1)[:, ::-1])


def bootstrap_growth(returns: np.ndarray, T, n_paths=20000, seed=1, shock_first_month=0.0) -> np.ndarray:
    """Same as simulate_growth, but each month's return is drawn at random (with replacement)
    from the real historical monthly returns: no normality assumption, fat tails kept."""
    rng = np.random.default_rng(seed)
    x = np.log1p(np.asarray(returns, float))
    logret = rng.choice(x, size=(n_paths, T), replace=True)
    if shock_first_month:
        logret[:, 0] = np.log1p(shock_first_month)
    return np.exp(np.cumsum(logret[:, ::-1], axis=1)[:, ::-1])


def wealth(G: np.ndarray, W: float, N: int, r: float, fee: float = 0.0) -> np.ndarray:
    """Terminal wealth of an N-tranche plan on simulated growth paths: W_T = G[:, :N] · a."""
    return G[:, :N] @ tranche_weights(W, N, r, fee)


def simulate(W, N, T, r, g, h, n_paths=20000, seed=1, shock_first_month=0.0, fee=0.0):
    """Terminal wealth of lump sum and DCA on the SAME paths."""
    check_inputs(W, N, T, g, h, r, fee)
    G = simulate_growth(T, g, h, n_paths, seed, shock_first_month)
    return wealth(G, W, 1, r, fee), wealth(G, W, N, r, fee)


def ce_exact(wealth: np.ndarray, gamma: float) -> float:
    """Certainty equivalent of a CRRA utility, computed on simulated wealth."""
    w = np.asarray(wealth, float)
    if abs(gamma - 1) < 1e-9:
        return float(np.exp(np.mean(np.log(w))))
    scale = np.median(w)
    x = (w / scale) ** (1 - gamma)
    return float(scale * np.mean(x) ** (1 / (1 - gamma)))


def tail_ce_factor(gamma: float, months: int, g: float | None = None, h: float | None = None,
                   returns: np.ndarray | None = None) -> float:
    """Certainty equivalent of B, the growth of 1 CHF over the months AFTER the buying period.

    Once the last instalment is in, both plans hold the same asset, so W_T = X · B with B
    independent of X. For CRRA utility E[(XB)^(1-γ)] = E[X^(1-γ)] · E[B^(1-γ)], hence
        CE(W_T) = CE(X) · CE(B).
    B is common to both plans: it scales both certainty equivalents by the same factor and
    cannot change which plan is better. CE(B) is exact:
      lognormal: log(1+R) ~ N(m, s²) → CE(B) = exp(months · (m + (1-γ) s²/2));
      bootstrap: CE(B) = E[(1+R)^(1-γ)]^(months/(1-γ)) over the historical returns."""
    if months <= 0:
        return 1.0
    if returns is not None:
        x = np.log1p(np.asarray(returns, float))
        if abs(gamma - 1) < 1e-9:
            return float(np.exp(months * x.mean()))
        c = x.max() if gamma < 1 else x.min()            # rescale to avoid overflow
        mom = np.mean(np.exp((1 - gamma) * (x - c)))
        return float(np.exp(months * (c + np.log(mom) / (1 - gamma))))
    s2 = np.log(h / g ** 2)
    m = np.log(g) - s2 / 2
    return float(np.exp(months * (m + (1 - gamma) * s2 / 2)))


@dataclass
class NewtonResult:
    gamma: float | None          # tipping point, None if the curves never cross on the grid
    iterations: int
    history: list[float]
    winner_if_none: str = ""     # "all now" or "spread" when there is no tipping point


GAMMA_GRID = np.linspace(0.5, 60, 120)


def newton_gamma_threshold(ls: np.ndarray, dca: np.ndarray, start: float | None = None,
                           tol: float = 1e-6, max_iter: int = 50) -> NewtonResult:
    """Exact break-even risk aversion: solve f(γ) = CE(lump sum; γ) − CE(DCA; γ) = 0.

    1. Scan a grid of γ for a sign change (no sign change → no tipping point, say who always wins).
    2. Safeguarded Newton inside the bracket [a, b]: γ ← γ − f(γ)/f'(γ) with f' by central
       differences; a step that leaves the bracket is replaced by a bisection step.
    The same simulated paths are used for every γ, so f is smooth."""
    f = lambda gm: ce_exact(ls, gm) - ce_exact(dca, gm)
    vals = np.array([f(gm) for gm in GAMMA_GRID])
    cross = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    if len(cross) == 0:
        return NewtonResult(None, 0, [], "all now" if vals[0] > 0 else "spread")
    a, b = float(GAMMA_GRID[cross[0]]), float(GAMMA_GRID[cross[0] + 1])
    fa = vals[cross[0]]
    gm = float(start) if start is not None and a < start < b else (a + b) / 2
    hist = [gm]
    eps = 1e-4
    for i in range(1, max_iter + 1):
        fg = f(gm)
        if np.sign(fg) == np.sign(fa):
            a, fa = gm, fg
        else:
            b = gm
        d = (f(gm + eps) - f(gm - eps)) / (2 * eps)
        new = gm - fg / d if d != 0 and np.isfinite(d) else (a + b) / 2
        if not a < new < b:
            new = (a + b) / 2                      # bisection safeguard
        hist.append(new)
        if abs(new - gm) < tol:
            return NewtonResult(new, i, hist)
        gm = new
    return NewtonResult(gm, max_iter, hist)


# --------------------------------------------------------------------------
# 4. Historical backtest
# --------------------------------------------------------------------------
def backtest(prices: pd.Series, N: int, monthly_rate: pd.Series | float,
             W: float = 1.0, fee: float = 0.0) -> pd.DataFrame:
    """For every start month s: value of DCA relative to lump sum once DCA is fully invested.

    Ratio(s) = sum_k ((W/N)·(cash growth over k months) − F) · P_s / P_(s+k) / (W − F).
    After month s+N−1 both strategies hold the same asset, so this ratio is
    also the ratio at ANY later horizon: who wins is decided in the first N months.
    """
    p = prices.dropna()
    if isinstance(monthly_rate, (int, float)):
        rate = pd.Series(float(monthly_rate), index=p.index)
    else:
        rate = monthly_rate.reindex(p.index).ffill().fillna(0.0)
    if len(p) < N + 12:
        raise ValueError("Not enough price history for this number of instalments.")
    P = p.to_numpy()
    R = rate.to_numpy()
    rows = []
    for s in range(0, len(P) - N + 1):
        cash = np.concatenate([[1.0], np.cumprod(1 + R[s:s + N - 1])])
        dca = float(np.sum((W / N * cash - fee) * P[s] / P[s:s + N]))
        ratio = dca / (W - fee)
        rows.append({"start": p.index[s], "dca_vs_lump_sum": ratio - 1})
    return pd.DataFrame(rows).set_index("start")
