# Validation (guideline 6)

All checks below are automated in `tests/` (`python -m pytest -q tests` → 38 tests).

## 1. Hand calculation

W = 12,000 CHF, horizon T = 2 months, N = 2 instalments, savings rate r = 0, g = E[1+R] = 1.01, monthly σ = 4 %, so h = E[(1+R)²] = g² + σ² = 1.0217.

**Expected values**

- All now (N = 1): E = 12,000 × 1.01² = **12,241.20**
- Two instalments: E = 6,000 × 1.01² + 6,000 × 1.01 = 6,120.60 + 6,060.00 = **12,180.60**

**Covariance matrix** C_jk = g^|k−j| · h^(T−max(j,k)) − g^(2T−j−k)

- C₀₀ = h² − g⁴ = 1.04387 − 1.04060 = 0.0032669
- C₀₁ = C₁₀ = g·h − g³ = 1.031917 − 1.030301 = 0.0016160
- C₁₁ = h − g² = 1.0217 − 1.0201 = 0.0016000

Sylvester: 0.0032669 > 0 and det C = 0.0032669 × 0.0016 − 0.001616² = 0.00000261 > 0 → positive definite.

**Variances**

- All now: Var = 12,000² × C₀₀ = **470,430.72** (sd 685.88)
- Two instalments: a = (6,000; 6,000), Var = a'Ca = 6,000² × (C₀₀ + 2C₀₁ + C₁₁) = 36,000,000 × 0.0080989 = **291,559.68** (sd 539.96)

Spreading costs 60.60 CHF of expected value and cuts the standard deviation by 21 %. ✅ The code returns the same numbers.

## 2. Independent checks

| What | Checked against | Tolerance | Result |
|---|---|---|---|
| Closed-form E[W_T] and Var[W_T] (N = 1, 6, 12) | Monte Carlo, 400,000 paths | 0.5 % / 3 % | ✅ |
| Continuous E[W_T] (integral, closed form) | `scipy.integrate.quad` | 10⁻¹⁰ | ✅ |
| Continuous E[W_T²] (double integral, closed form) | `scipy.integrate.dblquad` | 10⁻⁸ | ✅ |
| Continuous model vs monthly model (τ = N − 1) | each other | 0.1 % | ✅ |
| Newton's tipping point γ* | bisection (`scipy.optimize.brentq`) on the same paths | 10⁻⁴ | ✅ |
| CE(W_T) = CE(X_N) · CE(B) | certainty equivalent of the full 10-year simulation | 3 % | ✅ |
| Exact CE(B), lognormal and bootstrap | simulation, 300,000–400,000 paths | 1 % | ✅ |
| γ* unchanged when both plans are scaled by a common factor (any horizon) | each other | 10⁻⁶ | ✅ |
| Bootstrap simulation, E and Var | closed forms with the sample g and h | 0.3 % / 3 % | ✅ |
| Fee per purchase (2 instalments, CHF 10) | by hand: 5,990 × 1.01² + 5,990 × 1.01 | exact | ✅ |
| Backtest with fee | by hand: 1,470 / 990 − 1 | exact | ✅ |
| Merton share x* | derivative = 0 | 10⁻¹² | ✅ |
| Taylor CE (1-year horizon) | exact CRRA CE by simulation | 0.5 % | ✅ |
| Backtest | prices 100 → 50 → 100 by hand: DCA = +50 % | exact | ✅ |

## 3. Edge cases and invalid inputs

- N = 1 gives exactly the lump sum W·g^T.
- σ = 0 gives a variance of 0.
- Savings rate equal to the expected return gives the same expected value for every N.
- A −20 % crash in the first month: spreading wins in 100 % of paths (What if tab).
- No crossing between the two certainty equivalents (e.g. no risk at all): Newton reports "no tipping point, all now always wins" instead of looping.
- An SNB export holding several series: only the SARON rows are kept.
- Refused with a clear message: amount ≤ 0, N > horizon, N < 1, fee ≥ one instalment, h < g², γ < 0, missing data file, fewer than 36 months of data.

## 4. Real data (download of 6 Nov 2026, MSCI World in CHF, window Jan 2015 – Oct 2026)

- g = 1.00841, h = 1.01872 → about 10.1 % mean return and 14.9 % volatility per year.
- Bounds on the tipping point: γ_low = 4.44, γ_high = 8.88. Newton (N = 12) goes 6.750 → 6.531 → 6.532 and stops at **γ* = 6.53**, between the bounds. Across 6 different random seeds: 6.50 to 6.58.
- Backtest since 2010, 191 start months, waiting money earning the real monthly SARON: spreading over 12 months did better in **29 %** of cases, −3.1 % on average, best +17.6 % (start May 2010), worst −18.8 % (start Oct 2020). With a constant 0.5 % instead of the SARON: 30 % and −2.8 %. Consistent with Vanguard's finding that lump sum wins about two thirds of the time.

## 5. Robustness of the main result (CHF 20,000, 12 months, 10 years)

| Variant | Tipping point γ* |
|---|---|
| Baseline: MSCI World in CHF, lognormal, no fee | 6.5 |
| CHF 9 fee per purchase (cost of spreading 2,579 instead of 2,321) | 7.3 |
| Swiss equities (SPI) instead of MSCI World | 8.4 |
| Real returns redrawn (historical bootstrap) | 6.2 |

The tipping point stays between 6 and 8.5 in every variant, always far above a "Balanced" investor. Every variant supports the same advice for a "Balanced" investor (γ = 4): invest everything now.

## 6. Precision of the tipping point (correction of 26 Nov 2026)

A first version simulated the full 10 years on 20,000 paths and found γ* = 6.1, and no tipping point at all with the bootstrap. Comparing with ChatGPT (γ* = 6.52) made us check: with other random seeds that version gave anything from 5.4 to 7.2, and "no tipping point" in some bootstrap runs. The threshold depends on rare, very bad paths, so a 10-year simulation estimates it badly.

The fix is mathematical. After the last instalment both plans hold the same asset, so W_T = X_N · B, where X_N is the wealth at the end of the buying period and B the growth afterwards, independent of X_N. For CRRA utility, E[(X_N B)^(1−γ)] = E[X_N^(1−γ)] · E[B^(1−γ)], so CE(W_T) = CE(X_N) · CE(B). CE(B) has an exact formula, and it multiplies both plans by the same factor: **the winner and the tipping point depend only on the buying period, whatever the horizon.** The app now simulates only the first 24 months (200,000 paths) and multiplies by the exact CE(B). The spread of γ* across seeds fell from about 0.3 to 0.03.

## 7. Where the simple method fails

At a 10-year horizon the Taylor approximation says spreading is better for γ = 4, while the exact certainty equivalent says investing everything now is better. Taylor only uses mean and variance; when final wealth is very spread out (long horizons), that is not enough. The app therefore recommends on the exact certainty equivalent and shows the Taylor error.
