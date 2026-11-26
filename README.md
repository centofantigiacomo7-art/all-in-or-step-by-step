# All In or Step by Step? Timing the Investment of a Windfall

**For a young graduate who receives a windfall (a first bonus, a gift, an inheritance), the tool helps decide whether to invest it all at once or spread it over N months. It uses real Swiss and world market data and an expected value / variance model with a certainty equivalent, and finds the risk aversion above which spreading out is worth it.**

Project — Mathematics for Economics and Finance, HEC Lausanne, Fall 2026.

The asset is fixed (a world equity ETF in CHF, or Swiss equities). The only decision is **when** to invest.

## 0. The need and what already exists (guideline 2)

- **Evidence:** Vanguard (Finlay & Zorn, *Cost averaging: Invest now or temporarily hold your cash?*, Feb 2023) finds that over 1976–2022 investing a lump sum beat cost averaging about 68 % of the time, and that cost averaging may only suit investors with very high aversion to risk and losses ([PDF](https://corporate.vanguard.com/content/dam/corp/research/pdf/cost_averaging_invest_now_or_temporarily_hold_your_cash.pdf)). The study is in USD and its utility model uses three investor personas.
- **Existing tools:**
  - [Liberated Stock Trader — DCA vs lump sum calculator](https://www.liberatedstocktrader.com/dca-vs-lump-sum-investing-calculator/): quick and clear, includes the "cash drag" of the waiting money; but it works from an assumed return, with no risk measure and no personal risk aversion.
  - [Freedom Isn't Free — drip feed vs lump sum](https://freedomisntfree.co.uk/tools/drip-feed-vs-lump-sum): simulates both strategies and shows the median and a 10th–90th percentile range; but in GBP, no personal risk aversion and no historical backtest.
- **What our tool adds:** results in CHF with Swiss data, the user's own risk aversion (certainty equivalent), the tipping point γ*, fees per purchase, a backtest on every start month since 2010, and a robustness check with real returns.

## 1. How to run it (macOS)

```bash
cd ~/"Desktop/HEC LAUSANNE/02 MScF – Semestre 1 (automne 2026)/Mathematics for Economics and Finance/Projet All In or Step by Step"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py              # opens http://localhost:8501
```

The market data is already in `data/` (Yahoo Finance download of 6 Nov 2026, SNB SARON file published 2 Nov 2026). To refresh the prices: `python fetch_data.py`; to refresh the SARON, download https://data.snb.ch/api/cube/zimoma/data/csv/en again and save it as `data/SARON.csv`.

## 2. Data (guideline 4)

| File | Series | Source | Role |
|---|---|---|---|
| `data/ETF_MONDE.csv` | iShares Core MSCI World (`SWDA.SW`), USD | Yahoo Finance | Main asset, converted to CHF |
| `data/USDCHF.csv` | CHF per 1 USD (`CHF=X`) | Yahoo Finance | USD → CHF conversion |
| `data/CHSPI.csv` | iShares Core SPI ETF (CH), CHF | Yahoo Finance | Alternative asset (Swiss equities) |
| `data/SARON.csv` | SARON, monthly, % per year (SNB cube `zimoma`, published 2 Nov 2026) | SNB data portal | Rate earned by the money waiting to be invested, in the backtest |
| `data/sources.json` | Ticker, URL, retrieval date, period, currency | written by `fetch_data.py` | Provenance |

- **Field used:** `Adj Close` (dividends included). USD prices × CHF/USD of the same day.
- **Transformations:** month-end price → simple return `P_t / P_(t−1) − 1` → `g = mean(1+R)`, `h = mean((1+R)²)`. The incomplete current month is removed.
- **SARON:** the SNB file holds 14 money-market rates; only the 658 SARON rows are kept (329 usable monthly values, Jun 1999 – Oct 2026), converted to a monthly rate `(1 + R/100)^(1/12) − 1`. The backtest uses the real SARON of each month; the forward-looking simulation uses the savings rate set in the app (0.5 % by default, since the latest SARON is slightly negative, −0.06 % in Oct 2026, while savings accounts do not charge interest).
- **Observed vs calculated vs hypothetical:** prices and exchange rates are observations; returns, moments, expected values, variances and the backtest are calculations; simulated futures and What if scenarios are hypothetical.

## 3. Mathematical method (guideline 3)

Tranche k (k = 0 … N−1) waits k months on the savings account at rate r, then is invested until the horizon T.

- **Terminal wealth:** `W_T = Σ a_k G_k`, `a_k = (W/N)(1+r)^k`, `G_k = Π_(t=k+1..T)(1+R_t)`
- **Expectation:** `E[W_T] = Σ a_k g^(T−k)`; continuous version as an integral: `E[W_T] = (W/τ)∫₀^τ e^(rs) e^(μ(T−s)) ds`
- **Variance as a quadratic form:** `Var[W_T] = aᵀCa`, `C_jk = g^|k−j| h^(T−max(j,k)) − g^(2T−j−k)`; C is checked to be positive definite (Sylvester).
- **Certainty equivalent (Taylor, order 2):** `CE ≈ E − (γ/2)·Var/E`; compared with the exact CRRA certainty equivalent from simulation.
- **Exact certainty equivalent:** once the last instalment is in, both plans hold the same asset, so `W_T = X_N · B` with B independent of X_N and, for CRRA utility, `CE(W_T) = CE(X_N) · CE(B)`. CE(B) has a closed form (lognormal) or an exact sample formula (bootstrap); only the buying period is simulated (200,000 paths), which makes the tipping point precise (±0.03).
- **Best permanent share in equities:** derivative of `x(μ−r) − (γ/2)x²σ²` set to 0 → `x* = (μ−r)/(γσ²)`.
- **Tipping point γ*:** safeguarded Newton's method on `f(γ) = CE_all now(γ) − CE_N(γ)` on fixed simulated paths (a grid first checks that the curves cross; a step leaving the bracket becomes a bisection). For lognormal returns it lies between `γ_low = (μ−r)/σ²` and `γ_high = 2(μ−r)/σ²`.
- **Fees:** a fixed fee F per purchase is taken from every instalment (`a_k = (W/N)(1+r)^k − F`), so spreading pays it N times.
- **Two ways to simulate the future:** lognormal returns with the mean and variance of the data, or a historical bootstrap that redraws real monthly returns (fat tails kept).
- **Backtest:** every start month since 2010, real prices. Once fully invested both plans hold the same asset, so the winner is decided in the first N months.

**Main result:** for the default case (CHF 20,000, 12 months, 10 years, MSCI World in CHF since 2015) spreading out only pays for investors more risk-averse than γ* ≈ 6.5 — who should not hold 100 % equities in the first place. With a CHF 9 fee per purchase γ* rises to 7.3; with real returns redrawn (bootstrap) it is 6.2. Because both plans hold the same asset once the last instalment is in, the tipping point depends only on the buying period, not on the horizon. Spreading out is "just taking risk later" (Vanguard).

## 4. Validation (guideline 6)

`python -m pytest -q tests` → 38 tests. Details, hand calculation and real-data figures in `VALIDATION.md`.

## 5. Files

```
app.py          Streamlit interface (English)
style.py        visual identity: palette, typography, chart template, hero and plan timeline
.streamlit/     app theme
model.py        closed forms, covariance matrix, certainty equivalents, Newton, simulation, backtest
data.py         loading, cleaning, currency conversion, SARON, provenance
fetch_data.py   data download (run on your own computer)
tests/          automated validation (model, data, app)
VALIDATION.md   hand calculation and checks
AI_COMPARISON.md  prompt and answer grid for the comparison with an AI (guideline 7)
```

## 6. Limitations

Independent monthly returns (neither simulation keeps the order of real months, so long crashes are underestimated); past returns depend on the window; Taylor ignores the asymmetry of losses and fails over long horizons; only a fixed fee per purchase (no stamp duty or taxes); regret is not modelled. Not financial advice.

## 7. Use of AI (to disclose in the presentation appendix)

Claude (Anthropic) was used to structure the code, build the interface and write the tests. The group is responsible for the formulas, assumptions and the hand verification.
