"""
All In or Step by Step? Timing the Investment of a Windfall — Streamlit interface ("Maison" edition).
Run:  streamlit run app.py
"""
from fractions import Fraction

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import importlib

import style
from data import ASSETS, load_market, moments
from model import (RISK_PROFILES, backtest, bootstrap_growth, ce_exact, ce_taylor, check_definiteness,
                   cov_matrix, expected_wealth, gamma_bounds, merton_share, newton_gamma_threshold,
                   simulate_growth, strategy_table, tail_ce_factor, variance, wealth)

st.set_page_config(page_title="Windfall — all in or step by step?", page_icon="⏱", layout="wide",
                   initial_sidebar_state="collapsed")
importlib.reload(style)   # pick up style changes without restarting Streamlit
st.markdown(style.CSS, unsafe_allow_html=True)
style.register_plotly_template()
html = lambda s: st.markdown(s, unsafe_allow_html=True)


def chf(x):
    return f"CHF {x:,.0f}".replace(",", "'")


def in_n(p: float) -> str:
    """0.32 → '1 in 3' (closest fraction with a small denominator)."""
    f = Fraction(p).limit_denominator(5)
    return f"{f.numerator} in {f.denominator}" if 0 < p < 1 else f"{p:.0%}"


NUMBERS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
words = lambda n: NUMBERS[n] if n < len(NUMBERS) else str(n)

# ================================================================ header, sentence and settings
html(style.top())
sentence_slot = st.empty()
with st.expander("Change the numbers"):
    c1, c2, c3, c4 = st.columns(4)
    W = c1.number_input("Amount received (CHF)", 0, 5_000_000, 20_000, 1_000)
    N = c2.selectbox("Spread over", list(range(2, 25)), index=10, format_func=lambda n: f"{n} months")
    years = c3.selectbox("Horizon", list(range(1, 31)), index=9,
                         format_func=lambda y: f"{y} year" + ("s" if y > 1 else ""))
    profile = c4.selectbox("Attitude to risk", list(RISK_PROFILES), index=1,
                           help="Cautious: losses hurt you a lot. Bold: you accept big swings.")
    a1, a2, a3 = st.columns(3)
    gamma = a1.slider("Risk aversion γ", 0.5, 20.0, RISK_PROFILES[profile], 0.5, key=f"gamma_{profile}")
    rate_pa = a2.slider("Savings rate while waiting (%/year)", 0.0, 3.0, 0.5, 0.05) / 100
    fee = a3.number_input("Fee per purchase (CHF)", 0.0, 100.0, 0.0, 1.0,
                          help="Paid on every purchase: spreading over N months means N fees.")
    asset = a1.selectbox("Invest in", list(ASSETS))
    start_year = a2.slider("Use market data since", 2010, 2020, 2015)
    market_model = a3.radio("Simulated futures", ["Lognormal", "Historical bootstrap"], horizontal=True,
                            help="Bootstrap redraws real monthly returns and keeps their fat tails.")
n_paths = 20_000          # full 10-year paths: percentiles
n_first = 200_000         # buying period only: certainty equivalents, chances, tipping point
sentence_slot.markdown(style.sentence(chf(W), f"{N} months", f"{years} year" + ("s" if years > 1 else ""),
                                      profile.lower()), unsafe_allow_html=True)

T = 12 * years
r = (1 + rate_pa) ** (1 / 12) - 1
BOOT = market_model == "Historical bootstrap"

# ================================================================ data and checks
try:
    md = load_market(asset, start=f"{start_year}-01-01")
except (FileNotFoundError, ValueError) as err:
    st.error(f"Market data not available — {err} Run `python fetch_data.py` first (see README).")
    st.stop()
if W <= 0:
    st.error("Enter the amount you received (more than CHF 0).")
    st.stop()
if fee >= W / N:
    st.error(f"The fee of {chf(fee)} is larger than one instalment ({chf(W / N)}). Lower the fee or the number of months.")
    st.stop()
if N > T:
    st.error(f"You cannot spread the money over {N} months with a {years}-year horizon. "
             "Lower the number of months or raise the horizon.")
    st.stop()

g, h = moments(md.returns)
mu_pa, vol_pa = 12 * md.returns.mean(), md.returns.std() * np.sqrt(12)


@st.cache_data(show_spinner=False)
def growth_paths(T, g, h, n, shock=0.0, seed=1, boot=False, vol_mult=1.0):
    if boot:
        x = md.returns.to_numpy()
        x = x.mean() + (x - x.mean()) * vol_mult
        return bootstrap_growth(x, T, n, seed, shock)
    return simulate_growth(T, g, h, n, seed, shock)


def boot_returns(vol_mult):
    x = md.returns.to_numpy()
    return x.mean() + (x - x.mean()) * vol_mult


M = min(max(N, 24), T)    # months simulated in detail: the buying period of every plan compared below


def analyse(g_, h_, r_, gamma_, shock=0.0, vol_mult=1.0):
    G = growth_paths(T, g_, h_, n_paths, shock, boot=BOOT, vol_mult=vol_mult)
    F = growth_paths(M, g_, h_, n_first, shock, boot=BOOT, vol_mult=vol_mult)
    # After month M both plans hold the same asset: W_T = X_M · B, so CE(W_T) = CE(X_M) · CE(B)
    tail = (lambda gm: tail_ce_factor(gm, T - M, returns=boot_returns(vol_mult))) if BOOT \
        else (lambda gm: tail_ce_factor(gm, T - M, g_, h_))
    x_ls, x_dca = wealth(F, W, 1, r_, fee), wealth(F, W, N, r_, fee)
    ls, dca = wealth(G, W, 1, r_, fee), wealth(G, W, N, r_, fee)
    lo, hi = gamma_bounds(g_, h_, r_)
    newton = newton_gamma_threshold(x_ls, x_dca, start=hi if np.isfinite(hi) else None)
    E_ls, E_dca = expected_wealth(W, 1, T, r_, g_, fee), expected_wealth(W, N, T, r_, g_, fee)
    return dict(G=G, F=F, tail=tail, ls=ls, dca=dca,
                ce_ls=ce_exact(x_ls, gamma_) * tail(gamma_), ce_dca=ce_exact(x_dca, gamma_) * tail(gamma_),
                E_ls=E_ls, E_dca=E_dca, p_dca=float((x_dca > x_ls).mean()), cost=E_ls - E_dca,
                bad_ls=float(np.percentile(ls, 5)), bad_dca=float(np.percentile(dca, 5)),
                lo=lo, hi=hi, newton=newton, x_star=merton_share(g_, h_, r_, gamma_))


base = analyse(g, h, r, gamma)
nr = base["newton"]
dca_better = base["ce_dca"] > base["ce_ls"]
yrs_txt = f"{words(years)} year{'s' if years > 1 else ''}"

# ================================================================ verdict, plan, two figures
if not dca_better:
    html(style.verdict("Invest it", "all now.", "now"))
else:
    html(style.verdict("Spread it over", f"{words(N)} months.", "step"))
shown = max(12, N + 6) if N + 6 <= 30 else N
html(style.strip(chf(W / N), N, min(shown, T) if T >= N else N))

if not dca_better:
    html(style.figures([
        (chf(base["cost"]), f"What waiting would cost you on average, after {yrs_txt}."),
        (in_n(base["p_dca"]), "The chance that waiting ends ahead. In the other futures, investing at once does."),
    ]))
else:
    html(style.figures([
        (chf(base["cost"]), f"What waiting costs on average after {yrs_txt}: the price of a smoother ride."),
        (f"{min(base['x_star'], 1):.0%}", "Your ideal share in shares. Keeping the rest out of the market for good "
                                          "would suit you even better than waiting."),
    ]))
html(style.detail(f"After {years}\u00a0year{'s' if years > 1 else ''}", [
    ("Expected value", chf(base["E_ls"]), chf(base["E_dca"])),
    ("Bad case, 1&nbsp;in&nbsp;20", chf(base["bad_ls"]), chf(base["bad_dca"])),
    ("Worth to you", chf(base["ce_ls"]), chf(base["ce_dca"])),
], f"Over {words(N)} months"))

# ================================================================ who should wait
if nr.gamma is not None:
    who_title = "Only the very cautious should&nbsp;wait." if nr.gamma > gamma else "You are cautious enough to&nbsp;wait."
    who_text = (f"Below a risk aversion of {nr.gamma:.1f}, the return given up while waiting is worth more than the "
                f"risk avoided. Above it, a portfolio fully in shares is too risky in any case. Yours&nbsp;is&nbsp;{gamma:g}.")
else:
    who_title = "Nobody should&nbsp;wait." if nr.winner_if_none == "all now" else "Everybody should&nbsp;wait."
    who_text = ("With these settings, investing all now is better for every attitude to risk we tested."
                if nr.winner_if_none == "all now" else "With these settings, spreading out is better for every attitude to risk we tested.")
html(style.section_head("who", who_title, who_text) + style.scale(nr.gamma, gamma, nr.winner_if_none))

# ================================================================ the past
rate_hist = md.savings_rate if md.savings_rate is not None else r
bt = backtest(md.all_prices, N, rate_hist, W=W, fee=fee)
x = bt["dca_vs_lump_sum"]
share = (x > 0).mean()
past_title = (f"Since {md.all_prices.index[0]:%Y}, waiting won fewer than one start in&nbsp;three." if share < 1 / 3
              else f"Since {md.all_prices.index[0]:%Y}, waiting won {share:.0%} of&nbsp;starts.")
past_text = (f"{len(bt)} start months, real prices of {asset.split(' (')[0].lower()} in Swiss francs, waiting money earning "
             + ("the SARON" if md.savings_rate is not None else f"{rate_pa:.1%} a year")
             + f". Waiting finished ahead {share:.0%} of the time, and {abs(x.mean()):.1%} "
             + ("behind" if x.mean() < 0 else "ahead") + " on average.")
idx = bt.index
ticks = [str(round(idx[int(i)].year + (idx[int(i)].month - 1) / 12)) for i in np.linspace(0, len(idx) - 1, 5)]
ticks[-1] = str(idx[-1].year)
html(style.section_head("past", past_title, past_text) + style.bars(x.to_numpy(), ticks)
     + f"<p class='note' style='margin-top:20px'>Above the line, waiting finished ahead; below it, investing at once did. "
       f"Best start {x.idxmax():%B %Y}, {x.max():+.0%}. Worst {x.idxmin():%B %Y}, {x.min():+.0%}.</p>")

# ================================================================ a crash
quote_slot = st.empty()
with st.expander("Try another scenario"):
    w1, w2, w3 = st.columns(3)
    shock = w1.slider("Fall in the first month (%)", 0, 40, 20) / 100
    vol_mult = w2.slider("Volatility ×", 1.0, 3.0, 1.0, 0.25)
    rate_alt = w3.slider("Savings rate (%/year)", 0.0, 5.0, rate_pa * 100, 0.25) / 100
    scenario_slot = st.empty()
h_alt = g ** 2 + (h - g ** 2) * vol_mult ** 2
alt = analyse(g, h_alt, (1 + rate_alt) ** (1 / 12) - 1, gamma, shock=-shock, vol_mult=vol_mult)
if shock > 0:
    q = (f"If markets fall by {shock:.0%} the month after you invest, waiting wins in "
         f"{'every future' if alt['p_dca'] > 0.995 else f'{alt['p_dca']:.0%} of futures'}. "
         "The trouble is that nobody knows which month that will be.")
else:
    q = f"In this scenario, waiting ends ahead in {alt['p_dca']:.0%} of futures, against {base['p_dca']:.0%} today."
quote_slot.markdown(style.quote(q), unsafe_allow_html=True)
better_alt = "waiting" if alt["ce_dca"] > alt["ce_ls"] else "investing all now"
scenario_slot.markdown(
    f"<p class='result'>In this scenario waiting ends ahead in <b>{alt['p_dca']:.0%}</b> of futures "
    f"(today: {base['p_dca']:.0%}), and <b>{better_alt}</b> is the better choice for you.</p>", unsafe_allow_html=True)

# ================================================================ method
html("<div class='section' id='method'><h2>Method</h2></div>")
with st.expander("How it is calculated"):
    st.markdown(f"Tranche k waits k months on the savings account at rate r, then is invested until the horizon T. "
                f"Returns are measured on {asset} since {start_year}. Futures are simulated "
                f"({'redrawing real monthly returns' if BOOT else 'lognormal, same mean and variance'}): "
                f"{n_first:,} paths for the buying period and {n_paths:,} full paths for the bad and good cases.")
    st.latex(r"W_T=\sum_{k=0}^{N-1}a_k G_k,\quad a_k=\tfrac{W}{N}(1+r)^k-F,\quad G_k=\prod_{t=k+1}^{T}(1+R_t)")
    st.latex(r"E[W_T]=\sum_{k} a_k\, g^{T-k},\qquad g=E[1+R],\; h=E[(1+R)^2]")
    st.latex(r"\operatorname{Var}[W_T]=a^\top C a,\qquad C_{jk}=g^{|k-j|}h^{T-\max(j,k)}-g^{2T-j-k}")
    st.markdown("Continuous version, investing at a constant speed for τ months:")
    st.latex(r"E[W_T]=\frac{W}{\tau}\int_0^{\tau}e^{rs}e^{\mu(T-s)}ds=We^{\mu T}\frac{1-e^{-(\mu-r)\tau}}{(\mu-r)\tau}")
    st.markdown("Worth to you (certainty equivalent), second-order Taylor expansion of a CRRA utility:")
    st.latex(r"CE\approx E[W_T]-\frac{\gamma}{2}\,\frac{\operatorname{Var}[W_T]}{E[W_T]}")
    st.markdown("Best permanent share in shares, where the derivative of x(μ−r) − (γ/2)x²σ² is zero:")
    st.latex(r"x^*=\frac{\mu-r}{\gamma\sigma^2}")
    st.markdown("After the last instalment both plans hold the same asset, so the final wealth is the wealth at the "
                "end of the buying period times a common growth B, independent of it. With CRRA utility:")
    st.latex(r"CE(W_T)=CE(X_N)\cdot CE(B)")
    st.markdown("B scales both plans by the same factor: who wins, and the tipping point, depend only on the "
                "buying period, whatever the horizon.")
    st.markdown("Tipping point, by Newton's method on f(γ) = CE(all now) − CE(spread out):")
    st.latex(r"\gamma_{n+1}=\gamma_n-\frac{f(\gamma_n)}{f'(\gamma_n)}")
    rep = check_definiteness(cov_matrix(N, T, g, h))
    st.caption(f"Your numbers: g = {g:.5f}, h = {h:.5f} (about {mu_pa:.1%} return and {vol_pa:.1%} volatility a year); "
               f"r = {r:.5f} a month; best share in shares x* = {base['x_star']:.0%}; bounds {base['lo']:.2f} and "
               f"{base['hi']:.2f}. Newton: " + (" → ".join(f"{v:.3f}" for v in nr.history) if nr.gamma else "no crossing")
               + ". Matrix C is " + ("positive definite" if rep.ok else "NOT positive semi-definite")
               + " (first minors " + ", ".join(f"{m:.1e}" for m in rep.leading_minors[:3]) + ").")
    rows = []
    for n in sorted({1, 3, 6, 12, N} & set(range(1, min(24, T) + 1))):
        if fee >= W / n:
            continue
        w_n = wealth(base["G"], W, n, r, fee)
        E, V = expected_wealth(W, n, T, r, g, fee), variance(W, n, T, r, g, h, fee)
        rows.append({"Plan": "All now" if n == 1 else f"Over {n} months", "Expected value": chf(E),
                     "Bad case (5%)": chf(np.percentile(w_n, 5)), "Good case (95%)": chf(np.percentile(w_n, 95)),
                     "Worth to you (exact)": chf(ce_exact(wealth(base["F"], W, n, r, fee), gamma) * base["tail"](gamma)),
                     "Worth to you (Taylor)": chf(ce_taylor(E, V, gamma))})
    st.dataframe(pd.DataFrame(rows).set_index("Plan"), width="stretch")
    tab = strategy_table(W, T, r, g, h, gamma, n_max=min(24, T), fee=fee)
    exact = np.array([ce_exact(wealth(base["F"], W, n, r, fee), gamma) for n in tab.index]) * base["tail"](gamma)
    rel_exact, rel_taylor = exact / exact[0] - 1, tab["CE"].to_numpy() / tab["CE"].iloc[0] - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tab.index, y=rel_exact, name="Exact (simulation)", line=dict(color=style.INK, width=2)))
    fig.add_trace(go.Scatter(x=tab.index, y=rel_taylor, name="Taylor approximation",
                             line=dict(color=style.ACCENT, width=1.5, dash="dash")))
    fig.add_hline(y=0, line_color=style.LINE, line_width=1.5)
    fig.update_layout(template="windfall", height=300, xaxis_title="Number of monthly instalments (1 = all now)",
                      yaxis_title="Worth to you vs all now", yaxis_tickformat="+.1%")
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})
    st.caption(f"At {years} years the Taylor approximation points the wrong way, so the answer uses the exact value."
               if np.sign(rel_exact[-1]) != np.sign(rel_taylor[-1])
               else "The best plan is almost always one of the two ends: all now, or as slowly as possible.")

with st.expander("Data and sources"):
    if md.sources:
        st.dataframe(pd.DataFrame(md.sources).T, width="stretch")
    for line in md.log:
        st.caption(line)
    p0, p1 = md.prices.iloc[0], md.prices.iloc[1]
    st.markdown(f"Tracing one number: {md.prices.index[0]:%B %Y} CHF {p0:.4f} → {md.prices.index[1]:%B %Y} "
                f"CHF {p1:.4f}, a return of {p1 / p0 - 1:+.4%}.")
    st.markdown("Prices, exchange rates and the SARON are *observations*. Returns, expected values, variances and "
                "the backtest are *calculations*. Simulated futures and the crash scenario are *hypothetical*.")
    st.download_button("Download the monthly returns (CSV)", md.returns.to_csv().encode(), "returns.csv")

with st.expander("Limitations"):
    st.markdown("""
- Monthly returns are treated as independent: the tool cannot predict crashes, and nobody can time them.
- Past returns depend on the years used (Change the numbers).
- The Taylor formula uses only mean and variance; it ignores the asymmetry of losses and fails over long horizons.
- Only a fixed fee per purchase: no stamp duty, percentage fees or taxes.
- Regret is not modelled: some people spread out to avoid the pain of investing just before a fall.
- Not financial advice.
""")

html(f"<p class='foot'>{asset} and the SPI from Yahoo Finance, SARON from the Swiss National Bank, data to "
     f"{md.prices.index[-1]:%B %Y}. Values in CHF after a fee of {chf(fee)} per purchase, before taxes. "
     "Not financial advice.</p>")
