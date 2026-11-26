# Comparison with an AI assistant (guideline 7)

**To do:** paste the prompt below into ChatGPT, Claude or another assistant, *without* uploading our code. Then fill in the "AI answer" column and keep a screenshot for the presentation appendix.

- Assistant and model: ChatGPT
- Date: 26 November 2026
- Settings (web search on or off, etc.): ______________________

## Prompt (copy exactly)

> I am a young graduate in Switzerland and I just received CHF 20,000. I want to invest it in an MSCI World equity ETF, measured in Swiss francs. Should I invest everything now or spread it over 12 equal monthly instalments?
>
> Assumptions: monthly returns have a mean of 0.841% and a standard deviation of 4.27%, independent from month to month. Money that is waiting earns 0.5% per year on a savings account. No fees, no taxes. Horizon: 10 years after today. My preferences: CRRA utility with relative risk aversion γ = 4.
>
> Please give numbers for:
> 1. the expected value of each strategy after 10 years;
> 2. the probability that spreading over 12 months ends up ahead of investing everything now;
> 3. which strategy gives me the higher expected utility (certainty equivalent);
> 4. the risk aversion γ above which spreading becomes the better choice.

## Our tool's answers (same inputs, download of 6 Nov 2026, data since Jan 2015)

| Question | Our tool | How it is computed | AI answer | Match? |
|---|---|---|---|---|
| 1. Expected value, all now | CHF 54,648 | closed form W·g^T | CHF 54,637 | Yes (rounding of g in the prompt) |
| 1. Expected value, 12 months | CHF 52,326 (−2,321) | closed form Σ a_k g^(T−k) | CHF 52,319 (−2,318) | Yes |
| 2. Spreading ends ahead | 32 % of futures | 200,000 lognormal simulations of the buying period | 31.7 % | Yes |
| 2. Same question on real history | 29 % of start months since 2010 | backtest, real prices and real SARON | not done | Our addition |
| 3. Better for γ = 4 | All now (certainty equivalent 35,566 vs 34,967) | exact CRRA certainty equivalent | All now (35,544 vs 34,951) | Yes |
| 4. Tipping point γ* | 6.53 (between the bounds 4.4 and 8.9) | safeguarded Newton, CE(W_T) = CE(X_N)·CE(B) | 6.52 | Yes, after our correction (see note) |
| 4. With real returns redrawn (bootstrap) | 6.2 | bootstrap simulation | not done (assumes lognormal) | Our addition |

## Result

ChatGPT's numbers match ours closely when it is given the parameters. It also states, like our backtest, that the outcome is decided in the first 11 months, and it cites the Vanguard study.

**Note on the tipping point.** Our first version found γ* = 6.1 from 20,000 ten-year paths. ChatGPT's 6.52 made us check: with other random seeds that version moved between about 5.4 and 7.2. We fixed it mathematically: once the last instalment is in, both plans hold the same asset, so CE(W_T) = CE(X_N) · CE(B), and the common factor CE(B) has an exact formula. Only the buying period needs to be simulated, and γ* is now 6.53 ± 0.03 — the same as ChatGPT. The comparison with the AI found a weakness in our tool, and we corrected it.

**What the AI did not do:** estimate the inputs from real data, use the real SARON, backtest on real history (29 %), test the lognormal assumption with real returns (γ* = 6.2), or notice that the tipping point does not depend on the horizon.

## What to look for in the AI answer

- **Numbers or only principles?** A typical answer quotes the Vanguard result ("lump sum wins about two thirds of the time") without computing anything for these inputs.
- **Expected value:** does it compound correctly (0.841 % per month over 120 months) and give the waiting money its 0.5 %?
- **Risk aversion:** does it use γ = 4 at all, or only say "it depends on your risk tolerance"?
- **Tipping point:** does it find a threshold, and is it near 6?
- **Honesty:** if the AI gets something right, say so in the presentation (the guideline asks for a fair comparison).

## Sources for the context slide

- Vanguard, *Cost averaging: Invest now or temporarily hold your cash?*, Finlay & Zorn, February 2023 — lump sum beat cost averaging about 68 % of the time (1976–2022); cost averaging may suit investors with very high aversion to risk and losses. https://corporate.vanguard.com/content/dam/corp/research/pdf/cost_averaging_invest_now_or_temporarily_hold_your_cash.pdf
