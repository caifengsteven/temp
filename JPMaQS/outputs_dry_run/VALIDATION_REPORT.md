# [SYNTHETIC] 2Y OIS Fair Value Model - Validation Report

> **DRY_RUN = True.** All numbers below are generated from synthetic random-walk data. They verify the pipeline STRUCTURE only and are NOT usable for economic analysis. Re-run with `DRY_RUN = False` on a Bloomberg Terminal host for real results.

## 1. Headline improvement: nominal OIS vs real yield + Fisher proxy

The Kaggle edition of this model used `RYLDIRS02Y_NSA` (the 2Y **real** IRS yield) as the dependent variable, then reconstructed a nominal-like series via the Fisher equation (`nominal = real + 1Y_expected_inflation`). The Fisher proxy restores the expected positive inflation coefficient sign but introduces an extra layer of measurement error.

This Bloomberg edition pulls the 2Y **nominal** OIS swap rate directly (`ois_2y` in `BBG_TICKERS`) - no Fisher reconstruction. Expected effects on the regression:

- **Inflation coefficient**: should be more stable and closer to the note's published values, since we no longer compound two estimates of expected inflation.
- **Adjusted R2**: should be modestly HIGHER than the Kaggle version because the dependent variable is no longer a two-step reconstruction.
- **Policy-rate coefficient**: should remain close to 1 (the note reports 0.75-0.85 for G10 ex-JPY).

## 2. Structural sign checks

- Inflation coefficient POSITIVE in 8/8 currencies (expected: all).
- Policy rate is the DOMINANT regressor (largest |coef|) in 8/8 currencies (expected: all except JPY where it is anomalously large in the OTHER direction).

Per-mandate employment-sign check:
  - USD: empl_coef=+0.10 (mandate expects POSITIVE) -> OK
  - EUR: empl_coef=+0.10 (mandate expects negative) -> OFF-SIGN
  - GBP: empl_coef=+0.08 (mandate expects negative) -> OFF-SIGN
  - SEK: empl_coef=+0.08 (mandate expects negative) -> OFF-SIGN
  - NOK: empl_coef=+0.11 (mandate expects POSITIVE) -> OK
  - AUD: empl_coef=+0.07 (mandate expects POSITIVE) -> OK
  - NZD: empl_coef=+0.11 (mandate expects negative) -> OFF-SIGN
  - JPY: empl_coef=+0.06 (mandate expects negative) -> OFF-SIGN

- JPY policy coefficient = +0.91. The note flags JPY as anomalously large (published ~1.47) due to the YCC regime - the 2Y OIS is essentially pegged to the policy rate, producing a near-unity-or-higher beta.

## 3. Point-in-time caveat (IRREDUCIBLE)

Bloomberg `BDH` returns the **latest-vintage** value of each series. Macro releases - especially GDP (revised 3+ times after the advance print) and unemployment (annual benchmark revisions) - will be the value known as of TODAY, not the value known on each historical date in the regression window.

The JPMaQS dataset used by the note is **revision-free** (quantamental - records only the value that was actually available at each timestamp). This produces an irreducible divergence between this script's results and a true point-in-time replication:

- **GDP** is the worst offender: the 5Y-MA-based excess measure uses TODAY's revised GDP history. Periods of large benchmark revisions (e.g. UK ONS blue-book 2021-2022) will distort the excess-GDP regressor for the affected dates.
- **Unemployment** is monthly and revised - usually small revisions but annual benchmarks can shift the level by 0.1-0.3 pp.
- **Inflation swaps** and **OIS swap rates** are market quotes and ARE effectively point-in-time (no revision) - the headline improvement of this edition.
- **Policy rates** are essentially point-in-time (central-bank decisions are not revised after the fact).

Net effect: the OIS / policy / inflation-swap side of the regression is clean; the GDP / unemployment side carries look-ahead bias. A true point-in-time replication would require Bloomberg's `BLP_GDSC` / vintage data product, which is out of scope here.

## 4. Convention changes over the 10Y window

The 2014-2024 regression window spans four major benchmark-rate regime changes. Modern Bloomberg tickers (SOFR, EUR-flat, SONIA, TONA) back-fill the historical series using the new convention, but small regime breaks remain:

| Currency | Old convention | New convention | Switch date | Notes |
|----------|----------------|----------------|-------------|-------|
| USD | Fed Funds OIS | **SOFR OIS** (`USOSFR2`) | Oct 2020 | SOFR replaced Fed Funds as the standard risk-free rate. |
| EUR | EONIA OIS | **EUR OIS** (`EUSWEA2`) | Oct 2021 | EUR-flat replaced EONIA when EUR went negative-then-positive. |
| GBP | LIBOR-based swaps | **SONIA OIS** (`BPSWS2`) | Dec 2021 | LIBOR cessation; SONIA became the standard. |
| JPY | JPY LIBOR / TIBOR | **TONA OIS** (`JYOES2`) | Jul 2021 | TONA became the standard risk-free rate post-LIBOR. |

Implication: the first ~3-5 years of the 10Y window are back-filled under the new convention and may differ slightly from what was actually tradeable at the time. This affects level and volatility in the early part of the sample.

## 5. Ticker verification status

Confidence levels as documented in `BBG_TICKERS`. **Before trusting any regression output (DRY_RUN=False), run `verify_tickers()` on the terminal** and confirm every row below resolves. Tickers marked LOW need terminal-side verification (small-ccy OIS markets are thin and mnemonics vary by desk).

| Ccy | Field | Ticker | Confidence | Action if unresolved |
|-----|-------|--------|------------|----------------------|
| USD | ois_2y | `USOSFR2 Curncy` | HIGH | - |
| USD | policy | `FEDL01 Index` | HIGH | - |
| USD | unemp | `USURTOT Index` | HIGH | - |
| USD | infl_1y | `USSWIT1 Curncy` | MEDIUM | verify on terminal |
| USD | gdp | `GDP CHWG Index` | HIGH | - |
| EUR | ois_2y | `EUSWEA2 Curncy` | MEDIUM | EUR vs EONIA legacy - verify post-Oct 2021 |
| EUR | policy | `ECBDFR Index` | HIGH | - |
| EUR | unemp | `UMRTEMU Index` | HIGH | - |
| EUR | infl_1y | `EUSWIT1 Curncy` | MEDIUM | verify on terminal |
| EUR | gdp | `EUGNEMU Index` | HIGH | - |
| GBP | ois_2y | `BPSWS2 Curncy` | HIGH | - |
| GBP | policy | `UKBRBASE Index` | HIGH | - |
| GBP | unemp | `UKUEILOR Index` | HIGH | - |
| GBP | infl_1y | `BPSWIT1 Curncy` | MEDIUM | verify on terminal |
| GBP | gdp | `EUGNUK Index` | HIGH | - |
| SEK | ois_2y | `SDSOA2 Curncy` | LOW | verify; SIOR vs STIBOR conventions |
| SEK | policy | `SBREPO Index` | MEDIUM | alt: `SRREPO Index` |
| SEK | unemp | `SWUESART Index` | HIGH | - |
| SEK | infl_1y | `SDSWIT1 Curncy` | LOW | thin - verify |
| SEK | gdp | `EUGNSE Index` | HIGH | - |
| NOK | ois_2y | `NOSOAS2 Curncy` | LOW | verify; NOWA OIS is the post-2019 standard |
| NOK | policy | `NOWAO Index` | MEDIUM | alt: `NORWKEY Index` |
| NOK | unemp | `NOLBRATE Index` | HIGH | - |
| NOK | infl_1y | `NOSWIT1 Curncy` | LOW | thin - verify |
| NOK | gdp | `NOGDCOSA Index` | HIGH | - |
| AUD | ois_2y | `ADSOA2 Curncy` | LOW | try `ADSOAS2 Curncy` (AONIA swap); verify desk mnemonic |
| AUD | policy | `RBAOCR Index` | MEDIUM | verify on terminal |
| AUD | unemp | `AULFUNEM Index` | HIGH | - |
| AUD | infl_1y | `ADSWIT1 Curncy` | LOW | thin - consider hardcoding infl_target |
| AUD | gdp | `AUNAGDP Index` | HIGH | - |
| NZD | ois_2y | `NZSOA2 Curncy` | LOW | verify; NZ OIS market is very thin |
| NZD | policy | `RBNZOCR Index` | MEDIUM | verify on terminal |
| NZD | unemp | `NZLFUNER Index` | HIGH | - |
| NZD | infl_1y | `NZSWIT1 Curncy` | LOW | very thin - hardcode target recommended |
| NZD | gdp | `NZNTEGDP Index` | HIGH | - |
| JPY | ois_2y | `JYOES2 Curncy` | MEDIUM | verify TONA mnemonic |
| JPY | policy | `JBOCDR Index` | MEDIUM | BoJ policy-rate balance - verify |
| JPY | unemp | `JNUE Index` | HIGH | - |
| JPY | infl_1y | `JYSWIT1 Curncy` | LOW | thin market - fall back to hardcode target + CPI spread |
| JPY | gdp | `JGDPAGDP Index` | HIGH | - |

## 6. Window extension

The Kaggle edition's regression window was hard-capped at Dec-2023 (the end of the free JPMaQS sample). The Bloomberg edition is configured with a fixed 2014-12-20 -> 2024-12-20 10Y window for synthetic testing. **In live mode (`DRY_RUN = False`), the window end auto-advances to TODAY**, so the user can reproduce the note's actual Jul-2026 signal window and roll the model forward on subsequent days.

- 10Y window: `2014-12-20` -> `2024-12-20`
- 5Y window: `2019-12-20` -> `2024-12-20`

## 7. Latest 5Y residual z-score (rich / cheap signal)

| Ccy | z-score | Signal |
|-----|---------|--------|
| NZD | +1.60 | RICH (OIS > fair value) |
| USD | +1.57 | RICH (OIS > fair value) |
| EUR | +1.53 | RICH (OIS > fair value) |
| NOK | +1.18 | RICH (OIS > fair value) |
| AUD | +1.05 | RICH (OIS > fair value) |
| GBP | +0.93 | near fair value |
| JPY | +0.54 | near fair value |
| SEK | -0.15 | near fair value |
