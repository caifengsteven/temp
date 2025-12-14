## Using Open Interest (OI) to predict calendar spread / roll price

Open interest is useful for spreads because it helps you estimate **how much “roll demand” is still to come** and **how urgent it is**, which can create *predictable pressure* on the calendar spread (e.g., `Next − Front`). The key is to treat OI as a **position stock** and infer **flow + remaining inventory to roll**.

Two important caveats up front:

1) **OI is usually end-of-day** (published after close). It’s not a true intraday signal unless your venue provides intraday OI updates.  
2) **OI is not directional by itself** (every contract has a long and a short). The predictive value comes from **migration patterns across maturities** and **interaction with volume/price**.

---

## 1) What OI is telling you in a roll context

During a roll, many participants move exposure from front to next. That typically looks like:

- **Front OI ↓** (positions closed/rolled out)
- **Next OI ↑** (positions opened/rolled into)

This “OI migration” tends to be **concentrated in a known window** (e.g., X days before expiry, or around index roll conventions), and the spread price can drift because the roll is often **one-sided in urgency** (many need to do it; fewer naturally want the opposite at that time).

So the prediction target is usually not “level of spread forever”, but **expected drift over the remaining roll window**.

---

## 2) Build the right OI features (the ones that actually help)

Let:

- \( OI_1(t) \) = open interest of front contract  
- \( OI_2(t) \) = open interest of next contract  
- \( S(t) \) = calendar spread price (e.g., \(F_2 - F_1\), choose your sign and keep it consistent)

### A. Roll migration rate (core)
A simple, robust signal is the **share of OI migrating**:

\[
\text{RollSpeed}(t) = \frac{-\Delta OI_1(t)}{OI_1(t-1) + OI_2(t-1)}
\]

(Front OI dropping is positive roll progress.)

You can also track **net migration imbalance**:

\[
\text{MigImb}(t) = \frac{\Delta OI_2(t) - (-\Delta OI_1(t))}{OI_1(t-1)+OI_2(t-1)}
\]

- If \( \Delta OI_2 \approx -\Delta OI_1 \): migration is “clean roll”.
- If not, you may be seeing **new risk being added** or **positions being removed entirely**, which affects interpretation.

### B. Remaining roll inventory (how much still needs to move)
Define a *target* front share based on history. For each “days-to-expiry” \(d\), estimate from past cycles:

\[
p^*(d) = \text{median}\left(\frac{OI_1}{OI_1+OI_2}\right)\ \text{at days-to-expiry } d
\]

Then compute today’s deviation:

\[
\text{Remain}(t) = \frac{OI_1(t)}{OI_1(t)+OI_2(t)} - p^*(d_t)
\]

If `Remain` is high late in the cycle, it implies **more front OI than usual still needs to roll**, often leading to **continued roll pressure** in subsequent days.

### C. Urgency metric (inventory per remaining day)
If there are \(D(t)\) trading days left in your roll window:

\[
\text{Urgency}(t) = \frac{\max(0,\ \text{Remain}(t))}{D(t)}
\]

This tends to be more predictive than raw OI changes because it embeds the calendar constraint.

### D. OI + price/volume interaction (opening vs closing intensity)
Use the classic intuition:

- **Price up + OI up** → new positions added in direction of move (trend reinforcement)
- **Price up + OI down** → short covering / position reduction (less persistent)

For spreads, do the same with spread return \(r_S(t)\) and \(\Delta OI\) features. Even if you can’t infer “direction”, this helps identify whether today’s move is likely to persist.

---

## 3) Simple predictive models you can deploy

### Model 1: Next-day spread return from roll pressure (EOD OI)
Predict next day spread change:

\[
r_S(t+1) = a + b_1 \cdot \text{RollSpeed}(t) + b_2 \cdot \text{Urgency}(t) + b_3 \cdot \text{Regime}(t) + \epsilon
\]

Where `Regime` can include:
- outright vol regime (HSI/HHI realized vol)
- liquidity measures (spread book depth, bid/ask)
- day-to-expiry, day-of-week, holiday effects

**Practical tip:** run this model **only inside a roll window** (e.g., 15 trading days to expiry down to 3), otherwise OI migration has different meaning and coefficients will wash out.

### Model 2: “Deviation from normal migration path” mean reversion/trend
Use the `Remain(t)` feature:

- If `Remain` is **high** late in the cycle → expect **continued roll activity** → often implies **trend continuation** in the direction associated with rolling pressure in your market.
- If `Remain` is **low** (roll mostly done early) → expect **less forced flow** → spread may **mean-revert** toward fair carry.

You can trade:
- **trend-follow** during “still lots to roll + time is short”
- **mean-revert** when roll is largely completed and liquidity normalizes

### Model 3: Joint model with fair-value carry (recommended)
OI works best when combined with a fair-value anchor:

\[
S(t) = S_{\text{carry}}(t) + \lambda \cdot \text{Urgency}(t) + \text{microstructure noise}
\]

So you’re predicting **deviations from carry** using OI-driven pressure.

---

## 4) How to validate (avoid fooling yourself)

1) **Normalize the spread** (e.g., in index points *and* in bps-equivalent) so coefficients are stable across regimes.  
2) **Event-study by days-to-expiry**: average spread drift conditional on high vs low `Urgency`.  
3) **Out-of-sample by contract cycle**: train on past years, test on new roll cycles.  
4) Measure whether OI signals add value **after costs** (spreads can be small; fees/slippage dominate).

---

## 5) Common pitfalls
- Using OI levels without conditioning on **days-to-expiry** (seasonality is huge).
- Not separating “clean roll migration” from “new risk added/removed”.
- Assuming OI gives an intraday edge when it’s **EOD**.
- Ignoring that roll pressure can flip depending on who dominates flow (hedgers vs speculators), so you need **regime filters**.

---

## Questions to tailor this to your HSI/HHI roll trading
1) Do you get OI **intraday** or only **end-of-day** from HKEX?  
2) Are you trading **native calendar spread** order book or legging outrights?  
3) Which spread: **front-next (monthly)**, **quarterly roll**, or both?  
4) What’s your holding period (minutes vs days)?

If you answer these, I can propose a concrete signal definition (exact roll window, target \(p^*(d)\), and an entry/exit rule) and how to integrate it with your low-latency execution (maker vs taker, leg-risk handling).