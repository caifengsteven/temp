# 2330 Market-Making + 0050 Hedge (Queue-Theory-Based) — Single Document

> Purpose: Provide liquidity in **2330** (wide bid/ask) using **limit orders**, and hedge the market/index component using **0050** (tight bid/ask).  
> Core technique: Use a **queueing / limit-order-book fill probability model** to decide whether to **join** the best price or **improve** by 1 tick, and how large to quote.

---

## 1) One-sentence idea

**Earn spread in 2330 by posting limit orders, and neutralize systematic moves by immediately hedging fills using 0050; use queue-based fill probability to avoid “never fill / only fill when toxic.”**

---

## 2) What to model

You need:
1. **Execution probability in 2330** (queue position + depletion rates).
2. **Adverse selection** (probability midprice moves against you after fill).
3. **Hedge ratio and hedge cost** using 0050 (tight spread, fast execution).
4. **Inventory control** (avoid running big unhedged exposure).

---

## 3) Queueing model for 2330 fill probability

### 3.1 Queue position
When you place a limit order at the best price (e.g., best bid):
- Let `Q0` = shares (or lots) **ahead of you** at that price level at order time.
- Convert to queue units (prefer **lots**): `k = lots_ahead`.

To be filled, the queue in front must be depleted by:
- marketable orders that trade at that price
- cancellations of orders ahead of you

### 3.2 Depletion rate (Poisson approximation)
Estimate over a rolling window (e.g., last 60–300 seconds):
- `λ_MO` = rate of marketable orders consuming that best level
- `μ_cancel` = rate of cancellations ahead at that best level

Define:
- `δ = λ_MO + μ_cancel`  (queue depletion event rate)

Over horizon `τ` seconds:
- `N(τ) ~ Poisson(δ τ)` = number of depletion “events”

Approximate fill probability for reaching the front:
- `P_fill(τ) ≈ P(N(τ) ≥ k)`

Closed form (Poisson tail):
- `P_fill(τ) ≈ 1 - Σ_{i=0}^{k-1} exp(-δτ) * (δτ)^i / i!`

### 3.3 Join vs improve by 1 tick
For each side (bid/ask), evaluate:
- **Join** best: large `k_join` (deep queue), better economics
- **Improve** 1 tick: `k ≈ 0`, higher fill probability, but you give up 1 tick

Approximation for improve:
- `P_improve(τ) ≈ 1 - exp(-δ τ)` (since `k≈0`)

Pick the action that maximizes expected value (Section 6).

---

## 4) Adverse selection penalty (must-have)

Wide spreads often imply toxic flow: you may get filled right before the price moves against you.

Model a short-horizon adverse move cost:
- `AdverseCost ≈ P(mid moves against within h) * E(|Δmid|)`

Estimate `P(against)` from features such as:
- order flow imbalance near top levels
- recent trade signs (aggressive buy/sell)
- spread/volatility regime

---

## 5) Hedging with 0050 (tight spread)

### 5.1 Hedge ratio via regression (recommended)
Estimate rolling beta `b` using returns:
- `r_2330,t = a + b * r_0050,t + ε_t`

When you receive a fill `ΔS` shares of 2330 at price `P_S`:
- 2330 notional change: `ΔN_S = ΔS * P_S`
- target hedge notional in 0050: `ΔN_E = b * ΔN_S`
- 0050 shares to trade: `ΔE = - ΔN_E / P_E`

(`-` sign hedges the exposure.)

### 5.2 Weight-based hedge (simpler alternative)
If `w` = weight of 2330 inside 0050:
- `ΔE ≈ - (ΔS * P_S) / (w * P_E)`

(Usually less robust than regression beta.)

### 5.3 Execution choice
Typical practice:
- **2330**: passive (limit) to capture spread
- **0050**: aggressive/IOC/marketable to reduce risk quickly

---

## 6) Quoting rule: expected value + inventory penalty

For each candidate quote level `p` (join, improve, etc.) compute:

`EV(p) = P_fill(p) * (Edge(p) - AdverseCost(p) - HedgeCost) - InventoryPenalty`

Where:
- `Edge(p)`: expected spread capture vs mid or microprice
- `HedgeCost`: expected cost of hedging with 0050 (spread + fees + slippage)
- `InventoryPenalty`: discourages large exposure

### 6.1 Exposure definition (notional)
Let inventories be `I_2330` shares and `I_0050` shares:
- `X = I_2330 * mid(2330) - b * I_0050 * mid(0050)`

Penalty (example):
- `InventoryPenalty = γ * X^2`

### 6.2 Inventory skew (optional enhancement)
If you are too long 2330 (positive `X`), you want to sell faster and buy slower.
A simple skew in quotes:
- `p_bid = p_bid* - κX`
- `p_ask = p_ask* - κX`

(Exact sign convention depends on how you define `X`; goal is mean-revert inventory.)

---

## 7) End-to-end algorithm loop (high level)

### Inputs (real-time)
- L2 book for 2330 and 0050: best bid/ask + depths (ideally multi-level)
- Trades/timestamps/sizes
- Tick sizes, fees/taxes, order constraints

### Parameters
- `τ`: fill-prob horizon (e.g., 0.5s)
- rolling window for `(λ_MO, μ_cancel)` estimation (e.g., 60–300s)
- inventory limits, max order size
- news/volatility filters

---

## 8) Pseudocode

```text
Initialize:
  I_2330 = 0
  I_0050 = 0
  b = initial_beta_estimate
Loop every dt (e.g., 50–200 ms):

  # 1) Observe market
  S_bid, S_ask, depthS_bid, depthS_ask = L2(2330)
  E_bid, E_ask, depthE_bid, depthE_ask = L2(0050)

  # 2) Estimate 2330 queue depletion rates (rolling)
  lambda_hit_bid, mu_cancel_bid = estimate_rates(best_bid_level)
  lambda_hit_ask, mu_cancel_ask = estimate_rates(best_ask_level)
  delta_bid = lambda_hit_bid + mu_cancel_bid
  delta_ask = lambda_hit_ask + mu_cancel_ask

  # 3) Compute queue positions (lots ahead) for join
  k_join_bid = lots_ahead_at_best_bid()
  k_join_ask = lots_ahead_at_best_ask()

  # 4) Fill probabilities (Poisson tail)
  P_join_bid = 1 - PoissonCDF(k_join_bid - 1, delta_bid * τ)
  P_join_ask = 1 - PoissonCDF(k_join_ask - 1, delta_ask * τ)

  # Improve by 1 tick (k≈0)
  P_impr_bid = 1 - exp(-delta_bid * τ)
  P_impr_ask = 1 - exp(-delta_ask * τ)

  # 5) Adverse selection model (simple placeholder)
  adv_bid = adverse_cost_if_buy()
  adv_ask = adverse_cost_if_sell()

  # 6) Update hedge ratio slowly (e.g., every minute)
  b = rolling_regression_beta(returns_2330, returns_0050)

  # 7) Current net exposure
  X = I_2330 * mid(2330) - b * I_0050 * mid(0050)

  # 8) Evaluate expected value for quoting choices
  EV_join_bid = P_join_bid * (edge_join_bid - adv_bid - hedge_cost_per_share) - γ * X^2
  EV_impr_bid = P_impr_bid * (edge_impr_bid - adv_bid - hedge_cost_per_share) - γ * X^2

  EV_join_ask = P_join_ask * (edge_join_ask - adv_ask - hedge_cost_per_share) - γ * X^2
  EV_impr_ask = P_impr_ask * (edge_impr_ask - adv_ask - hedge_cost_per_share) - γ * X^2

  Choose bid quote option = argmax(EV_join_bid, EV_impr_bid) subject to inventory limits
  Choose ask quote option = argmax(EV_join_ask, EV_impr_ask) subject to inventory limits

  # 9) Place/cancel 2330 orders accordingly
  manage_orders_2330(bid_choice, ask_choice, size_rules)

  # 10) On any 2330 fill event:
     Update I_2330
     Hedge in 0050:
       ΔN_S = ΔS * P_S
       ΔN_E = b * ΔN_S
       ΔE   = -ΔN_E / P_E
     Send 0050 hedge order (marketable/IOC)
     Update I_0050

  # 11) Risk filters
  If vol spike, limit-up/down regime, 0050 liquidity drop, or news:
     cancel/widen 2330 quotes, reduce sizes, or pause