# Assignment3
# Algorithmic Weekly Trading of GOOGLE via Multi-Level ML Predictions

Jake Hongduo,SHAN
15 Aug 2025


---

## Abstract

This project implements an automated, **weekly** trading strategy for Alphabet (**GOOG**). A machine-learning model forecasts the **next 10-trading-day** log return using technical & statistical features; a **policy layer** maps the prediction to discrete actions (long/flat/short) using **risk-tuned percentiles** and a **regime filter**. The program is robust to data issues (rate limits, column variants, duplicate dates) and maintains a persistent **position state** to only rebalance on Fridays.

---


## Problem Description

* **Objective:** Automated trading for GOOG with a **multi-level pipeline**

  1. Predict the **10-day** forward log return on the latest bar.
  2. Convert the prediction into a **rolling percentile** vs. past weekly predictions (leakage-safe).
  3. Map percentile to action using a **risk band** (e.g., *moderate*: long ≥ 80th, short ≤ 20th).
  4. Apply a **regime filter** so we only long in uptrends and short in downtrends.
  5. **Rebalance weekly (Fridays)**; maintain the prior position on other days.

* **Outputs:** Human-readable JSON signal, signals ledger CSV, historical prediction CSV.

---

## Data Preparation & Robust Loader

* **Primary source:** `data/goog_historical_data.csv`.
* **Backfill:** Tries **Yahoo Finance** (GOOG, then **GOOGL**) and **Stooq** (via `pandas-datareader`) if rate-limited.
* **Normalization:**

  * Standardizes columns → `Open, High, Low, Close, Volume`.
  * Drops duplicates; coerces any accidental (N,1) DataFrame columns to 1-D Series.
  * Ensures a strictly increasing **Date** index, timezone-naive and **normalized to midnight**, which aligns with weekly rebalance checks.

---

## Features, Target, and Leakage Controls

* **Features (daily):** 1/5/10-day log returns, rolling vol (10/20), MA ratios (10/20/50/200), RSI(14), MACD (line/signal/hist), Bollinger z-score vs. MA20 & stdev20, volume z-score, intraday range (H–L)/Close, and (Close–Open)/Open.

* **Target:**

  $$
  y_t = \log(C_{t+10}) - \log(C_t)
  $$

  We **build features for all dates**, but labels only exist where $C_{t+10}$ is known, letting us **predict on the newest bar** without look-ahead.

* **Leakage controls:**

  * Train on the most recent **252 labeled** rows (≈ 1 trading year), **excluding the latest unlabeled bar**.
  * Percentiles are computed against **stored past weekly predictions** and **explicitly exclude today**.

---

## Model & Policy (Multi-Level Prediction → Action)

* **Model:** `HistGradientBoostingRegressor` (sklearn) inside a pipeline
  `Imputer(median) → StandardScaler(with_mean=False) → HGBR(max_depth=6, lr=0.05, max_iter=400)`.

* **Percentile mapping (risk bands):**

| Band     | Buy (Long) if pct ≥ | Short if pct ≤ |
| -------- | ------------------- | -------------- |
| low      | 0.90                | 0.10           |
| moderate | 0.80                | 0.20           |
| high     | 0.70                | 0.30           |

> Set in code via `RISK_BAND` and `RISK_BANDS`.

* **Regime filter:** 126-day momentum $\frac{C_t}{C_{t-126}}-1$

  * Uptrend (≥ 0): block **shorts**
  * Downtrend (≤ 0): block **longs**
    (Threshold can be softened, e.g., ±5%.)

* **Rebalance cadence:** `REBALANCE="W-FRI"` – only change exposure on Fridays.
  On other days, we **carry the prior position** (`live_state/position_state.json` keeps it).

* **ATR-based prompts:** Suggested stop/take (not orders) based on **ATR(14)**:

  * Long: stop = 1.5×ATR below, take = 3×ATR above
  * Short: stop = 1.5×ATR above, take = 3×ATR below

---

## Configuration (in `Google_Trading.py`)

```python
TICKER = "GOOG"
ALT_TICKER = "GOOGL"
REBALANCE = "W-FRI"     # or "D" for daily
LOOKBACK_D = 252
FWD_HORIZON_D = 10
PCTL_WINDOW_WEEKS = 20
LONG_ONLY = False       # True => long/flat only
USE_REGIME_FILTER = True
REGIME_WINDOW = 126
RISK_BAND = "moderate"  # "low" | "moderate" | "high"
ATR_WINDOW = 14
STOP_MULT = 1.5
TAKE_MULT = 3.0
```

---

## Program Outputs

* **`live_state/model_info_GOOG.json`** (latest signal snapshot)

  ```json
  {
    "timestamp": "2025-08-15 00:00:00",
    "ticker": "GOOG",
    "close": 204.91,
    "predicted_log_return_next_10d": 0.036017,
    "pred_percentile_20w": 0.0000,
    "risk_band": "moderate",
    "rebalance": "W-FRI",
    "long_only": false,
    "regime_filter": true,
    "signal_action": "HOLD/FLAT",
    "action_applied": "REBALANCE→FLAT",
    "current_position": 0,
    "atr_pct": 0.0212,
    "suggested_stop": null,
    "suggested_take_profit": null,
    "active_stop": null,
    "active_take_profit": null,
    "hypothetical_long_stop": 198.40,
    "hypothetical_long_take_profit": 217.92,
    "hypothetical_short_stop": 211.42,
    "hypothetical_short_take_profit": 191.90
  }
  ```

* **`live_state/signals_ledger_GOOG.csv`** 

* **`live_state/pred_history_GOOG.csv`**

But now I face rate limits

### What each field means

* **timestamp** – The bar being evaluated (normalized to midnight) → **2025-08-15** (a Friday), i.e., a rebalance day.
* **ticker / close** – Asset and latest close used by the model → **GOOG @ \$204.91**.
* **predicted\_log\_return\_next\_10d** – Model’s 10-trading-day **log** return forecast. Convert to simple % as $e^{x}-1$.

  * Here $e^{0.036017}-1 \approx \mathbf{+3.67\%}$.
* **pred\_percentile\_20w** – Where today’s prediction ranks vs the **last 20 weekly** predictions (excludes today).

  * **0.0000** = lowest in the window. Note: this is a **relative** rank; it can be lowest even if the forecast is still positive.
* **risk\_band** – Thresholds used to map percentile → action (for *moderate*: **buy ≥ 0.80**, **short ≤ 0.20**, else flat).
* **rebalance** – Decisions only applied on **Fridays** (`W-FRI`).
* **long\_only** – `false` means long **or** short are allowed **before** filters.
* **regime\_filter** – `true` means only long in **uptrends** and only short in **downtrends** (126-day trend).
* **signal\_action** – The **policy output** after applying thresholds **and** the regime filter.

  * Here it’s **HOLD/FLAT** (no new long because percentile < 0.80; would be a short by thresholds, but regime filter blocked it).
* **action\_applied** – What the engine actually **set** on this bar given the rebalance rule.

  * **REBALANCE→FLAT** because it’s Friday and the signal is flat after filters.
* **current\_position** – Resulting target position: **0** (flat).
* **atr\_pct** – ATR(14) as a % of price → **2.12%** (≈ \$4.35). Used to size suggested stop/target distances.
* **suggested\_stop / suggested\_take\_profit** – *Suggested* levels for the **current signal direction** only.

  * They’re `null` here because the signal is flat (no direction to size).
* **active\_stop / active\_take\_profit** – Levels attached to the **active/held** position (persist between rebalances).

  * Also `null` here because we’re flat.
* **hypothetical\_long\_* / hypothetical\_short\_*\*\* – What stops/targets **would** be if you were long/short **right now** using ATR sizing (1.5×ATR for stop, 3×ATR for take).

  * Long: **stop \$198.40**, **take \$217.92**
  * Short: **stop \$211.42**, **take \$191.90**

### Why is the strategy today flat?

1. Today’s prediction is **relatively weak** vs the past 20 weekly predictions (**percentile = 0.0000**).
2. Under the **moderate** band, low percentiles (≤ 0.20) would normally trigger a **short**.
3. The **regime filter** detected an **uptrend** (126-day momentum ≥ 0), so shorts are **blocked**.
4. It’s Friday (rebalance day), so the engine applies **FLAT**: `action_applied = "REBALANCE→FLAT"` and `current_position = 0`.

---

