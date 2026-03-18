# VWAP-MR Strategy Parity Audit
**Date:** 2026-03-17  
**Engine:** `/tmp/trading/capital_allocation_ai/vwap_control_flip_strategy.py`  
**Auditor:** Production infrastructure build (automated source inspection)

---

## 1. Real VWAP Construction

**Function:** `vwap_and_sigma(df, reset_hour_utc=23)`  
**Location:** `vwap_control_flip_strategy.py`, lines ~79-115

### Implementation (confirmed):
- **Price input:** Typical price `(H + L + C) / 3`
- **Reset logic:** Daily-reset at UTC 23:00 via `_reset_day_id()`
  - Shifts timestamp back by `reset_hour_utc` hours, then floors to day
  - Groups bars by this shifted day ID → each day starts fresh at 23:00 UTC
- **Accumulation:** Cumulative `TP * Volume / cumVol` — standard VWAP
- **Sigma:** Volume-weighted standard deviation around VWAP
  - `sqrt(sum(vol * (tp - vwap)^2) / cumVol)` — computed cumulatively within each day

### Confirmed:
- ✅ VWAP is **daily-reset at 23:00 UTC** (not expanding window)
- ✅ Uses typical price (not close)
- ✅ Sigma is volume-weighted, not simple std

---

## 2. Real Distance Filter (Entry Condition)

The strategy does NOT use a standalone "dist filter" parameter. Entry is gated by:

### VWAP Touch / Retest Detection:
```python
tol = self.p.touch_tol_atr_frac * row['atr20']  # default: 0.05 * ATR(20)

# Long entry condition:
retest_hold = self._touch(row['low'], row['vwap'], tol) AND (row['close'] > row['vwap'])

# Short entry condition:
retest_reject = self._touch(row['high'], row['vwap'], tol) AND (row['close'] < row['vwap'])
```

- `_touch(value, target, tol)` → `abs(value - target) <= tol`
- **Long:** Bar's low must wick within 5% ATR of VWAP, close above VWAP → bullish retest
- **Short:** Bar's high must wick within 5% ATR of VWAP, close below VWAP → bearish rejection

### Direction:
- LONG entry: price reclaims VWAP from below (cross_up) then retests from above → bullish
- SHORT entry: price breaks VWAP from above (cross_down) then retests from below → bearish
- **Not inverted** in the real module

---

## 3. Real Session Logic

**Function:** `in_sessions(ts, sessions)`

### Default sessions (UTC):
```python
sessions_utc = (("08:00", "10:00"), ("13:30", "17:00"))
```

- London open: 08:00–10:00 UTC
- NY open/overlap: 13:30–17:00 UTC
- **All other hours: NO trades**

### Per-strategy overrides:
- XAUUSD-5M: `session = "all"` (uses full-day session window or disabled filter)
- CHFJPY-1H (provisional): `session = "asia"` — not the default, would need custom sessions_utc config

### Real behavior:
- `require_session_filter` defaults to `True`
- If timestamp is outside all configured session windows → returns "hold" immediately
- Session check is UTC-based

---

## 4. Real Entry Logic (Full Gate)

Entry requires ALL of:
1. **Session filter:** timestamp within configured trading sessions
2. **Volatility cap:** `atr20 <= 2.5 * atr20_ema5` (no spike entries)
3. **VWAP cross recently:** within `cross_lookback_bars=12` bars
4. **Cross direction:** last cross was up (for long) or down (for short)
5. **Retest detected:** low touches VWAP + close > VWAP (long) or high touches + close < VWAP (short)
6. **Volume filter:** configurable (default: `vol >= 1.1 * vol_sma20`)
7. **Trend alignment:** `ema90` slope ≥ flat (for long) or ≤ flat (for short)
8. **VWMA slope:** `vwma10` slope agrees with direction
9. **Nth retest:** default requires 2nd or 3rd retest since cross (retest_count between 2-4)
10. **No open position:** flat only

---

## 5. Real Exit Logic

Exits are managed bar-by-bar via `_update_position_management()`:

### Long exit triggers (in order):
1. **Time stop:** exit if trade doesn't reach +0.5R within `time_stop_bars` (if configured)
2. **Break-even:** move stop to entry at +1R
3. **Partial TP:** take 40% off at +1R
4. **Trailing stop:** activates at +0.7% profit, trails at 0.7%
5. **Band progression exits:** price exceeds VWAP ± k*sigma bands
6. **Loss duration cap:** exit losing trades after N bars (if configured)
7. **Hard stop:** stop_price hit

### Short exits: mirror of long logic

---

## 6. Vectorized Backtest Discrepancies (Bug Report)

### Bug 1: VWAP Construction — **CONFIRMED PRESENT in vectorized engine**
- **Real module:** daily-reset at 23:00 UTC, groupby day_id
- **Vectorized (buggy):** used `df['close'].expanding().mean()` or similar — NOT daily-reset
- **Impact:** VWAP values are wrong for all bars after first bar of each day
- **Affected:** ALL JPY cross backtests (CHFJPY, GBPJPY, AUDJPY)
- **Status:** BUG CONFIRMED — corrected engine re-run PENDING

### Bug 2: Distance Filter Direction — **REPORTED, unverified in vectorized code**
- **Real module:** LONG when price retests VWAP from ABOVE (bullish hold)
- **Vectorized (alleged):** dist filter may have been inverted (entering on wrong side)
- **Impact:** Would reverse signal direction → results meaningless if true
- **Status:** ALLEGED — exact vectorized code not fully reviewed; marked PROVISIONAL

### Bug 3: Session Logic
- **Real module:** UTC session windows, checked per-bar via `in_sessions()`
- **Vectorized:** may have used local timezone or no session filter
- **Impact:** Trades in dead hours → inflated trade count, different win rate
- **Status:** UNVERIFIED

---

## 7. Per-Strategy Parity Status

| Strategy ID | Symbol | TF | Real Module Used | VWAP Correct | Dist Filter | Session | Parity Status |
|-------------|--------|----|-----------------|--------------|-------------|---------|---------------|
| VWAP-MR-XAUUSD-5M | XAUUSD | 5M | ✅ YES | ✅ daily-reset | ✅ correct | ✅ UTC | **PARITY VERIFIED** |
| VWAP-MR-CHFJPY-1H | CHFJPY | 1H | ❌ vectorized | ❌ expanding | ❓ unknown | ❓ unknown | **PARITY UNVERIFIED** |
| VWAP-MR-GBPJPY-1H | GBPJPY | 1H | ❌ vectorized | ❌ expanding | ❓ unknown | ❓ unknown | **PARITY UNVERIFIED** |
| VWAP-MR-AUDJPY-1H | AUDJPY | 1H | ❌ vectorized | ❌ expanding | ❓ unknown | ❓ unknown | **PARITY UNVERIFIED** |
| VWAP-MR-XAUUSD-1H | XAUUSD | 1H | ✅ YES | ✅ daily-reset | ✅ correct | ✅ UTC | **PARITY VERIFIED** (rejected on merit) |

---

## 8. Corrected Engine Status

### What needs to happen:
1. Run CHFJPY-1H, GBPJPY-1H, AUDJPY-1H through `VWAPControlFlipStrategy` (real module)
2. Use same IS/OOS split as original backtests
3. Record: OOS PF, OOS WR, OOS n, MaxDD
4. Apply governance rules:
   - OOS PF >= 1.20 AND n >= 25 → promote to SHADOW
   - OOS PF < 1.00 → REJECT
5. Update `strategy_registry.json` with new results

### Current blocker:
- No 1H OHLCV data files found for CHFJPY, GBPJPY, AUDJPY in `/tmp/trading/`
- Data acquisition required before corrected backtest can run

### Estimated impact of bug:
- If VWAP was expanding: VWAP converges to long-run mean → "signal" fires when price reverts to long-run mean, not intraday VWAP
- This can produce spurious profits on trending pairs (JPY crosses trend strongly)
- PF=5.86 for CHFJPY is almost certainly inflated; real PF likely 1.0–2.0 range at best

---

## 9. Recommendations

1. **BLOCKED:** Do not run any JPY cross strategy on live accounts until corrected backtest completes
2. **ACTION (Jefe):** Acquire 1H OHLCV data for CHFJPY, GBPJPY, AUDJPY (minimum 6 months)
3. **ACTION (engine):** Re-run backtests using `VWAPControlFlipStrategy` with same IS/OOS split
4. **ACTION (Jefe):** Create Track A demo account `OPTION_A_XAUUSD_5M_SHADOW` on GatesFX for XAUUSD-5M shadow validation
5. **MONITOR:** XAUUSD-5M shadow — accumulate OOS trades until n=25, then reassess for live promotion

---

*Audit generated from direct source inspection of `/tmp/trading/capital_allocation_ai/vwap_control_flip_strategy.py`*  
*Real module confirmed: daily-reset VWAP at 23:00 UTC, correct dist filter direction, UTC session windows*
