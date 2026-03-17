# MULTI-SYMBOL EDGE CLASSIFICATION

Generated: 2026-03-17 16:54:43.654654

## SIGMA FIX NOTE
Forex pairs (EURUSD, GBPJPY, etc.) from Yahoo Finance have `volume≈0`.
Fixed by using equal-weight expanding VWAP + rolling std as sigma fallback.

## SUMMARY TABLE

| Symbol | TF | ATR | BE_WR | IS WR | OOS WR | OOS PF | OOS DD | Classification | Action |
|--------|-----|-----|-------|-------|--------|--------|--------|----------------|--------|
| GBPJPY | 1H | 0.3794 | 25.0% | 49.7% | 52.6% | 2.81 | 0.0% | PROFITABLE_DEPLOYABLE | PROMOTE_TO_SHADOW |
| GBPJPY | 4H | 0.6692 | 33.3% | 58.8% | 60.0% | 1.96 | 0.0% | STRUCTURALLY_DEAD | RETIRE |
| AUDJPY | 1H | 0.2321 | 25.0% | 60.9% | 42.4% | 2.07 | 0.0% | PROFITABLE_DEPLOYABLE | PROMOTE_TO_SHADOW |
| AUDJPY | 4H | 0.4070 | 83.3% | 94.5% | 79.2% | 0.76 | 0.2% | STRUCTURALLY_DEAD | RETIRE |
| EURJPY | 1H | 0.3041 | 20.0% | 46.7% | 14.3% | 0.63 | 0.0% | STRUCTURALLY_DEAD | RETIRE |
| EURJPY | 4H | 0.5379 | 38.5% | 67.7% | 77.3% | 4.46 | 0.0% | STRUCTURALLY_DEAD | RETIRE |
| CHFJPY | 1H | 0.3067 | 25.0% | 50.4% | 65.2% | 5.86 | 0.0% | PROFITABLE_DEPLOYABLE | PROMOTE_TO_SHADOW |
| CHFJPY | 4H | 0.5346 | 33.3% | 50.0% | — | — | — | STRUCTURALLY_DEAD | RETIRE |
| JP225_NI | 1H | 225.0727 | 14.3% | 33.3% | 20.0% | 0.38 | 0.3% | STRUCTURALLY_DEAD | RETIRE |
| JP225_NI | 4H | 401.8151 | 7.7% | 23.1% | 8.3% | 0.48 | 0.7% | STRUCTURALLY_DEAD | RETIRE |
| US30 | 1H | 111.0899 | 7.7% | 27.3% | 0.0% | 0.00 | 0.3% | STRUCTURALLY_DEAD | RETIRE |
| US30 | 4H | 196.6944 | 20.0% | 26.7% | 42.9% | 0.21 | 0.2% | STRUCTURALLY_DEAD | RETIRE |
| NAS100 | 1H | 78.8427 | 14.3% | 25.0% | 11.1% | 0.22 | 0.2% | STRUCTURALLY_DEAD | RETIRE |
| NAS100 | 4H | 138.6414 | 20.0% | 43.8% | 30.0% | 0.30 | 0.2% | STRUCTURALLY_DEAD | RETIRE |
| AUDCAD | 1H | 0.0021 | 38.5% | 86.7% | 80.0% | 5.54 | 0.0% | STRUCTURALLY_DEAD | RETIRE |
| AUDCAD | 4H | 0.0031 | 83.3% | 98.2% | 81.0% | 0.76 | 0.2% | STRUCTURALLY_DEAD | RETIRE |
| EURUSD | 1H | 0.0015 | 7.7% | 17.2% | 0.0% | 0.00 | 0.1% | STRUCTURALLY_DEAD | RETIRE |
| EURUSD | 4H | 0.0027 | 25.0% | 54.2% | 18.2% | 0.44 | 0.0% | STRUCTURALLY_DEAD | RETIRE |
| EURTRY | 1H | 0.1390 | 83.3% | 99.0% | 98.5% | 25.70 | 0.0% | PROFITABLE_DEPLOYABLE | PROMOTE_TO_SHADOW |
| EURTRY | 4H | 0.2162 | 80.0% | 97.9% | 91.2% | 3.04 | 0.1% | PROFITABLE_DEPLOYABLE | PROMOTE_TO_SHADOW |
| XAUUSD | 1H | 13.9205 | 7.7% | 20.8% | 8.3% | 1.15 | 0.1% | STRUCTURALLY_DEAD | RETIRE |
| XAUUSD | 4H | 23.5429 | 33.3% | 64.5% | 41.2% | 0.71 | 0.3% | STRUCTURALLY_DEAD | RETIRE |
| XPTUSD | 1H | 10.9004 | 25.0% | 45.2% | 46.2% | 2.00 | 0.1% | STRUCTURALLY_DEAD | RETIRE |
| XPTUSD | 4H | 18.6102 | 38.5% | 88.9% | 38.5% | 1.09 | 0.2% | STRUCTURALLY_DEAD | RETIRE |
| BTCUSD | 1H | 639.3191 | 7.7% | 9.4% | 14.3% | -0.06 | 4.0% | STRUCTURALLY_DEAD | RETIRE |
| BTCUSD | 4H | 1188.9587 | — | — | — | — | — | STRUCTURALLY_DEAD | RETIRE |
| BCHUSD | 1H | 5.3746 | 7.7% | 20.4% | 10.5% | 1.77 | 0.2% | STRUCTURALLY_DEAD | RETIRE |
| BCHUSD | 4H | 10.1900 | 20.0% | 44.4% | 33.3% | 2.14 | 0.1% | STRUCTURALLY_DEAD | RETIRE |
| BTCEUR | 1H | 583.5612 | 7.7% | 23.1% | 0.0% | 0.00 | 1.6% | STRUCTURALLY_DEAD | RETIRE |
| BTCEUR | 4H | 1081.6166 | 33.3% | 77.8% | 12.5% | 0.10 | 2.3% | STRUCTURALLY_DEAD | RETIRE |

---

## DEPLOYABLE CANDIDATES

### ★ EURTRY 1H
- **Params**: TP=0.4× SL=2.0× Band=2.0 Dist=0.8 REXP=1.0 RSI=35-65 Session=asia
- **IS**: n=105 WR=99.0% PF=21.21 DD=0.1% EXP=$1.58
- **OOS**: n=67 WR=98.5% PF=25.70 DD=0.0% EXP=$1.20 | Robust PF=5.923448973196317

### ★ CHFJPY 1H
- **Params**: TP=0.75× SL=0.25× Band=2.0 Dist=0.5 REXP=2.0 RSI=35-65 Session=asia
- **IS**: n=135 WR=50.4% PF=2.90 DD=0.0% EXP=$0.43
- **OOS**: n=66 WR=65.2% PF=5.86 DD=0.0% EXP=$0.62 | Robust PF=4.946579322035271

### ★ EURTRY 4H
- **Params**: TP=0.5× SL=2.0× Band=2.0 Dist=0.3 REXP=2.0 RSI=30-70 Session=all
- **IS**: n=48 WR=97.9% PF=10.47 DD=0.1% EXP=$2.64
- **OOS**: n=34 WR=91.2% PF=3.04 DD=0.1% EXP=$1.49 | Robust PF=1.5103883370487345

### ★ GBPJPY 1H
- **Params**: TP=0.75× SL=0.25× Band=2.0 Dist=0.5 REXP=2.0 RSI=35-65 Session=asia
- **IS**: n=143 WR=49.7% PF=2.91 DD=0.0% EXP=$0.50
- **OOS**: n=76 WR=52.6% PF=2.81 DD=0.0% EXP=$0.37 | Robust PF=2.1556178722964683

### ★ AUDJPY 1H
- **Params**: TP=0.75× SL=0.25× Band=2.0 Dist=0.3 REXP=1.5 RSI=35-65 Session=asia
- **IS**: n=69 WR=60.9% PF=4.26 DD=0.0% EXP=$0.74
- **OOS**: n=33 WR=42.4% PF=2.07 DD=0.0% EXP=$0.34 | Robust PF=1.7369317713215688

## QUARANTINE (HIGH_WR_NEGATIVE_EV)

NONE

## REDESIGN CANDIDATES

NONE

## RETIRED (STRUCTURALLY_DEAD)

- **GBPJPY 4H**: Insufficient OOS trades: n=5 (need ≥25)
- **AUDJPY 4H**: Insufficient OOS trades: n=24 (need ≥25)
- **EURJPY 1H**: Insufficient OOS trades: n=7 (need ≥25)
- **EURJPY 4H**: Insufficient OOS trades: n=22 (need ≥25)
- **CHFJPY 4H**: No OOS trades or OOS artifact
- **JP225_NI 1H**: Insufficient OOS trades: n=5 (need ≥25)
- **JP225_NI 4H**: Insufficient OOS trades: n=12 (need ≥25)
- **US30 1H**: Insufficient OOS trades: n=5 (need ≥25)
- **US30 4H**: Insufficient OOS trades: n=7 (need ≥25)
- **NAS100 1H**: Insufficient OOS trades: n=9 (need ≥25)
- **NAS100 4H**: Insufficient OOS trades: n=10 (need ≥25)
- **AUDCAD 1H**: Insufficient OOS trades: n=15 (need ≥25)
- **AUDCAD 4H**: Insufficient OOS trades: n=21 (need ≥25)
- **EURUSD 1H**: Insufficient OOS trades: n=13 (need ≥25)
- **EURUSD 4H**: Insufficient OOS trades: n=11 (need ≥25)
- **XAUUSD 1H**: Insufficient OOS trades: n=12 (need ≥25)
- **XAUUSD 4H**: Insufficient OOS trades: n=17 (need ≥25)
- **XPTUSD 1H**: Insufficient OOS trades: n=13 (need ≥25)
- **XPTUSD 4H**: Insufficient OOS trades: n=13 (need ≥25)
- **BTCUSD 1H**: Insufficient OOS trades: n=21 (need ≥25)
- **BTCUSD 4H**: No filter combo produced ≥15 IS trades
- **BCHUSD 1H**: Insufficient OOS trades: n=19 (need ≥25)
- **BCHUSD 4H**: Insufficient OOS trades: n=9 (need ≥25)
- **BTCEUR 1H**: Insufficient OOS trades: n=8 (need ≥25)
- **BTCEUR 4H**: Insufficient OOS trades: n=8 (need ≥25)

---

## ARCHITECTURE VERDICT

WR≥80%+EV>0 zone exists on: EURTRY, AUDJPY, AUDCAD

### Redesign Paths
- **A. Scale-out exits** — 50% at TP×0.3, trail rest with ATR stop
- **B. Regime filter** — only trade bottom 40th percentile sigma (range markets)
- **C. 2nd touch filter** — enter only on 2nd consecutive band touch
- **D. Breakout family** — XAUUSD/BTC TP=1.5× SL=0.25× positive EV at WR~25%
