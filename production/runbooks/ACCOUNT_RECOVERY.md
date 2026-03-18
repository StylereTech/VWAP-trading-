# Account Recovery Runbook

## Context
Live accounts 703060 and 703062 have been operating without audit for 17+ days.
Their current state (balance, equity, positions, drawdown) is UNKNOWN.

## DO NOT assume these accounts are healthy.

---

## Step 1 — Login and verify

Go to: https://gatesfx.com  
Email: ryanlawrence217@hotmail.com  
Password: Life2026$$  
Server: GATESFX

For each account (703060, 703062):

1. Check current **Balance**
2. Check current **Equity**
3. Calculate **Drawdown**: `(Balance - Equity) / Balance × 100`
4. Check open **Positions** — are any open?
5. Check **Trade History** — what ran over the last 17 days?

---

## Step 2 — Apply kill-switch rules immediately

| Drawdown | Required action |
|----------|----------------|
| ≥ 15%    | FULL STOP — close all positions, disable auto-trading |
| ≥ 12%    | PAUSE — disable auto-trading, do not open new positions |
| ≥ 10%    | REDUCE — if position sizing available, reduce 60% |
| < 10%    | CONTINUE — resume normal operation |

---

## Step 3 — Record findings

Update `/tmp/trading/production/strategy_registry.json`:
- Set `last_audited` to today's date
- Update `balance` to current balance
- Set `audit_needed: false`
- Set `live_status` to `ACTIVE` or `PAUSED` based on drawdown rules

---

## Step 4 — Reconnect monitoring

Once TraderLocker API is accessible (non-WSL environment):
```bash
cd /tmp/trading/production
node -e "
const {ACCOUNTS} = require('./adapters/traderlocker_adapter');
ACCOUNTS['703060'].connect().then(() =>
  ACCOUNTS['703060'].getAccountSnapshot().then(s => console.log('703060:', s))
);
"
```

---

## Step 5 — Resume or quarantine

- If DD < 10%: **RESUME** — re-enable tracking, update registry
- If DD 10–15%: **RESTRICTED** — monitor only, no new trades
- If DD > 15%: **QUARANTINE** — flatten, document P&L, remove from live rotation
