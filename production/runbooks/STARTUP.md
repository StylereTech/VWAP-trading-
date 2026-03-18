# Trading Ops Startup Runbook

## Prerequisites
- [ ] TraderLocker API accessible (not blocked)
- [ ] Accounts 703060 + 703062 audited (see ACCOUNT_RECOVERY.md)
- [ ] At least one strategy status = APPROVED (not PROVISIONAL/SHADOW)
- [ ] Integrity reset complete (corrected JPY backtest results available)
- [ ] Risk controls config verified (production/risk_controls.json)

## Startup Sequence

### Step 1 — Verify environment
```bash
cd /tmp/trading/production
node -e "require('./adapters/traderlocker_adapter').ACCOUNTS['703060'].getConnectionState()"
```
Expected: `connected: true` — if false, resolve before proceeding.

### Step 2 — Start dashboard
```bash
cd /tmp/trading/production/dashboard
node server.js
# Access at http://localhost:3000
```

### Step 3 — Confirm no CRITICAL alerts
- Check `/api/runtime` → alerts array
- All CRITICAL alerts must be resolved before enabling live strategies

### Step 4 — Verify approved strategies
- Check `/api/strategies` → strategies where status = "APPROVED"
- Minimum: 1 strategy must be APPROVED with verified parity
- DO NOT proceed if all strategies are PROVISIONAL or SHADOW

### Step 5 — Enable live routing
- Currently: no mechanism to enable live routing (broker adapter scaffolded only)
- When adapter is connected: update `risk_controls.json` → `blocked_in_integrity_reset: false`
- Restart strategy engine with live mode flag

## Emergency Shutdown
```bash
# Kill switch via dashboard API
curl -X POST http://localhost:3000/api/kill \
  -H "Content-Type: application/json" \
  -d '{"account_id": "ALL", "action": "GLOBAL_KILL"}'
```

## Recovery After Crash
1. Check dashboard health at `/health`
2. Check account state (ACCOUNT_RECOVERY.md step 1)
3. Re-verify open positions vs expected state
4. If drift found: reconcile before resuming
