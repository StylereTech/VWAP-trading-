const express = require('express');
const fs = require('fs');
const path = require('path');
const app = express();

app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

// Strategy registry API
app.get('/api/strategies', (req, res) => {
  const registry = JSON.parse(fs.readFileSync(path.join(__dirname, '../strategy_registry.json'), 'utf8'));
  res.json(registry);
});

// Runtime state (would connect to TraderLocker API in production)
app.get('/api/runtime', (req, res) => {
  res.json({
    timestamp: new Date().toISOString(),
    accounts: [
      {
        id: "703060",
        name: "GFX-D1-S3S4",
        broker: "GATESFX",
        symbols: ["GBPJPY", "EURUSD", "XAUUSD"],
        balance: null,
        equity: null,
        drawdown_pct: null,
        status: "UNKNOWN — NEEDS AUDIT",
        last_seen: null,
        kill_switch: {dd_10: "reduce", dd_12: "pause", dd_15: "stop"},
        broker_connected: false,
        note: "TraderLocker API blocked from current environment. Manual audit required at gatesfx.com"
      },
      {
        id: "703062",
        name: "GFX-D17-S4",
        broker: "GATESFX",
        symbols: ["BTCEUR"],
        balance: null,
        equity: null,
        drawdown_pct: null,
        status: "UNKNOWN — NEEDS AUDIT",
        last_seen: null,
        kill_switch: {dd_10: "reduce", dd_12: "pause", dd_15: "stop"},
        broker_connected: false,
        note: "TraderLocker API blocked from current environment. Manual audit required at gatesfx.com"
      }
    ],
    engine: {
      status: "RUNNING",
      mode: "SHADOW",
      approved_live: 0,
      shadow_count: 1,
      provisional_count: 3,
      rejected_count: 1,
      pending_integrity_reset: true,
      note: "JPY cross backtest integrity reset in progress. No live strategies active from this engine."
    },
    alerts: [
      {level: "CRITICAL", msg: "Live accounts 703060 + 703062 have not been audited for 17+ days"},
      {level: "HIGH", msg: "JPY cross backtest integrity unresolved — 3 strategies PROVISIONAL"},
      {level: "HIGH", msg: "TraderLocker API unreachable from current environment"},
      {level: "MEDIUM", msg: "Track A demo account OPTION_A_XAUUSD_5M_SHADOW not yet created by Jefe"}
    ]
  });
});

// Kill switch API
app.post('/api/kill', (req, res) => {
  const {account_id, action} = req.body;
  console.log(`KILL SWITCH: account=${account_id} action=${action}`);
  res.json({
    status: 'logged',
    account_id,
    action,
    timestamp: new Date().toISOString(),
    note: 'Kill switch command logged. TraderLocker execution requires API access.'
  });
});

// Risk state
app.get('/api/risk', (req, res) => {
  res.json({
    global_kill_armed: false,
    max_daily_loss_pct: 5,
    max_position_exposure_pct: 10,
    guardrails: [
      {rule: "10% account DD → reduce size 60%", status: "ARMED", account: "703060"},
      {rule: "12% account DD → pause all signals", status: "ARMED", account: "703060"},
      {rule: "15% account DD → full stop", status: "ARMED", account: "703060"},
      {rule: "10% account DD → reduce size 60%", status: "ARMED", account: "703062"},
      {rule: "12% account DD → pause all signals", status: "ARMED", account: "703062"},
      {rule: "15% account DD → full stop", status: "ARMED", account: "703062"}
    ]
  });
});

// Detailed health check
app.get('/api/health/detailed', (req, res) => {
  res.json({
    components: [
      {name: 'Dashboard API',    status: 'HEALTHY',      latency_ms: 1},
      {name: 'Strategy Engine',  status: 'SHADOW_MODE',  note: 'No live strategies approved'},
      {name: 'Data Feed',        status: 'DISCONNECTED', note: 'TraderLocker blocked from WSL2 sandbox'},
      {name: 'Broker Adapter',   status: 'DISCONNECTED', note: 'TraderLocker API unreachable — 403 from WSL2'},
      {name: 'Account 703060',   status: 'STALE',        note: '17+ days since last sync', last_sync: null},
      {name: 'Account 703062',   status: 'STALE',        note: '17+ days since last sync', last_sync: null},
      {name: 'Integrity Reset',  status: 'IN_PROGRESS',  note: 'JPY cross corrected backtest pending'},
      {name: 'Track A Demo',     status: 'NOT_CREATED',  note: 'Jefe must create OPTION_A_XAUUSD_5M_SHADOW at gatesfx.com'},
    ],
    integrity_reset: {
      status: 'IN_PROGRESS',
      started: '2026-03-17',
      bug_found: 'expanding VWAP + inverted dist filter in vectorized backtest',
      symbols_pending: ['CHFJPY', 'GBPJPY', 'AUDJPY', 'USDJPY', 'CADJPY', 'NZDJPY', 'EURJPY', 'AUDCAD'],
      estimated_completion: 'Requires corrected backtest run (15-min compute job)',
    }
  });
});

// Health check
app.get('/health', (req, res) => {
  res.json({status: 'ok', timestamp: new Date().toISOString(), mode: 'SHADOW'});
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Trading Dashboard running on http://localhost:${PORT}`));
