const express = require('express');
const fs = require('fs');
const path = require('path');
const { readRuntimeSnapshot, configuredConnections } = require('../adapters/tradelocker_token_api');
const app = express();

app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

function requireDashboardToken(req, res, next) {
  const expected = process.env.DASHBOARD_ACCESS_TOKEN?.trim();
  if (!expected) return next();
  const supplied = String(req.get('x-dashboard-token') || req.query.token || '').trim();
  if (supplied === expected) return next();
  return res.status(401).json({ error: 'dashboard access token required' });
}

app.use('/api', requireDashboardToken);

// Strategy registry API
app.get('/api/strategies', (req, res) => {
  const registry = JSON.parse(fs.readFileSync(path.join(__dirname, '../strategy_registry.json'), 'utf8'));
  res.json(registry);
});

function accountsFromFilters() {
  return configuredConnections().flatMap((connection) => {
    const accountFilter = connection.accountFilter || '';
    return accountFilter.split(',').map((value) => value.trim()).filter(Boolean).map((id) => ({
      id,
      name: `${connection.label} ${id}`,
      broker: connection.label,
      environment: connection.environment,
      symbols: ["XAUUSD"],
      balance: null,
      equity: null,
      drawdown_pct: null,
      positions_count: 0,
      orders_count: 0,
      status: "AUTH_BLOCKED",
      last_seen: null,
      kill_switch: {dd_10: "reduce", dd_12: "pause", dd_15: "stop"},
      broker_connected: false,
      note: "TradeLocker token API auth blocked. Refresh credentials/token before trusting account state."
    }));
  });
}

function fallbackRuntime(error) {
  const environments = configuredConnections().map((connection) => ({
    id: connection.id,
    label: connection.label,
    environment: connection.environment,
    baseUrl: connection.baseUrl,
    status: 'AUTH_BLOCKED',
    visible_accounts: 0,
    total_accounts: null,
    account_filter: connection.accountFilter || 'ALL',
    last_sync: null,
    error: error.message,
    accounts: [],
  }));

  return {
    timestamp: new Date().toISOString(),
    source: 'fallback_static',
    accounts: accountsFromFilters(),
    environments,
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
      {level: "CRITICAL", msg: "TradeLocker live/demo account sync is auth-blocked"},
      {level: "HIGH", msg: "JPY cross backtest integrity unresolved — 3 strategies PROVISIONAL"},
      {level: "HIGH", msg: `TradeLocker token API unavailable: ${error.message}`},
      {level: "MEDIUM", msg: "Dashboard is deployed read-only; no broker execution route exists"}
    ]
  };
}

function withEngineState(snapshot) {
  const riskyAccounts = snapshot.accounts.filter((account) => account.positions_count > 0 || account.orders_count > 0);
  const connected = snapshot.environments.filter((environment) => environment.status === 'CONNECTED');
  const blocked = snapshot.environments.filter((environment) => environment.status !== 'CONNECTED');
  return {
    ...snapshot,
    engine: {
      status: "RUNNING",
      mode: "SHADOW",
      approved_live: 0,
      shadow_count: 1,
      provisional_count: 3,
      rejected_count: 1,
      pending_integrity_reset: true,
      note: "Dashboard is reading TradeLocker directly through JWT token auth. Execution remains disabled."
    },
    alerts: [
      ...(connected.length
        ? [{level: "HIGH", msg: `TradeLocker read-only connected: ${connected.map((environment) => `${environment.environment} ${environment.visible_accounts}/${environment.total_accounts}`).join(', ')}`}]
        : [{level: "CRITICAL", msg: "No TradeLocker environment is currently authenticated"}]),
      ...blocked.map((environment) => ({level: "HIGH", msg: `${environment.environment} auth blocked: ${environment.error}`})),
      {level: "HIGH", msg: "JPY cross backtest integrity unresolved — 3 strategies PROVISIONAL"},
      ...(riskyAccounts.length
        ? [{level: "CRITICAL", msg: `Open TradeLocker risk detected on ${riskyAccounts.length} account(s)`}]
        : []),
      {level: "MEDIUM", msg: "Kill switch buttons only log dashboard intent; broker execution remains disabled"}
    ]
  };
}

// Runtime state from TradeLocker token API. Read-only only.
app.get('/api/runtime', async (req, res) => {
  try {
    const snapshot = await readRuntimeSnapshot();
    res.json(withEngineState(snapshot));
  } catch (error) {
    res.json(fallbackRuntime(error));
  }
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
    note: 'Kill switch command logged. TradeLocker execution remains disabled.'
  });
});

// Risk state
app.get('/api/risk', async (req, res) => {
  let accounts = accountsFromFilters();
  try {
    const snapshot = await readRuntimeSnapshot();
    if (snapshot.accounts.length) accounts = snapshot.accounts;
  } catch {
    // Static account filters are enough for guardrail rendering during auth outages.
  }
  const guardrails = accounts.flatMap((account) => [
    {rule: "10% account DD -> reduce size 60%", status: "ARMED", account: account.id, environment: account.environment},
    {rule: "12% account DD -> pause all signals", status: "ARMED", account: account.id, environment: account.environment},
    {rule: "15% account DD -> full stop", status: "ARMED", account: account.id, environment: account.environment},
  ]);
  res.json({
    global_kill_armed: false,
    max_daily_loss_pct: 5,
    max_position_exposure_pct: 10,
    guardrails
  });
});

// Detailed health check
app.get('/api/health/detailed', async (req, res) => {
  let brokerStatus = {
    status: 'DISCONNECTED',
    note: 'TradeLocker token API unavailable',
    accountStatus: 'STALE',
    lastSync: null,
  };

  try {
    const snapshot = await readRuntimeSnapshot();
    const connected = snapshot.environments.filter((environment) => environment.status === 'CONNECTED');
    const blocked = snapshot.environments.filter((environment) => environment.status !== 'CONNECTED');
    const accountNote = snapshot.environments.map((environment) => (
      environment.status === 'CONNECTED'
        ? `${environment.environment}: ${environment.visible_accounts}/${environment.total_accounts} visible`
        : `${environment.environment}: ${environment.error}`
    )).join(' | ');
    brokerStatus = {
      status: connected.length ? (blocked.length ? 'DEGRADED' : 'HEALTHY') : 'DISCONNECTED',
      note: accountNote,
      accountStatus: snapshot.accounts.length ? 'HEALTHY' : 'STALE',
      lastSync: snapshot.timestamp,
    };
  } catch (error) {
    brokerStatus.note = `TradeLocker token API unavailable: ${error.message}`;
  }

  res.json({
    components: [
      {name: 'Dashboard API',    status: 'HEALTHY',      latency_ms: 1},
      {name: 'Strategy Engine',  status: 'SHADOW_MODE',  note: 'No live strategies approved'},
      {name: 'Data Feed',        status: brokerStatus.status, note: brokerStatus.note},
      {name: 'Broker Adapter',   status: brokerStatus.status, note: 'TradeLocker JWT token API, read-only endpoints'},
      {name: 'TradeLocker Accounts', status: brokerStatus.accountStatus, note: brokerStatus.note, last_sync: brokerStatus.lastSync},
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
