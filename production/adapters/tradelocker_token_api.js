const fs = require('fs');
const path = require('path');

const CREDENTIALS_PATH = process.env.STYLERE_CREDENTIALS_PATH || `${process.env.HOME}/.config/stylere/credentials.env`;
const DEFAULT_SYMBOL = process.env.TRADELOCKER_SYMBOL || 'XAUUSD';
const DEFAULT_CONNECTIONS = [
  {
    id: 'live',
    label: 'GatesFX Live',
    environment: 'LIVE',
    baseUrl: process.env.TRADELOCKER_LIVE_API_BASE || 'https://live.tradelocker.com/backend-api',
    jwtPath: process.env.TRADELOCKER_LIVE_JWT_PATH || process.env.TRADELOCKER_JWT_PATH || `${process.env.HOME}/.config/stylere/tradelocker.jwt.json`,
    accountFilter: process.env.TRADELOCKER_LIVE_ACCOUNTS || process.env.TRADELOCKER_DASHBOARD_ACCOUNTS || '703060,703062',
  },
  {
    id: 'demo',
    label: 'GatesFX Demo/Test',
    environment: 'DEMO',
    baseUrl: process.env.TRADELOCKER_DEMO_API_BASE || 'https://demo.tradelocker.com/backend-api',
    jwtPath: process.env.TRADELOCKER_DEMO_JWT_PATH || `${process.env.HOME}/.config/stylere/tradelocker-demo.jwt.json`,
    accountFilter: process.env.TRADELOCKER_DEMO_ACCOUNTS || '',
  },
];

const jwtCache = new Map();

function parseEnvFile(filePath) {
  if (!filePath || !fs.existsSync(filePath)) return {};
  const parsed = {};
  const raw = fs.readFileSync(filePath, 'utf8');
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const index = trimmed.indexOf('=');
    if (index === -1) continue;
    parsed[trimmed.slice(0, index)] = trimmed.slice(index + 1);
  }
  return parsed;
}

function envKey(connection, suffix) {
  return `TRADELOCKER_${connection.environment}_${suffix}`;
}

function configuredConnections() {
  if (process.env.TRADELOCKER_CONNECTIONS_JSON) {
    const parsed = JSON.parse(process.env.TRADELOCKER_CONNECTIONS_JSON);
    return parsed.map((connection, index) => ({
      id: connection.id || `connection_${index + 1}`,
      label: connection.label || connection.id || `TradeLocker ${index + 1}`,
      environment: (connection.environment || connection.id || `CONNECTION_${index + 1}`).toUpperCase(),
      baseUrl: connection.baseUrl,
      jwtPath: connection.jwtPath,
      accountFilter: connection.accountFilter || '',
    }));
  }
  return DEFAULT_CONNECTIONS;
}

function readJwt(connection) {
  const cached = jwtCache.get(connection.id);
  if (cached?.accessToken) return cached;

  const accessToken = process.env[envKey(connection, 'ACCESS_TOKEN')];
  const refreshToken = process.env[envKey(connection, 'REFRESH_TOKEN')];
  if (accessToken) {
    const jwt = {
      accessToken,
      refreshToken,
      expireDate: process.env[envKey(connection, 'EXPIRE_DATE')],
      baseUrl: connection.baseUrl,
      source: 'environment',
    };
    jwtCache.set(connection.id, jwt);
    return jwt;
  }

  if (!connection.jwtPath || !fs.existsSync(connection.jwtPath)) return null;
  const jwt = JSON.parse(fs.readFileSync(connection.jwtPath, 'utf8'));
  if (!jwt.accessToken) return null;
  jwtCache.set(connection.id, jwt);
  return jwt;
}

function writeJwt(connection, tokens) {
  const payload = {
    ...tokens,
    fetchedAt: new Date().toISOString(),
    baseUrl: connection.baseUrl,
  };

  jwtCache.set(connection.id, payload);
  if (connection.jwtPath) {
    fs.mkdirSync(path.dirname(connection.jwtPath), { recursive: true });
    fs.writeFileSync(connection.jwtPath, `${JSON.stringify(payload, null, 2)}\n`, { mode: 0o600 });
    fs.chmodSync(connection.jwtPath, 0o600);
  }
  return payload;
}

function tokenLooksExpired(jwt) {
  const expireMs = Date.parse(jwt?.expireDate || jwt?.expiresAt || '');
  return Number.isFinite(expireMs) && expireMs <= Date.now() + 60_000;
}

function credentialCandidates(connection) {
  const fileEnv = parseEnvFile(CREDENTIALS_PATH);
  const envPrefix = `GATESFX_TRADELOCKER_${connection.environment}`;
  const email = process.env[`${envPrefix}_EMAIL`] || process.env.GATESFX_TRADELOCKER_EMAIL ||
    process.env.GATESFX_EMAIL || fileEnv[`${envPrefix}_EMAIL`] ||
    fileEnv.GATESFX_TRADELOCKER_EMAIL || fileEnv.GATESFX_EMAIL;
  const server = process.env[`${envPrefix}_SERVER`] || process.env.GATESFX_TRADELOCKER_SERVER ||
    process.env.GATESFX_SERVER || fileEnv[`${envPrefix}_SERVER`] ||
    fileEnv.GATESFX_TRADELOCKER_SERVER || 'GATESFX';
  const passwords = [
    process.env[`${envPrefix}_PASSWORD`],
    process.env.GATESFX_TRADELOCKER_PASSWORD,
    process.env.GATESFX_PASSWORD,
    fileEnv[`${envPrefix}_PASSWORD`],
    fileEnv.GATESFX_TRADELOCKER_PASSWORD,
    fileEnv.GATESFX_PASSWORD,
  ].filter(Boolean);

  if (!email || passwords.length === 0 || !server) {
    throw new Error(`missing TradeLocker credentials for ${connection.label}`);
  }

  return [...new Set(passwords)].map((password) => ({ email, password, server }));
}

async function readResponse(response) {
  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text.slice(0, 1000) };
  }
}

function errorMessage(body) {
  return body?.errmsg || body?.message || JSON.stringify(body).slice(0, 300);
}

async function fetchJwtWithCredentials(connection) {
  let lastError = null;
  for (const credentials of credentialCandidates(connection)) {
    const response = await fetch(`${connection.baseUrl}/auth/jwt/token`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(credentials),
    });
    const body = await readResponse(response);
    if (response.ok) return writeJwt(connection, body);
    lastError = `${response.status} /auth/jwt/token: ${errorMessage(body)}`;
  }
  throw new Error(lastError || 'TradeLocker JWT auth failed');
}

async function refreshJwt(connection, jwt) {
  if (!jwt?.refreshToken) throw new Error('missing refreshToken');
  const response = await fetch(`${connection.baseUrl}/auth/jwt/refresh`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ refreshToken: jwt.refreshToken }),
  });
  const body = await readResponse(response);
  if (!response.ok) {
    throw new Error(`${response.status} /auth/jwt/refresh: ${errorMessage(body)}`);
  }
  return writeJwt(connection, body);
}

async function ensureJwt(connection) {
  const jwt = readJwt(connection);
  if (!jwt) return fetchJwtWithCredentials(connection);
  if (!tokenLooksExpired(jwt)) return jwt;
  try {
    return await refreshJwt(connection, jwt);
  } catch {
    return fetchJwtWithCredentials(connection);
  }
}

async function request(connection, pathname, { accNum, retry = true } = {}) {
  const jwt = await ensureJwt(connection);
  const response = await fetch(`${connection.baseUrl}${pathname}`, {
    headers: {
      Authorization: `Bearer ${jwt.accessToken}`,
      ...(accNum ? { accNum: String(accNum) } : {}),
    },
  });
  const body = await readResponse(response);
  if (!response.ok) {
    if (retry && response.status === 401) {
      try {
        await refreshJwt(connection, readJwt(connection));
      } catch {
        await fetchJwtWithCredentials(connection);
      }
      return request(connection, pathname, { accNum, retry: false });
    }
    throw new Error(`${response.status} ${pathname}: ${errorMessage(body)}`);
  }
  return body;
}

function columnsToObject(config, values) {
  const columns = config?.columns || [];
  const result = {};
  for (let index = 0; index < columns.length; index += 1) {
    result[columns[index].id] = values?.[index] ?? null;
  }
  return result;
}

function numberOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function configuredAccounts(connection, allAccounts) {
  if (!connection.accountFilter) return allAccounts;
  const wanted = new Set(String(connection.accountFilter).split(',').map((value) => value.trim()).filter(Boolean));
  return allAccounts.filter((account) => wanted.has(String(account.id)) || wanted.has(String(account.accNum)));
}

async function readAccountSnapshot(connection, account, config) {
  const accNum = account.accNum;
  const [stateResponse, positionsResponse, ordersResponse] = await Promise.all([
    request(connection, `/trade/accounts/${account.id}/state`, { accNum }),
    request(connection, `/trade/accounts/${account.id}/positions`, { accNum }),
    request(connection, `/trade/accounts/${account.id}/orders`, { accNum }),
  ]);

  const state = columnsToObject(config.d.accountDetailsConfig, stateResponse.d.accountDetailsData);
  const positions = (positionsResponse.d.positions || []).map((row) => columnsToObject(config.d.positionsConfig, row));
  const orders = (ordersResponse.d.orders || []).map((row) => columnsToObject(config.d.ordersConfig, row));
  const balance = numberOrNull(state.balance ?? state.cashBalance ?? account.accountBalance);
  const equity = numberOrNull(state.projectedBalance ?? state.balance ?? account.accountBalance);
  const drawdown = balance && equity !== null ? Number((((balance - equity) / balance) * 100).toFixed(2)) : null;

  return {
    id: String(account.id),
    accNum: Number(accNum),
    name: account.name || `TradeLocker ${account.id}`,
    broker: connection.label,
    environment: connection.environment,
    symbols: [DEFAULT_SYMBOL],
    balance,
    equity,
    availableFunds: numberOrNull(state.availableFunds),
    drawdown_pct: drawdown,
    positions_count: positions.length,
    orders_count: orders.length,
    status: positions.length || orders.length ? 'MONITORING_RISK' : 'READ_ONLY_CONNECTED',
    last_seen: new Date().toISOString(),
    kill_switch: { dd_10: 'reduce', dd_12: 'pause', dd_15: 'stop' },
    broker_connected: true,
    note: `Read-only ${connection.environment} TradeLocker sync. positions=${positions.length}, orders=${orders.length}`,
    state,
    positions,
    orders,
  };
}

async function readConnectionSnapshot(connection) {
  const [config, accountsResponse] = await Promise.all([
    request(connection, '/trade/config'),
    request(connection, '/auth/jwt/all-accounts'),
  ]);

  const allAccounts = accountsResponse.accounts || accountsResponse.d?.accounts || [];
  const accounts = configuredAccounts(connection, allAccounts);
  const snapshots = [];
  for (const account of accounts) {
    snapshots.push(await readAccountSnapshot(connection, account, config));
  }

  return {
    id: connection.id,
    label: connection.label,
    environment: connection.environment,
    baseUrl: connection.baseUrl,
    status: 'CONNECTED',
    visible_accounts: snapshots.length,
    total_accounts: allAccounts.length,
    account_filter: connection.accountFilter || 'ALL',
    last_sync: new Date().toISOString(),
    accounts: snapshots,
  };
}

async function readRuntimeSnapshot() {
  const connections = configuredConnections();
  const settled = await Promise.allSettled(connections.map(readConnectionSnapshot));
  const environments = settled.map((result, index) => {
    if (result.status === 'fulfilled') return result.value;
    const connection = connections[index];
    return {
      id: connection.id,
      label: connection.label,
      environment: connection.environment,
      baseUrl: connection.baseUrl,
      status: 'AUTH_BLOCKED',
      visible_accounts: 0,
      total_accounts: null,
      account_filter: connection.accountFilter || 'ALL',
      last_sync: null,
      error: result.reason.message,
      accounts: [],
    };
  });

  return {
    timestamp: new Date().toISOString(),
    source: 'tradelocker_token_api',
    accounts: environments.flatMap((environment) => environment.accounts),
    environments,
  };
}

module.exports = { readRuntimeSnapshot, configuredConnections };
