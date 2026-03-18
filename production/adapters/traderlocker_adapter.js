/**
 * TraderLocker Broker Adapter
 *
 * STATUS: SCAFFOLDED — not connected
 * BLOCKED: TraderLocker API returns 403 from WSL2 environment
 * READY FOR: Real connection when executed from non-WSL or with VPN
 *
 * Production endpoint: https://live.traderlocker.com/backend
 * Auth: POST /auth/jwt/token → JWT
 * Accounts: GET /trade/accounts
 * Positions: GET /trade/accounts/{id}/positions
 * Orders: POST /trade/accounts/{id}/orders
 */

const BASE_URL = process.env.TRADERLOCKER_URL || 'https://live.traderlocker.com/backend';

class TraderLockerAdapter {
  constructor(email, password, server, accountId) {
    this.email = email;
    this.password = password;
    this.server = server;
    this.accountId = accountId;
    this.token = null;
    this.tokenExpiry = null;
    this.connected = false;
    this.lastError = null;
  }

  async connect() {
    try {
      const resp = await fetch(`${BASE_URL}/auth/jwt/token`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({email: this.email, password: this.password, server: this.server})
      });
      if (!resp.ok) {
        this.lastError = `Auth failed: ${resp.status}`;
        this.connected = false;
        return false;
      }
      const data = await resp.json();
      this.token = data.accessToken;
      this.tokenExpiry = Date.now() + (data.expiresIn * 1000);
      this.connected = true;
      this.lastError = null;
      return true;
    } catch (err) {
      this.lastError = `Connection error: ${err.message}`;
      this.connected = false;
      return false;
    }
  }

  async getAccountSnapshot() {
    if (!this.connected) return {error: 'Not connected', status: 'DISCONNECTED'};
    try {
      const resp = await fetch(`${BASE_URL}/trade/accounts/${this.accountId}`, {
        headers: {'Authorization': `Bearer ${this.token}`}
      });
      if (!resp.ok) return {error: `API error: ${resp.status}`, status: 'ERROR'};
      const data = await resp.json();
      return {
        status: 'OK',
        accountId: this.accountId,
        balance: data.balance,
        equity: data.equity,
        margin: data.margin,
        freeMargin: data.freeMargin,
        drawdown_pct: ((data.balance - data.equity) / data.balance * 100).toFixed(2),
        timestamp: new Date().toISOString()
      };
    } catch (err) {
      return {error: err.message, status: 'ERROR'};
    }
  }

  async getPositions() {
    if (!this.connected) return [];
    const resp = await fetch(`${BASE_URL}/trade/accounts/${this.accountId}/positions`, {
      headers: {'Authorization': `Bearer ${this.token}`}
    });
    if (!resp.ok) return [];
    return resp.json();
  }

  async closeAllPositions() {
    // KILL SWITCH — close all open positions
    const positions = await this.getPositions();
    const results = [];
    for (const pos of positions) {
      const resp = await fetch(`${BASE_URL}/trade/accounts/${this.accountId}/positions/${pos.id}`, {
        method: 'DELETE',
        headers: {'Authorization': `Bearer ${this.token}`}
      });
      results.push({positionId: pos.id, closed: resp.ok, status: resp.status});
    }
    return results;
  }

  getConnectionState() {
    return {
      connected: this.connected,
      lastError: this.lastError,
      tokenValid: this.token && Date.now() < this.tokenExpiry,
      blocked: true,  // ALWAYS true until environment allows outbound to traderlocker.com
      note: 'Blocked from WSL2 — traderlocker.com returns 403. Run from non-WSL or with VPN.'
    };
  }
}

// Account configs (secrets via env vars in production)
const ACCOUNTS = {
  '703060': new TraderLockerAdapter(
    process.env.GATESFX_EMAIL    || 'ryanlawrence217@hotmail.com',
    process.env.GATESFX_PASSWORD || '',  // MUST be in env — never hardcode
    process.env.GATESFX_SERVER   || 'GATESFX',
    '703060'
  ),
  '703062': new TraderLockerAdapter(
    process.env.GATESFX_EMAIL    || 'ryanlawrence217@hotmail.com',
    process.env.GATESFX_PASSWORD || '',
    process.env.GATESFX_SERVER   || 'GATESFX',
    '703062'
  )
};

module.exports = { TraderLockerAdapter, ACCOUNTS };
