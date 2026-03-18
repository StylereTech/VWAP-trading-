/**
 * Market Data Feed Adapter
 *
 * STATUS: SCAFFOLDED — mock data only
 * PRODUCTION TARGET: yfinance / TraderLocker price feed / direct broker quotes
 *
 * Interface contract:
 *   subscribe(symbol, timeframe, callback) → real-time OHLCV bars
 *   getLatestBar(symbol, timeframe) → {open, high, low, close, volume, timestamp}
 *   getStatus() → {connected, symbols, lastTick, staleness_seconds}
 */

class DataFeedAdapter {
  constructor() {
    this.subscriptions = new Map();
    this.latestBars = new Map();
    this.connected = false;
    this.mode = 'MOCK'; // MOCK | LIVE
  }

  subscribe(symbol, timeframe, callback) {
    const key = `${symbol}_${timeframe}`;
    this.subscriptions.set(key, callback);
    console.log(`[FEED] Subscribed: ${key} (mock mode)`);
  }

  getLatestBar(symbol, timeframe) {
    const key = `${symbol}_${timeframe}`;
    if (this.latestBars.has(key)) return this.latestBars.get(key);
    // Mock bar — clearly labeled
    return {
      symbol, timeframe,
      open: null, high: null, low: null, close: null, volume: null,
      timestamp: null,
      status: 'NO_DATA',
      note: 'Mock adapter — real feed not connected'
    };
  }

  getStatus() {
    return {
      connected: false,
      mode: 'MOCK',
      subscriptions: Array.from(this.subscriptions.keys()),
      lastTick: null,
      staleness_seconds: null,
      blocked: true,
      note: 'Real data feed requires live broker connection (blocked in current environment)'
    };
  }
}

module.exports = { DataFeedAdapter };
