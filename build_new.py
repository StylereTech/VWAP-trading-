#!/usr/bin/env python3
"""Build TradingLand dashboard HTML."""
import json

with open('/tmp/vwap-dash/trades_array.json') as f:
    trades_json = f.read().strip()

with open('/tmp/vwap-dash/live_compact.json') as f:
    live_json = f.read().strip()

with open('/home/rlagents/.openclaw/workspace/trading-lab/configs/strategies.json') as f:
    config_json = f.read().strip()

html = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TradingLand — Trading Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
--bg:#0d1117;--bg2:#161b22;--bg3:#1c2333;--bg4:#21283b;
--border:#30363d;--border2:#444c56;
--text:#e6edf3;--text2:#8b949e;--text3:#6e7681;
--green:#3fb950;--green2:#238636;--green-bg:rgba(63,185,80,0.1);
--red:#f85149;--red2:#da3633;--red-bg:rgba(248,81,73,0.1);
--blue:#58a6ff;--blue2:#1f6feb;--blue-bg:rgba(88,166,255,0.1);
--gold:#d29922;--gold-bg:rgba(210,153,34,0.1);
--purple:#bc8cff;--orange:#f0883e;
}
body{font-family:'Inter',system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);overflow-x:hidden;min-height:100vh}
.app{display:flex;flex-direction:column;min-height:100vh}

/* Top Nav */
.topnav{background:var(--bg2);border-bottom:1px solid var(--border);padding:0 24px;display:flex;align-items:center;height:56px;position:sticky;top:0;z-index:100}
.logo{font-size:20px;font-weight:800;background:linear-gradient(135deg,var(--green),var(--blue));-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-right:32px;letter-spacing:-0.5px;cursor:pointer;white-space:nowrap}
.logo span{font-size:11px;font-weight:400;-webkit-text-fill-color:var(--text3);margin-left:4px}
.tabs{display:flex;gap:2px;height:100%;overflow-x:auto}
.tab{padding:0 16px;height:100%;display:flex;align-items:center;cursor:pointer;color:var(--text2);font-size:13px;font-weight:500;border-bottom:2px solid transparent;transition:all .2s;white-space:nowrap}
.tab:hover{color:var(--text);background:var(--bg3)}
.tab.active{color:var(--text);border-bottom-color:var(--blue)}
.tab .badge{background:var(--blue2);color:#fff;font-size:10px;padding:1px 6px;border-radius:10px;margin-left:6px}
.nav-right{margin-left:auto;display:flex;align-items:center;gap:12px;white-space:nowrap}
.nav-right .status{font-size:11px;color:var(--text3)}
.nav-right .dot{width:8px;height:8px;border-radius:50%;background:var(--green);display:inline-block;margin-right:4px}

/* Hamburger for mobile */
.hamburger{display:none;background:none;border:none;color:var(--text);font-size:24px;cursor:pointer;padding:4px 8px}

/* Main Content */
.content{flex:1;padding:24px;max-width:1600px;margin:0 auto;width:100%}
.page{display:none}
.page.active{display:block;animation:fadeIn .3s ease}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}

/* Cards */
.card{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:20px;margin-bottom:16px}
.card-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;flex-wrap:wrap;gap:8px}
.card-title{font-size:14px;font-weight:600;color:var(--text)}
.card-subtitle{font-size:12px;color:var(--text3)}

/* KPI Grid */
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:20px}
.kpi{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:16px}
.kpi-label{font-size:11px;font-weight:500;color:var(--text3);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px}
.kpi-value{font-size:22px;font-weight:700}
.kpi-value.positive{color:var(--green)}
.kpi-value.negative{color:var(--red)}
.kpi-value.neutral{color:var(--blue)}
.kpi-sub{font-size:11px;color:var(--text3);margin-top:4px}

/* Charts Grid */
.charts-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px}
.charts-grid .card{min-height:300px}
.chart-full{grid-column:1/-1}

/* Alerts */
.alerts-panel{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:20px}
.alert-card{padding:12px 16px;border-radius:8px;font-size:12px;font-weight:500;display:flex;align-items:center;gap:8px;flex:1;min-width:220px}
.alert-card.critical{background:rgba(248,81,73,0.12);border:1px solid rgba(248,81,73,0.3);color:var(--red)}
.alert-card.warning{background:rgba(210,153,34,0.12);border:1px solid rgba(210,153,34,0.3);color:var(--gold)}
.alert-card.info{background:rgba(88,166,255,0.12);border:1px solid rgba(88,166,255,0.3);color:var(--blue)}

/* Watchlist */
.watchlist-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:20px}
.watchlist-card{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:16px;text-align:center;transition:transform .2s}
.watchlist-card:hover{transform:translateY(-2px)}
.watchlist-symbol{font-size:14px;font-weight:700;margin-bottom:4px}
.watchlist-price{font-size:24px;font-weight:700;margin:8px 0}
.watchlist-sub{font-size:11px;color:var(--text3)}
.watchlist-badge{font-size:10px;padding:2px 8px;border-radius:4px;display:inline-block;margin-top:6px}

/* Calendar Heatmap */
.heatmap-wrap{overflow-x:auto;padding:8px 0}
.heatmap{display:flex;gap:3px}
.heatmap-week{display:flex;flex-direction:column;gap:3px}
.heatmap-day{width:14px;height:14px;border-radius:3px;cursor:pointer;position:relative}
.heatmap-day:hover{outline:2px solid var(--blue);outline-offset:1px}
.heatmap-day .tip{display:none;position:absolute;bottom:calc(100% + 6px);left:50%;transform:translateX(-50%);background:var(--bg4);border:1px solid var(--border2);border-radius:6px;padding:6px 10px;font-size:11px;white-space:nowrap;z-index:10;pointer-events:none}
.heatmap-day:hover .tip{display:block}

/* Strategy Cards */
.strat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:16px}
.strat-card{background:var(--bg2);border:1px solid var(--border);border-radius:12px;overflow:hidden;transition:transform .2s}
.strat-card:hover{transform:translateY(-2px)}
.strat-card-head{padding:16px 20px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--border)}
.strat-name{font-size:15px;font-weight:700}
.strat-instrument{font-size:12px;color:var(--text2);background:var(--bg3);padding:3px 8px;border-radius:6px}
.strat-card-body{padding:20px}
.strat-metrics{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:16px}
.strat-metric-label{font-size:10px;color:var(--text3);text-transform:uppercase}
.strat-metric-value{font-size:16px;font-weight:600;margin-top:2px}
.strat-chart{height:80px;margin-bottom:12px}
.strat-signals{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:12px}
.signal-tag{font-size:10px;padding:2px 8px;border-radius:4px;border:1px solid var(--border)}
.signal-tag.on{background:var(--green-bg);border-color:var(--green2);color:var(--green)}
.signal-tag.off{background:var(--bg3);color:var(--text3)}
.strat-banner{padding:10px 20px;font-size:12px;font-weight:500;display:flex;align-items:center;gap:6px}
.strat-banner.warn{background:rgba(210,153,34,0.1);color:var(--gold);border-top:1px solid rgba(210,153,34,0.2)}
.strat-banner.good{background:var(--green-bg);color:var(--green);border-top:1px solid rgba(63,185,80,0.2)}
.strat-coming{opacity:0.5;position:relative}
.strat-coming::after{content:'COMING SOON';position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);background:var(--bg);border:2px solid var(--gold);color:var(--gold);padding:8px 20px;border-radius:8px;font-weight:700;font-size:14px;letter-spacing:1px;z-index:5}
.strat-recent{border-top:1px solid var(--border);margin-top:12px;padding-top:12px}
.strat-recent-title{font-size:11px;color:var(--text3);margin-bottom:6px}
.strat-recent-row{display:flex;justify-content:space-between;font-size:11px;padding:3px 0;border-bottom:1px solid rgba(48,54,61,0.5)}

/* Tables */
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;padding:10px 12px;color:var(--text3);font-weight:600;text-transform:uppercase;font-size:10px;letter-spacing:0.5px;border-bottom:2px solid var(--border);cursor:pointer;user-select:none;white-space:nowrap}
th:hover{color:var(--text)}
td{padding:8px 12px;border-bottom:1px solid var(--border);white-space:nowrap}
tr:hover td{background:var(--bg3)}
.pnl-pos{color:var(--green);font-weight:600}
.pnl-neg{color:var(--red);font-weight:600}

/* Filters */
.filters{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px;align-items:center}
.filter-select,.filter-input{background:var(--bg3);border:1px solid var(--border);color:var(--text);padding:6px 12px;border-radius:6px;font-size:12px;font-family:inherit}
.filter-select:focus,.filter-input:focus{outline:none;border-color:var(--blue)}
.filter-label{font-size:11px;color:var(--text3);margin-right:4px}

/* Pagination */
.pagination{display:flex;gap:6px;align-items:center;justify-content:center;padding:16px 0;flex-wrap:wrap}
.pagination button{background:var(--bg3);border:1px solid var(--border);color:var(--text);padding:6px 14px;border-radius:6px;cursor:pointer;font-size:12px;font-family:inherit}
.pagination button:hover:not(:disabled){background:var(--bg4)}
.pagination button:disabled{opacity:0.4;cursor:default}
.pagination span{color:var(--text3);font-size:12px}

/* Live Account */
.live-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:20px}

/* Analytics */
.analytics-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.analytics-grid .card{min-height:250px}

/* Export buttons */
.btn{background:var(--bg3);border:1px solid var(--border);color:var(--text);padding:8px 16px;border-radius:6px;cursor:pointer;font-size:12px;font-family:inherit;font-weight:500;transition:all .2s;display:inline-flex;align-items:center;gap:6px}
.btn:hover{background:var(--bg4);border-color:var(--border2)}
.btn-primary{background:var(--blue2);border-color:var(--blue);color:#fff}
.btn-primary:hover{background:var(--blue)}

/* Footer */
.footer{background:var(--bg2);border-top:1px solid var(--border);padding:16px 24px;text-align:center;font-size:11px;color:var(--text3)}

/* VWAP Section */
.vwap-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px}
.vwap-stat{text-align:center;padding:20px}
.vwap-stat-value{font-size:28px;font-weight:700;margin:8px 0}
.vwap-stat-label{font-size:12px;color:var(--text3)}

/* Scrollbar */
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:var(--border2)}

/* Responsive */
@media(max-width:900px){
.charts-grid,.analytics-grid,.vwap-grid{grid-template-columns:1fr}
.strat-grid{grid-template-columns:1fr}
.kpi-grid{grid-template-columns:repeat(auto-fit,minmax(120px,1fr))}
.topnav{padding:0 12px;flex-wrap:wrap;height:auto;min-height:56px}
.tabs{display:none;width:100%;padding:8px 0}
.tabs.open{display:flex;flex-wrap:wrap}
.tab{padding:8px 14px;height:auto}
.hamburger{display:block}
.content{padding:12px}
.nav-right{display:none}
.watchlist-grid{grid-template-columns:1fr 1fr}
.alerts-panel{flex-direction:column}
.logo{margin-right:auto}
}
@media(max-width:500px){
.watchlist-grid{grid-template-columns:1fr}
.kpi-grid{grid-template-columns:1fr 1fr}
.strat-metrics{grid-template-columns:1fr 1fr}
}
</style>
</head>
<body>
<div class="app">
<nav class="topnav">
<div class="logo" onclick="document.querySelector('[data-page=dashboard]').click()">🏔️ TradingLand <span>v3.0</span></div>
<button class="hamburger" onclick="document.querySelector('.tabs').classList.toggle('open')">☰</button>
<div class="tabs">
<div class="tab active" data-page="dashboard">📊 Overview</div>
<div class="tab" data-page="demo">🧪 Strategies <span class="badge">4</span></div>
<div class="tab" data-page="live">💰 Live</div>
<div class="tab" data-page="journal">📒 Journal</div>
<div class="tab" data-page="analytics">📈 Analytics</div>
<div class="tab" data-page="vwap">🎯 VWAP</div>
</div>
<div class="nav-right">
<span class="status"><span class="dot"></span>Bots Running</span>
<span class="status" id="trade-count-nav"></span>
</div>
</nav>

<main class="content">
<!-- DASHBOARD PAGE -->
<div class="page active" id="page-dashboard">
<div id="alerts-panel" class="alerts-panel"></div>
<h3 style="font-size:14px;font-weight:600;color:var(--text3);margin-bottom:12px">📡 WATCHLIST</h3>
<div id="watchlist" class="watchlist-grid"></div>
<div id="kpi-row" class="kpi-grid"></div>
<div class="card">
<div class="card-header">
<span class="card-title">📅 Daily P&L Calendar</span>
<span class="card-subtitle" id="cal-range"></span>
</div>
<div class="heatmap-wrap"><div id="heatmap" class="heatmap"></div></div>
</div>
<div class="charts-grid">
<div class="card chart-full">
<div class="card-header"><span class="card-title">Cumulative P&L</span>
<div><select class="filter-select" id="cum-strat-filter"><option value="">All Strategies</option></select></div>
</div>
<canvas id="chart-cumPnl"></canvas>
</div>
<div class="card"><div class="card-header"><span class="card-title">P&L by Day of Week</span></div><canvas id="chart-dow"></canvas></div>
<div class="card"><div class="card-header"><span class="card-title">P&L by Hour</span></div><canvas id="chart-hour"></canvas></div>
<div class="card"><div class="card-header"><span class="card-title">P&L by Strategy</span></div><canvas id="chart-strat"></canvas></div>
<div class="card"><div class="card-header"><span class="card-title">Running Balance</span></div><canvas id="chart-balance"></canvas></div>
</div>
<div style="text-align:right;margin-bottom:16px">
<button class="btn" onclick="window.print()">🖨️ Print / PDF</button>
</div>
</div>

<!-- DEMO ACCOUNTS PAGE -->
<div class="page" id="page-demo">
<h2 style="font-size:18px;margin-bottom:16px;font-weight:700">🧪 Demo Strategy Accounts</h2>
<div id="strat-cards" class="strat-grid"></div>
</div>

<!-- LIVE ACCOUNT PAGE -->
<div class="page" id="page-live">
<h2 style="font-size:18px;margin-bottom:16px;font-weight:700">💰 GatesFX Live Account</h2>
<div id="live-content"></div>
</div>

<!-- TRADE JOURNAL PAGE -->
<div class="page" id="page-journal">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px">
<h2 style="font-size:18px;font-weight:700">📒 Trade Journal</h2>
<button class="btn btn-primary" id="btn-export-csv">📥 Export CSV</button>
</div>
<div class="filters" id="journal-filters"></div>
<div class="card" style="overflow-x:auto"><table id="journal-table"><thead></thead><tbody></tbody></table></div>
<div class="pagination" id="journal-pagination"></div>
</div>

<!-- ANALYTICS PAGE -->
<div class="page" id="page-analytics">
<h2 style="font-size:18px;margin-bottom:16px;font-weight:700">📈 Deep Analytics</h2>
<div class="analytics-grid" id="analytics-content"></div>
</div>

<!-- VWAP ANALYSIS PAGE -->
<div class="page" id="page-vwap">
<h2 style="font-size:18px;margin-bottom:16px;font-weight:700">🎯 VWAP Analysis</h2>
<div id="vwap-content"></div>
</div>
</main>

<footer class="footer">
TradingLand &copy; 2026 | Last updated: ''' + __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S PST') + r'''
</footer>
</div>

<script>
// ===== EMBEDDED DATA =====
const TRADES = ''' + trades_json + r''';
const CONFIG = ''' + config_json + r''';
const LIVE = ''' + live_json + r''';

// ===== UTILITIES =====
const fmt = (n,d=2) => n!=null ? Number(n).toFixed(d) : '—';
const fmtPct = n => n!=null ? (n*100).toFixed(1)+'%' : '—';
const fmtMoney = n => n!=null ? (n>=0?'+':'')+Number(n).toFixed(2) : '—';
const pnlClass = n => n>=0?'positive':'negative';
const pnlTd = n => '<span class="pnl-'+(n>=0?'pos':'neg')+'">'+fmtMoney(n)+'</span>';

// ===== TAB NAVIGATION =====
document.querySelectorAll('.tab').forEach(t=>{
t.addEventListener('click',()=>{
document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
document.querySelectorAll('.page').forEach(x=>x.classList.remove('active'));
t.classList.add('active');
const pg = document.getElementById('page-'+t.dataset.page);
if(pg) pg.classList.add('active');
// Close mobile menu
document.querySelector('.tabs').classList.remove('open');
});
});

// ===== COMPUTE METRICS =====
function computeMetrics(trades){
if(!trades.length) return {totalPnl:0,winRate:0,profitFactor:0,avgWin:0,avgLoss:0,trades:0,bestDay:0,worstDay:0,maxDrawdown:0,sharpe:0,expectancy:0,maxConsecWin:0,maxConsecLoss:0,daily:{}};
const wins = trades.filter(t=>t.pnl>0);
const losses = trades.filter(t=>t.pnl<0);
const totalPnl = trades.reduce((s,t)=>s+t.pnl,0);
const winRate = wins.length/trades.length;
const avgWin = wins.length? wins.reduce((s,t)=>s+t.pnl,0)/wins.length : 0;
const avgLoss = losses.length? Math.abs(losses.reduce((s,t)=>s+t.pnl,0)/losses.length) : 0;
const profitFactor = losses.length? wins.reduce((s,t)=>s+t.pnl,0)/Math.abs(losses.reduce((s,t)=>s+t.pnl,0)) : wins.length?Infinity:0;
const daily = {};
trades.forEach(t=>{const d=t.time.slice(0,10);daily[d]=(daily[d]||0)+t.pnl});
const dailyVals = Object.values(daily);
const avgDaily = dailyVals.reduce((a,b)=>a+b,0)/dailyVals.length;
const stdDaily = Math.sqrt(dailyVals.reduce((s,v)=>s+Math.pow(v-avgDaily,2),0)/dailyVals.length);
const sharpe = stdDaily?avgDaily/stdDaily*Math.sqrt(252):0;
let peak=0,maxDD=0,running=0;
dailyVals.forEach(v=>{running+=v;if(running>peak)peak=running;const dd=peak-running;if(dd>maxDD)maxDD=dd});
const bestDay = Math.max(...dailyVals);
const worstDay = Math.min(...dailyVals);
const expectancy = totalPnl/trades.length;
let maxConsecWin=0,maxConsecLoss=0,cw=0,cl=0;
trades.forEach(t=>{
if(t.pnl>0){cw++;cl=0;if(cw>maxConsecWin)maxConsecWin=cw}
else{cl++;cw=0;if(cl>maxConsecLoss)maxConsecLoss=cl}
});
return {totalPnl,winRate,profitFactor,avgWin,avgLoss,trades:trades.length,bestDay,worstDay,maxDrawdown:maxDD,sharpe,expectancy,maxConsecWin,maxConsecLoss,daily};
}

const allMetrics = computeMetrics(TRADES);
document.getElementById('trade-count-nav').textContent = TRADES.length.toLocaleString()+' trades';

// ===== ALERTS PANEL =====
(function(){
const alerts = [];
// Check each strategy
const stratMap = {};
TRADES.forEach(t=>{if(!stratMap[t.strategy])stratMap[t.strategy]=[];stratMap[t.strategy].push(t)});
Object.entries(stratMap).forEach(([k,v])=>{
const m = computeMetrics(v);
if(m.maxDrawdown > 0 && m.maxDrawdown / 500 > 0.10) alerts.push({type:'critical',text:'⛔ '+k+' drawdown '+fmtMoney(-m.maxDrawdown)+' (>'+(m.maxDrawdown/500*100).toFixed(0)+'%)'});
if(m.winRate < 0.35 && m.trades > 20) alerts.push({type:'warning',text:'⚠️ '+k+' win rate only '+fmtPct(m.winRate)});
if(m.maxConsecLoss > 5) alerts.push({type:'warning',text:'⚠️ '+k+' had '+m.maxConsecLoss+' consecutive losses'});
});
// Overall
if(allMetrics.maxConsecLoss > 5) alerts.push({type:'critical',text:'🔴 Max consecutive losses: '+allMetrics.maxConsecLoss});
if(allMetrics.winRate < 0.35) alerts.push({type:'warning',text:'⚠️ Overall win rate below 35%: '+fmtPct(allMetrics.winRate)});
// Pending journal entries (trades without notes - all trades are "pending" since we don't have notes)
const pending = TRADES.length;
if(pending > 0) alerts.push({type:'info',text:'📝 '+pending+' trades awaiting journal notes'});

const el = document.getElementById('alerts-panel');
if(alerts.length){
el.innerHTML = alerts.slice(0,4).map(a=>'<div class="alert-card '+a.type+'">'+a.text+'</div>').join('');
}
})();

// ===== WATCHLIST =====
(function(){
const el = document.getElementById('watchlist');
// Get last trade prices for each instrument
const lastPrices = {};
const instruments = {GBPJPY:'GBPJPY.B',BTCUSD:'BTCUSD',EURUSD:'EURUSD.B',XAUUSD:'XAUUSD.B'};
Object.entries(instruments).forEach(([sym,inst])=>{
const trades = TRADES.filter(t=>t.instrument===inst).sort((a,b)=>a.time.localeCompare(b.time));
if(trades.length){
const last = trades[trades.length-1];
const first = trades[0];
lastPrices[sym] = {price:last.exit, change:last.exit-first.entry, lastTrade:last.time};
}
});

const icons = {GBPJPY:'🇬🇧🇯🇵',BTCUSD:'₿',EURUSD:'🇪🇺🇺🇸',XAUUSD:'🥇'};
const precisions = {GBPJPY:3,BTCUSD:0,EURUSD:5,XAUUSD:2};

el.innerHTML = Object.entries(lastPrices).map(([sym,d])=>{
const up = d.change >= 0;
return '<div class="watchlist-card" style="border-left:3px solid '+(up?'var(--green)':'var(--red)')+'">'+
'<div class="watchlist-symbol">'+icons[sym]+' '+sym+'</div>'+
'<div class="watchlist-price" style="color:'+(up?'var(--green)':'var(--red)')+'">'+fmt(d.price,precisions[sym])+'</div>'+
'<div class="watchlist-sub">Last traded: '+d.lastTrade.slice(0,16).replace('T',' ')+'</div>'+
'<div class="watchlist-badge" style="background:'+(up?'var(--green-bg)':'var(--red-bg)')+';color:'+(up?'var(--green)':'var(--red)')+'">'+
(up?'▲':'▼')+' '+fmt(Math.abs(d.change),precisions[sym])+'</div>'+
'</div>';
}).join('');

// Also try fetching live BTC price
fetch('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT')
.then(r=>r.json()).then(d=>{
const card = el.querySelector('.watchlist-card:nth-child(2)');
if(card && d.price){
card.querySelector('.watchlist-price').textContent = Number(d.price).toFixed(0);
card.querySelector('.watchlist-sub').innerHTML = 'Live: Binance <span style="color:var(--green)">●</span>';
}
}).catch(()=>{});
})();

// ===== DASHBOARD: KPIs =====
const kpis = [
{label:'Net P&L',value:fmtMoney(allMetrics.totalPnl),cls:pnlClass(allMetrics.totalPnl)},
{label:'Win Rate',value:fmtPct(allMetrics.winRate),cls:allMetrics.winRate>=0.5?'positive':'negative'},
{label:'Profit Factor',value:fmt(allMetrics.profitFactor),cls:allMetrics.profitFactor>=1?'positive':'negative'},
{label:'Avg Win/Loss',value:allMetrics.avgLoss?fmt(allMetrics.avgWin/allMetrics.avgLoss):'—',cls:'neutral'},
{label:'Total Trades',value:allMetrics.trades.toLocaleString(),cls:'neutral'},
{label:'Best Day',value:fmtMoney(allMetrics.bestDay),cls:'positive'},
{label:'Worst Day',value:fmtMoney(allMetrics.worstDay),cls:'negative'},
{label:'Max Drawdown',value:fmtMoney(-allMetrics.maxDrawdown),cls:'negative'},
{label:'Sharpe Ratio',value:fmt(allMetrics.sharpe),cls:allMetrics.sharpe>=1?'positive':'neutral'},
{label:'Expectancy',value:fmtMoney(allMetrics.expectancy),cls:pnlClass(allMetrics.expectancy)},
];
document.getElementById('kpi-row').innerHTML = kpis.map(k=>'<div class="kpi"><div class="kpi-label">'+k.label+'</div><div class="kpi-value '+k.cls+'">'+k.value+'</div></div>').join('');

// ===== DASHBOARD: Calendar Heatmap =====
(function(){
const daily = allMetrics.daily;
const dates = Object.keys(daily).sort();
if(!dates.length) return;
const start = new Date(dates[0]);
const end = new Date(dates[dates.length-1]);
document.getElementById('cal-range').textContent = dates[0]+' → '+dates[dates.length-1];
const maxAbs = Math.max(...Object.values(daily).map(Math.abs),1);
let html = '';
let cur = new Date(start);
cur.setDate(cur.getDate()-cur.getDay());
let weekHtml = '';
let weekDay = 0;
while(cur<=end||weekDay>0){
const ds = cur.toISOString().slice(0,10);
const val = daily[ds]||0;
const intensity = Math.min(Math.abs(val)/maxAbs,1);
let color;
if(val>0) color = 'rgba(63,185,80,'+(0.15+intensity*0.85)+')';
else if(val<0) color = 'rgba(248,81,73,'+(0.15+intensity*0.85)+')';
else color = 'var(--bg3)';
weekHtml += '<div class="heatmap-day" style="background:'+color+'" title="'+ds+': '+fmtMoney(val)+'"><div class="tip">'+ds+'<br>'+fmtMoney(val)+'</div></div>';
weekDay++;
if(weekDay===7){html += '<div class="heatmap-week">'+weekHtml+'</div>';weekHtml='';weekDay=0;}
cur.setDate(cur.getDate()+1);
}
if(weekHtml) html += '<div class="heatmap-week">'+weekHtml+'</div>';
document.getElementById('heatmap').innerHTML = html;
})();

// ===== CHART DEFAULTS =====
Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#30363d';
Chart.defaults.font.family = 'Inter, system-ui';
Chart.defaults.font.size = 11;

// Strategy filter for cumPnl
const strategies = [...new Set(TRADES.map(t=>t.strategy))].sort();
const cumFilter = document.getElementById('cum-strat-filter');
strategies.forEach(s=>{const o=document.createElement('option');o.value=s;o.textContent=s;cumFilter.appendChild(o)});

let cumChart = null;
function renderCumPnl(stratFilter){
const src = stratFilter ? TRADES.filter(t=>t.strategy===stratFilter) : TRADES;
const sorted = [...src].sort((a,b)=>a.time.localeCompare(b.time));
let cum=0;
const data = sorted.map(t=>{cum+=t.pnl;return cum});
const labels = sorted.map((_,i)=>i);
if(cumChart) cumChart.destroy();
cumChart = new Chart(document.getElementById('chart-cumPnl'),{
type:'line',
data:{labels,datasets:[{data,borderColor:'#58a6ff',backgroundColor:'rgba(88,166,255,0.1)',fill:true,pointRadius:0,borderWidth:1.5,tension:0.1}]},
options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{display:false},y:{grid:{color:'#1c2333'}}}}
});
}
renderCumPnl('');
cumFilter.addEventListener('change',()=>renderCumPnl(cumFilter.value));

// ===== DASHBOARD: P&L by Day of Week =====
(function(){
const dow = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
const pnl = Array(7).fill(0);
TRADES.forEach(t=>{const d=new Date(t.time).getDay();pnl[d]+=t.pnl});
new Chart(document.getElementById('chart-dow'),{
type:'bar',
data:{labels:dow,datasets:[{data:pnl,backgroundColor:pnl.map(v=>v>=0?'rgba(63,185,80,0.7)':'rgba(248,81,73,0.7)'),borderRadius:4}]},
options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{grid:{color:'#1c2333'}}}},
});
})();

// ===== DASHBOARD: P&L by Hour =====
(function(){
const pnl = Array(24).fill(0);
TRADES.forEach(t=>{const h=parseInt(t.time.slice(11,13));pnl[h]+=t.pnl});
new Chart(document.getElementById('chart-hour'),{
type:'bar',
data:{labels:Array.from({length:24},(_,i)=>i+':00'),datasets:[{data:pnl,backgroundColor:pnl.map(v=>v>=0?'rgba(63,185,80,0.7)':'rgba(248,81,73,0.7)'),borderRadius:3}]},
options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{grid:{color:'#1c2333'}}}},
});
})();

// ===== DASHBOARD: P&L by Strategy (Interactive Donut) =====
(function(){
const strats = {};
TRADES.forEach(t=>{strats[t.strategy]=(strats[t.strategy]||0)+t.pnl});
const labels = Object.keys(strats);
const data = Object.values(strats);
const colors = ['#58a6ff','#3fb950','#f0883e','#bc8cff','#d29922'];
const chart = new Chart(document.getElementById('chart-strat'),{
type:'doughnut',
data:{labels,datasets:[{data:data.map(Math.abs),backgroundColor:colors.slice(0,labels.length),borderWidth:0}]},
options:{responsive:true,maintainAspectRatio:false,
onClick:(e,elements)=>{
if(elements.length){
const idx = elements[0].index;
const strat = labels[idx];
cumFilter.value = strat;
cumFilter.dispatchEvent(new Event('change'));
}
},
plugins:{legend:{position:'right',labels:{boxWidth:10,padding:8,font:{size:11},generateLabels:(chart)=>{
return chart.data.labels.map((l,i)=>({text:l+' ('+fmtMoney(data[i])+')',fillStyle:colors[i],strokeStyle:colors[i]}))}}},
tooltip:{callbacks:{label:(ctx)=>ctx.label+': '+fmtMoney(data[ctx.dataIndex])}}}},
});
})();

// ===== DASHBOARD: Running Balance =====
(function(){
const sorted = [...TRADES].sort((a,b)=>a.time.localeCompare(b.time));
const data = sorted.map(t=>t.balance);
new Chart(document.getElementById('chart-balance'),{
type:'line',
data:{labels:sorted.map((_,i)=>i),datasets:[{data,borderColor:'#d29922',backgroundColor:'rgba(210,153,34,0.1)',fill:true,pointRadius:0,borderWidth:1.5,tension:0.1}]},
options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{display:false},y:{grid:{color:'#1c2333'}}}}
});
})();

// ===== DEMO ACCOUNTS =====
(function(){
const stratDefs = [
{key:'hft_momentum',name:'HFT Momentum',icon:'⚡',instrument:'GBPJPY'},
{key:'vwap_reversion',name:'VWAP Reversion',icon:'📊',instrument:'BTCUSD'},
{key:'range_scalper',name:'Range Scalper',icon:'📐',instrument:'EURUSD'},
{key:'gold_momentum',name:'Gold Momentum',icon:'🥇',instrument:'XAUUSD'},
{key:'trend_follower',name:'Trend Follower',icon:'📈',instrument:'TBD',coming:true},
];
const container = document.getElementById('strat-cards');
stratDefs.forEach((sd,idx)=>{
const trades = TRADES.filter(t=>t.strategy===sd.key).sort((a,b)=>a.time.localeCompare(b.time));
const m = computeMetrics(trades);
const cfg = CONFIG.strategies[sd.key];
const recent = trades.slice(-5).reverse();
let bannerHtml = '';
if(m.trades>0){
if(m.winRate<0.45) bannerHtml = '<div class="strat-banner warn">⚠️ Win rate '+fmtPct(m.winRate)+' — consider tuning signals</div>';
else if(m.profitFactor<1) bannerHtml = '<div class="strat-banner warn">⚠️ Profit factor '+fmt(m.profitFactor)+' — strategy losing money</div>';
else if(m.profitFactor>1.5) bannerHtml = '<div class="strat-banner good">✅ Strong PF '+fmt(m.profitFactor)+' — strategy performing well</div>';
}
const signalHtml = cfg? Object.entries(cfg.signals||{}).map(([k,v])=>'<span class="signal-tag '+(v.enabled?'on':'off')+'">'+k+'</span>').join('') : '';
const configHtml = cfg? '<div style="font-size:11px;color:var(--text3);margin-top:8px">TP: '+cfg.tp_pips+' | SL: '+cfg.sl_pips+' | Max Pos: '+cfg.max_positions+'</div>' : '';
const recentHtml = recent.length? '<div class="strat-recent"><div class="strat-recent-title">Recent Trades</div>'+recent.map(t=>'<div class="strat-recent-row"><span>'+t.time.slice(5,16)+'</span><span>'+t.side+'</span><span>'+t.signal+'</span>'+pnlTd(t.pnl)+'<span style="color:var(--text3)">'+t.reason+'</span></div>').join('')+'</div>' : '';

container.innerHTML += '<div class="strat-card '+(sd.coming?'strat-coming':'')+'"><div class="strat-card-head"><span class="strat-name">'+sd.icon+' '+sd.name+'</span><span class="strat-instrument">'+sd.instrument+'</span></div><div class="strat-card-body"><div class="strat-chart"><canvas id="strat-chart-'+idx+'"></canvas></div><div class="strat-metrics"><div><div class="strat-metric-label">Win Rate</div><div class="strat-metric-value" style="color:'+(m.winRate>=0.5?'var(--green)':'var(--red)')+'">'+fmtPct(m.winRate||0)+'</div></div><div><div class="strat-metric-label">Net P&L</div><div class="strat-metric-value" style="color:'+((m.totalPnl||0)>=0?'var(--green)':'var(--red)')+'">'+fmtMoney(m.totalPnl||0)+'</div></div><div><div class="strat-metric-label">Trades</div><div class="strat-metric-value">'+(m.trades||0)+'</div></div><div><div class="strat-metric-label">Profit Factor</div><div class="strat-metric-value">'+fmt(m.profitFactor||0)+'</div></div><div><div class="strat-metric-label">Max DD</div><div class="strat-metric-value" style="color:var(--red)">'+fmtMoney(-(m.maxDrawdown||0))+'</div></div><div><div class="strat-metric-label">Avg W/L</div><div class="strat-metric-value">'+(m.avgLoss?fmt(m.avgWin/m.avgLoss):'—')+'</div></div></div><div class="strat-signals">'+signalHtml+'</div>'+configHtml+recentHtml+'</div>'+bannerHtml+'</div>';

if(trades.length && !sd.coming){
setTimeout(()=>{
let cum=0;
const data = trades.map(t=>{cum+=t.pnl;return cum});
new Chart(document.getElementById('strat-chart-'+idx),{
type:'line',
data:{labels:trades.map((_,i)=>i),datasets:[{data,borderColor:data[data.length-1]>=0?'#3fb950':'#f85149',backgroundColor:data[data.length-1]>=0?'rgba(63,185,80,0.1)':'rgba(248,81,73,0.1)',fill:true,pointRadius:0,borderWidth:1.5,tension:0.1}]},
options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{display:false},y:{display:false}}},
});
},100*(idx+1));
}
});
})();

// ===== LIVE ACCOUNT =====
(function(){
const el = document.getElementById('live-content');
const a = LIVE.account||{};
const s = LIVE.state||{};
const connected = LIVE.connected;
const syncTime = LIVE.fetched_at ? new Date(LIVE.fetched_at+'Z').toLocaleString() : 'Unknown';
const balance = s.balance || a.accountBalance || 0;
const equity = s.projectedBalance || balance;
const freeMargin = s.availableFunds || balance;
const marginUsed = s.initialMarginReq || 0;
const todayPnl = s.todayNet || 0;
const posCount = s.positionsCount || 0;
const startBalance = 500;
const totalPnl = balance - startBalance;

el.innerHTML = '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;flex-wrap:wrap;gap:8px"><div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap"><div style="display:flex;align-items:center;gap:6px"><span style="width:10px;height:10px;border-radius:50%;background:'+(connected?'var(--green)':'var(--red)')+';display:inline-block"></span><span style="font-size:13px;font-weight:600;color:var(--text)">'+(connected?'Connected':'Disconnected')+'</span></div><span style="font-size:12px;color:var(--text3);background:var(--bg3);padding:3px 10px;border-radius:6px">GatesFX • Account #'+(a.id||684012)+'</span><span style="font-size:12px;color:var(--text3);background:var(--bg3);padding:3px 10px;border-radius:6px">'+(a.status||'ACTIVE')+'</span></div><div style="font-size:11px;color:var(--text3)">🔄 Last synced: '+syncTime+'</div></div>'+
'<div class="kpi-grid" style="margin-bottom:20px"><div class="kpi"><div class="kpi-label">Balance</div><div class="kpi-value neutral">$'+fmt(balance)+'</div></div><div class="kpi"><div class="kpi-label">Equity</div><div class="kpi-value neutral">$'+fmt(equity)+'</div></div><div class="kpi"><div class="kpi-label">Free Margin</div><div class="kpi-value neutral">$'+fmt(freeMargin)+'</div></div><div class="kpi"><div class="kpi-label">Margin Used</div><div class="kpi-value">$'+fmt(marginUsed)+'</div></div><div class="kpi"><div class="kpi-label">Total P&L</div><div class="kpi-value '+(totalPnl>=0?'positive':'negative')+'">'+fmtMoney(totalPnl)+'</div><div class="kpi-sub">from $'+fmt(startBalance)+' start</div></div><div class="kpi"><div class="kpi-label">Today P&L</div><div class="kpi-value '+(todayPnl>=0?'positive':'negative')+'">'+fmtMoney(todayPnl)+'</div></div></div>'+
'<div class="card"><div class="card-header"><span class="card-title">📊 Financial Summary</span></div><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;padding:4px 0"><div><div style="font-size:11px;color:var(--text3);margin-bottom:4px">Starting Balance</div><div style="font-size:16px;font-weight:600">$'+fmt(startBalance)+'</div></div><div><div style="font-size:11px;color:var(--text3);margin-bottom:4px">Current Balance</div><div style="font-size:16px;font-weight:600;color:var(--blue)">$'+fmt(balance)+'</div></div><div><div style="font-size:11px;color:var(--text3);margin-bottom:4px">Total P&L</div><div style="font-size:16px;font-weight:600;color:'+(totalPnl>=0?'var(--green)':'var(--red)')+'">'+fmtMoney(totalPnl)+'</div></div><div><div style="font-size:11px;color:var(--text3);margin-bottom:4px">Return</div><div style="font-size:16px;font-weight:600;color:'+(totalPnl>=0?'var(--green)':'var(--red)'">'+(totalPnl/startBalance*100).toFixed(1)+'%</div></div></div></div>'+
'<div class="card"><div class="card-header"><span class="card-title">📈 Open Positions</span><span class="card-subtitle">'+posCount+' position'+(posCount!==1?'s':'')+'</span></div>'+
(LIVE.positions&&LIVE.positions.length?'<table><tr><th>Instrument</th><th>Side</th><th>Size</th><th>Entry</th><th>P&L</th></tr>'+LIVE.positions.map(function(p){return '<tr><td>'+JSON.stringify(p)+'</td></tr>'}).join('')+'</table>':
'<div style="padding:32px;text-align:center;color:var(--text3)"><div style="font-size:32px;margin-bottom:10px;opacity:0.5">📭</div><div style="font-size:13px">No open positions</div></div>')+'</div>'+
'<div class="card"><div class="card-header"><span class="card-title">📋 Recent Executions</span></div>'+
(LIVE.executions&&LIVE.executions.length?'<table><tr><th>Time</th><th>Instrument</th><th>Side</th><th>Price</th><th>Size</th></tr>'+LIVE.executions.map(function(e){return '<tr><td>'+JSON.stringify(e)+'</td></tr>'}).join('')+'</table>':
'<div style="padding:32px;text-align:center;color:var(--text3)"><div style="font-size:32px;margin-bottom:10px;opacity:0.5">📋</div><div style="font-size:13px">No trade history yet</div></div>')+'</div>'+
'<div class="card"><div class="card-header"><span class="card-title">⚙️ Account Details</span></div><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;font-size:12px"><div><span style="color:var(--text3)">Withdrawal Available:</span> <strong>$'+fmt(s.withdrawalAvailable||0)+'</strong></div><div><span style="color:var(--text3)">Cash Balance:</span> <strong>$'+fmt(s.cashBalance||0)+'</strong></div><div><span style="color:var(--text3)">Blocked Balance:</span> <strong>$'+fmt(s.blockedBalance||0)+'</strong></div><div><span style="color:var(--text3)">Warning Margin:</span> <strong>'+fmt(s.marginWarningLevel||0)+'%</strong></div><div><span style="color:var(--text3)">Stop Out Level:</span> <strong>'+fmt(s.stopOutLevel||0)+'%</strong></div><div><span style="color:var(--text3)">Today Trades:</span> <strong>'+(s.todayTradesCount||0)+'</strong></div></div></div>';
})();

// ===== TRADE JOURNAL =====
(function(){
const PAGE_SIZE = 50;
let page = 0;
let filtered = [...TRADES].sort((a,b)=>b.time.localeCompare(a.time));
const strategies = [...new Set(TRADES.map(t=>t.strategy))].sort();
const instruments = [...new Set(TRADES.map(t=>t.instrument))].sort();
const signals = [...new Set(TRADES.map(t=>t.signal))].sort();

const filtersEl = document.getElementById('journal-filters');
filtersEl.innerHTML = '<span class="filter-label">Strategy:</span><select class="filter-select" id="f-strat"><option value="">All</option>'+strategies.map(function(s){return '<option value="'+s+'">'+s+'</option>'}).join('')+'</select>'+
'<span class="filter-label">Instrument:</span><select class="filter-select" id="f-inst"><option value="">All</option>'+instruments.map(function(s){return '<option value="'+s+'">'+s+'</option>'}).join('')+'</select>'+
'<span class="filter-label">Signal:</span><select class="filter-select" id="f-signal"><option value="">All</option>'+signals.map(function(s){return '<option value="'+s+'">'+s+'</option>'}).join('')+'</select>'+
'<span class="filter-label">Win/Loss:</span><select class="filter-select" id="f-wl"><option value="">All</option><option value="win">Win</option><option value="loss">Loss</option></select>'+
'<span class="filter-label">Search:</span><input class="filter-input" id="f-search" placeholder="reason, signal..." style="width:140px">';

function applyFilters(){
const strat = document.getElementById('f-strat').value;
const inst = document.getElementById('f-inst').value;
const signal = document.getElementById('f-signal').value;
const wl = document.getElementById('f-wl').value;
const search = document.getElementById('f-search').value.toLowerCase();
filtered = TRADES.filter(function(t){
if(strat && t.strategy!==strat) return false;
if(inst && t.instrument!==inst) return false;
if(signal && t.signal!==signal) return false;
if(wl==='win' && t.pnl<=0) return false;
if(wl==='loss' && t.pnl>=0) return false;
if(search && !JSON.stringify(t).toLowerCase().includes(search)) return false;
return true;
}).sort((a,b)=>b.time.localeCompare(a.time));
page = 0;
renderJournal();
}

filtersEl.querySelectorAll('select,input').forEach(function(el){el.addEventListener('change',applyFilters)});
document.getElementById('f-search').addEventListener('input',applyFilters);

// Export CSV
document.getElementById('btn-export-csv').addEventListener('click',function(){
const header = 'Date/Time,Strategy,Instrument,Side,Signal,Entry,Exit,Pips,P&L,Reason,Config,Balance\n';
const rows = filtered.map(function(t){
return [t.time,t.strategy,t.instrument,t.side,t.signal,t.entry,t.exit,t.pips,t.pnl,t.reason,t.config_version,t.balance].join(',');
}).join('\n');
const blob = new Blob([header+rows],{type:'text/csv'});
const a = document.createElement('a');
a.href = URL.createObjectURL(blob);
a.download = 'tradingland_journal_'+new Date().toISOString().slice(0,10)+'.csv';
a.click();
});

let sortCol = null, sortDir = 1;
const cols = [
{key:'time',label:'Date/Time'},{key:'strategy',label:'Strategy'},{key:'instrument',label:'Instrument'},
{key:'side',label:'Side'},{key:'signal',label:'Signal'},{key:'entry',label:'Entry'},{key:'exit',label:'Exit'},
{key:'pips',label:'Pips'},{key:'pnl',label:'P&L'},{key:'reason',label:'Reason'},{key:'config_version',label:'Config'},{key:'balance',label:'Balance'},
];

function renderJournal(){
const start = page * PAGE_SIZE;
const pageData = filtered.slice(start, start + PAGE_SIZE);
const thead = document.querySelector('#journal-table thead');
const tbody = document.querySelector('#journal-table tbody');
thead.innerHTML = '<tr>'+cols.map(function(c){return '<th data-col="'+c.key+'">'+c.label+(sortCol===c.key?(sortDir>0?' ▲':' ▼'):'')+'</th>'}).join('')+'</tr>';
thead.querySelectorAll('th').forEach(function(th){
th.addEventListener('click',function(){
const col = th.dataset.col;
if(sortCol===col) sortDir*=-1; else { sortCol=col; sortDir=1; }
filtered.sort(function(a,b){
const av=a[col],bv=b[col];
if(typeof av==='number') return (av-bv)*sortDir;
return String(av||'').localeCompare(String(bv||''))*sortDir;
});
renderJournal();
});
});
tbody.innerHTML = pageData.map(function(t){
return '<tr><td>'+t.time.slice(0,19).replace('T',' ')+'</td><td>'+t.strategy+'</td><td>'+t.instrument+'</td><td style="color:'+(t.side==='buy'?'var(--green)':'var(--red)')+'">'+t.side.toUpperCase()+'</td><td><span class="signal-tag on">'+t.signal+'</span></td><td>'+fmt(t.entry,t.instrument.includes('BTC')?0:t.instrument.includes('JPY')?3:5)+'</td><td>'+fmt(t.exit,t.instrument.includes('BTC')?0:t.instrument.includes('JPY')?3:5)+'</td><td style="color:'+(t.pips>=0?'var(--green)':'var(--red)')+'">'+fmt(t.pips,1)+'</td><td>'+pnlTd(t.pnl)+'</td><td>'+t.reason+'</td><td>v'+t.config_version+'</td><td>'+fmt(t.balance)+'</td></tr>';
}).join('');
const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
document.getElementById('journal-pagination').innerHTML = '<button '+(page===0?'disabled':'')+' onclick="window._jPage('+(page-1)+')">← Prev</button><span>Page '+(page+1)+' of '+totalPages+' ('+filtered.length+' trades)</span><button '+(page>=totalPages-1?'disabled':'')+' onclick="window._jPage('+(page+1)+')">Next →</button>';
}
window._jPage = function(p){ page=p; renderJournal(); };
renderJournal();
})();

// ===== ANALYTICS =====
(function(){
const el = document.getElementById('analytics-content');
const stratMap = {};
TRADES.forEach(function(t){if(!stratMap[t.strategy])stratMap[t.strategy]=[];stratMap[t.strategy].push(t)});

let stratHtml = '<table class="analytics-table"><tr><th>Strategy</th><th>Trades</th><th>Win Rate</th><th>Net P&L</th><th>PF</th><th>Max DD</th></tr>';
Object.entries(stratMap).forEach(function([k,v]){
const m=computeMetrics(v);
stratHtml += '<tr><td style="font-weight:600">'+k+'</td><td>'+m.trades+'</td><td style="color:'+(m.winRate>=0.5?'var(--green)':'var(--red)')+'">'+fmtPct(m.winRate)+'</td><td>'+pnlTd(m.totalPnl)+'</td><td>'+fmt(m.profitFactor)+'</td><td class="pnl-neg">'+fmtMoney(-m.maxDrawdown)+'</td></tr>';
});
stratHtml += '</table>';

const instMap = {};
TRADES.forEach(function(t){if(!instMap[t.instrument])instMap[t.instrument]=[];instMap[t.instrument].push(t)});
let instHtml = '<table class="analytics-table"><tr><th>Instrument</th><th>Trades</th><th>Win Rate</th><th>Net P&L</th><th>PF</th></tr>';
Object.entries(instMap).forEach(function([k,v]){
const m=computeMetrics(v);
instHtml += '<tr><td style="font-weight:600">'+k+'</td><td>'+m.trades+'</td><td style="color:'+(m.winRate>=0.5?'var(--green)':'var(--red)')+'">'+fmtPct(m.winRate)+'</td><td>'+pnlTd(m.totalPnl)+'</td><td>'+fmt(m.profitFactor)+'</td></tr>';
});
instHtml += '</table>';

const sigMap = {};
TRADES.forEach(function(t){if(!sigMap[t.signal])sigMap[t.signal]=[];sigMap[t.signal].push(t)});
let sigHtml = '<table class="analytics-table"><tr><th>Signal</th><th>Trades</th><th>Win Rate</th><th>Net P&L</th><th>PF</th></tr>';
Object.entries(sigMap).sort(function(a,b){return computeMetrics(b[1]).totalPnl-computeMetrics(a[1]).totalPnl}).forEach(function([k,v]){
const m=computeMetrics(v);
sigHtml += '<tr><td style="font-weight:600">'+k+'</td><td>'+m.trades+'</td><td style="color:'+(m.winRate>=0.5?'var(--green)':'var(--red)')+'">'+fmtPct(m.winRate)+'</td><td>'+pnlTd(m.totalPnl)+'</td><td>'+fmt(m.profitFactor)+'</td></tr>';
});
sigHtml += '</table>';

const cfgMap = {};
TRADES.forEach(function(t){const k='v'+t.config_version;if(!cfgMap[k])cfgMap[k]=[];cfgMap[k].push(t)});
let cfgHtml = '<table class="analytics-table"><tr><th>Version</th><th>Trades</th><th>Win Rate</th><th>Net P&L</th><th>PF</th></tr>';
Object.entries(cfgMap).sort(function(a,b){return parseInt(a[0].slice(1))-parseInt(b[0].slice(1))}).forEach(function([k,v]){
const m=computeMetrics(v);
cfgHtml += '<tr><td style="font-weight:600">'+k+'</td><td>'+m.trades+'</td><td style="color:'+(m.winRate>=0.5?'var(--green)':'var(--red)')+'">'+fmtPct(m.winRate)+'</td><td>'+pnlTd(m.totalPnl)+'</td><td>'+fmt(m.profitFactor)+'</td></tr>';
});
cfgHtml += '</table>';

const sessions = {Asian:[],London:[],NewYork:[],Other:[]};
TRADES.forEach(function(t){
const h=parseInt(t.time.slice(11,13));
if(h>=0&&h<8) sessions.Asian.push(t);
else if(h>=8&&h<14) sessions.London.push(t);
else if(h>=14&&h<21) sessions.NewYork.push(t);
else sessions.Other.push(t);
});
let sessionHtml = '<table class="analytics-table"><tr><th>Session</th><th>Trades</th><th>Win Rate</th><th>Net P&L</th><th>PF</th></tr>';
Object.entries(sessions).forEach(function([k,v]){
if(!v.length) return;
const m=computeMetrics(v);
sessionHtml += '<tr><td style="font-weight:600">'+k+'</td><td>'+m.trades+'</td><td style="color:'+(m.winRate>=0.5?'var(--green)':'var(--red)')+'">'+fmtPct(m.winRate)+'</td><td>'+pnlTd(m.totalPnl)+'</td><td>'+fmt(m.profitFactor)+'</td></tr>';
});
sessionHtml += '</table>';

const wins = TRADES.filter(function(t){return t.pnl>0});
const losses = TRADES.filter(function(t){return t.pnl<0});
const largestWin = wins.length?Math.max(...wins.map(function(t){return t.pnl})):0;
const largestLoss = losses.length?Math.min(...losses.map(function(t){return t.pnl})):0;
const riskHtml = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px"><div><div class="kpi-label">Max Consecutive Wins</div><div class="kpi-value positive" style="font-size:18px">'+allMetrics.maxConsecWin+'</div></div><div><div class="kpi-label">Max Consecutive Losses</div><div class="kpi-value negative" style="font-size:18px">'+allMetrics.maxConsecLoss+'</div></div><div><div class="kpi-label">Largest Winner</div><div class="kpi-value positive" style="font-size:18px">'+fmtMoney(largestWin)+'</div></div><div><div class="kpi-label">Largest Loser</div><div class="kpi-value negative" style="font-size:18px">'+fmtMoney(largestLoss)+'</div></div><div><div class="kpi-label">Recovery Factor</div><div class="kpi-value neutral" style="font-size:18px">'+(allMetrics.maxDrawdown?fmt(allMetrics.totalPnl/allMetrics.maxDrawdown):'—')+'</div></div><div><div class="kpi-label">Expectancy</div><div class="kpi-value '+pnlClass(allMetrics.expectancy)+'" style="font-size:18px">'+fmtMoney(allMetrics.expectancy)+'</div></div></div>';

el.innerHTML = '<div class="card"><div class="card-header"><span class="card-title">📊 By Strategy</span></div>'+stratHtml+'</div>'+
'<div class="card"><div class="card-header"><span class="card-title">🏦 By Instrument</span></div>'+instHtml+'</div>'+
'<div class="card"><div class="card-header"><span class="card-title">📡 By Signal</span></div>'+sigHtml+'</div>'+
'<div class="card"><div class="card-header"><span class="card-title">🔄 By Config Version</span></div>'+cfgHtml+'</div>'+
'<div class="card"><div class="card-header"><span class="card-title">🕐 By Session</span></div>'+sessionHtml+'</div>'+
'<div class="card"><div class="card-header"><span class="card-title">⚠️ Risk Analysis</span></div>'+riskHtml+'</div>';

// P&L Distribution
const distCard = document.createElement('div');
distCard.className = 'card';
distCard.style.gridColumn = '1/-1';
distCard.innerHTML = '<div class="card-header"><span class="card-title">📊 P&L Distribution</span></div><canvas id="chart-dist" style="height:250px"></canvas>';
el.appendChild(distCard);
setTimeout(function(){
const pnls = TRADES.map(function(t){return t.pnl});
const min = Math.floor(Math.min(...pnls)*10)/10;
const max = Math.ceil(Math.max(...pnls)*10)/10;
const step = Math.max(0.1, (max-min)/30);
const bins = [];
for(let b=min;b<max;b+=step) bins.push({x:b,count:pnls.filter(function(p){return p>=b&&p<b+step}).length});
new Chart(document.getElementById('chart-dist'),{
type:'bar',
data:{labels:bins.map(function(b){return fmt(b.x,2)}),datasets:[{data:bins.map(function(b){return b.count}),backgroundColor:bins.map(function(b){return b.x>=0?'rgba(63,185,80,0.6)':'rgba(248,81,73,0.6)'}),borderRadius:2}]},
options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{title:{display:true,text:'P&L'}},y:{title:{display:true,text:'Frequency'},grid:{color:'#1c2333'}}}},
});
},200);
})();

// ===== VWAP ANALYSIS =====
(function(){
const el = document.getElementById('vwap-content');
// Categorize trades by signal names that relate to VWAP
const vwapTrades = TRADES.filter(function(t){return t.strategy==='vwap_reversion'});
const aboveTrades = vwapTrades.filter(function(t){return t.signal==='above_upper_band'});
const belowTrades = vwapTrades.filter(function(t){return t.signal==='below_lower_band'});
const otherVwap = vwapTrades.filter(function(t){return t.signal!=='above_upper_band'&&t.signal!=='below_lower_band'});
const mAbove = computeMetrics(aboveTrades);
const mBelow = computeMetrics(belowTrades);
const mAll = computeMetrics(vwapTrades);
const mOther = computeMetrics(otherVwap);

// Also categorize all trades by buy below entry vs above (rough VWAP proxy)
const allBuys = TRADES.filter(function(t){return t.side==='buy'});
const allSells = TRADES.filter(function(t){return t.side==='sell'});
const mBuys = computeMetrics(allBuys);
const mSells = computeMetrics(allSells);

el.innerHTML = '<div class="card"><div class="card-header"><span class="card-title">🎯 VWAP Reversion Strategy Overview</span></div>'+
'<div class="vwap-grid"><div class="card" style="margin:0"><div class="vwap-stat"><div class="vwap-stat-label">Total VWAP Trades</div><div class="vwap-stat-value" style="color:var(--blue)">'+mAll.trades+'</div><div class="vwap-stat-label">Net P&L: <span style="color:'+(mAll.totalPnl>=0?'var(--green)':'var(--red)')+'">'+fmtMoney(mAll.totalPnl)+'</span></div></div></div>'+
'<div class="card" style="margin:0"><div class="vwap-stat"><div class="vwap-stat-label">Win Rate</div><div class="vwap-stat-value" style="color:'+(mAll.winRate>=0.5?'var(--green)':'var(--red)')+'">'+fmtPct(mAll.winRate)+'</div><div class="vwap-stat-label">Profit Factor: '+fmt(mAll.profitFactor)+'</div></div></div></div></div>'+

'<div class="card"><div class="card-header"><span class="card-title">📊 Above vs Below VWAP Band</span></div>'+
'<div class="vwap-grid"><div class="card" style="margin:0;border-left:3px solid var(--red)"><div class="vwap-stat"><div class="vwap-stat-label">🔴 Above Upper Band (Short)</div><div class="vwap-stat-value">'+mAbove.trades+' trades</div><div style="font-size:12px;color:var(--text2);margin-top:8px"><div>Win Rate: <span style="color:'+(mAbove.winRate>=0.5?'var(--green)':'var(--red)')+'">'+fmtPct(mAbove.winRate)+'</span></div><div>Net P&L: '+pnlTd(mAbove.totalPnl)+'</div><div>PF: '+fmt(mAbove.profitFactor)+'</div></div></div></div>'+
'<div class="card" style="margin:0;border-left:3px solid var(--green)"><div class="vwap-stat"><div class="vwap-stat-label">🟢 Below Lower Band (Long)</div><div class="vwap-stat-value">'+mBelow.trades+' trades</div><div style="font-size:12px;color:var(--text2);margin-top:8px"><div>Win Rate: <span style="color:'+(mBelow.winRate>=0.5?'var(--green)':'var(--red)')+'">'+fmtPct(mBelow.winRate)+'</span></div><div>Net P&L: '+pnlTd(mBelow.totalPnl)+'</div><div>PF: '+fmt(mBelow.profitFactor)+'</div></div></div></div></div></div>'+

(otherVwap.length?'<div class="card"><div class="card-header"><span class="card-title">🔄 Other VWAP Signals (ob_momentum, etc)</span></div><div class="vwap-stat"><div>'+mOther.trades+' trades | Win Rate: '+fmtPct(mOther.winRate)+' | Net P&L: '+fmtMoney(mOther.totalPnl)+'</div></div></div>':'')+

'<div class="card"><div class="card-header"><span class="card-title">📈 VWAP Strategy Equity Curve</span></div><canvas id="chart-vwap-eq" style="height:300px"></canvas></div>'+

'<div class="card"><div class="card-header"><span class="card-title">⚖️ Overall Buy vs Sell Performance</span></div>'+
'<div class="vwap-grid"><div class="card" style="margin:0;border-left:3px solid var(--green)"><div class="vwap-stat"><div class="vwap-stat-label">🟢 All Buys</div><div class="vwap-stat-value">'+mBuys.trades+'</div><div style="font-size:12px;color:var(--text2)">Win: '+fmtPct(mBuys.winRate)+' | P&L: '+fmtMoney(mBuys.totalPnl)+'</div></div></div>'+
'<div class="card" style="margin:0;border-left:3px solid var(--red)"><div class="vwap-stat"><div class="vwap-stat-label">🔴 All Sells</div><div class="vwap-stat-value">'+mSells.trades+'</div><div style="font-size:12px;color:var(--text2)">Win: '+fmtPct(mSells.winRate)+' | P&L: '+fmtMoney(mSells.totalPnl)+'</div></div></div></div></div>';

// VWAP equity curve
if(vwapTrades.length){
setTimeout(function(){
const sorted = [...vwapTrades].sort(function(a,b){return a.time.localeCompare(b.time)});
let cum=0;
const data = sorted.map(function(t){cum+=t.pnl;return cum});
new Chart(document.getElementById('chart-vwap-eq'),{
type:'line',
data:{labels:sorted.map(function(_,i){return i}),datasets:[{data:data,borderColor:'#bc8cff',backgroundColor:'rgba(188,140,255,0.1)',fill:true,pointRadius:0,borderWidth:2,tension:0.1}]},
options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{display:false},y:{grid:{color:'#1c2333'}}}},
});
},300);
}
})();
</script>
</body>
</html>'''

with open('/tmp/vwap-dash/index.html', 'w') as f:
    f.write(html)

print("Done! Size:", len(html))
