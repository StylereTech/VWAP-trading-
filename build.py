#!/usr/bin/env python3
"""Build the trading dashboard HTML with embedded data."""
import json, os

# Load trade data
trades = []
with open("/home/rlagents/.openclaw/workspace/trading-lab/data/all-trades.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            trades.append(json.loads(line))

# Load strategies config
with open("/home/rlagents/.openclaw/workspace/trading-lab/configs/strategies.json") as f:
    strategies = json.load(f)

# Load live data
with open("/tmp/vwap-dash/live_data.json") as f:
    live_data = json.load(f)

trades_json = json.dumps(trades)
strategies_json = json.dumps(strategies)
live_json = json.dumps(live_data)

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TradeZilla — Trading Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
:root{{
--bg:#0d1117;--bg2:#161b22;--bg3:#1c2333;--bg4:#21283b;
--border:#30363d;--border2:#444c56;
--text:#e6edf3;--text2:#8b949e;--text3:#6e7681;
--green:#3fb950;--green2:#238636;--green-bg:rgba(63,185,80,0.1);
--red:#f85149;--red2:#da3633;--red-bg:rgba(248,81,73,0.1);
--blue:#58a6ff;--blue2:#1f6feb;--blue-bg:rgba(88,166,255,0.1);
--gold:#d29922;--gold-bg:rgba(210,153,34,0.1);
--purple:#bc8cff;--orange:#f0883e;
}}
body{{font-family:'Inter',system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);overflow-x:hidden;min-height:100vh}}
.app{{display:flex;flex-direction:column;min-height:100vh}}

/* Top Nav */
.topnav{{background:var(--bg2);border-bottom:1px solid var(--border);padding:0 24px;display:flex;align-items:center;height:56px;position:sticky;top:0;z-index:100}}
.logo{{font-size:20px;font-weight:800;background:linear-gradient(135deg,var(--green),var(--blue));-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-right:40px;letter-spacing:-0.5px}}
.logo span{{font-size:12px;font-weight:400;-webkit-text-fill-color:var(--text3);margin-left:6px}}
.tabs{{display:flex;gap:2px;height:100%}}
.tab{{padding:0 20px;height:100%;display:flex;align-items:center;cursor:pointer;color:var(--text2);font-size:13px;font-weight:500;border-bottom:2px solid transparent;transition:all .2s}}
.tab:hover{{color:var(--text);background:var(--bg3)}}
.tab.active{{color:var(--text);border-bottom-color:var(--blue)}}
.tab .badge{{background:var(--blue2);color:#fff;font-size:10px;padding:1px 6px;border-radius:10px;margin-left:6px}}
.nav-right{{margin-left:auto;display:flex;align-items:center;gap:12px}}
.nav-right .status{{font-size:11px;color:var(--text3)}}
.nav-right .dot{{width:8px;height:8px;border-radius:50%;background:var(--green);display:inline-block;margin-right:4px}}

/* Main Content */
.content{{flex:1;padding:24px;max-width:1600px;margin:0 auto;width:100%}}
.page{{display:none}}
.page.active{{display:block}}

/* Cards */
.card{{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:20px;margin-bottom:16px}}
.card-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}}
.card-title{{font-size:14px;font-weight:600;color:var(--text)}}
.card-subtitle{{font-size:12px;color:var(--text3)}}

/* KPI Grid */
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:20px}}
.kpi{{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:16px}}
.kpi-label{{font-size:11px;font-weight:500;color:var(--text3);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px}}
.kpi-value{{font-size:22px;font-weight:700}}
.kpi-value.positive{{color:var(--green)}}
.kpi-value.negative{{color:var(--red)}}
.kpi-value.neutral{{color:var(--blue)}}
.kpi-sub{{font-size:11px;color:var(--text3);margin-top:4px}}

/* Charts Grid */
.charts-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px}}
.charts-grid .card{{min-height:300px}}
.chart-full{{grid-column:1/-1}}

/* Calendar Heatmap */
.heatmap-wrap{{overflow-x:auto;padding:8px 0}}
.heatmap{{display:flex;gap:3px}}
.heatmap-week{{display:flex;flex-direction:column;gap:3px}}
.heatmap-day{{width:14px;height:14px;border-radius:3px;cursor:pointer;position:relative}}
.heatmap-day:hover{{outline:2px solid var(--blue);outline-offset:1px}}
.heatmap-day .tip{{display:none;position:absolute;bottom:calc(100% + 6px);left:50%;transform:translateX(-50%);background:var(--bg4);border:1px solid var(--border2);border-radius:6px;padding:6px 10px;font-size:11px;white-space:nowrap;z-index:10;pointer-events:none}}
.heatmap-day:hover .tip{{display:block}}
.heatmap-labels{{display:flex;flex-direction:column;gap:3px;margin-right:6px;font-size:10px;color:var(--text3)}}

/* Strategy Cards */
.strat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:16px}}
.strat-card{{background:var(--bg2);border:1px solid var(--border);border-radius:12px;overflow:hidden;transition:transform .2s}}
.strat-card:hover{{transform:translateY(-2px)}}
.strat-card-head{{padding:16px 20px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--border)}}
.strat-name{{font-size:15px;font-weight:700}}
.strat-instrument{{font-size:12px;color:var(--text2);background:var(--bg3);padding:3px 8px;border-radius:6px}}
.strat-card-body{{padding:20px}}
.strat-metrics{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:16px}}
.strat-metric-label{{font-size:10px;color:var(--text3);text-transform:uppercase}}
.strat-metric-value{{font-size:16px;font-weight:600;margin-top:2px}}
.strat-chart{{height:80px;margin-bottom:12px}}
.strat-signals{{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:12px}}
.signal-tag{{font-size:10px;padding:2px 8px;border-radius:4px;border:1px solid var(--border)}}
.signal-tag.on{{background:var(--green-bg);border-color:var(--green2);color:var(--green)}}
.signal-tag.off{{background:var(--bg3);color:var(--text3)}}
.strat-banner{{padding:10px 20px;font-size:12px;font-weight:500;display:flex;align-items:center;gap:6px}}
.strat-banner.warn{{background:rgba(210,153,34,0.1);color:var(--gold);border-top:1px solid rgba(210,153,34,0.2)}}
.strat-banner.good{{background:var(--green-bg);color:var(--green);border-top:1px solid rgba(63,185,80,0.2)}}
.strat-coming{{opacity:0.5;position:relative}}
.strat-coming::after{{content:'COMING SOON';position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);background:var(--bg);border:2px solid var(--gold);color:var(--gold);padding:8px 20px;border-radius:8px;font-weight:700;font-size:14px;letter-spacing:1px}}
.strat-recent{{border-top:1px solid var(--border);margin-top:12px;padding-top:12px}}
.strat-recent-title{{font-size:11px;color:var(--text3);margin-bottom:6px}}
.strat-recent-row{{display:flex;justify-content:space-between;font-size:11px;padding:3px 0;border-bottom:1px solid rgba(48,54,61,0.5)}}

/* Tables */
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{text-align:left;padding:10px 12px;color:var(--text3);font-weight:600;text-transform:uppercase;font-size:10px;letter-spacing:0.5px;border-bottom:2px solid var(--border);cursor:pointer;user-select:none;white-space:nowrap}}
th:hover{{color:var(--text)}}
td{{padding:8px 12px;border-bottom:1px solid var(--border);white-space:nowrap}}
tr:hover td{{background:var(--bg3)}}
.pnl-pos{{color:var(--green);font-weight:600}}
.pnl-neg{{color:var(--red);font-weight:600}}

/* Filters */
.filters{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px;align-items:center}}
.filter-select,.filter-input{{background:var(--bg3);border:1px solid var(--border);color:var(--text);padding:6px 12px;border-radius:6px;font-size:12px;font-family:inherit}}
.filter-select:focus,.filter-input:focus{{outline:none;border-color:var(--blue)}}
.filter-label{{font-size:11px;color:var(--text3);margin-right:4px}}

/* Pagination */
.pagination{{display:flex;gap:6px;align-items:center;justify-content:center;padding:16px 0}}
.pagination button{{background:var(--bg3);border:1px solid var(--border);color:var(--text);padding:6px 14px;border-radius:6px;cursor:pointer;font-size:12px;font-family:inherit}}
.pagination button:hover{{background:var(--bg4)}}
.pagination button.active{{background:var(--blue2);border-color:var(--blue)}}
.pagination span{{color:var(--text3);font-size:12px}}

/* Live Account */
.live-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:20px}}
.live-kpi{{background:var(--bg3);border-radius:10px;padding:16px;text-align:center}}
.live-kpi-label{{font-size:11px;color:var(--text3);text-transform:uppercase;letter-spacing:0.5px}}
.live-kpi-value{{font-size:28px;font-weight:700;margin-top:4px}}
.error-banner{{background:rgba(248,81,73,0.1);border:1px solid var(--red2);border-radius:8px;padding:16px;color:var(--red);font-size:13px;margin-bottom:16px}}

/* Analytics */
.analytics-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.analytics-grid .card{{min-height:250px}}
.analytics-table{{font-size:12px}}
.analytics-table th{{font-size:10px}}

/* Scrollbar */
::-webkit-scrollbar{{width:8px;height:8px}}
::-webkit-scrollbar-track{{background:var(--bg)}}
::-webkit-scrollbar-thumb{{background:var(--border);border-radius:4px}}
::-webkit-scrollbar-thumb:hover{{background:var(--border2)}}

/* Responsive */
@media(max-width:900px){{
.charts-grid,.analytics-grid{{grid-template-columns:1fr}}
.strat-grid{{grid-template-columns:1fr}}
.kpi-grid{{grid-template-columns:repeat(auto-fit,minmax(130px,1fr))}}
}}
</style>
</head>
<body>
<div class="app">
<nav class="topnav">
<div class="logo">TradeZilla <span>v2.0</span></div>
<div class="tabs">
<div class="tab active" data-page="dashboard">📊 Dashboard</div>
<div class="tab" data-page="demo">🧪 Demo Accounts <span class="badge">4</span></div>
<div class="tab" data-page="live">💰 Live Account</div>
<div class="tab" data-page="journal">📒 Trade Journal</div>
<div class="tab" data-page="analytics">📈 Analytics</div>
</div>
<div class="nav-right">
<span class="status"><span class="dot"></span>Bots Running</span>
<span class="status" id="trade-count-nav"></span>
</div>
</nav>

<main class="content">
<!-- DASHBOARD PAGE -->
<div class="page active" id="page-dashboard">
<div id="kpi-row" class="kpi-grid"></div>
<div class="card">
<div class="card-header"><span class="card-title">📅 Daily P&L Calendar</span><span class="card-subtitle" id="cal-range"></span></div>
<div class="heatmap-wrap"><div id="heatmap" class="heatmap"></div></div>
</div>
<div class="charts-grid">
<div class="card chart-full"><div class="card-header"><span class="card-title">Cumulative P&L</span></div><canvas id="chart-cumPnl"></canvas></div>
<div class="card"><div class="card-header"><span class="card-title">P&L by Day of Week</span></div><canvas id="chart-dow"></canvas></div>
<div class="card"><div class="card-header"><span class="card-title">P&L by Hour</span></div><canvas id="chart-hour"></canvas></div>
<div class="card"><div class="card-header"><span class="card-title">P&L by Strategy</span></div><canvas id="chart-strat"></canvas></div>
<div class="card"><div class="card-header"><span class="card-title">Running Balance</span></div><canvas id="chart-balance"></canvas></div>
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
<h2 style="font-size:18px;margin-bottom:12px;font-weight:700">📒 Trade Journal</h2>
<div class="filters" id="journal-filters"></div>
<div class="card" style="overflow-x:auto"><table id="journal-table"><thead></thead><tbody></tbody></table></div>
<div class="pagination" id="journal-pagination"></div>
</div>

<!-- ANALYTICS PAGE -->
<div class="page" id="page-analytics">
<h2 style="font-size:18px;margin-bottom:16px;font-weight:700">📈 Deep Analytics</h2>
<div class="analytics-grid" id="analytics-content"></div>
</div>
</main>
</div>

<script>
// ===== EMBEDDED DATA =====
const TRADES = {trades_json};
const CONFIG = {strategies_json};
const LIVE = {live_json};

// ===== UTILITIES =====
const fmt = (n,d=2) => n!=null ? Number(n).toFixed(d) : '—';
const fmtPct = n => n!=null ? (n*100).toFixed(1)+'%' : '—';
const fmtMoney = n => n!=null ? (n>=0?'+':'')+Number(n).toFixed(2) : '—';
const pnlClass = n => n>=0?'positive':'negative';
const pnlTd = n => `<span class="pnl-${{n>=0?'pos':'neg'}}">${{fmtMoney(n)}}</span>`;

// ===== TAB NAVIGATION =====
document.querySelectorAll('.tab').forEach(t=>{{
t.addEventListener('click',()=>{{
document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
document.querySelectorAll('.page').forEach(x=>x.classList.remove('active'));
t.classList.add('active');
document.getElementById('page-'+t.dataset.page).classList.add('active');
}});
}});

// ===== COMPUTE METRICS =====
function computeMetrics(trades){{
if(!trades.length) return {{}};
const wins = trades.filter(t=>t.pnl>0);
const losses = trades.filter(t=>t.pnl<0);
const totalPnl = trades.reduce((s,t)=>s+t.pnl,0);
const winRate = wins.length/trades.length;
const avgWin = wins.length? wins.reduce((s,t)=>s+t.pnl,0)/wins.length : 0;
const avgLoss = losses.length? Math.abs(losses.reduce((s,t)=>s+t.pnl,0)/losses.length) : 0;
const profitFactor = losses.length? wins.reduce((s,t)=>s+t.pnl,0)/Math.abs(losses.reduce((s,t)=>s+t.pnl,0)) : wins.length?Infinity:0;

// Daily PnL for drawdown/sharpe
const daily = {{}};
trades.forEach(t=>{{const d=t.time.slice(0,10);daily[d]=(daily[d]||0)+t.pnl}});
const dailyVals = Object.values(daily);
const avgDaily = dailyVals.reduce((a,b)=>a+b,0)/dailyVals.length;
const stdDaily = Math.sqrt(dailyVals.reduce((s,v)=>s+Math.pow(v-avgDaily,2),0)/dailyVals.length);
const sharpe = stdDaily?avgDaily/stdDaily*Math.sqrt(252):0;

// Max drawdown
let peak=0,maxDD=0,running=0;
dailyVals.forEach(v=>{{running+=v;if(running>peak)peak=running;const dd=peak-running;if(dd>maxDD)maxDD=dd}});

const bestDay = Math.max(...dailyVals);
const worstDay = Math.min(...dailyVals);
const expectancy = totalPnl/trades.length;

// Consecutive
let maxConsecWin=0,maxConsecLoss=0,cw=0,cl=0;
trades.forEach(t=>{{
if(t.pnl>0){{cw++;cl=0;if(cw>maxConsecWin)maxConsecWin=cw}}
else{{cl++;cw=0;if(cl>maxConsecLoss)maxConsecLoss=cl}}
}});

return {{totalPnl,winRate,profitFactor,avgWin,avgLoss,trades:trades.length,bestDay,worstDay,maxDrawdown:maxDD,sharpe,expectancy,maxConsecWin,maxConsecLoss,daily}};
}}

const allMetrics = computeMetrics(TRADES);
document.getElementById('trade-count-nav').textContent = TRADES.length.toLocaleString()+' trades';

// ===== DASHBOARD: KPIs =====
const kpis = [
{{label:'Net P&L',value:fmtMoney(allMetrics.totalPnl),cls:pnlClass(allMetrics.totalPnl)}},
{{label:'Win Rate',value:fmtPct(allMetrics.winRate),cls:allMetrics.winRate>=0.5?'positive':'negative'}},
{{label:'Profit Factor',value:fmt(allMetrics.profitFactor),cls:allMetrics.profitFactor>=1?'positive':'negative'}},
{{label:'Avg Win/Loss',value:allMetrics.avgLoss?fmt(allMetrics.avgWin/allMetrics.avgLoss):'—',cls:'neutral'}},
{{label:'Total Trades',value:allMetrics.trades.toLocaleString(),cls:'neutral'}},
{{label:'Best Day',value:fmtMoney(allMetrics.bestDay),cls:'positive'}},
{{label:'Worst Day',value:fmtMoney(allMetrics.worstDay),cls:'negative'}},
{{label:'Max Drawdown',value:fmtMoney(-allMetrics.maxDrawdown),cls:'negative'}},
{{label:'Sharpe Ratio',value:fmt(allMetrics.sharpe),cls:allMetrics.sharpe>=1?'positive':'neutral'}},
{{label:'Expectancy',value:fmtMoney(allMetrics.expectancy),cls:pnlClass(allMetrics.expectancy)}},
];
document.getElementById('kpi-row').innerHTML = kpis.map(k=>`<div class="kpi"><div class="kpi-label">${{k.label}}</div><div class="kpi-value ${{k.cls}}">${{k.value}}</div></div>`).join('');

// ===== DASHBOARD: Calendar Heatmap =====
(function(){{
const daily = allMetrics.daily;
const dates = Object.keys(daily).sort();
if(!dates.length) return;
const start = new Date(dates[0]);
const end = new Date(dates[dates.length-1]);
document.getElementById('cal-range').textContent = dates[0]+' → '+dates[dates.length-1];

const maxAbs = Math.max(...Object.values(daily).map(Math.abs),1);
let html = '';
let cur = new Date(start);
cur.setDate(cur.getDate()-cur.getDay()); // Start on Sunday
let weekHtml = '';
let weekDay = 0;

while(cur<=end||weekDay>0){{
const ds = cur.toISOString().slice(0,10);
const val = daily[ds]||0;
const intensity = Math.min(Math.abs(val)/maxAbs,1);
let color;
if(val>0) color = `rgba(63,185,80,${{0.15+intensity*0.85}})`;
else if(val<0) color = `rgba(248,81,73,${{0.15+intensity*0.85}})`;
else color = 'var(--bg3)';
weekHtml += `<div class="heatmap-day" style="background:${{color}}" title="${{ds}}: ${{fmtMoney(val)}}"><div class="tip">${{ds}}<br>${{fmtMoney(val)}}</div></div>`;
weekDay++;
if(weekDay===7){{
html += `<div class="heatmap-week">${{weekHtml}}</div>`;
weekHtml = '';
weekDay = 0;
}}
cur.setDate(cur.getDate()+1);
}}
if(weekHtml) html += `<div class="heatmap-week">${{weekHtml}}</div>`;
document.getElementById('heatmap').innerHTML = html;
}})();

// ===== CHART DEFAULTS =====
Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#30363d';
Chart.defaults.font.family = 'Inter, system-ui';
Chart.defaults.font.size = 11;
const chartOpts = (title) => ({{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},title:{{display:false}}}}}});

// ===== DASHBOARD: Cumulative PnL =====
(function(){{
const sorted = [...TRADES].sort((a,b)=>a.time.localeCompare(b.time));
let cum = 0;
const data = sorted.map(t=>{{cum+=t.pnl;return cum}});
const labels = sorted.map((_,i)=>i);
new Chart(document.getElementById('chart-cumPnl'),{{
type:'line',
data:{{labels,datasets:[{{data,borderColor:'#58a6ff',backgroundColor:'rgba(88,166,255,0.1)',fill:true,pointRadius:0,borderWidth:1.5,tension:0.1}}]}},
options:{{...chartOpts(),scales:{{x:{{display:false}},y:{{grid:{{color:'#1c2333'}}}}}}}}
}});
}})();

// ===== DASHBOARD: P&L by Day of Week =====
(function(){{
const dow = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
const pnl = Array(7).fill(0);
TRADES.forEach(t=>{{const d=new Date(t.time).getDay();pnl[d]+=t.pnl}});
new Chart(document.getElementById('chart-dow'),{{
type:'bar',
data:{{labels:dow,datasets:[{{data:pnl,backgroundColor:pnl.map(v=>v>=0?'rgba(63,185,80,0.7)':'rgba(248,81,73,0.7)'),borderRadius:4}}]}},
options:{{...chartOpts(),scales:{{y:{{grid:{{color:'#1c2333'}}}}}}}},
}});
}})();

// ===== DASHBOARD: P&L by Hour =====
(function(){{
const pnl = Array(24).fill(0);
TRADES.forEach(t=>{{const h=parseInt(t.time.slice(11,13));pnl[h]+=t.pnl}});
new Chart(document.getElementById('chart-hour'),{{
type:'bar',
data:{{labels:Array.from({{length:24}},(_,i)=>i+':00'),datasets:[{{data:pnl,backgroundColor:pnl.map(v=>v>=0?'rgba(63,185,80,0.7)':'rgba(248,81,73,0.7)'),borderRadius:3}}]}},
options:{{...chartOpts(),scales:{{y:{{grid:{{color:'#1c2333'}}}}}}}},
}});
}})();

// ===== DASHBOARD: P&L by Strategy (Donut) =====
(function(){{
const strats = {{}};
TRADES.forEach(t=>{{strats[t.strategy]=(strats[t.strategy]||0)+t.pnl}});
const labels = Object.keys(strats);
const data = Object.values(strats);
const colors = ['#58a6ff','#3fb950','#f0883e','#bc8cff','#d29922'];
new Chart(document.getElementById('chart-strat'),{{
type:'doughnut',
data:{{labels,datasets:[{{data:data.map(Math.abs),backgroundColor:colors.slice(0,labels.length),borderWidth:0}}]}},
options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{position:'right',labels:{{boxWidth:10,padding:8,font:{{size:11}},generateLabels:(chart)=>{{
return chart.data.labels.map((l,i)=>({{text:l+' ('+fmtMoney(data[i])+')',fillStyle:colors[i],strokeStyle:colors[i]}}))}}}}}}}}}},
}});
}})();

// ===== DASHBOARD: Running Balance =====
(function(){{
const sorted = [...TRADES].sort((a,b)=>a.time.localeCompare(b.time));
const data = sorted.map(t=>t.balance);
new Chart(document.getElementById('chart-balance'),{{
type:'line',
data:{{labels:sorted.map((_,i)=>i),datasets:[{{data,borderColor:'#d29922',backgroundColor:'rgba(210,153,34,0.1)',fill:true,pointRadius:0,borderWidth:1.5,tension:0.1}}]}},
options:{{...chartOpts(),scales:{{x:{{display:false}},y:{{grid:{{color:'#1c2333'}}}}}}}}
}});
}})();

// ===== DEMO ACCOUNTS =====
(function(){{
const stratDefs = [
{{key:'hft_momentum',name:'HFT Momentum',icon:'⚡',instrument:'GBPJPY'}},
{{key:'vwap_reversion',name:'VWAP Reversion',icon:'📊',instrument:'BTCUSD'}},
{{key:'range_scalper',name:'Range Scalper',icon:'📐',instrument:'EURUSD'}},
{{key:'gold_momentum',name:'Gold Momentum',icon:'🥇',instrument:'XAUUSD'}},
{{key:'trend_follower',name:'Trend Follower',icon:'📈',instrument:'TBD',coming:true}},
];

const container = document.getElementById('strat-cards');
stratDefs.forEach((sd,idx)=>{{
const trades = TRADES.filter(t=>t.strategy===sd.key).sort((a,b)=>a.time.localeCompare(b.time));
const m = computeMetrics(trades);
const cfg = CONFIG.strategies[sd.key];
const recent = trades.slice(-5).reverse();

let bannerHtml = '';
if(m.trades>0){{
if(m.winRate<0.45) bannerHtml = `<div class="strat-banner warn">⚠️ Win rate ${{fmtPct(m.winRate)}} — consider tuning signals</div>`;
else if(m.profitFactor<1) bannerHtml = `<div class="strat-banner warn">⚠️ Profit factor ${{fmt(m.profitFactor)}} — strategy losing money</div>`;
else if(m.profitFactor>1.5) bannerHtml = `<div class="strat-banner good">✅ Strong PF ${{fmt(m.profitFactor)}} — strategy performing well</div>`;
}}

const signalHtml = cfg? Object.entries(cfg.signals||{{}}).map(([k,v])=>`<span class="signal-tag ${{v.enabled?'on':'off'}}">${{k}}</span>`).join('') : '';

const configHtml = cfg? `<div style="font-size:11px;color:var(--text3);margin-top:8px">TP: ${{cfg.tp_pips}} | SL: ${{cfg.sl_pips}} | Max Pos: ${{cfg.max_positions}}</div>` : '';

const recentHtml = recent.length? `<div class="strat-recent"><div class="strat-recent-title">Recent Trades</div>${{recent.map(t=>`<div class="strat-recent-row"><span>${{t.time.slice(5,16)}}</span><span>${{t.side}}</span><span>${{t.signal}}</span>${{pnlTd(t.pnl)}}<span style="color:var(--text3)">${{t.reason}}</span></div>`).join('')}}</div>` : '';

container.innerHTML += `
<div class="strat-card ${{sd.coming?'strat-coming':''}}">
<div class="strat-card-head">
<span class="strat-name">${{sd.icon}} ${{sd.name}}</span>
<span class="strat-instrument">${{sd.instrument}}</span>
</div>
<div class="strat-card-body">
<div class="strat-chart"><canvas id="strat-chart-${{idx}}"></canvas></div>
<div class="strat-metrics">
<div><div class="strat-metric-label">Win Rate</div><div class="strat-metric-value" style="color:${{m.winRate>=0.5?'var(--green)':'var(--red)'}}">${{fmtPct(m.winRate||0)}}</div></div>
<div><div class="strat-metric-label">Net P&L</div><div class="strat-metric-value" style="color:${{(m.totalPnl||0)>=0?'var(--green)':'var(--red)'}}">${{fmtMoney(m.totalPnl||0)}}</div></div>
<div><div class="strat-metric-label">Trades</div><div class="strat-metric-value">${{m.trades||0}}</div></div>
<div><div class="strat-metric-label">Profit Factor</div><div class="strat-metric-value">${{fmt(m.profitFactor||0)}}</div></div>
<div><div class="strat-metric-label">Max DD</div><div class="strat-metric-value" style="color:var(--red)">${{fmtMoney(-(m.maxDrawdown||0))}}</div></div>
<div><div class="strat-metric-label">Avg Win/Loss</div><div class="strat-metric-value">${{m.avgLoss?fmt(m.avgWin/m.avgLoss):'—'}}</div></div>
</div>
<div class="strat-signals">${{signalHtml}}</div>
${{configHtml}}
${{recentHtml}}
</div>
${{bannerHtml}}
</div>`;

// Mini equity curve
if(trades.length && !sd.coming){{
setTimeout(()=>{{
let cum=0;
const data = trades.map(t=>{{cum+=t.pnl;return cum}});
new Chart(document.getElementById('strat-chart-'+idx),{{
type:'line',
data:{{labels:trades.map((_,i)=>i),datasets:[{{data,borderColor:data[data.length-1]>=0?'#3fb950':'#f85149',backgroundColor:data[data.length-1]>=0?'rgba(63,185,80,0.1)':'rgba(248,81,73,0.1)',fill:true,pointRadius:0,borderWidth:1.5,tension:0.1}}]}},
options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{display:false}},y:{{display:false}}}}}},
}});
}},100);
}}
}});
}})();

// ===== LIVE ACCOUNT =====
(function(){{
const el = document.getElementById('live-content');
if(LIVE.error){{
el.innerHTML = `
<div class="error-banner">⚠️ TradeLocker API connection failed. Credentials may need updating.<br><small>${{LIVE.error.slice(0,200)}}</small></div>
<div class="card">
<div class="card-header"><span class="card-title">Account: GatesFX #684012</span></div>
<div style="padding:40px;text-align:center;color:var(--text3)">
<div style="font-size:48px;margin-bottom:16px">🔒</div>
<div style="font-size:14px">Live data unavailable — authentication failed</div>
<div style="font-size:12px;margin-top:8px">Check TradeLocker credentials and re-run fetch_live.py</div>
</div>
</div>`;
return;
}}
const a = LIVE.account||{{}};
el.innerHTML = `
<div class="live-grid">
<div class="live-kpi"><div class="live-kpi-label">Balance</div><div class="live-kpi-value">$${{fmt(a.balance)}}</div></div>
<div class="live-kpi"><div class="live-kpi-label">Equity</div><div class="live-kpi-value">$${{fmt(a.equity)}}</div></div>
<div class="live-kpi"><div class="live-kpi-label">Margin Used</div><div class="live-kpi-value">$${{fmt(a.margin)}}</div></div>
<div class="live-kpi"><div class="live-kpi-label">Free Margin</div><div class="live-kpi-value">$${{fmt(a.free_margin)}}</div></div>
<div class="live-kpi"><div class="live-kpi-label">Margin Level</div><div class="live-kpi-value">${{a.margin_level||'—'}}%</div></div>
</div>
<div class="card">
<div class="card-header"><span class="card-title">Open Positions</span></div>
<div style="padding:20px;text-align:center;color:var(--text3)">No open positions</div>
</div>
<div class="card">
<div class="card-header"><span class="card-title">Trade History</span></div>
<div style="padding:20px;text-align:center;color:var(--text3)">No trade history available</div>
</div>`;
}})();

// ===== TRADE JOURNAL =====
(function(){{
const PAGE_SIZE = 50;
let page = 0;
let filtered = [...TRADES].sort((a,b)=>b.time.localeCompare(a.time));
const strategies = [...new Set(TRADES.map(t=>t.strategy))].sort();
const instruments = [...new Set(TRADES.map(t=>t.instrument))].sort();
const signals = [...new Set(TRADES.map(t=>t.signal))].sort();

// Filters
const filtersEl = document.getElementById('journal-filters');
filtersEl.innerHTML = `
<span class="filter-label">Strategy:</span>
<select class="filter-select" id="f-strat"><option value="">All</option>${{strategies.map(s=>`<option value="${{s}}">${{s}}</option>`).join('')}}</select>
<span class="filter-label">Instrument:</span>
<select class="filter-select" id="f-inst"><option value="">All</option>${{instruments.map(s=>`<option value="${{s}}">${{s}}</option>`).join('')}}</select>
<span class="filter-label">Signal:</span>
<select class="filter-select" id="f-signal"><option value="">All</option>${{signals.map(s=>`<option value="${{s}}">${{s}}</option>`).join('')}}</select>
<span class="filter-label">Win/Loss:</span>
<select class="filter-select" id="f-wl"><option value="">All</option><option value="win">Win</option><option value="loss">Loss</option></select>
<span class="filter-label">Search:</span>
<input class="filter-input" id="f-search" placeholder="reason, signal..." style="width:140px">
`;

function applyFilters(){{
const strat = document.getElementById('f-strat').value;
const inst = document.getElementById('f-inst').value;
const signal = document.getElementById('f-signal').value;
const wl = document.getElementById('f-wl').value;
const search = document.getElementById('f-search').value.toLowerCase();
filtered = TRADES.filter(t=>{{
if(strat && t.strategy!==strat) return false;
if(inst && t.instrument!==inst) return false;
if(signal && t.signal!==signal) return false;
if(wl==='win' && t.pnl<=0) return false;
if(wl==='loss' && t.pnl>=0) return false;
if(search && !JSON.stringify(t).toLowerCase().includes(search)) return false;
return true;
}}).sort((a,b)=>b.time.localeCompare(a.time));
page = 0;
renderJournal();
}}

filtersEl.querySelectorAll('select,input').forEach(el=>el.addEventListener('change',applyFilters));
document.getElementById('f-search').addEventListener('input',applyFilters);

let sortCol = null, sortDir = 1;
const cols = [
{{key:'time',label:'Date/Time'}},
{{key:'strategy',label:'Strategy'}},
{{key:'instrument',label:'Instrument'}},
{{key:'side',label:'Side'}},
{{key:'signal',label:'Signal'}},
{{key:'entry',label:'Entry'}},
{{key:'exit',label:'Exit'}},
{{key:'pips',label:'Pips'}},
{{key:'pnl',label:'P&L'}},
{{key:'reason',label:'Reason'}},
{{key:'config_version',label:'Config'}},
{{key:'balance',label:'Balance'}},
];

function renderJournal(){{
const start = page * PAGE_SIZE;
const pageData = filtered.slice(start, start + PAGE_SIZE);
const thead = document.querySelector('#journal-table thead');
const tbody = document.querySelector('#journal-table tbody');

thead.innerHTML = '<tr>'+cols.map(c=>`<th data-col="${{c.key}}">${{c.label}}${{sortCol===c.key?(sortDir>0?' ▲':' ▼'):''}}</th>`).join('')+'</tr>';
thead.querySelectorAll('th').forEach(th=>{{
th.addEventListener('click',()=>{{
const col = th.dataset.col;
if(sortCol===col) sortDir*=-1; else {{ sortCol=col; sortDir=1; }}
filtered.sort((a,b)=>{{
const av=a[col],bv=b[col];
if(typeof av==='number') return (av-bv)*sortDir;
return String(av||'').localeCompare(String(bv||''))*sortDir;
}});
renderJournal();
}});
}});

tbody.innerHTML = pageData.map(t=>`<tr>
<td>${{t.time.slice(0,19).replace('T',' ')}}</td>
<td>${{t.strategy}}</td>
<td>${{t.instrument}}</td>
<td style="color:${{t.side==='buy'?'var(--green)':'var(--red)'}}">${{t.side.toUpperCase()}}</td>
<td><span class="signal-tag on">${{t.signal}}</span></td>
<td>${{fmt(t.entry,t.instrument.includes('BTC')?0:t.instrument.includes('JPY')?3:5)}}</td>
<td>${{fmt(t.exit,t.instrument.includes('BTC')?0:t.instrument.includes('JPY')?3:5)}}</td>
<td style="color:${{t.pips>=0?'var(--green)':'var(--red)'}}">${{fmt(t.pips,1)}}</td>
<td>${{pnlTd(t.pnl)}}</td>
<td>${{t.reason}}</td>
<td>v${{t.config_version}}</td>
<td>${{fmt(t.balance)}}</td>
</tr>`).join('');

const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
const pagEl = document.getElementById('journal-pagination');
pagEl.innerHTML = `
<button ${{page===0?'disabled':''}} onclick="window._jPage(${{page-1}})">← Prev</button>
<span>Page ${{page+1}} of ${{totalPages}} (${{filtered.length}} trades)</span>
<button ${{page>=totalPages-1?'disabled':''}} onclick="window._jPage(${{page+1}})">Next →</button>
`;
}}
window._jPage = (p)=>{{ page=p; renderJournal(); }};
renderJournal();
}})();

// ===== ANALYTICS =====
(function(){{
const el = document.getElementById('analytics-content');

// By Strategy
const stratMap = {{}};
TRADES.forEach(t=>{{
if(!stratMap[t.strategy]) stratMap[t.strategy]=[];
stratMap[t.strategy].push(t);
}});
let stratHtml = '<table class="analytics-table"><tr><th>Strategy</th><th>Trades</th><th>Win Rate</th><th>Net P&L</th><th>PF</th><th>Max DD</th></tr>';
Object.entries(stratMap).forEach(([k,v])=>{{
const m=computeMetrics(v);
stratHtml += `<tr><td style="font-weight:600">${{k}}</td><td>${{m.trades}}</td><td style="color:${{m.winRate>=0.5?'var(--green)':'var(--red)'}}">${{fmtPct(m.winRate)}}</td><td>${{pnlTd(m.totalPnl)}}</td><td>${{fmt(m.profitFactor)}}</td><td class="pnl-neg">${{fmtMoney(-m.maxDrawdown)}}</td></tr>`;
}});
stratHtml += '</table>';

// By Instrument
const instMap = {{}};
TRADES.forEach(t=>{{
if(!instMap[t.instrument]) instMap[t.instrument]=[];
instMap[t.instrument].push(t);
}});
let instHtml = '<table class="analytics-table"><tr><th>Instrument</th><th>Trades</th><th>Win Rate</th><th>Net P&L</th><th>PF</th></tr>';
Object.entries(instMap).forEach(([k,v])=>{{
const m=computeMetrics(v);
instHtml += `<tr><td style="font-weight:600">${{k}}</td><td>${{m.trades}}</td><td style="color:${{m.winRate>=0.5?'var(--green)':'var(--red)'}}">${{fmtPct(m.winRate)}}</td><td>${{pnlTd(m.totalPnl)}}</td><td>${{fmt(m.profitFactor)}}</td></tr>`;
}});
instHtml += '</table>';

// By Signal
const sigMap = {{}};
TRADES.forEach(t=>{{
if(!sigMap[t.signal]) sigMap[t.signal]=[];
sigMap[t.signal].push(t);
}});
let sigHtml = '<table class="analytics-table"><tr><th>Signal</th><th>Trades</th><th>Win Rate</th><th>Net P&L</th><th>PF</th></tr>';
Object.entries(sigMap).sort((a,b)=>{{const ma=computeMetrics(a[1]),mb=computeMetrics(b[1]);return mb.totalPnl-ma.totalPnl}}).forEach(([k,v])=>{{
const m=computeMetrics(v);
sigHtml += `<tr><td style="font-weight:600">${{k}}</td><td>${{m.trades}}</td><td style="color:${{m.winRate>=0.5?'var(--green)':'var(--red)'}}">${{fmtPct(m.winRate)}}</td><td>${{pnlTd(m.totalPnl)}}</td><td>${{fmt(m.profitFactor)}}</td></tr>`;
}});
sigHtml += '</table>';

// By Config Version
const cfgMap = {{}};
TRADES.forEach(t=>{{
const k='v'+t.config_version;
if(!cfgMap[k]) cfgMap[k]=[];
cfgMap[k].push(t);
}});
let cfgHtml = '<table class="analytics-table"><tr><th>Version</th><th>Trades</th><th>Win Rate</th><th>Net P&L</th><th>PF</th></tr>';
Object.entries(cfgMap).sort((a,b)=>parseInt(a[0].slice(1))-parseInt(b[0].slice(1))).forEach(([k,v])=>{{
const m=computeMetrics(v);
cfgHtml += `<tr><td style="font-weight:600">${{k}}</td><td>${{m.trades}}</td><td style="color:${{m.winRate>=0.5?'var(--green)':'var(--red)'}}">${{fmtPct(m.winRate)}}</td><td>${{pnlTd(m.totalPnl)}}</td><td>${{fmt(m.profitFactor)}}</td></tr>`;
}});
cfgHtml += '</table>';

// By Session (time)
const sessions = {{Asian:[],London:[],NewYork:[],Other:[]}};
TRADES.forEach(t=>{{
const h=parseInt(t.time.slice(11,13));
if(h>=0&&h<8) sessions.Asian.push(t);
else if(h>=8&&h<14) sessions.London.push(t);
else if(h>=14&&h<21) sessions.NewYork.push(t);
else sessions.Other.push(t);
}});
let sessionHtml = '<table class="analytics-table"><tr><th>Session</th><th>Trades</th><th>Win Rate</th><th>Net P&L</th><th>PF</th></tr>';
Object.entries(sessions).forEach(([k,v])=>{{
if(!v.length) return;
const m=computeMetrics(v);
sessionHtml += `<tr><td style="font-weight:600">${{k}}</td><td>${{m.trades}}</td><td style="color:${{m.winRate>=0.5?'var(--green)':'var(--red)'}}">${{fmtPct(m.winRate)}}</td><td>${{pnlTd(m.totalPnl)}}</td><td>${{fmt(m.profitFactor)}}</td></tr>`;
}});
sessionHtml += '</table>';

// Risk Analysis
const wins = TRADES.filter(t=>t.pnl>0);
const losses = TRADES.filter(t=>t.pnl<0);
const largestWin = wins.length?Math.max(...wins.map(t=>t.pnl)):0;
const largestLoss = losses.length?Math.min(...losses.map(t=>t.pnl)):0;
const riskHtml = `
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
<div><div class="kpi-label">Max Consecutive Wins</div><div class="kpi-value positive" style="font-size:18px">${{allMetrics.maxConsecWin}}</div></div>
<div><div class="kpi-label">Max Consecutive Losses</div><div class="kpi-value negative" style="font-size:18px">${{allMetrics.maxConsecLoss}}</div></div>
<div><div class="kpi-label">Largest Winner</div><div class="kpi-value positive" style="font-size:18px">${{fmtMoney(largestWin)}}</div></div>
<div><div class="kpi-label">Largest Loser</div><div class="kpi-value negative" style="font-size:18px">${{fmtMoney(largestLoss)}}</div></div>
<div><div class="kpi-label">Recovery Factor</div><div class="kpi-value neutral" style="font-size:18px">${{allMetrics.maxDrawdown?fmt(allMetrics.totalPnl/allMetrics.maxDrawdown):'—'}}</div></div>
<div><div class="kpi-label">Expectancy</div><div class="kpi-value ${{pnlClass(allMetrics.expectancy)}}" style="font-size:18px">${{fmtMoney(allMetrics.expectancy)}}</div></div>
</div>`;

el.innerHTML = `
<div class="card"><div class="card-header"><span class="card-title">📊 By Strategy</span></div>${{stratHtml}}</div>
<div class="card"><div class="card-header"><span class="card-title">🏦 By Instrument</span></div>${{instHtml}}</div>
<div class="card"><div class="card-header"><span class="card-title">📡 By Signal</span></div>${{sigHtml}}</div>
<div class="card"><div class="card-header"><span class="card-title">🔄 By Config Version</span></div>${{cfgHtml}}</div>
<div class="card"><div class="card-header"><span class="card-title">🕐 By Session</span></div>${{sessionHtml}}</div>
<div class="card"><div class="card-header"><span class="card-title">⚠️ Risk Analysis</span></div>${{riskHtml}}</div>
`;

// P&L Distribution histogram
const distCard = document.createElement('div');
distCard.className = 'card';
distCard.style.gridColumn = '1/-1';
distCard.innerHTML = '<div class="card-header"><span class="card-title">📊 P&L Distribution</span></div><canvas id="chart-dist" style="height:250px"></canvas>';
el.appendChild(distCard);

setTimeout(()=>{{
const pnls = TRADES.map(t=>t.pnl);
const min = Math.floor(Math.min(...pnls)*10)/10;
const max = Math.ceil(Math.max(...pnls)*10)/10;
const step = Math.max(0.1, (max-min)/30);
const bins = [];
for(let b=min;b<max;b+=step){{
const count = pnls.filter(p=>p>=b&&p<b+step).length;
bins.push({{x:b,count}});
}}
new Chart(document.getElementById('chart-dist'),{{
type:'bar',
data:{{labels:bins.map(b=>fmt(b.x,2)),datasets:[{{data:bins.map(b=>b.count),backgroundColor:bins.map(b=>b.x>=0?'rgba(63,185,80,0.6)':'rgba(248,81,73,0.6)'),borderRadius:2}}]}},
options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{title:{{display:true,text:'P&L'}}}},y:{{title:{{display:true,text:'Frequency'}},grid:{{color:'#1c2333'}}}}}}}},
}});
}},200);
}})();
</script>
</body>
</html>'''

with open('/tmp/vwap-dash/index.html', 'w') as f:
    f.write(html)

print(f"Dashboard built: {len(html):,} bytes, {len(trades)} trades embedded")
