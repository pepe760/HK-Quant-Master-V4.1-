import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import json
import os
import requests
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 設定觀察名單與參數
# ==============================================================================
# 核心港股觀察清單 (全範圍雷達)
WATCHLIST = [
    '0001.HK', '0002.HK', '0003.HK', '0005.HK', '0006.HK', '0011.HK', '0012.HK', '0016.HK', '0017.HK', '0020.HK',
    '0027.HK', '0066.HK', '0083.HK', '0101.HK', '0119.HK', '0135.HK', '0144.HK', '0151.HK', '0168.HK', '0175.HK',
    '0200.HK', '0241.HK', '0256.HK', '0267.HK', '0268.HK', '0270.HK', '0272.HK', '0285.HK', '0288.HK', '0291.HK',
    '0316.HK', '0322.HK', '0336.HK', '0345.HK', '0354.HK', '0358.HK', '0386.HK', '0388.HK', '0390.HK', '0460.HK',
    '0520.HK', '0522.HK', '0552.HK', '0576.HK', '0586.HK', '0598.HK', '0604.HK', '0656.HK', '0669.HK', '0688.HK',
    '0700.HK', '0728.HK', '0753.HK', '0762.HK', '0772.HK', '0778.HK', '0780.HK', '0813.HK', '0823.HK', '0836.HK',
    '0853.HK', '0857.HK', '0861.HK', '0868.HK', '0883.HK', '0902.HK', '0909.HK', '0914.HK', '0916.HK', '0934.HK',
    '0939.HK', '0941.HK', '0960.HK', '0968.HK', '0981.HK', '0992.HK', '0998.HK', '1024.HK', '1030.HK', '1038.HK',
    '1044.HK', '1055.HK', '1066.HK', '1071.HK', '1088.HK', '1093.HK', '1099.HK', '1109.HK', '1113.HK', '1119.HK',
    '1138.HK', '1157.HK', '1177.HK', '1193.HK', '1209.HK', '1211.HK', '1258.HK', '1299.HK', '1308.HK', '1313.HK',
    '1316.HK', '1336.HK', '1339.HK', '1347.HK', '1368.HK', '1378.HK', '1398.HK', '1516.HK', '1530.HK', '1658.HK',
    '1772.HK', '1787.HK', '1801.HK', '1810.HK', '1818.HK', '1833.HK', '1876.HK', '1898.HK', '1919.HK', '1928.HK',
    '1929.HK', '1997.HK', '2005.HK', '2007.HK', '2013.HK', '2015.HK', '2018.HK', '2020.HK', '2186.HK', '2192.HK',
    '2202.HK', '2238.HK', '2269.HK', '2313.HK', '2318.HK', '2319.HK', '2331.HK', '2333.HK', '2359.HK', '2380.HK',
    '2388.HK', '2600.HK', '2618.HK', '2628.HK', '2669.HK', '2688.HK', '2689.HK', '2727.HK', '2858.HK', '2866.HK',
    '2869.HK', '2877.HK', '2883.HK', '2899.HK', '3311.HK', '3319.HK', '3323.HK', '3328.HK', '3331.HK', '3606.HK',
    '3618.HK', '3633.HK', '3690.HK', '3692.HK', '3738.HK', '3800.HK', '3868.HK', '3888.HK', '3899.HK', '3900.HK',
    '3908.HK', '3933.HK', '3958.HK', '3968.HK', '3983.HK', '3988.HK', '3990.HK', '3993.HK', '6030.HK', '6098.HK',
    '6110.HK', '6160.HK', '6618.HK', '6690.HK', '6806.HK', '6837.HK', '6862.HK', '6865.HK', '6881.HK', '6969.HK',
    '9618.HK', '9633.HK', '9866.HK', '9868.HK', '9888.HK', '9922.HK', '9959.HK', '9988.HK', '9992.HK', '9999.HK'
]

PORTFOLIO_FILE = 'portfolio.json'
INITIAL_CAPITAL = 100000.0
TRADE_SIZE = 10000.0 
today_str = datetime.datetime.now().strftime('%Y-%m-%d')
today_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

print("⏳ 1/5 正在下載最新市場數據 (包含歷史配息)...")
hsi_df = yf.download("2800.HK", period="1y", progress=False, threads=False)
if hsi_df.empty:
    hsi_df = yf.download("^HSI", period="1y", progress=False, threads=False)

hsi_c = hsi_df['Close'].iloc[:, 0].ffill() if isinstance(hsi_df.columns, pd.MultiIndex) else hsi_df['Close'].ffill()

data = yf.download(WATCHLIST, period="1y", progress=False, threads=True, actions=True)
if isinstance(data.columns, pd.MultiIndex):
    closes = data.xs('Close', level=1, axis=1).ffill() if 'Close' not in data.columns.get_level_values(0) else data['Close'].ffill()
    highs = data.xs('High', level=1, axis=1).ffill() if 'High' not in data.columns.get_level_values(0) else data['High'].ffill()
    lows = data.xs('Low', level=1, axis=1).ffill() if 'Low' not in data.columns.get_level_values(0) else data['Low'].ffill()
    vols = data.xs('Volume', level=1, axis=1).ffill() if 'Volume' not in data.columns.get_level_values(0) else data['Volume'].ffill()
    
    if 'Dividends' in data.columns.get_level_values(0):
        divs = data['Dividends'].fillna(0)
    elif 'Dividends' in data.columns.get_level_values(1):
        divs = data.xs('Dividends', level=1, axis=1).fillna(0)
    else:
        divs = pd.DataFrame(columns=WATCHLIST)
else:
    closes, highs, lows, vols = data[['Close']].ffill(), data[['High']].ffill(), data[['Low']].ffill(), data[['Volume']].ffill()
    divs = data[['Dividends']].fillna(0) if 'Dividends' in data.columns else pd.DataFrame(columns=WATCHLIST)

print("⏳ 2/5 計算技術指標與大盤狀態...")
hsi_200ma = hsi_c.rolling(200).mean()
current_hsi_price, current_hsi_200ma = hsi_c.iloc[-1], hsi_200ma.iloc[-1]
is_bull_market = current_hsi_price > current_hsi_200ma

market_status = "🟢 牛市狀態 (大盤 > 200MA)" if is_bull_market else "🔴 熊市/震盪狀態 (大盤 < 200MA)"
active_strategy = "Donchian Turtle (海龜20日突破)" if is_bull_market else "Mean Reversion (RSI超賣抄底)"

delta = closes.diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss.replace(0, np.nan)
rsi = 100 - (100 / (1 + rs))

sma20 = closes.rolling(20).mean()
std20 = closes.rolling(20).std()
lower_bb = sma20 - (2 * std20)
donchian_high = highs.rolling(20).max().shift(1)
avg_vol_20 = vols.rolling(20).mean()

# ==============================================================================
# 3. 虛擬實盤帳戶管理
# ==============================================================================
print("⏳ 3/5 載入並更新虛擬實盤帳戶...")
if os.path.exists(PORTFOLIO_FILE):
    with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
        portfolio = json.load(f)
else:
    portfolio = {"cash": INITIAL_CAPITAL, "open_positions": [], "closed_trades": [], "equity_curve": []}

remaining_positions = []
for pos in portfolio['open_positions']:
    ticker = pos['ticker']
    if ticker not in closes.columns:
        remaining_positions.append(pos)
        continue
    
    cur_c, cur_h, cur_l = closes[ticker].iloc[-1], highs[ticker].iloc[-1], lows[ticker].iloc[-1]
    days_held = (datetime.datetime.strptime(today_str, "%Y-%m-%d") - datetime.datetime.strptime(pos['entry_date'], "%Y-%m-%d")).days
    
    exit_price, exit_reason = None, ""
    
    if cur_l <= pos['sl']:
        exit_price, exit_reason = pos['sl'], "🔴 觸發止損"
    elif cur_h >= pos['tp']:
        exit_price, exit_reason = pos['tp'], "🟢 觸發止盈"
    elif days_held >= 30:
        exit_price, exit_reason = cur_c, "⏱️ 30日強制平倉"
        
    if exit_price:
        pnl = (exit_price - pos['entry_price']) * pos['shares']
        portfolio['cash'] += (exit_price * pos['shares'])
        portfolio['closed_trades'].append({
            "ticker": ticker, "entry_date": pos['entry_date'], "exit_date": today_str,
            "entry_price": pos['entry_price'], "exit_price": round(exit_price, 2),
            "pnl": round(pnl, 2), "reason": exit_reason
        })
    else:
        remaining_positions.append(pos)

portfolio['open_positions'] = remaining_positions

print("⏳ 4/5 掃描訊號、抓取基本面護城河...")
signals = []
open_tickers = [p['ticker'] for p in portfolio['open_positions']]

def safe_list(series):
    return [None if pd.isna(x) else round(float(x), 2) for x in series.tolist()]

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'})

for ticker in closes.columns:
    if ticker not in WATCHLIST: continue
    cur_c = closes[ticker].iloc[-1]
    if pd.isna(cur_c): continue

    recent_c, recent_rsi, recent_lbb, recent_dh = closes[ticker].tail(5), rsi[ticker].tail(5), lower_bb[ticker].tail(5), donchian_high[ticker].tail(5)
    trigger_type, trigger_days, sl_price, tp_price = None, 0, 0.0, 0.0
    
    if is_bull_market:
        is_triggered = recent_c > recent_dh
        if is_triggered.iloc[-1]:
            trigger_type, sl_price, tp_price = "海龜突破", cur_c * 0.90, cur_c * 1.30
            for val in reversed(is_triggered.values):
                if val: trigger_days += 1
                else: break
    else:
        is_triggered = (recent_rsi < 30) & (recent_c < recent_lbb)
        if is_triggered.iloc[-1]:
            trigger_type, sl_price, tp_price = "RSI抄底", cur_c * 0.88, cur_c * 1.20
            for val in reversed(is_triggered.values):
                if val: trigger_days += 1
                else: break

    if trigger_type:
        cur_h52 = highs[ticker].max() 
        drawdown_52w = ((cur_c - cur_h52) / cur_h52) * 100 if cur_h52 > 0 else 0
        cur_vol = avg_vol_20[ticker].iloc[-1]
        daily_turnover_est = (cur_vol * cur_c) / 1000000 
        
        div_yield_pct = 0.0
        if ticker in divs.columns:
            annual_div = divs[ticker].sum()
            if cur_c > 0:
                div_yield_pct = round((annual_div / cur_c) * 100, 2)
                
        earn_label = "無資料"
        try:
            tk_info = yf.Ticker(ticker, session=session).info
            earn_growth = tk_info.get('earningsGrowth') or tk_info.get('revenueGrowth') or 0
            earn_growth_pct = round(earn_growth * 100, 2)
            if earn_growth_pct >= 15: earn_label = f"強勁成長 (+{earn_growth_pct}%)"
            elif earn_growth_pct > 0: earn_label = f"溫和復甦 (+{earn_growth_pct}%)"
            elif earn_growth_pct < 0: earn_label = f"衰退中 ({earn_growth_pct}%)"
            else: earn_label = "未公佈"
        except:
            pass
            
        # 轉換 Ticker 給 TradingView 用 (例如 '0700.HK' -> 'HKEX:700')
        tv_ticker = f"HKEX:{int(ticker.split('.')[0])}" if ticker.split('.')[0].isdigit() else ticker

        signals.append({
            "ticker": ticker, "tv_ticker": tv_ticker, "price": round(cur_c, 2), 
            "type": trigger_type, "sl": round(sl_price, 2), "tp": round(tp_price, 2), 
            "rsi": round(recent_rsi.iloc[-1], 1), "trigger_days": trigger_days,
            "dd_52w": round(drawdown_52w, 1), "turnover_m": round(daily_turnover_est, 1),
            "div_yield": div_yield_pct, "earn_label": earn_label,
            "chart_dates": closes.index[-100:].strftime('%m-%d').tolist(),
            "chart_prices": safe_list(closes[ticker].tail(100)),
            "chart_sma20": safe_list(sma20[ticker].tail(100)),
            "chart_lbb": safe_list(lower_bb[ticker].tail(100))
        })
        
        if ticker not in open_tickers and portfolio['cash'] >= TRADE_SIZE:
            shares = int(TRADE_SIZE / cur_c)
            portfolio['cash'] -= shares * cur_c
            portfolio['open_positions'].append({
                "ticker": ticker, "entry_date": today_str, "entry_price": round(cur_c, 2),
                "shares": shares, "sl": round(sl_price, 2), "tp": round(tp_price, 2), "type": trigger_type,
                "div_yield": div_yield_pct, "earn_label": earn_label
            })

total_equity = portfolio['cash']
for pos in portfolio['open_positions']:
    if pos['ticker'] in closes.columns:
        total_equity += (closes[pos['ticker']].iloc[-1] * pos['shares'])

if len(portfolio['equity_curve']) > 0 and portfolio['equity_curve'][-1]['date'] == today_str:
    portfolio['equity_curve'][-1]['equity'] = round(total_equity, 2)
else:
    portfolio['equity_curve'].append({"date": today_str, "equity": round(total_equity, 2)})

with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
    json.dump(portfolio, f, indent=4)

print("⏳ 5/5 正在生成 HTML Dashboard...")

eq_dates = [e['date'][5:] for e in portfolio['equity_curve']]
eq_values = [e['equity'] for e in portfolio['equity_curve']]

html_content = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <title>HK Quant Master V6.2 - 終極旗艦版</title>
    <style>
        body {{ background-color: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        .card {{ background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; overflow: hidden; }}
        .bull-text {{ color: #10b981; }} .bear-text {{ color: #ef4444; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #334155; font-size: 14px; }}
        th {{ background-color: #0f172a; color: #94a3b8; font-size: 12px; text-transform: uppercase; }}
        tr:hover {{ background-color: #334155; }}
        .tab-btn {{ padding: 10px 20px; font-weight: bold; color: #94a3b8; border-bottom: 2px solid transparent; cursor: pointer; transition: 0.3s; }}
        .tab-btn:hover {{ color: #e2e8f0; }}
        .tab-active {{ color: #3b82f6; border-bottom: 2px solid #3b82f6; }}
        .tab-content {{ display: none; }}
        .content-active {{ display: block; }}
        .info-badge {{ font-size: 0.75rem; padding: 4px 6px; border-radius: 6px; background: rgba(30, 41, 59, 0.8); border: 1px solid #475569; text-align: center; }}
    </style>
</head>
<body class="p-4 md:p-8">
    <div class="max-w-7xl mx-auto space-y-6">
        <div class="card p-6 flex flex-col md:flex-row justify-between items-center shadow-lg border-b-4 border-blue-500">
            <div>
                <h1 class="text-3xl font-black text-white mb-2">HK Quant Master V6.2 <span class="text-blue-500">終極旗艦版</span></h1>
                <p class="text-slate-400 text-sm">防阻擋數據核心 ✕ 技財雙重掃描 | 更新時間：{today_time_str}</p>
            </div>
            <div class="mt-4 md:mt-0 text-right bg-slate-900 p-4 rounded-lg border border-slate-700">
                <p class="text-xs text-slate-400 mb-1">大盤狀態 ({hsi_c.iloc[-1]:.2f} vs 200MA {hsi_200ma.iloc[-1]:.2f})</p>
                <div class="text-xl font-bold {'bull-text' if is_bull_market else 'bear-text'}">{market_status}</div>
                <div class="text-sm mt-1 text-slate-300">啟動引擎：<span class="font-bold text-yellow-400">{active_strategy}</span></div>
            </div>
        </div>

        <div class="flex space-x-2 border-b border-slate-700">
            <button class="tab-btn tab-active" onclick="switchTab('tab-scanner', this)">🎯 今日訊號與圖表</button>
            <button class="tab-btn" onclick="switchTab('tab-portfolio', this)">📈 實盤績效與庫存</button>
            <button class="tab-btn" onclick="switchTab('tab-manual', this)">📖 系統使用說明書</button>
        </div>

        <div id="tab-scanner" class="tab-content content-active">
            <div class="flex flex-col lg:flex-row gap-6">
                <div class="lg:w-2/5 flex flex-col gap-4 overflow-y-auto max-h-[750px] pr-2">
                    <h2 class="text-xl font-bold border-b border-slate-700 pb-2">🎯 今日觸發標的 ({len(signals)})</h2>
"""
if not signals:
    html_content += """<div class="card p-6 text-center text-slate-500"><p class="text-4xl mb-2">☕</p><p>今日無訊號。</p></div>"""
else:
    for sig in signals:
        badge = "bg-green-900/50 text-green-400 border-green-700" if "海龜" in sig['type'] else "bg-red-900/50 text-red-400 border-red-700"
        div_color = "text-green-400 font-black" if sig['div_yield'] >= 6 else "text-slate-300"
        earn_color = "text-green-400" if "+" in sig['earn_label'] else "text-red-400" if "衰退" in sig['earn_label'] else "text-slate-400"
        
        html_content += f"""
                    <div class="card p-4 cursor-pointer hover:bg-slate-700 transition shadow border-l-4 border-blue-500" onclick="loadStockChart('{sig['ticker']}')">
                        <div class="flex justify-between items-center mb-2">
                            <div class="text-2xl font-black text-white">{sig['ticker']}</div>
                            <div class="text-xs px-2 py-1 rounded border {badge}">{sig['type']}</div>
                        </div>
                        
                        <div class="grid grid-cols-2 gap-2 mb-3 bg-slate-800 p-2 rounded">
                            <div class="info-badge">
                                <span class="text-slate-400 block mb-1">股息殖利率</span>
                                <span class="{div_color}">{sig['div_yield']}%</span>
                            </div>
                            <div class="info-badge">
                                <span class="text-slate-400 block mb-1">近期業績</span>
                                <span class="{earn_color} font-bold">{sig['earn_label']}</span>
                            </div>
                            <div class="info-badge">
                                <span class="text-slate-400 block mb-1">52週回撤</span>
                                <span class="text-slate-300">{sig['dd_52w']}%</span>
                            </div>
                            <div class="info-badge">
                                <span class="text-slate-400 block mb-1">現價/RSI</span>
                                <span class="text-white">${sig['price']} (RSI:{sig['rsi']})</span>
                            </div>
                        </div>
                    </div>
"""
html_content += f"""
                </div>

                <div class="lg:w-3/5">
                    <div class="card p-4 mb-4 bg-slate-800 flex justify-between items-center">
                        <div>
                            <h3 id="stock_chart_title" class="text-xl font-bold text-white">點擊左側股票載入圖表</h3>
                            <p class="text-xs text-slate-400 mt-1">藍線: 收盤價 | 橘虛線: 20日均線 | 紅虛線: 布林下軌</p>
                        </div>
                        <a id="tv_link" href="#" target="_blank" class="hidden bg-blue-600 hover:bg-blue-500 text-white text-sm font-bold py-2 px-4 rounded transition shadow">
                            📊 在 TradingView 開啟
                        </a>
                    </div>
                    <div class="card p-4 h-[500px] flex items-center justify-center relative shadow-inner bg-slate-900">
                        <p id="chart_placeholder" class="text-slate-500 absolute z-0">個股圖表顯示區 (近100日走勢)</p>
                        <canvas id="stockChart" class="w-full h-full relative z-10 hidden"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <div id="tab-portfolio" class="tab-content space-y-6">
            <div class="card p-6 bg-gradient-to-br from-slate-800 to-slate-900">
                <h2 class="text-xl font-bold border-b border-slate-700 pb-2 mb-4">📈 系統自動實盤績效 (Paper Trading)</h2>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    <div class="bg-slate-900 p-4 rounded text-center border border-slate-700">
                        <p class="text-xs text-slate-400">總資產 (Total Equity)</p>
                        <p class="text-2xl font-black text-white">${total_equity:,.2f}</p>
                    </div>
                    <div class="bg-slate-900 p-4 rounded text-center border border-slate-700">
                        <p class="text-xs text-slate-400">可用現金 (Cash)</p>
                        <p class="text-xl font-bold text-blue-400">${portfolio['cash']:,.2f}</p>
                    </div>
                    <div class="bg-slate-900 p-4 rounded text-center border border-slate-700">
                        <p class="text-xs text-slate-400">總回報率 (Return)</p>
                        <p class="text-xl font-bold {'text-green-400' if total_equity >= INITIAL_CAPITAL else 'text-red-400'}">{((total_equity/INITIAL_CAPITAL)-1)*100:+.2f}%</p>
                    </div>
                    <div class="bg-slate-900 p-4 rounded text-center border border-slate-700">
                        <p class="text-xs text-slate-400">持倉檔數</p>
                        <p class="text-xl font-bold text-yellow-400">{len(portfolio['open_positions'])} 檔</p>
                    </div>
                </div>
                <div class="h-[250px] w-full"><canvas id="equityChart"></canvas></div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="card">
                    <div class="bg-slate-800 p-3 font-bold border-b border-slate-700">💼 當前持倉 (包含護城河指標)</div>
                    <div class="overflow-x-auto max-h-[350px] overflow-y-auto">
                        <table>
                            <tr><th>代碼</th><th>買入價</th><th>現價</th><th>損益(%)</th><th>殖利率</th><th>業績</th></tr>
"""
for pos in reversed(portfolio['open_positions']):
    cur_px = closes[pos['ticker']].iloc[-1] if pos['ticker'] in closes.columns else pos['entry_price']
    pnl_pct = (cur_px / pos['entry_price'] - 1) * 100
    color = "text-green-400" if pnl_pct > 0 else "text-red-400"
    dy = pos.get('div_yield', 0.0)
    el = pos.get('earn_label', '無資料')
    dy_color = "text-green-400 font-bold" if dy >= 6 else "text-slate-400"
    html_content += f"<tr><td class='font-bold text-blue-400'>{pos['ticker']}</td><td>${pos['entry_price']}</td><td>${cur_px:.2f}</td><td class='font-bold {color}'>{pnl_pct:+.2f}%</td><td class='{dy_color}'>{dy}%</td><td class='text-xs text-slate-400'>{el}</td></tr>"

if not portfolio['open_positions']: html_content += "<tr><td colspan='6' class='text-center text-slate-500'>目前無持倉空手中</td></tr>"

html_content += f"""
                        </table>
                    </div>
                </div>
                <div class="card">
                    <div class="bg-slate-800 p-3 font-bold border-b border-slate-700">📜 歷史平倉 (Closed Trades)</div>
                    <div class="overflow-x-auto max-h-[350px] overflow-y-auto">
                        <table>
                            <tr><th>代碼</th><th>損益($)</th><th>原因</th></tr>
"""
for pos in reversed(portfolio['closed_trades'][-20:]):
    color = "text-green-400" if pos['pnl'] > 0 else "text-red-400"
    html_content += f"<tr><td class='font-bold'>{pos['ticker']}</td><td class='font-bold {color}'>${pos['pnl']}</td><td class='text-xs'>{pos['reason']}</td></tr>"
if not portfolio['closed_trades']: html_content += "<tr><td colspan='3' class='text-center text-slate-500'>尚無平倉紀錄</td></tr>"

html_content += f"""
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <div id="tab-manual" class="tab-content">
            <div class="card p-8 space-y-6 text-slate-300">
                <h2 class="text-2xl font-black text-white border-b border-slate-700 pb-2">📖 HK Quant Master 系統使用說明書</h2>
                
                <div>
                    <h3 class="text-lg font-bold text-blue-400 mb-2">1. 系統大腦：狀態切換邏輯 (Regime Switching)</h3>
                    <p class="mb-2">本系統的核心優勢在於「判斷大環境」。系統每天會掃描香港盈富基金 (2800.HK) 的價格與 200 日移動平均線 (200MA) 之間的關係：</p>
                    <ul class="list-disc pl-6 space-y-1 text-sm">
                        <li><b>🟢 牛市狀態 (大盤 > 200MA)</b>：系統啟動「海龜突破法」，專挑創下 20 日新高的強勢股，順勢追擊。</li>
                        <li><b>🔴 熊市狀態 (大盤 < 200MA)</b>：港股大部分時間處於此狀態。系統會關閉追高策略，啟動「RSI 超賣抄底法」，專門尋找被市場錯殺、跌穿布林通道下軌的股票進行低吸。</li>
                    </ul>
                </div>

                <div>
                    <h3 class="text-lg font-bold text-blue-400 mb-2">2. 面板資訊解讀 (如何看懂訊號)</h3>
                    <p class="mb-2">在「實盤庫存」或「今日訊號」分頁中，您會看到強大的<b>「基本面護城河」</b>標籤：</p>
                    <ul class="list-disc pl-6 space-y-1 text-sm">
                        <li><span class="text-green-400 font-bold">高股息防護 (綠色顯示)</span>：當殖利率 > 6% 時會以綠色高亮。在熊市抄底時，高股息能為您提供強大的「下跌緩衝墊」，就算被套牢也能靠配息回本。</li>
                        <li><span class="text-green-400 font-bold">業績動能確認</span>：如果一檔股票暴跌，但業績顯示「強勁成長 (+)」，這就是黃金坑。若顯示「衰退中 (-)」，代表下跌是有合理原因的，強烈建議反彈即走，不可長抱。</li>
                    </ul>
                </div>

                <div>
                    <h3 class="text-lg font-bold text-blue-400 mb-2">3. 虛擬實盤基金 (Paper Trading) 運作機制</h3>
                    <p class="mb-2">這是一個全自動運行的機器人公開基金：</p>
                    <ul class="list-disc pl-6 space-y-1 text-sm">
                        <li><b>初始資金</b>：預設為 $100,000 港幣。</li>
                        <li><b>自動建倉</b>：只要出現符合條件的新訊號，且帳戶可用現金大於 $10,000，系統就會自動動用約 $1 萬港幣買入該股票。</li>
                        <li><b>無情平倉</b>：系統每天會結算價格。如果股價跌穿「嚴格止損價(SL)」或突破「目標止盈價(TP)」，系統隔天就會自動賣出，記錄在歷史平倉區。</li>
                    </ul>
                </div>

                <div class="bg-blue-900/30 p-4 border border-blue-800 rounded">
                    <h3 class="text-md font-bold text-white mb-1">⏱️ 自動化更新時間</h3>
                    <p class="text-sm">本網站託管於 GitHub Pages，透過 GitHub Actions 每天在 <b>香港時間下午 4:30 (港股收盤後)</b> 自動執行。您不需要做任何事，每天傍晚打開網頁，就能看到最新的實盤戰況與明日的交易計畫！</p>
                </div>
            </div>
        </div>

    </div>

    <script>
        // 1. 分頁切換功能 (Tab Switching)
        function switchTab(tabId, btnElement) {{
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('content-active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('tab-active'));
            document.getElementById(tabId).classList.add('content-active');
            btnElement.classList.add('tab-active');
        }}

        // 2. 繪製總資產曲線 (Equity Chart)
        const ctxEq = document.getElementById('equityChart').getContext('2d');
        new Chart(ctxEq, {{
            type: 'line',
            data: {{
                labels: {json.dumps(eq_dates)},
                datasets: [{{
                    label: '總資產 (HKD)', data: {json.dumps(eq_values)},
                    borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.2)',
                    borderWidth: 3, fill: true, tension: 0.3, pointRadius: 2
                }}]
            }},
            options: {{
                responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ grid: {{ color: '#334155' }}, ticks: {{ color: '#94a3b8' }} }},
                    y: {{ grid: {{ color: '#334155' }}, ticks: {{ color: '#94a3b8' }} }}
                }}
            }}
        }});

        // 3. 繪製個股圖表 (Stock Chart)
        const signalsData = {json.dumps(signals)};
        let stockChart = null;

        function loadStockChart(ticker) {{
            const sig = signalsData.find(s => s.ticker === ticker);
            if (!sig) return;

            document.getElementById('chart_placeholder').classList.add('hidden');
            const canvas = document.getElementById('stockChart');
            canvas.classList.remove('hidden');
            document.getElementById('stock_chart_title').innerText = ticker + " (近100日技術走勢)";

            // 更新 TradingView 按鈕連結
            const tvLink = document.getElementById('tv_link');
            tvLink.href = `https://www.tradingview.com/chart/?symbol=${{sig.tv_ticker}}`;
            tvLink.classList.remove('hidden');

            const ctxStock = canvas.getContext('2d');
            if (stockChart) stockChart.destroy();

            stockChart = new Chart(ctxStock, {{
                type: 'line',
                data: {{
                    labels: sig.chart_dates,
                    datasets: [
                        {{ label: '收盤價', data: sig.chart_prices, borderColor: '#3b82f6', borderWidth: 2, pointRadius: 0, tension: 0.1 }},
                        {{ label: '20日均線', data: sig.chart_sma20, borderColor: '#f59e0b', borderWidth: 1.5, borderDash: [5, 5], pointRadius: 0 }},
                        {{ label: '布林下軌', data: sig.chart_lbb, borderColor: '#ef4444', borderWidth: 1.5, borderDash: [2, 2], pointRadius: 0 }}
                    ]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    interaction: {{ mode: 'index', intersect: false }},
                    plugins: {{ legend: {{ labels: {{ color: '#e2e8f0' }} }} }},
                    scales: {{
                        x: {{ ticks: {{ color: '#94a3b8', maxTicksLimit: 10 }}, grid: {{ color: '#334155' }} }},
                        y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }}
                    }}
                }}
            }});
        }}

        // 預設載入第一檔股票的圖表
        if (signalsData.length > 0) {{
            setTimeout(() => loadStockChart(signalsData[0].ticker), 100);
        }}
    </script>
</body>
</html>
"""

with open("index.html", 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"🎉 成功！已生成 V6.2 終極體驗優化版網頁 (index.html)")
