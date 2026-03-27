import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 設定觀察名單與參數
# ==============================================================================
WATCHLIST = [
    '0700.HK', '09988.HK', '3690.HK', '1211.HK', '2318.HK', '0388.HK', '0005.HK', '1299.HK', 
    '0941.HK', '0883.HK', '0857.HK', '0386.HK', '0001.HK', '0016.HK', '0066.HK', '0823.HK',
    '1928.HK', '2020.HK', '1093.HK', '1177.HK', '2269.HK', '0291.HK', '2388.HK', '0011.HK',
    '0939.HK', '1398.HK', '3988.HK', '2628.HK', '0175.HK'
]

PORTFOLIO_FILE = 'portfolio.json'
INITIAL_CAPITAL = 100000.0
TRADE_SIZE = 10000.0 # 每筆交易固定投入金額
today_str = datetime.datetime.now().strftime('%Y-%m-%d')
today_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

print("⏳ 1/5 正在下載最新市場數據...")
hsi_df = yf.download("2800.HK", period="1y", progress=False, threads=False)
if hsi_df.empty:
    hsi_df = yf.download("^HSI", period="1y", progress=False, threads=False)

hsi_c = hsi_df['Close'].iloc[:, 0].ffill() if isinstance(hsi_df.columns, pd.MultiIndex) else hsi_df['Close'].ffill()

data = yf.download(WATCHLIST, period="1y", progress=False, threads=True)
if isinstance(data.columns, pd.MultiIndex):
    closes = data.xs('Close', level=1, axis=1).ffill() if 'Close' not in data.columns.get_level_values(0) else data['Close'].ffill()
    highs = data.xs('High', level=1, axis=1).ffill() if 'High' not in data.columns.get_level_values(0) else data['High'].ffill()
    lows = data.xs('Low', level=1, axis=1).ffill() if 'Low' not in data.columns.get_level_values(0) else data['Low'].ffill()
    vols = data.xs('Volume', level=1, axis=1).ffill() if 'Volume' not in data.columns.get_level_values(0) else data['Volume'].ffill()
else:
    closes, highs, lows, vols = data[['Close']].ffill(), data[['High']].ffill(), data[['Low']].ffill(), data[['Volume']].ffill()

print("⏳ 2/5 計算技術指標與大盤狀態...")
hsi_200ma = hsi_c.rolling(200).mean()
current_hsi_price, current_hsi_200ma = hsi_c.iloc[-1], hsi_200ma.iloc[-1]
is_bull_market = current_hsi_price > current_hsi_200ma

market_status = "🟢 牛市狀態 (基準 > 200MA)" if is_bull_market else "🔴 熊市/震盪狀態 (基準 < 200MA)"
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
# 3. 虛擬實盤帳戶管理 (Paper Trading Portfolio Manager)
# ==============================================================================
print("⏳ 3/5 載入並更新虛擬實盤帳戶...")
if os.path.exists(PORTFOLIO_FILE):
    with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
        portfolio = json.load(f)
else:
    portfolio = {"cash": INITIAL_CAPITAL, "open_positions": [], "closed_trades": [], "equity_curve": []}

# 處理出場邏輯 (Check Exits)
remaining_positions = []
for pos in portfolio['open_positions']:
    ticker = pos['ticker']
    if ticker not in closes.columns:
        remaining_positions.append(pos)
        continue
    
    cur_c, cur_h, cur_l = closes[ticker].iloc[-1], highs[ticker].iloc[-1], lows[ticker].iloc[-1]
    days_held = (datetime.datetime.strptime(today_str, "%Y-%m-%d") - datetime.datetime.strptime(pos['entry_date'], "%Y-%m-%d")).days
    
    exit_price = None
    exit_reason = ""
    
    # 判斷是否觸及止損或止盈
    if cur_l <= pos['sl']:
        exit_price = pos['sl']
        exit_reason = "🔴 觸發止損"
    elif cur_h >= pos['tp']:
        exit_price = pos['tp']
        exit_reason = "🟢 觸發止盈"
    elif days_held >= 30: # 30天強制平倉
        exit_price = cur_c
        exit_reason = "⏱️ 30日強制平倉"
        
    if exit_price:
        pnl = (exit_price - pos['entry_price']) * pos['shares']
        portfolio['cash'] += (exit_price * pos['shares'])
        portfolio['closed_trades'].append({
            "ticker": ticker, "entry_date": pos['entry_date'], "exit_date": today_str,
            "entry_price": pos['entry_price'], "exit_price": round(exit_price, 2),
            "pnl": round(pnl, 2), "pnl_pct": round((exit_price/pos['entry_price']-1)*100, 2), "reason": exit_reason
        })
        print(f"   💰 平倉: {ticker} ({exit_reason}), 獲利: ${round(pnl, 2)}")
    else:
        remaining_positions.append(pos)

portfolio['open_positions'] = remaining_positions

print("⏳ 4/5 掃描新訊號與自動建倉...")
signals = []
open_tickers = [p['ticker'] for p in portfolio['open_positions']]

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
        signals.append({"ticker": ticker, "price": round(cur_c, 2), "type": trigger_type, "sl": round(sl_price, 2), "tp": round(tp_price, 2), "rsi": round(recent_rsi.iloc[-1], 1)})
        
        # 自動建倉邏輯 (如果不在庫存中，且現金足夠)
        if ticker not in open_tickers and portfolio['cash'] >= TRADE_SIZE:
            shares = int(TRADE_SIZE / cur_c)
            actual_cost = shares * cur_c
            portfolio['cash'] -= actual_cost
            portfolio['open_positions'].append({
                "ticker": ticker, "entry_date": today_str, "entry_price": round(cur_c, 2),
                "shares": shares, "sl": round(sl_price, 2), "tp": round(tp_price, 2), "type": trigger_type
            })
            print(f"   🛒 買入: {ticker} @ ${round(cur_c,2)}, 數量: {shares}")

# 計算最新總資產 (Total Equity)
total_equity = portfolio['cash']
for pos in portfolio['open_positions']:
    if pos['ticker'] in closes.columns:
        total_equity += (closes[pos['ticker']].iloc[-1] * pos['shares'])

# 更新資產曲線
if len(portfolio['equity_curve']) > 0 and portfolio['equity_curve'][-1]['date'] == today_str:
    portfolio['equity_curve'][-1]['equity'] = round(total_equity, 2)
else:
    portfolio['equity_curve'].append({"date": today_str, "equity": round(total_equity, 2)})

# 存檔 Portfolio
with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
    json.dump(portfolio, f, indent=4)

print(f"✅ 掃描與結算完成！今日總資產: ${round(total_equity, 2)}")

# ==============================================================================
# 5. 生成互動式 HTML Dashboard (V5.0 實盤追蹤版)
# ==============================================================================
print("⏳ 5/5 正在生成 Dashboard HTML...")

# 將資料轉給前端 Chart.js 使用
eq_dates = [e['date'][5:] for e in portfolio['equity_curve']] # 只取 MM-DD
eq_values = [e['equity'] for e in portfolio['equity_curve']]

html_content = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <title>HK Quant Master V5 - 雲端實盤公開基金</title>
    <style>
        body {{ background-color: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        .card {{ background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; overflow: hidden; }}
        .bull-text {{ color: #10b981; }} .bear-text {{ color: #ef4444; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background-color: #0f172a; color: #94a3b8; font-size: 12px; text-transform: uppercase; }}
        tr:hover {{ background-color: #334155; }}
    </style>
</head>
<body class="p-4 md:p-8">
    <div class="max-w-7xl mx-auto space-y-6">
        <div class="card p-6 flex flex-col md:flex-row justify-between items-center shadow-lg border-b-4 border-indigo-500">
            <div>
                <h1 class="text-3xl font-black text-white mb-2">HK Quant Master V5 <span class="text-indigo-400">雲端公開基金</span></h1>
                <p class="text-slate-400 text-sm">全自動訊號掃描 ✕ 虛擬實盤資金追蹤 | 最後更新：{today_time_str}</p>
            </div>
            <div class="mt-4 md:mt-0 text-right bg-slate-900 p-4 rounded-lg border border-slate-700">
                <p class="text-xs text-slate-400 mb-1">大盤狀態 ({hsi_c.iloc[-1]:.2f} vs 200MA {hsi_200ma.iloc[-1]:.2f})</p>
                <div class="text-xl font-bold {'bull-text' if is_bull_market else 'bear-text'}">{market_status}</div>
                <div class="text-sm mt-1 text-slate-300">啟動引擎：<span class="font-bold text-yellow-400">{active_strategy}</span></div>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div class="lg:col-span-1 flex flex-col gap-4">
                <h2 class="text-xl font-bold border-b border-slate-700 pb-2">🎯 今日觸發訊號 ({len(signals)})</h2>
                <div class="overflow-y-auto max-h-[600px] space-y-4 pr-2">
"""
if not signals:
    html_content += `<div class="card p-6 text-center text-slate-500"><p class="text-4xl mb-2">☕</p><p>今日無訊號。</p></div>`
else:
    for sig in signals:
        badge = "text-green-400 border-green-700 bg-green-900/30" if "海龜" in sig['type'] else "text-red-400 border-red-700 bg-red-900/30"
        html_content += f"""
                    <div class="card p-4 shadow">
                        <div class="flex justify-between items-center mb-2">
                            <div class="text-2xl font-black text-white">{sig['ticker']}</div>
                            <div class="text-xs px-2 py-1 rounded border {badge}">{sig['type']}</div>
                        </div>
                        <div class="grid grid-cols-3 gap-2 text-center bg-slate-900 p-2 rounded border border-slate-700 mt-3">
                            <div><div class="text-[10px] text-slate-500">現價</div><div class="font-bold text-white">${sig['price']}</div></div>
                            <div><div class="text-[10px] text-red-400">止損</div><div class="font-bold text-red-400">${sig['sl']}</div></div>
                            <div><div class="text-[10px] text-green-400">止盈</div><div class="font-bold text-green-400">${sig['tp']}</div></div>
                        </div>
                    </div>
        """
html_content += f"""
                </div>
            </div>

            <div class="lg:col-span-2 space-y-6">
                <div class="card p-6 bg-gradient-to-br from-slate-800 to-slate-900">
                    <h2 class="text-xl font-bold border-b border-slate-700 pb-2 mb-4">📈 系統自動實盤績效 (Paper Trading)</h2>
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                        <div class="bg-slate-900 p-4 rounded border border-slate-700 text-center">
                            <p class="text-xs text-slate-400">總資產 (Total Equity)</p>
                            <p class="text-2xl font-black text-white">${total_equity:,.2f}</p>
                        </div>
                        <div class="bg-slate-900 p-4 rounded border border-slate-700 text-center">
                            <p class="text-xs text-slate-400">可用現金 (Cash)</p>
                            <p class="text-xl font-bold text-blue-400">${portfolio['cash']:,.2f}</p>
                        </div>
                        <div class="bg-slate-900 p-4 rounded border border-slate-700 text-center">
                            <p class="text-xs text-slate-400">總回報率 (Return)</p>
                            <p class="text-xl font-bold {'text-green-400' if total_equity >= INITIAL_CAPITAL else 'text-red-400'}">{((total_equity/INITIAL_CAPITAL)-1)*100:+.2f}%</p>
                        </div>
                        <div class="bg-slate-900 p-4 rounded border border-slate-700 text-center">
                            <p class="text-xs text-slate-400">持倉檔數</p>
                            <p class="text-xl font-bold text-yellow-400">{len(portfolio['open_positions'])} 檔</p>
                        </div>
                    </div>
                    
                    <div class="h-[250px] w-full">
                        <canvas id="equityChart"></canvas>
                    </div>
                </div>

                <div class="grid grid-cols-1 gap-6">
                    <div class="card">
                        <div class="bg-slate-800 p-3 border-b border-slate-700 font-bold">💼 當前持倉 (Open Positions)</div>
                        <div class="overflow-x-auto max-h-[250px] overflow-y-auto">
                            <table>
                                <tr><th>代碼</th><th>策略</th><th>買入日</th><th>買入價</th><th>現價</th><th>損益(%)</th></tr>
"""
for pos in reversed(portfolio['open_positions']):
    cur_px = closes[pos['ticker']].iloc[-1] if pos['ticker'] in closes.columns else pos['entry_price']
    pnl_pct = (cur_px / pos['entry_price'] - 1) * 100
    color = "text-green-400" if pnl_pct > 0 else "text-red-400"
    html_content += f"<tr><td class='font-bold'>{pos['ticker']}</td><td>{pos['type']}</td><td>{pos['entry_date']}</td><td>${pos['entry_price']}</td><td>${cur_px:.2f}</td><td class='font-bold {color}'>{pnl_pct:+.2f}%</td></tr>"

if not portfolio['open_positions']: html_content += "<tr><td colspan='6' class='text-center text-slate-500'>目前無持倉空手中</td></tr>"

html_content += f"""
                            </table>
                        </div>
                    </div>

                    <div class="card">
                        <div class="bg-slate-800 p-3 border-b border-slate-700 font-bold">📜 最近平倉紀錄 (Closed Trades)</div>
                        <div class="overflow-x-auto max-h-[250px] overflow-y-auto">
                            <table>
                                <tr><th>代碼</th><th>買入日</th><th>賣出日</th><th>買入價</th><th>賣出價</th><th>損益($)</th><th>原因</th></tr>
"""
for pos in reversed(portfolio['closed_trades'][-20:]): # 顯示最近 20 筆
    color = "text-green-400" if pos['pnl'] > 0 else "text-red-400"
    html_content += f"<tr><td class='font-bold'>{pos['ticker']}</td><td>{pos['entry_date']}</td><td>{pos['exit_date']}</td><td>${pos['entry_price']}</td><td>${pos['exit_price']}</td><td class='font-bold {color}'>${pos['pnl']}</td><td class='text-xs'>{pos['reason']}</td></tr>"

if not portfolio['closed_trades']: html_content += "<tr><td colspan='7' class='text-center text-slate-500'>尚無平倉紀錄</td></tr>"

html_content += f"""
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('equityChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(eq_dates)},
                datasets: [{{
                    label: '總資產 (HKD)',
                    data: {json.dumps(eq_values)},
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.2)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.3,
                    pointRadius: { '3' if len(eq_values) < 10 else '0' },
                    pointHoverRadius: 6
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ grid: {{ color: '#334155', drawBorder: false }}, ticks: {{ color: '#94a3b8', maxTicksLimit: 10 }} }},
                    y: {{ grid: {{ color: '#334155', drawBorder: false }}, ticks: {{ color: '#94a3b8' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

with open("index.html", 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"🎉 成功！已生成 V5 雲端公開基金網頁 (index.html)")
