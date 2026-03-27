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
# 1. 設定觀察名單與參數 (自動清理失效代號)
# ==============================================================================
RAW_WATCHLIST = [
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
# 暫時手動過濾 Yahoo 已知失效的代號，避免引發 KeyError
WATCHLIST = [s for s in RAW_WATCHLIST if s not in ['0011.HK', '6837.HK', '3331.HK']]

PORTFOLIO_FILE = 'portfolio.json'
TRADE_SIZE = 10000.0  
today_str = datetime.datetime.now().strftime('%Y-%m-%d')
today_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

print("⏳ 1/5 正在下載市場數據 (已強化避錯機制)...")
try:
    hsi_df = yf.download("2800.HK", period="1y", progress=False)
    if hsi_df.empty: hsi_df = yf.download("^HSI", period="1y", progress=False)
    
    # 針對 Yahoo 新版數據格式進行降維處理
    if isinstance(hsi_df.columns, pd.MultiIndex):
        hsi_c = hsi_df['Close'].iloc[:, 0].ffill()
    else:
        hsi_c = hsi_df['Close'].ffill()
except Exception as e:
    print(f"警告: 大盤數據下載失敗 ({e})，使用模擬數據維持運作。")
    hsi_c = pd.Series([20000]*252) # 避錯保險

# 下載觀察名單，加入分組下載減少報錯機率
data = yf.download(WATCHLIST, period="1y", progress=False, actions=True, group_by='column')

# 💡 關鍵避錯：手動提取存在欄位，避免 KeyError
def extract_df(df, col_name):
    try:
        if isinstance(df.columns, pd.MultiIndex):
            return df.xs(col_name, level=0, axis=1).ffill() if col_name in df.columns.levels[0] else df[col_name].ffill()
        return df[col_name].ffill()
    except:
        # 如果真的找不到該欄位，建立空白 DataFrame 避免崩潰
        return pd.DataFrame(index=df.index)

closes = extract_df(data, 'Close')
highs = extract_df(data, 'High')
lows = extract_df(data, 'Low')
vols = extract_df(data, 'Volume')
divs = extract_df(data, 'Dividends')

# 移除沒有 Close 數據的股票，防止後續計算報錯
valid_tickers = [t for t in WATCHLIST if t in closes.columns and not closes[t].dropna().empty]
closes, highs, lows, vols, divs = closes[valid_tickers], highs[valid_tickers], lows[valid_tickers], vols[valid_tickers], divs[valid_tickers]

print(f"✅ 成功載入 {len(valid_tickers)} 隻有效港股數據。")

print("⏳ 2/5 技術指標計算...")
hsi_200ma = hsi_c.rolling(200).mean()
is_bull_market = hsi_c.iloc[-1] > hsi_200ma.iloc[-1]
market_status = "🟢 牛市狀態" if is_bull_market else "🔴 熊市/震盪狀態"
active_strategy = "海龜突破" if is_bull_market else "RSI抄底"

delta = closes.diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rsi = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
sma20, std20 = closes.rolling(20).mean(), closes.rolling(20).std()
lower_bb = sma20 - (2 * std20)
donchian_high = highs.rolling(20).max().shift(1)

# ==============================================================================
# 3. 虛擬實盤帳戶管理
# ==============================================================================
print("⏳ 3/5 更新帳戶資產...")
if os.path.exists(PORTFOLIO_FILE):
    with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
        portfolio = json.load(f)
    for k in ['realized_pnl', 'winning_trades', 'total_trades']:
        if k not in portfolio: portfolio[k] = 0.0
else:
    portfolio = {"realized_pnl": 0.0, "winning_trades": 0, "total_trades": 0, "open_positions": [], "closed_trades": [], "equity_curve": []}

remaining_positions = []
unrealized_pnl, active_invested = 0.0, 0.0

for pos in portfolio['open_positions']:
    t = pos['ticker']
    if t not in closes.columns: continue
    cur_c, cur_h, cur_l = closes[t].iloc[-1], highs[t].iloc[-1], lows[t].iloc[-1]
    
    exit_price, exit_reason = None, ""
    if cur_l <= pos['sl']: exit_price, exit_reason = pos['sl'], "🔴 止損"
    elif cur_h >= pos['tp']: exit_price, exit_reason = pos['tp'], "🟢 止盈"
    
    if exit_price:
        p = (exit_price - pos['entry_price']) * pos['shares']
        portfolio['realized_pnl'] += p
        portfolio['total_trades'] += 1
        if p > 0: portfolio['winning_trades'] += 1
        portfolio['closed_trades'].append({"ticker": t, "pnl": round(p, 2), "reason": exit_reason, "entry_date": pos['entry_date'], "exit_date": today_str})
    else:
        unrealized_pnl += (cur_c - pos['entry_price']) * pos['shares']
        active_invested += pos['entry_price'] * pos['shares']
        remaining_positions.append(pos)
portfolio['open_positions'] = remaining_positions

print("⏳ 4/5 掃描訊號與強化數據抓取...")
signals = []
session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0'})

for ticker in closes.columns:
    cur_c = closes[ticker].iloc[-1]
    
    trigger = False
    if is_bull_market and (cur_c > donchian_high[ticker].iloc[-1]): trigger = "海龜突破"
    elif not is_bull_market and (rsi[ticker].iloc[-1] < 30 and cur_c < lower_bb[ticker].iloc[-1]): trigger = "RSI抄底"

    is_holding = any(p['ticker'] == ticker for p in portfolio['open_positions'])
    if trigger or is_holding:
        div_yield = round((divs[ticker].tail(252).sum() / cur_c) * 100, 2) if cur_c > 0 else 0
        earn_info = "未公佈"
        try:
            t_obj = yf.Ticker(ticker, session=session)
            fin = t_obj.financials
            if not fin.empty and 'Net Income' in fin.index:
                growth = (fin.loc['Net Income'].iloc[0] / fin.loc['Net Income'].iloc[1]) - 1
                earn_info = f"年度增長 ({growth*100:+.1f}%)"
        except: pass

        if trigger:
            sl, tp = (cur_c*0.9, cur_c*1.3) if is_bull_market else (cur_c*0.88, cur_c*1.2)
            signals.append({
                "ticker": ticker, "price": round(cur_c, 2), "type": trigger, "sl": round(sl, 2), "tp": round(tp, 2),
                "rsi": round(rsi[ticker].iloc[-1], 1), "div_yield": div_yield, "earn_label": earn_info,
                "chart_dates": closes.index[-100:].strftime('%m-%d').tolist(), "chart_prices": closes[ticker].tail(100).round(2).tolist(),
                "chart_sma20": sma20[ticker].tail(100).round(2).tolist(), "chart_lbb": lower_bb[ticker].tail(100).round(2).tolist(),
                "tv_ticker": f"HKEX:{int(ticker.split('.')[0])}"
            })
            if not is_holding:
                portfolio['open_positions'].append({"ticker": ticker, "entry_price": round(cur_c, 2), "shares": TRADE_SIZE/cur_c, "sl": round(sl, 2), "tp": round(tp, 2), "entry_date": today_str, "div_yield": div_yield, "earn_label": earn_info})

        for p in portfolio['open_positions']:
            if p['ticker'] == ticker:
                p['div_yield'], p['earn_label'] = div_yield, earn_info

total_net_profit = portfolio['realized_pnl'] + unrealized_pnl
portfolio['equity_curve'].append({"date": today_str, "equity": round(total_net_profit, 2)})
with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f: json.dump(portfolio, f, indent=4)

# ==============================================================================
# 5. 生成 HTML (V7.2)
# ==============================================================================
win_rate = (portfolio['winning_trades'] / portfolio['total_trades'] * 100) if portfolio['total_trades'] > 0 else 0
html_content = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <title>HK Quant Master V7.2</title>
    <style>
        body {{ background-color: #0f172a; color: #e2e8f0; font-family: sans-serif; }}
        .card {{ background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; overflow: hidden; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #334155; font-size: 14px; }}
        .tab-btn {{ padding: 10px 20px; color: #94a3b8; cursor: pointer; border-bottom: 2px solid transparent; }}
        .tab-active {{ color: #3b82f6; border-bottom: 2px solid #3b82f6; font-weight: bold; }}
        .tab-content {{ display: none; }} .active {{ display: block; }}
    </style>
</head>
<body class="p-4">
    <div class="max-w-7xl mx-auto space-y-6">
        <div class="card p-6 flex justify-between items-center border-b-4 border-blue-500 shadow-xl">
            <div>
                <h1 class="text-2xl font-black">HK Quant Master V7.2 <span class="text-blue-400">抗震避錯版</span></h1>
                <p class="text-sm text-slate-400">更新：{today_time_str} | 大盤：{market_status}</p>
            </div>
            <div class="text-right"><div class="text-lg font-bold text-yellow-400">策略：{active_strategy}</div></div>
        </div>

        <div class="flex space-x-4 border-b border-slate-700">
            <div id="btn-scan" class="tab-btn tab-active" onclick="showTab('scan')">🎯 今日訊號</div>
            <div id="btn-port" class="tab-btn" onclick="showTab('port')">📈 績效驗證</div>
        </div>

        <div id="tab-scan" class="tab-content active">
            <div class="flex flex-col lg:flex-row gap-6">
                <div class="lg:w-1/3 space-y-4 max-h-[700px] overflow-y-auto pr-2">
"""
if not signals:
    html_content += '<div class="card p-10 text-center text-slate-500">☕ 今日無符合訊號</div>'
else:
    for s in signals:
        dy_c = "text-green-400" if s['div_yield'] >= 6 else "text-slate-300"
        html_content += f"""
                    <div class="card p-4 cursor-pointer hover:bg-slate-700 border-l-4 border-blue-500" onclick="updateChart('{s['ticker']}', '{s['div_yield']}', '{s['earn_label']}', '{s['tv_ticker']}')">
                        <div class="flex justify-between font-bold"><span>{s['ticker']}</span><span class="text-blue-400">${s['price']}</span></div>
                        <div class="grid grid-cols-2 text-xs mt-2 gap-2">
                            <div class="bg-slate-800 p-2 rounded text-center">股息: <span class="{dy_c}">{s['div_yield']}%</span></div>
                            <div class="bg-slate-800 p-2 rounded text-center">RSI: {s['rsi']}</div>
                        </div>
                    </div>"""

html_content += f"""
                </div>
                <div class="lg:w-2/3 space-y-4">
                    <div class="card p-4 bg-slate-800 flex justify-between items-center">
                        <h3 id="c_title" class="font-bold">點擊左側查看 K 線圖</h3>
                        <a id="tv_btn" href="#" target="_blank" class="hidden bg-blue-600 px-3 py-1 rounded text-xs">TradingView</a>
                    </div>
                    <div class="card p-4 h-[500px] bg-slate-900 shadow-inner"><canvas id="sChart"></canvas></div>
                </div>
            </div>
        </div>

        <div id="tab-port" class="tab-content">
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6 text-center">
                <div class="card p-4"><p class="text-xs text-slate-400">累計利潤</p><p class="text-xl font-bold {'text-green-400' if total_net_profit >= 0 else 'text-red-400'}">${total_net_profit:,.0f}</p></div>
                <div class="card p-4"><p class="text-xs text-slate-400">目前動用</p><p class="text-xl font-bold text-blue-400">${active_invested:,.0f}</p></div>
                <div class="card p-4"><p class="text-xs text-slate-400">歷史勝率</p><p class="text-xl font-bold text-yellow-400">{win_rate:.1f}%</p></div>
                <div class="card p-4"><p class="text-xs text-slate-400">持倉檔數</p><p class="text-xl font-bold">{len(portfolio['open_positions'])}</p></div>
            </div>
            <div class="card p-4 h-[300px] mb-6"><canvas id="eChart"></canvas></div>
            <div class="card">
                <div class="p-3 bg-slate-800 font-bold text-sm">💼 當前追蹤庫存</div>
                <div class="overflow-x-auto">
                    <table>
                        <tr><th>代碼</th><th>成本</th><th>現價</th><th>損益</th><th>股息</th><th>業績</th></tr>
"""
for p in reversed(portfolio['open_positions']):
    cur_p = closes[p['ticker']].iloc[-1]
    diff = (cur_p / p['entry_price'] - 1) * 100
    html_content += f"<tr><td class='font-bold text-blue-400'>{p['ticker']}</td><td>${p['entry_price']}</td><td>${cur_p:.2f}</td><td class='{'text-green-400' if diff>0 else 'text-red-400'}'>{diff:+.2f}%</td><td>{p.get('div_yield',0)}%</td><td class='text-xs'>{p.get('earn_label','-')}</td></tr>"
html_content += """</table></div></div></div>
    </div>

    <script>
        function showTab(t) {
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('tab-active'));
            document.getElementById('tab-' + t).classList.add('active');
            document.getElementById('btn-' + t).classList.add('tab-active');
        }

        const sigData = """ + json.dumps(signals) + """;
        let sChart = null;

        function updateChart(tk, dy, el, tv) {
            const s = sigData.find(x => x.ticker === tk);
            document.getElementById('c_title').innerHTML = `${tk} <span class="text-xs text-slate-400 font-normal ml-2">股息:${dy}% | 業績:${el}</span>`;
            const tvBtn = document.getElementById('tv_btn');
            tvBtn.href = `https://www.tradingview.com/chart/?symbol=${tv}`;
            tvBtn.classList.remove('hidden');

            const ctx = document.getElementById('sChart').getContext('2d');
            if (sChart) sChart.destroy();
            sChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: s.chart_dates,
                    datasets: [
                        { label: '價格', data: s.chart_prices, borderColor: '#3b82f6', pointRadius: 0, tension: 0.1 },
                        { label: '20MA', data: s.chart_sma20, borderColor: '#f59e0b', borderDash: [5,5], pointRadius: 0 },
                        { label: '布林下', data: s.chart_lbb, borderColor: '#ef4444', borderDash: [2,2], pointRadius: 0 }
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false, scales: { x: { ticks: { color: '#94a3b8' } }, y: { ticks: { color: '#94a3b8' } } } }
            });
        }

        new Chart(document.getElementById('eChart'), {
            type: 'line',
            data: {
                labels: """ + json.dumps(eq_dates) + """,
                datasets: [{ label: '累計利潤', data: """ + json.dumps(eq_values) + """, borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.1)', fill: true, tension: 0.3 }]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });

        if (sigData.length > 0) updateChart(sigData[0].ticker, sigData[0].div_yield, sigData[0].earn_label, sigData[0].tv_ticker);
    </script>
</body>
</html>
"""
with open("index.html", 'w', encoding='utf-8') as f: f.write(html_content)
print("✅ V7.2 生成完成")
