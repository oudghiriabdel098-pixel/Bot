# bot_hybrid_smart_early.py
import os
import requests
import time
import pandas as pd
import ta
import numpy as np
from datetime import datetime

# ========== CONFIG ==========
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

TIMEFRAME = "5m"
KLINES_LIMIT = 60

# Strategy thresholds
PUMP_PERCENT = 3.0
TP_PERCENT = 3.0
SL_PERCENT = -1.5
MIN_KLINES = 12

# Early pump
EARLY_PUMP_THRESHOLD = 2.5
EARLY_TIME_WINDOW = 60

# Filters
VOLUME_SPIKE_MULT = 2.0
BUY_ASK_RATIO_MIN = 1.6
ASK_WALL_THRESHOLD = 20000.0
WICK_RATIO_MAX = 2.5
RSI_MAX = 75

SYMBOL_DELAY = 0.6
LOOP_DELAY = 8

# ---------- SEND ALERT USING TOKEN ----------
def send_alert(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg})
    except Exception as e:
        print("Telegram send error:", e)

# ---------- BINANCE HELPERS ----------
def get_active_usdt_pairs():
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        data = requests.get(url, timeout=10).json()
        symbols = data.get("symbols", [])
        return {s["symbol"] for s in symbols if s.get("status") == "TRADING" and s.get("symbol","").endswith("USDT")}
    except:
        return set()

def get_top_gainers(active_pairs, min_change=3.0, top_n=30):
    url = "https://api.binance.com/api/v3/ticker/24hr"
    try:
        tickers = requests.get(url, timeout=10).json()
    except:
        return []
    arr = []
    for t in tickers:
        s = t.get("symbol")
        if not s or s not in active_pairs:
            continue
        try: ch = float(t.get("priceChangePercent", 0))
        except: ch = 0.0
        if ch >= min_change:
            arr.append((s, ch))
    arr.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in arr[:top_n]]

def get_klines_df(symbol, interval=TIMEFRAME, limit=KLINES_LIMIT):
    url = "https://api.binance.com/api/v3/klines"
    try:
        r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=10)
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return None
        df = pd.DataFrame(data, columns=["t","o","h","l","c","v","ct","qv","n","tb","tbq","ignore"])
        df[["o","h","l","c","v","qv"]] = df[["o","h","l","c","v","qv"]].astype(float)
        return df
    except:
        return None

def get_orderbook_quote(symbol, depth=20):
    url = "https://api.binance.com/api/v3/depth"
    try:
        r = requests.get(url, params={"symbol": symbol, "limit": 100}, timeout=6)
        data = r.json()
        bids = data.get("bids", [])[:depth]
        asks = data.get("asks", [])[:depth]
        buy_q = sum(float(p)*float(q) for p,q in bids)
        sell_q = sum(float(p)*float(q) for p,q in asks)
        top_ask = float(asks[0][0]) * float(asks[0][1]) if asks else 0.0
        return {"buy_quote": buy_q, "sell_quote": sell_q, "top_ask": top_ask}
    except:
        return None

# ---------- INDICATORS ----------
def calculate_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.isna().all() else None

def is_uptrend(df):
    if len(df["c"]) < 50:
        return False
    ma20 = df["c"].rolling(20).mean()
    ma50 = df["c"].rolling(50).mean()
    return ma20.iloc[-1] > ma50.iloc[-1]

def detect_reverse_dump(last, prev):
    try:
        body = abs(last["c"] - last["o"])
        upper_wick = last["h"] - max(last["c"], last["o"])
        ratio = upper_wick / body if body > 0 else 999
        return ratio >= WICK_RATIO_MAX and last["c"] < last["o"]
    except:
        return False

# ---------- EARLY PUMP DETECTOR ----------
def early_pump_detector(df, symbol):

    try:
        r = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}", timeout=5)
        live = float(r.json().get("price", 0))
    except:
        return False

    open_time = df["t"].iloc[-1]
    open_price = df["o"].iloc[-1]

    seconds_passed = time.time() - (open_time / 1000)
    if seconds_passed > EARLY_TIME_WINDOW:
        return False

    change = ((live - open_price) / open_price) * 100
    if change < EARLY_PUMP_THRESHOLD:
        return False

    rsi = calculate_rsi(df["c"])
    if rsi >= (RSI_MAX + 5):
        return False

    if not is_uptrend(df):
        return False

    msg = (
        f"🚀 EARLY PUMP DETECTED\n"
        f"Pair: {symbol}\n+{change:.2f}% in {int(seconds_passed)}s\nRSI:{rsi:.1f}"
    )
    send_alert(msg)
    print(f"[EARLY] {symbol} +{change:.2f}%")
    return True

# ---------- MAIN ANALYZER ----------
def analyze_symbol_combined(symbol):
    df = get_klines_df(symbol)
    if df is None or len(df) < MIN_KLINES:
        return

    if early_pump_detector(df, symbol):
        return

    df["ema20"] = ta.trend.EMAIndicator(df["c"], 20).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(df["c"], 50).ema_indicator()
    df["rsi"] = ta.momentum.RSIIndicator(df["c"], 14).rsi()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    change_5m = ((last["c"] - prev["o"]) / prev["o"]) * 100

    qv_mean = df["qv"][:-1].mean()
    qv_last = last["qv"]
    vol_spike = qv_last > qv_mean * VOLUME_SPIKE_MULT if qv_mean > 0 else False

    trend_up = last["c"] > last["ema20"] and last["c"] > last["ema50"]
    rsi_ok = last["rsi"] < RSI_MAX

    ob = get_orderbook_quote(symbol)
    if ob is None:
        return

    buy_q = ob["buy_quote"]
    sell_q = ob["sell_quote"]
    top_ask = ob["top_ask"]

    if top_ask >= ASK_WALL_THRESHOLD:
        return

    ratio = buy_q / sell_q if sell_q > 0 else 999
    buy_pressure_ok = ratio >= BUY_ASK_RATIO_MIN

    reversal = detect_reverse_dump(last, prev)

    if (
        change_5m >= PUMP_PERCENT
        and vol_spike
        and trend_up
        and rsi_ok
        and buy_pressure_ok
        and not reversal
    ):
        msg = (
            f"🚀 SMART-HYBRID ENTRY\n"
            f"{symbol}\n"
            f"Change: {change_5m:.2f}%\n"
            f"Buy/Sell:{ratio:.2f}"
        )
        send_alert(msg)

# ---------- MAIN LOOP ----------
def main():
    send_alert("🔥 SMART-HYBRID-EARLY BOT Started")
    active = get_active_usdt_pairs()

    while True:
        try:
            gainers = get_top_gainers(active, min_change=3.0, top_n=30)

            if not gainers:
                time.sleep(LOOP_DELAY)
                continue

            for s in gainers:
                analyze_symbol_combined(s)
                time.sleep(SYMBOL_DELAY)

            time.sleep(LOOP_DELAY)

        except Exception as e:
            print("ERROR in main loop:", e)
            send_alert(f"⚠️ ERROR in bot loop:\n{e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
