
import os, json, re, asyncio, threading, math
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from openai import OpenAI

# ============================================================
# ENV - set these in Azure App Service Configuration
# ============================================================
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

app = FastAPI(title="Commodity Sentiment Terminal - By Abbas")
executor = ThreadPoolExecutor(max_workers=8)

# ============================================================
# AZURE GLOBAL HIT COUNTER + SIGNAL MEMORY
# ============================================================
HIT_COUNTER_FILE = Path(os.getenv("HIT_COUNTER_FILE", "/home/site/hit_counter.json"))
SIGNAL_TRACKER_FILE = Path(os.getenv("SIGNAL_TRACKER_FILE", "/home/site/signal_tracker.json"))
if not HIT_COUNTER_FILE.parent.exists():
    HIT_COUNTER_FILE = Path("hit_counter.json")
if not SIGNAL_TRACKER_FILE.parent.exists():
    SIGNAL_TRACKER_FILE = Path("signal_tracker.json")

HIT_LOCK = threading.Lock()
TRACKER_LOCK = threading.Lock()


def read_hit_count() -> int:
    try:
        if HIT_COUNTER_FILE.exists():
            data = json.loads(HIT_COUNTER_FILE.read_text())
            return int(data.get("hits", 0))
    except Exception:
        pass
    return 0


def write_hit_count(count: int) -> None:
    try:
        HIT_COUNTER_FILE.parent.mkdir(parents=True, exist_ok=True)
        HIT_COUNTER_FILE.write_text(json.dumps({"hits": int(count)}))
    except Exception:
        pass


ASSETS = {
    "GOLD": {"name": "Gold Spot", "icon": "🟡", "yf": "GC=F", "keywords": ["gold", "xau", "bullion", "safe haven", "inflation", "fed", "rates", "dollar"]},
    "SILVER": {"name": "Silver", "icon": "⚪", "yf": "SI=F", "keywords": ["silver", "xag", "precious metal", "industrial metal", "solar", "dollar", "rates"]},
    "WTI": {"name": "Crude Oil WTI", "icon": "🛢️", "yf": "CL=F", "keywords": ["wti", "crude", "oil", "opec", "iran", "gulf", "hormuz", "sanctions", "us strikes", "middle east"]},
    "BRENT": {"name": "Brent Crude", "icon": "🛢️", "yf": "BZ=F", "keywords": ["brent", "crude", "oil", "opec", "iran", "gulf", "hormuz", "sanctions", "shipping"]},
    "BTC": {"name": "Bitcoin", "icon": "₿", "yf": "BTC-USD", "keywords": ["bitcoin", "btc", "crypto", "etf", "risk assets", "liquidity", "fed", "rates"]},
    "USTEC100": {"name": "USTEC 100 Future", "icon": "📈", "yf": "NQ=F", "keywords": ["nasdaq", "nasdaq 100", "nq futures", "tech stocks", "ai stocks", "fed", "rates", "yields"]},
}

INTERVALS = {
    "1M": {"yf": "1m", "period": "1d"},
    "15M": {"yf": "15m", "period": "5d"},
    "30M": {"yf": "30m", "period": "10d"},
    "1H": {"yf": "1h", "period": "30d"},
    "1D": {"yf": "1d", "period": "6mo"},
}


def clamp(x, lo=-100, hi=100):
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return 0.0


def safe_float(x, default=0.0):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def clean(x):
    return re.sub(r"\s+", " ", str(x or "")).strip()


# ============================================================
# MARKET DATA
# ============================================================
def load_candles(asset_key, tf):
    meta = ASSETS[asset_key]
    cfg = INTERVALS[tf]
    symbols = meta.get("yf")
    if isinstance(symbols, str):
        symbols = [symbols]

    attempts = [(cfg["yf"], cfg["period"])]
    if tf == "1M":
        attempts += [("2m", "5d"), ("5m", "5d"), ("15m", "5d"), ("30m", "10d")]
    elif tf == "15M":
        attempts += [("30m", "10d"), ("1h", "30d")]
    elif tf == "30M":
        attempts += [("15m", "5d"), ("1h", "30d")]

    def _normalize(raw):
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(c[0] if c[0] else c[-1]).strip() for c in df.columns.to_list()]
        df = df.reset_index()
        df.columns = [str(c).strip() for c in df.columns]
        colmap = {c.lower().replace(" ", ""): c for c in df.columns}
        time_col = None
        for k in ["datetime", "date", "index"]:
            if k in colmap:
                time_col = colmap[k]
                break
        if time_col is None:
            return pd.DataFrame()
        needed = {}
        for want in ["open", "high", "low", "close", "volume"]:
            if want in colmap:
                needed[want] = colmap[want]
            elif want == "close" and "adjclose" in colmap:
                needed[want] = colmap["adjclose"]
        if not all(k in needed for k in ["open", "high", "low", "close"]):
            return pd.DataFrame()
        out = pd.DataFrame({
            "time": pd.to_datetime(df[time_col], errors="coerce"),
            "open": pd.to_numeric(df[needed["open"]], errors="coerce"),
            "high": pd.to_numeric(df[needed["high"]], errors="coerce"),
            "low": pd.to_numeric(df[needed["low"]], errors="coerce"),
            "close": pd.to_numeric(df[needed["close"]], errors="coerce"),
            "volume": pd.to_numeric(df[needed["volume"]], errors="coerce") if "volume" in needed else 0,
        })
        out = out.dropna(subset=["time", "open", "high", "low", "close"])
        out = out[(out["open"] > 0) & (out["high"] > 0) & (out["low"] > 0) & (out["close"] > 0)]
        out = out.sort_values("time").drop_duplicates("time")
        if len(out) < 30 or out["close"].nunique() <= 2:
            return pd.DataFrame()
        return out.tail(800)

    for symbol in symbols:
        for interval, period in attempts:
            for mode in ("download", "history"):
                try:
                    if mode == "download":
                        raw = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False, threads=False, group_by="column")
                    else:
                        raw = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False, actions=False)
                    df = _normalize(raw)
                    if not df.empty:
                        return df
                except Exception:
                    pass
    return pd.DataFrame()


# ============================================================
# INDICATORS + INSTITUTIONAL FILTERS
# ============================================================
def add_indicators(df):
    d = df.copy().sort_values("time")
    d["ema9"] = d["close"].ewm(span=9, adjust=False).mean()
    d["ema21"] = d["close"].ewm(span=21, adjust=False).mean()
    d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()
    d["ema200"] = d["close"].ewm(span=200, adjust=False).mean()

    delta = d["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    tr1 = d["high"] - d["low"]
    tr2 = (d["high"] - d["close"].shift()).abs()
    tr3 = (d["low"] - d["close"].shift()).abs()
    d["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    d["atr"] = d["tr"].ewm(alpha=1/14, adjust=False).mean().fillna(d["tr"].mean())
    d["atr_pct"] = (d["atr"] / d["close"] * 100).replace([np.inf, -np.inf], np.nan).fillna(0)

    # ADX calculation for market regime detection
    up_move = d["high"].diff()
    down_move = -d["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = d["atr"].replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=d.index).ewm(alpha=1/14, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=d.index).ewm(alpha=1/14, adjust=False).mean() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    d["adx"] = dx.ewm(alpha=1/14, adjust=False).mean().fillna(15)
    d["plus_di"] = plus_di.fillna(0)
    d["minus_di"] = minus_di.fillna(0)

    d["vol_ma"] = d["volume"].rolling(20).mean().fillna(d["volume"].mean())
    d["ret"] = d["close"].pct_change().fillna(0)
    return d


def support_resistance(df, lookback=80):
    d = df.tail(lookback).copy()
    price = safe_float(d["close"].iloc[-1])
    supports = []
    resistances = []
    for i in range(2, len(d) - 2):
        lo = d["low"].iloc[i]
        hi = d["high"].iloc[i]
        if lo <= d["low"].iloc[i-2:i+3].min():
            supports.append(float(lo))
        if hi >= d["high"].iloc[i-2:i+3].max():
            resistances.append(float(hi))
    supports = [x for x in supports if x < price]
    resistances = [x for x in resistances if x > price]
    sup = max(supports) if supports else float(d["low"].min())
    res = min(resistances) if resistances else float(d["high"].max())
    return {"support": round(sup, 4), "resistance": round(res, 4)}


def volume_profile(df, bins=24):
    d = df.tail(160).copy()
    typical = (d["high"] + d["low"] + d["close"]) / 3
    vol = d["volume"].replace(0, np.nan)
    if vol.isna().all() or vol.sum(skipna=True) <= 0:
        vol = pd.Series(np.ones(len(d)), index=d.index)
    try:
        cats = pd.cut(typical, bins=bins, duplicates="drop")
        grouped = vol.groupby(cats, observed=False).sum()
        if grouped.empty:
            return {"poc": safe_float(d["close"].iloc[-1]), "bias": 0}
        poc_interval = grouped.idxmax()
        poc = float((poc_interval.left + poc_interval.right) / 2)
        price = safe_float(d["close"].iloc[-1])
        atr = safe_float(add_indicators(d)["atr"].iloc[-1], price * 0.005)
        bias = clamp((price - poc) / max(atr, price * 0.001) * 10, -15, 15)
        return {"poc": round(poc, 4), "bias": round(bias, 2)}
    except Exception:
        return {"poc": safe_float(d["close"].iloc[-1]), "bias": 0}


def market_regime(df):
    d = add_indicators(df)
    last = d.iloc[-1]
    adx = safe_float(last.adx, 15)
    atr_pct = safe_float(last.atr_pct, 0)
    ema_spread = abs(safe_float(last.ema21 - last.ema50)) / max(safe_float(last.close), 1) * 100
    if adx >= 25 and ema_spread > atr_pct * 0.20:
        regime = "TRENDING"
        multiplier = 1.12
    elif atr_pct > d["atr_pct"].tail(120).quantile(0.75):
        regime = "HIGH VOLATILITY"
        multiplier = 0.88
    elif adx < 17:
        regime = "RANGING"
        multiplier = 0.78
    else:
        regime = "NORMAL"
        multiplier = 1.0
    return {"regime": regime, "adx": round(adx, 2), "atr_pct": round(atr_pct, 3), "multiplier": multiplier}


def raw_technical_model(df):
    d = add_indicators(df)
    if len(d) < 60:
        return 0.0
    last = d.iloc[-1]
    prev = d.iloc[-4] if len(d) > 4 else d.iloc[-2]
    score = 0.0

    # Trend stack + price location
    score += 18 if last.ema9 > last.ema21 else -18
    score += 16 if last.ema21 > last.ema50 else -16
    score += 16 if last.close > last.ema200 else -16
    score += 10 if last.close > last.ema21 else -10

    # Momentum quality
    score += 14 if last.macd > last.macd_signal else -14
    score += clamp(last.macd_hist / max(last.atr, last.close * 0.001) * 25, -12, 12)
    mom10 = (last.close - d["close"].iloc[-10]) / max(d["close"].iloc[-10], 1) * 100
    mom20 = (last.close - d["close"].iloc[-20]) / max(d["close"].iloc[-20], 1) * 100
    score += clamp(mom10 * 18, -12, 12)
    score += clamp(mom20 * 10, -10, 10)

    # RSI: continuation zone is stronger than overbought/oversold alone
    rsi = safe_float(last.rsi, 50)
    if 52 <= rsi <= 68:
        score += 10
    elif 32 <= rsi <= 48:
        score -= 10
    elif rsi > 75:
        score -= 8
    elif rsi < 25:
        score += 8

    # ADX directional confirmation
    if last.adx >= 20:
        score += 8 if last.plus_di > last.minus_di else -8

    # Breakout / breakdown against recent range
    recent_high = d["high"].iloc[-31:-1].max()
    recent_low = d["low"].iloc[-31:-1].min()
    if last.close > recent_high:
        score += 10
    elif last.close < recent_low:
        score -= 10

    # Penalize sudden reversal against the signal
    if score > 0 and last.close < prev.close and last.macd_hist < prev.macd_hist:
        score -= 8
    if score < 0 and last.close > prev.close and last.macd_hist > prev.macd_hist:
        score += 8

    return clamp(score)


def backtest_technical(df, horizon=3):
    d = add_indicators(df).tail(420).reset_index(drop=True)
    results = []
    if len(d) < 90:
        return {"win_rate": 0.50, "trades": 0, "expectancy": 0.0, "score_adj": 0}
    for i in range(70, len(d) - horizon):
        sub = d.iloc[:i+1].copy()
        score = raw_technical_model(sub)
        if abs(score) < 24:
            continue
        direction = 1 if score > 0 else -1
        entry = safe_float(d["close"].iloc[i])
        exitp = safe_float(d["close"].iloc[i+horizon])
        atr = max(safe_float(d["atr"].iloc[i], entry * 0.005), entry * 0.0005)
        r_mult = ((exitp - entry) * direction) / atr
        results.append(r_mult)
    if not results:
        return {"win_rate": 0.50, "trades": 0, "expectancy": 0.0, "score_adj": 0}
    arr = np.array(results, dtype=float)
    win_rate = float((arr > 0).mean())
    expectancy = float(arr.mean())
    score_adj = clamp((win_rate - 0.50) * 50 + expectancy * 8, -12, 12)
    return {"win_rate": round(win_rate, 3), "trades": int(len(arr)), "expectancy": round(expectancy, 3), "score_adj": round(score_adj, 2)}


def multi_timeframe_confirmation(dfs):
    # Signals are still based on 30M; higher frames only confirm or reduce confidence.
    scores = {}
    for tf, df in dfs.items():
        if df is not None and not df.empty:
            scores[tf] = raw_technical_model(df)
    base = scores.get("30M", 0)
    if not scores:
        return {"score": 0, "alignment": 0, "scores": {}}
    signs = []
    for tf in ["15M", "30M", "1H", "1D"]:
        s = scores.get(tf)
        if s is None:
            continue
        signs.append(1 if s > 12 else -1 if s < -12 else 0)
    base_sign = 1 if base > 0 else -1 if base < 0 else 0
    aligned = sum(1 for s in signs if s == base_sign and s != 0)
    opposed = sum(1 for s in signs if s == -base_sign and s != 0)
    alignment = aligned - opposed
    score = clamp(alignment * 7, -18, 18)
    return {"score": round(score, 2), "alignment": alignment, "scores": {k: round(v, 2) for k, v in scores.items()}}


def economic_event_filter(news, asset_key):
    text = " ".join([(n.get("headline", "") + " " + n.get("summary", "")) for n in news]).lower()
    high_impact = ["fed", "fomc", "powell", "cpi", "inflation", "jobs report", "nonfarm", "nfp", "pce", "rate decision", "ecb", "opec", "eia", "inventory", "war", "strike", "hormuz", "sanction"]
    hits = [w for w in high_impact if w in text]
    risk_penalty = min(12, len(hits) * 2)
    if asset_key in ["WTI", "BRENT"] and any(w in text for w in ["opec", "eia", "inventory", "hormuz", "iran"]):
        risk_penalty = max(risk_penalty, 6)
    if asset_key in ["GOLD", "SILVER", "USTEC100", "BTC"] and any(w in text for w in ["fed", "cpi", "inflation", "powell", "rate"]):
        risk_penalty = max(risk_penalty, 6)
    return {"hits": hits[:6], "risk_penalty": risk_penalty}


def optimized_levels(df, direction):
    d = add_indicators(df)
    last = d.iloc[-1]
    price = safe_float(last.close)
    atr = max(safe_float(last.atr, price * 0.006), price * 0.001)
    sr = support_resistance(d)
    vp = volume_profile(d)
    if direction >= 0:
        raw_stop = price - atr * 1.35
        sr_stop = min(raw_stop, sr["support"] - atr * 0.15) if sr["support"] < price else raw_stop
        stop = sr_stop
        target = price + max(atr * 2.15, (price - stop) * 1.65)
    else:
        raw_stop = price + atr * 1.35
        sr_stop = max(raw_stop, sr["resistance"] + atr * 0.15) if sr["resistance"] > price else raw_stop
        stop = sr_stop
        target = price - max(atr * 2.15, (stop - price) * 1.65)
    return {"entry": round(price, 4), "target": round(target, 4), "stop": round(stop, 4), "support": sr["support"], "resistance": sr["resistance"], "poc": vp["poc"]}


def technical_score(df, mtf=None):
    d = add_indicators(df)
    last = d.iloc[-1]
    base = raw_technical_model(d)
    regime = market_regime(d)
    sr = support_resistance(d)
    vp = volume_profile(d)
    bt = backtest_technical(d)
    mtf_score = safe_float((mtf or {}).get("score", 0))

    # Support/resistance and volume profile confirmation
    price = safe_float(last.close)
    sr_bias = 0
    if price > sr["resistance"]:
        sr_bias += 8
    elif price < sr["support"]:
        sr_bias -= 8
    vp_bias = safe_float(vp.get("bias", 0))

    final_score = (base + mtf_score + sr_bias + vp_bias + bt["score_adj"]) * regime["multiplier"]
    final_score = clamp(final_score)
    direction = 1 if final_score >= 0 else -1
    levels = optimized_levels(d, direction)

    return {
        "score": round(final_score, 2),
        "price": round(price, 4),
        "rsi": round(safe_float(last.rsi), 2),
        "macd": round(safe_float(last.macd), 4),
        "atr": round(safe_float(last.atr, price * 0.01), 4),
        "entry": levels["entry"],
        "target": levels["target"],
        "stop": levels["stop"],
        "support": levels["support"],
        "resistance": levels["resistance"],
        "poc": levels["poc"],
        "regime": regime,
        "backtest": bt,
        "raw_score": round(base, 2),
    }


# ============================================================
# NEWS + AI - news display kept as original behavior
# ============================================================
async def fetch_finnhub_news(asset_key):
    if not FINNHUB_API_KEY:
        return []
    url = "https://finnhub.io/api/v1/news"
    params = {"category": "general", "token": FINNHUB_API_KEY}
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            raw = r.json()
    except Exception:
        return []

    keywords = [k.lower() for k in ASSETS[asset_key]["keywords"]]
    geopolitics = ["iran", "hormuz", "gulf", "sanction", "strike", "middle east", "shipping", "war", "ceasefire", "deal"]
    out = []
    for n in raw:
        headline = clean(n.get("headline"))
        summary = clean(n.get("summary"))
        source = clean(n.get("source"))
        text = f"{headline} {summary}".lower()
        relevance = sum(1 for k in keywords if k in text)
        relevance += sum(1 for k in geopolitics if k in text) * 1.3
        if relevance > 0:
            out.append({"headline": headline, "summary": summary[:180], "source": source, "url": n.get("url", ""), "relevance": relevance})
    return sorted(out, key=lambda x: x["relevance"], reverse=True)[:10]


def news_lexicon_score(news, asset_key):
    if not news:
        return 0
    text = " ".join([n["headline"] + " " + n["summary"] for n in news]).lower()
    bullish = ["rally", "surge", "gain", "rise", "rebound", "strong demand", "supply cut", "disruption", "shortage", "drawdown", "inventory draw", "sanctions", "strike", "hormuz", "shipping risk", "safe haven", "rate cut", "weak dollar", "breakout", "record high"]
    bearish = ["drop", "fall", "slump", "selloff", "weak demand", "inventory build", "surplus", "oversupply", "recession", "strong dollar", "rate hike", "peace deal", "ceasefire", "supply restored", "risk off"]
    score = 0
    for w in bullish:
        if w in text:
            score += 8
    for w in bearish:
        if w in text:
            score -= 8
    if asset_key in ["WTI", "BRENT"]:
        if any(x in text for x in ["iran", "hormuz", "gulf", "sanction", "strike", "shipping risk"]):
            score += 20
        if any(x in text for x in ["inventory build", "oversupply", "weak demand"]):
            score -= 20
    return clamp(score)


async def openai_sentiment(asset_key, news, tech, institutional_context=None):
    if not OPENAI_API_KEY:
        return {"score": 0, "bias": "NEUTRAL", "summary": "OpenAI key missing. Using fallback sentiment.", "risk": "AI sentiment unavailable."}
    headlines = [f"{n['source']}: {n['headline']} - {n['summary']}" for n in news[:8]]
    ctx = institutional_context or {}
    prompt = f"""
Asset: {ASSETS[asset_key]['name']}
30M Technical Score={tech['score']}
Price={tech['price']} RSI={tech['rsi']} MACD={tech['macd']} ATR={tech['atr']}
Regime={tech.get('regime', {}).get('regime')} Backtest={tech.get('backtest')}
Support={tech.get('support')} Resistance={tech.get('resistance')} VolumePOC={tech.get('poc')}
MTF={ctx.get('mtf')} EventRisk={ctx.get('event_risk')}

News:
{chr(10).join(headlines)}

Return only JSON:
{{"score": number between -100 and 100,"bias": "BULLISH" or "BEARISH" or "NEUTRAL","summary": "short reason","risk": "short risk"}}
"""
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Return valid JSON only. Be conservative and do not overrule strong 30M technical evidence without clear news catalyst."}, {"role": "user", "content": prompt}],
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
        text = re.sub(r"^```json|```$", "", text, flags=re.I).strip()
        data = json.loads(text)
        return {"score": clamp(data.get("score", 0)), "bias": data.get("bias", "NEUTRAL"), "summary": data.get("summary", ""), "risk": data.get("risk", "")}
    except Exception as e:
        return {"score": 0, "bias": "NEUTRAL", "summary": "OpenAI sentiment failed. Fallback active.", "risk": str(e)[:100]}


def read_tracker():
    try:
        if SIGNAL_TRACKER_FILE.exists():
            return json.loads(SIGNAL_TRACKER_FILE.read_text())
    except Exception:
        pass
    return {"signals": []}


def write_tracker(data):
    try:
        SIGNAL_TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
        SIGNAL_TRACKER_FILE.write_text(json.dumps(data)[-250000:])
    except Exception:
        pass


def update_signal_tracker(asset_key, signal, price, confidence):
    # lightweight live paper-trading validation memory; does not alter UI schema
    if signal not in ["BUY SIGNAL", "SELL SIGNAL"]:
        return {"paper_trades": 0, "recent_accuracy": 0.5}
    with TRACKER_LOCK:
        data = read_tracker()
        rows = data.get("signals", [])
        now = datetime.now(timezone.utc).isoformat()
        direction = 1 if signal == "BUY SIGNAL" else -1
        rows.append({"asset": asset_key, "time": now, "signal": signal, "direction": direction, "price": price, "confidence": confidence})
        rows = rows[-300:]
        same = [r for r in rows if r.get("asset") == asset_key]
        # Approximate live tracker from subsequent observed prices on refreshes.
        closed = []
        if len(same) >= 2:
            latest_price = price
            for r in same[:-1][-80:]:
                move = (latest_price - safe_float(r.get("price"))) * int(r.get("direction", 1))
                closed.append(1 if move > 0 else 0)
        acc = float(np.mean(closed)) if closed else 0.5
        data["signals"] = rows
        write_tracker(data)
        return {"paper_trades": len(closed), "recent_accuracy": round(acc, 3)}


def probability_from_inputs(fusion, tech, backtest, event_risk):
    bt_wr = safe_float(backtest.get("win_rate", 0.5), 0.5)
    bt_trades = int(backtest.get("trades", 0) or 0)
    bt_weight = min(1.0, bt_trades / 40.0)
    base = 1 / (1 + math.exp(-abs(fusion) / 24.0))
    prob = (base * 0.72) + ((bt_wr * bt_weight + 0.5 * (1 - bt_weight)) * 0.28)
    prob -= safe_float(event_risk.get("risk_penalty", 0)) / 250.0
    return round(max(0.45, min(0.86, prob)), 3)


def fusion_signal(tech, news_score, ai, mtf=None, event_risk=None, tracker=None):
    tech_s = clamp(tech["score"])
    news_s = clamp(news_score)
    ai_s = clamp(ai.get("score", 0))
    mtf_s = clamp((mtf or {}).get("score", 0))
    event_penalty = safe_float((event_risk or {}).get("risk_penalty", 0))

    # Adaptive fusion: if news/AI are neutral/missing, do not force HOLD; let 30M technical + MTF dominate.
    active_news = abs(news_s) >= 8
    active_ai = abs(ai_s) >= 8
    if active_news and active_ai:
        fusion = tech_s * 0.50 + news_s * 0.18 + ai_s * 0.22 + mtf_s * 0.10
    elif active_ai:
        fusion = tech_s * 0.62 + ai_s * 0.25 + mtf_s * 0.13
    elif active_news:
        fusion = tech_s * 0.64 + news_s * 0.22 + mtf_s * 0.14
    else:
        fusion = tech_s * 0.78 + mtf_s * 0.22

    # Penalize confidence during high-impact event risk, but do not blindly flip direction.
    if fusion > 0:
        fusion -= event_penalty * 0.45
    elif fusion < 0:
        fusion += event_penalty * 0.45
    fusion = clamp(fusion)

    regime_name = tech.get("regime", {}).get("regime", "NORMAL")
    threshold = 22
    if regime_name == "TRENDING":
        threshold = 18
    elif regime_name == "RANGING":
        threshold = 28
    elif regime_name == "HIGH VOLATILITY":
        threshold = 30

    prob = probability_from_inputs(fusion, tech, tech.get("backtest", {}), event_risk or {})

    if fusion >= threshold and prob >= 0.54:
        signal = "BUY SIGNAL"
        label = "BULLISH"
    elif fusion <= -threshold and prob >= 0.54:
        signal = "SELL SIGNAL"
        label = "BEARISH"
    else:
        signal = "HOLD / WAIT"
        label = "NEUTRAL"

    tracker_acc = safe_float((tracker or {}).get("recent_accuracy", 0.5), 0.5)
    tracker_boost = (tracker_acc - 0.5) * 10 if (tracker or {}).get("paper_trades", 0) >= 5 else 0
    confidence = min(100, max(25, abs(fusion) * 1.28 + prob * 25 + tracker_boost))

    return {
        "fusion": round(fusion, 2),
        "confidence": round(confidence, 1),
        "probability": prob,
        "signal": signal,
        "label": label,
        "tech_percent": round((tech_s + 100) / 2, 1),
        "news_percent": round((news_s + 100) / 2, 1),
        "ai_percent": round((ai_s + 100) / 2, 1),
        "threshold": threshold,
    }


# ============================================================
# CHART - UI unchanged
# ============================================================
def build_chart(df, asset_name):
    d = add_indicators(df)
    x = pd.to_datetime(d["time"])
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=x, open=d["open"], high=d["high"], low=d["low"], close=d["close"], name=asset_name, increasing_line_width=1, decreasing_line_width=1))
    fig.add_trace(go.Scatter(x=x, y=d["ema9"], name="EMA9", mode="lines", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=x, y=d["ema21"], name="EMA21", mode="lines", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=x, y=d["ema50"], name="EMA50", mode="lines", line=dict(width=1)))
    y_min = float(d["low"].tail(200).min())
    y_max = float(d["high"].tail(200).max())
    pad = max((y_max - y_min) * 0.14, y_max * 0.003)
    fig.update_layout(template="plotly_dark", height=330, margin=dict(l=58, r=34, t=22, b=46), paper_bgcolor="#111a26", plot_bgcolor="#111a26", font=dict(color="#dce7f3", size=10), xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9)), uirevision="keep")
    fig.update_xaxes(type="date", showgrid=True, automargin=True)
    fig.update_yaxes(range=[y_min - pad, y_max + pad], fixedrange=False, automargin=True, zeroline=False)
    return json.loads(fig.to_json())


async def process(asset_key, tf):
    loop = asyncio.get_running_loop()

    # Chart follows UI timeframe; signal and all technicals remain fixed to 30M.
    chart_future = loop.run_in_executor(executor, load_candles, asset_key, tf)
    signal_future = loop.run_in_executor(executor, load_candles, asset_key, "30M")
    h1_future = loop.run_in_executor(executor, load_candles, asset_key, "1H")
    d1_future = loop.run_in_executor(executor, load_candles, asset_key, "1D")

    chart_df, signal_df, h1_df, d1_df = await asyncio.gather(chart_future, signal_future, h1_future, d1_future)
    if chart_df.empty and signal_df.empty:
        raise RuntimeError("No candle data returned.")
    if chart_df.empty:
        chart_df = signal_df.copy()
    if signal_df.empty:
        signal_df = chart_df.copy()

    mtf = multi_timeframe_confirmation({"30M": signal_df, "1H": h1_df, "1D": d1_df})
    tech = technical_score(signal_df, mtf=mtf)
    news = await fetch_finnhub_news(asset_key)
    news_score = news_lexicon_score(news, asset_key)
    event_risk = economic_event_filter(news, asset_key)
    ai = await openai_sentiment(asset_key, news, tech, {"mtf": mtf, "event_risk": event_risk})

    # First fusion before tracker, then update tracker and finalize confidence.
    fusion_pre = fusion_signal(tech, news_score, ai, mtf=mtf, event_risk=event_risk)
    tracker = update_signal_tracker(asset_key, fusion_pre["signal"], tech["price"], fusion_pre["confidence"])
    fusion = fusion_signal(tech, news_score, ai, mtf=mtf, event_risk=event_risk, tracker=tracker)

    chart = build_chart(chart_df, ASSETS[asset_key]["name"])

    return {
        "asset_key": asset_key,
        "asset": ASSETS[asset_key],
        "tf": tf,
        "signal_tf": "30M",
        "tech": tech,
        "news": news,
        "news_score": round(news_score, 2),
        "ai": ai,
        "fusion": fusion,
        "chart": chart,
        "institutional": {"mtf": mtf, "event_risk": event_risk, "tracker": tracker},
        "updated": datetime.now().strftime("%H:%M:%S"),
    }


@app.get("/api/hit")
async def api_hit():
    with HIT_LOCK:
        hits = read_hit_count() + 1
        write_hit_count(hits)
    return {"hits": hits}


@app.get("/api/hits")
async def api_hits():
    return {"hits": read_hit_count()}


@app.get("/api/signal")
async def api_signal(asset: str = Query("WTI"), tf: str = Query("1M")):
    asset = asset.upper()
    tf = tf.upper()
    if asset not in ASSETS:
        return JSONResponse({"error": "Invalid asset"}, status_code=400)
    if tf not in INTERVALS:
        return JSONResponse({"error": "Invalid timeframe"}, status_code=400)
    try:
        return await process(asset, tf)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=200)

HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Commodity Sentiment Terminal - By Abbas</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
:root{
  --bg:#050812; --panel:#0b1220; --panel2:#0f1a2e; --line:rgba(148,163,184,.20);
  --text:#eaf2ff; --muted:#94a3b8; --green:#22c55e; --red:#ef4444; --amber:#f59e0b;
  --cyan:#38bdf8; --blue:#2563eb; --purple:#8b5cf6;
}
*{box-sizing:border-box} html,body{margin:0;min-height:100%;background:var(--bg);color:var(--text);font-family:Inter,Segoe UI,Arial,sans-serif}
body{overflow:hidden}.shell{width:100vw;height:100vh;padding:12px;background:
 radial-gradient(circle at 20% -10%,rgba(56,189,248,.24),transparent 32%),radial-gradient(circle at 90% 0%,rgba(139,92,246,.20),transparent 30%),linear-gradient(180deg,#050812,#06101e 70%,#030711)}
.topbar{height:74px;border:1px solid var(--line);border-radius:22px;background:linear-gradient(135deg,rgba(15,23,42,.88),rgba(15,23,42,.55));backdrop-filter:blur(16px);display:flex;align-items:center;gap:14px;padding:12px 16px;box-shadow:0 22px 70px rgba(0,0,0,.35)}
.brandmark{width:48px;height:48px;border-radius:16px;background:linear-gradient(135deg,var(--cyan),var(--blue));display:grid;place-items:center;font-size:24px;box-shadow:0 0 40px rgba(56,189,248,.28)}
.brand h1{font-size:20px;margin:0;letter-spacing:.3px}.brand p{margin:3px 0 0;color:var(--muted);font-size:12px}.pill{border:1px solid var(--line);background:rgba(2,6,23,.35);border-radius:999px;padding:9px 12px;font-size:12px;color:var(--muted)}
.spacer{flex:1}.hitbox{text-align:right;min-width:135px}.hitbox span{display:block;font-size:11px;color:var(--muted);font-weight:800;letter-spacing:.12em}.hitbox b{font-size:24px;color:var(--green)}
.progress{height:4px;background:rgba(148,163,184,.17);border-radius:99px;overflow:hidden;margin:10px 3px 0}.progress div{height:100%;width:0;background:linear-gradient(90deg,var(--green),var(--cyan));animation:bar 30s linear infinite}@keyframes bar{from{width:0}to{width:100%}}
.controls{height:74px;display:grid;grid-template-columns:1.2fr .8fr 1fr;gap:10px;margin-top:10px}.control,.action{border:1px solid var(--line);border-radius:18px;background:rgba(15,23,42,.78);padding:10px 12px;min-width:0}.control label{display:block;font-size:11px;color:var(--muted);font-weight:800;margin-bottom:6px;text-transform:uppercase;letter-spacing:.09em}select{width:100%;background:transparent;border:0;color:var(--text);font-weight:900;font-size:15px;outline:0}option{background:#0b1220}.action{color:#fff;font-weight:900;font-size:13px;cursor:pointer}.run{background:linear-gradient(135deg,#ef4444,#f97316);border:0}.download{background:linear-gradient(135deg,#1e293b,#334155)}
#error{height:18px;color:#fb7185;font-size:12px;font-weight:900;padding:3px 8px}.dashboard{height:calc(100vh - 272px);display:grid;grid-template-columns:310px 1fr 340px;grid-template-rows:150px minmax(260px,1fr) 150px;gap:10px}.card{border:1px solid var(--line);border-radius:22px;background:linear-gradient(180deg,rgba(15,23,42,.88),rgba(15,23,42,.62));box-shadow:0 20px 55px rgba(0,0,0,.22);overflow:hidden;position:relative}.card:before{content:"";position:absolute;inset:0;background:linear-gradient(120deg,rgba(255,255,255,.08),transparent 30%);pointer-events:none}.card-head{height:42px;display:flex;align-items:center;justify-content:space-between;padding:12px 14px;border-bottom:1px solid rgba(148,163,184,.13);font-weight:900;font-size:13px;letter-spacing:.05em;text-transform:uppercase}.muted{color:var(--muted)}
.signal{grid-column:1;grid-row:1 / 3;padding:14px;display:flex;flex-direction:column}.asset-badge{display:flex;align-items:center;gap:10px;margin-bottom:10px}.asset-icon{width:44px;height:44px;border-radius:15px;background:rgba(56,189,248,.12);display:grid;place-items:center;font-size:24px}.asset-name{font-weight:950;font-size:20px}.top-price{font-size:30px;font-weight:1000;color:var(--red);line-height:1;margin-top:4px}.updated{font-size:12px;color:var(--muted)}.signal-main{border-radius:22px;background:radial-gradient(circle at 50% 0%,rgba(34,197,94,.25),transparent 45%),rgba(2,6,23,.40);padding:18px;text-align:center;margin:5px 0 12px}.signal-text{font-size:30px;font-weight:1000;letter-spacing:.5px}.signal-sub{color:var(--muted);font-size:12px;margin-top:4px}.gauge-wrap{display:grid;place-items:center;margin:6px 0 12px}.gauge{width:230px;height:115px;border-radius:230px 230px 0 0;background:conic-gradient(from 180deg,var(--red),var(--amber),#eab308,var(--green),var(--green));position:relative;overflow:hidden}.gauge:after{content:"";position:absolute;left:26px;top:26px;width:178px;height:89px;border-radius:178px 178px 0 0;background:#0b1220}.needle{width:4px;height:92px;position:absolute;left:113px;bottom:0;background:#fff;z-index:2;transform-origin:bottom center;transition:.45s ease}.confidence{font-size:44px;font-weight:1000;color:var(--green);margin-top:-45px;z-index:3}.mini-grid{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-top:auto}.mini{border:1px solid rgba(148,163,184,.16);border-radius:16px;background:rgba(2,6,23,.28);padding:10px}.mini span{display:block;color:var(--muted);font-size:11px;font-weight:800}.mini b{font-size:16px;margin-top:4px;display:block}.chart{grid-column:2;grid-row:1 / 3;padding:0;overflow:hidden}.chart #chart{width:100%;height:calc(100% - 42px);min-height:430px;padding:6px 10px 10px 8px}.ai{grid-column:3;grid-row:1;padding:0}.ai-body{padding:13px 14px;font-size:13px;line-height:1.45}.ai-body p{margin:0;color:#dbeafe}.ai-tags{display:flex;gap:7px;margin-top:10px;flex-wrap:wrap}.tag{border:1px solid rgba(148,163,184,.18);border-radius:999px;padding:6px 9px;font-size:11px;font-weight:900;background:rgba(2,6,23,.35)}.drivers{grid-column:3;grid-row:2;padding:0}.driver-list{padding:12px 14px}.driver{margin:13px 0}.driver-top{display:flex;justify-content:space-between;font-size:12px;font-weight:900;margin-bottom:7px}.track{height:10px;border-radius:99px;background:rgba(148,163,184,.15);overflow:hidden}.fill{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--green),var(--cyan));transition:.35s}.fill.neg{background:linear-gradient(90deg,var(--red),var(--amber))}.stats{grid-column:1 / 3;grid-row:3;display:grid;grid-template-columns:repeat(6,1fr);gap:10px;background:transparent;border:0;box-shadow:none}.stat-card{border:1px solid var(--line);border-radius:20px;background:rgba(15,23,42,.75);padding:13px;min-width:0}.stat-card span{display:block;color:var(--muted);font-size:11px;font-weight:900}.stat-card b{display:block;font-size:18px;margin-top:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.news{grid-column:3;grid-row:3;padding:0}.news-list{height:calc(100% - 42px);overflow:auto;padding:10px 12px}.news-item{border-left:3px solid var(--green);padding:8px 8px 8px 10px;background:rgba(2,6,23,.30);border-radius:11px;margin-bottom:8px}.news-item a{color:var(--text);font-size:12px;font-weight:850;text-decoration:none;line-height:1.35;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}.bottom-actions{height:58px;margin-top:10px;display:grid;grid-template-columns:1fr;gap:10px}.bottom-actions .action{height:58px;border-radius:18px}.loading{opacity:.55;filter:saturate(.55)}
@media(max-width:1100px){body{overflow:auto}.shell{height:auto;min-height:100vh}.controls{height:auto;grid-template-columns:1fr 1fr}.bottom-actions{height:auto;grid-template-columns:1fr}.dashboard{height:auto;grid-template-columns:1fr;grid-template-rows:auto}.signal,.chart,.ai,.drivers,.stats,.news{grid-column:1;grid-row:auto}.chart{height:470px}.stats{grid-template-columns:repeat(2,1fr)}.topbar{height:auto;flex-wrap:wrap}.spacer{display:none}}
@media(max-width:640px){.shell{padding:8px}.brand h1{font-size:16px}.controls{grid-template-columns:1fr}.bottom-actions{grid-template-columns:1fr}.stats{grid-template-columns:1fr}.signal-text{font-size:24px}.hitbox{text-align:left}}
</style>
</head>
<body>
<div class="shell">
  <header class="topbar">
    <div class="brandmark">📊</div><div class="brand"><h1>Commodity Sentiment Terminal <span style="color:#f87171">By Abbas</span></h1><p>AI + Technicals + Market News Fusion Engine</p></div>
    <div class="pill" id="livePill">● Live feed</div><div class="pill">Auto refresh: 30s</div><div class="spacer"></div>
    <div class="hitbox"><span>TOTAL HITS</span><b id="hitCounter">0</b></div>
  </header>
  <div class="progress"><div id="refreshBar"></div></div>
  <section class="controls">
    <div class="control"><label>Asset</label><select id="assetSelect"></select></div>
    <div class="control"><label>Timeframe</label><select id="tfSelect"></select></div>
    <div class="control"><label>Status</label><select disabled><option id="statusText">Ready</option></select></div>
  </section>
  <div id="error"></div>
  <main class="dashboard" id="dashboard">
    <section class="card signal">
      <div class="asset-badge"><div class="asset-icon" id="assetIcon">🛢️</div><div><div class="asset-name" id="assetName">WTI</div><div class="top-price" id="topPrice">--</div><div class="updated" id="updated">Updated --</div></div></div>
      <div class="signal-main"><div class="signal-text" id="signalText">LOADING</div><div class="signal-sub" id="signalSub">Waiting for fusion engine</div></div>
      <div class="gauge-wrap"><div class="gauge"><div class="needle" id="needle"></div></div><div class="confidence" id="meterValue">--</div></div>
      <div class="mini-grid">
        <div class="mini"><span>FUSION SCORE</span><b id="fusionScore">--</b></div><div class="mini"><span>AI BIAS</span><b id="aiBiasMini">--</b></div>
        <div class="mini"><span>ENTRY</span><b id="entryVal">--</b></div><div class="mini"><span>RISK NOTE</span><b id="riskMini">--</b></div>
      </div>
    </section>
    <section class="card chart"><div class="card-head"><span>Price Action Chart</span><span class="muted" id="chartLabel">Candles + EMA 9/21/50</span></div><div id="chart"></div></section>
    <section class="card ai"><div class="card-head"><span>AI Sentiment</span><span class="muted" id="aiScore">--</span></div><div class="ai-body"><p id="aiSummary">Loading market narrative...</p><div class="ai-tags"><span class="tag" id="biasTag">BIAS --</span><span class="tag" id="confTag">CONF --</span></div></div></section>
    <section class="card drivers"><div class="card-head"><span>Signal Drivers</span><span class="muted">weighted</span></div><div class="driver-list" id="drivers"></div></section>
    <section class="stats" id="techStats"></section>
    <section class="card news"><div class="card-head"><span>Flash News Stream</span><span class="muted" id="newsCount">0 items</span></div><div class="news-list" id="news"></div></section>
  </main>
  <section class="bottom-actions">
    <button class="action download" onclick="window.print()">DOWNLOAD REPORT</button>
  </section>
</div>
<script>
const ASSETS={GOLD:["🟡","Gold Spot"],SILVER:["⚪","Silver"],WTI:["🛢️","Crude Oil WTI"],BRENT:["🛢️","Brent Crude"],BTC:["₿","Bitcoin"],USTEC100:["📈","USTEC 100 Future"]};
const TFS=["1M","15M","30M","1H","1D"]; let asset="WTI",tf="1M",busy=false;
async function updateHits(){
  try{
    const r = await fetch(`/api/hit?_=${Date.now()}`);
    const d = await r.json();
    hitCounter.innerText = Number(d.hits || 0).toLocaleString();
  }catch(e){
    hitCounter.innerText = "--";
  }
}
function init(){Object.keys(ASSETS).forEach(k=>assetSelect.innerHTML+=`<option value="${k}">${ASSETS[k][0]} ${ASSETS[k][1]}</option>`);TFS.forEach(t=>tfSelect.innerHTML+=`<option value="${t}">${t}</option>`);assetSelect.value=asset;tfSelect.value=tf;assetSelect.onchange=()=>{asset=assetSelect.value;loadData(true)};tfSelect.onchange=()=>{tf=tfSelect.value;loadData(true)};updateHits();loadData(true);setInterval(()=>loadData(false),30000)}
function colorFor(label){if(label==="BULLISH")return "#22c55e"; if(label==="BEARISH")return "#ef4444"; return "#f59e0b"}
function setLoading(v){busy=v;dashboard.classList.toggle('loading',v);statusText.innerText=v?'Loading':'Ready';livePill.innerText=v?'● Updating':'● Live feed'}
async function loadData(manual=false){if(busy&&!manual)return;setLoading(true);try{error.innerText="";refreshBar.style.animation='none';void refreshBar.offsetWidth;refreshBar.style.animation='bar 30s linear infinite';const r=await fetch(`/api/signal?asset=${asset}&tf=${tf}&_=${Date.now()}`);const d=await r.json();if(d.error)throw new Error(d.error);render(d)}catch(e){error.innerText="Error: "+e.message}finally{setLoading(false)}}
function render(d){const label=d.fusion.label||'NEUTRAL', sig=d.fusion.signal||'HOLD / WAIT', c=Number(d.fusion.confidence||0), fusion=Number(d.fusion.fusion||0), col=colorFor(label);assetIcon.innerText=d.asset.icon;assetName.innerText=d.asset.name;topPrice.innerText=d.tech.price;updated.innerText=`Updated ${d.updated} • Chart ${d.tf} • Signal ${d.signal_tf||"30M"}`;signalText.innerText=sig;signalText.style.color=col;signalSub.innerText=`${label} setup from technical + news + AI fusion`;meterValue.innerText=c.toFixed(1);meterValue.style.color=col;needle.style.transform=`rotate(${(c/100*180)-90}deg)`;fusionScore.innerText=(fusion>0?'+':'')+fusion.toFixed(2);aiBiasMini.innerText=d.ai.bias||label;entryVal.innerText=d.tech.entry;riskMini.innerText=(d.ai.risk||'Normal').slice(0,18);aiSummary.innerText=d.ai.summary||'No AI summary returned.';aiScore.innerText=`AI ${Number(d.ai.score||0).toFixed(0)}`;biasTag.innerText='BIAS '+label;biasTag.style.color=col;confTag.innerText='CONF '+c.toFixed(1)+'%';chartLabel.innerText=`${d.asset.name} • Chart ${d.tf} • Signal fixed ${d.signal_tf||"30M"} • Candles + EMA 9/21/50`;
techStats.innerHTML=[['PRICE',d.tech.price],['RSI 14',d.tech.rsi],['MACD',d.tech.macd],['ATR 14',d.tech.atr],['TARGET',d.tech.target],['STOP LOSS',d.tech.stop]].map(([a,b])=>`<div class="stat-card"><span>${a}</span><b>${b}</b></div>`).join('');
let layout=d.chart.layout||{};layout.autosize=true;layout.height=null;layout.margin={l:58,r:34,t:22,b:46};layout.paper_bgcolor='rgba(0,0,0,0)';layout.plot_bgcolor='rgba(2,6,23,.34)';layout.font={color:'#eaf2ff',size:11};layout.legend={orientation:'h',y:1.04,x:0,font:{size:10}};layout.xaxis={...(layout.xaxis||{}),type:'date',rangeslider:{visible:false},gridcolor:'rgba(148,163,184,.10)',automargin:true};layout.yaxis={...(layout.yaxis||{}),gridcolor:'rgba(148,163,184,.10)',automargin:true,zeroline:false};Plotly.react('chart',d.chart.data,layout,{displayModeBar:false,responsive:true});setTimeout(()=>Plotly.Plots.resize('chart'),180);
const rows=[['Technical Analysis',Math.round(d.fusion.tech_percent-50)],['News Sentiment',Math.round(d.fusion.news_percent-50)],['Fusion Momentum',Math.round(d.fusion.fusion)],['OpenAI Sentiment',Math.round(d.fusion.ai_percent-50)],['Risk Adjustment',label==='NEUTRAL'?-8:14]];drivers.innerHTML=rows.map(([n,v])=>`<div class="driver"><div class="driver-top"><span>${n}</span><span style="color:${v<0?'#ef4444':'#22c55e'}">${v>0?'+':''}${v}</span></div><div class="track"><div class="fill ${v<0?'neg':''}" style="width:${Math.min(100,Math.abs(v)*2)}%"></div></div></div>`).join('');
newsCount.innerText=(d.news||[]).length+' items';news.innerHTML=(d.news||[]).slice(0,8).map(n=>`<div class="news-item"><a href="${n.url||'#'}" target="_blank">[${n.source||'News'}] ${n.headline||''}</a></div>`).join('')||'<div class="news-item"><a>No matching news returned.</a></div>'}
window.addEventListener('resize',()=>{try{Plotly.Plots.resize('chart')}catch(e){}});init();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(HTML)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
    )
