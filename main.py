import os, json, re, asyncio, threading
from datetime import datetime
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


FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

app = FastAPI(title="Commodity Sentiment Terminal - By Abbas")
executor = ThreadPoolExecutor(max_workers=8)

# ============================================================
# AZURE GLOBAL HIT COUNTER
# ============================================================
# Uses Azure App Service persistent /home storage when available.
# For local development it falls back to the current app folder.
HIT_COUNTER_FILE = Path(os.getenv("HIT_COUNTER_FILE", "/home/site/hit_counter.json"))
if not HIT_COUNTER_FILE.parent.exists():
    HIT_COUNTER_FILE = Path("hit_counter.json")

HIT_LOCK = threading.Lock()


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
    "GOLD": {
        "name": "Gold",
        "icon": "🟡",
        "yf": "GC=F",
        "keywords": ["gold", "xau", "bullion", "safe haven", "inflation", "fed", "rates", "dollar"],
    },
    "SILVER": {
        "name": "Silver",
        "icon": "⚪",
        "yf": "SI=F",
        "keywords": ["silver", "xag", "precious metal", "industrial metal", "solar", "dollar", "rates"],
    },
    "WTI": {
        "name": "Crude Oil WTI",
        "icon": "🛢️",
        "yf": "CL=F",
        "keywords": ["wti", "crude", "oil", "opec", "iran", "gulf", "hormuz", "sanctions", "us strikes", "middle east"],
    },
    "BRENT": {
        "name": "Brent Crude",
        "icon": "🛢️",
        "yf": "BZ=F",
        "keywords": ["brent", "crude", "oil", "opec", "iran", "gulf", "hormuz", "sanctions", "shipping"],
    },
    "BTC": {
        "name": "Bitcoin",
        "icon": "₿",
        "yf": "BTC-USD",
        "keywords": ["bitcoin", "btc", "crypto", "etf", "risk assets", "liquidity", "fed", "rates"],
    },
    "USTEC100": {
        "name": "USTEC 100 Future",
        "icon": "📈",
        "yf": "NQ=F",
        "keywords": ["nasdaq", "nasdaq 100", "nq futures", "tech stocks", "ai stocks", "fed", "rates", "yields"],
    },
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


def load_candles(asset_key, tf):
    meta = ASSETS[asset_key]
    cfg = INTERVALS[tf]

    df = yf.download(
        meta["yf"],
        period=cfg["period"],
        interval=cfg["yf"],
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns=str.title).dropna().reset_index()
    time_col = "Datetime" if "Datetime" in df.columns else "Date"

    df["time"] = pd.to_datetime(df[time_col])
    df = df[["time", "Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["time", "open", "high", "low", "close", "volume"]

    return df.tail(500)


def add_indicators(df):
    d = df.copy()

    d["ema9"] = d["close"].ewm(span=9, adjust=False).mean()
    d["ema21"] = d["close"].ewm(span=21, adjust=False).mean()
    d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()
    d["ema200"] = d["close"].ewm(span=200, adjust=False).mean()

    delta = d["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["rsi"] = 100 - (100 / (1 + rs))
    d["rsi"] = d["rsi"].fillna(50)

    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()

    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()

    tr1 = d["high"] - d["low"]
    tr2 = (d["high"] - d["close"].shift()).abs()
    tr3 = (d["low"] - d["close"].shift()).abs()

    d["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    d["atr"] = d["tr"].rolling(14).mean().fillna(d["tr"].mean())

    d["vol_ma"] = d["volume"].rolling(20).mean()

    return d


def technical_score(df):
    d = add_indicators(df)
    last = d.iloc[-1]

    score = 0

    score += 15 if last.ema9 > last.ema21 else -15
    score += 15 if last.ema21 > last.ema50 else -15
    score += 15 if last.close > last.ema200 else -15
    score += 15 if last.macd > last.macd_signal else -15

    if 50 < last.rsi < 70:
        score += 10
    elif 30 < last.rsi <= 50:
        score -= 5
    elif last.rsi >= 70:
        score -= 8
    elif last.rsi <= 30:
        score += 8

    if len(d) > 20:
        momentum = (last.close - d["close"].iloc[-10]) / d["close"].iloc[-10] * 100
        score += clamp(momentum * 3, -15, 15)

    price = safe_float(last.close)
    atr = safe_float(last.atr, price * 0.01)

    score = clamp(score)
    direction = 1 if score >= 0 else -1

    return {
        "score": round(score, 2),
        "price": round(price, 4),
        "rsi": round(safe_float(last.rsi), 2),
        "macd": round(safe_float(last.macd), 4),
        "atr": round(atr, 4),
        "entry": round(price, 4),
        "target": round(price + direction * atr * 2.2, 4),
        "stop": round(price - direction * atr * 1.25, 4),
    }


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
            out.append({
                "headline": headline,
                "summary": summary[:180],
                "source": source,
                "url": n.get("url", ""),
                "relevance": relevance,
            })

    return sorted(out, key=lambda x: x["relevance"], reverse=True)[:10]


def news_lexicon_score(news, asset_key):
    if not news:
        return 0

    text = " ".join([n["headline"] + " " + n["summary"] for n in news]).lower()

    bullish = [
        "rally", "surge", "gain", "rise", "rebound", "strong demand",
        "supply cut", "disruption", "shortage", "drawdown", "inventory draw",
        "sanctions", "strike", "hormuz", "shipping risk", "safe haven",
        "rate cut", "weak dollar"
    ]

    bearish = [
        "drop", "fall", "slump", "selloff", "weak demand",
        "inventory build", "surplus", "oversupply", "recession",
        "strong dollar", "rate hike", "peace deal", "ceasefire",
        "supply restored"
    ]

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


async def openai_sentiment(asset_key, news, tech):
    if not OPENAI_API_KEY:
        return {
            "score": 0,
            "bias": "NEUTRAL",
            "summary": "OpenAI key missing. Using fallback sentiment.",
            "risk": "AI sentiment unavailable.",
        }

    headlines = [
        f"{n['source']}: {n['headline']} - {n['summary']}"
        for n in news[:8]
    ]

    prompt = f"""
Asset: {ASSETS[asset_key]["name"]}

Technical:
Price={tech["price"]}
RSI={tech["rsi"]}
MACD={tech["macd"]}
Technical Score={tech["score"]}

News:
{chr(10).join(headlines)}

Return only JSON:
{{
"score": number between -100 and 100,
"bias": "BULLISH" or "BEARISH" or "NEUTRAL",
"summary": "short reason",
"risk": "short risk"
}}
"""

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        data = json.loads(response.choices[0].message.content.strip())

        return {
            "score": clamp(data.get("score", 0)),
            "bias": data.get("bias", "NEUTRAL"),
            "summary": data.get("summary", ""),
            "risk": data.get("risk", ""),
        }

    except Exception as e:
        return {
            "score": 0,
            "bias": "NEUTRAL",
            "summary": "OpenAI sentiment failed. Fallback active.",
            "risk": str(e)[:100],
        }


def fusion_signal(tech, news_score, ai):
    tech_s = clamp(tech["score"])
    news_s = clamp(news_score)
    ai_s = clamp(ai["score"])

    fusion = clamp((tech_s * 0.45) + (news_s * 0.25) + (ai_s * 0.30))
    confidence = min(100, abs(fusion) * 1.15 + 20)

    if fusion >= 35:
        signal = "BUY SIGNAL"
        label = "BULLISH"
    elif fusion <= -35:
        signal = "SELL SIGNAL"
        label = "BEARISH"
    else:
        signal = "HOLD / WAIT"
        label = "NEUTRAL"

    return {
        "fusion": round(fusion, 2),
        "confidence": round(confidence, 1),
        "signal": signal,
        "label": label,
        "tech_percent": round((tech_s + 100) / 2, 1),
        "news_percent": round((news_s + 100) / 2, 1),
        "ai_percent": round((ai_s + 100) / 2, 1),
    }


def build_chart(df, asset_name):
    d = add_indicators(df)

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=d["time"].astype(str),
        open=d["open"],
        high=d["high"],
        low=d["low"],
        close=d["close"],
        name=asset_name,
    ))

    fig.add_trace(go.Scatter(x=d["time"].astype(str), y=d["ema9"], name="EMA9", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=d["time"].astype(str), y=d["ema21"], name="EMA21", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=d["time"].astype(str), y=d["ema50"], name="EMA50", line=dict(width=1)))

    fig.update_layout(
        template="plotly_dark",
        height=330,
        margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor="#111a26",
        plot_bgcolor="#111a26",
        font=dict(color="#dce7f3", size=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9)),
    )

    return json.loads(fig.to_json())


async def process(asset_key, tf):
    loop = asyncio.get_running_loop()

    df = await loop.run_in_executor(executor, load_candles, asset_key, tf)

    if df.empty:
        raise RuntimeError("No candle data returned.")

    tech = technical_score(df)
    news = await fetch_finnhub_news(asset_key)
    news_score = news_lexicon_score(news, asset_key)
    ai = await openai_sentiment(asset_key, news, tech)
    fusion = fusion_signal(tech, news_score, ai)
    chart = build_chart(df, ASSETS[asset_key]["name"])

    return {
        "asset_key": asset_key,
        "asset": ASSETS[asset_key],
        "tf": tf,
        "tech": tech,
        "news": news,
        "news_score": round(news_score, 2),
        "ai": ai,
        "fusion": fusion,
        "chart": chart,
        "updated": datetime.now().strftime("%H:%M:%S"),
    }


@app.get("/api/hit")
async def api_hit():
    """
    Global hit counter for Azure Web App deployment.
    Every page load calls this endpoint once.
    Shared across users/browsers/devices.
    Persists in /home/site/hit_counter.json on Azure App Service.
    """
    with HIT_LOCK:
        hits = read_hit_count() + 1
        write_hit_count(hits)

    return {"hits": hits}


@app.get("/api/hits")
async def api_hits():
    """
    Read current hit count without incrementing.
    Useful for diagnostics.
    """
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
        return JSONResponse({"error": str(e)}, status_code=500)


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
.controls{height:74px;display:grid;grid-template-columns:1.2fr .8fr .9fr .9fr 1fr;gap:10px;margin-top:10px}.control,.action{border:1px solid var(--line);border-radius:18px;background:rgba(15,23,42,.78);padding:10px 12px;min-width:0}.control label{display:block;font-size:11px;color:var(--muted);font-weight:800;margin-bottom:6px;text-transform:uppercase;letter-spacing:.09em}select{width:100%;background:transparent;border:0;color:var(--text);font-weight:900;font-size:15px;outline:0}option{background:#0b1220}.action{color:#fff;font-weight:900;font-size:13px;cursor:pointer}.run{background:linear-gradient(135deg,#ef4444,#f97316);border:0}.download{background:linear-gradient(135deg,#1e293b,#334155)}
#error{height:18px;color:#fb7185;font-size:12px;font-weight:900;padding:3px 8px}.dashboard{height:calc(100vh - 202px);display:grid;grid-template-columns:310px 1fr 340px;grid-template-rows:150px minmax(260px,1fr) 150px;gap:10px}.card{border:1px solid var(--line);border-radius:22px;background:linear-gradient(180deg,rgba(15,23,42,.88),rgba(15,23,42,.62));box-shadow:0 20px 55px rgba(0,0,0,.22);overflow:hidden;position:relative}.card:before{content:"";position:absolute;inset:0;background:linear-gradient(120deg,rgba(255,255,255,.08),transparent 30%);pointer-events:none}.card-head{height:42px;display:flex;align-items:center;justify-content:space-between;padding:12px 14px;border-bottom:1px solid rgba(148,163,184,.13);font-weight:900;font-size:13px;letter-spacing:.05em;text-transform:uppercase}.muted{color:var(--muted)}
.signal{grid-column:1;grid-row:1 / 3;padding:14px;display:flex;flex-direction:column}.asset-badge{display:flex;align-items:center;gap:10px;margin-bottom:10px}.asset-icon{width:44px;height:44px;border-radius:15px;background:rgba(56,189,248,.12);display:grid;place-items:center;font-size:24px}.asset-name{font-weight:950;font-size:20px}.updated{font-size:12px;color:var(--muted)}.signal-main{border-radius:22px;background:radial-gradient(circle at 50% 0%,rgba(34,197,94,.25),transparent 45%),rgba(2,6,23,.40);padding:18px;text-align:center;margin:5px 0 12px}.signal-text{font-size:30px;font-weight:1000;letter-spacing:.5px}.signal-sub{color:var(--muted);font-size:12px;margin-top:4px}.gauge-wrap{display:grid;place-items:center;margin:6px 0 12px}.gauge{width:230px;height:115px;border-radius:230px 230px 0 0;background:conic-gradient(from 180deg,var(--red),var(--amber),#eab308,var(--green),var(--green));position:relative;overflow:hidden}.gauge:after{content:"";position:absolute;left:26px;top:26px;width:178px;height:89px;border-radius:178px 178px 0 0;background:#0b1220}.needle{width:4px;height:92px;position:absolute;left:113px;bottom:0;background:#fff;z-index:2;transform-origin:bottom center;transition:.45s ease}.confidence{font-size:44px;font-weight:1000;color:var(--green);margin-top:-45px;z-index:3}.mini-grid{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-top:auto}.mini{border:1px solid rgba(148,163,184,.16);border-radius:16px;background:rgba(2,6,23,.28);padding:10px}.mini span{display:block;color:var(--muted);font-size:11px;font-weight:800}.mini b{font-size:16px;margin-top:4px;display:block}.chart{grid-column:2;grid-row:1 / 3;padding:0}.chart #chart{width:100%;height:calc(100% - 42px);min-height:430px}.ai{grid-column:3;grid-row:1;padding:0}.ai-body{padding:13px 14px;font-size:13px;line-height:1.45}.ai-body p{margin:0;color:#dbeafe}.ai-tags{display:flex;gap:7px;margin-top:10px;flex-wrap:wrap}.tag{border:1px solid rgba(148,163,184,.18);border-radius:999px;padding:6px 9px;font-size:11px;font-weight:900;background:rgba(2,6,23,.35)}.drivers{grid-column:3;grid-row:2;padding:0}.driver-list{padding:12px 14px}.driver{margin:13px 0}.driver-top{display:flex;justify-content:space-between;font-size:12px;font-weight:900;margin-bottom:7px}.track{height:10px;border-radius:99px;background:rgba(148,163,184,.15);overflow:hidden}.fill{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--green),var(--cyan));transition:.35s}.fill.neg{background:linear-gradient(90deg,var(--red),var(--amber))}.stats{grid-column:1 / 3;grid-row:3;display:grid;grid-template-columns:repeat(6,1fr);gap:10px;background:transparent;border:0;box-shadow:none}.stat-card{border:1px solid var(--line);border-radius:20px;background:rgba(15,23,42,.75);padding:13px;min-width:0}.stat-card span{display:block;color:var(--muted);font-size:11px;font-weight:900}.stat-card b{display:block;font-size:18px;margin-top:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.news{grid-column:3;grid-row:3;padding:0}.news-list{height:calc(100% - 42px);overflow:auto;padding:10px 12px}.news-item{border-left:3px solid var(--green);padding:8px 8px 8px 10px;background:rgba(2,6,23,.30);border-radius:11px;margin-bottom:8px}.news-item a{color:var(--text);font-size:12px;font-weight:850;text-decoration:none;line-height:1.35;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}.loading{opacity:.55;filter:saturate(.55)}
@media(max-width:1100px){body{overflow:auto}.shell{height:auto;min-height:100vh}.controls{height:auto;grid-template-columns:1fr 1fr}.dashboard{height:auto;grid-template-columns:1fr;grid-template-rows:auto}.signal,.chart,.ai,.drivers,.stats,.news{grid-column:1;grid-row:auto}.chart{height:470px}.stats{grid-template-columns:repeat(2,1fr)}.topbar{height:auto;flex-wrap:wrap}.spacer{display:none}}
@media(max-width:640px){.shell{padding:8px}.brand h1{font-size:16px}.controls{grid-template-columns:1fr}.stats{grid-template-columns:1fr}.signal-text{font-size:24px}.hitbox{text-align:left}}
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
    <button class="action run" onclick="loadData(true)">RUN ANALYSIS</button>
    <button class="action download" onclick="window.print()">DOWNLOAD REPORT</button>
    <div class="control"><label>Status</label><select disabled><option id="statusText">Ready</option></select></div>
  </section>
  <div id="error"></div>
  <main class="dashboard" id="dashboard">
    <section class="card signal">
      <div class="asset-badge"><div class="asset-icon" id="assetIcon">🛢️</div><div><div class="asset-name" id="assetName">WTI</div><div class="updated" id="updated">Updated --</div></div></div>
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
</div>
<script>
const ASSETS={GOLD:["🟡","Gold"],SILVER:["⚪","Silver"],WTI:["🛢️","Crude Oil WTI"],BRENT:["🛢️","Brent Crude"],BTC:["₿","Bitcoin"],USTEC100:["📈","USTEC 100 Future"]};
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
function render(d){const label=d.fusion.label||'NEUTRAL', sig=d.fusion.signal||'HOLD / WAIT', c=Number(d.fusion.confidence||0), fusion=Number(d.fusion.fusion||0), col=colorFor(label);assetIcon.innerText=d.asset.icon;assetName.innerText=d.asset.name;updated.innerText=`Updated ${d.updated} • ${d.tf}`;signalText.innerText=sig;signalText.style.color=col;signalSub.innerText=`${label} setup from technical + news + AI fusion`;meterValue.innerText=c.toFixed(1);meterValue.style.color=col;needle.style.transform=`rotate(${(c/100*180)-90}deg)`;fusionScore.innerText=(fusion>0?'+':'')+fusion.toFixed(2);aiBiasMini.innerText=d.ai.bias||label;entryVal.innerText=d.tech.entry;riskMini.innerText=(d.ai.risk||'Normal').slice(0,18);aiSummary.innerText=d.ai.summary||'No AI summary returned.';aiScore.innerText=`AI ${Number(d.ai.score||0).toFixed(0)}`;biasTag.innerText='BIAS '+label;biasTag.style.color=col;confTag.innerText='CONF '+c.toFixed(1)+'%';chartLabel.innerText=`${d.asset.name} • ${d.tf} • Candles + EMA 9/21/50`;
techStats.innerHTML=[['PRICE',d.tech.price],['RSI 14',d.tech.rsi],['MACD',d.tech.macd],['ATR 14',d.tech.atr],['TARGET',d.tech.target],['STOP LOSS',d.tech.stop]].map(([a,b])=>`<div class="stat-card"><span>${a}</span><b>${b}</b></div>`).join('');
let layout=d.chart.layout||{};layout.autosize=true;layout.height=null;layout.margin={l:42,r:18,t:12,b:34};layout.paper_bgcolor='rgba(0,0,0,0)';layout.plot_bgcolor='rgba(2,6,23,.34)';layout.font={color:'#eaf2ff',size:11};layout.legend={orientation:'h',y:1.04,x:0,font:{size:10}};layout.xaxis={...(layout.xaxis||{}),rangeslider:{visible:false},gridcolor:'rgba(148,163,184,.10)'};layout.yaxis={...(layout.yaxis||{}),gridcolor:'rgba(148,163,184,.10)'};Plotly.react('chart',d.chart.data,layout,{displayModeBar:false,responsive:true});setTimeout(()=>Plotly.Plots.resize('chart'),180);
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
