import json
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import mean
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen


HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
INTERVAL_FALLBACKS = {
    "1d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"],
    "5d": ["5m", "15m", "30m", "60m", "1h", "1d"],
    "1mo": ["5m", "15m", "30m", "60m", "1h", "1d"],
    "3mo": ["1h", "1d", "1wk"],
    "6mo": ["1h", "1d", "1wk"],
    "1y": ["1d", "1wk", "1mo"],
    "2y": ["1d", "1wk", "1mo"],
    "5y": ["1wk", "1mo", "3mo"],
    "10y": ["1wk", "1mo", "3mo"],
    "ytd": ["1d", "1wk", "1mo"],
    "max": ["1wk", "1mo", "3mo"],
}
CANADIAN_STOCKS = [
    "BTB-UN.TO", "CASH.TO", "CCO.TO", "CJ.TO", "CLS.TO",
    "CSU.TO", "DFN.TO", "DOL.TO", "ENB.TO", "FFU.NE",
    "FTN.TO", "GIL.TO", "HDIV.TO", "HHIS.TO",
    "KILO.TO", "KITS.TO", "KMP-UN.TO", "MSA.TO", "NA.TO", "NXE.TO",
    "QSR.TO", "RGPM.NE", "RY.TO", "SIA.NE", "SHOP.TO",
    "TECK-A.TO", "UCU.NE", "XEQT.TO", "YNVD.NE",
    "ZQQ.TO", "ZSP.TO"
]


def fetch_json(url: str) -> dict:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def sma(values, period):
    if len(values) < period:
        return None
    return mean(values[-period:])


def ema_series(values, period):
    if len(values) < period:
        return []

    multiplier = 2 / (period + 1)
    seed = mean(values[:period])
    series = [seed]

    for price in values[period:]:
        series.append((price - series[-1]) * multiplier + series[-1])
    return series


def rsi(values, period=14):
    if len(values) <= period:
        return None

    gains = []
    losses = []
    for index in range(1, period + 1):
        delta = values[index] - values[index - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))

    avg_gain = mean(gains)
    avg_loss = mean(losses)

    for index in range(period + 1, len(values)):
        delta = values[index] - values[index - 1]
        gain = max(delta, 0)
        loss = max(-delta, 0)
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(values):
    ema12 = ema_series(values, 12)
    ema26 = ema_series(values, 26)
    if not ema12 or not ema26:
        return None

    offset = len(ema12) - len(ema26)
    macd_line = [ema12[index + offset] - ema26[index] for index in range(len(ema26))]
    signal_line = ema_series(macd_line, 9)
    if not signal_line:
        return None

    hist = macd_line[-1] - signal_line[-1]
    return {
        "macd": round(macd_line[-1], 4),
        "signal": round(signal_line[-1], 4),
        "histogram": round(hist, 4),
    }


def bollinger_bands(values, period=20, multiplier=2):
    if len(values) < period:
        return None
    window = values[-period:]
    center = mean(window)
    variance = sum((value - center) ** 2 for value in window) / period
    deviation = variance ** 0.5
    upper = center + multiplier * deviation
    lower = center - multiplier * deviation
    return {
        "middle": round(center, 4),
        "upper": round(upper, 4),
        "lower": round(lower, 4),
    }


def atr(points, period=14):
    if len(points) <= period:
        return None

    true_ranges = []
    for index in range(1, len(points)):
        high = points[index].get("high")
        low = points[index].get("low")
        prev_close = points[index - 1].get("close")
        if high is None or low is None or prev_close is None:
            continue
        true_ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))

    if len(true_ranges) < period:
        return None
    return round(mean(true_ranges[-period:]), 4)


def percent_change(current, reference):
    if reference in (0, None):
        return None
    return ((current - reference) / reference) * 100


def normalize_chart_result(raw):
    chart = raw.get("chart", {})
    results = chart.get("result") or []
    if not results:
        error = chart.get("error") or {}
        raise ValueError(error.get("description") or "No chart data returned.")

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators") or {}).get("quote") or [{}]
    closes = quote[0].get("close") or []
    opens = quote[0].get("open") or []
    highs = quote[0].get("high") or []
    lows = quote[0].get("low") or []
    volumes = quote[0].get("volume") or []
    meta = result.get("meta") or {}

    points = []
    for index, timestamp in enumerate(timestamps):
        close = closes[index] if index < len(closes) else None
        if close is None:
            continue

        points.append(
            {
                "timestamp": timestamp,
                "datetime": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                "open": opens[index] if index < len(opens) else None,
                "high": highs[index] if index < len(highs) else None,
                "low": lows[index] if index < len(lows) else None,
                "close": close,
                "volume": volumes[index] if index < len(volumes) else None,
            }
        )

    if not points:
        raise ValueError("No usable price points returned.")

    return {
        "symbol": meta.get("symbol"),
        "currency": meta.get("currency"),
        "exchangeName": meta.get("exchangeName"),
        "instrumentType": meta.get("instrumentType"),
        "regularMarketPrice": meta.get("regularMarketPrice"),
        "previousClose": meta.get("previousClose"),
        "chartPreviousClose": meta.get("chartPreviousClose"),
        "timezone": meta.get("exchangeTimezoneName"),
        "dataGranularity": meta.get("dataGranularity"),
        "range": meta.get("range"),
        "validRanges": meta.get("validRanges") or [],
        "points": points,
    }


def compute_analysis(chart_data, quote_data=None):
    points = chart_data["points"]
    closes = [point["close"] for point in points]
    current = closes[-1]
    sma20 = sma(closes, 20)
    sma50 = sma(closes, 50)
    sma200 = sma(closes, 200)
    rsi14 = rsi(closes, 14)
    macd_data = macd(closes)
    bbands = bollinger_bands(closes, 20, 2)
    atr14 = atr(points, 14)
    prev_close = chart_data.get("previousClose") or chart_data.get("chartPreviousClose")
    price_change_pct = percent_change(current, prev_close)

    score = 0
    reasons = []
    indicator_cards = []

    if sma20 and current > sma20:
        score += 1
        reasons.append("Price is above the 20-period average.")
        indicator_cards.append({"name": "SMA 20", "value": round(sma20, 4), "signal": "BUY", "meaning": "Short-term trend is above its average."})
    elif sma20:
        score -= 1
        reasons.append("Price is below the 20-period average.")
        indicator_cards.append({"name": "SMA 20", "value": round(sma20, 4), "signal": "SELL", "meaning": "Short-term price is trading below its average."})

    if sma50 and current > sma50:
        score += 1
        reasons.append("Price is above the 50-period average.")
        indicator_cards.append({"name": "SMA 50", "value": round(sma50, 4), "signal": "BUY", "meaning": "Medium-term trend remains constructive."})
    elif sma50:
        score -= 1
        reasons.append("Price is below the 50-period average.")
        indicator_cards.append({"name": "SMA 50", "value": round(sma50, 4), "signal": "SELL", "meaning": "Medium-term trend remains under pressure."})

    if sma20 and sma50 and sma20 > sma50:
        score += 1
        reasons.append("Short-term momentum is stronger than medium-term momentum.")
    elif sma20 and sma50:
        score -= 1
        reasons.append("Short-term momentum is weaker than medium-term momentum.")

    if rsi14 is not None:
        if rsi14 < 30:
            score += 1
            reasons.append("RSI is in oversold territory, which can support a bounce.")
            indicator_cards.append({"name": "RSI 14", "value": round(rsi14, 2), "signal": "BUY", "meaning": "Oversold conditions can support a rebound."})
        elif rsi14 > 70:
            score -= 1
            reasons.append("RSI is in overbought territory, which can signal exhaustion.")
            indicator_cards.append({"name": "RSI 14", "value": round(rsi14, 2), "signal": "SELL", "meaning": "Overbought conditions can mean short-term exhaustion."})
        else:
            reasons.append("RSI is in a neutral range.")
            indicator_cards.append({"name": "RSI 14", "value": round(rsi14, 2), "signal": "HOLD", "meaning": "Momentum is balanced, not stretched."})

    if macd_data:
        if macd_data["macd"] > macd_data["signal"]:
            score += 1
            reasons.append("MACD is above its signal line.")
            indicator_cards.append({"name": "MACD", "value": macd_data["macd"], "signal": "BUY", "meaning": "Momentum is strengthening versus the signal line."})
        else:
            score -= 1
            reasons.append("MACD is below its signal line.")
            indicator_cards.append({"name": "MACD", "value": macd_data["macd"], "signal": "SELL", "meaning": "Momentum is weakening versus the signal line."})

    if bbands:
        if current < bbands["lower"]:
            indicator_cards.append({"name": "Bollinger", "value": bbands["lower"], "signal": "BUY", "meaning": "Price is below the lower band, which can mean oversold."})
        elif current > bbands["upper"]:
            indicator_cards.append({"name": "Bollinger", "value": bbands["upper"], "signal": "SELL", "meaning": "Price is above the upper band, which can mean overextended."})
        else:
            indicator_cards.append({"name": "Bollinger", "value": bbands["middle"], "signal": "HOLD", "meaning": "Price is trading inside a normal volatility range."})

    if atr14 is not None:
        indicator_cards.append({"name": "ATR 14", "value": atr14, "signal": "INFO", "meaning": "Higher ATR means larger price swings and more volatility risk."})

    signal = "HOLD"
    if score >= 3:
        signal = "BUY"
    elif score <= -3:
        signal = "SELL"

    confidence = min(95, 50 + abs(score) * 9)

    trend = "Sideways"
    if sma20 and sma50 and current > sma20 > sma50:
        trend = "Bullish"
    elif sma20 and sma50 and current < sma20 < sma50:
        trend = "Bearish"

    analysis = {
        "signal": signal,
        "score": score,
        "confidence": confidence,
        "trend": trend,
        "currentPrice": round(current, 4),
        "previousClose": prev_close,
        "changePercentFromClose": round(price_change_pct, 2) if price_change_pct is not None else None,
        "indicators": {
            "sma20": round(sma20, 4) if sma20 else None,
            "sma50": round(sma50, 4) if sma50 else None,
            "sma200": round(sma200, 4) if sma200 else None,
            "rsi14": round(rsi14, 2) if rsi14 is not None else None,
            "macd": macd_data,
            "bollinger": bbands,
            "atr14": atr14,
        },
        "indicatorCards": indicator_cards,
        "reasons": reasons,
    }

    if quote_data:
        analysis["quote"] = quote_data

    return analysis


def latest_volume_ratio(points, period=20):
    volumes = [point.get("volume") for point in points if point.get("volume")]
    if len(volumes) <= period:
        return None
    average_volume = mean(volumes[-period - 1:-1])
    if not average_volume:
        return None
    return volumes[-1] / average_volume


def latest_change_percent(points):
    if len(points) < 2:
        return None
    return percent_change(points[-1].get("close"), points[-2].get("close"))


def risk_levels(current, atr14, signal):
    if current is None or atr14 is None:
        return {"stop": None, "target": None}
    if signal == "BUY":
        return {
            "stop": round(max(0, current - atr14 * 1.5), 4),
            "target": round(current + atr14 * 2.5, 4),
        }
    if signal == "SELL":
        return {
            "stop": round(current + atr14 * 1.5, 4),
            "target": round(max(0, current - atr14 * 2.5), 4),
        }
    return {"stop": None, "target": None}


def classify_setup(mode, analysis, points):
    indicators = analysis.get("indicators") or {}
    current = analysis.get("currentPrice")
    sma20 = indicators.get("sma20")
    sma50 = indicators.get("sma50")
    rsi14 = indicators.get("rsi14")
    macd_data = indicators.get("macd") or {}
    atr14 = indicators.get("atr14")
    volume_ratio = latest_volume_ratio(points)
    atr_percent = percent_change(current + atr14, current) if current and atr14 else None

    tags = []
    score = analysis["score"]

    if analysis["trend"] == "Bullish":
        tags.append("uptrend")
    elif analysis["trend"] == "Bearish":
        tags.append("downtrend")
    if volume_ratio and volume_ratio >= 1.5:
        score += 1
        tags.append("volume surge")
    if atr_percent and atr_percent >= (1.8 if mode == "day" else 2.8):
        tags.append("high volatility")
    if sma20 and sma50 and current and current > sma20 > sma50 and macd_data.get("histogram", 0) > 0:
        score += 1
        tags.append("momentum continuation")
    if rsi14 is not None and 45 <= rsi14 <= 65 and analysis["trend"] == "Bullish":
        tags.append("balanced RSI")
    if rsi14 is not None and rsi14 > 72:
        score -= 1
        tags.append("extended")

    if mode == "day":
        buy_threshold = 3
        sell_threshold = -3
        label = "Day"
    else:
        buy_threshold = 4
        sell_threshold = -4
        label = "Swing"

    signal = "WATCH"
    if score >= buy_threshold:
        signal = "LONG"
    elif score <= sell_threshold:
        signal = "AVOID"

    if signal == "LONG" and "volume surge" in tags:
        setup = f"{label} momentum"
    elif signal == "LONG":
        setup = f"{label} trend pullback"
    elif signal == "AVOID":
        setup = "Weak or extended"
    else:
        setup = "Watchlist only"

    return {
        "signal": signal,
        "setup": setup,
        "screenScore": score,
        "tags": tags[:4],
        "volumeRatio": round(volume_ratio, 2) if volume_ratio is not None else None,
        "atrPercent": round(atr_percent, 2) if atr_percent is not None else None,
        **risk_levels(current, atr14, "BUY" if signal == "LONG" else analysis["signal"]),
    }


def screen_one_symbol(symbol, mode):
    range_value = "5d" if mode == "day" else "6mo"
    interval = "15m" if mode == "day" else "1d"
    chart_data = fetch_chart(symbol, range_value, interval)
    quote_data = build_quote_from_chart(chart_data, fetch_quote(symbol))
    analysis = compute_analysis(chart_data, quote_data)
    setup = classify_setup(mode, analysis, chart_data["points"])
    quote_name = quote_data.get("shortName") or quote_data.get("longName") or symbol

    return {
        "symbol": symbol,
        "name": quote_name,
        "price": analysis["currentPrice"],
        "currency": quote_data.get("currency") or chart_data.get("currency") or "CAD",
        "changePercent": quote_data.get("regularMarketChangePercent") if quote_data.get("regularMarketChangePercent") is not None else latest_change_percent(chart_data["points"]),
        "volume": quote_data.get("regularMarketVolume") or chart_data["points"][-1].get("volume"),
        "signal": setup["signal"],
        "setup": setup["setup"],
        "score": setup["screenScore"],
        "confidence": analysis["confidence"],
        "trend": analysis["trend"],
        "rsi14": analysis["indicators"]["rsi14"],
        "atrPercent": setup["atrPercent"],
        "volumeRatio": setup["volumeRatio"],
        "stop": setup["stop"],
        "target": setup["target"],
        "tags": setup["tags"],
        "resolvedInterval": chart_data.get("resolvedInterval"),
        "range": chart_data.get("range"),
    }


def parse_symbols(raw_symbols):
    if not raw_symbols:
        return CANADIAN_STOCKS
    symbols = []
    for item in raw_symbols.replace("\n", ",").split(","):
        symbol = item.strip().upper()
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols[:80]


def screen_symbols(symbols, mode):
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(screen_one_symbol, symbol, mode): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({"symbol": symbol, "error": str(exc)})

    signal_order = {"LONG": 0, "WATCH": 1, "AVOID": 2}
    results.sort(key=lambda row: (signal_order.get(row["signal"], 9), -row["score"], -(row["volumeRatio"] or 0)))
    return results, errors


def fetch_quote(symbol: str):
    params = urlencode({"symbols": symbol})
    try:
        raw = fetch_json(f"{YAHOO_QUOTE_URL}?{params}")
    except HTTPError as exc:
        if exc.code == 401:
            return None
        raise
    result = ((raw.get("quoteResponse") or {}).get("result") or [])
    if not result:
        return None

    quote = result[0]
    return {
        "symbol": quote.get("symbol"),
        "shortName": quote.get("shortName"),
        "longName": quote.get("longName"),
        "currency": quote.get("currency"),
        "marketState": quote.get("marketState"),
        "regularMarketPrice": quote.get("regularMarketPrice"),
        "regularMarketChange": quote.get("regularMarketChange"),
        "regularMarketChangePercent": quote.get("regularMarketChangePercent"),
        "regularMarketVolume": quote.get("regularMarketVolume"),
        "fiftyTwoWeekHigh": quote.get("fiftyTwoWeekHigh"),
        "fiftyTwoWeekLow": quote.get("fiftyTwoWeekLow"),
        "averageDailyVolume3Month": quote.get("averageDailyVolume3Month"),
        "marketCap": quote.get("marketCap"),
    }


def build_quote_from_chart(chart_data, quote_data=None):
    if quote_data:
        return quote_data

    points = chart_data.get("points") or []
    latest_volume = points[-1].get("volume") if points else None

    return {
        "symbol": chart_data.get("symbol"),
        "shortName": chart_data.get("symbol"),
        "longName": chart_data.get("symbol"),
        "currency": chart_data.get("currency"),
        "marketState": chart_data.get("exchangeName") or "Available",
        "regularMarketPrice": chart_data.get("regularMarketPrice"),
        "regularMarketChange": None,
        "regularMarketChangePercent": None,
        "regularMarketVolume": latest_volume,
        "fiftyTwoWeekHigh": None,
        "fiftyTwoWeekLow": None,
        "averageDailyVolume3Month": None,
        "marketCap": None,
    }


def enrich_quote_metrics(symbol: str, quote_data: dict):
    try:
        history = fetch_chart(symbol, "1y", "1d")
    except Exception:
        return quote_data

    closes = [point["close"] for point in history["points"]]
    volumes = [point.get("volume") for point in history["points"] if point.get("volume") is not None]
    non_zero_volumes = [value for value in volumes if value]

    enriched = dict(quote_data)
    if closes:
        enriched["fiftyTwoWeekHigh"] = enriched.get("fiftyTwoWeekHigh") or round(max(closes), 4)
        enriched["fiftyTwoWeekLow"] = enriched.get("fiftyTwoWeekLow") or round(min(closes), 4)
    if len(non_zero_volumes) >= 20:
        enriched["averageDailyVolume3Month"] = enriched.get("averageDailyVolume3Month") or round(mean(non_zero_volumes[-20:]))
    if not enriched.get("regularMarketVolume") and non_zero_volumes:
        enriched["regularMarketVolume"] = non_zero_volumes[-1]
    return enriched


def build_action(signal: str, trend: str, confidence: int, label: str):
    if signal == "BUY":
        return f"{label}: bias is bullish. Consider entries on pullbacks while risk stays controlled."
    if signal == "SELL":
        return f"{label}: bias is defensive. Reduce size, wait, or look for confirmation before buying."
    if trend == "Bullish":
        return f"{label}: trend is still positive, but the setup is not strong enough for a fresh buy signal."
    if trend == "Bearish":
        return f"{label}: trend is still weak, so patience is safer than forcing an entry."
    return f"{label}: mixed setup. Best action is to wait for a cleaner trend or momentum break."


def build_timeframe_guidance(symbol: str):
    frames = [
        ("1 Hour", "5d", "1h"),
        ("1 Day", "6mo", "1d"),
        ("3 Days", "5d", "1d"),
        ("1 Week", "2y", "1wk"),
        ("1 Month", "1y", "1wk"),
        ("6 Months", "2y", "1mo"),
        ("12 Months", "5y", "1mo"),
    ]

    guidance = []
    for label, range_value, interval in frames:
        try:
            chart = fetch_chart(symbol, range_value, interval)
            analysis = compute_analysis(chart)
            guidance.append(
                {
                    "label": label,
                    "signal": analysis["signal"],
                    "trend": analysis["trend"],
                    "confidence": analysis["confidence"],
                    "action": build_action(analysis["signal"], analysis["trend"], analysis["confidence"], label),
                }
            )
        except Exception:
            continue
    return guidance


def fetch_chart_once(symbol: str, range_value: str, interval: str):
    params = urlencode({"range": range_value, "interval": interval, "includePrePost": "true"})
    url = YAHOO_CHART_URL.format(symbol=symbol) + f"?{params}"
    raw = fetch_json(url)
    chart_data = normalize_chart_result(raw)
    chart_data["requestedInterval"] = interval
    chart_data["resolvedInterval"] = interval
    return chart_data


def fetch_chart(symbol: str, range_value: str, interval: str):
    candidates = INTERVAL_FALLBACKS.get(range_value, [interval, "1d"])
    ordered_intervals = []

    for candidate in [interval, *candidates]:
        if candidate not in ordered_intervals:
            ordered_intervals.append(candidate)

    last_error = None
    for candidate in ordered_intervals:
        try:
            chart_data = fetch_chart_once(symbol, range_value, candidate)
            chart_data["requestedInterval"] = interval
            chart_data["resolvedInterval"] = candidate
            chart_data["intervalAdjusted"] = candidate != interval
            return chart_data
        except HTTPError as exc:
            last_error = exc
            if exc.code == 422:
                continue
            raise

    if last_error:
        raise last_error
    raise ValueError("No chart data returned.")


class StockDashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            return self.serve_file(STATIC_DIR / "index.html")
        if path == "/monitor":
            return self.serve_file(STATIC_DIR / "monitor.html")
        if path.startswith("/static/"):
            relative_path = path.replace("/static/", "", 1)
            return self.serve_file(STATIC_DIR / relative_path)
        if path == "/api/stock":
            return self.handle_stock_api(parsed.query)
        if path == "/api/screen":
            return self.handle_screen_api(parsed.query)

        self.send_error(404, "Not Found")

    def handle_stock_api(self, query_string):
        query = parse_qs(query_string)
        symbol = (query.get("symbol", ["AAPL"])[0] or "AAPL").upper()
        range_value = query.get("range", ["1d"])[0]
        interval = query.get("interval", ["5m"])[0]

        try:
            chart_data = fetch_chart(symbol, range_value, interval)
            quote_data = enrich_quote_metrics(symbol, build_quote_from_chart(chart_data, fetch_quote(symbol)))
            analysis = compute_analysis(chart_data, quote_data)
            guidance = build_timeframe_guidance(symbol)
            self.send_json(
                {
                    "ok": True,
                    "quote": quote_data,
                    "chart": chart_data,
                    "analysis": analysis,
                    "timeframes": guidance,
                }
            )
        except ValueError as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=400)
        except HTTPError as exc:
            self.send_json({"ok": False, "error": f"Upstream HTTP error: {exc.code}"}, status=502)
        except URLError:
            self.send_json(
                {
                    "ok": False,
                    "error": "Could not reach the market data provider. Check your connection and try again.",
                },
                status=502,
            )
        except Exception as exc:
            self.send_json({"ok": False, "error": f"Unexpected error: {exc}"}, status=500)

    def handle_screen_api(self, query_string):
        query = parse_qs(query_string)
        mode = (query.get("mode", ["swing"])[0] or "swing").lower()
        if mode not in {"swing", "day"}:
            mode = "swing"
        symbols = parse_symbols(query.get("symbols", [""])[0])

        try:
            results, errors = screen_symbols(symbols, mode)
            self.send_json(
                {
                    "ok": True,
                    "mode": mode,
                    "symbolsScanned": len(symbols),
                    "results": results,
                    "errors": errors,
                    "source": "Yahoo Finance chart/quote endpoints. Data may be delayed.",
                }
            )
        except URLError:
            self.send_json(
                {
                    "ok": False,
                    "error": "Could not reach the free market data provider. Check your connection and try again.",
                },
                status=502,
            )
        except Exception as exc:
            self.send_json({"ok": False, "error": f"Unexpected error: {exc}"}, status=500)

    def serve_file(self, file_path: Path):
        if not file_path.exists() or not file_path.is_file():
            return self.send_error(404, "File Not Found")

        content_type, _ = mimetypes.guess_type(str(file_path))
        content_type = content_type or "application/octet-stream"

        try:
            with open(file_path, "rb") as file:
                body = file.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            self.send_error(500, "Could not read file")

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


def run():
    server = ThreadingHTTPServer((HOST, PORT), StockDashboardHandler)
    print(f"Serving on http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
