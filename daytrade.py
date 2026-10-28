"""
daytrade.py - intraday (day-trading) analysis for Euronext shares.

Inputs : intraday OHLCV bars (Paris time, naive index) + daily close history.
Outputs: VWAP, EMA 9/21, opening range, classic pivots, expected daily range
         (GARCH), an intraday score, a BUY / WAIT / SELL plan with entry, stop
         and targets, and a bootstrap Monte Carlo of the rest of the session.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import quant as q

OPEN_MIN, CLOSE_MIN = 9 * 60, 17 * 60 + 30     # Euronext continuous trading 09:00-17:30
SESSION_MIN = CLOSE_MIN - OPEN_MIN
WEIGHTS = {"Daily trend": 0.20, "Daily regime": 0.10, "VWAP": 0.25,
           "EMA 9/21": 0.20, "Opening range": 0.15, "RSI": 0.10}
INTRADAY = ("VWAP", "EMA 9/21", "Opening range")


# --------------------------------------------------------------------------- #
# Bars & indicators
# --------------------------------------------------------------------------- #
def session_bars(bars: pd.DataFrame) -> pd.DataFrame:
    m = bars.index.hour * 60 + bars.index.minute
    return bars[(m >= OPEN_MIN) & (m < CLOSE_MIN)]


def bar_minutes(idx: pd.DatetimeIndex) -> int:
    d = pd.Series(idx).diff().dropna()
    d = d[d < pd.Timedelta(hours=2)]
    return max(1, int(round(d.median().total_seconds() / 60))) if len(d) else 5


def add_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    b = bars.copy()
    day = b.index.normalize()
    tp = (b.High + b.Low + b.Close) / 3
    cum_mean = tp.groupby(day).cumsum() / (tp.groupby(day).cumcount() + 1)
    vol = b.Volume.fillna(0)
    if vol.sum() > 0:
        vw = (tp * vol).groupby(day).cumsum() / vol.groupby(day).cumsum().replace(0, np.nan)
        b["VWAP"] = vw.fillna(cum_mean)
    else:                                   # indices have no volume
        b["VWAP"] = cum_mean
    b["EMA9"] = b.Close.ewm(span=9, adjust=False).mean()
    b["EMA21"] = b.Close.ewm(span=21, adjust=False).mean()
    prev = b.Close.shift()
    tr = pd.concat([b.High - b.Low, (b.High - prev).abs(), (b.Low - prev).abs()],
                   axis=1).max(axis=1)
    b["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    b["RSI"] = q.rsi(b.Close)
    return b


def daily_ohlc(bars: pd.DataFrame) -> pd.DataFrame:
    g = bars.groupby(bars.index.normalize())
    return pd.DataFrame({"Open": g.Open.first(), "High": g.High.max(), "Low": g.Low.min(),
                         "Close": g.Close.last(), "Volume": g.Volume.sum()})


def pivots(row) -> dict:
    H, L, C = float(row.High), float(row.Low), float(row.Close)
    P = (H + L + C) / 3
    return {"R2": P + (H - L), "R1": 2 * P - L, "Pivot": P, "S1": 2 * P - H,
            "S2": P - (H - L)}


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #
@dataclass
class DayPlan:
    call: str                     # BUY | WAIT | SELL
    headline: str
    live: bool
    session_date: pd.Timestamp
    bar_min: int
    price: float
    prev_close: float
    day_open: float
    day_high: float
    day_low: float
    vwap: float
    or_high: float
    or_low: float
    or_complete: bool
    or_end: pd.Timestamp
    pivots: dict
    exp1: tuple
    exp2: tuple
    daily_atr: float
    bar_atr: float
    range_used: float
    score: float
    components: dict
    entry_low: float
    entry_high: float
    stop: float
    t1: float
    t2: float
    rr_t2: float
    triggers: dict
    mc: dict
    ladder: pd.DataFrame
    bars: pd.DataFrame
    reasons: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _nearest(values, ref, above: bool):
    c = sorted(v for v in values if (v > ref if above else v < ref))
    return (c[0] if above else c[-1]) if c else None


def analyze_day(bars: pd.DataFrame, daily_close: pd.Series, sigma_d: float, p_bear: float,
                or_minutes: int = 30, now: pd.Timestamp | None = None,
                n_sims: int = 4000, seed: int = 1) -> DayPlan:
    bars = session_bars(bars.dropna(subset=["Close"]))
    if len(bars) < 40:
        raise ValueError("Not enough intraday bars.")
    bm = bar_minutes(bars.index)
    per = max(1, SESSION_MIN // bm)
    b = add_indicators(bars)
    d = daily_ohlc(bars)
    last_day = d.index[-1]
    now = now if now is not None else pd.Timestamp.now(tz="Europe/Paris").tz_localize(None)
    mnow = now.hour * 60 + now.minute
    live = bool(now.normalize() == last_day and OPEN_MIN <= mnow < CLOSE_MIN
                and now.weekday() < 5)
    tb = b[b.index.normalize() == last_day]

    done = d.iloc[:-1] if live else d                       # completed sessions
    ref = done.iloc[-1] if len(done) else d.iloc[-1]
    prev_close = float(ref.Close)
    piv = pivots(ref)
    pc = done.Close.shift()
    tr = pd.concat([done.High - done.Low, (done.High - pc).abs(), (done.Low - pc).abs()],
                   axis=1).max(axis=1)
    daily_atr = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]) if len(done) >= 3 \
        else float((d.High - d.Low).mean())

    price = float(tb.Close.iloc[-1])
    vwap = float(tb.VWAP.iloc[-1])
    or_n = max(1, int(np.ceil(or_minutes / bm)))
    orb = tb.iloc[:or_n]
    or_complete = len(tb) >= or_n or not live
    or_high, or_low = float(orb.High.max()), float(orb.Low.min())
    or_end = orb.index[-1] + pd.Timedelta(minutes=bm)
    day_open, day_high, day_low = float(tb.Open.iloc[0]), float(tb.High.max()), float(tb.Low.min())
    range_used = (day_high - day_low) / daily_atr if live and daily_atr > 0 else np.nan
    base = prev_close if live else price
    exp1 = (base * np.exp(-sigma_d), base * np.exp(sigma_d))
    exp2 = (base * np.exp(-2 * sigma_d), base * np.exp(2 * sigma_d))
    ema9, ema21 = float(tb.EMA9.iloc[-1]), float(tb.EMA21.iloc[-1])
    rsi_now = float(tb.RSI.iloc[-1])
    bar_atr = float(tb.ATR.iloc[-1])

    ma50 = float(daily_close.rolling(50).mean().iloc[-1])
    ma200 = float(daily_close.rolling(200).mean().iloc[-1])
    if price > ma200:
        trend = 1.0 if ma50 > ma200 else 0.5
    else:
        trend = -0.5 if ma50 > ma200 else -1.0
    comp = {
        "Daily trend": trend,
        "Daily regime": float(np.clip(1 - 2 * p_bear, -1, 1)),
        "VWAP": float(np.clip((price - vwap) / (0.25 * daily_atr), -1, 1)),
        "EMA 9/21": float(np.clip((ema9 - ema21) / max(bar_atr, 1e-9), -1, 1)),
        "Opening range": (1.0 if price > or_high else -1.0 if price < or_low else 0.0)
        if or_complete else 0.0,
        "RSI": float(-np.clip((rsi_now - 70) / 20, 0, 1) + np.clip((30 - rsi_now) / 20, 0, 1)),
    }
    score = 100 * sum(WEIGHTS[k] * v for k, v in comp.items())
    warnings = []
    if live and range_used > 1.0:
        score *= 0.6
        warnings.append(f"The share has already moved {range_used:.0%} of its typical daily "
                        "range – late entries have little room left.")
    intraday = sum(WEIGHTS[k] * comp[k] for k in INTRADAY)   # must confirm the direction
    call = ("BUY" if score >= 35 and intraday > 0.1 else
            "SELL" if score <= -35 and intraday < -0.1 else "WAIT")

    # ---- levels -------------------------------------------------------------
    ru = max(1.5 * bar_atr, 0.2 * daily_atr)                  # risk unit
    piv_vals = list(piv.values())
    triggers = {}
    lo = hi = stop = t1 = t2 = rr = np.nan
    if call == "BUY":
        hi = price
        lo = max(vwap, price - ru) if price > vwap else price - 0.5 * ru
        lo = min(lo, hi - 0.25 * ru)
        stop = lo - ru
        mid = (lo + hi) / 2
        risk = mid - stop
        t1 = mid + 1.5 * risk
        nxt = _nearest(piv_vals, t1, True)
        t2 = nxt if nxt is not None and nxt <= mid + 4 * risk else mid + 2.5 * risk
        rr = (t2 - mid) / risk
    elif call == "SELL":
        lo = price
        hi = min(vwap, price + ru) if price < vwap else price + 0.5 * ru
        hi = max(hi, lo + 0.25 * ru)
        stop = hi + ru
        mid = (lo + hi) / 2
        risk = stop - mid
        t1 = mid - 1.5 * risk
        nxt = _nearest(piv_vals, t1, False)
        t2 = nxt if nxt is not None and nxt >= mid - 4 * risk else mid - 2.5 * risk
        rr = (mid - t2) / risk
    L = or_high if or_complete and or_high > price else price + 0.5 * ru
    S = or_low if or_complete and or_low < price else price - 0.5 * ru
    triggers = {"buy_above": L, "buy_stop": L - ru, "buy_target": L + 1.5 * ru,
                "sell_below": S, "sell_stop": S + ru, "sell_target": S - 1.5 * ru}

    # ---- Monte Carlo of the rest of the session (bootstrap of intraday returns) --
    lr = np.log(b.Close).diff()
    same = pd.Series(b.index.normalize(), index=b.index).diff().eq(pd.Timedelta(0))
    pool = lr[same].dropna().values[-20 * per:]
    pool = pool - pool.mean()
    typical = pool.std() * np.sqrt(per) if len(pool) > 10 else np.nan
    scale = float(np.clip(sigma_d / typical, 0.5, 2.0)) if typical and typical > 0 else 1.0
    n_rem = max(1, per - len(tb)) if live else per
    rng = np.random.default_rng(seed)
    steps = rng.choice(pool, (n_sims, n_rem)) * scale if len(pool) > 10 else \
        rng.normal(0, sigma_d / np.sqrt(per), (n_sims, n_rem))
    paths = price * np.exp(np.cumsum(steps, axis=1))
    paths = np.hstack([np.full((n_sims, 1), price), paths])
    pmax, pmin, close = paths.max(1), paths.min(1), paths[:, -1]

    def touched(level):
        if level is None or np.isnan(level):
            return np.nan
        return float((pmax >= level).mean() if level >= price else (pmin <= level).mean())

    def first(level, up):
        hit = paths >= level if up else paths <= level
        return np.where(hit.any(1), hit.argmax(1), paths.shape[1] + 1)

    mc = {"p_close_up": float((close > price).mean()),
          "close_q": {p: float(np.quantile(close, p)) for p in (0.05, 0.5, 0.95)},
          "bars_left": n_rem}
    if call in ("BUY", "SELL"):
        up = call == "BUY"
        it, isl = first(t1, up), first(stop, not up)
        mc["p_t1_first"] = float((it < isl).mean())
        mc["p_stop_first"] = float((isl < it).mean())

    rows = []
    if call in ("BUY", "SELL"):
        rows += [("Target 2", t2, "Plan"), ("Target 1", t1, "Plan"),
                 ("Entry zone – top", hi, "Plan"), ("Entry zone – bottom", lo, "Plan"),
                 ("Stop", stop, "Plan")]
    else:
        rows += [("Buy trigger (break above)", L, "Trigger"),
                 ("Sell trigger (break below)", S, "Trigger")]
    rows += [("Current price", price, "Now"), ("VWAP", vwap, "Intraday"),
             ("Opening range high", or_high, "Intraday"), ("Opening range low", or_low, "Intraday"),
             ] + ([("Previous close", prev_close, "Reference")] if live else [])
    rows += [(k, v, "Pivot") for k, v in piv.items()]
    ladder = pd.DataFrame(rows, columns=["Level", "Price", "Type"])
    ladder["vs current"] = ladder.Price / price - 1
    ladder["P(touched by close)"] = [1.0 if r.Level == "Current price" else touched(r.Price)
                                     for r in ladder.itertuples()]
    ladder = ladder.sort_values("Price", ascending=False, ignore_index=True)

    # ---- wording -----------------------------------------------------------
    when = "" if live else "Next session – "
    if call == "BUY":
        headline = (f"{when}BUY – go long between {lo:,.2f} and {hi:,.2f} (dips toward VWAP), "
                    f"stop {stop:,.2f}, targets {t1:,.2f} / {t2:,.2f}. Close before 17:30.")
    elif call == "SELL":
        headline = (f"{when}SELL – intraday trend is down: exit longs or avoid buying; "
                    f"sell into {lo:,.2f}–{hi:,.2f}, stop {stop:,.2f}, targets "
                    f"{t1:,.2f} / {t2:,.2f}.")
    else:
        headline = (f"{when}WAIT – no clear intraday edge. Buy only on a break above "
                    f"{L:,.2f}; sell / avoid below {S:,.2f}.")
    reasons = [
        f"Daily trend: price {'above' if price > ma200 else 'below'} its 200-day average, "
        f"50-day {'above' if ma50 > ma200 else 'below'} 200-day; bear-regime probability "
        f"{p_bear:.0%}.",
        f"Price {'above' if price > vwap else 'below'} VWAP ({vwap:,.2f}) and EMA 9 "
        f"{'above' if ema9 > ema21 else 'below'} EMA 21.",
        (f"Opening range {or_low:,.2f}–{or_high:,.2f}: price is "
         + ("above it (breakout)." if price > or_high else
            "below it (breakdown)." if price < or_low else "inside it."))
        if or_complete else "Opening range still forming.",
        f"RSI ({bm}-min bars) {rsi_now:.0f}.",
        f"Expected range {'today' if live else 'next session'} (1σ, GARCH): "
        f"{exp1[0]:,.2f}–{exp1[1]:,.2f}"
        + (f"; {range_used:.0%} of a typical day's range already used." if live else "."),
    ]
    if live and not or_complete:
        warnings.append("Opening range still forming – early signals are less reliable.")
    if live and mnow >= CLOSE_MIN - 30:
        warnings.append("Less than 30 minutes of trading left – avoid opening new positions.")
    if call == "BUY" and p_bear > 0.5:
        warnings.append("Buying against a daily bear regime – keep the position small.")
    if call == "SELL" and trend > 0:
        warnings.append("Selling against the daily uptrend – intraday dips may be bought.")

    return DayPlan(call, headline, live, last_day, bm, price, prev_close, day_open, day_high,
                   day_low, vwap, or_high, or_low, or_complete, or_end, piv, exp1, exp2,
                   daily_atr, bar_atr, range_used, float(score), comp, float(lo), float(hi),
                   float(stop), float(t1), float(t2), float(rr), triggers, mc, ladder, b,
                   reasons, warnings)


# --------------------------------------------------------------------------- #
# Demo intraday data, consistent with the daily demo closes
# --------------------------------------------------------------------------- #
def synthetic_intraday(daily_close: pd.Series, interval_min: int = 5, n_days: int = 25,
                       partial: float = 0.6, key: str = "") -> pd.DataFrame:
    rng = np.random.default_rng(zlib.crc32((key or "x").encode()) + interval_min)
    closes = daily_close.dropna().iloc[-(n_days + 1):]
    per = SESSION_MIN // interval_min
    sig = float(np.log(daily_close).diff().iloc[-60:].std())
    t = np.arange(1, per + 1) / per
    u_shape = 1 + 2.5 * ((t - 0.5) * 2) ** 2
    days = list(closes.index[1:])            # last day = today's session, still running
    frames, prev = [], float(closes.iloc[0])
    for i, day in enumerate(days):
        last = i == len(days) - 1
        o = prev * np.exp(rng.normal(0, 0.3 * sig))
        w = np.cumsum(rng.normal(0, sig / np.sqrt(per), per))
        if last:
            path = np.log(o) + w
            n = int(per * partial)
        else:
            c1 = float(closes.iloc[i + 1])
            path = np.log(o) + w - t * w[-1] + t * (np.log(c1) - np.log(o))
            n = per
        c = np.exp(path)[:n]
        op = np.r_[o, c[:-1]]
        wig = np.abs(rng.normal(0, 0.6 * sig / np.sqrt(per), (2, n)))
        hi = np.maximum(op, c) * np.exp(wig[0])
        lo = np.minimum(op, c) * np.exp(-wig[1])
        vol = (2e4 * u_shape[:n] * rng.lognormal(0, 0.4, n)).round()
        idx = pd.Timestamp(day).normalize() + pd.Timedelta(hours=9) + \
            pd.to_timedelta(np.arange(n) * interval_min, unit="min")
        frames.append(pd.DataFrame({"Open": op, "High": hi, "Low": lo, "Close": c,
                                    "Volume": vol}, index=idx))
        prev = float(c[-1])
    return pd.concat(frames)
