"""
insights.py - one-sentence, plain-language readings of each tab's results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import quant as q


def _p(x, d=0):
    return f"{x:.{d}%}"


# --------------------------------------------------------------------------- #
# Single share
# --------------------------------------------------------------------------- #
def price(a: q.Analysis, name: str) -> str:
    s = a.stats
    above = s["last"] > s["ma200"]
    cross = "50-day above 200-day (uptrend)" if s["ma50"] > s["ma200"] else \
        "50-day below 200-day (downtrend)"
    h = s["hurst"]
    beh = ("trending" if h > 0.55 else "mean-reverting" if h < 0.45 else "close to a random walk")
    return (f"{name} trades {'above' if above else 'below'} its 200-day average with the "
            f"{cross}; the model reads a **{s['regime']}** regime and price behaviour is "
            f"{beh} (Hurst {h:.2f}).")


def monte_carlo(a: q.Analysis, horizon: int, target: float, stop: float) -> str:
    m = a.mc
    base = stop / (target + stop)      # odds of a driftless random walk
    pt = m["p_target_first"] / max(1e-9, m["p_target_first"] + m["p_stop_first"])
    view = ("favour" if pt > base + 0.03 else "work against" if pt < base - 0.03
            else "are roughly neutral for")
    return (f"Over {horizon} days the median outcome is {m['median_ret']:+.1%} and 90% of "
            f"paths end between {m['q'][0.05]:+.1%} and {m['q'][0.95]:+.1%}; the odds "
            f"**{view}** a +{target:.0%} / −{stop:.0%} trade (when one of the two is hit, "
            f"the target comes first {_p(pt)} of the time vs {_p(base)} for a pure "
            f"random walk).")


def volatility(a: q.Analysis) -> str:
    g = a.garch
    r = g.vol_ratio
    lvl = ("well above" if r > 1.3 else "above" if r > 1.05 else
           "below" if r < 0.95 else "in line with")
    nxt = ("expect wider-than-usual swings and give stops more room" if r > 1.05 else
           "swings should be calmer than usual, a good backdrop for entries" if r < 0.95
           else "expect normal swings")
    return (f"Volatility is **{lvl}** its long-run level ({_p(g.next_vol_ann)} vs "
            f"{_p(g.long_run_vol_ann)} a year); shocks fade with a half-life of about "
            f"{g.half_life_days:.0f} days, so {nxt}.")


def regimes(a: q.Analysis) -> str:
    hm = a.hmm
    i = int(np.argmax(hm.next_day))
    return (f"The share is most likely in a **{hm.labels[i]}** regime "
            f"({_p(hm.next_day[i])} probability), which typically lasts about "
            f"{hm.expected_duration[i]:.0f} trading days; the chance of being in the Bear "
            f"regime tomorrow is {_p(hm.next_day[0])}.")


def limits(a: q.Analysis, horizon: int) -> str:
    lt = a.limits
    ok = lt[lt.p_fill >= 0.5]
    row = ok.iloc[-1] if not ok.empty else lt.iloc[0]
    gain = row.p_gain_if_filled
    return (f"A buy limit at **{row.limit_price:,.2f}** (−{row.discount:.1%}) has about a "
            f"{_p(row.p_fill)} chance of filling within {horizon} days"
            + (f", and {_p(gain)} of those fills end in profit" if pd.notna(gain) else "")
            + "; deeper limits are cheaper but fill less often.")


def backtest(tab: pd.DataFrame, fwd: int) -> str:
    top = next((i for i in tab.index if str(i).startswith("Score ≥")), None)
    if top is None or tab.loc[top, "days"] < 20:
        return ("The score rarely reached the 'favourable' zone on this share, so its "
                "track record cannot be judged yet – rely on it with caution.")
    t, al = tab.loc[top], tab.loc["All days"]
    better = t.mean_fwd > al.mean_fwd and t.hit_rate >= al.hit_rate
    return (f"Historically, days with a high entry score were followed by "
            f"{t.mean_fwd:+.1%} on average over {fwd} days vs {al.mean_fwd:+.1%} for all "
            f"days (hit rate {_p(t.hit_rate)} vs {_p(al.hit_rate)}), so the score has "
            + ("**added value** on this share." if better
               else "**not added value** on this share – treat its signals with caution."))


def trade(tp: q.TradePlan, name: str) -> str:
    tail = {"ENTRY": "conditions line up for a buy in the green zone.",
            "NEUTRAL": "the set-up is not compelling yet – wait for the green zone or a "
                       "better score.",
            "OUT": "risk dominates – stay out or reduce."}[tp.call]
    mid = (tp.entry_low + tp.entry_high) / 2
    where = ("If it later drops into the re-entry zone, buying" if tp.call == "OUT"
             else "Buying")
    return (f"For {name}, the call is **{tp.call}**: {tail} {where} in the middle of the "
            f"zone ({mid:,.2f}) risks {1 - tp.stop / mid:.1%} to the stop for "
            f"{tp.tp1 / mid - 1:+.1%} to the first take-profit (risk/reward "
            f"{tp.risk_reward:.1f}).")


# --------------------------------------------------------------------------- #
# Portfolio
# --------------------------------------------------------------------------- #
def holdings(total: float, pnl: float, ps: dict, var_h: int) -> str:
    b = ps["beta"]
    beta_txt = ("" if pd.isna(b) else
                f"; with a beta of {b:.2f} it moves {'more' if b > 1.05 else 'less' if b < 0.95 else 'in line'}"
                f"{' than' if abs(b - 1) >= 0.05 else ' with'} the CAC 40")
    sign = "+" if pnl >= 0 else "−"
    return (f"Your portfolio is worth €{total:,.0f} with an unrealised P&L of "
            f"{sign}€{abs(pnl):,.0f}"
            f"{beta_txt}, and in a bad {var_h}-day period (1 in 20) it could lose about "
            f"€{ps['var95'] * total:,.0f}.")


def performance(ps: dict, bench: pd.Series | None) -> str:
    cum = ps["cum"]
    pr = cum.iloc[-1] / cum.iloc[0] - 1
    txt = f"With today's weights the portfolio would have returned {pr:+.0%} over the period"
    if bench is not None:
        b = bench.reindex(cum.index).dropna()
        if len(b) > 1:
            br = b.iloc[-1] / b.iloc[0] - 1
            txt += (f" vs {br:+.0%} for the CAC 40 price index "
                    f"({'outperforming' if pr > br else 'underperforming'})")
    return txt + f", with a worst peak-to-trough fall of {abs(ps['mdd']):.0%}."


def risk(ps: dict, w: pd.Series, names: dict | None = None) -> str:
    gap = (ps["risk_contrib"] - w).sort_values()
    hi = gap.index[-1]
    c = ps["corr"].values
    n = len(c)
    avg = (c.sum() - n) / (n * (n - 1)) if n > 1 else np.nan
    div = ("" if pd.isna(avg) else
           f"; average correlation {avg:.2f} means diversification is "
           f"{'weak' if avg > 0.6 else 'moderate' if avg > 0.35 else 'good'}")
    label = (names or {}).get(hi, hi)
    return (f"**{label}** adds the most risk relative to its size "
            f"({_p(ps['risk_contrib'][hi])} of risk for {_p(w[hi])} of value){div}.")


def market(label: str, p_bear: float) -> str:
    return (f"The CAC 40 itself is in a **{label}** regime with a {_p(p_bear)} bear "
            "probability – " +
            ("a defensive stance (smaller positions, tighter stops) is sensible."
             if p_bear > 0.5 else
             "a supportive backdrop for selective entries." if label == "Bull"
             else "no strong market-wide signal; pick shares on their own merits."))


def screener(sc: pd.DataFrame) -> str:
    fav = sc[sc.Verdict == "Favourable entry"]
    risky = sc[sc["Decline risk"] >= 55]
    best = sc.sort_values("Entry score", ascending=False).iloc[0]
    return (f"{len(fav)} of {len(sc)} shares show a favourable entry today and "
            f"{len(risky)} carry high decline risk; the best-ranked is "
            f"**{best.Name}** (score {best['Entry score']:.0f}).")
