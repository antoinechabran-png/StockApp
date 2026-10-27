"""
CAC 40 Stochastic Lab - Streamlit app
    streamlit run app.py

Tab 1  Single share : GARCH-t volatility, HMM regimes, Monte Carlo, OU mean
                      reversion -> entry score, decline-risk score, limit levels
Tab 2  CAC 40 portfolio : holdings P&L, risk (VaR/CVaR, beta, drawdown, risk
                      contributions), market regime, CAC 40 entry screener
Decision aid only - not investment advice.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import quant as q

st.set_page_config(page_title="CAC 40 Stochastic Lab", page_icon="📈", layout="wide")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
# Indicative composition - the index is reviewed quarterly by Euronext.
# Check https://live.euronext.com and edit the list in the Portfolio tab.
CAC40 = {
    "Accor": "AC.PA", "Air Liquide": "AI.PA", "Airbus": "AIR.PA",
    "ArcelorMittal": "MT.AS", "AXA": "CS.PA", "BNP Paribas": "BNP.PA",
    "bioMérieux": "BIM.PA", "Bouygues": "EN.PA", "Bureau Veritas": "BVI.PA",
    "Capgemini": "CAP.PA", "Carrefour": "CA.PA", "Crédit Agricole": "ACA.PA",
    "Danone": "BN.PA", "Dassault Systèmes": "DSY.PA", "Edenred": "EDEN.PA",
    "Engie": "ENGI.PA", "EssilorLuxottica": "EL.PA", "Eurofins Scientific": "ERF.PA",
    "Hermès": "RMS.PA", "Kering": "KER.PA", "L'Oréal": "OR.PA", "Legrand": "LR.PA",
    "LVMH": "MC.PA", "Michelin": "ML.PA", "Orange": "ORA.PA", "Pernod Ricard": "RI.PA",
    "Publicis": "PUB.PA", "Renault": "RNO.PA", "Safran": "SAF.PA",
    "Saint-Gobain": "SGO.PA", "Sanofi": "SAN.PA", "Schneider Electric": "SU.PA",
    "Société Générale": "GLE.PA", "Stellantis": "STLAP.PA",
    "STMicroelectronics": "STMPA.PA", "Thales": "HO.PA", "TotalEnergies": "TTE.PA",
    "Unibail-Rodamco-Westfield": "URW.PA", "Veolia": "VIE.PA", "Vinci": "DG.PA",
}
NAME_OF = {v: k for k, v in CAC40.items()}
INDEX = "^FCHI"
PERIOD_DAYS = {"3y": 756, "5y": 1260, "10y": 2520, "15y": 3780}

# chart palette (validated categorical order) + regime colours
C_BLUE, C_ORANGE, C_AQUA, C_GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
REGIME_COL = {"Bear": "#e34948", "Correction": "#eb6834", "Neutral": "#8a8984",
              "Calm": "#2a78d6", "Bull": "#1baf7a"}


def style(fig: go.Figure, h: int = 380, ytitle: str | None = None) -> go.Figure:
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10),
                      hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0),
                      yaxis_title=ytitle)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch")


def pct(x, d=1):
    return "–" if x is None or pd.isna(x) else f"{x:.{d}%}"


# --------------------------------------------------------------------------- #
# Data layer
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=3600, show_spinner=False)
def load_yahoo(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(list(tickers), period=period, auto_adjust=True,
                     progress=False, threads=True)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]
    else:
        close = df[["Close"]].rename(columns={"Close": tickers[0]})
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.dropna(how="all")


@st.cache_data(show_spinner=False)
def load_demo(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    return q.synthetic_prices(tickers, PERIOD_DAYS.get(period, 2520))


@st.cache_data(show_spinner=False)
def parse_csv(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    date_col = next((c for c in df.columns if c.lower() in ("date", "datetime", "time")),
                    df.columns[0])
    raw_dates = df[date_col].astype(str)
    iso = raw_dates.str.match(r"^\d{4}-\d{2}-\d{2}").all()
    df[date_col] = (pd.to_datetime(raw_dates, format="ISO8601", errors="coerce") if iso
                    else pd.to_datetime(raw_dates, dayfirst=True, errors="coerce"))
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(" ", "")
                                  .str.replace(",", "."), errors="coerce")
    return df.select_dtypes("number")


def get_prices(tickers: list[str], period: str) -> pd.DataFrame:
    tickers = tuple(sorted(set(tickers)))
    src = st.session_state.get("source", "Yahoo Finance")
    if src == "Demo (synthetic)":
        return load_demo(tickers, period)
    if src == "Upload CSV":
        up = st.session_state.get("csv_df")
        if up is None:
            return pd.DataFrame()
        cols = [t for t in tickers if t in up.columns]
        return up[cols].iloc[-PERIOD_DAYS.get(period, 10**6):]
    try:
        return load_yahoo(tickers, period)
    except Exception as e:  # network blocked, rate-limited, API change...
        st.error(f"Yahoo Finance download failed: {e}. Switch to *Demo* or *Upload CSV* "
                 "in the sidebar.")
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# Cached computations
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=30)
def run_analysis(prices: pd.Series, horizon, n_paths, model, drift, target, stop, n_states):
    return q.analyze(prices, horizon=horizon, n_paths=n_paths, model=model, drift=drift,
                     target=target, stop=stop, n_states=n_states)


@st.cache_data(show_spinner=False, max_entries=10)
def run_backtest(_a: q.Analysis, key: str, fwd: int):
    return q.signal_backtest(_a, fwd=fwd)


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def run_screener(prices: pd.DataFrame, horizon: int, target: float, stop: float):
    rows = []
    bar = st.progress(0.0, text="Scanning…")
    cols = list(prices.columns)
    for i, tk in enumerate(cols):
        s = prices[tk].dropna()
        bar.progress((i + 1) / len(cols), text=f"Scanning {NAME_OF.get(tk, tk)}…")
        if len(s) < 300:
            continue
        try:
            a = q.analyze(s, horizon=horizon, n_paths=2000, target=target, stop=stop,
                          n_states=3, hmm_years=4)
        except Exception:
            continue
        rows.append({
            "Ticker": tk, "Name": NAME_OF.get(tk, tk), "Last": a.stats["last"],
            "1M": a.stats["chg_1m"], "Regime": a.stats["regime"],
            "P(bear)": a.stats["p_bear_next"], "OU z": a.ou.z, "RSI": a.stats["rsi"],
            "Vol ratio": a.garch.vol_ratio, "Entry score": a.entry_score,
            "Decline risk": a.decline_score, "Verdict": a.verdict,
        })
    bar.empty()
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Data")
    st.radio("Source", ["Yahoo Finance", "Upload CSV", "Demo (synthetic)"], key="source",
             help="Yahoo = free, unofficial, may be rate-limited. CSV = Date column + one "
                  "close-price column per ticker (e.g. MC.PA). Demo = simulated prices.")
    if st.session_state.source == "Upload CSV":
        f = st.file_uploader("Price history (CSV)", type=["csv", "txt"])
        if f is not None:
            st.session_state.csv_df = parse_csv(f.getvalue())
            st.caption(f"{st.session_state.csv_df.shape[1]} series, "
                       f"{len(st.session_state.csv_df)} rows")
    period = st.selectbox("History", list(PERIOD_DAYS), index=2,
                          help="10y+ gives GARCH and HMM enough crises to learn from.")
    if st.button("↻ Refresh data"):
        st.cache_data.clear()
    st.divider()
    st.caption("Decision aid based on statistical models. Probabilities, not certainties – "
               "not investment advice.")

st.title("📈 CAC 40 Stochastic Lab")
tab1, tab2, tab3 = st.tabs(["🎯 Single share", "🧺 CAC 40 portfolio", "ℹ️ Method"])

# =========================================================================== #
# TAB 1 - single share
# =========================================================================== #
with tab1:
    c1, c2 = st.columns([2, 1])
    with c1:
        names = list(CAC40) + ["Other ticker…"]
        pick = st.selectbox("Share", names, index=names.index("LVMH"))
    with c2:
        other = st.text_input("Yahoo ticker", "SU.PA",
                              disabled=pick != "Other ticker…",
                              help="Paris shares end in .PA, e.g. GTT.PA")
    ticker = other.strip().upper() if pick == "Other ticker…" else CAC40[pick]

    with st.expander("Model settings", expanded=False):
        s1, s2, s3, s4 = st.columns(4)
        horizon = s1.slider("Horizon (trading days)", 5, 120, 20, 5)
        model = s2.selectbox("Monte Carlo model", ["GARCH-t", "GBM", "Bootstrap"],
                             help="GARCH-t: today's volatility + fat tails (recommended). "
                                  "GBM: constant vol, normal returns. Bootstrap: resamples "
                                  "real 5-day blocks of history.")
        drift = s3.selectbox("Drift", ["Zero", "Historical"],
                             help="Zero = no assumed trend (conservative). Historical = past "
                                  "average return continues.")
        n_states = s4.selectbox("HMM regimes", [2, 3, 4], index=1)
        s5, s6, s7 = st.columns(3)
        target = s5.number_input("Profit target %", 1.0, 50.0, 8.0, 0.5) / 100
        stop = s6.number_input("Stop / decline threshold %", 1.0, 50.0, 5.0, 0.5) / 100
        n_paths = s7.select_slider("Simulated paths", [1000, 2000, 5000, 10000, 20000], 5000)

    px = get_prices([ticker], period)
    if px.empty or ticker not in px or px[ticker].dropna().shape[0] < 300:
        st.warning(f"Not enough price history for {ticker} (need ≥ 300 trading days).")
        st.stop()
    s = px[ticker].dropna()

    with st.spinner("Fitting GARCH, HMM and simulating paths…"):
        a = run_analysis(s, horizon, n_paths, model, drift, target, stop, n_states)
    stt = a.stats

    # ---- headline -------------------------------------------------------------
    k = st.columns(6)
    k[0].metric("Last price", f"{stt['last']:,.2f}", pct(stt["chg_1d"], 2))
    k[1].metric("Entry score", f"{a.entry_score:.0f}/100")
    k[2].metric("Decline risk", f"{a.decline_score:.0f}/100")
    k[3].metric("Regime (next day)", stt["regime"], f"P(bear) {stt['p_bear_next']:.0%}",
                delta_color="off")
    k[4].metric("GARCH vol (ann.)", pct(a.garch.next_vol_ann, 0),
                f"long-run {pct(a.garch.long_run_vol_ann, 0)}", delta_color="off")
    k[5].metric(f"{horizon}d VaR 95%", pct(a.mc["var95"]),
                f"CVaR {pct(a.mc['cvar95'])}", delta_color="off")

    box = {"Favourable entry": st.success, "Neutral – scale in / use limits": st.info,
           "Unfavourable – wait": st.warning, "Decline risk – wait": st.error}[a.verdict]
    box(f"**{a.verdict}** · {NAME_OF.get(ticker, ticker)} ({ticker})")
    cl, cr = st.columns(2)
    with cl:
        st.markdown("**Why**")
        st.markdown("\n".join(f"- {r}" for r in a.reasons))
    with cr:
        st.markdown("**Decline warnings**")
        st.markdown("\n".join(f"- ⚠️ {w}" for w in a.warnings) if a.warnings
                    else "- No active decline warning.")
        comp = pd.DataFrame({"component": list(a.entry_components),
                             "score": list(a.entry_components.values()),
                             "weight": [q.ENTRY_WEIGHTS[c] for c in a.entry_components]})
        st.dataframe(comp, hide_index=True, width="stretch",
                     column_config={"score": st.column_config.ProgressColumn(
                         "score (0–1)", min_value=0, max_value=1, format="%.2f"),
                         "weight": st.column_config.NumberColumn(format="%.2f")})

    t_price, t_mc, t_vol, t_reg, t_lvl, t_bt = st.tabs(
        ["Price & regimes", "Monte Carlo", "Volatility (GARCH)", "Regimes (HMM)",
         "Entry levels", "Signal check"])

    # ---- price with regime shading ------------------------------------------
    with t_price:
        look = st.select_slider("Window", ["6M", "1Y", "2Y", "5Y", "All"], "2Y", key="win")
        n = {"6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260, "All": len(s)}[look]
        p = s.iloc[-n:]
        fig = go.Figure()
        state = a.hmm.filtered.reindex(p.index).idxmax(axis=1).ffill()
        seg = (state != state.shift()).cumsum()
        for _, g in state.groupby(seg):
            fig.add_vrect(x0=g.index[0], x1=g.index[-1], line_width=0, opacity=0.12,
                          fillcolor=REGIME_COL.get(g.iloc[0], C_GRAY))
        fig.add_scatter(x=p.index, y=p, name="Price", line=dict(color=C_BLUE, width=2))
        fig.add_scatter(x=p.index, y=s.rolling(50).mean().reindex(p.index), name="50-day MA",
                        line=dict(color=C_ORANGE, width=1.5))
        fig.add_scatter(x=p.index, y=s.rolling(200).mean().reindex(p.index),
                        name="200-day MA", line=dict(color=C_AQUA, width=1.5))
        if a.ou.mean_reverting:
            fig.add_hline(y=a.ou.fair_value, line_dash="dot", line_color=C_GRAY,
                          annotation_text="OU fair value")
        show(style(fig, 440, "Price"))
        st.caption("Background colour = most likely HMM regime that day (filtered, no "
                   "look-ahead): " + " · ".join(f"{l}" for l in a.hmm.labels) +
                   f". Hurst exponent {stt['hurst']:.2f} (<0.5 mean-reverting, >0.5 trending).")

    # ---- Monte Carlo ----------------------------------------------------------
    with t_mc:
        P = a.paths
        days = np.arange(P.shape[1])
        qs = np.quantile(P, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
        fig = go.Figure()
        hist = s.iloc[-60:]
        x_hist = np.arange(-len(hist) + 1, 1)
        fig.add_scatter(x=x_hist, y=hist.values, name="Last 60 days",
                        line=dict(color=C_GRAY, width=2))
        fig.add_scatter(x=days, y=qs[4], line=dict(width=0), showlegend=False,
                        hoverinfo="skip")
        fig.add_scatter(x=days, y=qs[0], fill="tonexty", line=dict(width=0),
                        fillcolor="rgba(42,120,214,0.15)", name="5–95%")
        fig.add_scatter(x=days, y=qs[3], line=dict(width=0), showlegend=False,
                        hoverinfo="skip")
        fig.add_scatter(x=days, y=qs[1], fill="tonexty", line=dict(width=0),
                        fillcolor="rgba(42,120,214,0.30)", name="25–75%")
        fig.add_scatter(x=days, y=qs[2], name="Median", line=dict(color=C_BLUE, width=2))
        for i in range(8):
            fig.add_scatter(x=days, y=P[i], line=dict(width=0.8, color="rgba(128,128,128,.4)"),
                            showlegend=False, hoverinfo="skip")
        S0 = P[0, 0]
        fig.add_hline(y=S0 * (1 + target), line_dash="dash", line_color=C_AQUA,
                      annotation_text=f"+{target:.0%}")
        fig.add_hline(y=S0 * (1 - stop), line_dash="dash", line_color=REGIME_COL["Bear"],
                      annotation_text=f"-{stop:.0%}")
        fig.update_xaxes(title="Trading days from today")
        show(style(fig, 440, "Price"))

        m = st.columns(5)
        m[0].metric("P(up at horizon)", pct(a.mc["p_up"], 0))
        m[1].metric("Median return", pct(a.mc["median_ret"]))
        m[2].metric(f"Hits +{target:.0%} first", pct(a.mc["p_target_first"], 0))
        m[3].metric(f"Hits −{stop:.0%} first", pct(a.mc["p_stop_first"], 0))
        m[4].metric(f"Touches −{stop:.0%}", pct(a.mc["p_touch_stop"], 0))

        tr = a.mc["term_returns"]
        fig = go.Figure(go.Histogram(x=tr * 100, nbinsx=80, marker_color=C_BLUE,
                                     marker_line=dict(color="white", width=1),
                                     name="Simulated return"))
        fig.add_vline(x=-a.mc["var95"] * 100, line_dash="dash",
                      line_color=REGIME_COL["Bear"], annotation_text="VaR 95%")
        fig.update_xaxes(title=f"Return after {horizon} days (%)")
        show(style(fig, 300, "Paths"))

    # ---- GARCH ----------------------------------------------------------------
    with t_vol:
        g = a.garch
        vol = g.sigma * np.sqrt(q.TRADING_DAYS) * 100
        rv = a.returns.rolling(21).std() * np.sqrt(q.TRADING_DAYS) * 100
        fig = go.Figure()
        fig.add_scatter(x=vol.index, y=vol, name="GARCH conditional vol",
                        line=dict(color=C_BLUE, width=2))
        fig.add_scatter(x=rv.index, y=rv, name="21-day realised vol",
                        line=dict(color=C_GRAY, width=1))
        fig.add_hline(y=g.long_run_vol_ann * 100, line_dash="dot", line_color=C_ORANGE,
                      annotation_text="long-run")
        show(style(fig, 380, "Annualised vol (%)"))
        fc = g.forecast_vol(max(horizon, 60)) * np.sqrt(q.TRADING_DAYS) * 100
        fig = go.Figure(go.Scatter(x=np.arange(1, len(fc) + 1), y=fc, name="Forecast",
                                   line=dict(color=C_BLUE, width=2)))
        fig.update_xaxes(title="Days ahead")
        show(style(fig, 260, "Forecast vol (%)"))
        st.markdown(
            f"α = {g.alpha:.3f} (reaction to shocks) · β = {g.beta:.3f} (memory) · "
            f"persistence α+β = {g.persistence:.3f} → a vol shock halves in "
            f"≈ {g.half_life_days:.0f} days · Student-t ν = {g.nu:.1f} "
            f"({'fat' if g.nu < 10 else 'moderate'} tails).")

    # ---- HMM ------------------------------------------------------------------
    with t_reg:
        f = a.hmm.filtered.iloc[-756:]
        fig = go.Figure()
        for lab in a.hmm.labels:
            fig.add_scatter(x=f.index, y=f[lab], name=lab, stackgroup="one",
                            line=dict(width=0.5, color=REGIME_COL.get(lab, C_GRAY)))
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        show(style(fig, 340, "Filtered probability"))
        reg = pd.DataFrame({
            "Regime": a.hmm.labels,
            "Daily mean": [f"{m:+.3%}" for m in a.hmm.means],
            "Ann. vol": [pct(v * np.sqrt(q.TRADING_DAYS), 0) for v in a.hmm.vols],
            "Avg duration (days)": a.hmm.expected_duration.round(0),
            "Now": [pct(x, 0) for x in a.hmm.current],
            "Tomorrow": [pct(x, 0) for x in a.hmm.next_day],
        })
        st.dataframe(reg, hide_index=True, width="stretch")
        st.markdown("**Transition matrix** (row = today, column = tomorrow)")
        st.dataframe(pd.DataFrame(a.hmm.trans, index=a.hmm.labels, columns=a.hmm.labels)
                     .style.format("{:.1%}"), width="stretch")

    # ---- entry levels ---------------------------------------------------------
    with t_lvl:
        lt = a.limits.copy()
        st.markdown(f"Buy-limit orders placed below today's price ({stt['last']:,.2f}), "
                    f"valid for {horizon} trading days, using the simulated paths.")
        fig = go.Figure(go.Bar(x=[f"-{d:.1%}" for d in lt.discount], y=lt.p_fill,
                               marker_color=C_BLUE, name="P(fill)",
                               text=[pct(v, 0) for v in lt.p_fill], textposition="outside"))
        fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
        fig.update_xaxes(title="Limit below current price")
        show(style(fig, 300, "Probability of fill"))
        st.dataframe(lt, hide_index=True, width="stretch", column_config={
            "discount": st.column_config.NumberColumn("Discount", format="percent"),
            "limit_price": st.column_config.NumberColumn("Limit price", format="%.2f"),
            "p_fill": st.column_config.NumberColumn("P(fill)", format="percent"),
            "p_gain_if_filled": st.column_config.NumberColumn("P(gain at horizon | filled)",
                                                              format="percent"),
            "exp_ret_if_filled": st.column_config.NumberColumn("Mean return | filled",
                                                               format="percent")})
        st.caption("Deeper limits fill less often. With zero drift, a fill often means the "
                   "share kept falling – compare the conditional gain probability before "
                   "choosing a level.")

    # ---- backtest -------------------------------------------------------------
    with t_bt:
        fwd = st.slider("Forward window (days)", 5, 60, 20, 5)
        tab, score = run_backtest(a, f"{ticker}-{period}-{n_states}-{len(s)}", fwd)
        st.dataframe(tab, width="stretch", column_config={
            "days": st.column_config.NumberColumn("Days"),
            "mean_fwd": st.column_config.NumberColumn(f"Mean {fwd}d return", format="percent"),
            "hit_rate": st.column_config.NumberColumn("Hit rate", format="percent"),
            "worst": st.column_config.NumberColumn("5% worst", format="percent")})
        fig = go.Figure(go.Scatter(x=score.index, y=score, name="Entry score",
                                   line=dict(color=C_BLUE, width=1.5)))
        fig.add_hline(y=65, line_dash="dot", line_color=C_AQUA)
        fig.add_hline(y=35, line_dash="dot", line_color=REGIME_COL["Bear"])
        show(style(fig, 280, "Entry score"))
        st.caption("Uses the score without its Monte Carlo part, computed day by day from "
                   "information available then. Model parameters are fitted on the full "
                   "history, so results are somewhat optimistic. A useful score shows a "
                   "higher mean return and hit rate in the top bucket than in 'All days'.")

# =========================================================================== #
# TAB 2 - portfolio
# =========================================================================== #
with tab2:
    st.subheader("My holdings")
    if "holdings" not in st.session_state:
        st.session_state.holdings = pd.DataFrame({
            "Ticker": ["MC.PA", "TTE.PA", "AIR.PA", "SAN.PA", "BNP.PA"],
            "Quantity": [10, 50, 20, 30, 40],
            "Avg cost": [600.0, 55.0, 160.0, 90.0, 60.0]})
    h = st.data_editor(st.session_state.holdings, num_rows="dynamic", width="stretch",
                       key="hold_editor", column_config={
                           "Ticker": st.column_config.TextColumn(help="Yahoo ticker, e.g. OR.PA"),
                           "Quantity": st.column_config.NumberColumn(min_value=0, step=1),
                           "Avg cost": st.column_config.NumberColumn(min_value=0, format="%.2f")})
    h = h.dropna(subset=["Ticker"]).assign(Ticker=lambda d: d.Ticker.str.strip().str.upper())
    h = h[(h.Quantity > 0)].groupby("Ticker", as_index=False).agg(
        Quantity=("Quantity", "sum"), **{"Avg cost": ("Avg cost", "mean")})

    pc1, pc2, pc3 = st.columns(3)
    var_h = pc1.slider("VaR horizon (days)", 1, 60, 10)
    rf = pc2.number_input("Risk-free rate % (Sharpe)", 0.0, 10.0, 2.0, 0.25) / 100
    reg_states = pc3.selectbox("Market regimes", [2, 3], index=1)

    if h.empty:
        st.info("Add at least one holding.")
    else:
        allpx = get_prices(list(h.Ticker) + [INDEX], period)
        missing = [t for t in h.Ticker if t not in allpx or allpx[t].dropna().empty]
        if missing:
            st.warning(f"No data for: {', '.join(missing)} – ignored.")
        h = h[~h.Ticker.isin(missing)]
        if not h.empty:
            px_h = allpx[list(h.Ticker)].dropna()
            last = px_h.iloc[-1]
            h["Name"] = h.Ticker.map(lambda t: NAME_OF.get(t, t))
            h["Last"] = h.Ticker.map(last)
            h["Value"] = h.Quantity * h.Last
            h["P&L"] = h.Quantity * (h.Last - h["Avg cost"])
            h["P&L %"] = h.Last / h["Avg cost"] - 1
            h["1D"] = h.Ticker.map(px_h.iloc[-1] / px_h.iloc[-2] - 1)
            h["Weight"] = h.Value / h.Value.sum()
            w = h.set_index("Ticker").Weight
            bench = allpx[INDEX].dropna() if INDEX in allpx else None
            ps = q.portfolio_stats(px_h, w, bench, rf=rf, horizon=var_h)

            # market regime banner
            if bench is not None and len(bench) > 500:
                mh = q.fit_hmm(q.log_returns(bench).iloc[-4 * q.TRADING_DAYS:], reg_states)
                lab = mh.labels[int(np.argmax(mh.next_day))]
                msg = (f"CAC 40 market regime: **{lab}** · P(bear tomorrow) "
                       f"{mh.next_day[0]:.0%}")
                (st.error if mh.next_day[0] > 0.5 else st.success if lab == "Bull"
                 else st.info)(msg)

            tot, pnl = h.Value.sum(), h["P&L"].sum()
            k = st.columns(6)
            k[0].metric("Value", f"€{tot:,.0f}",
                        f"{(h.Value * h['1D'] / (1 + h['1D'])).sum():+,.0f} today")
            k[1].metric("Unrealised P&L", f"€{pnl:,.0f}",
                        pct(pnl / (tot - pnl)))
            k[2].metric("Ann. vol", pct(ps["ann_vol"], 0))
            k[3].metric("Sharpe", f"{ps['sharpe']:.2f}")
            k[4].metric("Beta vs CAC 40", f"{ps['beta']:.2f}" if pd.notna(ps["beta"]) else "–")
            k[5].metric(f"{var_h}d VaR 95%", f"€{ps['var95'] * tot:,.0f}",
                        f"CVaR €{ps['cvar95'] * tot:,.0f}", delta_color="off")

            st.dataframe(h[["Ticker", "Name", "Quantity", "Avg cost", "Last", "Value", "P&L",
                            "P&L %", "1D", "Weight"]], hide_index=True, width="stretch",
                         column_config={
                             "Avg cost": st.column_config.NumberColumn(format="%.2f"),
                             "Last": st.column_config.NumberColumn(format="%.2f"),
                             "Value": st.column_config.NumberColumn(format="€%.0f"),
                             "P&L": st.column_config.NumberColumn(format="€%.0f"),
                             "P&L %": st.column_config.NumberColumn(format="percent"),
                             "1D": st.column_config.NumberColumn(format="percent"),
                             "Weight": st.column_config.ProgressColumn(
                                 format="percent", min_value=0, max_value=1)})

            g1, g2 = st.columns(2)
            with g1:
                cum = ps["cum"]
                fig = go.Figure()
                fig.add_scatter(x=cum.index, y=cum * 100, name="Portfolio (today's weights)",
                                line=dict(color=C_BLUE, width=2))
                if bench is not None:
                    b = bench.reindex(cum.index).dropna()
                    fig.add_scatter(x=b.index, y=b / b.iloc[0] * 100, name="CAC 40",
                                    line=dict(color=C_GRAY, width=1.5))
                show(style(fig, 340, "Growth of 100"))
                fig = go.Figure(go.Scatter(x=ps["dd"].index, y=ps["dd"] * 100, fill="tozeroy",
                                           line=dict(color=REGIME_COL["Bear"], width=1),
                                           name="Drawdown"))
                show(style(fig, 220, f"Drawdown % (max {ps['mdd']:.0%})"))
            with g2:
                rc = pd.DataFrame({"Weight": w, "Risk contribution": ps["risk_contrib"]})
                fig = go.Figure()
                fig.add_bar(x=rc.index, y=rc.Weight, name="Weight", marker_color=C_GRAY)
                fig.add_bar(x=rc.index, y=rc["Risk contribution"], name="Risk contribution",
                            marker_color=C_ORANGE)
                fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.05)
                fig.update_yaxes(tickformat=".0%")
                show(style(fig, 300, "Share of total"))
                corr = ps["corr"]
                fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                                           zmin=-1, zmax=1, colorscale="RdBu_r",
                                           text=corr.round(2).values, texttemplate="%{text}"))
                show(style(fig, 280, None).update_layout(hovermode="closest"))
            st.caption("Historical risk uses today's weights applied to past returns. VaR is "
                       "from a 5-day block bootstrap of joint returns, which keeps the "
                       "correlations between holdings.")

    # ---- screener -----------------------------------------------------------
    st.divider()
    st.subheader("CAC 40 entry screener")
    with st.expander("Constituents (indicative – check the latest Euronext review)"):
        txt = st.text_area("Yahoo tickers, comma-separated",
                           ", ".join(CAC40.values()), height=110)
    universe = [t.strip().upper() for t in txt.split(",") if t.strip()]
    sc1, sc2, sc3 = st.columns(3)
    s_h = sc1.slider("Horizon", 5, 60, 20, 5, key="sh")
    s_t = sc2.number_input("Target %", 1.0, 30.0, 8.0, 0.5, key="st") / 100
    s_s = sc3.number_input("Stop %", 1.0, 30.0, 5.0, 0.5, key="ss") / 100
    if st.button(f"Scan {len(universe)} shares", type="primary"):
        upx = get_prices(universe, period)
        if upx.empty:
            st.error("No data returned.")
        else:
            st.session_state.scan = run_screener(upx, s_h, s_t, s_s)
    if "scan" in st.session_state and not st.session_state.scan.empty:
        sc = st.session_state.scan.sort_values("Entry score", ascending=False)
        st.dataframe(sc, hide_index=True, width="stretch", height=520, column_config={
            "Last": st.column_config.NumberColumn(format="%.2f"),
            "1M": st.column_config.NumberColumn(format="percent"),
            "P(bear)": st.column_config.NumberColumn(format="percent"),
            "OU z": st.column_config.NumberColumn(format="%.2f"),
            "RSI": st.column_config.NumberColumn(format="%.0f"),
            "Vol ratio": st.column_config.NumberColumn(format="%.2f"),
            "Entry score": st.column_config.ProgressColumn(min_value=0, max_value=100,
                                                           format="%.0f"),
            "Decline risk": st.column_config.ProgressColumn(min_value=0, max_value=100,
                                                            format="%.0f")})
        st.caption("Open a share in the Single share tab for the full analysis.")

# =========================================================================== #
# TAB 3 - method
# =========================================================================== #
with tab3:
    st.markdown(r"""
### Models
| Model | What it captures | Used for |
|---|---|---|
| **GARCH(1,1), Student-t** | volatility clustering & fat tails: $\sigma^2_t=\omega+\alpha\varepsilon^2_{t-1}+\beta\sigma^2_{t-1}$ | today's risk, vol forecast, Monte Carlo |
| **Gaussian HMM** (Baum-Welch) | hidden market regimes with their own mean & volatility | bear probability, regime shading |
| **Monte Carlo** (GARCH-t / GBM / bootstrap) | distribution of future paths | VaR, CVaR, target-vs-stop odds, limit fill probabilities |
| **Ornstein-Uhlenbeck** on log price (120 days) | pull back towards a fair value | how stretched the price is (z-score) |
| Moving averages, RSI, Hurst | trend, momentum, persistence | trend filter, overbought check |

### Entry score (0–100)
Weighted mix of: regime (1 − P(bear tomorrow), 30%), discount to OU fair value (25%),
trend above the 200-day average (15%), calm volatility vs long run (15%), Monte Carlo
odds of hitting the target before the stop compared with a pure random walk (15%).

### Decline-risk score (0–100)
P(bear regime) 35%, probability of touching the stop 25%, price below 200-day / death
cross 15%, volatility stress 15%, RSI overbought 10%.

### Verdicts
*Favourable entry*: entry ≥ 65 and decline < 40 · *Decline risk – wait*: decline ≥ 55 ·
*Neutral*: entry ≥ 50 · otherwise *Unfavourable*.

### Limits
Filtered probabilities use only past data, but parameters are estimated on the full
history. Models assume the future resembles the past; earnings, dividends (prices are
dividend-adjusted on Yahoo) and news shocks are not modelled. Check the **Signal check**
tab before relying on the score.
""")
