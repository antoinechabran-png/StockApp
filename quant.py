"""
quant.py - stochastic models behind the share analysis app.

Pure numpy / pandas / scipy (no compiled extras), so it installs anywhere
Streamlit runs, including restricted corporate sandboxes.

Models
------
* GARCH(1,1) with Student-t innovations  (volatility clustering, fat tails)
* Gaussian Hidden Markov Model           (bull / neutral / bear regimes)
* Monte Carlo price paths                (GARCH-t, GBM, or block bootstrap)
* Ornstein-Uhlenbeck mean reversion      (how stretched the price is)
* Hurst exponent, RSI, moving averages   (trend / momentum context)

All signals use information available at the decision date only, except that
model parameters are estimated on the full history (stated in the backtest).
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, signal, special, stats

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Basics
# --------------------------------------------------------------------------- #
def log_returns(prices: pd.Series) -> pd.Series:
    prices = prices.dropna()
    return np.log(prices).diff().dropna()


def max_drawdown(prices: pd.Series) -> tuple[float, pd.Series]:
    peak = prices.cummax()
    dd = prices / peak - 1.0
    return float(dd.min()), dd


def rsi(prices: pd.Series, n: int = 14) -> pd.Series:
    d = prices.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def hurst_exponent(prices: pd.Series, max_lag: int = 50) -> float:
    """H < 0.5 mean-reverting, ~0.5 random walk, > 0.5 trending."""
    x = np.log(prices.dropna().values[-750:])
    lags = np.arange(2, min(max_lag, len(x) // 4))
    tau = np.array([np.std(x[lag:] - x[:-lag]) for lag in lags])
    ok = tau > 0
    if ok.sum() < 5:
        return np.nan
    return float(np.polyfit(np.log(lags[ok]), np.log(tau[ok]), 1)[0])


# --------------------------------------------------------------------------- #
# GARCH(1,1) Student-t
# --------------------------------------------------------------------------- #
@dataclass
class GarchFit:
    mu: float          # daily mean, % units
    omega: float
    alpha: float
    beta: float
    nu: float          # Student-t degrees of freedom
    sigma: pd.Series   # conditional daily vol (decimal), aligned to returns
    next_var: float    # one-step-ahead variance, %^2
    loglik: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_var(self) -> float:
        return self.omega / max(1e-9, 1 - self.persistence)

    @property
    def long_run_vol_ann(self) -> float:
        return np.sqrt(self.long_run_var * TRADING_DAYS) / 100

    @property
    def next_vol_ann(self) -> float:
        return np.sqrt(self.next_var * TRADING_DAYS) / 100

    @property
    def vol_ratio(self) -> float:
        """Next-day vol / long-run vol (>1 = stressed, <1 = calm)."""
        return float(np.sqrt(self.next_var / self.long_run_var))

    @property
    def half_life_days(self) -> float:
        p = self.persistence
        return float(np.log(0.5) / np.log(p)) if 0 < p < 1 else np.inf

    def forecast_vol(self, horizon: int) -> np.ndarray:
        """Daily vol forecasts (decimal) for steps 1..horizon."""
        k = np.arange(horizon)
        var = self.long_run_var + self.persistence ** k * (self.next_var - self.long_run_var)
        return np.sqrt(var) / 100


def _garch_var(params, r):
    mu, omega, alpha, beta, _ = params
    eps = r - mu
    s0 = eps.var()
    x = omega + alpha * np.r_[s0, eps[:-1] ** 2]
    sigma2 = signal.lfilter([1.0], [1.0, -beta], x, zi=[beta * s0])[0]
    return eps, np.maximum(sigma2, 1e-12)


def _garch_nll(params, r):
    nu = params[4]
    eps, s2 = _garch_var(params, r)
    z2 = eps ** 2 / s2
    ll = (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2)
          - 0.5 * np.log(np.pi * (nu - 2)) - 0.5 * np.log(s2)
          - (nu + 1) / 2 * np.log1p(z2 / (nu - 2)))
    val = -ll.sum()
    return val if np.isfinite(val) else 1e12


def fit_garch(returns: pd.Series) -> GarchFit:
    r = returns.values * 100.0
    v = r.var()
    x0 = [r.mean(), v * 0.02, 0.08, 0.90, 8.0]
    bounds = [(-1, 1), (1e-6, 10 * v), (1e-6, 0.5), (0.0, 0.999), (2.1, 60.0)]
    cons = [{"type": "ineq", "fun": lambda p: 0.998 - p[2] - p[3]}]
    res = optimize.minimize(_garch_nll, x0, args=(r,), method="SLSQP",
                            bounds=bounds, constraints=cons,
                            options={"maxiter": 300, "ftol": 1e-9})
    p = res.x
    eps, s2 = _garch_var(p, r)
    next_var = p[1] + p[2] * eps[-1] ** 2 + p[3] * s2[-1]
    return GarchFit(mu=p[0], omega=p[1], alpha=p[2], beta=p[3], nu=p[4],
                    sigma=pd.Series(np.sqrt(s2) / 100, index=returns.index),
                    next_var=float(next_var), loglik=-float(res.fun))


# --------------------------------------------------------------------------- #
# Gaussian Hidden Markov Model (Baum-Welch), states sorted Bear -> Bull
# --------------------------------------------------------------------------- #
@dataclass
class HMMFit:
    means: np.ndarray        # daily mean return (decimal) per state
    vols: np.ndarray         # daily vol (decimal) per state
    trans: np.ndarray        # transition matrix
    filtered: pd.DataFrame   # P(state_t | returns up to t)  - no look-ahead
    labels: list[str]
    loglik: float

    @property
    def current(self) -> np.ndarray:
        return self.filtered.iloc[-1].values

    @property
    def next_day(self) -> np.ndarray:
        return self.current @ self.trans

    @property
    def expected_duration(self) -> np.ndarray:
        return 1.0 / np.maximum(1e-9, 1 - np.diag(self.trans))

    @property
    def bear_prob_series(self) -> pd.Series:
        return self.filtered.iloc[:, 0]


def _state_labels(k: int) -> list[str]:
    return {2: ["Bear", "Bull"], 3: ["Bear", "Neutral", "Bull"],
            4: ["Bear", "Correction", "Calm", "Bull"]}.get(k, [f"S{i}" for i in range(k)])


def fit_hmm(returns: pd.Series, n_states: int = 3, n_iter: int = 80,
            tol: float = 1e-4) -> HMMFit:
    r = returns.values * 100.0
    T, K = len(r), n_states
    # initialise from quantile chunks of sorted returns
    chunks = np.array_split(np.sort(r), K)
    mu = np.array([c.mean() for c in chunks])
    sd = np.array([max(c.std(), 0.3) for c in chunks])
    sd = np.maximum(sd, 0.5 * r.std())
    A = np.full((K, K), 0.05 / (K - 1)) + np.eye(K) * (0.95 - 0.05 / (K - 1))
    pi = np.full(K, 1.0 / K)
    prev = -np.inf
    alpha = np.zeros((T, K))
    beta = np.zeros((T, K))
    c = np.zeros(T)
    for _ in range(n_iter):
        B = np.maximum(stats.norm.pdf(r[:, None], mu, sd), 1e-300)
        # forward (scaled)
        a = pi * B[0]
        c[0] = a.sum()
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ A) * B[t]
            s = a.sum()
            c[t] = s
            alpha[t] = a / s
        # backward
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(1, keepdims=True)
        xi = (alpha[:-1, :, None] * A[None] * (B[1:] * beta[1:])[:, None, :]
              / c[1:, None, None]).sum(0)
        # M-step
        pi = gamma[0]
        A = xi / xi.sum(1, keepdims=True)
        w = gamma.sum(0)
        mu = (gamma * r[:, None]).sum(0) / w
        sd = np.sqrt(np.maximum((gamma * (r[:, None] - mu) ** 2).sum(0) / w, 0.01))
        ll = np.log(c).sum()
        if ll - prev < tol:
            break
        prev = ll
    # final filtered probabilities with final parameters
    B = np.maximum(stats.norm.pdf(r[:, None], mu, sd), 1e-300)
    a = pi * B[0]
    alpha[0] = a / a.sum()
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * B[t]
        alpha[t] = a / a.sum()
    order = np.argsort(mu / sd)  # Bear = worst return per unit of risk
    labels = _state_labels(K)
    filt = pd.DataFrame(alpha[:, order], index=returns.index, columns=labels)
    return HMMFit(means=mu[order] / 100, vols=sd[order] / 100,
                  trans=A[np.ix_(order, order)], filtered=filt,
                  labels=labels, loglik=float(ll))


# --------------------------------------------------------------------------- #
# Mean reversion (discrete Ornstein-Uhlenbeck / AR(1) on log price)
# --------------------------------------------------------------------------- #
@dataclass
class OUFit:
    mean_reverting: bool
    half_life: float
    fair_value: float
    z: float


def fit_ou(prices: pd.Series, window: int = 120) -> OUFit:
    x = np.log(prices.dropna().values[-window:])
    dx, xl = np.diff(x), x[:-1]
    b, a = np.polyfit(xl, dx, 1)
    resid = dx - (a + b * xl)
    phi = 1 + b
    half_life = np.log(2) / -b if b < 0 else np.inf
    # only trust the OU fit when reversion is fast relative to the window;
    # a weak pull (trending price) puts the "fair value" far outside the data
    if b >= 0 or phi <= 0 or half_life > window / 2:
        # no mean reversion: fall back to a rolling z-score
        z = (x[-1] - x.mean()) / (x.std() + 1e-12)
        return OUFit(False, np.inf, float(np.exp(x.mean())), float(np.clip(z, -4, 4)))
    sd_stat = resid.std() / np.sqrt(max(1e-9, 1 - phi ** 2))
    mean = -a / b
    return OUFit(True, float(half_life), float(np.exp(mean)),
                 float(np.clip((x[-1] - mean) / sd_stat, -4, 4)))


# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #
def _std_t(rng, nu, size):
    return rng.standard_t(nu, size) * np.sqrt((nu - 2) / nu)


def simulate_paths(prices: pd.Series, garch: GarchFit, model: str = "GARCH-t",
                   horizon: int = 20, n_paths: int = 5000, drift: str = "Zero",
                   seed: int = 42) -> np.ndarray:
    """Return price paths of shape (n_paths, horizon + 1)."""
    rng = np.random.default_rng(seed)
    S0 = float(prices.iloc[-1])
    rets = log_returns(prices).values
    hist_mu = rets.mean()
    mu = hist_mu if drift == "Historical" else 0.0

    if model == "GBM":
        sig = rets[-TRADING_DAYS:].std()
        z = rng.standard_normal((n_paths, horizon))
        logr = (mu - 0.5 * sig ** 2) + sig * z
    elif model == "Bootstrap":
        block = 5
        base = rets - hist_mu + mu
        n_blocks = int(np.ceil(horizon / block))
        starts = rng.integers(0, len(base) - block, (n_paths, n_blocks))
        idx = (starts[:, :, None] + np.arange(block)).reshape(n_paths, -1)[:, :horizon]
        logr = base[idx]
    else:  # GARCH-t
        var = np.full(n_paths, garch.next_var)
        logr = np.empty((n_paths, horizon))
        for k in range(horizon):
            eps = np.sqrt(var) * _std_t(rng, garch.nu, n_paths)
            logr[:, k] = (mu * 100 + eps) / 100 - 0.5 * var / 1e4
            var = garch.omega + garch.alpha * eps ** 2 + garch.beta * var
    paths = S0 * np.exp(np.cumsum(logr, axis=1))
    return np.hstack([np.full((n_paths, 1), S0), paths])


def mc_summary(paths: np.ndarray, target: float, stop: float) -> dict:
    S0 = paths[0, 0]
    term = paths[:, -1] / S0 - 1
    up, dn = S0 * (1 + target), S0 * (1 - stop)
    H = paths.shape[1]
    hit_up = paths >= up
    hit_dn = paths <= dn
    i_up = np.where(hit_up.any(1), hit_up.argmax(1), H + 1)
    i_dn = np.where(hit_dn.any(1), hit_dn.argmax(1), H + 1)
    q05 = np.quantile(term, 0.05)
    return {
        "p_up": float((term > 0).mean()),
        "exp_ret": float(term.mean()),
        "median_ret": float(np.median(term)),
        "q": {p: float(np.quantile(term, p)) for p in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "var95": float(-q05),
        "cvar95": float(-term[term <= q05].mean()),
        "p_target_first": float((i_up < i_dn).mean()),
        "p_stop_first": float((i_dn < i_up).mean()),
        "p_touch_stop": float(hit_dn.any(1).mean()),
        "term_returns": term,
    }


def limit_fill_table(paths: np.ndarray, discounts=(0.01, 0.02, 0.03, 0.05, 0.075, 0.10)) -> pd.DataFrame:
    """For a buy-limit placed d% below today: fill probability and outcome."""
    S0 = paths[0, 0]
    mins = paths.min(1)
    term = paths[:, -1]
    rows = []
    for d in discounts:
        lvl = S0 * (1 - d)
        filled = mins <= lvl
        p = filled.mean()
        if filled.any():
            r = term[filled] / lvl - 1
            rows.append((d, lvl, p, (r > 0).mean(), r.mean()))
        else:
            rows.append((d, lvl, 0.0, np.nan, np.nan))
    return pd.DataFrame(rows, columns=["discount", "limit_price", "p_fill",
                                       "p_gain_if_filled", "exp_ret_if_filled"])


# --------------------------------------------------------------------------- #
# Signals: entry score & decline-risk score
# --------------------------------------------------------------------------- #
ENTRY_WEIGHTS = {"regime": 0.30, "discount": 0.25, "trend": 0.15, "vol": 0.15, "mc": 0.15}


def _entry_components(p_bear, z, above_200, vol_ratio, mc_edge=None):
    comp = {
        "regime": 1.0 - p_bear,
        "discount": np.clip(0.5 - z / 4.0, 0, 1),      # z=-2 -> 1, z=+2 -> 0
        "trend": np.where(above_200, 1.0, 0.3),          # avoid falling knives
        "vol": np.clip(1.5 - vol_ratio, 0, 1),           # calm vol preferred
    }
    if mc_edge is not None:
        comp["mc"] = mc_edge
    return comp


def _weighted(comp):
    w = {k: ENTRY_WEIGHTS[k] for k in comp}
    tot = sum(w.values())
    return 100 * sum(w[k] * comp[k] for k in comp) / tot


@dataclass
class Analysis:
    prices: pd.Series
    returns: pd.Series
    garch: GarchFit
    hmm: HMMFit
    ou: OUFit
    paths: np.ndarray
    mc: dict
    limits: pd.DataFrame
    entry_score: float
    entry_components: dict
    decline_score: float
    verdict: str
    reasons: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def analyze(prices: pd.Series, horizon: int = 20, n_paths: int = 5000,
            model: str = "GARCH-t", drift: str = "Zero", target: float = 0.08,
            stop: float = 0.05, n_states: int = 3, hmm_years: float | None = None,
            seed: int = 42) -> Analysis:
    prices = prices.dropna().astype(float)
    rets = log_returns(prices)
    garch = fit_garch(rets)
    hmm_rets = rets if hmm_years is None else rets.iloc[-int(hmm_years * TRADING_DAYS):]
    hmm = fit_hmm(hmm_rets, n_states=n_states)
    ou = fit_ou(prices)
    paths = simulate_paths(prices, garch, model, horizon, n_paths, drift, seed)
    mc = mc_summary(paths, target, stop)
    limits = limit_fill_table(paths)

    last = float(prices.iloc[-1])
    ma50 = prices.rolling(50).mean().iloc[-1]
    ma200 = prices.rolling(200).mean().iloc[-1]
    rsi_now = float(rsi(prices).iloc[-1])
    mdd, dd = max_drawdown(prices.iloc[-TRADING_DAYS:])
    p_bear_next = float(hmm.next_day[0])
    # edge vs a driftless random walk (whose odds are stop / (target + stop))
    hit = mc["p_target_first"] / max(1e-9, mc["p_target_first"] + mc["p_stop_first"])
    edge = float(np.clip(0.5 + 2 * (hit - stop / (target + stop)), 0, 1))

    comp = _entry_components(p_bear_next, ou.z, last > ma200, garch.vol_ratio, edge)
    comp = {k: float(v) for k, v in comp.items()}
    entry = float(_weighted(comp))

    # ---- decline-risk score ------------------------------------------------
    death_cross = ma50 < ma200
    d_comp = {
        "bear regime": (0.35, p_bear_next),
        "MC stop touch": (0.25, mc["p_touch_stop"]),
        "below 200d MA": (0.15, float(last < ma200) * (0.6 + 0.4 * death_cross)),
        "vol stress": (0.15, float(np.clip(garch.vol_ratio - 0.8, 0, 1))),
        "overbought": (0.10, float(np.clip((rsi_now - 60) / 20, 0, 1))),
    }
    decline = float(100 * sum(w * v for w, v in d_comp.values()))

    reasons, warns = [], []
    lab = hmm.labels[int(np.argmax(hmm.next_day))]
    reasons.append(f"HMM regime: **{lab}** most likely tomorrow "
                   f"(bear probability {p_bear_next:.0%}).")
    if ou.mean_reverting:
        reasons.append(f"Mean reversion: price is {ou.z:+.2f} σ from its OU fair value "
                       f"{ou.fair_value:,.2f} (half-life ≈ {ou.half_life:.0f} days).")
    else:
        reasons.append(f"No clear mean reversion on 120 days; rolling z-score {ou.z:+.2f}.")
    reasons.append(f"Trend: price {'above' if last > ma200 else 'below'} its 200-day average "
                   f"({ma200:,.2f}).")
    reasons.append(f"GARCH volatility {garch.next_vol_ann:.0%} annualised vs long-run "
                   f"{garch.long_run_vol_ann:.0%} (ratio {garch.vol_ratio:.2f}).")
    reasons.append(f"Monte Carlo ({model}, {horizon} days): +{target:.0%} reached before "
                   f"-{stop:.0%} in {mc['p_target_first']:.0%} of paths vs "
                   f"{mc['p_stop_first']:.0%} the other way.")

    if p_bear_next > 0.5:
        warns.append("Bear regime is the most likely state – declines tend to cluster.")
    if death_cross and last < ma200:
        warns.append("Death cross: 50-day average below 200-day and price below both.")
    if garch.vol_ratio > 1.3:
        warns.append("Volatility is well above its long-run level – wide swings likely.")
    if rsi_now > 70:
        warns.append(f"RSI {rsi_now:.0f}: overbought, pull-back risk.")
    if mc["p_touch_stop"] > 0.45:
        warns.append(f"{mc['p_touch_stop']:.0%} chance of touching -{stop:.0%} within "
                     f"{horizon} days.")

    if entry >= 65 and decline < 40:
        verdict = "Favourable entry"
    elif decline >= 55:
        verdict = "Decline risk – wait"
    elif entry >= 50:
        verdict = "Neutral – scale in / use limits"
    else:
        verdict = "Unfavourable – wait"

    st_ = {
        "last": last, "chg_1d": float(prices.iloc[-1] / prices.iloc[-2] - 1),
        "chg_1m": float(prices.iloc[-1] / prices.iloc[-22] - 1) if len(prices) > 22 else np.nan,
        "ma50": float(ma50), "ma200": float(ma200), "rsi": rsi_now,
        "hurst": hurst_exponent(prices), "mdd_1y": mdd,
        "regime": lab, "p_bear_next": p_bear_next,
        "decline_components": {k: v[1] for k, v in d_comp.items()},
    }
    return Analysis(prices, rets, garch, hmm, ou, paths, mc, limits, entry, comp,
                    decline, verdict, reasons, warns, st_)


# --------------------------------------------------------------------------- #
# Historical check of the entry score (without the Monte Carlo component)
# --------------------------------------------------------------------------- #
def signal_backtest(a: Analysis, fwd: int = 20, hi: float = 65, lo: float = 35):
    p = a.prices
    logp = np.log(p)
    z = (logp - logp.rolling(120).mean()) / logp.rolling(120).std()
    above = p > p.rolling(200).mean()
    vol_ratio = a.garch.sigma / (np.sqrt(a.garch.long_run_var) / 100)
    # use the next-day bear probability implied by the filtered state
    nxt = a.hmm.filtered.values @ a.hmm.trans
    bear = pd.Series(nxt[:, 0], index=a.hmm.filtered.index)
    df = pd.DataFrame({"z": z, "above": above, "vr": vol_ratio, "bear": bear}).dropna()
    comp = _entry_components(df["bear"].values, df["z"].values,
                             df["above"].values.astype(bool), df["vr"].values)
    df["score"] = _weighted(comp)
    df["fwd"] = p.shift(-fwd).reindex(df.index) / p.reindex(df.index) - 1
    df = df.dropna(subset=["fwd"])
    df["bucket"] = np.where(df["score"] >= hi, f"Score ≥ {hi:.0f}",
                            np.where(df["score"] <= lo, f"Score ≤ {lo:.0f}", "In between"))
    tab = df.groupby("bucket")["fwd"].agg(days="count", mean_fwd="mean",
                                          hit_rate=lambda s: (s > 0).mean(),
                                          worst=lambda s: s.quantile(0.05))
    allrow = pd.DataFrame({"days": [len(df)], "mean_fwd": [df["fwd"].mean()],
                           "hit_rate": [(df["fwd"] > 0).mean()],
                           "worst": [df["fwd"].quantile(0.05)]}, index=["All days"])
    order = [f"Score ≥ {hi:.0f}", "In between", f"Score ≤ {lo:.0f}"]
    tab = pd.concat([tab.reindex([o for o in order if o in tab.index]), allrow])
    return tab, df["score"]


# --------------------------------------------------------------------------- #
# Portfolio
# --------------------------------------------------------------------------- #
def portfolio_stats(prices: pd.DataFrame, weights: pd.Series, bench: pd.Series | None,
                    rf: float = 0.0, horizon: int = 10, n_paths: int = 10000,
                    seed: int = 7) -> dict:
    rets = np.log(prices[weights.index]).diff().dropna()
    simple = np.expm1(rets)
    port = simple @ weights.values
    cum = (1 + port).cumprod()
    ann_ret = port.mean() * TRADING_DAYS
    ann_vol = port.std() * np.sqrt(TRADING_DAYS)
    mdd, dd = max_drawdown(cum)
    cov = simple.cov().values * TRADING_DAYS
    w = weights.values
    pvar = w @ cov @ w
    rc = pd.Series(w * (cov @ w) / pvar, index=weights.index)
    beta = np.nan
    if bench is not None:
        b = np.expm1(np.log(bench).diff()).reindex(port.index).dropna()
        pp = port.reindex(b.index)
        beta = float(np.cov(pp, b)[0, 1] / b.var())
    # block-bootstrap of joint daily returns keeps the correlations
    rng = np.random.default_rng(seed)
    R = rets.values
    block = 5
    nb = int(np.ceil(horizon / block))
    starts = rng.integers(0, len(R) - block, (n_paths, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_paths, -1)[:, :horizon]
    asset_tot = np.expm1(R[idx].sum(1))          # (n_paths, n_assets)
    term = asset_tot @ w
    q05 = np.quantile(term, 0.05)
    return {
        "daily": port, "cum": cum, "dd": dd, "ann_ret": ann_ret, "ann_vol": ann_vol,
        "sharpe": (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan,
        "mdd": mdd, "beta": beta, "risk_contrib": rc,
        "corr": simple.corr(), "var95": -q05, "cvar95": -term[term <= q05].mean(),
        "mc_term": term,
    }


# --------------------------------------------------------------------------- #
# Demo data (regime-switching, market factor) for offline testing
# --------------------------------------------------------------------------- #
def synthetic_prices(tickers, n_days: int = 2520, end=None) -> pd.DataFrame:
    end = pd.Timestamp.today().normalize() if end is None else pd.Timestamp(end)
    idx = pd.bdate_range(end=end, periods=n_days)
    rng = np.random.default_rng(2026)
    # market regime chain: calm / stress
    P = np.array([[0.985, 0.015], [0.04, 0.96]])
    s = np.zeros(n_days, int)
    for t in range(1, n_days):
        s[t] = rng.random() > P[s[t - 1], 0]
    mkt_mu, mkt_sd = np.array([0.0006, -0.0012]), np.array([0.009, 0.022])
    mkt = mkt_mu[s] + mkt_sd[s] * rng.standard_t(5, n_days) * np.sqrt(3 / 5)
    out = {}
    for tk in tickers:
        if tk == "^FCHI":
            r = mkt
            p0 = 5000.0
        else:
            g = np.random.default_rng(zlib.crc32(tk.encode()))
            beta = g.uniform(0.6, 1.4)
            idio = g.uniform(0.008, 0.02) * g.standard_t(4, n_days) * np.sqrt(2 / 4)
            r = g.uniform(-0.0002, 0.0004) + beta * mkt + idio
            p0 = float(g.uniform(15, 600))
        out[tk] = p0 * np.exp(np.cumsum(r))
    return pd.DataFrame(out, index=idx)
