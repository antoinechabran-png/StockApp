# CAC 40 Stochastic Lab

Streamlit app for stochastic analysis of Paris-listed shares.

* **Single share** – GARCH(1,1)-t volatility, Hidden Markov regimes, Monte Carlo
  (GARCH-t / GBM / bootstrap), Ornstein-Uhlenbeck mean reversion → entry score,
  decline-risk score, buy-limit fill probabilities, and a historical signal check.
* **CAC 40 portfolio** – holdings P&L, VaR/CVaR, beta, drawdown, risk contributions,
  correlation, CAC 40 market regime, and an entry screener over the 40 constituents.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Deploy on Streamlit Community Cloud
Push `app.py`, `quant.py`, `requirements.txt` to a GitHub repo → share.streamlit.io →
New app → main file `app.py`.

## Data
* **Yahoo Finance** (default, free, unofficial; tickers like `MC.PA`, index `^FCHI`).
* **Upload CSV** – a `Date` column plus one close-price column per ticker
  (column names = tickers). Use this if Yahoo is blocked on your network.
* **Demo** – simulated prices, for trying the app offline.

The CAC 40 list in `app.py` is indicative: update it after each quarterly Euronext review
(editable in the Portfolio tab too).

Decision aid only – not investment advice.
