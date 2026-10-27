"""
isin.py - ISIN validation and ISIN -> Yahoo ticker resolution.

Resolution order
1. Built-in table of CAC 40 ISINs (works offline)
2. OpenFIGI mapping API (free, no key needed for light use)
3. Yahoo Finance search (yfinance.Search)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# CAC 40 constituents: ISIN -> (Yahoo ticker, name)
CAC40_ISIN = {
    "FR0000120404": ("AC.PA", "Accor"),
    "FR0000120073": ("AI.PA", "Air Liquide"),
    "NL0000235190": ("AIR.PA", "Airbus"),
    "LU1598757687": ("MT.AS", "ArcelorMittal"),
    "FR0000120628": ("CS.PA", "AXA"),
    "FR0000131104": ("BNP.PA", "BNP Paribas"),
    "FR0013280286": ("BIM.PA", "bioMérieux"),
    "FR0000120503": ("EN.PA", "Bouygues"),
    "FR0006174348": ("BVI.PA", "Bureau Veritas"),
    "FR0000125338": ("CAP.PA", "Capgemini"),
    "FR0000120172": ("CA.PA", "Carrefour"),
    "FR0000045072": ("ACA.PA", "Crédit Agricole"),
    "FR0000120644": ("BN.PA", "Danone"),
    "FR0014003TT8": ("DSY.PA", "Dassault Systèmes"),
    "FR0010908533": ("EDEN.PA", "Edenred"),
    "FR0010208488": ("ENGI.PA", "Engie"),
    "FR0000121667": ("EL.PA", "EssilorLuxottica"),
    "FR0014000MR3": ("ERF.PA", "Eurofins Scientific"),
    "FR0000052292": ("RMS.PA", "Hermès"),
    "FR0000121485": ("KER.PA", "Kering"),
    "FR0000120321": ("OR.PA", "L'Oréal"),
    "FR0010307819": ("LR.PA", "Legrand"),
    "FR0000121014": ("MC.PA", "LVMH"),
    "FR001400AJ45": ("ML.PA", "Michelin"),
    "FR0000133308": ("ORA.PA", "Orange"),
    "FR0000120693": ("RI.PA", "Pernod Ricard"),
    "FR0000130577": ("PUB.PA", "Publicis"),
    "FR0000131906": ("RNO.PA", "Renault"),
    "FR0000073272": ("SAF.PA", "Safran"),
    "FR0000125007": ("SGO.PA", "Saint-Gobain"),
    "FR0000120578": ("SAN.PA", "Sanofi"),
    "FR0000121972": ("SU.PA", "Schneider Electric"),
    "FR0000130809": ("GLE.PA", "Société Générale"),
    "NL00150001Q9": ("STLAP.PA", "Stellantis"),
    "NL0000226223": ("STMPA.PA", "STMicroelectronics"),
    "FR0000121329": ("HO.PA", "Thales"),
    "FR0000120271": ("TTE.PA", "TotalEnergies"),
    "FR0013326246": ("URW.PA", "Unibail-Rodamco-Westfield"),
    "FR0000124141": ("VIE.PA", "Veolia"),
    "FR0000125486": ("DG.PA", "Vinci"),
}
ISIN_OF_TICKER = {t: i for i, (t, _) in CAC40_ISIN.items()}

# Bloomberg/OpenFIGI exchange code -> Yahoo suffix (preferred order first)
FIGI_EXCH = {"FP": ".PA", "NA": ".AS", "BB": ".BR", "PL": ".LS", "ID": ".IR",
             "GY": ".DE", "GR": ".F", "IM": ".MI", "SM": ".MC", "SW": ".SW",
             "SE": ".SW", "LN": ".L", "SS": ".ST", "DC": ".CO", "NO": ".OL",
             "FH": ".HE", "AV": ".VI", "US": "", "UN": "", "UW": "", "CN": ".TO"}
# prefer the home market given by the ISIN country prefix
HOME_EXCH = {"FR": "FP", "NL": "NA", "BE": "BB", "PT": "PL", "IE": "ID", "DE": "GY",
             "IT": "IM", "ES": "SM", "CH": "SW", "GB": "LN", "SE": "SS", "DK": "DC",
             "NO": "NO", "FI": "FH", "AT": "AV", "US": "US", "CA": "CN", "LU": "NA"}

_ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}\d$")


@dataclass
class Resolved:
    isin: str
    ticker: str
    name: str
    exchange: str
    source: str


def clean_isin(code: str) -> str:
    return re.sub(r"[\s\-]", "", (code or "").upper())


def isin_valid(code: str) -> bool:
    """Format + Luhn check digit (ISO 6166)."""
    code = clean_isin(code)
    if not _ISIN_RE.match(code):
        return False
    digits = "".join(str(int(ch, 36)) for ch in code[:-1])
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2 == 0:          # double every second digit from the right
            n *= 2
            n = n - 9 if n > 9 else n
        total += n
    return (10 - total % 10) % 10 == int(code[-1])


def _openfigi(isin: str, timeout: float = 8.0) -> Resolved | None:
    import requests
    r = requests.post("https://api.openfigi.com/v3/mapping",
                      json=[{"idType": "ID_ISIN", "idValue": isin}], timeout=timeout)
    r.raise_for_status()
    data = (r.json() or [{}])[0].get("data") or []
    data = [d for d in data if d.get("exchCode") in FIGI_EXCH and d.get("ticker")]
    if not data:
        return None
    home = HOME_EXCH.get(isin[:2])
    order = ([home] if home else []) + list(FIGI_EXCH)
    data.sort(key=lambda d: order.index(d["exchCode"]) if d["exchCode"] in order else 99)
    d = data[0]
    tk = d["ticker"].replace("/", "-").replace(" ", "-") + FIGI_EXCH[d["exchCode"]]
    return Resolved(isin, tk, (d.get("name") or tk).title(), d["exchCode"], "OpenFIGI")


def _yahoo_search(isin: str) -> Resolved | None:
    import yfinance as yf
    quotes = [x for x in (yf.Search(isin, max_results=8).quotes or []) if x.get("symbol")]
    if not quotes:
        return None
    pref = {"FR": ".PA", "NL": ".AS", "BE": ".BR", "DE": ".DE", "IT": ".MI",
            "ES": ".MC", "GB": ".L", "CH": ".SW"}.get(isin[:2], "")
    quotes.sort(key=lambda x: (not x["symbol"].endswith(pref) if pref else False,
                               x.get("quoteType") != "EQUITY"))
    x = quotes[0]
    return Resolved(isin, x["symbol"], x.get("longname") or x.get("shortname") or x["symbol"],
                    x.get("exchange", ""), "Yahoo search")


def resolve_isin(code: str, online: bool = True) -> Resolved:
    isin = clean_isin(code)
    if not isin_valid(isin):
        raise ValueError(f"'{code}' is not a valid ISIN (12 characters, check digit "
                         "must match – e.g. FR0000121014).")
    if isin in CAC40_ISIN:
        tk, name = CAC40_ISIN[isin]
        return Resolved(isin, tk, name, "Euronext", "CAC 40 table")
    if not online:
        raise LookupError(f"{isin} is not a CAC 40 share and online lookup is off "
                          "(demo / CSV mode).")
    errors = []
    for fn in (_openfigi, _yahoo_search):
        try:
            res = fn(isin)
            if res:
                return res
        except Exception as e:   # network blocked, rate limit, API change
            errors.append(f"{fn.__name__.strip('_')}: {e}")
    raise LookupError(f"No listing found for {isin}." +
                      (f" ({'; '.join(errors)})" if errors else ""))
