import streamlit as st
import pandas as pd
import requests
import re
import html as ihtml
import unicodedata
import urllib.parse
from entsoe import EntsoePandasClient
from datetime import datetime, timedelta
import pytz

tz_utc = pytz.UTC
tz_be = pytz.timezone("Europe/Brussels")

@st.cache_data(ttl=24*3600)
def fetch_daily_prices(token: str, zone: str, start_date: str, end_inclusive_date: str) -> pd.DataFrame:
    """
    Fetch day-ahead prices via ENTSO-E API.
    Returns DataFrame with date, avg, mn, mx, n columns.
    """
    try:
        client = EntsoePandasClient(api_key=token)
        start = pd.Timestamp(start_date, tz=tz_utc)
        end = pd.Timestamp(end_inclusive_date, tz=tz_utc) + pd.Timedelta(days=1)

        months = pd.date_range(start.normalize(), end.normalize(), freq="MS", tz=tz_utc)
        series = []

        for i, t0 in enumerate(months):
            t1 = months[i+1] if i+1 < len(months) else end
            try:
                s = client.query_day_ahead_prices(zone, start=t0, end=t1)
                series.append(s)
            except Exception as e:
                st.warning(f"Failed to fetch data for period {t0.date()} to {t1.date()}: {e}")
                continue

        if not series:
            return pd.DataFrame()

        s_all = pd.concat(series).sort_index()
        s_all = s_all.tz_convert(tz_be)

        df = s_all.to_frame("price").reset_index().rename(columns={"index": "ts"})
        df["date"] = df["ts"].dt.date

        out = df.groupby("date")["price"].agg(
            avg="mean",
            mn="min",
            mx="max",
            n="count"
        ).reset_index()

        out[["avg", "mn", "mx"]] = out[["avg", "mn", "mx"]].round(2)
        return out

    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        return pd.DataFrame()

def calculate_decision(daily: pd.DataFrame, lookback_days: int = 180) -> dict:
    """Calculate trading decision based on recent price history."""
    if daily.empty:
        return {
            "reco": "—",
            "raison": "Pas de données.",
            "last": None,
            "p10": None,
            "p30": None,
            "p70": None
        }

    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    last_price = float(df.iloc[-1]["avg"])
    ref_end = df["date"].max()
    ref_start = ref_end - pd.Timedelta(days=lookback_days)

    ref = df[df["date"] >= ref_start]
    if len(ref) < 30:
        ref = df

    p10 = ref["avg"].quantile(0.10)
    p30 = ref["avg"].quantile(0.30)
    p70 = ref["avg"].quantile(0.70)

    if last_price <= p10:
        return {
            "reco": "FIXER 40–60 %",
            "raison": f"Dernier prix {last_price:.2f} ≤ P10 {p10:.2f} : opportunité forte.",
            "last": round(last_price, 2),
            "p10": round(p10, 2),
            "p30": round(p30, 2),
            "p70": round(p70, 2)
        }

    if last_price <= p30:
        return {
            "reco": "FIXER 20–30 %",
            "raison": f"Dernier prix {last_price:.2f} ≤ P30 {p30:.2f} : fenêtre favorable.",
            "last": round(last_price, 2),
            "p10": round(p10, 2),
            "p30": round(p30, 2),
            "p70": round(p70, 2)
        }

    if last_price >= p70:
        return {
            "reco": "ATTENDRE (clairement)",
            "raison": f"Dernier prix {last_price:.2f} ≥ P70 {p70:.2f} : marché cher.",
            "last": round(last_price, 2),
            "p10": round(p10, 2),
            "p30": round(p30, 2),
            "p70": round(p70, 2)
        }

    return {
        "reco": "ATTENDRE",
        "raison": f"Dernier prix {last_price:.2f} entre P30 {p30:.2f} et P70 {p70:.2f} : pas de signal fort.",
        "last": round(last_price, 2),
        "p10": round(p10, 2),
        "p30": round(p30, 2),
        "p70": round(p70, 2)
    }

@st.cache_data(ttl=60*30)
def fetch_flexypower_cals(url: str = "https://flexypower.eu/prix-de-lenergie/") -> dict:
    """Fetch CAL prices from FlexyPower website with multiple fallback strategies."""

    def _normalize(raw: str) -> str:
        txt = ihtml.unescape(raw).replace("\xa0", " ")
        txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"\s+", " ", txt)

    def _parse_block(text: str) -> dict:
        vals = {"CAL-26": None, "CAL-27": None, "CAL-28": None, "date": None}
        m_elec = re.search(r"Electricite.*?(?=Gaz naturel|<h2|</section>|$)", text, flags=re.I)
        block = m_elec.group(0) if m_elec else text

        dm = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", block)
        if dm:
            vals["date"] = dm.group(1)

        for yy in ("26", "27", "28"):
            m = re.search(rf"CAL\s*[-]?\s*{yy}\D*?([0-9]+(?:[.,][0-9]+)?)", block, flags=re.I)
            if m:
                try:
                    vals[f"CAL-{yy}"] = float(m.group(1).replace(",", "."))
                except:
                    pass
        return vals

    try:
        r = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8"
            },
            timeout=20
        )
        r.raise_for_status()
        vals = _parse_block(_normalize(r.text))
        if any(vals[k] is not None for k in ("CAL-26", "CAL-27", "CAL-28")):
            return vals
    except:
        pass

    try:
        pr = urllib.parse.urlparse(url)
        proxy_url = f"https://r.jina.ai/http://{pr.netloc}{pr.path}"
        r2 = requests.get(proxy_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r2.raise_for_status()
        vals = _parse_block(_normalize(r2.text))
        if any(vals[k] is not None for k in ("CAL-26", "CAL-27", "CAL-28")):
            return vals
    except:
        pass

    try:
        tables = pd.read_html(url)
        vals = {"CAL-26": None, "CAL-27": None, "CAL-28": None, "date": None}
        for df in tables:
            prod_col = df.columns[0]
            price_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            for yy in ("26", "27", "28"):
                mask = df[prod_col].astype(str).str.upper().str.contains(f"CAL[- ]?{yy}")
                if mask.any():
                    rawv = str(df.loc[mask, price_col].iloc[0])
                    rawv = re.sub(r"[^\d,\.]", "", rawv).replace(",", ".")
                    try:
                        vals[f"CAL-{yy}"] = float(rawv)
                    except:
                        pass
        if any(vals[k] is not None for k in ("CAL-26", "CAL-27", "CAL-28")):
            return vals
    except:
        pass

    return {"CAL-26": None, "CAL-27": None, "CAL-28": None, "date": None}
