# app.py — version corrigée

import streamlit as st
import pandas as pd
import requests
from io import StringIO
from datetime import datetime, timedelta, timezone
import pytz

from entsoe import EntsoePandasClient

# ---------------- Config page
st.set_page_config(page_title="BE Day-Ahead – MVP", layout="wide")
st.title("🇧🇪 BE Day‑Ahead – MVP Timing")

# ---------------- Constantes
ENTSOE_BASE = "https://web-api.tp.entsoe.eu/api"
BID_BE = "10YBE----------2"

# ---------------- Helpers API spot (XML brut)
@st.cache_data(ttl=60*60)
def get_entsoe_xml(token: str, start_utc: datetime, end_utc: datetime) -> str:
    url = (
        f"{ENTSOE_BASE}?documentType=A44&"
        f"in_Domain={BID_BE}&out_Domain={BID_BE}&"
        f"periodStart={start_utc.strftime('%Y%m%d%H%M')}&"
        f"periodEnd={end_utc.strftime('%Y%m%d%H%M')}&"
        f"securityToken={token}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text

def parse_publication_xml(xml_text: str) -> pd.DataFrame:
    # Si l'API répond explicitement "No matching data..."
    if "<Acknowledgement_MarketDocument" in xml_text:
        return pd.DataFrame()

    df = pd.read_xml(StringIO(xml_text))
    # Colonnes dynamiques : on sécurise
    price_col = next((c for c in df.columns if "price" in c.lower()), None)
    pos_col   = next((c for c in df.columns if "position" in c.lower()), None)
    start_col = next((c for c in df.columns if "timeinterval_start" in c.lower()), None)
    if not price_col or not pos_col or not start_col:
        return pd.DataFrame()

    out = pd.DataFrame({
        "price": pd.to_numeric(df[price_col], errors="coerce"),
        "position": pd.to_numeric(df[pos_col], errors="coerce"),
        "period_start": pd.to_datetime(df[start_col], utc=True).ffill()
    }).dropna()

    out["ts_utc"] = out["period_start"] + pd.to_timedelta(out["position"] - 1, unit="h")
    out["date_be"] = out["ts_utc"].dt.tz_convert("Europe/Brussels").dt.date
    return out[["ts_utc", "date_be", "price"]].dropna()

def daily_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.groupby("date_be")["price"]
    daily = g.agg(avg="mean", mn="min", mx="max", n="count").reset_index()
    daily["neg_hours"] = df.assign(neg=(df["price"] < 0).astype(int)) \
                            .groupby("date_be")["neg"].sum().values
    for c in ("avg","mn","mx"):
        daily[c] = daily[c].round(2)
    return daily

def simple_score(daily: pd.DataFrame, fwd_cal: float|None, fwd_ma12: float|None) -> float:
    if daily.empty: 
        return 0.0
    last7 = daily["avg"].iloc[-7:].mean() if len(daily)>=7 else daily["avg"].mean()
    comp1 = 50
    if fwd_cal and fwd_ma12 and fwd_ma12>0:
        comp1 = 50 + 50*(fwd_ma12 - fwd_cal)/fwd_ma12
        comp1 = max(0, min(100, comp1))
    comp2 = 50
    if fwd_cal and fwd_cal>0:
        comp2 = 50 + ((last7 - fwd_cal) / fwd_cal)*100
        comp2 = max(0, min(100, comp2))
    score = 0.6*comp1 + 0.4*comp2
    return round(max(0, min(100, score)), 1)

def recommendation(score: float) -> str:
    if score < 35: return "ATTENDRE (0 %)"
    if score < 60: return "FIXER 20–30 %"
    return "FIXER 40–60 %"

# ---------------- Historique via entsoe-py (définie AVANT usage)
@st.cache_data(ttl=24*3600)
def fetch_dayahead_history_entsoe(token: str, start_date="2023-01-01", end_date=None):
    tz = pytz.UTC
    if end_date is None:
        end_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    start = pd.Timestamp(start_date, tz=tz)
    end   = pd.Timestamp(end_date,   tz=tz) + pd.Timedelta(days=1)  # exclusif

    client = EntsoePandasClient(api_key=token)
    zone = BID_BE

    months = pd.date_range(start.normalize(), end.normalize(), freq="MS", tz=tz)
    series = []
    for i, t0 in enumerate(months):
        t1 = months[i+1] if i+1 < len(months) else end
        s = client.query_day_ahead_prices(zone, start=t0, end=t1)
        series.append(s)

    s_all = pd.concat(series).sort_index()

    s_all.index = s_all.index.tz_convert("Europe/Brussels")
    df = s_all.to_frame("price").copy()
    df["date"] = df.index.date
    df["neg"] = (df["price"] < 0).astype(int)
    daily = df.groupby("date").agg(
        avg=("price","mean"),
        mn =("price","min"),
        mx =("price","max"),
        n  =("price","count"),
        negative_hours=("neg","sum"),
    ).reset_index()
    daily[["avg","mn","mx"]] = daily[["avg","mn","mx"]].round(2)
    return daily

# ---------------- UI (sidebar)
token = st.secrets.get("ENTSOE_TOKEN", "")
st.sidebar.info("Ton token ENTSO‑E est lu depuis les *Secrets* Streamlit.")
days = st.sidebar.slider("Jours à afficher", min_value=7, max_value=90, value=14, step=1)
fwd_cal = st.sidebar.number_input("Forward CAL (€/MWh) – provisoire", min_value=0.0, value=0.0, step=1.0)
fwd_ma  = st.sidebar.number_input("Moyenne 12 mois CAL (€/MWh) – provisoire", min_value=0.0, value=0.0, step=1.0)

# ---------------- Intervalle robuste (J-1 uniquement, en UTC)
now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
last_published_utc = (now_utc - timedelta(days=1)).replace(hour=22, minute=0, second=0, microsecond=0)
start_utc = last_published_utc - timedelta(days=days-1)
end_utc   = last_published_utc + timedelta(days=1)  # exclusif

# ---------------- Action 1 : N derniers jours (spot)
st.subheader("N derniers jours Day-Ahead 🇧🇪")
if st.button("Charger / Mettre à jour"):
    if not token:
        st.warning("Ajoute ENTSOE_TOKEN dans les *Secrets* Streamlit.")
    else:
        try:
            xml = get_entsoe_xml(token, start_utc, end_utc)
            raw = parse_publication_xml(xml)
            if raw.empty:
                st.error("Pas de données pour l’intervalle (publie ~13:00 UTC la veille) ou bornes invalides.")
            else:
                daily = daily_stats(raw)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Moyenne (dernier jour)", f"{daily['avg'].iloc[-1]} €/MWh")
                c2.metric("Min / Max", f"{daily['mn'].iloc[-1]} / {daily['mx'].iloc[-1]} €/MWh")
                c3.metric("Heures négatives (dernier jour)", int(daily['neg_hours'].iloc[-1]))
                c4.metric("Points", int(daily['n'].iloc[-1]))

                st.line_chart(daily.set_index("date_be")["avg"])
                st.dataframe(daily, use_container_width=True)

                st.subheader("Score & Recommandation (MVP)")
                score = simple_score(daily, fwd_cal if fwd_cal>0 else None, fwd_ma if fwd_ma>0 else None)
                st.metric("Score (0–100)", score)
                st.success(recommendation(score))

                st.download_button("Télécharger CSV (N jours)",
                                   data=daily.to_csv(index=False),
                                   file_name="be_dayahead_summary.csv",
                                   mime="text/csv")
        except Exception as e:
            st.error(f"Erreur: {e}")
            st.caption("Vérifie le token, l’URL API, et l’intervalle (UTC).")

# ---------------- Action 2 : Backfill historique (entsoe-py)
st.subheader("Historique Day-Ahead 🇧🇪 (2023 → hier)")
if st.button("Backfill historique ENTSO‑E"):
    if not token:
        st.error("Ajoute ENTSOE_TOKEN dans les *Secrets* Streamlit.")
    else:
        try:
            with st.spinner("Récupération ENTSO‑E…"):
                hist = fetch_dayahead_history_entsoe(token, start_date="2023-01-01")
            st.success(f"{len(hist)} jours récupérés.")
            st.line_chart(hist.set_index("date")["avg"])
            st.dataframe(hist.tail(10), use_container_width=True)
            st.download_button(
                "Télécharger CSV historique",
                data=hist.to_csv(index=False),
                file_name="be_dayahead_2023_to_yesterday.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Erreur: {e}")
            st.caption("Vérifie les Secrets, la dépendance 'entsoe-py' et les quotas API.")
