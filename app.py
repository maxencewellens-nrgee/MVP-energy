import streamlit as st
import requests, pandas as pd
from io import StringIO
from datetime import datetime, timedelta, time
import pytz, re

st.set_page_config(page_title="DIAG ENTSO-E", layout="wide")
st.title("🔧 Diagnostic ENTSO‑E (BE Day‑Ahead)")

TOKEN = st.secrets.get("ENTSOE_TOKEN", "")
if not TOKEN:
    st.error("Secret ENTSOE_TOKEN manquant (Settings → Secrets).")
    st.stop()

BASE = "https://web-api.tp.entsoe.eu/api"
ZONE = "10YBE----------2"

def parse_prices(xml_text: str) -> pd.DataFrame:
    # 1) si ACK => pas de données
    if "<Acknowledgement_MarketDocument" in xml_text:
        return pd.DataFrame()
    # 2) extraction robuste Period/Point (ignore namespaces)
    rows = []
    for start_s, end_s, inner in re.findall(
        r"<Period>.*?<timeInterval>.*?<start>(.*?)</start>.*?<end>(.*?)</end>.*?</timeInterval>(.*?)</Period>",
        xml_text, flags=re.S
    ):
        try:
            start_dt = datetime.fromisoformat(start_s.replace("Z","+00:00"))
        except:
            start_dt = pd.to_datetime(start_s, utc=True).to_pydatetime()
        for pos_s, price_s in re.findall(
            r"<Point>.*?<position>(\d+)</position>.*?<price.amount>(-?\d+\.?\d*)</price.amount>.*?</Point>",
            inner, flags=re.S
        ):
            pos = int(pos_s); price = float(price_s)
            ts_utc = start_dt + timedelta(hours=pos-1)
            rows.append((ts_utc, price))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts_utc","price"])
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["date_be"] = df["ts_utc"].dt.tz_convert("Europe/Brussels").dt.date
    return df

def call_api(period_start_yyyymmddhhmm: str, period_end_yyyymmddhhmm: str) -> tuple[str,str]:
    url = (
        f"{BASE}?documentType=A44&in_Domain={ZONE}&out_Domain={ZONE}"
        f"&periodStart={period_start_yyyymmddhhmm}&periodEnd={period_end_yyyymmddhhmm}"
        f"&securityToken={TOKEN}"
    )
    r = requests.get(url, timeout=45)
    return url, r.text

st.markdown("### Test 1 — **journée connue** (doit renvoyer des données)")
st.caption("24 août 2025 en BE → 23/08/2025 22:00Z → 24/08/2025 22:00Z (fin exclusive)")
if st.button("Charger (test 24 août 2025)"):
    url, xml = call_api("202508232200","202508242200")
    st.code(url.replace(TOKEN, "***"), language="text")
    st.text(xml[:600])
    df = parse_prices(xml)
    if df.empty:
        st.error("❌ Pas de données (token/URL/intervalle).")
    else:
        st.success(f"✅ Points lus: {len(df)} (attendu ~24)")
        daily = df.groupby("date_be")["price"].agg(avg="mean", mn="min", mx="max", n="count").round(2).reset_index()
        st.write(daily)
        st.metric("Moyenne (€/MWh)", round(df["price"].mean(), 2))

st.markdown("---")
st.markdown("### Test 2 — **J‑7 → J‑1** (dates dynamiques robustes)")
days = st.slider("Nombre de jours", 7, 30, 7)
if st.button("Charger J‑7 → J‑1"):
    tz_be = pytz.timezone("Europe/Brussels")
    now_be = datetime.now(tz_be)
    last_day = (now_be.date() - timedelta(days=1))
    start_local = tz_be.localize(datetime.combine(last_day - timedelta(days=days-1), time(0,0)))
    end_local   = tz_be.localize(datetime.combine(last_day + timedelta(days=1), time(0,0)))  # exclusif
    start_utc = start_local.astimezone(pytz.UTC)
    end_utc   = end_local.astimezone(pytz.UTC)
    st.caption(f"Local BE : {start_local} → {end_local} (excl.)")
    st.caption(f"UTC      : {start_utc} → {end_utc} (excl.)")

    url, xml = call_api(start_utc.strftime("%Y%m%d%H%M"), end_utc.strftime("%Y%m%d%H%M"))
    st.code(url.replace(TOKEN, "***"), language="text")
    df = parse_prices(xml)
    if df.empty:
        st.error("❌ Pas de données (intervalle trop récent ou token).")
        st.text(xml[:600])
    else:
        st.success(f"✅ Points lus: {len(df)}")
        daily = df.groupby("date_be")["price"].agg(avg="mean", mn="min", mx="max", n="count").round(2).reset_index()
        st.line_chart(daily.set_index("date_be")["avg"])
        st.dataframe(daily.tail(10), use_container_width=True)
