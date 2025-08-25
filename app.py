import streamlit as st
import requests, pandas as pd
from io import StringIO

st.set_page_config(page_title="DIAG ENTSO-E", layout="wide")
st.title("🔧 Diagnostic ENTSO‑E (Belgique Day‑Ahead)")

TOKEN = st.secrets.get("ENTSOE_TOKEN", "")
if not TOKEN:
    st.error("Secret ENTSOE_TOKEN manquant dans Streamlit Cloud → Settings → Secrets.")
    st.stop()

# ⚠️ Période FIXE qui a des données : 24 août 2025 (en heure locale BE)
# En UTC pour ENTSO‑E : 23/08/2025 22:00Z → 24/08/2025 22:00Z (fin exclusive)
URL = (
    "https://web-api.tp.entsoe.eu/api?"
    "documentType=A44&"
    "in_Domain=10YBE----------2&out_Domain=10YBE----------2&"
    "periodStart=202508232200&"
    "periodEnd=202508242200&"
    f"securityToken={TOKEN}"
)

st.code(URL, language="text")
r = requests.get(URL, timeout=30)
st.write("HTTP status:", r.status_code)

text = r.text
st.text(text[:800])  # affiche le début de la réponse

if "<Acknowledgement_MarketDocument" in text:
    st.error("⚠️ L’API répond 'No matching data' → problème de période OU token invalide.")
    st.stop()

# Parse robuste : pandas.read_xml direct
try:
    df = pd.read_xml(StringIO(text))
    price_col = next(c for c in df.columns if "price" in c.lower())
    pos_col   = next(c for c in df.columns if "position" in c.lower())
    start_col = next(c for c in df.columns if "timeinterval_start" in c.lower())
    out = pd.DataFrame({
        "price": pd.to_numeric(df[price_col], errors="coerce"),
        "position": pd.to_numeric(df[pos_col], errors="coerce"),
        "period_start": pd.to_datetime(df[start_col], utc=True).ffill()
    }).dropna()
    st.success(f"Points lus: {len(out)} (attendu ≈ 24)")
    st.write(out.head())
    st.metric("Moyenne (€/MWh)", round(out["price"].mean(), 2))
except Exception as e:
    st.error(f"Parse XML échoué: {e}")
