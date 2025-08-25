import streamlit as st, requests

st.set_page_config(page_title="DIAG ENTSO-E — DOIT CHANGER VISUELLEMENT", layout="wide")
st.title("🔧 DIAGNOSTIC — ENTSO‑E BE Day‑Ahead")

TOKEN = st.secrets.get("ENTSOE_TOKEN", "")
st.write("Secret présent ?", bool(TOKEN))  # doit afficher True

BASE = "https://web-api.tp.entsoe.eu/api"
ZONE = "10YBE----------2"

# Journée fixe qui contient des données (24 août 2025 local)
start = "202508232200"  # UTC
end   = "202508242200"  # UTC
url = f"{BASE}?documentType=A44&in_Domain={ZONE}&out_Domain={ZONE}&periodStart={start}&periodEnd={end}&securityToken={TOKEN}"

if st.button("TEST 24 août 2025 (doit renvoyer des données)"):
    st.code(url.replace(TOKEN, "***"))
    r = requests.get(url, timeout=30)
    st.write("HTTP status:", r.status_code)
    txt = r.text
    st.text(txt[:600])
    if "<Publication_MarketDocument" in txt and "<TimeSeries>" in txt:
        st.success("✅ L’API renvoie des TimeSeries. Token OK.")
    elif "<Acknowledgement_MarketDocument" in txt:
        st.error("❌ ACK: No matching data → période/token.")
    else:
        st.error("❌ Réponse inattendue (voir texte).")
