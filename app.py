import streamlit as st
import pandas as pd
from entsoe import EntsoePandasClient
from datetime import datetime, timedelta
import pytz

# --- Config Streamlit
st.set_page_config(page_title="BE Day-Ahead – Simple", layout="wide")
st.title("🇧🇪 BE Day-Ahead – Prix moyens (2023 → 23 août 2025)")

# --- Récupération du token
token = st.secrets.get("ENTSOE_TOKEN", "")
if not token:
    st.error("Ajoute ENTSOE_TOKEN dans les secrets Streamlit.")
    st.stop()

# --- Paramètres API
client = EntsoePandasClient(api_key=token)
zone = "10YBE----------2"  # Belgique
tz = pytz.UTC

# Période : du 1er janvier 2023 au 23 août 2025 inclus
start = pd.Timestamp("2023-01-01", tz=tz)
end   = pd.Timestamp("2025-08-24", tz=tz)  # exclusif

# --- Récupération des données
with st.spinner("Récupération des prix ENTSO-E…"):
    try:
        s = client.query_day_ahead_prices(zone, start=start, end=end)
    except Exception as e:
        st.error(f"Erreur API ENTSO-E : {e}")
        st.stop()

# --- Traitement
s = s.tz_convert("Europe/Brussels")
df = s.to_frame("price").reset_index()
df["date"] = df["index"].dt.date
daily = df.groupby("date")["price"].mean().reset_index()

# --- Affichage
st.metric("Moyenne totale (2023 → 23 août 2025)",
          f"{daily['price'].mean():.2f} €/MWh")

st.line_chart(daily.set_index("date")["price"])
st.dataframe(daily.tail(10), use_container_width=True)

# Bouton CSV
st.download_button("Télécharger CSV complet",
                   data=daily.to_csv(index=False),
                   file_name="be_dayahead_2023_2025.csv",
                   mime="text/csv")
