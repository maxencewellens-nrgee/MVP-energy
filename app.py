import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pytz
from entsoe import EntsoePandasClient

st.set_page_config(page_title="BE Day-Ahead – Moyenne 2023 → 2025-08-23", layout="wide")
st.title("🇧🇪 Moyenne Day‑Ahead Belgique (2023 → 2025‑08‑23)")

TOKEN = st.secrets.get("ENTSOE_TOKEN", "")
if not TOKEN:
    st.error("Secret ENTSOE_TOKEN manquant (Streamlit Cloud → Settings → Secrets).")
    st.stop()

# bornes fixes demandées (tu pourras changer end_date plus tard dans l'UI)
DEFAULT_START = "2023-01-01"
DEFAULT_END   = "2025-08-23"  # inclus (on ajoutera +1 jour côté API)

@st.cache_data(ttl=24*3600)
def fetch_history_avg(token: str, start_date: str, end_date: str) -> tuple[pd.DataFrame, float]:
    """
    Récupère les prix Day-Ahead (heure) Belgique via entsoe-py, par MOIS,
    agrège en JOUR, puis calcule la moyenne sur toute la période.
    """
    tz = pytz.UTC
    start = pd.Timestamp(start_date, tz=tz)
    # end exclusif pour l'API -> +1 jour
    end   = pd.Timestamp(end_date, tz=tz) + pd.Timedelta(days=1)

    client = EntsoePandasClient(api_key=token)
    zone = "10YBE----------2"  # Belgique

    # boucle mensuelle pour éviter les limites de l'API
    months = pd.date_range(start.normalize(), end.normalize(), freq="MS", tz=tz)
    series = []
    for i, t0 in enumerate(months):
        t1 = months[i+1] if i+1 < len(months) else end
        s = client.query_day_ahead_prices(zone, start=t0, end=t1)
        series.append(s)

    s_all = pd.concat(series).sort_index()

    # Convertit en heure Belgique et agrège par jour
    s_all.index = s_all.index.tz_convert("Europe/Brussels")
    df = s_all.to_frame("price").copy()
    df["date"] = df.index.date
    daily = df.groupby("date")["price"].agg(avg="mean").reset_index()
    daily["avg"] = daily["avg"].round(2)

    # garde uniquement jusqu'à end_date inclus (au cas où)
    daily = daily[daily["date"] <= pd.to_datetime(end_date).date()]

    overall_avg = round(daily["avg"].mean(), 2) if not daily.empty else float("nan")
    return daily, overall_avg

# UI simple
colA, colB, colC = st.columns([1,1,1])
with colA:
    start_input = st.text_input("Date début (YYYY-MM-DD)", value=DEFAULT_START)
with colB:
    end_input = st.text_input("Date fin incluse (YYYY-MM-DD)", value=DEFAULT_END)
with colC:
    run = st.button("Calculer")

if run:
    try:
        with st.spinner("Récupération ENTSO‑E (boucle mois par mois)…"):
            daily, overall_avg = fetch_history_avg(TOKEN, start_input, end_input)

        if daily.empty:
            st.error("Aucune donnée renvoyée sur l'intervalle. Vérifie les dates et réessaie.")
        else:
            st.metric(f"Moyenne {start_input} → {end_input}", f"{overall_avg} €/MWh")
            st.line_chart(daily.set_index("date")["avg"])
            st.download_button(
                "Télécharger CSV (moyennes journalières)",
                data=daily.to_csv(index=False),
                file_name=f"be_dayahead_{start_input}_to_{end_input}.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Erreur : {e}")
        st.caption("Vérifie : token (Secrets), dates valides, et réessaie.")
else:
    st.info("Clique sur **Calculer** pour récupérer 2023 → 2025‑08‑23 et afficher la moyenne.")
