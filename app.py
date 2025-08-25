# app.py — MVP décision "Attendre / Fixer" 🇧🇪
import streamlit as st
import pandas as pd
from entsoe import EntsoePandasClient
from datetime import datetime, timedelta
import pytz

st.set_page_config(page_title="BE Day-Ahead – Décision", layout="wide")
st.title("🇧🇪 Électricité BE – Dernier prix & décision (MVP)")

# --- Token
TOKEN = st.secrets.get("ENTSOE_TOKEN", "")
if not TOKEN:
    st.error("Secret ENTSOE_TOKEN manquant (Streamlit Cloud → Settings → Secrets).")
    st.stop()

# --- Params
ZONE = "10YBE----------2"  # Belgique
tz_utc = pytz.UTC
tz_be  = pytz.timezone("Europe/Brussels")

# --- Bornes : 2023-01-01 → J-1 (fin exclusive = J)
DEFAULT_START = "2023-01-01"
today_be = datetime.now(tz_be).date()
end_inclusive = today_be - timedelta(days=1)     # J-1 (inclus)
end_exclusive = end_inclusive + timedelta(days=1)  # exclusif pour l'API

client = EntsoePandasClient(api_key=TOKEN)

@st.cache_data(ttl=24*3600)
def fetch_daily(start_date: str, end_inclusive_date: str) -> pd.DataFrame:
    start = pd.Timestamp(start_date, tz=tz_utc)
    end   = pd.Timestamp(end_inclusive_date, tz=tz_utc) + pd.Timedelta(days=1)  # exclusif
    # Boucle mensuelle (robuste quotas)
    months = pd.date_range(start.normalize(), end.normalize(), freq="MS", tz=tz_utc)
    series = []
    for i, t0 in enumerate(months):
        t1 = months[i+1] if i+1 < len(months) else end
        s = client.query_day_ahead_prices(ZONE, start=t0, end=t1)
        series.append(s)
    s_all = pd.concat(series).sort_index()

    s_all = s_all.tz_convert(tz_be)  # heure BE
    df = s_all.to_frame("price").reset_index().rename(columns={"index":"ts"})
    df["date"] = df["ts"].dt.date
    # agrégat journalier
    daily = df.groupby("date")["price"].agg(["mean","min","max","count"]).reset_index()
    daily = daily.rename(columns={"mean":"avg","min":"mn","max":"mx","count":"n"})
    daily[["avg","mn","mx"]] = daily[["avg","mn","mx"]].round(2)
    return daily

def recommendation_from_history(daily: pd.DataFrame, lookback_days: int = 180) -> dict:
    """Reco simple par quantiles sur l'historique récent."""
    if daily.empty: 
        return {"reco":"—", "comment":"Pas de données.", "score":None}
    last_day = daily.iloc[-1]
    ref = daily.copy()
    # fenêtre de référence
    ref = ref[ref["date"] >= (pd.to_datetime(end_inclusive) - pd.Timedelta(days=lookback_days)).date()]
    if len(ref) < 30:  # garde-fou
        ref = daily
    q10 = ref["avg"].quantile(0.10)
    q30 = ref["avg"].quantile(0.30)
    q70 = ref["avg"].quantile(0.70)

    last_avg = last_day["avg"]
    if last_avg <= q10:
        reco = "FIXER 40–60 %"
        comment = "Prix très bas vs historique récent (≤ P10). Opportunité forte de couverture."
        score = 80
    elif last_avg <= q30:
        reco = "FIXER 20–30 %"
        comment = "Prix bas vs historique (≤ P30). Couverture partielle conseillée."
        score = 60
    elif last_avg <= q70:
        reco = "ATTENDRE"
        comment = "Prix médian (P30–P70). Pas de signal clair, observer encore."
        score = 45
    else:
        reco = "ATTENDRE (clairement)"
        comment = "Prix élevés (≥ P70). Éviter de fixer maintenant."
        score = 25

    return {
        "reco": reco,
        "comment": comment,
        "score": score,
        "last_avg": round(last_avg,2),
        "q10": round(q10,2), "q30": round(q30,2), "q70": round(q70,2)
    }

# --- UI contrôles
with st.sidebar:
    st.subheader("Paramètres")
    start_input = st.text_input("Date début (YYYY-MM-DD)", value=DEFAULT_START)
    end_input   = st.text_input("Date fin incluse (J-1 par défaut)", value=str(end_inclusive))
    lookback = st.slider("Fenêtre d'analyse (jours)", 90, 365, 180, step=30)
    run = st.button("Calculer / Mettre à jour")

if run:
    try:
        with st.spinner("Récupération ENTSO‑E (mois par mois)…"):
            daily = fetch_daily(start_input, end_input)
        if daily.empty:
            st.error("Aucune donnée sur l'intervalle demandé.")
            st.stop()

        # KPIs
        overall_avg = round(daily["avg"].mean(), 2)
        last_row = daily.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Moyenne totale", f"{overall_avg} €/MWh")
        c2.metric("Dernier prix accessible (J-1)", f"{last_row['avg']:.2f} €/MWh")
        c3.metric("Min / Max (J-1)", f"{last_row['mn']:.2f} / {last_row['mx']:.2f} €/MWh")
        c4.metric("Points (J-1)", int(last_row["n"]))

        # Recommandation
        rec = recommendation_from_history(daily, lookback_days=lookback)
        st.subheader("Décision du jour (MVP)")
        colA, colB = st.columns([1,2])
        with colA:
            st.metric("Recommandation", rec["reco"])
            st.metric("Score (0–100)", rec["score"])
        with colB:
            st.info(
                f"**Dernier prix (J-1)** : {rec['last_avg']} €/MWh  \n"
                f"Référence {lookback} j – Quantiles : P10 {rec['q10']} · P30 {rec['q30']} · P70 {rec['q70']} €/MWh  \n"
                f"**Commentaire** : {rec['comment']}"
            )

        # Graph & tableau
        st.subheader("Moyennes journalières (€/MWh)")
        st.line_chart(daily.set_index("date")["avg"])
        st.dataframe(daily.tail(14), use_container_width=True)
        st.download_button(
            "Télécharger CSV (jour par jour)",
            data=daily.to_csv(index=False),
            file_name=f"be_dayahead_{start_input}_to_{end_input}.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Erreur : {e}")
        st.caption("Vérifie le secret ENTSOE_TOKEN et les dates (format YYYY-MM-DD).")
else:
    st.info("Choisis les dates puis clique **Calculer / Mettre à jour**. Par défaut, la fin est J‑1.")
