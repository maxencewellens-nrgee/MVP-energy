# app.py — MVP décision + contrat 🇧🇪
import streamlit as st
import pandas as pd
from entsoe import EntsoePandasClient
from datetime import datetime, timedelta, date
import pytz
import altair as alt

st.set_page_config(page_title="BE Day-Ahead – Décision + Contrat", layout="wide")
st.title("🇧🇪 Électricité BE – Dernier prix & décision (MVP + Contrat)")

# --- Token
TOKEN = st.secrets.get("ENTSOE_TOKEN", "")
if not TOKEN:
    st.error("Secret ENTSOE_TOKEN manquant (Streamlit Cloud → Settings → Secrets).")
    st.stop()

# --- Constantes
ZONE = "10YBE----------2"  # Belgique
tz_utc = pytz.UTC
tz_be  = pytz.timezone("Europe/Brussels")

client = EntsoePandasClient(api_key=TOKEN)

# --- Utilitaires
@st.cache_data(ttl=24*3600)
def fetch_daily(start_date: str, end_inclusive_date: str) -> pd.DataFrame:
    """Récupère day-ahead via entsoe-py, agrège par jour (avg/mn/mx/n) en heure BE."""
    start = pd.Timestamp(start_date, tz=tz_utc)
    end   = pd.Timestamp(end_inclusive_date, tz=tz_utc) + pd.Timedelta(days=1)  # exclusif
    months = pd.date_range(start.normalize(), end.normalize(), freq="MS", tz=tz_utc)
    series = []
    for i, t0 in enumerate(months):
        t1 = months[i+1] if i+1 < len(months) else end
        s = client.query_day_ahead_prices(ZONE, start=t0, end=t1)
        series.append(s)
    s_all = pd.concat(series).sort_index()
    s_all = s_all.tz_convert(tz_be)
    df = s_all.to_frame("price").reset_index().rename(columns={"index":"ts"})
    df["date"] = df["ts"].dt.date
    daily = df.groupby("date")["price"].agg(["mean","min","max","count"]).reset_index()
    daily = daily.rename(columns={"mean":"avg","min":"mn","max":"mx","count":"n"})
    daily[["avg","mn","mx"]] = daily[["avg","mn","mx"]].round(2)
    return daily

def reco_quantiles(daily: pd.DataFrame, lookback_days: int = 180):
    """Reco MVP basée sur quantiles P10/P30/P70 sur l'historique récent."""
    if daily.empty:
        return {"reco":"—","score":None,"comment":"Pas de données."}
    end_inclusive = daily["date"].max()
    ref_start = (pd.to_datetime(end_inclusive) - pd.Timedelta(days=lookback_days)).date()
    ref = daily[daily["date"] >= ref_start]
    if len(ref) < 30:  # garde-fou
        ref = daily
    q10 = ref["avg"].quantile(0.10)
    q30 = ref["avg"].quantile(0.30)
    q70 = ref["avg"].quantile(0.70)
    last_avg = daily.iloc[-1]["avg"]
    if last_avg <= q10:
        return {"reco":"FIXER 40–60 %","score":80,
                "comment":"Prix ≤ P10 (très bas). Fenêtre de couverture.",
                "last_avg":round(last_avg,2),"q10":round(q10,2),"q30":round(q30,2),"q70":round(q70,2)}
    if last_avg <= q30:
        return {"reco":"FIXER 20–30 %","score":60,
                "comment":"Prix ≤ P30 (bas). Couverture partielle conseillée.",
                "last_avg":round(last_avg,2),"q10":round(q10,2),"q30":round(q30,2),"q70":round(q70,2)}
    if last_avg <= q70:
        return {"reco":"ATTENDRE","score":45,
                "comment":"Prix entre P30 et P70. Pas de signal fort.",
                "last_avg":round(last_avg,2),"q10":round(q10,2),"q30":round(q30,2),"q70":round(q70,2)}
    return {"reco":"ATTENDRE (clairement)","score":25,
            "comment":"Prix ≥ P70 (élevés). Éviter de fixer maintenant.",
            "last_avg":round(last_avg,2),"q10":round(q10,2),"q30":round(q30,2),"q70":round(q70,2)}

# === Bloc existant : données jusqu'à J-1 ===
today_be = datetime.now(tz_be).date()
end_inclusive_default = today_be - timedelta(days=1)  # J-1
start_default = "2023-01-01"

with st.sidebar:
    st.subheader("Paramètres marché")
    start_input = st.text_input("Date début (YYYY-MM-DD)", value=start_default)
    end_input   = st.text_input("Date fin incluse (par défaut J-1)", value=str(end_inclusive_default))
    lookback    = st.slider("Fenêtre quantiles (jours)", 90, 365, 180, step=30)
    run_market  = st.button("Charger / Mettre à jour")

if run_market:
    try:
        with st.spinner("Récupération ENTSO-E…"):
            daily = fetch_daily(start_input, end_input)
        if daily.empty:
            st.error("Aucune donnée sur l'intervalle demandé.")
        else:
            last = daily.iloc[-1]
            overall_avg = round(daily["avg"].mean(), 2)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Moyenne totale", f"{overall_avg} €/MWh")
            c2.metric("Dernier prix accessible (J-1)", f"{last['avg']:.2f} €/MWh")
            c3.metric("Min/Max (J-1)", f"{last['mn']:.2f} / {last['mx']:.2f} €/MWh")
            c4.metric("Points (J-1)", int(last["n"]))

            rec = reco_quantiles(daily, lookback_days=lookback)
            st.subheader("Décision marché (MVP)")
            colA, colB = st.columns([1,2])
            with colA:
                st.metric("Recommandation", rec["reco"])
                st.metric("Score (0–100)", rec["score"])
            with colB:
                st.info(
                    f"**Dernier prix (J-1)**: {rec['last_avg']} €/MWh  \n"
                    f"Références {lookback} j : P10 {rec['q10']} · P30 {rec['q30']} · P70 {rec['q70']} €/MWh  \n"
                    f"**Commentaire** : {rec['comment']}"
                )

            st.subheader("Historique – Moyenne journalière (€/MWh)")
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

st.markdown("---")

# === 1) Formulaire d’entrée utilisateur (Contrat) ===
st.subheader("Contrat client — entrées")
with st.form("form_contrat"):
    col1, col2, col3 = st.columns(3)
    with col1:
        date_debut_contrat = st.date_input("Date début contrat", value=date(today_be.year, 1, 1))
    with col2:
        duree_contrat = st.radio("Durée du contrat", options=[12, 24, 36], index=2, format_func=lambda m: f"{m//12} an(s)")
    with col3:
        volume_total_mwh = st.number_input("Volume total (MWh)", min_value=0.0, value=200.0, step=10.0)

    col4, col5, col6 = st.columns(3)
    with col4:
        volume_deja_fixe_mwh = st.number_input("Volume déjà fixé (MWh)", min_value=0.0, value=120.0, step=10.0)
    with col5:
        prix_fixe_moyen = st.number_input("Prix fixe moyen (€/MWh)", min_value=0.0, value=85.0, step=1.0)
    with col6:
        date_fixage_moyen = st.date_input("Date de fixation moyenne", value=date(today_be.year, 1, 10))

    submit_contrat = st.form_submit_button("Mettre à jour le contrat")

# Calculs contrat
if submit_contrat:
    date_fin_contrat = date_debut_contrat + timedelta(days=30*duree_contrat)  # approx. mois → correct visuel
    # Couvertures
    couverture_pct = 0.0 if volume_total_mwh == 0 else min(100.0, round(100*volume_deja_fixe_mwh/volume_total_mwh, 2))
    reste_mwh = max(0.0, volume_total_mwh - volume_deja_fixe_mwh)

    st.success(f"Date fin contrat (calculée) : **{date_fin_contrat}**")

    # === 2) Timeline (ligne du temps) ===
    st.subheader("Ligne du temps du contrat")
    # Prépare les points clés
    nb_mois = duree_contrat
    # Position actuelle (en mois depuis début)
    months_elapsed = max(0, (today_be.year - date_debut_contrat.year) * 12 + (today_be.month - date_debut_contrat.month))
    months_elapsed = int(min(nb_mois, max(0, months_elapsed)))
    # Data pour Altair
    timeline_df = pd.DataFrame({
        "segment":["Écoulé","Restant"],
        "mois":[months_elapsed, max(0, nb_mois - months_elapsed)]
    })
    base = alt.Chart(timeline_df).mark_bar().encode(
        x=alt.X('sum(mois):Q', title=f"Durée contrat: {nb_mois} mois"),
        color=alt.Color('segment:N', scale=alt.Scale(range=["#4CAF50","#E0E0E0"]))
    ).properties(height=40, width=600)
    # Curseur "aujourd'hui"
    today_pos = min(nb_mois, months_elapsed)
    marker_df = pd.DataFrame({"mois":[today_pos]})
    marker = alt.Chart(marker_df).mark_rule(size=2).encode(x='mois:Q', color=alt.value("#212121"))
    st.altair_chart(base + marker, use_container_width=True)
    st.caption(f"⏱️ Aujourd’hui = mois {months_elapsed} / {nb_mois}")

    # === 3) Barre de couverture (progression des volumes fixés) ===
    st.subheader("Couverture du contrat (volumes)")
    cover_df = pd.DataFrame({
        "Part":["Fixé","À fixer"],
        "MWh":[min(volume_deja_fixe_mwh, volume_total_mwh), max(0.0, volume_total_mwh - volume_deja_fixe_mwh)],
        "Info":[f"{volume_deja_fixe_mwh:.0f} MWh fixés à {prix_fixe_moyen:.2f} €/MWh le {date_fixage_moyen}",
                f"{reste_mwh:.0f} MWh restants"]
    })
    cover_chart = alt.Chart(cover_df).mark_bar().encode(
        x=alt.X('sum(MWh):Q', title=f"Total {volume_total_mwh:.0f} MWh"),
        color=alt.Color('Part:N', scale=alt.Scale(range=["#4CAF50","#BDBDBD"])),
        tooltip=['Part','MWh','Info']
    ).properties(height=40, width=600)
    st.altair_chart(cover_chart, use_container_width=True)
    st.caption(f"Couverture : **{couverture_pct:.1f} %**")

    # === 4) Bloc recommandation simple (prix marché vs prix fixe)
    st.subheader("Recommandation simple (prix marché vs prix fixe)")
    # Dernier prix marché : si tu as déjà chargé les données marché, on réutilise; sinon on met None.
    dernier_prix_marche = None
    if 'daily' in locals() and not daily.empty:
        dernier_prix_marche = float(daily.iloc[-1]["avg"])  # J-1
    # fallback : tu peux hardcoder une valeur si besoin
    dernier_prix_marche = st.number_input("Dernier prix marché (€/MWh)", min_value=0.0,
                                          value=dernier_prix_marche if dernier_prix_marche else 100.0, step=1.0)
    colR1, colR2 = st.columns(2)
    if dernier_prix_marche < prix_fixe_moyen:
        colR1.success("🟢 Prix intéressant.")
        colR2.write("Le prix de marché est **inférieur** à votre prix fixe moyen → vous pouvez envisager **de fixer** une tranche supplémentaire.")
    elif dernier_prix_marche > prix_fixe_moyen:
        colR1.error("🔴 Prix élevé.")
        colR2.write("Le prix de marché est **supérieur** à votre prix fixe moyen → **attendez** une fenêtre plus favorable.")
    else:
        colR1.info("🟡 Égalité.")
        colR2.write("Le prix de marché est proche de votre prix fixe moyen → pas de signal évident.")

    # Petit rappel des chiffres
    st.caption(
        f"📌 Récap : Prix fixe moyen **{prix_fixe_moyen:.2f} €/MWh**, "
        f"Dernier prix marché **{dernier_prix_marche:.2f} €/MWh**, "
        f"Couverture **{couverture_pct:.1f}%**."
    )
