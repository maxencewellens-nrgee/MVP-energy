# app.py — MVP Énergie (BE Day-Ahead + Contrat + FlexyPower CAL)
import streamlit as st
import pandas as pd
import altair as alt
import requests, re
from entsoe import EntsoePandasClient
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import pytz
import html as ihtml
import unicodedata


# ----------------------------- Config
st.set_page_config(page_title="MVP Énergie — BE Day-Ahead", layout="wide")
st.title("Gestion contrat futur, Recommandation & Prise de décision")

# ----------------------------- Secrets / Token
TOKEN = st.secrets.get("ENTSOE_TOKEN", "")
if not TOKEN:
    st.error("Secret ENTSOE_TOKEN manquant (Streamlit Cloud → Settings → Secrets).")
    st.stop()

# ----------------------------- Constantes
ZONE = "10YBE----------2"  # Belgique
tz_utc = pytz.UTC
tz_be  = pytz.timezone("Europe/Brussels")
client = EntsoePandasClient(api_key=TOKEN)

# ----------------------------- Helpers
def fmt_be(d) -> str:
    """Format JJ/MM/AAAA."""
    return pd.to_datetime(d).strftime("%d/%m/%Y")

@st.cache_data(ttl=24*3600)
def fetch_daily(start_date: str, end_inclusive_date: str) -> pd.DataFrame:
    """
    Récupère prix day-ahead (heure) via entsoe-py, agrège en jour (avg/mn/mx/n) en heure BE.
    start_date, end_inclusive_date: 'YYYY-MM-DD'. end est inclus (on ajoute +1 jour côté API).
    """
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
    out = df.groupby("date")["price"].agg(avg="mean", mn="min", mx="max", n="count").reset_index()
    out[["avg","mn","mx"]] = out[["avg","mn","mx"]].round(2)
    return out

def decision_from_last(daily: pd.DataFrame, lookback_days: int = 180) -> dict:
    """
    Décision ancrée sur le dernier prix (J-1) avec garde-fous par quantiles (P10/P30/P70)
    calculés sur une fenêtre lookback_days (par défaut 180).
    """
    if daily.empty:
        return {"reco":"—","raison":"Pas de données.","last":None,"p10":None,"p30":None,"p70":None}
    last_price = float(daily.iloc[-1]["avg"])
    ref_end = pd.to_datetime(daily["date"].max())
    ref_start = (ref_end - pd.Timedelta(days=lookback_days)).date()
    ref = daily[daily["date"] >= ref_start]
    if len(ref) < 30:
        ref = daily
    p10 = ref["avg"].quantile(0.10)
    p30 = ref["avg"].quantile(0.30)
    p70 = ref["avg"].quantile(0.70)

    if last_price <= p10:
        return {"reco":"FIXER 40–60 %","raison":f"Dernier prix {last_price:.2f} ≤ P10 {p10:.2f} : opportunité forte.",
                "last":round(last_price,2),"p10":round(p10,2),"p30":round(p30,2),"p70":round(p70,2)}
    if last_price <= p30:
        return {"reco":"FIXER 20–30 %","raison":f"Dernier prix {last_price:.2f} ≤ P30 {p30:.2f} : fenêtre favorable.",
                "last":round(last_price,2),"p10":round(p10,2),"p30":round(p30,2),"p70":round(p70,2)}
    if last_price >= p70:
        return {"reco":"ATTENDRE (clairement)","raison":f"Dernier prix {last_price:.2f} ≥ P70 {p70:.2f} : marché cher.",
                "last":round(last_price,2),"p10":round(p10,2),"p30":round(p30,2),"p70":round(p70,2)}
    return {"reco":"ATTENDRE","raison":f"Dernier prix {last_price:.2f} entre P30 {p30:.2f} et P70 {p70:.2f} : pas de signal fort.",
            "last":round(last_price,2),"p10":round(p10,2),"p30":round(p30,2),"p70":round(p70,2)}

@st.cache_data(ttl=60*30)

@st.cache_data(ttl=60*30)
def fetch_flexypower_cals(url: str = "https://flexypower.eu/prix-de-lenergie/", debug: bool = False) -> dict:
    """
    Récupère CAL-26/27/28 (électricité) + date JJ/MM/AAAA depuis flexypower.eu.
    Retour: {'CAL-26': float|None, 'CAL-27': float|None, 'CAL-28': float|None, 'date': str|None}
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
    }
    vals = {"CAL-26": None, "CAL-27": None, "CAL-28": None, "date": None}
    try:
        r = requests.get(url, headers=headers, timeout=25)
        r.raise_for_status()
        raw = r.text

        # Normalise : entités HTML, espaces, accents
        txt = ihtml.unescape(raw).replace("\xa0", " ")
        txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")
        txt_compact = re.sub(r"\s+", " ", txt)

        # Isole bloc Electricite
        m_elec = re.search(r"Electricite.*?(?=Gaz naturel|<h2|</section>|$)", txt_compact, flags=re.I)
        block = m_elec.group(0) if m_elec else txt_compact

        # Date JJ/MM/AAAA
        dm = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", block)
        if dm:
            vals["date"] = dm.group(1)

        # CAL-26/27/28 (tolère "CAL 26" / "CAL-26")
        for yy in ("26", "27", "28"):
            m = re.search(rf"CAL\s*[-]?\s*{yy}\D*?([0-9]+(?:[.,][0-9]+)?)", block, flags=re.I)
            if m:
                try:
                    vals[f"CAL-{yy}"] = float(m.group(1).replace(",", "."))
                except:
                    vals[f"CAL-{yy}"] = None

        # Fallback: tenter read_html si rien trouvé
        if all(vals[k] is None for k in ("CAL-26", "CAL-27", "CAL-28")):
            try:
                tables = pd.read_html(raw)  # nécessite lxml
                for df in tables:
                    df_cols_u = [str(c).upper() for c in df.columns]
                    prod_col = next((c for c in df.columns if str(c).upper() in df_cols_u and
                                     ("PRODUIT" in str(c).upper() or "PRODUCT" in str(c).upper() or "CAL" == str(c).upper())), None)
                    price_col = next((c for c in df.columns if "PRIX" in str(c).upper() or "PRICE" in str(c).upper()), None)
                    if prod_col is None or price_col is None:
                        if len(df.columns) >= 2:
                            prod_col, price_col = df.columns[0], df.columns[1]
                        else:
                            continue
                    for yy in ("26", "27", "28"):
                        mask = df[prod_col].astype(str).str.upper().str.contains(f"CAL[- ]?{yy}")
                        if mask.any():
                            rawv = str(df.loc[mask, price_col].iloc[0])
                            rawv = re.sub(r"[^\d,\.]", "", rawv).replace(",", ".")
                            try:
                                vals[f"CAL-{yy}"] = float(rawv)
                            except:
                                pass
            except Exception:
                pass

        if debug:
            st.write("DEBUG FlexyPower:", {"len_html": len(raw), "found_elec_block": bool(m_elec), "vals": vals})
        return vals

    except Exception as e:
        if debug:
            st.write("DEBUG FlexyPower Exception:", str(e))
        return vals


# ----------------------------- Marché : bornes automatiques (sans UI)
today_be = datetime.now(tz_be).date()
END_INCLUSIVE = str(today_be - timedelta(days=1))   # J-1
START_HISTORY = "2023-01-01"                        # élargis si tu veux plus long
LOOKBACK_DAYS = 180                                 # pour les quantiles (non visible côté client)

# Variables utilisées plus bas (pas d’UI publique)
start_input = START_HISTORY
end_input   = END_INCLUSIVE
lookback    = LOOKBACK_DAYS
run_market  = False  # plus de bouton; chargement auto géré dans le bloc suivant

# ----------------------------- Marché : chargement & affichage (AUTO)
def load_market(start_date: str, end_date: str):
    with st.spinner("Récupération ENTSO-E (par mois)…"):
        data = fetch_daily(start_date, end_date)
    return data

# 1) Auto-chargement au premier affichage (et à chaque redeploy/cache clear)
if "market_daily" not in st.session_state:
    try:
        st.session_state["market_daily"] = load_market(start_input, end_input)
        st.session_state["market_params"] = (start_input, end_input, lookback)
    except Exception as e:
        st.error(f"Erreur : {e}")
        st.stop()

# 2) Rendu permanent (toujours visible)
daily = st.session_state.get("market_daily", pd.DataFrame())
if daily.empty:
    st.error("Aucune donnée sur l'intervalle demandé.")
else:
    # Titre demandé
    st.subheader("Historique prix marché électricité")

    # GRAND GRAPHIQUE
    vis = daily.copy()
    vis["date"] = pd.to_datetime(vis["date"])
    chart = (
        alt.Chart(vis)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("avg:Q", title="€/MWh")
        )
        .properties(height=420, width="container")
    )
    st.altair_chart(chart, use_container_width=True)

  # ----------------------------- Synthèse (REMPLACEMENT ENTIER, AUTO-SUFFISANT)
st.subheader("Synthèse")

# ----------------------------- Synthèse (remplacement entier, autonome)
st.subheader("Synthèse")

_daily = st.session_state.get("market_daily", pd.DataFrame())
if _daily.empty:
    st.error("Aucune donnée marché chargée (daily vide).")
else:
    daily_syn = _daily.copy()

    # KPI spot
    overall_avg = round(daily_syn["avg"].mean(), 2)
    last = daily_syn.iloc[-1]
    last_day_dt = pd.to_datetime(daily_syn["date"].max())
    mask_month = (
        (pd.to_datetime(daily_syn["date"]).dt.year == last_day_dt.year) &
        (pd.to_datetime(daily_syn["date"]).dt.month == last_day_dt.month)
    )
    month_avg = round(daily_syn.loc[mask_month, "avg"].mean(), 2)

    k1, k2, k3 = st.columns(3)
    k1.metric("Moyenne depuis le début visible", f"{overall_avg} €/MWh")
    k2.metric("Moyenne mois en cours (jusqu’à J−1)", f"{month_avg} €/MWh")
    k3.metric("Dernier prix accessible (J−1)", f"{last['avg']:.2f} €/MWh")

    # Forwards CAL (FlexyPower)
    try:
        cal = fetch_flexypower_cals()
        cal_date = cal.get("date") or "—"
        f1, f2, f3 = st.columns(3)
        f1.metric(f"CAL-26 (élec) – {cal_date}",
                  f"{cal.get('CAL-26'):.2f} €/MWh" if cal.get('CAL-26') is not None else "—")
        f2.metric(f"CAL-27 (élec) – {cal_date}",
                  f"{cal.get('CAL-27'):.2f} €/MWh" if cal.get('CAL-27') is not None else "—")
        f3.metric(f"CAL-28 (élec) – {cal_date}",
                  f"{cal.get('CAL-28'):.2f} €/MWh" if cal.get('CAL-28') is not None else "—")
    except Exception as e:
        st.warning(f"CAL FlexyPower indisponible : {e}")

    
# ----------------------------- Contrat : formulaire & couverture
st.subheader("Contrat client — entrées")
with st.form("form_contrat"):
    col1, col2, col3 = st.columns(3)
    with col1:
        date_debut_contrat = st.date_input("Date début contrat", value=date(datetime.now().year, 1, 1))
    with col2:
        duree_contrat_mois = st.radio("Durée du contrat", options=[12, 24, 36], index=2, format_func=lambda m: f"{m//12} an(s)")
    with col3:
        volume_total_mwh = st.number_input("Volume total (MWh)", min_value=0.0, value=200.0, step=10.0)

    col4, col5 = st.columns(2)
    with col4:
        volume_deja_fixe_mwh = st.number_input("Volume déjà fixé (MWh)", min_value=0.0, value=120.0, step=10.0)
    with col5:
        prix_fixe_moyen = st.number_input("Prix fixe moyen (€/MWh)", min_value=0.0, value=85.0, step=1.0)

    submit_contrat = st.form_submit_button("Mettre à jour le contrat")

if submit_contrat:
    # Fin = début + durée - 1 jour
    date_fin_contrat = (date_debut_contrat + relativedelta(months=duree_contrat_mois)) - timedelta(days=1)

    # Couverture %
    couverture_pct = 0.0 if volume_total_mwh == 0 else min(100.0, round(100 * volume_deja_fixe_mwh / volume_total_mwh, 2))
    reste_mwh = max(0.0, volume_total_mwh - volume_deja_fixe_mwh)

    st.success(f"Début : **{fmt_be(date_debut_contrat)}**  ·  Fin : **{fmt_be(date_fin_contrat)}**")

    # "Couverture du contrat en cours" (titre + chiffres)
    st.subheader("Couverture du contrat en cours")
    cA, cB, cC = st.columns(3)
    cA.metric("Couverture", f"{couverture_pct:.1f} %")
    cB.metric("Fixé", f"{volume_deja_fixe_mwh:.0f} MWh")
    cC.metric("À fixer", f"{reste_mwh:.0f} MWh")

    # (Optionnel) rappel prix marché vs prix fixe moyen
    st.caption(f"Référence : Prix fixe moyen **{prix_fixe_moyen:.2f} €/MWh**."
               " Pour une décision, croisez avec la section 'Décision (ancrée sur le dernier prix)'.")
