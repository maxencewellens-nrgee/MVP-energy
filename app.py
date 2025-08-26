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
import urllib.parse 

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

# --- FlexyPower: robuste (normal -> fallback via r.jina.ai -> read_html) ---

@st.cache_data(ttl=60*30)
def fetch_flexypower_cals(url: str = "https://flexypower.eu/prix-de-lenergie/", debug: bool = False) -> dict:
    """
    Essaie 3 voies : HTML brut -> proxy textuel (r.jina.ai) -> pandas.read_html.
    Retour: {'CAL-26':float|None,'CAL-27':...,'CAL-28':...,'date':str|None}
    """
    def _normalize(raw: str) -> str:
        txt = ihtml.unescape(raw).replace("\xa0", " ")
        txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"\s+", " ", txt)

    def _parse_block(text: str) -> dict:
        vals = {"CAL-26": None, "CAL-27": None, "CAL-28": None, "date": None}
        # isole bloc Electricite si possible
        m_elec = re.search(r"Electricite.*?(?=Gaz naturel|<h2|</section>|$)", text, flags=re.I)
        block = m_elec.group(0) if m_elec else text
        dm = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", block)
        if dm: vals["date"] = dm.group(1)
        for yy in ("26","27","28"):
            m = re.search(rf"CAL\s*[-]?\s*{yy}\D*?([0-9]+(?:[.,][0-9]+)?)", block, flags=re.I)
            if m:
                try:
                    vals[f"CAL-{yy}"] = float(m.group(1).replace(",", "."))
                except: pass
        return vals

    # 1) tentative HTML direct
    try:
        r = requests.get(url, headers={
            "User-Agent":"Mozilla/5.0", "Accept-Language":"fr-FR,fr;q=0.9,en;q=0.8"
        }, timeout=20)
        r.raise_for_status()
        vals = _parse_block(_normalize(r.text))
        if any(vals[k] is not None for k in ("CAL-26","CAL-27","CAL-28")):
            if debug: st.write("FlexyPower: direct OK")
            return vals
    except Exception as e:
        if debug: st.write("Flexy direct FAIL:", e)

    # 2) fallback proxy texte (souvent ça contourne JS/anti-bot)
    try:
        pr = urllib.parse.urlparse(url)
        proxy_url = f"https://r.jina.ai/http://{pr.netloc}{pr.path}"
        r2 = requests.get(proxy_url, headers={"User-Agent":"Mozilla/5.0"}, timeout=20)
        r2.raise_for_status()
        vals = _parse_block(_normalize(r2.text))
        if any(vals[k] is not None for k in ("CAL-26","CAL-27","CAL-28")):
            if debug: st.write("FlexyPower: proxy r.jina.ai OK")
            return vals
    except Exception as e:
        if debug: st.write("Flexy proxy FAIL:", e)

    # 3) dernier recours: read_html sur la page d’origine
    try:
        tables = pd.read_html(url)
        for df in tables:
            cols = [str(c).upper() for c in df.columns]
            prod_col = df.columns[0]
            price_col = df.columns[1] if len(df.columns)>1 else df.columns[0]
            for yy in ("26","27","28"):
                mask = df[prod_col].astype(str).str.upper().str.contains(f"CAL[- ]?{yy}")
                if mask.any():
                    rawv = str(df.loc[mask, price_col].iloc[0])
                    rawv = re.sub(r"[^\d,\.]", "", rawv).replace(",", ".")
                    try:
                        outv = float(rawv)
                    except:
                        outv = None
                    # on construit un résultat incrémental
                    if 'vals' not in locals():
                        vals = {"CAL-26": None, "CAL-27": None, "CAL-28": None, "date": None}
                    vals[f"CAL-{yy}"] = outv
        if 'vals' in locals():
            if debug: st.write("FlexyPower: read_html OK")
            return vals
    except Exception as e:
        if debug: st.write("Flexy read_html FAIL:", e)

    # échec total
    return {"CAL-26": None, "CAL-27": None, "CAL-28": None, "date": None}



# ----------------------------- Marché : bornes automatiques (sans UI)
today_be = datetime.now(tz_be).date()
END_INCLUSIVE = str(today_be - timedelta(days=1))   # J-1
START_HISTORY = "2025-01-01"                        # élargis si tu veux plus long
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

    # GRAND GRAPHIQUE + moyenne mobile
st.subheader("Moyenne 30-60-90 jours")

# Choix fenêtre pour la moyenne mobile
mm_window = st.selectbox("Moyenne mobile (jours)", [30, 60, 90], index=0, key="mm_win")

vis = daily.copy()
vis["date"] = pd.to_datetime(vis["date"])
# moyenne mobile simple (SMA). min_periods pour éviter trous au début
vis = vis.sort_values("date")
vis["sma"] = vis["avg"].rolling(window=int(mm_window), min_periods=max(5, int(mm_window)//3)).mean()

price_line = (
    alt.Chart(vis)
    .mark_line()
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("avg:Q", title="€/MWh"),
        color=alt.value("#1f2937"),  # ligne principale (optionnel)
        tooltip=[alt.Tooltip("date:T", title="Date"),
                 alt.Tooltip("avg:Q", title="BE spot (€/MWh)", format=".2f"),
                 alt.Tooltip("sma:Q", title=f"SMA {mm_window}j (€/MWh)", format=".2f")]
    )
)

sma_line = (
    alt.Chart(vis.dropna(subset=["sma"]))
    .mark_line(strokeWidth=3)
    .encode(
        x="date:T",
        y="sma:Q",
        color=alt.value("#22c55e")  # vert pour la moyenne (optionnel)
    )
)

chart = (price_line + sma_line).properties(height=420, width="container")
st.altair_chart(chart, use_container_width=True)


# ----------------------------- Synthèse (unique)
st.subheader("Synthèse Prix Spot et Forward")

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

    # CAL FlexyPower (utilise ta fonction fetch_flexypower_cals() définie plus haut)
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

    
# ===================== CONTRAT — VOLUME & CLICS (REMPLACEMENT ENTIER) =====================
st.markdown("---")
st.subheader("Contrat client — entrées & clics")

# --- 0) État initial en session
if "contrat_total_mwh" not in st.session_state:
    st.session_state["contrat_total_mwh"] = 200.0  # valeur par défaut
if "contrat_fixes" not in st.session_state:
    # liste de dicts: {"date": date, "price": float €/MWh, "volume": float MWh}
    st.session_state["contrat_fixes"] = []

# --- 1) Paramètres principaux (volume total)
colA, colB = st.columns([2,1])
with colA:
    total_mwh = st.number_input("Volume total (MWh)", min_value=0.0,
                                value=float(st.session_state["contrat_total_mwh"]), step=10.0,
                                help="Volume total du contrat à couvrir.")
    st.session_state["contrat_total_mwh"] = total_mwh

# --- 2) Formulaire d'ajout d'un 'clic' (blocage)
with st.expander("Ajouter un clic (blocage)"):
    c1, c2, c3 = st.columns(3)
    with c1:
        d_click = st.date_input("Date du clic", value=date.today(), key="click_date")
    with c2:
        p_click = st.number_input("Prix (€/MWh)", min_value=0.0, value=0.0, step=0.1, key="click_price")
    with c3:
        v_click = st.number_input("Volume (MWh)", min_value=0.0, value=0.0, step=1.0, key="click_volume")

    add = st.button("➕ Ajouter ce clic")
    if add:
        if v_click <= 0 or p_click <= 0:
            st.warning("Prix et volume doivent être > 0.")
        else:
            st.session_state["contrat_fixes"].append({
                "date": d_click,
                "price": float(p_click),
                "volume": float(v_click),
            })
            st.success("Clic ajouté.")
            # reset inputs (optionnel)
        

# ===== Affichage sous le formulaire : synthèse des clics + totaux =====
fixes = st.session_state.get("contrat_fixes", [])
total_mwh = float(st.session_state.get("contrat_total_mwh", 0.0))

fixes_df = pd.DataFrame(fixes)
if fixes_df.empty:
    st.info("Aucun clic enregistré pour l’instant.")
else:
    # Tableau des clics
    fixes_df = fixes_df.copy()
    fixes_df["date"] = pd.to_datetime(fixes_df["date"]).dt.date
    fixes_df["% du total"] = fixes_df["volume"].apply(
        lambda v: round((v / total_mwh * 100.0), 2) if total_mwh > 0 else 0.0
    )

    st.markdown("### Clics enregistrés")
    st.dataframe(
        fixes_df.rename(columns={
            "date": "Date",
            "price": "Prix (€/MWh)",
            "volume": "Volume (MWh)"
        })[["Date", "Prix (€/MWh)", "Volume (MWh)", "% du total"]],
        use_container_width=True
    )

    # Totaux
    fixed_mwh = float(fixes_df["Volume (MWh)"].sum())
    restant_mwh = max(0.0, total_mwh - fixed_mwh)
    pct_couverture = round((fixed_mwh / total_mwh * 100.0), 2) if total_mwh > 0 else 0.0
    pmp = round(
        (fixes_df["Prix (€/MWh)"] * fixes_df["Volume (MWh)"]).sum() / fixed_mwh, 2
    ) if fixed_mwh > 0 else None

    st.markdown("### Synthèse du contrat")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Volume total", f"{total_mwh:.0f} MWh")
    m2.metric("Total cliqué", f"{fixed_mwh:.0f} MWh")
    m3.metric("Total restant", f"{restant_mwh:.0f} MWh")
    m4.metric("Couverture", f"{pct_couverture:.1f} %")

    if pmp is not None:
        st.caption(f"Prix moyen pondéré des clics : **{pmp:.2f} €/MWh**")

    # (optionnel) barre de progression
    st.progress(min(1.0, pct_couverture/100.0))

    # Export CSV
    csv = fixes_df.rename(columns={"% du total":"pct_total_%"}).to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger l’historique des clics (CSV)", data=csv,
                       file_name="clics_blocages.csv", mime="text/csv")

# --- 3) Tableau des clics + suppression
fixes_df = pd.DataFrame(st.session_state["contrat_fixes"])
if not fixes_df.empty:
    # % calculé par rapport au total
    fixes_df = fixes_df.copy()
    fixes_df["pct_total_%"] = fixes_df["volume"].apply(
        lambda v: round((v / total_mwh * 100.0), 2) if total_mwh > 0 else 0.0
    )
    fixes_df["date"] = pd.to_datetime(fixes_df["date"]).dt.date

    st.markdown("**Clics enregistrés**")
    st.dataframe(
        fixes_df.rename(columns={
            "date": "Date",
            "price": "Prix (€/MWh)",
            "volume": "Volume (MWh)",
            "pct_total_%": "% du total"
        }),
        use_container_width=True
    )

    # Suppression d'une ligne
    del_col1, del_col2 = st.columns([3,1])
    with del_col1:
        idx_to_delete = st.selectbox(
            "Supprimer un clic (sélectionne la ligne)",
            options=list(range(len(fixes_df))),
            format_func=lambda i: f"{i+1} — {fixes_df.iloc[i]['date']} | {fixes_df.iloc[i]['volume']} MWh @ {fixes_df.iloc[i]['price']} €/MWh",
            index=0
        )
    with del_col2:
        if st.button("🗑️ Supprimer"):
            st.session_state["contrat_fixes"].pop(int(idx_to_delete))
            st.experimental_rerun()
else:
    st.info("Aucun clic enregistré pour l’instant.")

# --- 4) Synthèse contrat (totaux, restant, % couverture, prix moyen pondéré)
fixes_df = pd.DataFrame(st.session_state["contrat_fixes"])
fixed_mwh = float(fixes_df["volume"].sum()) if not fixes_df.empty else 0.0
restant_mwh = max(0.0, total_mwh - fixed_mwh)
pct_couverture = round((fixed_mwh / total_mwh * 100.0), 2) if total_mwh > 0 else 0.0
pmp = round((fixes_df["price"].mul(fixes_df["volume"]).sum() / fixed_mwh), 2) if fixed_mwh > 0 else None

st.subheader("Couverture du contrat en cours")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Volume total", f"{total_mwh:.0f} MWh")
m2.metric("Total fixé", f"{fixed_mwh:.0f} MWh")
m3.metric("Total restant à cliquer", f"{restant_mwh:.0f} MWh")
m4.metric("Couverture", f"{pct_couverture:.1f} %")

# Affiche le prix moyen pondéré des clics si dispo
if pmp is not None:
    st.caption(f"Prix moyen pondéré des clics : **{pmp:.2f} €/MWh**")

# Export CSV (historique des clics)
if not fixes_df.empty:
    csv_bytes = fixes_df.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger l’historique des clics (CSV)", data=csv_bytes,
                       file_name="clics_blocages.csv", mime="text/csv")
