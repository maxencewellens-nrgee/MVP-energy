
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

# ----------------------------- Configuration
st.set_page_config(page_title="MVP Énergie — BE Day-Ahead", layout="wide")
st.title("Gérer mes contrats; recommandations & prise de décision")

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
    Décision basée sur le dernier prix et des quantiles P10/P30/P70
    calculés sur une fenêtre de lookback. Robuste aux types de dates.
    """
    if daily.empty:
        return {"reco":"—","raison":"Pas de données.","last":None,"p10":None,"p30":None,"p70":None}

    df = daily.copy()
    # 🔧 Standardise: s'assure que 'date' est bien datetime64[ns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    last_price = float(df.iloc[-1]["avg"])
    ref_end = df["date"].max()                           # Timestamp
    ref_start = ref_end - pd.Timedelta(days=lookback_days)  # Timestamp

    # Filtre sur la fenêtre
    ref = df[df["date"] >= ref_start]
    if len(ref) < 30:
        ref = df

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

# ===================== Graphique interactif BE spot =====================

mm_window = st.selectbox("Moyenne mobile (jours)", [30, 60, 90], index=0, key="mm_win")

# --- Préparation des données
vis = daily.copy()
vis["date"] = pd.to_datetime(vis["date"])
vis = vis.sort_values("date")
vis["sma"] = vis["avg"].rolling(
    window=int(mm_window),
    min_periods=max(5, int(mm_window)//3)
).mean()

# Champs formatés FR pour affichage
vis["date_str"] = vis["date"].dt.strftime("%d/%m/%y")                      # ex : 21/05/25
vis["spot_str"] = vis["avg"].apply(lambda v: f"{v:.2f}".replace(".", ",") + "€")  # ex : 99,75€

# --- Sélection souris : suit le mouvement, ne se vide jamais
hover = alt.selection_point(
    fields=["date"],
    nearest=True,
    on="mousemove",   # pas besoin de viser la boule
    empty="none",
    clear=False       # reste affiché quand on sort du graphe
)

base = alt.Chart(vis).encode(
    x=alt.X("date:T", title="Date")
)

# Courbe spot (sans tooltip)
spot_line = base.mark_line(strokeWidth=1.5, color="#1f2937").encode(
    y=alt.Y("avg:Q", title="€/MWh"),
    tooltip=[]
)

# Courbe moyenne mobile (sans tooltip)
sma_line = base.transform_filter("datum.sma != null").mark_line(
    strokeWidth=3, color="#22c55e"
).encode(
    y="sma:Q",
    tooltip=[]
)

# Points invisibles pour accrocher la sélection
points = base.mark_point(opacity=0).encode(
    y="avg:Q",
    tooltip=[]
).add_params(hover)

# Point visible au survol
hover_point = base.mark_circle(size=60, color="#1f2937").encode(
    y="avg:Q",
    tooltip=[]
).transform_filter(hover)

# Règle verticale — réactive le "cadran" (tooltip)
v_rule = base.mark_rule(color="#9ca3af").encode(
    tooltip=[
        alt.Tooltip("date_str:N", title="Date"),
        alt.Tooltip("spot_str:N", title="Spot")
    ]
).transform_filter(hover)

# Labels persistants (avec halo blanc + décalage pour ne pas coller à la boule)
label_price_halo = base.mark_text(
    dx=14, dy=-16, align="left", fontSize=12, fontWeight="bold",
    stroke="white", strokeWidth=5, opacity=1
).encode(
    y="avg:Q",
    text="spot_str:N"
).transform_filter(hover)

label_price = base.mark_text(
    dx=14, dy=-16, align="left", fontSize=12, fontWeight="bold",
    color="#111827", opacity=1
).encode(
    y="avg:Q",
    text="spot_str:N"
).transform_filter(hover)

label_date_halo = base.mark_text(
    dx=14, dy=4, align="left", fontSize=11,
    stroke="white", strokeWidth=5, opacity=1
).encode(
    y="avg:Q",
    text="date_str:N"
).transform_filter(hover)

label_date = base.mark_text(
    dx=14, dy=4, align="left", fontSize=11, color="#374151", opacity=1
).encode(
    y="avg:Q",
    text="date_str:N"
).transform_filter(hover)

chart = alt.layer(
    spot_line, sma_line, points, v_rule, hover_point,
    label_price_halo, label_price,
    label_date_halo, label_date
).properties(
    height=420, width="container"
).interactive()

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
    k2.metric("Moyenne mois en cours", f"{month_avg} €/MWh")
    k3.metric("Dernier prix accessible", f"{last['avg']:.2f} €/MWh")

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

# ================= Récapitulatif contrats passés (2024 / 2025) =================
st.markdown("### Contrats passés — récapitulatif 2024 / 2025")

# Récupérer les inputs (tu peux les mettre dans une sidebar ou un form ailleurs)
volume_2024 = st.session_state.get("contrat2024_volume", 250)
prix_2024 = st.session_state.get("contrat2024_prix", 95.0)
budget_2024 = volume_2024 * prix_2024

volume_2025 = st.session_state.get("contrat2025_volume", 300)
prix_2025 = st.session_state.get("contrat2025_prix", 103.0)
budget_2025 = volume_2025 * prix_2025

# Cartes sous forme verticale (longues et fines)
with st.container():
    st.markdown("#### Contrat 2024 (fixe saisi)")
    st.write(f"**Volume :** {volume_2024} MWh")
    st.write(f"**Prix :** {prix_2024:.2f} €/MWh")
    st.write(f"**Budget total :** {budget_2024:,.0f} €")
    st.caption(f"Calcul : {volume_2024} MWh × {prix_2024:.2f} €/MWh = {budget_2024:,.0f} €")

with st.container():
    st.markdown("#### Contrat 2025 (fixe saisi)")
    st.write(f"**Volume :** {volume_2025} MWh")
    st.write(f"**Prix :** {prix_2025:.2f} €/MWh")
    st.write(f"**Budget total :** {budget_2025:,.0f} €")
    st.caption(f"Calcul : {volume_2025} MWh × {prix_2025:.2f} €/MWh = {budget_2025:,.0f} €")


# ===================== DÉCIDEUR DE CLIC — IMPACT BUDGET =====================
st.subheader("Décider un clic maintenant (par année)")

# 0) S'assure d'avoir les CAL du jour disponibles ici
cal_used = st.session_state.get("CAL_USED")
cal_date = st.session_state.get("CAL_DATE", "—")
if cal_used is None:
    try:
        _cal = fetch_flexypower_cals()
    except Exception:
        _cal = {"CAL-26": None, "CAL-27": None, "CAL-28": None, "date": None}
    CAL_FALLBACK = {"CAL-26": 84.13, "CAL-27": 79.33, "CAL-28": 74.49}
    cal_used = {
        "y2026": _cal.get("CAL-26") or CAL_FALLBACK["CAL-26"],
        "y2027": _cal.get("CAL-27") or CAL_FALLBACK["CAL-27"],
        "y2028": _cal.get("CAL-28") or CAL_FALLBACK["CAL-28"],
    }
    cal_date = _cal.get("date") or "—"
    st.session_state["CAL_USED"] = cal_used
    st.session_state["CAL_DATE"] = cal_date

# 1) Choix de la baseline pour comparer le clic
def _safe_mean(series):
    s = pd.to_numeric(series, errors="coerce")
    return float(s.mean()) if len(s.dropna()) else None

spot_avg_7  = _safe_mean(daily.sort_values("date")["avg"].iloc[-7:])  if not daily.empty and len(daily)>=7  else None
spot_avg_30 = _safe_mean(daily.sort_values("date")["avg"].iloc[-30:]) if not daily.empty and len(daily)>=30 else None
spot_avg_90 = _safe_mean(daily.sort_values("date")["avg"].iloc[-90:]) if not daily.empty and len(daily)>=90 else None

cbl1, cbl2, cbl3 = st.columns([1.2, 1, 1])
with cbl1:
    baseline_choice = st.selectbox(
        "Comparer mon clic à…",
        ["CAL du jour (delta = 0)", "Variable — moyenne 7j", "Variable — moyenne 30j", "Variable — moyenne 90j", "Mon scénario (€/MWh)"],
        index=2
    )
with cbl2:
    if baseline_choice == "Mon scénario (€/MWh)":
        baseline_scenario = st.number_input("Prix scénario (€/MWh)", min_value=0.0, step=1.0, format="%.2f", key="baseline_scenario")
    else:
        baseline_scenario = None
with cbl3:
    st.caption(f"Prix CAL du jour (source {cal_date}) — "
               f"26: {cal_used['y2026']:.2f} €/MWh | 27: {cal_used['y2027']:.2f} €/MWh | 28: {cal_used['y2028']:.2f} €/MWh")

def baseline_price():
    if baseline_choice == "CAL du jour (delta = 0)":
        return None  # on traitera “delta=0” année par année (CAL ≈ baseline)
    if baseline_choice == "Variable — moyenne 7j":
        return spot_avg_7
    if baseline_choice == "Variable — moyenne 30j":
        return spot_avg_30
    if baseline_choice == "Variable — moyenne 90j":
        return spot_avg_90
    return baseline_scenario

# 2) Fonctions utilitaires
def _fmt_eur(amount, dec=0):
    if amount is None:
        return "—"
    s = f"{amount:,.{dec}f}".replace(",", " ")
    return f"{s} €"

def compute_state(ns: str):
    total_key  = f"{ns}__total_mwh"
    clicks_key = f"{ns}__clicks"
    total_mwh  = float(st.session_state.get(total_key, 0.0))
    df_clicks  = pd.DataFrame(st.session_state.get(clicks_key, []))
    if not df_clicks.empty:
        df_clicks["volume"] = pd.to_numeric(df_clicks["volume"], errors="coerce").fillna(0.0)
        df_clicks["price"]  = pd.to_numeric(df_clicks["price"],  errors="coerce").fillna(0.0)
    fixed_mwh = float(df_clicks["volume"].sum()) if not df_clicks.empty else 0.0
    fixed_mwh = min(fixed_mwh, total_mwh) if total_mwh > 0 else 0.0
    rest_mwh  = max(0.0, total_mwh - fixed_mwh)
    avg_pond  = (df_clicks["price"] * df_clicks["volume"]).sum()/fixed_mwh if fixed_mwh > 0 else None
    cal_now   = cal_used.get(ns)
    return total_mwh, fixed_mwh, rest_mwh, avg_pond, cal_now

def render_decider(ns: str, year_label: str):
    total_mwh, fixed_mwh, rest_mwh, avg_pond, cal_now = compute_state(ns)

    st.markdown(f"### {year_label} — volume restant **{rest_mwh:.0f} MWh**  |  CAL du jour **{(cal_now or 0):.2f} €/MWh**")

    # Slider de décision
    pct = st.slider(f"% du volume restant à cliquer ({year_label})", 0, 100, 25, 5, key=f"{ns}__decide_pct")
    extra = rest_mwh * pct/100.0

    # Avant clic (référence interne)
    # Total estimé AVANT = (fixé × avg_pond) + (restant × CAL du jour)  -> projection interne
    total_before = (avg_pond or 0.0) * fixed_mwh + (cal_now or 0.0) * rest_mwh
    unit_before  = (total_before/total_mwh) if total_mwh > 0 else None

    # Après clic : on fixe 'extra' au CAL du jour
    new_fixed_mwh   = fixed_mwh + extra
    new_fixed_cost  = (avg_pond or 0.0) * fixed_mwh + (cal_now or 0.0) * extra
    remaining_after = max(0.0, total_mwh - new_fixed_mwh)
    projected_after = (cal_now or 0.0) * remaining_after
    total_after     = new_fixed_cost + projected_after
    unit_after      = (total_after/total_mwh) if total_mwh > 0 else None

    # NB: si la projection utilise déjà le CAL du jour pour le restant, total_after == total_before (delta=0).
    # Pour donner un "impact budget" intuitif, on compare le clic au PRIX DE BASELINE choisi.
    base = baseline_price()  # peut être None (CAL du jour)
    delta_vs_baseline = None
    if extra > 0:
        if base is None:
            # baseline = CAL du jour => delta = 0 (on l'affiche clairement)
            delta_vs_baseline = 0.0
        else:
            delta_vs_baseline = (cal_now - base) * extra if cal_now is not None else None

    # Prix moyen du FIXÉ (utile quand on a déjà 100 MWh @ 95 et on clique à 80)
    fixed_avg_before = avg_pond
    fixed_avg_after  = None
    if new_fixed_mwh > 0:
        fixed_avg_after = ((avg_pond or 0.0) * fixed_mwh + (cal_now or 0.0) * extra) / new_fixed_mwh

    # Affichage des impacts
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Nouveau prix moyen du contrat", f"{unit_after:.2f} €/MWh" if unit_after is not None else "—",
                  delta=(f"{(unit_after - unit_before):+.2f} €/MWh" if (unit_after is not None and unit_before is not None) else None))
    with c2:
        st.metric("Nouveau prix moyen du fixé", f"{fixed_avg_after:.2f} €/MWh" if fixed_avg_after is not None else ("—" if fixed_avg_before is None else f"{fixed_avg_before:.2f} €/MWh"),
                  delta=(f"{( (fixed_avg_after or fixed_avg_before) - (fixed_avg_before or 0) ):+.2f} €/MWh" if fixed_avg_before is not None and fixed_avg_after is not None else None))
    with c3:
        st.metric("Couverture après clic", f"{(new_fixed_mwh/total_mwh*100.0):.1f} %" if total_mwh>0 else "—",
                  delta=(f"{(extra/total_mwh*100.0):+.1f} pts" if total_mwh>0 else None))
    with c4:
        st.metric("Impact vs baseline choisie", _fmt_eur(delta_vs_baseline) if delta_vs_baseline is not None else "—",
                  help=("Baseline = CAL du jour → delta 0 €" if base is None else
                        f"Baseline = {base:.2f} €/MWh ; impact = {extra:.0f} MWh × ({(cal_now or 0):.2f} − {base:.2f})"))

    # Micro-copie explicative
    expl = []
    expl.append(f"Avant clic : prix moyen contrat ≈ {unit_before:.2f} €/MWh" if unit_before is not None else "Avant clic : —")
    if base is None:
        expl.append("Baseline = CAL du jour → un clic déplace du 'projeté' vers du 'fixé' sans changer le total projeté.")
    else:
        expl.append(f"Baseline = {base:.2f} €/MWh (ex: variable {baseline_choice.lower()}).")
    if fixed_avg_after is not None and fixed_avg_before is not None:
        expl.append(f"Le prix moyen du **fixé** passe de {fixed_avg_before:.2f} à {fixed_avg_after:.2f} €/MWh "
                    f"(car on ajoute {extra:.0f} MWh @ {(cal_now or 0):.2f}).")
    st.caption(" • ".join(expl))

# 3) Rendu par année (onglets ou vertical selon ton choix)
st.markdown("#### Simulation par année")
coltabs = st.tabs(["2026", "2027", "2028"])
with coltabs[0]:
    render_decider("y2026", "2026")
with coltabs[1]:
    render_decider("y2027", "2027")
with coltabs[2]:
    render_decider("y2028", "2028")
# ===================== FIN DÉCIDEUR DE CLIC =====================


# ===================== CONTRATS MULTI-MODULES (SIDEBAR + REGLAGES) =====================

# 1) Récup prix CAL depuis la synthèse (fallback si indispo)
try:
    _cal = fetch_flexypower_cals()
except Exception:
    _cal = {"CAL-26": None, "CAL-27": None, "CAL-28": None, "date": None}

CAL_FALLBACK = {"CAL-26": 84.13, "CAL-27": 79.33, "CAL-28": 74.49}
CAL_USED = {
    "y2026": _cal.get("CAL-26") or CAL_FALLBACK["CAL-26"],
    "y2027": _cal.get("CAL-27") or CAL_FALLBACK["CAL-27"],
    "y2028": _cal.get("CAL-28") or CAL_FALLBACK["CAL-28"],
}
CAL_DATE = _cal.get("date") or "—"

YEARS = [("y2026", "2026"), ("y2027", "2027"), ("y2028", "2028")]

def _fmt_eur(amount: float, dec: int = 0) -> str:
    s = f"{amount:,.{dec}f}".replace(",", " ")
    return f"{s} €"

# 2) INIT des clés + PANNEAU LATERAL (réglages par année)
# ===================== BARRE LATÉRALE — CONTRATS =====================
st.sidebar.header("Paramètres contrats")

# --- petit helper local (évite collision de nom)
def _fmt_eur_sb(amount: float, dec: int = 0) -> str:
    s = f"{amount:,.{dec}f}".replace(",", " ")
    return f"{s} €"

# ---------- 1) CONTRATS PASSÉS (saisie simple) : 2024 & 2025 ----------
st.sidebar.subheader("Contrats passés (saisie simple)")

for ns, y in [("y2024", "2024"), ("y2025", "2025")]:
    vol_key   = f"{ns}__fixed_volume"
    price_key = f"{ns}__fixed_price"
    budg_key  = f"{ns}__fixed_budget"

    # init si absent
    if vol_key not in st.session_state:   st.session_state[vol_key] = 0.0
    if price_key not in st.session_state: st.session_state[price_key] = 0.0

    with st.sidebar.expander(f"Contrat {y}", expanded=False):
        st.number_input("Volume (MWh)", min_value=0.0, step=5.0, format="%.0f", key=vol_key)
        st.number_input("Prix (€/MWh)", min_value=0.0, step=1.0, format="%.0f", key=price_key)

        vol   = float(st.session_state[vol_key])
        price = float(st.session_state[price_key])
        budget = vol * price
        st.session_state[budg_key] = budget

        st.metric("Budget total", _fmt_eur_sb(budget))
        st.caption(f"Calcul : {vol:.0f} MWh × {price:.2f} €/MWh = {_fmt_eur_sb(budget)}")

# séparation visuelle
st.sidebar.divider()

# ---------- 2) CONTRATS FUTURS (avec clics) : 2026, 2027, 2028 ----------
st.sidebar.subheader("Contrats futurs (avec clics)")

FUTURE_YEARS = [("y2026", "2026"), ("y2027", "2027"), ("y2028", "2028")]
for ns, y in FUTURE_YEARS:
    total_key  = f"{ns}__total_mwh"
    max_key    = f"{ns}__max_clicks"
    clicks_key = f"{ns}__clicks"
    init_key   = f"{ns}__initialized"

    # init propre si première exécution
    if init_key not in st.session_state:
        st.session_state[total_key]  = 200.0
        st.session_state[max_key]    = 5
        st.session_state[clicks_key] = []
        st.session_state[init_key]   = True

    with st.sidebar.expander(f"Contrat {y}", expanded=(ns == "y2026")):
        st.number_input("Volume total (MWh)", min_value=0.0, step=5.0, format="%.0f", key=total_key)
        st.number_input("Clics max autorisés", min_value=1, max_value=20, step=1, format="%d", key=max_key)
        # rappel rapide d’usage
        used = len(st.session_state.get(clicks_key, []))
        st.caption(f"Clics utilisés : {used}/{int(st.session_state[max_key])}.")
# ===================== FIN BARRE LATÉRALE =====================

# 3) MODULE PAR ANNEE (lit les réglages depuis la sidebar)
def render_contract_module(title: str, ns: str):
    with st.container(border=True):
        st.subheader(title)

        # --- Clés/état
        total_key   = f"{ns}__total_mwh"
        max_key     = f"{ns}__max_clicks"
        clicks_key  = f"{ns}__clicks"
        date_key    = f"{ns}__new_click_date"
        price_key   = f"{ns}__new_click_price"
        vol_key     = f"{ns}__new_click_volume"
        add_btn     = f"{ns}__btn_add_click"
        del_select  = f"{ns}__delete_click_selector"
        del_btn     = f"{ns}__btn_delete_click"
        dl_btn      = f"{ns}__dl_csv"

        total_mwh  = float(st.session_state.get(total_key, 0.0))
        max_clicks = int(st.session_state.get(max_key, 5))
        clicks     = st.session_state.get(clicks_key, [])
        df_clicks  = pd.DataFrame(clicks)

        # types sûrs
        if not df_clicks.empty:
            df_clicks["volume"] = pd.to_numeric(df_clicks["volume"], errors="coerce").fillna(0.0)
            df_clicks["price"]  = pd.to_numeric(df_clicks["price"],  errors="coerce").fillna(0.0)

        fixed_mwh = float(df_clicks["volume"].sum()) if not df_clicks.empty else 0.0
        fixed_mwh = min(fixed_mwh, total_mwh) if total_mwh > 0 else 0.0
        rest_mwh  = max(0.0, total_mwh - fixed_mwh)
        cov_pct   = round((fixed_mwh / total_mwh * 100.0), 2) if total_mwh > 0 else 0.0

        avg_simple = round(float(df_clicks["price"].mean()), 2) if not df_clicks.empty else None
        avg_pond   = round(((df_clicks["price"]*df_clicks["volume"]).sum()/fixed_mwh), 2) if fixed_mwh > 0 else None
        cal_price  = CAL_USED.get(ns)

        # --- (A) Synthèse
        c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1.2])
        c1.metric("Volume total", f"{total_mwh:.0f} MWh", help="Modifiable dans la barre latérale.")
        c2.metric("Déjà fixé", f"{fixed_mwh:.0f} MWh")
        c3.metric("Restant", f"{rest_mwh:.0f} MWh")
        c4.metric("Couverture", f"{cov_pct:.1f} %")
        c5.metric(f"CAL utilisé ({CAL_DATE})", f"{cal_price:.2f} €/MWh" if cal_price is not None else "—",
                  help="Forward utilisé pour estimer le budget restant.")
        st.progress(min(cov_pct/100.0, 1.0), text=f"Couverture {cov_pct:.1f}%")

        # --- (B) Budget (carte unique)
        budget_fixe    = (fixed_mwh * avg_pond) if avg_pond is not None else 0.0
        budget_restant = rest_mwh * float(cal_price or 0.0)
        budget_total   = budget_fixe + budget_restant
        unit_cost      = (budget_total / total_mwh) if total_mwh > 0 else None

        with st.container(border=True):
            st.markdown("#### Budget")
            b1, b2, b3, b4 = st.columns([1,1,1,1])
            b1.metric("Budget fixé", _fmt_eur(budget_fixe))
            b2.metric("Budget restant projeté", _fmt_eur(budget_restant))
            b3.metric("Budget total estimé", _fmt_eur(budget_total))
            b4.metric("Coût unitaire estimé", f"{unit_cost:.2f} €/MWh" if unit_cost is not None else "—")

            if avg_pond is not None:
                st.caption(
                    f"• Fixé = {fixed_mwh:.0f} MWh × {avg_pond:.2f} €/MWh  |  "
                    f"• Restant = {rest_mwh:.0f} MWh × {cal_price:.2f} €/MWh"
                )
            else:
                st.caption(f"• Restant = {rest_mwh:.0f} MWh × {cal_price:.2f} €/MWh")

        # --- (C) Ajouter un clic
        with st.container(border=True):
            st.markdown("#### Ajouter un clic")
            col1, col2, col3, col4 = st.columns([1, 1, 1, 0.8])
            with col1:
                new_date = st.date_input("Date", value=date.today(), key=date_key)
            with col2:
                new_price = st.number_input("Prix (€/MWh)", min_value=0.0, step=1.0, format="%.0f", key=price_key)
            with col3:
                new_vol = st.number_input("Volume (MWh)", min_value=0.0, step=5.0, format="%.0f", key=vol_key)
            with col4:
                st.markdown("&nbsp;")
                add_click = st.button("➕ Ajouter", key=add_btn, use_container_width=True)

            used_clicks = len(clicks)
            st.caption(f"Clics utilisés : {used_clicks}/{max_clicks} (modifiable dans la barre latérale).")

            if add_click:
                if used_clicks >= int(max_clicks):
                    st.error(f"Limite atteinte ({int(max_clicks)} clics).")
                elif new_vol <= 0 or new_price <= 0:
                    st.warning("Prix et volume doivent être > 0.")
                else:
                    st.session_state[clicks_key].append(
                        {"date": new_date, "price": float(new_price), "volume": float(new_vol)}
                    )
                    st.success("Clic ajouté.")
                    for k in (price_key, vol_key):
                        st.session_state.pop(k, None)
                    st.rerun()

        # --- (D) Historique (expander)
        with st.expander("Clics enregistrés", expanded=not df_clicks.empty):
            if df_clicks.empty:
                st.caption("Aucun clic pour l’instant.")
            else:
                df_disp = df_clicks.copy()
                df_disp["date"] = pd.to_datetime(df_disp["date"]).dt.date
                df_disp["% du total"] = df_disp["volume"].apply(
                    lambda v: round((v / total_mwh * 100.0), 2) if total_mwh > 0 else 0.0
                )
                df_disp = df_disp.rename(columns={
                    "date": "Date", "price": "Prix (€/MWh)", "volume": "Volume (MWh)",
                })[["Date", "Prix (€/MWh)", "Volume (MWh)", "% du total"]]
                df_disp.index = range(1, len(df_disp) + 1)
                df_disp.index.name = "Clic #"

                st.dataframe(df_disp, use_container_width=True)

                del_idx = st.selectbox(
                    "Supprimer un clic",
                    options=df_disp.index.tolist(),
                    format_func=lambda i: (
                        f"{i} — {df_disp.loc[i, 'Date']} | "
                        f"{df_disp.loc[i, 'Volume (MWh)']} MWh @ "
                        f"{df_disp.loc[i, 'Prix (€/MWh)']} €/MWh"
                    ),
                    key=del_select,
                )
                cdel, cdl = st.columns([1,1])
                with cdel:
                    if st.button("🗑️ Supprimer la ligne sélectionnée", key=del_btn, use_container_width=True):
                        st.session_state[clicks_key].pop(del_idx - 1)
                        st.rerun()
                with cdl:
                    csv_bytes = df_disp.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Télécharger l’historique (CSV)",
                        data=csv_bytes,
                        file_name=f"clics_blocages_{ns}.csv",
                        mime="text/csv",
                        key=dl_btn,
                        use_container_width=True
                    )

# ======= Rendu avec des onglets =======
tab2026, tab2027, tab2028 = st.tabs(["Contrat 2026", "Contrat 2027", "Contrat 2028"])

with tab2026:
    render_contract_module("Couverture du contrat 2026", ns="y2026")

with tab2027:
    render_contract_module("Couverture du contrat 2027", ns="y2027")

with tab2028:
    render_contract_module("Couverture du contrat 2028", ns="y2028")
# ===================== FIN CONTRATS MULTI-MODULES =====================

# ===================== FIN CONTRATS MULTI-MODULES =====================
