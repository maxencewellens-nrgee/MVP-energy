# app.py — MVP Énergie (BE Day-Ahead + Contrat + FlexyPower CAL) — version épurée UX/UI
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
import urllib.parse  # ⚠️ corrige l’espace insécable après 'parse'
# ----------------------------- Configuration
st.set_page_config(page_title="MVP Énergie — BE Day-Ahead", layout="wide")
st.title("Gérer mes contrats d'énergie")

    
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

# ----------------------------- Helpers format (unifiés)
def eur(amount: float, dec: int = 0) -> str:
    s = f"{amount:,.{dec}f}".replace(",", " ")
    return f"{s} €"

def price_eur_mwh(p: float) -> str:
    return f"{p:,.2f} €/MWh".replace(",", " ")

def mwh(v: float, dec: int = 0) -> str:
    return f"{v:,.{dec}f} MWh".replace(",", " ")

def fmt_be(d) -> str:
    return pd.to_datetime(d).strftime("%d/%m/%Y")

# ----------------------------- Data market
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
    Décision basée sur le dernier prix et quantiles P10/P30/P70 sur fenêtre de lookback.
    """
    if daily.empty:
        return {"reco":"—","raison":"Pas de données.","last":None,"p10":None,"p30":None,"p70":None}

    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    last_price = float(df.iloc[-1]["avg"])
    ref_end = df["date"].max()
    ref_start = ref_end - pd.Timedelta(days=lookback_days)

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

# --- FlexyPower: robuste (normal -> fallback via r.jina.ai -> read_html)
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
        m_elec = re.search(r"Electricite.*?(?=Gaz naturel|<h2|</section>|$)", text, flags=re.I)
        block = m_elec.group(0) if m_elec else text
        dm = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", block)
        if dm: vals["date"] = dm.group(1)
        for yy in ("26","27","28"):
            m = re.search(rf"CAL\s*[-]?\s*{yy}\D*?([0-9]+(?:[.,][0-9]+)?)", block, flags=re.I)
            if m:
                try:
                    vals[f"CAL-{yy}"] = float(m.group(1).replace(",", "."))
                except: 
                    pass
        return vals

    # 1) direct
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0","Accept-Language":"fr-FR,fr;q=0.9,en;q=0.8"}, timeout=20)
        r.raise_for_status()
        vals = _parse_block(_normalize(r.text))
        if any(vals[k] is not None for k in ("CAL-26","CAL-27","CAL-28")):
            if debug: st.write("FlexyPower: direct OK")
            return vals
    except Exception as e:
        if debug: st.write("Flexy direct FAIL:", e)

    # 2) proxy texte
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

    # 3) read_html
    try:
        tables = pd.read_html(url)
        for df in tables:
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
                    if 'vals' not in locals():
                        vals = {"CAL-26": None, "CAL-27": None, "CAL-28": None, "date": None}
                    vals[f"CAL-{yy}"] = outv
        if 'vals' in locals():
            if debug: st.write("FlexyPower: read_html OK")
            return vals
    except Exception as e:
        if debug: st.write("Flexy read_html FAIL:", e)

    return {"CAL-26": None, "CAL-27": None, "CAL-28": None, "date": None}

# ----------------------------- Marché : bornes automatiques (sans UI)
today_be = datetime.now(tz_be).date()
END_INCLUSIVE = str(today_be - timedelta(days=1))   # J-1
START_HISTORY = "2025-01-01"
LOOKBACK_DAYS = 180

start_input = START_HISTORY
end_input   = END_INCLUSIVE
lookback    = LOOKBACK_DAYS

# ----------------------------- Marché : chargement & affichage (AUTO)
def load_market(start_date: str, end_date: str):
    # skeleton loader : KPI placeholders + chart container
    with st.spinner("Récupération ENTSO-E (par mois)…"):
        data = fetch_daily(start_date, end_date)
    return data

# init unique
if "market_daily" not in st.session_state:
    try:
        st.session_state["market_daily"] = load_market(start_input, end_input)
        st.session_state["market_params"] = (start_input, end_input, lookback)
    except Exception as e:
        st.error(f"Erreur : {e}")
        st.stop()

daily = st.session_state.get("market_daily", pd.DataFrame())
if daily.empty:
    st.error("Aucune donnée sur l'intervalle demandé.")
else:
    st.subheader("Market Data & Actions")

# ===================== NAVIGATION PAR ONGLETS (plein écran) =====================

def ensure_cal_used():
    """Stocke CAL_USED & CAL_DATE dans session (source FlexyPower + fallback)."""
    cal_used = st.session_state.get("CAL_USED")
    cal_date = st.session_state.get("CAL_DATE")
    if not cal_used:
        try:
            cal = fetch_flexypower_cals()
        except Exception:
            cal = {"CAL-26": None, "CAL-27": None, "CAL-28": None, "date": None}
        fallback = {"CAL-26": 82.61, "CAL-27": 77.82, "CAL-28": 74.38}
        cal_used = {
            "y2026": float(cal.get("CAL-26") or fallback["CAL-26"]),
            "y2027": float(cal.get("CAL-27") or fallback["CAL-27"]),
            "y2028": float(cal.get("CAL-28") or fallback["CAL-28"]),
        }
        cal_date = cal.get("date") or pd.Timestamp.today().strftime("%d/%m/%Y")
        st.session_state["CAL_USED"] = cal_used
        st.session_state["CAL_DATE"] = cal_date
    return cal_used, cal_date

# ===================== NAVIGATION PERSISTANTE (haut de page) =====================
NAV_ITEMS = ["Graphique & Synthèse", "Contrats 2024–2025", "Simulation & Couverture"]

# Init sélection (persistant)
if "page" not in st.session_state:
    st.session_state["page"] = NAV_ITEMS[0]

# Barre d’onglets horizontale persistante
page = st.radio("Navigation", NAV_ITEMS, key="page", horizontal=True, label_visibility="collapsed")

# ------------------ PAGES ------------------

def render_page_graph():
    st.subheader("Historique prix marché électricité")

    # daily DOIT déjà exister (chargé plus haut dans ton code)
    vis = daily.copy()
    vis["date"] = pd.to_datetime(vis["date"])
    vis = vis.sort_values("date")

    # clé unique pour éviter conflits avec d’autres pages
    mm_window = st.selectbox("Moyenne mobile (jours)", [30, 60, 90], index=0, key="mm_win_graph")

    vis["sma"] = vis["avg"].rolling(window=int(mm_window),
                                    min_periods=max(5, int(mm_window)//3)).mean()
    vis["date_str"] = vis["date"].dt.strftime("%d/%m/%y")
    vis["spot_str"] = vis["avg"].apply(lambda v: f"{v:.2f}".replace(".", ",") + "€")

    hover = alt.selection_point(fields=["date"], nearest=True, on="mousemove", empty="none", clear=False)
    base = alt.Chart(vis).encode(x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %y")))

    spot_line = base.mark_line(strokeWidth=1.5, color="#1f2937").encode(y=alt.Y("avg:Q", title="€/MWh"), tooltip=[])
    sma_line  = base.transform_filter("datum.sma != null").mark_line(strokeWidth=3, color="#22c55e").encode(y="sma:Q")
    points    = base.mark_point(opacity=0).encode(y="avg:Q").add_params(hover)
    hover_pt  = base.mark_circle(size=60, color="#1f2937").encode(y="avg:Q").transform_filter(hover)
    v_rule    = base.mark_rule(color="#9ca3af").encode(
        tooltip=[alt.Tooltip("date_str:N", title="Date"), alt.Tooltip("spot_str:N", title="Spot")]
    ).transform_filter(hover)
    lbl_ph    = base.mark_text(dx=14, dy=-16, align="left", fontSize=12, fontWeight="bold",
                               stroke="white", strokeWidth=5, opacity=1).encode(y="avg:Q", text="spot_str:N").transform_filter(hover)
    lbl_p     = base.mark_text(dx=14, dy=-16, align="left", fontSize=12, fontWeight="bold",
                               color="#111827", opacity=1).encode(y="avg:Q", text="spot_str:N").transform_filter(hover)
    lbl_dh    = base.mark_text(dx=14, dy=4, align="left",  fontSize=11,
                               stroke="white", strokeWidth=5, opacity=1).encode(y="avg:Q", text="date_str:N").transform_filter(hover)
    lbl_d     = base.mark_text(dx=14, dy=4, align="left",  fontSize=11,
                               color="#374151", opacity=1).encode(y="avg:Q", text="date_str:N").transform_filter(hover)

    legend_sel  = alt.selection_point(fields=['serie'], bind='legend')
    layer_lines = alt.layer(
        spot_line.transform_calculate(serie='"Spot"'),
        sma_line.transform_calculate(serie='"Moyenne mobile"')
    ).add_params(legend_sel).transform_filter(legend_sel)

    chart = alt.layer(layer_lines, points, v_rule, hover_pt, lbl_ph, lbl_p, lbl_dh, lbl_d).properties(
        height=420, width="container"
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

    last_visible_date = vis["date"].max()
    st.caption(f"Dernière donnée spot : {fmt_be(last_visible_date)} • Fuseau : Europe/Brussels")

    # ---------- Synthèse Prix Spot et Forward ----------
    st.subheader("Synthèse Prix Spot et Forward")
    overall_avg = round(daily["avg"].mean(), 2)
    last = daily.iloc[-1]
    last_day_dt = pd.to_datetime(daily["date"].max())
    mask_month = (
        (pd.to_datetime(daily["date"]).dt.year  == last_day_dt.year) &
        (pd.to_datetime(daily["date"]).dt.month == last_day_dt.month)
    )
    month_avg = round(daily.loc[mask_month, "avg"].mean(), 2)
    k1, k2, k3 = st.columns(3)
    k1.metric("Moyenne depuis le début visible", f"{overall_avg:.2f} €/MWh")
    k2.metric("Moyenne mois en cours",           f"{month_avg:.2f} €/MWh")
    k3.metric("Dernier prix accessible",         f"{last['avg']:.2f} €/MWh")

    # CAL du jour (via ta fonction déjà définie)
    cal_used, cal_date = ensure_cal_used()
    f1, f2, f3 = st.columns(3)
    f1.metric(f"CAL-26 (élec) – {cal_date}", f"{cal_used['y2026']:.2f} €/MWh")
    f2.metric(f"CAL-27 (élec) – {cal_date}", f"{cal_used['y2027']:.2f} €/MWh")
    f3.metric(f"CAL-28 (élec) – {cal_date}", f"{cal_used['y2028']:.2f} €/MWh")


def render_page_recap_contracts():
    st.subheader("Contrats passés — récapitulatif 2024 / 2025")

    def _get(ns):
        vol   = float(st.session_state.get(f"{ns}__fixed_volume", 0.0))
        prix  = float(st.session_state.get(f"{ns}__fixed_price", 0.0))
        budget = vol * prix
        return vol, prix, budget

    # 2024
    vol24, px24, bud24 = _get("y2024")
    with st.container(border=True):
        st.markdown("**Récap contrat 2024**")
        if vol24 <= 0 or px24 <= 0:
            st.info("Aucun contrat saisi pour 2024 — renseignez Volume & Prix dans la page Simulation & Couverture.")
        c1, c2, c3 = st.columns([1,1,1])
        with c1: st.metric("Volume", f"{vol24:,.0f} MWh".replace(",", " "))
        with c2: st.metric("Prix",   f"{px24:,.2f} €/MWh".replace(",", " ") if px24>0 else "—")
        with c3: st.metric("Budget total", eur(bud24))
        st.caption(f"Calcul : {vol24:.0f} MWh × {px24:.2f} €/MWh = {eur(bud24)}")

    # 2025
    vol25, px25, bud25 = _get("y2025")
    with st.container(border=True):
        st.markdown("**Récap contrat 2025**")
        if vol25 <= 0 or px25 <= 0:
            st.info("Aucun contrat saisi pour 2025 — renseignez Volume & Prix dans la page Simulation & Couverture.")
        c1, c2, c3 = st.columns([1,1,1])
        with c1: st.metric("Volume", f"{vol25:,.0f} MWh".replace(",", " "))
        with c2: st.metric("Prix",   f"{px25:,.2f} €/MWh".replace(",", " ") if px25>0 else "—")
        with c3: st.metric("Budget total", eur(bud25))
        st.caption(f"Calcul : {vol25:.0f} MWh × {px25:.2f} €/MWh = {eur(bud25)}")


def render_page_simulation():
    st.subheader("Simulation & Couverture")

    # Paramètres (saisie) — pour éviter la sidebar, on met les inputs ici avec clés uniques
    st.markdown("##### Paramètres contrats")
    g2024, g2025 = st.columns(2)
    with g2024:
        st.markdown("**Contrat 2024 (saisie simple)**")
        st.number_input("Volume 2024 (MWh)", min_value=0.0, step=5.0, format="%.0f", key="y2024__fixed_volume")
        st.number_input("Prix 2024 (€/MWh)", min_value=0.0, step=1.0, format="%.0f", key="y2024__fixed_price")
    with g2025:
        st.markdown("**Contrat 2025 (saisie simple)**")
        st.number_input("Volume 2025 (MWh)", min_value=0.0, step=5.0, format="%.0f", key="y2025__fixed_volume")
        st.number_input("Prix 2025 (€/MWh)", min_value=0.0, step=1.0, format="%.0f", key="y2025__fixed_price")

    st.divider()
    st.markdown("##### Contrats futurs (avec fixations)")

    # Réglages des 3 années (avec clés stables/uniques)
    cols = st.columns(3)
    for (ns, label), col in zip([("y2026","2026"),("y2027","2027"),("y2028","2028")], cols):
        with col:
            st.markdown(f"**{label} — réglages**")
            st.session_state.setdefault(f"{ns}__total_mwh", 200.0)
            st.session_state.setdefault(f"{ns}__max_clicks", 5)
            st.session_state.setdefault(f"{ns}__clicks", [])
            st.number_input("Volume total (MWh)", min_value=0.0, step=5.0, format="%.0f", key=f"{ns}__total_mwh")
            st.number_input("Fixations max", min_value=1, max_value=20, step=1, format="%d", key=f"{ns}__max_clicks")

    st.divider()
    st.markdown("##### Fixations & historique")

    # Onglets par année (réutilise ta fonction render_year / render_contract_module existante)
    t1, t2, t3 = st.tabs(["2026", "2027", "2028"])
    with t1:
        render_year("y2026", "2026")  # ta fonction de simulation MWh au CAL du jour
        st.markdown("---")
        render_contract_module("Couverture du contrat 2026", ns="y2026")  # ta fonction de gestion des fixations
    with t2:
        render_year("y2027", "2027")
        st.markdown("---")
        render_contract_module("Couverture du contrat 2027", ns="y2027")
    with t3:
        render_year("y2028", "2028")
        st.markdown("---")
        render_contract_module("Couverture du contrat 2028", ns="y2028")


# ------------------ ROUTER ------------------
if page == "Graphique & Synthèse":
    render_page_graph()
elif page == "Contrats 2024–2025":
    render_page_recap_contracts()
else:
    render_page_simulation()
# ===================== FIN NAVIGATION PERSISTANTE =====================
