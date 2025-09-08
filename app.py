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

# ---------- Page 1 : Marché (graphique + synthèse)
def render_page_market(daily: pd.DataFrame):
    st.subheader("Historique prix marché électricité")

    vis = daily.copy()
    vis["date"] = pd.to_datetime(vis["date"]).sort_values()
    mm_window = st.selectbox("Moyenne mobile (jours)", [30, 60, 90], index=0, key="mm_win")
    vis["sma"] = vis["avg"].rolling(window=int(mm_window), min_periods=max(5, int(mm_window)//3)).mean()

    vis["date_str"] = vis["date"].dt.strftime("%d/%m/%y")
    vis["spot_str"] = vis["avg"].apply(lambda v: f"{v:.2f}".replace(".", ",") + "€")

    hover = alt.selection_point(fields=["date"], nearest=True, on="mousemove", empty="none", clear=False)
    base = alt.Chart(vis).encode(x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %y")))

    spot_line = base.mark_line(strokeWidth=1.5, color="#1f2937").encode(y=alt.Y("avg:Q", title="€/MWh"), tooltip=[])
    sma_line  = base.transform_filter("datum.sma != null").mark_line(strokeWidth=3, color="#22c55e").encode(y="sma:Q", tooltip=[])
    points    = base.mark_point(opacity=0).encode(y="avg:Q").add_params(hover)
    hover_pt  = base.mark_circle(size=60, color="#1f2937").encode(y="avg:Q").transform_filter(hover)
    v_rule    = base.mark_rule(color="#9ca3af").encode(
        tooltip=[alt.Tooltip("date_str:N", title="Date"), alt.Tooltip("spot_str:N", title="Spot")]
    ).transform_filter(hover)

    label_price_halo = base.mark_text(dx=14, dy=-18, align="left", fontSize=12, fontWeight="bold",
                                      stroke="white", strokeWidth=5, opacity=1).encode(y="avg:Q", text="spot_str:N").transform_filter(hover)
    label_price = base.mark_text(dx=14, dy=-18, align="left", fontSize=12, fontWeight="bold",
                                 color="#111827", opacity=1).encode(y="avg:Q", text="spot_str:N").transform_filter(hover)
    label_date_halo = base.mark_text(dx=14, dy=6, align="left", fontSize=11,
                                     stroke="white", strokeWidth=5, opacity=1).encode(y="avg:Q", text="date_str:N").transform_filter(hover)
    label_date = base.mark_text(dx=14, dy=6, align="left", fontSize=11,
                                color="#374151", opacity=1).encode(y="avg:Q", text="date_str:N").transform_filter(hover)

    legend_sel = alt.selection_point(fields=['serie'], bind='legend')
    layer_lines = alt.layer(
        spot_line.transform_calculate(serie='"Spot"'),
        sma_line.transform_calculate(serie='"Moyenne mobile"'),
    ).add_params(legend_sel).transform_filter(legend_sel)

    chart = alt.layer(layer_lines, points, v_rule, hover_pt, label_price_halo, label_price, label_date_halo, label_date)\
               .properties(height=420, width="container").interactive()
    st.altair_chart(chart, use_container_width=True)

    last_visible_date = vis["date"].max()
    st.caption(f"Dernière donnée spot : {fmt_be(last_visible_date)} • Fuseau : Europe/Brussels")

    # Synthèse
    st.subheader("Synthèse Prix Spot et Forward")
    overall_avg = round(daily["avg"].mean(), 2)
    last = daily.iloc[-1]
    last_day_dt = pd.to_datetime(daily["date"].max())
    mask_month = (
        (pd.to_datetime(daily["date"]).dt.year == last_day_dt.year) &
        (pd.to_datetime(daily["date"]).dt.month == last_day_dt.month)
    )
    month_avg = round(daily.loc[mask_month, "avg"].mean(), 2)
    k1, k2, k3 = st.columns(3)
    k1.metric("Moyenne depuis le début visible", price_eur_mwh(overall_avg))
    k2.metric("Moyenne mois en cours",           price_eur_mwh(month_avg))
    k3.metric("Dernier prix accessible",         price_eur_mwh(last['avg']))

    CAL_USED, CAL_DATE = ensure_cal_used()
    f1, f2, f3 = st.columns(3)
    f1.metric(f"CAL-26 (élec) – {CAL_DATE}", price_eur_mwh(CAL_USED['y2026']))
    f2.metric(f"CAL-27 (élec) – {CAL_DATE}", price_eur_mwh(CAL_USED['y2027']))
    f3.metric(f"CAL-28 (élec) – {CAL_DATE}", price_eur_mwh(CAL_USED['y2028']))


# ---------- Page 2 : Contrats passés (2024/2025) — saisie + récap
def render_page_past():
    st.subheader("Contrats passés — 2024 & 2025")

    def edit(ns: str, label: str):
        vol_key, price_key, budg_key = f"{ns}__fixed_volume", f"{ns}__fixed_price", f"{ns}__fixed_budget"
        if vol_key not in st.session_state:   st.session_state[vol_key] = 0.0
        if price_key not in st.session_state: st.session_state[price_key] = 0.0

        with st.container(border=True):
            st.markdown(f"**Contrat {label} — saisie**")
            c1, c2, c3 = st.columns([1,1,1])
            with c1: st.number_input("Volume (MWh)", min_value=0.0, step=5.0, format="%.0f", key=vol_key)
            with c2: st.number_input("Prix (€/MWh)", min_value=0.0, step=1.0, format="%.0f", key=price_key)

            vol   = float(st.session_state[vol_key])
            price = float(st.session_state[price_key])
            budget = vol * price
            st.session_state[budg_key] = budget
            with c3: st.metric("Budget total", eur(budget))
            st.caption(f"Calcul : {mwh(vol,0)} × {price_eur_mwh(price) if price>0 else '—'} = {eur(budget)}")

    # Saisie + récap compact
    edit("y2024", "2024")
    edit("y2025", "2025")
    st.markdown("### Récapitulatif")
    for ns, label in [("y2024", "2024"), ("y2025", "2025")]:
        vol   = float(st.session_state.get(f"{ns}__fixed_volume", 0.0))
        price = float(st.session_state.get(f"{ns}__fixed_price", 0.0))
        budg  = vol * price
        with st.container(border=True):
            st.markdown(f"**Récap contrat {label}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Volume", mwh(vol, 0))
            c2.metric("Prix", price_eur_mwh(price) if price > 0 else "—")
            c3.metric("Budget total", eur(budg))


# ---------- Utilitaires simulation/contrats (sans sidebar)
def _year_state(ns: str):
    """total (MWh), fixed_mwh, avg_fixed (€/MWh), rest_mwh, cal_now (€/MWh) — lit session."""
    CAL_USED = st.session_state.get("CAL_USED", {"y2026":82.61,"y2027":77.82,"y2028":74.38})
    total = float(st.session_state.get(f"{ns}__total_mwh", 0.0) or 0.0)
    clicks = pd.DataFrame(st.session_state.get(f"{ns}__clicks", []))
    if not clicks.empty:
        clicks["volume"] = pd.to_numeric(clicks["volume"], errors="coerce").fillna(0.0)
        clicks["price"]  = pd.to_numeric(clicks["price"],  errors="coerce").fillna(0.0)
        fixed_mwh = float(clicks["volume"].sum())
        avg_fixed = float((clicks["price"] * clicks["volume"]).sum() / fixed_mwh) if fixed_mwh > 0 else None
    else:
        fixed_mwh, avg_fixed = 0.0, None
    fixed_mwh = min(fixed_mwh, total) if total > 0 else 0.0
    rest_mwh  = max(0.0, total - fixed_mwh)
    cal_now   = float(CAL_USED.get(ns) or 0.0)
    return total, fixed_mwh, avg_fixed, rest_mwh, cal_now

def render_year(ns: str, title: str):
    CAL_USED, CAL_DATE = ensure_cal_used()
    total, fixed_mwh, avg_fixed, rest_mwh, cal_now = _year_state(ns)

    with st.container(border=True):
        st.markdown(f"### {title} — restant **{rest_mwh:.0f} MWh** · CAL du jour **{cal_now:.2f} €/MWh** (source {CAL_DATE})")

        # Slider MWh
        if rest_mwh <= 0:
            st.info("Plus aucun MWh à fixer pour cette année.")
            extra = 0.0
        else:
            if rest_mwh >= 20:
                step, def_val = 1.0, max(0.0, min(rest_mwh, round(rest_mwh * 0.25)))
            elif rest_mwh >= 1:
                step, def_val = 0.5, max(0.0, min(rest_mwh, round(rest_mwh * 0.25, 1)))
            else:
                step, def_val = (round(rest_mwh / 5, 3) or 0.001), round(rest_mwh / 2, 3)
            def_val = max(0.0, min(float(rest_mwh), float(def_val)))
            extra = st.slider(f"MWh à fixer aujourd’hui ({title})",
                              min_value=0.0, max_value=float(rest_mwh),
                              step=float(step), value=float(def_val),
                              key=f"{ns}__mw_click",
                              help="Choisissez directement la quantité en MWh à fixer aujourd’hui.")

        # AVANT/APRÈS
        budget_before = (avg_fixed or 0.0) * fixed_mwh + cal_now * rest_mwh
        new_fixed_mwh   = fixed_mwh + extra
        new_fixed_cost  = (avg_fixed or 0.0) * fixed_mwh + cal_now * extra
        remaining_after = max(0.0, total - new_fixed_mwh)
        budget_after    = new_fixed_cost + cal_now * remaining_after
        fixed_avg_after = ((avg_fixed or 0.0) * fixed_mwh + cal_now * extra) / new_fixed_mwh if new_fixed_mwh > 0 else None

        # KPIs (deltas : vert si prix baisse)
        c1, c2, c3 = st.columns(3)
        with c1:
            delta_price = (fixed_avg_after - avg_fixed) if (fixed_avg_after is not None and avg_fixed is not None) else None
            st.metric("Prix d'achat moyen (après fixation)",
                      f"{fixed_avg_after:.2f} €/MWh" if fixed_avg_after is not None else ("—" if avg_fixed is None else f"{avg_fixed:.2f} €/MWh"),
                      delta=(f"{delta_price:+.2f} €/MWh" if delta_price is not None else None),
                      delta_color="inverse")  # vert si baisse
        with c2:
            cover_after = (new_fixed_mwh/total*100.0) if total>0 else 0.0
            st.metric("Couverture (après fixation)", f"{cover_after:.1f} %",
                      delta=(f"{(extra/total*100.0):+.1f} %" if total>0 else None))
        with c3:
            st.metric("Budget total estimé (après fixation)", eur(budget_after))

        # Barre MWh
        seg = pd.DataFrame({"segment":["Fixé existant","Nouvelle fixation","Restant après"],
                            "mwh":[fixed_mwh, extra, remaining_after]})
        bar = alt.Chart(seg).mark_bar(height=20).encode(
            x=alt.X("sum(mwh):Q", stack="zero", title=f"Répartition {title} (MWh) — Total {total:.0f}"),
            color=alt.Color("segment:N", scale=alt.Scale(
                domain=["Fixé existant","Nouvelle fixation","Restant après"],
                range=["#22c55e","#3b82f6","#9ca3af"])),
            tooltip=[alt.Tooltip("segment:N"), alt.Tooltip("mwh:Q", format=".0f", title="MWh")]
        ).properties(width="container")
        st.altair_chart(bar, use_container_width=True)

        st.caption("Le budget projeté valorise déjà le **restant** au **CAL du jour** ; fixer aujourd’hui "
                   "**déplace** du ‘projeté’ vers du ‘fixé’. Impact principal : **prix moyen du fixé** et **couverture**.")


def render_contract_module(title: str, ns: str):
    # On ne crée plus les widgets "Volume total" / "Fixations max" ici,
    # on ne fait que LIRE les valeurs définies dans la page "Simulation & Couverture".
    CAL_USED, CAL_DATE = ensure_cal_used()

    with st.container(border=True):
        st.subheader(title)

        # --- Clés / état
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

        # --- LECTURE (pas de widgets de réglage ici)
        total_mwh  = float(st.session_state.get(total_key, 0.0))
        max_clicks = int(st.session_state.get(max_key, 5))
        clicks     = st.session_state.get(clicks_key, [])
        df_clicks  = pd.DataFrame(clicks)

        if not df_clicks.empty:
            df_clicks["volume"] = pd.to_numeric(df_clicks["volume"], errors="coerce").fillna(0.0)
            df_clicks["price"]  = pd.to_numeric(df_clicks["price"],  errors="coerce").fillna(0.0)

        fixed_mwh = float(df_clicks["volume"].sum()) if not df_clicks.empty else 0.0
        fixed_mwh = min(fixed_mwh, total_mwh) if total_mwh > 0 else 0.0
        rest_mwh  = max(0.0, total_mwh - fixed_mwh)
        cov_pct   = round((fixed_mwh / total_mwh * 100.0), 2) if total_mwh > 0 else 0.0

        # Prix moyen pondéré du fixé
        total_cost_fixed = float((df_clicks["price"] * df_clicks["volume"]).sum()) if fixed_mwh > 0 else 0.0
        avg_fixed_mwh    = float(total_cost_fixed / fixed_mwh) if fixed_mwh > 0 else None

        cal_price  = st.session_state["CAL_USED"].get(ns)

        # --- Synthèse couverture
        c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1.2])
        c1.metric("Volume total", f"{total_mwh:.0f} MWh")
        c2.metric("Déjà fixé",    f"{fixed_mwh:.0f} MWh")
        c3.metric("Restant",      f"{rest_mwh:.0f} MWh")
        c4.metric("Couverture",   f"{cov_pct:.1f} %")
        c5.metric(f"CAL utilisé ({CAL_DATE})", f"{cal_price:.2f} €/MWh" if cal_price is not None else "—")
        st.progress(min(cov_pct/100.0, 1.0), text=f"Couverture {cov_pct:.1f}%")

        # --- Budget (fixé uniquement)
        with st.container(border=True):
            st.markdown("#### Budget (déjà fixé)")
            b1, b2, b3 = st.columns([1,1,1])
            b1.metric("Volume fixé", f"{fixed_mwh:.0f} MWh")
            b2.metric("Prix moyen fixé", f"{avg_fixed_mwh:.2f} €/MWh" if avg_fixed_mwh is not None else "—")
            b3.metric("Budget fixé", f"{total_cost_fixed:,.0f} €".replace(",", " "))
            if avg_fixed_mwh is not None:
                st.caption(
                    f"Calcul : Σ(Volume × Prix) / Volume fixé = {avg_fixed_mwh:.2f} €/MWh "
                    f"(Σ = {total_cost_fixed:,.0f} €, Volume = {fixed_mwh:.0f} MWh)".replace(",", " ")
                )
            else:
                st.caption("Aucune fixation enregistrée pour l’instant (prix moyen du fixé indisponible).")

        # --- Ajouter une fixation (ces widgets sont uniques par ns → pas de collision)
        with st.container(border=True):
            st.markdown("#### Ajouter une fixation")
            col1, col2, col3, col4 = st.columns([1, 1, 1, 0.8])
            with col1:
                new_date = st.date_input("Date", value=date.today(), key=date_key)
            with col2:
                new_price = st.number_input("Prix (€/MWh)", min_value=0.0, step=1.0, format="%.0f", key=price_key)
            with col3:
                new_vol = st.number_input("Volume (MWh)",  min_value=0.0, step=5.0, format="%.0f", key=vol_key)
            with col4:
                st.markdown("&nbsp;")
                used = len(clicks)
                can_add = (used < int(max_clicks)) and (rest_mwh > 0) and (new_vol > 0) and (new_price > 0)
                add_click = st.button("➕ Ajouter", key=add_btn, use_container_width=True, disabled=not can_add)

            st.caption(f"Fixations utilisées : {used}/{max_clicks} (les paramètres se règlent dans l’onglet “Simulation & Couverture”).")

            if add_click:
                if used >= int(max_clicks):
                    st.error(f"Limite atteinte ({int(max_clicks)} fixations).")
                elif new_vol <= 0 or new_price <= 0:
                    st.warning("Prix et volume doivent être > 0.")
                else:
                    st.session_state[clicks_key].append(
                        {"date": new_date, "price": float(new_price), "volume": float(new_vol)}
                    )
                    st.success("Fixation ajoutée.")
                    for k in (price_key, vol_key):
                        st.session_state.pop(k, None)
                    st.rerun()

        # --- Historique
        with st.expander("Fixations enregistrées", expanded=not df_clicks.empty):
            if df_clicks.empty:
                st.caption("Aucune fixation pour l’instant.")
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
                df_disp.index.name = "Fixation #"

                st.dataframe(df_disp, use_container_width=True)

                del_idx = st.selectbox(
                    "Supprimer une fixation",
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
                        file_name=f"fixations_{ns}.csv",
                        mime="text/csv",
                        key=dl_btn,
                        use_container_width=True
                    )

# ---------- Page 3 : Simulation & Couverture (2026–2028)
def render_page_simulation():
    ensure_cal_used()  # garantit CAL_USED/CAL_DATE
    st.subheader("Réglages des contrats 2026–2028")
    for ns, y in [("y2026","2026"),("y2027","2027"),("y2028","2028")]:
        total_key, max_key = f"{ns}__total_mwh", f"{ns}__max_clicks"
        if total_key not in st.session_state: st.session_state[total_key] = 200.0
        if max_key not in st.session_state:   st.session_state[max_key]   = 5
        with st.expander(f"Contrat {y} — paramètres", expanded=(ns=="y2026")):
            c1, c2 = st.columns([1,1])
            with c1: st.number_input("Volume total (MWh)", min_value=0.0, step=5.0, format="%.0f", key=total_key)
            with c2: st.number_input("Fixations max autorisées", min_value=1, max_value=20, step=1, format="%d", key=max_key)

    st.subheader("Simuler une fixation aujourd’hui (en MWh, au CAL du jour)")
    sub2026, sub2027, sub2028 = st.tabs(["2026", "2027", "2028"])
    with sub2026: render_year("y2026", "2026")
    with sub2027: render_year("y2027", "2027")
    with sub2028: render_year("y2028", "2028")

    st.divider()
    st.subheader("Couverture du contrat (gestion des fixations)")
    g2026, g2027, g2028 = st.tabs(["Contrat 2026", "Contrat 2027", "Contrat 2028"])
    with g2026: render_contract_module("Couverture du contrat 2026", ns="y2026")
    with g2027: render_contract_module("Couverture du contrat 2027", ns="y2027")
    with g2028: render_contract_module("Couverture du contrat 2028", ns="y2028")

# ===================== Onglets TOP-LEVEL =====================
tab_market, tab_past, tab_sim = st.tabs(["📈 Marché", "📒 Contrats passés", "🧮 Simulation & Couverture"])

with tab_market:
    render_page_market(daily)

with tab_past:
    render_page_past()

with tab_sim:
    render_page_simulation()
