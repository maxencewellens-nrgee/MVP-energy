
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

# ===================== Graphique interactif BE spot =====================
st.subheader("Moyenne 30-60-90 jours")

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

# ===================== CONTRATS MULTI-MODULES (REMPLACEMENT ENTIER) =====================

# Dictionnaire des prix CAL (issus de ta synthèse)
cal_prices = {
    "y2026": 84.13,
    "y2027": 79.33,
    "y2028": 74.49,
}

def render_contract_module(title: str, ns: str, default_total: float = 200.0, default_max_clicks: int = 5):
    with st.container(border=True):
        st.subheader(title)

        # ---------- Clés uniques
        total_key   = f"{ns}__total_mwh"
        clicks_key  = f"{ns}__clicks"
        max_key     = f"{ns}__max_clicks"
        init_key    = f"{ns}__initialized"

        date_key    = f"{ns}__new_click_date"
        price_key   = f"{ns}__new_click_price"
        vol_key     = f"{ns}__new_click_volume"
        add_btn     = f"{ns}__btn_add_click"

        del_select  = f"{ns}__delete_click_selector"
        del_btn     = f"{ns}__btn_delete_click"
        dl_btn      = f"{ns}__dl_csv"

        # ---------- INIT
        if init_key not in st.session_state:
            st.session_state[total_key]  = float(default_total)
            st.session_state[clicks_key] = []
            st.session_state[max_key]    = int(default_max_clicks)
            st.session_state[init_key]   = True

        # ---------- 1) Couverture
        total_mwh = st.number_input(
            "Volume total (MWh)",
            min_value=0.0, step=5.0, format="%.0f", key=total_key,
        )

        clicks = st.session_state.get(clicks_key, [])
        _df = pd.DataFrame(clicks)

        fixed_mwh = float(_df["volume"].sum()) if not _df.empty else 0.0
        rest_mwh  = max(0.0, total_mwh - fixed_mwh)
        cov_pct   = round((fixed_mwh / total_mwh * 100.0), 2) if total_mwh > 0 else 0.0

        avg_simple = round(float(_df["price"].mean()), 2) if not _df.empty else None
        avg_pond   = round(((_df["price"] * _df["volume"]).sum() / fixed_mwh), 2) if fixed_mwh > 0 else None

        # --- Calcul budget actuel & projeté
        cal_price = cal_prices.get(ns, None)
        budget_actuel   = fixed_mwh * avg_pond if avg_pond is not None else 0.0
        budget_projete  = rest_mwh * cal_price if cal_price is not None else 0.0
        budget_total    = budget_actuel + budget_projete

        # ---------- Affichage métriques
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Volume total", f"{total_mwh:.0f} MWh")
        c2.metric("Total déjà fixé", f"{fixed_mwh:.0f} MWh")
        c3.metric("Total restant", f"{rest_mwh:.0f} MWh")
        c4.metric("Couverture", f"{cov_pct:.1f} %")
        c5.metric("Prix d’achat moyen", f"{avg_simple:.2f} €/MWh" if avg_simple is not None else "—")

        st.progress(min(cov_pct/100.0, 1.0))

        if avg_pond is not None:
            st.caption(f"(Référence) Prix moyen pondéré : **{avg_pond:.2f} €/MWh**")

        # ---------- Nouveau bloc budget
        st.markdown("### Budget")
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Budget actuel", f"{budget_actuel:,.0f} €")
        bc2.metric("Budget projeté", f"{budget_projete:,.0f} €" if cal_price else "—")
        bc3.metric("Budget total estimé", f"{budget_total:,.0f} €")

        # (le reste de ton code inchangé : nombre de clics autorisés, ajout clics, suppression, CSV etc.)


        # ---------- 2) Entrées / clics
        st.subheader("Entrées / clics")
        col1, col2, col3, col4 = st.columns([1, 1, 1, 0.8])
        with col1:
            new_date = st.date_input("Date du clic", value=date.today(), key=date_key)
        with col2:
            new_price = st.number_input("Prix (€/MWh)", min_value=0.0, step=1.0, format="%.0f", key=price_key)
        with col3:
            new_vol = st.number_input("Volume (MWh)", min_value=0.0, step=5.0, format="%.0f", key=vol_key)
        with col4:
            st.markdown("&nbsp;")
            add_click = st.button("➕ Ajouter ce clic", key=add_btn, use_container_width=True)

        used_clicks = len(clicks)
        if add_click:
            if used_clicks >= int(st.session_state[max_key]):
                st.error(f"Limite atteinte ({int(st.session_state[max_key])} clics).")
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

        # ---------- 3) Clics enregistrés + suppression + CSV
        clicks_df = pd.DataFrame(st.session_state.get(clicks_key, []))
        display_df = pd.DataFrame(columns=["Date", "Prix (€/MWh)", "Volume (MWh)", "% du total"])
        if not clicks_df.empty:
            df = clicks_df.copy()
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df["pct_total"] = df["volume"].apply(
                lambda v: round((v / total_mwh * 100.0), 2) if total_mwh > 0 else 0.0
            )
            display_df = df.rename(columns={
                "date": "Date",
                "price": "Prix (€/MWh)",
                "volume": "Volume (MWh)",
                "pct_total": "% du total",
            })[["Date", "Prix (€/MWh)", "Volume (MWh)", "% du total"]]
            display_df.index = range(1, len(display_df) + 1)
            display_df.index.name = "Clic #"

        st.markdown("### Clics enregistrés")
        st.dataframe(display_df, width="stretch")

        if not display_df.empty:
            del_idx = st.selectbox(
                "Supprimer un clic",
                options=display_df.index.tolist(),
                format_func=lambda i: (
                    f"{i} — {display_df.loc[i, 'Date']} | "
                    f"{display_df.loc[i, 'Volume (MWh)']} MWh @ "
                    f"{display_df.loc[i, 'Prix (€/MWh)']} €/MWh"
                ),
                key=del_select,
            )
            if st.button("🗑️ Supprimer la ligne sélectionnée", key=del_btn):
                st.session_state[clicks_key].pop(del_idx - 1)
                st.rerun()
        else:
            st.caption("Aucun clic à supprimer pour l’instant.")

        if not display_df.empty:
            csv_bytes = display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger l’historique (CSV)",
                data=csv_bytes,
                file_name=f"clics_blocages_{ns}.csv",
                mime="text/csv",
                key=dl_btn,
            )

        # ---------- 4) BUDGETS (les 3 cadrans)
        # Budget actuel = fixed_mwh * avg_pond (si pas de clic → 0)
        current_budget = (fixed_mwh * avg_pond) if avg_pond is not None else 0.0
        # Budget projeté = rest_mwh * cal_price
        projected_budget = rest_mwh * float(cal_price or 0.0)
        # Total estimé
        total_budget = current_budget + projected_budget
        # Coût unitaire estimé
        unit_cost = (total_budget / total_mwh) if total_mwh > 0 else None

        st.markdown("### Budget (actuel / projeté / total)")
        b1, b2, b3, b4 = st.columns([1, 1, 1, 1])
        b1.metric("Budget actuel", f"{current_budget:,.0f} €".replace(",", " "))
        b2.metric("Budget projeté", f"{projected_budget:,.0f} €".replace(",", " "))
        b3.metric("Budget total estimé", f"{total_budget:,.0f} €".replace(",", " "))
        b4.metric("Coût unitaire estimé", f"{unit_cost:.2f} €/MWh" if unit_cost is not None else "—")

        st.caption(
            f"• Actuel = {fixed_mwh:.0f} MWh × {avg_pond:.2f} €/MWh "
            f"| • Projets = {rest_mwh:.0f} MWh × {cal_price:.2f} €/MWh"
            if avg_pond is not None else
            f"• Projets = {rest_mwh:.0f} MWh × {cal_price:.2f} €/MWh"
        )

# ======= Appel des trois modules (2026, 2027, 2028) avec un CAL par défaut =======
render_contract_module("Couverture du contrat 2026", ns="y2026", default_cal_price=st.session_state.get("cal_26_price", 95.00))
render_contract_module("Couverture du contrat 2027", ns="y2027", default_cal_price=st.session_state.get("cal_27_price", 88.50))
render_contract_module("Couverture du contrat 2028", ns="y2028", default_cal_price=st.session_state.get("cal_28_price", 84.20))
# ===================== FIN CONTRATS MULTI-MODULES =====================

