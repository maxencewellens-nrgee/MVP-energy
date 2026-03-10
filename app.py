import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, date, timedelta
import pytz

from utils.database import get_supabase_client, DatabaseManager
from utils.formatters import eur, price_eur_mwh, mwh, fmt_be
from utils.market_data import fetch_daily_prices, calculate_decision, fetch_flexypower_cals
from utils.network_costs import get_network_costs, get_dsos_for_year, get_segments_for_dso

st.set_page_config(page_title="MVP Énergie — BE Day-Ahead", layout="wide")

if "_base_css_done" not in st.session_state:
    st.markdown("""
    <style>
      .stApp header {position: sticky; top: 0; z-index: 50; background: white;}
      .block-container {padding-top: 1.2rem;}
      div[role="radiogroup"] > label {padding: 6px 10px; border-radius: 999px; margin-right: 6px; border:1px solid #e5e7eb;}
      div[role="radiogroup"] > label[data-checked="true"] {background:#eef2ff; border-color:#c7d2fe;}
      .eq-card {padding:14px 16px;border:1px solid #e5e7eb;border-radius:10px;background:#f9fafb;}
      .eq-sum  {padding:14px 16px;border:1px solid #d1d5db;border-radius:10px;background:#f3f4f6;}
      .muted   {color:#6b7280;}
      .big     {font-size:26px;font-weight:800;}
      .mid     {font-size:18px;font-weight:700;}
      .center  {text-align:center;}
      .op      {font-size:28px; line-height:1; font-weight:700; color:#9ca3af;}
    </style>
    """, unsafe_allow_html=True)
    st.session_state["_base_css_done"] = True

def init_altair_theme():
    """Initialize Altair theme."""
    if "_alt_theme_done" not in st.session_state:
        def _alt_theme():
            return {
                "config": {
                    "view": {"stroke": "transparent"},
                    "axis": {"labelFontSize": 12, "titleFontSize": 12, "grid": True, "gridOpacity": 0.2},
                    "legend": {"labelFontSize": 12, "title": None},
                    "range": {"category": ["#1f2937","#22c55e","#3b82f6","#f59e0b","#ef4444"]},
                    "font": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial"
                }
            }
        alt.themes.register("clean", _alt_theme)
        alt.themes.enable("clean")
        st.session_state["_alt_theme_done"] = True

init_altair_theme()
st.title("Gérer mes contrats d'énergie")

ENTSOE_TOKEN = st.secrets.get("ENTSOE_TOKEN", "")
if not ENTSOE_TOKEN:
    st.error("Secret ENTSOE_TOKEN manquant. Configurez-le dans Settings → Secrets.")
    st.stop()

ZONE = "10YBE----------2"
tz_be = pytz.timezone("Europe/Brussels")

try:
    supabase = get_supabase_client()
    db = DatabaseManager(supabase)
except Exception as e:
    st.error(f"Impossible de se connecter à la base de données: {e}")
    st.stop()

user = supabase.auth.get_user()
if not user or not user.user:
    st.warning("Vous devez être connecté pour utiliser cette application.")

    tab1, tab2 = st.tabs(["Connexion", "Inscription"])

    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Mot de passe", type="password")
            submit = st.form_submit_button("Se connecter")

            if submit:
                try:
                    response = supabase.auth.sign_in_with_password({
                        "email": email,
                        "password": password
                    })
                    st.success("Connexion réussie")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur de connexion: {e}")

    with tab2:
        with st.form("signup_form"):
            new_email = st.text_input("Email")
            new_password = st.text_input("Mot de passe", type="password")
            confirm_password = st.text_input("Confirmer le mot de passe", type="password")
            signup = st.form_submit_button("S'inscrire")

            if signup:
                if new_password != confirm_password:
                    st.error("Les mots de passe ne correspondent pas")
                elif len(new_password) < 6:
                    st.error("Le mot de passe doit contenir au moins 6 caractères")
                else:
                    try:
                        response = supabase.auth.sign_up({
                            "email": new_email,
                            "password": new_password
                        })
                        st.success("Inscription réussie. Vous pouvez maintenant vous connecter.")
                    except Exception as e:
                        st.error(f"Erreur d'inscription: {e}")
    st.stop()

USER_ID = user.user.id

with st.sidebar:
    st.write(f"Connecté: {user.user.email}")
    if st.button("Déconnexion"):
        supabase.auth.sign_out()
        st.rerun()

def ensure_cal_data():
    """Ensure CAL data is available."""
    if "CAL_USED" not in st.session_state:
        try:
            cal = fetch_flexypower_cals()
        except:
            cal = {"CAL-26": None, "CAL-27": None, "CAL-28": None, "date": None}

        fallback = {"CAL-26": 82.61, "CAL-27": 77.82, "CAL-28": 74.38}
        st.session_state["CAL_USED"] = {
            "y2026": float(cal.get("CAL-26") or fallback["CAL-26"]),
            "y2027": float(cal.get("CAL-27") or fallback["CAL-27"]),
            "y2028": float(cal.get("CAL-28") or fallback["CAL-28"]),
        }
        st.session_state["CAL_DATE"] = cal.get("date") or pd.Timestamp.today().strftime("%d/%m/%Y")

    return st.session_state["CAL_USED"], st.session_state["CAL_DATE"]

def load_market_data():
    """Load market data with caching."""
    if "market_daily" not in st.session_state:
        today_be = datetime.now(tz_be).date()
        end_date = str(today_be - timedelta(days=1))
        start_date = "2025-01-01"

        cached_df = db.get_market_data_cache(start_date, end_date)

        if not cached_df.empty:
            st.session_state["market_daily"] = cached_df
        else:
            with st.spinner("Récupération des données ENTSO-E..."):
                df = fetch_daily_prices(ENTSOE_TOKEN, ZONE, start_date, end_date)
                if not df.empty:
                    st.session_state["market_daily"] = df
                    db.cache_market_data(df)
                else:
                    st.session_state["market_daily"] = pd.DataFrame()

    return st.session_state.get("market_daily", pd.DataFrame())

daily = load_market_data()
ensure_cal_data()

def render_page_market(daily: pd.DataFrame):
    """Render market data page."""
    st.subheader("Historique prix marché électricité")

    if daily.empty:
        st.warning("Aucune donnée disponible")
        return

    vis = daily.copy()
    vis["date"] = pd.to_datetime(vis["date"])
    vis = vis.sort_values("date")

    mm_window = st.selectbox("Moyenne mobile (jours)", [30, 60, 90], index=0)
    vis["sma"] = vis["avg"].rolling(window=int(mm_window), min_periods=max(5, int(mm_window)//3)).mean()

    vis["date_str"] = vis["date"].dt.strftime("%d/%m/%y")
    vis["spot_str"] = vis["avg"].apply(lambda v: f"{v:.2f}".replace(".", ",") + "€")

    hover = alt.selection_point(fields=["date"], nearest=True, on="mousemove", empty="none", clear=False)
    base = alt.Chart(vis).encode(x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %y")))

    spot_line = base.mark_line(strokeWidth=1.5, color="#1f2937").encode(y=alt.Y("avg:Q", title="€/MWh"), tooltip=[])
    sma_line = base.transform_filter("datum.sma != null").mark_line(strokeWidth=3, color="#22c55e").encode(y="sma:Q", tooltip=[])
    points = base.mark_point(opacity=0).encode(y="avg:Q").add_params(hover)
    hover_pt = base.mark_circle(size=60, color="#1f2937").encode(y="avg:Q").transform_filter(hover)
    v_rule = base.mark_rule(color="#9ca3af").encode(
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
    k2.metric("Moyenne mois en cours", price_eur_mwh(month_avg))
    k3.metric("Dernier prix accessible", price_eur_mwh(last['avg']))

    CAL_USED, CAL_DATE = ensure_cal_data()
    f1, f2, f3 = st.columns(3)
    f1.metric(f"CAL-26 (élec) – {CAL_DATE}", price_eur_mwh(CAL_USED['y2026']))
    f2.metric(f"CAL-27 (élec) – {CAL_DATE}", price_eur_mwh(CAL_USED['y2027']))
    f3.metric(f"CAL-28 (élec) – {CAL_DATE}", price_eur_mwh(CAL_USED['y2028']))

def render_page_past():
    """Render past contracts page."""
    st.subheader("Contrats passés — 2024 & 2025")

    for year in [2024, 2025]:
        with st.container(border=True):
            st.markdown(f"**Contrat {year}**")

            past = db.get_past_contract(USER_ID, year)
            vol_default = past["fixed_volume"] if past else 0.0
            price_default = past["fixed_price"] if past else 0.0

            with st.form(f"form_past_{year}"):
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    vol = st.number_input("Volume (MWh)", min_value=0.0, step=5.0, format="%.0f",
                                         value=float(vol_default), key=f"vol_{year}")
                with c2:
                    price = st.number_input("Prix (€/MWh)", min_value=0.0, step=1.0, format="%.0f",
                                           value=float(price_default), key=f"price_{year}")
                submitted = st.form_submit_button("Enregistrer")

                if submitted:
                    if db.upsert_past_contract(USER_ID, year, vol, price):
                        st.success("Contrat enregistré")
                        st.rerun()

            budget = vol * price
            c1, c2, c3 = st.columns([1, 1, 1])
            with c3:
                st.metric("Budget total", eur(budget))
            st.caption(f"Calcul : {mwh(vol,0)} × {price_eur_mwh(price) if price>0 else '—'} = {eur(budget)}")

def get_contract_state(year: int):
    """Get contract state from database."""
    contract = db.get_or_create_contract(USER_ID, year, {
        "total_mwh": 200.0,
        "max_fixations": 5,
        "dso": "ORES",
        "segment": "BT"
    })

    if not contract:
        return None, [], 0.0, None, 0.0

    fixations = db.get_fixations(contract["id"])

    total_mwh = float(contract["total_mwh"])

    if fixations:
        fixed_mwh = sum(float(f["volume"]) for f in fixations)
        total_cost = sum(float(f["price"]) * float(f["volume"]) for f in fixations)
        avg_fixed = total_cost / fixed_mwh if fixed_mwh > 0 else None
    else:
        fixed_mwh = 0.0
        avg_fixed = None

    rest_mwh = max(0.0, total_mwh - fixed_mwh)
    CAL_USED, _ = ensure_cal_data()
    cal_now = float(CAL_USED.get(f"y{year}", 0.0))

    return contract, fixations, fixed_mwh, avg_fixed, rest_mwh

def render_page_simulation():
    """Render simulation page."""
    st.subheader("Réglages des contrats 2026–2028")

    for year in [2026, 2027, 2028]:
        contract, fixations, fixed_mwh, avg_fixed, rest_mwh = get_contract_state(year)

        if not contract:
            st.error(f"Impossible de charger le contrat {year}")
            continue

        with st.expander(f"Contrat {year} — paramètres", expanded=(year == 2026)):
            with st.form(f"form_params_{year}"):
                c1, c2 = st.columns(2)
                with c1:
                    total_mwh = st.number_input("Volume total (MWh)", min_value=0.0, step=5.0, format="%.0f",
                                               value=float(contract["total_mwh"]), key=f"total_{year}")
                with c2:
                    max_fix = st.number_input("Fixations max autorisées", min_value=1, max_value=20, step=1,
                                             value=int(contract["max_fixations"]), key=f"max_{year}")

                if st.form_submit_button("Enregistrer"):
                    if db.update_contract(contract["id"], {
                        "total_mwh": float(total_mwh),
                        "max_fixations": int(max_fix)
                    }):
                        st.success("Paramètres mis à jour")
                        st.rerun()

    st.subheader("Couverture des contrats")

    tabs = st.tabs(["Contrat 2026", "Contrat 2027", "Contrat 2028"])

    for idx, year in enumerate([2026, 2027, 2028]):
        with tabs[idx]:
            render_contract_management(year)

def render_contract_management(year: int):
    """Render contract management interface."""
    contract, fixations, fixed_mwh, avg_fixed, rest_mwh = get_contract_state(year)

    if not contract:
        st.error("Impossible de charger le contrat")
        return

    CAL_USED, CAL_DATE = ensure_cal_data()
    cal_now = float(CAL_USED.get(f"y{year}", 0.0))
    total_mwh = float(contract["total_mwh"])
    max_fixations = int(contract["max_fixations"])

    cov_pct = round((fixed_mwh / total_mwh * 100.0), 2) if total_mwh > 0 else 0.0

    with st.container(border=True):
        st.markdown(f"### Contrat {year}")

        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.2])
        c1.metric("Volume total", f"{total_mwh:.0f} MWh")
        c2.metric("Déjà fixé", f"{fixed_mwh:.0f} MWh")
        c3.metric("Restant", f"{rest_mwh:.0f} MWh")
        c4.metric("Couverture", f"{cov_pct:.1f} %")
        c5.metric(f"CAL utilisé ({CAL_DATE})", f"{cal_now:.2f} €/MWh")

        st.progress(min(cov_pct/100.0, 1.0), text=f"Couverture {cov_pct:.1f}%")

        with st.container(border=True):
            st.markdown("#### Ajouter une fixation")

            col1, col2, col3, col4 = st.columns([1, 1, 1, 0.8])
            with col1:
                new_date = st.date_input("Date", value=date.today(), key=f"date_{year}")
            with col2:
                new_price = st.number_input("Prix (€/MWh)", min_value=0.0, step=1.0, format="%.0f", key=f"price_{year}")
            with col3:
                new_vol = st.number_input("Volume (MWh)", min_value=0.0, step=5.0, format="%.0f", key=f"vol_{year}")
            with col4:
                st.markdown("&nbsp;")
                used = len(fixations)
                can_add = (used < max_fixations) and (rest_mwh > 0) and (new_vol > 0) and (new_price > 0)

                if st.button("Ajouter", key=f"add_{year}", use_container_width=True, disabled=not can_add):
                    if new_vol > rest_mwh:
                        st.error(f"Volume trop élevé (max: {rest_mwh:.0f} MWh)")
                    elif db.add_fixation(contract["id"], USER_ID, new_date, new_price, new_vol):
                        st.success("Fixation ajoutée")
                        st.rerun()

            st.caption(f"Fixations utilisées : {used}/{max_fixations}")

        with st.expander("Fixations enregistrées", expanded=bool(fixations)):
            if not fixations:
                st.caption("Aucune fixation")
            else:
                df_disp = pd.DataFrame(fixations)
                df_disp["date"] = pd.to_datetime(df_disp["date"]).dt.date
                df_disp["% du total"] = df_disp["volume"].apply(
                    lambda v: round((v / total_mwh * 100.0), 2) if total_mwh > 0 else 0.0
                )
                df_disp = df_disp.rename(columns={
                    "date": "Date", "price": "Prix (€/MWh)", "volume": "Volume (MWh)",
                })[["Date", "Prix (€/MWh)", "Volume (MWh)", "% du total"]]

                st.dataframe(df_disp, use_container_width=True)

                if fixations:
                    fix_options = {f["id"]: f"{f['date']} | {f['volume']} MWh @ {f['price']} €/MWh"
                                  for f in fixations}
                    selected = st.selectbox("Supprimer une fixation", options=list(fix_options.keys()),
                                           format_func=lambda x: fix_options[x], key=f"del_sel_{year}")

                    if st.button("Supprimer", key=f"del_{year}"):
                        if db.delete_fixation(selected):
                            st.success("Fixation supprimée")
                            st.rerun()

def render_page_total_cost():
    """Render total cost summary page."""
    st.subheader("Coût total (réel) — Résumé")
    st.caption("Énergie = (fixé au prix moyen des clics) + (restant au CAL). Réseau = Transport (Elia) + Distribution (GRD). TVA = 21 % (B2B).")

    year = st.radio("Année", ["2026", "2027", "2028"], horizontal=True, key="tc_year")
    year_int = int(year)

    dsos = get_dsos_for_year(year_int) or ["ORES", "RESA", "AIEG", "AIESH", "REW"]
    dso = st.selectbox("GRD (distributeur)", options=dsos, key="tc_dso")

    seg_opts = get_segments_for_dso(year_int, dso) or ["BT (≤56 kVA)", "MT (>56 kVA)"]
    seg_label = st.selectbox("Tension", options=seg_opts, key="tc_seg")

    contract, fixations, fixed_mwh, avg_fixed, rest_mwh = get_contract_state(year_int)

    if not contract:
        st.error("Impossible de charger le contrat")
        return

    total_mwh = float(contract["total_mwh"])

    if total_mwh <= 0:
        st.warning("Définis le volume total dans Simulation & Couverture")
        return

    CAL_USED, CAL_DATE = ensure_cal_data()
    cal_now = float(CAL_USED.get(f"y{year_int}", 0.0))

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Volume total", mwh(total_mwh, 0))
        c2.metric("Fixé", mwh(fixed_mwh, 0))
        c3.metric("Restant (valorisé CAL)", mwh(rest_mwh, 0))
        c4.metric(f"CAL {year} ({CAL_DATE})", price_eur_mwh(cal_now))

    st.caption(f"Contexte : **{dso}** — **{seg_label}** — Année **{year}**")

    transport, dso_var, dso_fixe_an = get_network_costs(year_int, dso, seg_label)
    dso_fixe_eur_mwh = (dso_fixe_an / total_mwh) if total_mwh > 0 else 0.0

    energy_fixed_eur = (fixed_mwh * (avg_fixed or 0.0))
    energy_rest_eur = (rest_mwh * cal_now)
    energy_budget_eur = energy_fixed_eur + energy_rest_eur

    reseau_eur_mwh = transport + dso_var + dso_fixe_eur_mwh
    reseau_budget_eur = reseau_eur_mwh * total_mwh

    ht_budget_eur = energy_budget_eur + reseau_budget_eur
    tva_budget_eur = ht_budget_eur * 0.21
    ttc_budget_eur = ht_budget_eur + tva_budget_eur

    st.markdown("#### Décomposition budgétaire (€/an)")

    row1 = st.columns([3.5, 0.8, 3.5, 0.8, 3.5])
    with row1[0]:
        st.markdown("<div class='eq-card'><div class='muted'>Énergie — fixé</div>"
                    f"<div class='mid'>{eur(energy_fixed_eur, 0)}</div>"
                    f"<div class='muted'>{mwh(fixed_mwh,0)} × {price_eur_mwh((avg_fixed or 0.0))}</div></div>",
                    unsafe_allow_html=True)
    with row1[1]:
        st.markdown("<div class='center op'>+</div>", unsafe_allow_html=True)
    with row1[2]:
        st.markdown("<div class='eq-card'><div class='muted'>Énergie — restant au CAL</div>"
                    f"<div class='mid'>{eur(energy_rest_eur, 0)}</div>"
                    f"<div class='muted'>{mwh(rest_mwh,0)} × {price_eur_mwh(cal_now)}</div></div>",
                    unsafe_allow_html=True)
    with row1[3]:
        st.markdown("<div class='center op'>=</div>", unsafe_allow_html=True)
    with row1[4]:
        st.markdown("<div class='eq-sum'><div class='muted'>Énergie — sous-total</div>"
                    f"<div class='mid'>{eur(energy_budget_eur, 0)}</div></div>",
                    unsafe_allow_html=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)

    row2 = st.columns([3.5, 0.8, 3.5, 0.8, 3.5])
    with row2[0]:
        st.markdown("<div class='eq-card'><div class='muted'>Réseau — Transport (Elia)</div>"
                    f"<div class='mid'>{eur(transport*total_mwh, 0)}</div>"
                    f"<div class='muted'>{price_eur_mwh(transport)} × {mwh(total_mwh,0)}</div></div>",
                    unsafe_allow_html=True)
    with row2[1]:
        st.markdown("<div class='center op'>+</div>", unsafe_allow_html=True)
    with row2[2]:
        st.markdown("<div class='eq-card'><div class='muted'>Réseau — Distribution variable</div>"
                    f"<div class='mid'>{eur(dso_var*total_mwh, 0)}</div>"
                    f"<div class='muted'>{price_eur_mwh(dso_var)} × {mwh(total_mwh,0)}</div></div>",
                    unsafe_allow_html=True)
    with row2[3]:
        st.markdown("<div class='center op'>+</div>", unsafe_allow_html=True)
    with row2[4]:
        st.markdown("<div class='eq-card'><div class='muted'>Réseau — Fixe (1 site)</div>"
                    f"<div class='mid'>{eur(dso_fixe_an, 0)}</div>"
                    f"<div class='muted'>{price_eur_mwh(dso_fixe_eur_mwh)} × {mwh(total_mwh,0)}</div></div>",
                    unsafe_allow_html=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)

    row3 = st.columns([3.5, 0.8, 3.5, 0.8, 3.5])
    with row3[0]:
        st.markdown("<div class='eq-sum'><div class='muted'>Sous-total HT</div>"
                    f"<div class='mid'>{eur(ht_budget_eur, 0)}</div></div>",
                    unsafe_allow_html=True)
    with row3[1]:
        st.markdown("<div class='center op'>+</div>", unsafe_allow_html=True)
    with row3[2]:
        st.markdown("<div class='eq-card'><div class='muted'>TVA 21 %</div>"
                    f"<div class='mid'>{eur(tva_budget_eur, 0)}</div></div>",
                    unsafe_allow_html=True)
    with row3[3]:
        st.markdown("<div class='center op'>=</div>", unsafe_allow_html=True)
    with row3[4]:
        st.markdown("<div class='eq-sum'><div class='muted'>Total TTC</div>"
                    f"<div class='big'>{eur(ttc_budget_eur, 0)}</div></div>",
                    unsafe_allow_html=True)

    st.markdown("### Tableau récapitulatif")
    df = pd.DataFrame([
        ["Énergie — fixé", (avg_fixed or 0.0), energy_fixed_eur],
        ["Énergie — restant au CAL", cal_now, energy_rest_eur],
        ["Énergie — SOUS-TOTAL", None, energy_budget_eur],
        ["Transport (Elia)", transport, transport * total_mwh],
        ["Distribution — variable", dso_var, dso_var * total_mwh],
        ["Distribution — fixe → €/MWh (1 site)", dso_fixe_eur_mwh, dso_fixe_an],
        ["**SOUS-TOTAL HT**", None, ht_budget_eur],
        ["TVA 21 %", None, tva_budget_eur],
        ["**TOTAL TTC**", None, ttc_budget_eur],
    ], columns=["Poste", "€/MWh", "€ / an"])

    def _row_style(row):
        label = str(row["Poste"])
        if "SOUS-TOTAL" in label and "TOTAL" not in label:
            return ["background-color: #f3f4f6; font-weight: 700;" if c != "€/MWh" else "background-color: #f3f4f6;"
                   for c in df.columns]
        if "TOTAL TTC" in label:
            return ["background-color: #eef2ff; font-weight: 800;" if c != "€/MWh" else "background-color: #eef2ff;"
                   for c in df.columns]
        return [""] * len(df.columns)

    st.dataframe(
        df.style.apply(_row_style, axis=1).format({
            "€/MWh": (lambda v: "" if (v is None or pd.isna(v)) else price_eur_mwh(float(v))),
            "€ / an": (lambda v: eur(float(v), 0)),
        }),
        use_container_width=True
    )

NAV_ITEMS = ["Marché", "Contrats passés", "Simulation & Couverture", "Coût total (réel)"]

q = st.query_params
if "page" not in st.session_state:
    st.session_state["page"] = q.get("nav", NAV_ITEMS[0])

page = st.radio("Navigation", NAV_ITEMS, key="page", horizontal=True, label_visibility="collapsed")
st.query_params["nav"] = page

if page == "Marché":
    render_page_market(daily)
elif page == "Contrats passés":
    render_page_past()
elif page == "Simulation & Couverture":
    render_page_simulation()
else:
    render_page_total_cost()
