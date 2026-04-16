"""
TariffFloat - Palm Oil Tariff Impact Simulator
AI-powered policy decision support for India's edible oil imports
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import requests
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Page config with palm theme
st.set_page_config(
    page_title="TariffFloat - Palm Oil Tariff Simulator",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for palm oil theme
st.markdown("""
<style>
    /* Main background with subtle palm plantation gradient */
    .main {
        background: linear-gradient(180deg, 
            rgba(245, 250, 245, 1) 0%, 
            rgba(240, 245, 238, 1) 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, 
            #f5faf5 0%, 
            #e8f5e8 50%,
            #f0f5ee 100%);
        background-attachment: fixed;
    }
    
    h1 {
        background: linear-gradient(135deg, #1a5f3a 0%, #2d8659 50%, #3da673 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
        font-size: 3.5rem !important;
        letter-spacing: 3px;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    .caption-text {
        color: #c17a3b;
        font-size: 1.15rem;
        font-weight: 600;
        margin-top: 0.5rem;
        letter-spacing: 1px;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.9rem;
        font-weight: 700;
        color: #1a5f3a;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem;
        font-weight: 600;
        color: #2d5016 !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #ffffff, #fafff7);
        padding: 1.3rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(26, 95, 58, 0.12), 
                    0 2px 4px rgba(0,0,0,0.06);
        border: 2px solid rgba(26, 95, 58, 0.3);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 6px 16px rgba(26, 95, 58, 0.18), 
                    0 3px 6px rgba(0,0,0,0.08);
        transform: translateY(-2px);
    }
    
    .info-box {
        background: linear-gradient(135deg, #fff9f0 0%, #fffcf7 100%);
        border-left: 5px solid #d4862a;
        padding: 1.3rem;
        border-radius: 8px;
        margin: 1.2rem 0;
        box-shadow: 0 3px 8px rgba(212, 134, 42, 0.15);
        color: #1a3a0f;
        font-weight: 500;
    }

    .info-box strong {
        color: #1a5f3a;
    }
    
    h3 {
        color: #1a5f3a !important;
        font-weight: 700;
        border-bottom: 3px solid #1a5f3a;
        padding-bottom: 0.5rem;
        margin-top: 2rem !important;
        margin-bottom: 1.5rem !important;
    }
    
    .stNumberInput input, .stSelectbox select {
        border: 2px solid #3da673 !important;
        border-radius: 8px;
        background-color: #ffffff !important;
        color: #1a5f3a !important;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #1a5f3a !important;
        box-shadow: 0 0 0 3px rgba(61, 166, 115, 0.15) !important;
    }
    
    .stSlider {
        padding: 1rem 0;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1a5f3a 0%, #2d8659 100%);
        color: white;
        font-weight: 700;
        font-size: 1.15rem;
        padding: 0.8rem 2.5rem;
        border: none;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(26, 95, 58, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2d8659 0%, #1a5f3a 100%);
        box-shadow: 0 6px 18px rgba(26, 95, 58, 0.6);
        transform: translateY(-2px);
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #d4862a 0%, #e89a3c 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        box-shadow: 0 3px 8px rgba(212, 134, 42, 0.3);
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #e89a3c 0%, #d4862a 100%);
        box-shadow: 0 5px 12px rgba(212, 134, 42, 0.5);
        transform: translateY(-1px);
    }
    
    .stAlert {
        border-radius: 10px;
        border-left-width: 5px;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        backdrop-filter: blur(8px);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a5f3a 0%, #2d8659 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #e8f5ed !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #ffd699 !important;
        border-bottom-color: #3da673 !important;
    }
    
    [data-testid="stSidebar"] strong {
        color: #ffffff !important;
    }
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(61, 166, 115, 0.15);
        border-radius: 8px 8px 0 0;
        color: #1a5f3a;
        font-weight: 600;
        padding: 0.8rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, #3da673 0%, #2d8659 100%);
        color: white !important;
    }
    
    .js-plotly-plot {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(26, 95, 58, 0.1);
        background: white;
    }
    
    hr {
        border-color: rgba(61, 166, 115, 0.6) !important;
        margin: 2.5rem 0;
    }
    
    label {
        color: #1a5f3a !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        price_model    = joblib.load('price_model.pkl')
        import_model   = joblib.load('import_model.pkl')
        price_features = joblib.load('price_features.pkl')
        import_features= joblib.load('import_features.pkl')
        return price_model, import_model, price_features, import_features
    except Exception:
        st.error("⚠ Models not found! Run `python ml_models.py` first.")
        st.stop()


@st.cache_data(ttl=3600)
def get_live_inr_usd() -> tuple:
    try:
        r = requests.get(
            "https://api.frankfurter.app/latest?from=USD&to=INR", timeout=6
        )
        if r.status_code == 200:
            data = r.json()
            return float(data['rates']['INR']), data['date']
    except Exception:
        pass
    return 83.0, "fallback"


@st.cache_data(ttl=3600)
def load_data():
    try:
        conn = sqlite3.connect('palm_oil_data.db')
        data = pd.read_sql('SELECT * FROM palm_oil_data ORDER BY date', conn)
        try:
            last_updated = pd.read_sql(
                "SELECT value FROM metadata WHERE key='last_updated'", conn
            )['value'].iloc[0]
        except Exception:
            last_updated = None
        conn.close()
        data['date'] = pd.to_datetime(data['date'])
        return data, last_updated
    except Exception:
        pass
    try:
        data = pd.read_csv('palm_oil_data.csv')
        data['date'] = pd.to_datetime(data['date'])
        return data, None
    except Exception:
        st.error("⚠ Data not found! Run `python data_collector.py` first.")
        st.stop()


# ── THE FIX: safe Prophet CSV loader ─────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_prophet_forecast():
    """
    Load prophet_forecast.csv with bulletproof date handling.
    Returns (hist_fitted_df, future_fc_df) or (None, None) on failure.
    """
    try:
        df = pd.read_csv('prophet_forecast.csv')

        # Convert date column to plain Python datetime via string parsing
        # — avoids every pandas / Prophet timezone / offset compatibility issue
        df['date'] = pd.to_datetime(
            df['date'].astype(str).str[:10],   # take only YYYY-MM-DD part
            format='%Y-%m-%d'
        )

        # Reset index BEFORE any filtering so indices are 0-based and contiguous
        df = df.reset_index(drop=True)

        hist = df[df['is_forecast'] == False].copy().reset_index(drop=True)
        fut  = df[df['is_forecast'] == True ].copy().reset_index(drop=True)

        # Convert to plain strings BEFORE returning — ensures zero pandas Timestamp
        # objects ever reach Plotly, which internally does Timestamp arithmetic that
        # breaks on newer pandas ("Addition of integers with Timestamp" error).
        hist['date'] = hist['date'].dt.strftime('%Y-%m-%d')
        fut['date']  = fut['date'].dt.strftime('%Y-%m-%d')

        return hist, fut
    except FileNotFoundError:
        return None, None
    except Exception as e:
        return None, str(e)


# ── Resources ─────────────────────────────────────────────────────────────────
price_model, import_model, price_features, import_features = load_models()
historical_data, db_last_updated = load_data()
live_inr_usd, inr_rate_date = get_live_inr_usd()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 About TariffFloat")
    st.markdown("""
    **Purpose:** Evidence-based policy decisions for India's crude palm oil imports
    
    **Key Capabilities:**
    - Price impact forecasting
    - Import volume predictions
    - Forex savings analysis
    - Import dependency tracking
    
    **Technology:**
    - Random Forest (Price Model)
    - Gradient Boosting (Import Model)
    - Facebook Prophet (12-month forecast)
    - Live data: FRED API + Frankfurter
    
    **Problem Context:**
    India imports 94% of its palm oil needs, spending $6.5B annually. This creates:
    - Foreign exchange drain
    - Price volatility
    - Policy uncertainty
    - Challenges for domestic farmers
    """)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; margin-bottom: 3rem;">
    <div style="font-size: 4rem; margin-bottom: -1rem;">🌴</div>
    <h1 style="margin: 0;">TariffFloat</h1>
    <p class="caption-text">AI-Powered Palm Oil Tariff Impact Simulator</p>
</div>
""", unsafe_allow_html=True)


# ── Current Market State ──────────────────────────────────────────────────────
st.markdown("### 📍 Current Market State")

_src_label  = f"DB updated: {db_last_updated[:10]}" if db_last_updated else "Source: CSV"
_rate_label = f"Rate as of: {inr_rate_date}" if inr_rate_date != "fallback" else "Rate: fallback ₹83"
st.caption(f"Data source — {_src_label}  |  INR/USD — {_rate_label}")

col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    st.metric("Global CPO Price",
              f"${historical_data['global_cpo_price_usd_tonne'].iloc[-1]:.0f}/tonne",
              help="International market price for crude palm oil")
with col_s2:
    st.metric("Current Import Tariff",
              f"{historical_data['tariff_pct'].iloc[-1]:.1f}%",
              help="Active customs duty on CPO imports")
with col_s3:
    st.metric("Monthly Import Volume",
              f"{historical_data['import_volume_tonnes'].iloc[-1]/1000:.0f}K tonnes",
              help="Recent average monthly imports")
with col_s4:
    inr_delta = f"{live_inr_usd - 83.0:+.2f} vs ₹83 base" if inr_rate_date != "fallback" else None
    st.metric("Live INR/USD", f"₹{live_inr_usd:.2f}",
              delta=inr_delta, delta_color="off",
              help=f"Live rate from Frankfurter API ({inr_rate_date})")

st.markdown("---")


# ── Policy Simulation Controls ────────────────────────────────────────────────
st.markdown("### ⚙️ Policy Simulation Controls")

col_c1, col_c2, col_c3, col_c4 = st.columns([2, 2, 2, 1])
with col_c1:
    current_global_price = st.number_input(
        "Global CPO Price (USD/tonne)", min_value=500.0, max_value=2000.0,
        value=float(historical_data['global_cpo_price_usd_tonne'].iloc[-1]),
        step=10.0)
with col_c2:
    current_tariff = st.number_input(
        "Base Tariff (%)", min_value=0.0, max_value=30.0,
        value=float(historical_data['tariff_pct'].iloc[-1]), step=0.5)
with col_c3:
    tariff_change = st.slider(
        "Tariff Adjustment (± percentage points)",
        min_value=-10.0, max_value=20.0, value=0.0, step=0.5)
with col_c4:
    forecast_months = st.selectbox("Forecast Period", [3, 6, 9, 12], index=3)

new_tariff = current_tariff + tariff_change

if tariff_change != 0:
    st.markdown(f"""
    <div class="info-box">
        <strong>Policy Scenario:</strong> Tariff adjustment of
        <strong>{tariff_change:+.1f} pp</strong>
        (from {current_tariff:.1f}% to <strong>{new_tariff:.1f}%</strong>)
    </div>""", unsafe_allow_html=True)

run_simulation = st.button("🚀 Run Simulation", type="primary", use_container_width=True)


# ── Helper functions ──────────────────────────────────────────────────────────
def prepare_forecast_features(base_data, months, new_tariff_val, global_price):
    last_row = base_data.iloc[-1]
    forecast_data = []
    for i in range(months):
        ts    = pd.Timestamp.now() + timedelta(days=30 * i)
        month = ts.month
        quarter = (month - 1) // 3 + 1
        price_features_dict = {
            'global_cpo_price_usd_tonne': global_price,
            'tariff_pct':                 new_tariff_val,
            'inr_usd':                    live_inr_usd,
            'year':                       ts.year,
            'month':                      month,
            'quarter':                    quarter,
            'global_price_lag1':          last_row['global_cpo_price_usd_tonne'] if i == 0 else global_price,
            'global_price_lag2':          last_row['global_cpo_price_usd_tonne'],
            'domestic_price_lag1':        last_row['domestic_cpo_price_inr_10kg'],
            'global_price_ma3':           global_price,
            'price_tariff_interaction':   global_price * new_tariff_val,
            'import_dependency':          last_row.get('import_volume_tonnes', 900000) /
                                          (last_row.get('import_volume_tonnes', 900000) +
                                           last_row.get('domestic_production_tonnes', 30000)) * 100,
        }
        import_features_dict = {
            'global_cpo_price_usd_tonne':  global_price,
            'tariff_pct':                  new_tariff_val,
            'domestic_production_tonnes':  last_row.get('domestic_production_tonnes', 30000),
            'month':                       month,
            'quarter':                     quarter,
            'import_volume_lag1':          last_row.get('import_volume_tonnes', 900000),
            'import_volume_lag2':          last_row.get('import_volume_tonnes', 900000),
            'import_volume_ma3':           last_row.get('import_volume_tonnes', 900000),
            'price_tariff_interaction':    global_price * new_tariff_val,
            'domestic_cpo_price_inr_10kg': 0,
        }
        forecast_data.append({
            'price_features':  price_features_dict,
            'import_features': import_features_dict,
            'date':            ts,
        })
    return forecast_data


def run_forecast(forecast_data):
    results = []
    for item in forecast_data:
        p_in = pd.DataFrame([item['price_features']])[price_features]
        dom_price = price_model.predict(p_in)[0]
        item['import_features']['domestic_cpo_price_inr_10kg'] = dom_price
        i_in = pd.DataFrame([item['import_features']])[import_features]
        imp_vol = import_model.predict(i_in)[0]
        results.append({'date': item['date'],
                        'domestic_price': dom_price,
                        'import_volume':  imp_vol})
    return pd.DataFrame(results)


# ── Main simulation ───────────────────────────────────────────────────────────
if run_simulation:
    with st.spinner("🔄 Running AI simulation..."):
        baseline_results = run_forecast(
            prepare_forecast_features(historical_data, forecast_months,
                                      current_tariff, current_global_price))
        policy_results = run_forecast(
            prepare_forecast_features(historical_data, forecast_months,
                                      new_tariff, current_global_price))

        avg_price_baseline  = baseline_results['domestic_price'].mean()
        avg_price_policy    = policy_results['domestic_price'].mean()
        price_impact_pct    = (avg_price_policy - avg_price_baseline) / avg_price_baseline * 100

        avg_import_baseline = baseline_results['import_volume'].mean()
        avg_import_policy   = policy_results['import_volume'].mean()
        import_impact_pct   = (avg_import_policy - avg_import_baseline) / avg_import_baseline * 100

        usd_per_kg        = current_global_price / 1000
        forex_baseline    = avg_import_baseline * usd_per_kg * forecast_months
        forex_policy      = avg_import_policy   * usd_per_kg * forecast_months
        forex_savings     = forex_baseline - forex_policy
        forex_savings_inr = forex_savings * live_inr_usd

    st.success("✅ Simulation Complete")
    st.markdown("---")

    # Impact metrics
    st.markdown("### 📊 Forecasted Impact Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Domestic Price", f"₹{avg_price_policy:.0f}/10kg",
                  f"{price_impact_pct:+.1f}%", delta_color="inverse")
    with col2:
        st.metric("Import Volume", f"{avg_import_policy/1000:.0f}K tonnes/mo",
                  f"{import_impact_pct:+.1f}%")
    with col3:
        forex_label = "Savings" if forex_savings > 0 else "Additional Cost"
        st.metric(f"Forex {forex_label}", f"${abs(forex_savings)/1e6:.1f}M",
                  f"≈ ₹{abs(forex_savings_inr)/1e7:.1f} Cr  ({forecast_months} mo)",
                  delta_color="normal" if forex_savings > 0 else "inverse")
    with col4:
        import_dep = avg_import_policy / (avg_import_policy + 30000) * 100
        st.metric("Import Dependency", f"{import_dep:.1f}%",
                  f"{import_impact_pct:+.1f}pp", delta_color="inverse")

    st.markdown("---")

    # Forecast comparison charts
    st.markdown("### 📈 Forecast Comparison")
    col_left, col_right = st.columns(2)

    with col_left:
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=baseline_results['date'], y=baseline_results['domestic_price'],
            mode='lines+markers', name=f'Baseline ({current_tariff:.1f}%)',
            line=dict(color='#3b7a57', width=3), marker=dict(size=8)))
        fig_price.add_trace(go.Scatter(
            x=policy_results['date'], y=policy_results['domestic_price'],
            mode='lines+markers', name=f'Policy Scenario ({new_tariff:.1f}%)',
            line=dict(color='#d4a574', width=3, dash='dash'), marker=dict(size=8)))
        fig_price.update_layout(
            title="Domestic Price Forecast (₹/10kg)",
            xaxis_title="Month", yaxis_title="Price (INR/10kg)",
            hovermode='x unified',
            plot_bgcolor='#ffffff', paper_bgcolor='#f0f7f0',
            font=dict(family="Arial", size=12, color='#1a3a0f'),
            xaxis=dict(gridcolor='#c8e6c9', linecolor='#1a5f3a',
                       tickfont=dict(color='#1a3a0f'), title_font=dict(color='#1a3a0f')),
            yaxis=dict(gridcolor='#c8e6c9', linecolor='#1a5f3a',
                       tickfont=dict(color='#1a3a0f'), title_font=dict(color='#1a3a0f')),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                        font=dict(color='#1a3a0f'), bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='#3da673', borderwidth=1))
        st.plotly_chart(fig_price, use_container_width=True)

    with col_right:
        fig_import = go.Figure()
        fig_import.add_trace(go.Scatter(
            x=baseline_results['date'], y=baseline_results['import_volume']/1000,
            mode='lines+markers', name=f'Baseline ({current_tariff:.1f}%)',
            line=dict(color='#5a7c3e', width=3), marker=dict(size=8)))
        fig_import.add_trace(go.Scatter(
            x=policy_results['date'], y=policy_results['import_volume']/1000,
            mode='lines+markers', name=f'Policy Scenario ({new_tariff:.1f}%)',
            line=dict(color='#c17a3b', width=3, dash='dash'), marker=dict(size=8)))
        fig_import.update_layout(
            title="Import Volume Forecast (K tonnes/month)",
            xaxis_title="Month", yaxis_title="Import Volume (K tonnes)",
            hovermode='x unified',
            plot_bgcolor='#ffffff', paper_bgcolor='#f0f7f0',
            font=dict(family="Arial", size=12, color='#1a3a0f'),
            xaxis=dict(gridcolor='#c8e6c9', linecolor='#1a5f3a',
                       tickfont=dict(color='#1a3a0f'), title_font=dict(color='#1a3a0f')),
            yaxis=dict(gridcolor='#c8e6c9', linecolor='#1a5f3a',
                       tickfont=dict(color='#1a3a0f'), title_font=dict(color='#1a3a0f')),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                        font=dict(color='#1a3a0f'), bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='#3da673', borderwidth=1))
        st.plotly_chart(fig_import, use_container_width=True)

    st.markdown("---")

    # ── Prophet chart — FIXED ─────────────────────────────────────────────────
    st.markdown("### 🔮 Prophet Price Forecast (12-Month Horizon)")

    hist_fitted, future_fc = load_prophet_forecast()

    if hist_fitted is None and future_fc is None:
        st.info("Prophet forecast not found. Run `python ml_models.py` to generate **prophet_forecast.csv**.")
    elif isinstance(future_fc, str):
        st.warning(f"Could not load Prophet forecast: {future_fc}")
    else:
        try:
            fig_prophet = go.Figure()

            # Trace 1 — actual historical prices
            fig_prophet.add_trace(go.Scatter(
                x=historical_data['date'].dt.strftime('%Y-%m-%d').tolist(),
                y=historical_data['domestic_cpo_price_inr_10kg'],
                mode='lines', name='Actual (historical)',
                line=dict(color='#1a5f3a', width=2.5)))

            # Trace 2 — Prophet in-sample fitted
            fig_prophet.add_trace(go.Scatter(
                x=hist_fitted['date'].tolist(),
                y=hist_fitted['forecast_price_inr_10kg'].tolist(),
                mode='lines', name='Prophet fitted',
                line=dict(color='#3da673', width=1.5, dash='dot')))

            # Trace 3 — 80% CI band via fill='tonexty' (two traces, no x-concatenation)
            # fill='toself' with list + list triggers Plotly integer-arithmetic on
            # date strings; fill='tonexty' avoids that path entirely.
            x_fut = future_fc['date'].tolist()
            fig_prophet.add_trace(go.Scatter(
                x=x_fut,
                y=future_fc['lower_80pct'].tolist(),
                mode='lines',
                line=dict(color='rgba(0,0,0,0)'),
                hoverinfo='skip',
                showlegend=False,
                name='_ci_lower'))

            fig_prophet.add_trace(go.Scatter(
                x=x_fut,
                y=future_fc['upper_80pct'].tolist(),
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(61,166,115,0.15)',
                line=dict(color='rgba(0,0,0,0)'),
                hoverinfo='skip',
                showlegend=True,
                name='80% confidence interval'))

            # Trace 5 — 12-month forecast line
            fig_prophet.add_trace(go.Scatter(
                x=x_fut,
                y=future_fc['forecast_price_inr_10kg'].tolist(),
                mode='lines+markers', name='Forecast (next 12 mo)',
                line=dict(color='#d4862a', width=2.5, dash='dash'),
                marker=dict(size=7, symbol='diamond')))

            # Vertical line at forecast start — Scatter trace avoids add_vline's
            # internal integer-offset arithmetic on string x values.
            y_min = float(hist_fitted['forecast_price_inr_10kg'].min())
            y_max = float(future_fc['upper_80pct'].max())
            fig_prophet.add_trace(go.Scatter(
                x=[x_fut[0], x_fut[0]],
                y=[y_min, y_max],
                mode='lines',
                line=dict(color='#888888', width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip',
                name='_vline'))

            fig_prophet.update_layout(
                title='Domestic CPO Price — Historical + Prophet 12-Month Forecast (₹/10 kg)',
                xaxis_title='Date', yaxis_title='Price (INR / 10 kg)',
                hovermode='x unified',
                plot_bgcolor='#ffffff', paper_bgcolor='#f0f7f0',
                font=dict(family='Arial', size=12, color='#1a3a0f'),
                xaxis=dict(gridcolor='#c8e6c9', linecolor='#1a5f3a',
                           tickfont=dict(color='#1a3a0f'), title_font=dict(color='#1a3a0f')),
                yaxis=dict(gridcolor='#c8e6c9', linecolor='#1a5f3a',
                           tickfont=dict(color='#1a3a0f'), title_font=dict(color='#1a3a0f')),
                legend=dict(orientation='h', yanchor='bottom', y=1.02,
                            xanchor='center', x=0.5,
                            font=dict(color='#1a3a0f'), bgcolor='rgba(255,255,255,0.8)',
                            bordercolor='#3da673', borderwidth=1))

            st.plotly_chart(fig_prophet, use_container_width=True)

            p_mean = future_fc['forecast_price_inr_10kg'].mean()
            p_lo   = future_fc['lower_80pct'].min()
            p_hi   = future_fc['upper_80pct'].max()
            st.caption(
                f"Prophet forecast mean: **₹{p_mean:.0f}/10kg** | "
                f"80% CI: ₹{p_lo:.0f} – ₹{p_hi:.0f} | "
                f"Regressors: global price, tariff, INR/USD | "
                f"Retrain: `python ml_models.py`")

        except Exception as _e:
            st.warning(f"Could not render Prophet chart: {_e}")

    st.markdown("---")

    # AI Policy Insights
    st.markdown("### 🎯 AI-Generated Policy Insights")
    ins1, ins2 = st.columns(2)
    with ins1:
        if price_impact_pct > 5:
            st.warning(f"⚠️ **Consumer Price Alert**\n\nDomestic prices may increase by **{price_impact_pct:.1f}%**. Consider:\n- Gradual tariff implementation\n- Consumer subsidies for vulnerable groups\n- Communication strategy for price changes")
        elif price_impact_pct < -5:
            st.success(f"✅ **Consumer Benefit**\n\nDomestic prices may decrease by **{abs(price_impact_pct):.1f}%**. Benefits:\n- Lower cooking oil costs\n- Reduced inflation pressure\n- Support for food processing industry")
        else:
            st.info(f"ℹ️ **Neutral Price Impact**\n\nPrice change of **{price_impact_pct:.1f}%** is within acceptable range.")
    with ins2:
        if import_impact_pct < -10:
            st.success(f"✅ **Import Reduction Achieved**\n\nImports may decrease by **{abs(import_impact_pct):.1f}%**. Benefits:\n- Forex savings: **${abs(forex_savings)/1e6:.1f}M** (≈ ₹{abs(forex_savings_inr)/1e7:.1f} Cr)\n- Support for domestic farmers\n- Progress toward self-reliance goals")
        elif import_impact_pct > 10:
            st.warning(f"⚠️ **Import Surge Warning**\n\nImports may increase by **{import_impact_pct:.1f}%**. Monitor:\n- Forex outflow implications\n- Impact on domestic production\n- Trade balance effects")
        else:
            st.info(f"ℹ️ **Stable Import Levels**\n\nImport change of **{import_impact_pct:.1f}%** maintains current trade balance.")

    # Export
    st.markdown("---")
    st.markdown("### 💾 Export Simulation Results")
    export_df = pd.DataFrame({
        'Date':                  policy_results['date'],
        'Domestic_Price_INR_10kg': policy_results['domestic_price'].round(2),
        'Import_Volume_Tonnes':  policy_results['import_volume'].round(0),
        'Tariff_Pct':            new_tariff,
        'Price_Impact_Pct':      price_impact_pct,
        'Import_Impact_Pct':     import_impact_pct,
        'Forex_Savings_USD':     forex_savings,
    })
    st.download_button(
        label="📥 Download Forecast Report (CSV)",
        data=export_df.to_csv(index=False),
        file_name=f"tariff_impact_forecast_{new_tariff:.1f}pct_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True)

else:
    st.markdown("---")
    st.markdown("### 🚀 Get Started")
    st.info("**Adjust the policy simulation controls above and click 'Run Simulation' to forecast tariff impacts.**")
    st.markdown("### 🎯 What This Tool Does")
    ci1, ci2 = st.columns(2)
    with ci1:
        st.markdown("""
        **Forecasts the impact of CPO tariff changes on:**
        - Domestic palm oil prices (consumer impact)
        - Import volumes (trade balance)
        - Foreign exchange costs (forex savings)
        - Import dependency (self-reliance metrics)
        """)
    with ci2:
        st.markdown("""
        **Powered by machine learning models:**
        - Random Forest for price prediction
        - Gradient Boosting for import volumes
        - Facebook Prophet for 12-month forecast
        - Live data: FRED API + Frankfurter
        """)

# Footer
st.markdown("---")
st.caption(
    f"🌴 TariffFloat | Evidence-Based Policy Decisions for India's Edible Oil Security | "
    f"CPO prices: FRED PPOILUSDM | INR/USD: Frankfurter API (₹{live_inr_usd:.2f}, {inr_rate_date}) | "
    f"Storage: SQLite")