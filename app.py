# ============================================================
#  AI-Based Hazardous Waste Prediction & Classification System
#  For Automobile Workshops
#  Author  : Mechanical Engineering Project Expo
#  Stack   : Python · Streamlit · Scikit-learn · Plotly
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG & GLOBAL STYLES
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HazWaste AI | Automobile Workshop",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for a dark industrial-tech look
st.markdown("""
<style>
  /* ── Base ── */
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700;900&display=swap');

  html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: #0a0e1a;
    color: #cdd6f4;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629 0%, #111827 100%);
    border-right: 1px solid #1e2d45;
  }

  /* ── Metric cards ── */
  div[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 0 18px rgba(56,189,248,0.07);
  }
  div[data-testid="metric-container"] label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7dd3fc !important;
    font-family: 'Share Tech Mono', monospace;
  }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-size: 1.9rem !important;
    font-weight: 700;
    color: #e2e8f0 !important;
  }

  /* ── Buttons ── */
  div.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #0ea5e9);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2.2rem;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    transition: all 0.25s;
    box-shadow: 0 4px 20px rgba(14,165,233,0.3);
  }
  div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 28px rgba(14,165,233,0.5);
  }

  /* ── Section headers ── */
  .section-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 4px;
  }
  .big-title {
    font-size: 1.55rem;
    font-weight: 900;
    color: #f1f5f9;
    margin-bottom: 18px;
    line-height: 1.2;
  }

  /* ── Hazard badge ── */
  .hazard-badge {
    display: inline-block;
    padding: 6px 22px;
    border-radius: 999px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.05rem;
    font-weight: bold;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }
  .badge-low    { background:#052e16; color:#4ade80; border:1.5px solid #16a34a; }
  .badge-medium { background:#422006; color:#fbbf24; border:1.5px solid #d97706; }
  .badge-high   { background:#450a0a; color:#f87171; border:1.5px solid #dc2626; }

  /* ── Recommendation card ── */
  .rec-card {
    background: #111827;
    border-left: 4px solid #38bdf8;
    border-radius: 0 10px 10px 0;
    padding: 10px 16px;
    margin-bottom: 9px;
    font-size: 0.9rem;
    line-height: 1.5;
  }

  /* ── Alert boxes ── */
  .alert-high   { background:#1f0707; border:1px solid #991b1b; border-radius:10px; padding:14px 18px; }
  .alert-medium { background:#1c1007; border:1px solid #92400e; border-radius:10px; padding:14px 18px; }
  .alert-low    { background:#022c22; border:1px solid #065f46; border-radius:10px; padding:14px 18px; }

  /* Plotly chart background harmony */
  .js-plotly-plot .plotly .bg { fill: #111827 !important; }

  /* Input widgets */
  div[data-baseweb="input"] input,
  div[data-baseweb="slider"] {
    background: #1e293b !important;
  }
  label[data-testid="stWidgetLabel"] {
    font-size: 0.82rem;
    color: #94a3b8;
    font-weight: 600;
    letter-spacing: 0.05em;
  }

  /* Divider */
  hr { border-color: #1e3a5f; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CONSTANTS & LOOKUP TABLES
# ─────────────────────────────────────────────

# Waste generation coefficients (litres / kg per service event)
WASTE_COEFFICIENTS = {
    "used_oil_per_oil_change":      4.5,   # litres
    "coolant_per_vehicle":          0.8,   # litres
    "acid_per_battery":             1.2,   # litres of battery acid
    "oil_from_brake_service":       0.3,   # litres of brake fluid
    "coolant_from_brake":           0.1,
    "general_oil_per_vehicle":      0.25,  # drip / residual
}

# Regulatory disposal thresholds (litres per day)
THRESHOLDS = {
    "oil":     {"low": 10, "medium": 25},
    "coolant": {"low": 5,  "medium": 12},
    "acid":    {"low": 2,  "medium": 5},
}

HAZARD_COLORS = {
    "Low":    "#4ade80",
    "Medium": "#fbbf24",
    "High":   "#f87171",
}

RECOMMENDATIONS_DB = {
    "Low": [
        "✅ Store used oil in sealed, labelled HDPE drums (≤ 200 L capacity).",
        "✅ Schedule quarterly pick-up with a certified waste hauler.",
        "✅ Maintain a waste log; update after every 10 oil changes.",
        "✅ Keep absorbent pads and spill kits stocked at service bays.",
        "✅ Train new staff on basic spill-response procedures.",
    ],
    "Medium": [
        "⚠️ Increase collection frequency to bi-weekly certified disposal.",
        "⚠️ Install secondary containment (drip trays, bunded storage) immediately.",
        "⚠️ Label all waste containers with hazard class, date, and volume.",
        "⚠️ Conduct monthly internal audits against local EPA waste regulations.",
        "⚠️ Equip staff with nitrile gloves, goggles, and chemical aprons.",
        "⚠️ Document battery acid handling with material safety data sheets (MSDS).",
    ],
    "High": [
        "🚨 IMMEDIATE ACTION: Contact licensed hazardous-waste contractor today.",
        "🚨 Do NOT mix acid waste with oil or coolant — risk of toxic reaction.",
        "🚨 Install acid-neutralisation tank before next battery service batch.",
        "🚨 Report volumes to the Pollution Control Board if thresholds are exceeded.",
        "🚨 Restrict access to waste storage area — signage and lock required.",
        "🚨 Implement ISO 14001 environmental management plan within 30 days.",
        "🚨 Arrange third-party environmental compliance audit immediately.",
    ],
}


# ─────────────────────────────────────────────
#  SYNTHETIC TRAINING DATASET
# ─────────────────────────────────────────────

def generate_training_data(n: int = 500) -> pd.DataFrame:
    """
    Generates synthetic historical records that mimic real workshop data.
    Features: vehicles, oil_changes, battery_replacements, brake_services
    Targets: oil_waste, coolant_waste, acid_waste (litres/day)
    """
    rng = np.random.default_rng(42)

    vehicles     = rng.integers(5, 80, n).astype(float)
    oil_changes  = (vehicles * rng.uniform(0.25, 0.55, n)).astype(int).astype(float)
    batteries    = (vehicles * rng.uniform(0.05, 0.18, n)).astype(int).astype(float)
    brakes       = (vehicles * rng.uniform(0.10, 0.30, n)).astype(int).astype(float)

    noise = lambda scale: rng.normal(0, scale, n)

    oil_waste     = (oil_changes * WASTE_COEFFICIENTS["used_oil_per_oil_change"]
                     + vehicles  * WASTE_COEFFICIENTS["general_oil_per_vehicle"]
                     + brakes    * WASTE_COEFFICIENTS["oil_from_brake_service"]
                     + noise(1.2)).clip(0)

    coolant_waste = (vehicles   * WASTE_COEFFICIENTS["coolant_per_vehicle"]
                     + brakes   * WASTE_COEFFICIENTS["coolant_from_brake"]
                     + noise(0.5)).clip(0)

    acid_waste    = (batteries  * WASTE_COEFFICIENTS["acid_per_battery"]
                     + noise(0.3)).clip(0)

    return pd.DataFrame({
        "vehicles": vehicles,
        "oil_changes": oil_changes,
        "battery_replacements": batteries,
        "brake_services": brakes,
        "oil_waste": oil_waste,
        "coolant_waste": coolant_waste,
        "acid_waste": acid_waste,
    })


# ─────────────────────────────────────────────
#  ML MODEL TRAINING
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def train_models():
    """
    Trains three separate Linear Regression pipelines (with StandardScaler)
    — one each for oil, coolant, and acid waste prediction.
    Returns fitted pipeline objects and training R² scores.
    """
    df = generate_training_data()
    features = ["vehicles", "oil_changes", "battery_replacements", "brake_services"]
    X = df[features].values

    models, scores = {}, {}
    for target in ["oil_waste", "coolant_waste", "acid_waste"]:
        y = df[target].values
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ])
        pipe.fit(X, y)
        y_pred = pipe.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        scores[target] = round(1 - ss_res / ss_tot, 4)
        models[target] = pipe

    return models, scores, df


# ─────────────────────────────────────────────
#  WASTE CALCULATION (Physics-based)
# ─────────────────────────────────────────────

def calculate_waste(vehicles: float, oil_changes: float,
                    batteries: float, brakes: float) -> dict:
    """
    Deterministic calculation of daily waste volumes using
    known engineering coefficients — no ML needed here.
    """
    oil = (oil_changes * WASTE_COEFFICIENTS["used_oil_per_oil_change"]
           + vehicles  * WASTE_COEFFICIENTS["general_oil_per_vehicle"]
           + brakes    * WASTE_COEFFICIENTS["oil_from_brake_service"])
    coolant = (vehicles * WASTE_COEFFICIENTS["coolant_per_vehicle"]
               + brakes * WASTE_COEFFICIENTS["coolant_from_brake"])
    acid = batteries * WASTE_COEFFICIENTS["acid_per_battery"]
    return {"oil": round(oil, 2), "coolant": round(coolant, 2), "acid": round(acid, 2)}


# ─────────────────────────────────────────────
#  HAZARD CLASSIFICATION
# ─────────────────────────────────────────────

def classify_hazard(waste: dict) -> tuple[str, dict]:
    """
    Classifies each waste stream and returns an overall hazard level.
    Uses a rule-based system on top of ML predictions.
    Returns (overall_level, per_stream_levels)
    """
    levels, priority = {}, {"Low": 0, "Medium": 1, "High": 2}
    for stream, volume in waste.items():
        thr = THRESHOLDS[stream]
        if volume >= thr["medium"]:
            levels[stream] = "High"
        elif volume >= thr["low"]:
            levels[stream] = "Medium"
        else:
            levels[stream] = "Low"
    overall = max(levels.values(), key=lambda l: priority[l])
    return overall, levels


# ─────────────────────────────────────────────
#  7-DAY FORECAST
# ─────────────────────────────────────────────

def forecast_7_days(models: dict, base_inputs: np.ndarray,
                    variation: float = 0.10) -> pd.DataFrame:
    """
    Generates a 7-day waste forecast using the trained ML models.
    Each day's inputs are slightly perturbed (±variation) to simulate
    realistic daily fluctuations in workshop activity.
    """
    rng = np.random.default_rng(datetime.now().toordinal())
    today = datetime.today()
    records = []

    for i in range(7):
        noise = rng.uniform(1 - variation, 1 + variation, size=base_inputs.shape)
        day_inputs = (base_inputs * noise).clip(min=0)
        row = {"date": (today + timedelta(days=i)).strftime("%b %d")}
        for target, model in models.items():
            row[target] = max(0.0, round(model.predict(day_inputs.reshape(1, -1))[0], 2))
        records.append(row)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
#  PLOTLY CHART HELPERS
# ─────────────────────────────────────────────

CHART_LAYOUT = dict(
    paper_bgcolor="#111827",
    plot_bgcolor="#0f172a",
    font=dict(family="Exo 2, sans-serif", color="#94a3b8"),
    title_font=dict(family="Share Tech Mono, monospace", color="#e2e8f0", size=14),
    legend=dict(bgcolor="#111827", bordercolor="#1e3a5f", borderwidth=1),
    xaxis=dict(gridcolor="#1e293b", zerolinecolor="#1e293b"),
    yaxis=dict(gridcolor="#1e293b", zerolinecolor="#1e293b"),
)


def plot_waste_bar(waste: dict) -> go.Figure:
    """Horizontal bar chart of today's waste volumes with hazard-coloured bars."""
    streams   = ["Oil", "Coolant", "Acid"]
    volumes   = [waste["oil"], waste["coolant"], waste["acid"]]
    _, levels = classify_hazard(waste)
    colours   = [HAZARD_COLORS[levels["oil"]],
                 HAZARD_COLORS[levels["coolant"]],
                 HAZARD_COLORS[levels["acid"]]]

    fig = go.Figure(go.Bar(
        x=volumes, y=streams, orientation="h",
        marker=dict(color=colours, line=dict(color="#0f172a", width=1.5)),
        text=[f"{v} L" for v in volumes],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=13),
        hovertemplate="<b>%{y}</b>: %{x:.2f} L<extra></extra>",
    ))
    fig.update_layout(
        title="Daily Waste Generation (Litres)",
        xaxis_title="Volume (L)",
        yaxis_title="",
        height=290,
        margin=dict(l=20, r=60, t=50, b=30),
        **CHART_LAYOUT,
    )
    return fig


def plot_forecast(forecast_df: pd.DataFrame) -> go.Figure:
    """Multi-line 7-day forecast chart with filled area bands."""
    fig = go.Figure()
    stream_cfg = {
        "oil_waste":     {"name": "Used Oil",     "color": "#f97316"},
        "coolant_waste": {"name": "Coolant",       "color": "#38bdf8"},
        "acid_waste":    {"name": "Battery Acid",  "color": "#a78bfa"},
    }
    for col, cfg in stream_cfg.items():
        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df[col],
            name=cfg["name"],
            mode="lines+markers",
            line=dict(color=cfg["color"], width=2.5),
            marker=dict(size=8, symbol="circle"),
            fill="tozeroy",
            fillcolor=cfg["color"].rstrip(")") + ",0.08)" if cfg["color"].startswith("rgb") else cfg["color"] + "14",
            hovertemplate=f"<b>{cfg['name']}</b><br>%{{x}}: %{{y:.2f}} L<extra></extra>",
        ))
    fig.update_layout(
        title="7-Day AI Waste Forecast",
        xaxis_title="Date",
        yaxis_title="Volume (Litres)",
        height=340,
        margin=dict(l=20, r=20, t=50, b=30),
        hovermode="x unified",
        **CHART_LAYOUT,
    )
    return fig


def plot_hazard_gauge(overall_level: str) -> go.Figure:
    """Gauge indicator for overall hazard level."""
    level_map = {"Low": 1, "Medium": 2, "High": 3}
    colour_map = {"Low": "#4ade80", "Medium": "#fbbf24", "High": "#f87171"}
    val = level_map[overall_level]

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=val,
        number=dict(
            suffix=f"  {overall_level.upper()}",
            font=dict(size=22, color=colour_map[overall_level],
                      family="Share Tech Mono, monospace"),
            valueformat=".0f",
        ),
        gauge=dict(
            axis=dict(range=[0, 3], tickvals=[1, 2, 3],
                      ticktext=["LOW", "MED", "HIGH"],
                      tickfont=dict(color="#94a3b8", size=11)),
            bar=dict(color=colour_map[overall_level], thickness=0.28),
            bgcolor="#0f172a",
            steps=[
                dict(range=[0, 1], color="#052e16"),
                dict(range=[1, 2], color="#1c1007"),
                dict(range=[2, 3], color="#1f0707"),
            ],
            threshold=dict(
                line=dict(color="white", width=3),
                thickness=0.82,
                value=val,
            ),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#111827",
        height=220,
        margin=dict(l=30, r=30, t=30, b=20),
        font=dict(family="Exo 2, sans-serif", color="#94a3b8"),
    )
    return fig


def plot_forecast_stacked(forecast_df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart showing total daily hazardous waste composition."""
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Used Oil",    x=forecast_df["date"], y=forecast_df["oil_waste"],
                         marker_color="#f97316", hovertemplate="%{y:.2f} L<extra>Oil</extra>"))
    fig.add_trace(go.Bar(name="Coolant",     x=forecast_df["date"], y=forecast_df["coolant_waste"],
                         marker_color="#38bdf8", hovertemplate="%{y:.2f} L<extra>Coolant</extra>"))
    fig.add_trace(go.Bar(name="Battery Acid",x=forecast_df["date"], y=forecast_df["acid_waste"],
                         marker_color="#a78bfa", hovertemplate="%{y:.2f} L<extra>Acid</extra>"))
    fig.update_layout(
        barmode="stack",
        title="Forecasted Waste Composition (Stacked)",
        xaxis_title="Date",
        yaxis_title="Total Volume (L)",
        height=310,
        margin=dict(l=20, r=20, t=50, b=30),
        **CHART_LAYOUT,
    )
    return fig


# ─────────────────────────────────────────────
#  SIDEBAR — INPUTS
# ─────────────────────────────────────────────

def render_sidebar() -> dict:
    st.sidebar.markdown("""
    <div style='text-align:center; padding: 10px 0 20px;'>
      <div style='font-family:"Share Tech Mono",monospace; font-size:0.68rem;
                  letter-spacing:0.22em; color:#38bdf8; text-transform:uppercase;'>
        Workshop Input Panel
      </div>
      <div style='font-size:1.35rem; font-weight:900; color:#f1f5f9; margin-top:4px;'>
        Daily Parameters
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    vehicles = st.sidebar.slider(
        "🚗  Vehicles Serviced Per Day", 1, 80, 20,
        help="Total number of vehicles entering the workshop in a single day."
    )
    oil_changes = st.sidebar.slider(
        "🛢️  Oil Changes", 0, 50, 8,
        help="Number of engine oil change services performed."
    )
    battery_replacements = st.sidebar.slider(
        "🔋  Battery Replacements", 0, 20, 3,
        help="Number of lead-acid batteries replaced today."
    )
    brake_services = st.sidebar.slider(
        "🔧  Brake Services", 0, 30, 6,
        help="Number of brake fluid flushes / brake system services."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='font-family:"Share Tech Mono",monospace; font-size:0.65rem;
                color:#475569; text-transform:uppercase; letter-spacing:0.15em;
                text-align:center; padding-top:4px;'>
      Model Info
    </div>
    """, unsafe_allow_html=True)

    return dict(
        vehicles=vehicles,
        oil_changes=oil_changes,
        batteries=battery_replacements,
        brakes=brake_services,
    )


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────

def main():
    # ── Train models (cached) ──────────────────
    with st.spinner("🔬 Training AI models on historical workshop data…"):
        models, r2_scores, train_df = train_models()

    # ── Inputs from sidebar ───────────────────
    inputs = render_sidebar()

    # Show model R² in sidebar
    st.sidebar.markdown(f"""
    <div style='background:#0f172a; border:1px solid #1e3a5f; border-radius:8px;
                padding:12px; font-family:"Share Tech Mono",monospace; font-size:0.76rem;'>
      <div style='color:#38bdf8; margin-bottom:6px;'>MODEL ACCURACY (R²)</div>
      <div>🛢️ Oil Waste     <b style='color:#f97316'>{r2_scores['oil_waste']:.4f}</b></div>
      <div>💧 Coolant       <b style='color:#38bdf8'>{r2_scores['coolant_waste']:.4f}</b></div>
      <div>⚡ Battery Acid  <b style='color:#a78bfa'>{r2_scores['acid_waste']:.4f}</b></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Header ────────────────────────────────
    col_logo, col_title = st.columns([1, 9])
    with col_title:
        st.markdown("""
        <div style='padding: 10px 0 6px;'>
          <div class='section-title'>⚠ Artificial Intelligence System</div>
          <div style='font-size:2.1rem; font-weight:900; color:#f1f5f9; line-height:1.1;'>
            Hazardous Waste Prediction & Classification
          </div>
          <div style='color:#64748b; font-size:0.88rem; margin-top:4px;'>
            Automobile Workshop Environmental Monitoring Dashboard
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 6px 0 18px;'>", unsafe_allow_html=True)

    # ── Compute waste & classify ───────────────
    waste         = calculate_waste(**inputs)
    overall, per  = classify_hazard(waste)
    total_waste   = sum(waste.values())

    # Prepare feature vector for ML
    X_input = np.array([
        inputs["vehicles"],
        inputs["oil_changes"],
        inputs["batteries"],
        inputs["brakes"],
    ])

    # 7-day forecast
    forecast_df = forecast_7_days(models, X_input)

    # ── KPI Metrics Row ────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🚗 Vehicles / Day",   f"{inputs['vehicles']}")
    m2.metric("🛢️ Oil Waste (L)",    f"{waste['oil']:.1f}",
              delta=f"{waste['oil'] - THRESHOLDS['oil']['low']:.1f} vs threshold")
    m3.metric("💧 Coolant (L)",      f"{waste['coolant']:.1f}")
    m4.metric("⚡ Battery Acid (L)", f"{waste['acid']:.1f}")
    m5.metric("🧪 Total Waste (L)",  f"{total_waste:.1f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Waste bar + Hazard gauge ────────
    r1c1, r1c2 = st.columns([3, 2])
    with r1c1:
        st.plotly_chart(plot_waste_bar(waste), use_container_width=True)
    with r1c2:
        st.markdown("""
        <div class='section-title' style='margin-bottom:8px;'>Overall Hazard Level</div>
        """, unsafe_allow_html=True)
        st.plotly_chart(plot_hazard_gauge(overall), use_container_width=True)

        badge_cls = f"badge-{overall.lower()}"
        st.markdown(f"""
        <div style='text-align:center; margin-top:4px;'>
          <span class='hazard-badge {badge_cls}'>{overall} Hazard</span>
        </div>
        """, unsafe_allow_html=True)

        # Per-stream badges
        st.markdown("<br>", unsafe_allow_html=True)
        cols_b = st.columns(3)
        for i, (stream, level) in enumerate(per.items()):
            bc = f"badge-{level.lower()}"
            cols_b[i].markdown(f"""
            <div style='text-align:center;'>
              <div style='font-size:0.68rem; color:#64748b; font-family:"Share Tech Mono",monospace;
                          text-transform:uppercase; letter-spacing:0.1em;'>{stream}</div>
              <span class='hazard-badge {bc}' style='font-size:0.72rem; padding:3px 10px;'>
                {level}
              </span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 2: 7-Day forecast charts ──────────
    st.markdown("<div class='big-title'>📈 7-Day AI Forecast</div>", unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    with f1:
        st.plotly_chart(plot_forecast(forecast_df), use_container_width=True)
    with f2:
        st.plotly_chart(plot_forecast_stacked(forecast_df), use_container_width=True)

    # Forecast data table
    with st.expander("📋 View Raw Forecast Data Table"):
        display_df = forecast_df.rename(columns={
            "date": "Date",
            "oil_waste": "Oil Waste (L)",
            "coolant_waste": "Coolant (L)",
            "acid_waste": "Acid (L)",
        })
        display_df["Total (L)"] = (display_df["Oil Waste (L)"]
                                   + display_df["Coolant (L)"]
                                   + display_df["Acid (L)"]).round(2)
        st.dataframe(display_df.set_index("Date"), use_container_width=True)

    st.markdown("---")

    # ── Row 3: Alert + Recommendations ────────
    st.markdown("<div class='big-title'>🛡️ Hazard Alert & Recommendations</div>",
                unsafe_allow_html=True)

    alert_cls   = f"alert-{overall.lower()}"
    alert_icon  = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[overall]
    alert_desc  = {
        "Low":    "Waste levels are within safe operational limits. Maintain current practices.",
        "Medium": "Waste generation is approaching regulatory thresholds. Immediate preventive action required.",
        "High":   "CRITICAL: Hazardous waste exceeds safe limits. Regulatory action may be required.",
    }[overall]

    st.markdown(f"""
    <div class='{alert_cls}' style='margin-bottom:20px;'>
      <div style='font-size:1.1rem; font-weight:700; margin-bottom:5px;'>
        {alert_icon} {overall.upper()} HAZARD ALERT
      </div>
      <div style='color:#cbd5e1; font-size:0.9rem;'>{alert_desc}</div>
    </div>
    """, unsafe_allow_html=True)

    rc1, rc2 = st.columns([1, 1])
    with rc1:
        st.markdown("<div class='section-title' style='margin-bottom:10px;'>Treatment Recommendations</div>",
                    unsafe_allow_html=True)
        for rec in RECOMMENDATIONS_DB[overall]:
            st.markdown(f"<div class='rec-card'>{rec}</div>", unsafe_allow_html=True)

    with rc2:
        st.markdown("<div class='section-title' style='margin-bottom:10px;'>Waste Stream Analysis</div>",
                    unsafe_allow_html=True)
        for stream, level in per.items():
            thr    = THRESHOLDS[stream]
            volume = waste[stream]
            pct    = min(100, round(volume / thr["medium"] * 100, 1))
            bar_col = HAZARD_COLORS[level]
            label   = stream.replace("_", " ").title()
            st.markdown(f"""
            <div style='margin-bottom:14px;'>
              <div style='display:flex; justify-content:space-between; font-size:0.82rem;
                          color:#94a3b8; margin-bottom:5px;'>
                <span>{label}</span>
                <span style='color:{bar_col}; font-weight:700;'>{volume} L — {level}</span>
              </div>
              <div style='background:#1e293b; border-radius:999px; height:8px; overflow:hidden;'>
                <div style='width:{pct}%; height:100%; background:{bar_col};
                            border-radius:999px; transition:width 0.5s;'></div>
              </div>
              <div style='font-size:0.7rem; color:#475569; margin-top:3px;'>
                Threshold: {thr["low"]} L (low) / {thr["medium"]} L (medium)
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Footer ────────────────────────────────
    st.markdown(f"""
    <div style='text-align:center; padding:16px 0 8px;
                font-family:"Share Tech Mono",monospace; font-size:0.68rem;
                color:#334155; letter-spacing:0.12em;'>
      HAZWASTE AI v2.0 · MECHANICAL ENGINEERING PROJECT EXPO ·
      MODEL: LINEAR REGRESSION PIPELINE · TRAINING SAMPLES: 500 ·
      LAST RUN: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
