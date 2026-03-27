# app/streamlit_app.py

"""
F1 2026 Regulation Impact Analysis — Streamlit Dashboard
"""

import os
import sys
import base64

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

for path in [APP_DIR, PROJECT_ROOT, SRC_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard_data import (
    load_simulation_results,
    load_validation_results,
    load_gold_data_summary,
    load_driver_simulation_data,
    load_shap_summary,
    load_mlflow_summary,
    load_model_metadata,
    get_plot_path,
    get_impact_color,
    get_impact_label
)

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="F1 2026 Regulation Impact Analysis",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourname/F1-2026-regulation-impact-analysis",
        "Report a bug": "https://github.com/yourname/F1-2026-regulation-impact-analysis/issues",
        "About": "F1 2026 Regulation Impact Analysis — A Data Science Portfolio Project"
    }
)

# ============================================================
# Helpers
# ============================================================

def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default

def img_to_base64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_path = os.path.join(APP_DIR, "assets", "f1_bg.jpg")
bg_base64 = img_to_base64(bg_path)

# ============================================================
# Custom CSS
# ============================================================

background_css = f"""
    .stApp {{
        background:
            linear-gradient(rgba(10,12,18,0.90), rgba(10,12,18,0.92)),
            url("data:image/jpg;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
"""

st.markdown(f"""
<style>
    {background_css}

    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}

    section[data-testid="stSidebar"] {{
        background: rgba(28, 30, 39, 0.92);
        border-right: 1px solid rgba(255,255,255,0.08);
    }}

    .metric-card {{
        background: rgba(34, 37, 49, 0.92);
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid #E8002D;
        margin-bottom: 12px;
        color: #F5F7FA;
    }}

    .content-panel {{
        background: rgba(18, 20, 28, 0.88);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }}

    .section-header {{
        color: #F5F7FA;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #E8002D;
        padding-bottom: 6px;
    }}

    .insight-box {{
        background: rgba(20, 50, 95, 0.35);
        border-radius: 10px;
        padding: 14px 16px;
        border-left: 4px solid #0067FF;
        margin: 10px 0;
        font-size: 0.95rem;
        color: #F5F7FA;
    }}

    .warning-box {{
        background: rgba(110, 70, 20, 0.28);
        border-radius: 10px;
        padding: 14px 16px;
        border-left: 4px solid #FF8C00;
        margin: 10px 0;
        font-size: 0.95rem;
        color: #F5F7FA;
    }}

    .dark-card {{
        background: rgba(34, 37, 49, 0.92);
        border-radius: 14px;
        padding: 20px;
        border-top: 4px solid #0067FF;
        color: #F5F7FA;
        margin-bottom: 10px;
    }}

    .hero-card {{
        background: linear-gradient(135deg, rgba(27,27,39,0.95) 0%, rgba(38,38,58,0.92) 100%);
        padding: 40px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }}

    div[data-testid="stDataFrame"] {{
        background: rgba(22, 24, 32, 0.95);
        border-radius: 10px;
        padding: 6px;
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Color Constants
# ============================================================

F1_RED    = "#E8002D"
F1_DARK   = "#15151E"
F1_BLUE   = "#0067FF"
F1_SILVER = "#C0C0C0"
F1_WHITE  = "#FFFFFF"

# ============================================================
# Data Loading
# ============================================================

@st.cache_data(show_spinner=False)
def load_all_data():
    with st.spinner("Loading F1 data..."):
        sim_df      = load_simulation_results()
        val_df      = load_validation_results()
        gold_df     = load_gold_data_summary()
        driver_df   = load_driver_simulation_data()
        shap_data   = load_shap_summary()
        mlflow_data = load_mlflow_summary()
        model_meta  = load_model_metadata()
    return sim_df, val_df, gold_df, driver_df, shap_data, mlflow_data, model_meta

(
    sim_df, val_df, gold_df,
    driver_df, shap_data, mlflow_data, model_meta
) = load_all_data()

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown(
        "<h1 style='color:#FF1744; font-size:2rem; margin-bottom:0;'>🏎️ F1 2026</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='color:#B8BCC8; font-size:0.9rem;'>Regulation Impact Analysis</p>",
        unsafe_allow_html=True
    )
    st.divider()

    page = st.radio(
        "Navigate to",
        options=[
            "🏠 Home",
            "🏁 Circuit Impact",
            "👤 Driver Analysis",
            "🔬 Model Insights",
            "✅ Simulation Validation"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    if not sim_df.empty and "lap_time_change_seconds" in sim_df.columns:
        st.markdown(
            "<p style='color:#9AA1B2; font-size:0.75rem;'>SIMULATION COVERAGE</p>",
            unsafe_allow_html=True
        )
        st.metric("Circuits", len(sim_df))
        st.metric(
            "Avg Predicted Change",
            f"{safe_float(sim_df['lap_time_change_seconds'].mean()):+.3f}s"
        )
        faster = int((sim_df["lap_time_change_seconds"] < 0).sum())
        st.metric("Circuits Faster", f"{faster}/{len(sim_df)}")

    st.divider()

    if model_meta:
        metrics = model_meta.get("metrics", {})
        rmse = metrics.get("rmse")
        st.markdown(
            "<p style='color:#9AA1B2; font-size:0.75rem;'>MODEL PERFORMANCE</p>",
            unsafe_allow_html=True
        )
        st.metric("Best Model", model_meta.get("model_name", "LightGBM"))
        if rmse is not None:
            st.metric("Test RMSE", f"{rmse:.4f}s")

# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":
    st.markdown(
        """
        <div class='hero-card'>
            <h1 style='color:#FF1744; font-size:2.5rem; margin:0;'>🏎️ F1 2026 Regulation Impact Analysis</h1>
            <p style='color:#E6EAF2; font-size:1.2rem; margin-top:10px;'>
                Predictive Modeling of Formula 1 Lap Time Dynamics and 2026 Regulation Impact
            </p>
            <p style='color:#AEB6C6; font-size:0.95rem; margin-top:8px;'>
                End-to-end data pipeline, machine learning, explainability, and regulation simulation.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not sim_df.empty and "lap_time_change_seconds" in sim_df.columns:
        col1, col2, col3, col4, col5 = st.columns(5)
        fastest_circuit = sim_df.iloc[0]
        n_faster = int((sim_df["lap_time_change_seconds"] < 0).sum())

        with col1:
            st.metric("Circuits Analyzed", len(sim_df))
        with col2:
            st.metric("Seasons of Data", "2023-2025")
        with col3:
            st.metric(
                "Most Improved Circuit",
                fastest_circuit["circuit"],
                delta=f"{safe_float(fastest_circuit['lap_time_change_seconds']):+.3f}s",
                delta_color="inverse"
            )
        with col4:
            st.metric("Circuits Faster in 2026", f"{n_faster}/{len(sim_df)}")
        with col5:
            rmse = model_meta.get("metrics", {}).get("rmse") if model_meta else None
            st.metric("Model RMSE", f"{rmse:.4f}s" if rmse is not None else "N/A")

    st.divider()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("<div class='section-header'>Project Overview</div>", unsafe_allow_html=True)
        st.markdown("""
        This project builds a complete end-to-end data science pipeline that:

        1. Ingests Formula 1 telemetry and lap data into Bronze → Silver → Gold MySQL layers  
        2. Engineers race-physics features such as fuel weight, tire degradation, and grip  
        3. Trains LightGBM / XGBoost / Random Forest with temporal holdout validation  
        4. Uses SHAP for explainability  
        5. Simulates 2026 regulation effects on circuit lap times  
        6. Validates against early 2026 race data  
        """)

        st.markdown(
            "<div class='section-header' style='margin-top:16px;'>2026 Regulation Changes Modeled</div>",
            unsafe_allow_html=True
        )
        reg_col1, reg_col2, reg_col3 = st.columns(3)
        with reg_col1:
            st.markdown(
                "<div class='metric-card'>"
                "<h4 style='color:#E8002D; margin:0;'>⚖️ -30kg</h4>"
                "<p style='font-size:0.85rem; margin:4px 0 0;'>Minimum car weight reduction</p>"
                "</div>",
                unsafe_allow_html=True
            )
        with reg_col2:
            st.markdown(
                "<div class='metric-card'>"
                "<h4 style='color:#E8002D; margin:0;'>⚡ 50% Electric</h4>"
                "<p style='font-size:0.85rem; margin:4px 0 0;'>Power split from ~20% to 50% electric</p>"
                "</div>",
                unsafe_allow_html=True
            )
        with reg_col3:
            st.markdown(
                "<div class='metric-card'>"
                "<h4 style='color:#E8002D; margin:0;'>🔋 Less Combustion</h4>"
                "<p style='font-size:0.85rem; margin:4px 0 0;'>Reduced peak combustion output</p>"
                "</div>",
                unsafe_allow_html=True
            )

    with col_right:
        st.markdown("<div class='section-header'>Technology Stack</div>", unsafe_allow_html=True)
        tech_df = pd.DataFrame({
            "Layer": [
                "Data Collection", "Data Storage", "Processing",
                "Machine Learning", "Explainability",
                "Experiment Tracking", "Dashboard"
            ],
            "Technology": [
                "FastF1 API", "MySQL 8.0", "Pandas / NumPy",
                "LightGBM / XGBoost", "SHAP",
                "MLflow", "Streamlit / Plotly"
            ]
        })
        st.dataframe(tech_df, hide_index=True, use_container_width=True)

        st.markdown("<div class='section-header'>Data Pipeline Architecture</div>", unsafe_allow_html=True)
        st.markdown("""
        ```
        FastF1 API
             ↓
        MySQL Bronze (Raw)
             ↓
        MySQL Silver (Clean)
             ↓
        MySQL Gold (Features)
             ↓
        LightGBM Model
             ↓
        2026 Simulation
             ↓
        Streamlit Dashboard
        ```
        """)

    st.divider()
    st.markdown("<div class='section-header'>Key Finding — Circuit Impact Preview</div>", unsafe_allow_html=True)

    if not sim_df.empty and "lap_time_change_seconds" in sim_df.columns:
        fig = px.bar(
            sim_df.sort_values("lap_time_change_seconds"),
            x="lap_time_change_seconds",
            y="circuit",
            orientation="h",
            color="lap_time_change_seconds",
            color_continuous_scale=["#0067FF", "#808080", "#E8002D"],
            color_continuous_midpoint=0,
            labels={
                "lap_time_change_seconds": "Predicted Lap Time Change (s)",
                "circuit": "Circuit"
            }
        )
        fig.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            plot_bgcolor="rgba(20,22,30,0.92)",
            paper_bgcolor="rgba(20,22,30,0)",
            font=dict(color="#F5F7FA"),
            height=420
        )
        fig.add_vline(x=0, line_color=F1_WHITE, line_width=1.5)

        st.markdown("<div class='content-panel'>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "<div class='insight-box'>🔵 Weight reduction is the dominant effect across circuits, "
            "while combustion penalties partially offset gains on more power-sensitive tracks.</div>",
            unsafe_allow_html=True
        )

# ============================================================
# CIRCUIT IMPACT
# ============================================================

elif page == "🏁 Circuit Impact":
    st.markdown("<h2 class='section-header'>🏁 Circuit Impact Analysis</h2>", unsafe_allow_html=True)
    st.markdown(
        "Explore the predicted impact of 2026 regulations on each circuit. "
        "Select a circuit to inspect detailed contributions and compare it against the full field."
    )

    if sim_df.empty:
        st.error("No simulation data available.")
        st.stop()

    selected_circuit = st.selectbox("Select Circuit", options=sim_df["circuit"].tolist())
    circuit_row = sim_df[sim_df["circuit"] == selected_circuit].iloc[0]

    change = safe_float(circuit_row.get("lap_time_change_seconds"))
    weight_eff = safe_float(circuit_row.get("weight_effect_seconds"))
    power_eff = safe_float(
        circuit_row.get(
            "combustion_effect_seconds",
            circuit_row.get("power_effect_seconds", 0.0)
        )
    )
    elec_eff = safe_float(circuit_row.get("electric_effect_seconds"))
    power_sens = safe_float(circuit_row.get("power_sensitivity_score"), default=0.5)

    impact = get_impact_label(change)
    color = get_impact_color(change)

    st.markdown(
        f"""
        <div class='dark-card' style='border-top:4px solid {color};'>
            <h3 style='color:{color}; margin:0;'>{selected_circuit} — {impact}</h3>
            <p style='color:#D9E0EA;'>Predicted lap time change: <b>{change:+.4f}s</b></p>
            <p style='color:#9AA1B2; font-size:0.85rem;'>Power sensitivity: {power_sens:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Change", f"{change:+.4f}s")
    with m2:
        st.metric("Weight Effect", f"{weight_eff:+.4f}s")
    with m3:
        st.metric("Power Unit Effect", f"{power_eff:+.4f}s")
    with m4:
        st.metric("Electric Effect", f"{elec_eff:+.4f}s")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        sorted_sim = sim_df.sort_values("lap_time_change_seconds")
        fig_all = go.Figure()
        fig_all.add_trace(go.Bar(
            x=sorted_sim["lap_time_change_seconds"],
            y=sorted_sim["circuit"],
            orientation="h",
            marker_color=[F1_BLUE if v < 0 else F1_RED for v in sorted_sim["lap_time_change_seconds"]],
            hovertemplate="<b>%{y}</b><br>Change: %{x:+.4f}s<extra></extra>"
        ))
        fig_all.add_vline(x=0, line_color=F1_WHITE, line_width=1.5)
        fig_all.update_layout(
            plot_bgcolor="rgba(20,22,30,0.92)",
            paper_bgcolor="rgba(20,22,30,0)",
            font=dict(color="#F5F7FA"),
            height=420
        )

        st.markdown("<div class='content-panel'>", unsafe_allow_html=True)
        st.plotly_chart(fig_all, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col2:
        effects = {
            "Weight Reduction": weight_eff,
            "Power Unit Change": power_eff,
            "Electric Benefit": elec_eff
        }
        fig_decomp = go.Figure()
        fig_decomp.add_trace(go.Bar(
            x=list(effects.keys()),
            y=list(effects.values()),
            marker_color=[F1_BLUE if v < 0 else F1_RED for v in effects.values()],
            hovertemplate="<b>%{x}</b><br>Effect: %{y:+.4f}s<extra></extra>"
        ))
        fig_decomp.add_hline(y=0, line_color=F1_WHITE, line_width=1.5)
        fig_decomp.update_layout(
            plot_bgcolor="rgba(20,22,30,0.92)",
            paper_bgcolor="rgba(20,22,30,0)",
            font=dict(color="#F5F7FA"),
            height=420
        )

        st.markdown("<div class='content-panel'>", unsafe_allow_html=True)
        st.plotly_chart(fig_decomp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if power_sens > 0.7:
        insight = (
            f"**{selected_circuit}** is a high power-sensitivity circuit, so combustion losses "
            f"offset more of the weight benefit here."
        )
    elif power_sens < 0.4:
        insight = (
            f"**{selected_circuit}** is a lower power-sensitivity circuit, so it benefits more "
            f"from the electric torque and weight combination."
        )
    else:
        insight = (
            f"**{selected_circuit}** is relatively balanced, with weight reduction still the "
            f"dominant performance effect."
        )

    st.markdown(f"<div class='insight-box'>🔵 {insight}</div>", unsafe_allow_html=True)

    display_cols = [
        "circuit",
        "lap_time_change_seconds",
        "weight_effect_seconds",
        "combustion_effect_seconds",
        "power_effect_seconds",
        "electric_effect_seconds",
        "power_sensitivity_score"
    ]
    available_cols = [c for c in display_cols if c in sim_df.columns]
    display_df = sim_df[available_cols].copy()

    # handle fallback naming
    if "combustion_effect_seconds" not in display_df.columns and "power_effect_seconds" in display_df.columns:
        display_df = display_df.rename(columns={"power_effect_seconds": "combustion_effect_seconds"})

    display_df = display_df.rename(columns={
        "circuit": "Circuit",
        "lap_time_change_seconds": "Total Change (s)",
        "weight_effect_seconds": "Weight Effect (s)",
        "combustion_effect_seconds": "Power Effect (s)",
        "electric_effect_seconds": "Electric Effect (s)",
        "power_sensitivity_score": "Power Sensitivity"
    })

    st.markdown("#### Complete Circuit Results Table")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ============================================================
# DRIVER ANALYSIS
# ============================================================

elif page == "👤 Driver Analysis":
    st.markdown("<h2 class='section-header'>👤 Driver Analysis</h2>", unsafe_allow_html=True)
    st.markdown(
        "Compare drivers using their 2025 performance signatures. "
        "Driver skill scores are computed from lap-time deltas relative to session medians."
    )

    if driver_df.empty:
        st.warning("Driver data not available.")
        st.stop()

    all_drivers = sorted(driver_df["driver"].unique().tolist())
    all_circuits_d = sorted(driver_df["circuit"].unique().tolist())

    col_d1, col_d2, col_d3 = st.columns([2, 2, 1])
    with col_d1:
        driver_1 = st.selectbox("Driver 1", options=all_drivers, index=0)
    with col_d2:
        driver_2 = st.selectbox("Driver 2", options=all_drivers, index=min(1, len(all_drivers)-1))
    with col_d3:
        compare_circuit = st.selectbox("Circuit", options=["All Circuits"] + all_circuits_d, index=0)

    st.markdown(
        "<div class='insight-box'>🔵 Driver skill score is target-encoded from 2025 relative lap performance.</div>",
        unsafe_allow_html=True
    )

    driver_skills = driver_df.groupby(["driver", "team"])["skill_score"].mean().reset_index().sort_values("skill_score")
    fig_skills = go.Figure()
    fig_skills.add_trace(go.Bar(
        x=driver_skills["driver"],
        y=driver_skills["skill_score"],
        marker_color=[F1_RED if d in [driver_1, driver_2] else "#888888" for d in driver_skills["driver"]],
        hovertemplate="<b>%{x}</b><br>Skill Score: %{y:+.4f}s<extra></extra>"
    ))
    fig_skills.add_hline(y=0, line_color=F1_WHITE, line_dash="dash")
    fig_skills.update_layout(
        plot_bgcolor="rgba(20,22,30,0.92)",
        paper_bgcolor="rgba(20,22,30,0)",
        font=dict(color="#F5F7FA"),
        height=340
    )

    st.markdown("<div class='content-panel'>", unsafe_allow_html=True)
    st.plotly_chart(fig_skills, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if compare_circuit != "All Circuits":
        d1_data = driver_df[(driver_df["driver"] == driver_1) & (driver_df["circuit"] == compare_circuit)]
        d2_data = driver_df[(driver_df["driver"] == driver_2) & (driver_df["circuit"] == compare_circuit)]
    else:
        d1_data = driver_df[driver_df["driver"] == driver_1]
        d2_data = driver_df[driver_df["driver"] == driver_2]

    def render_driver_card(data, driver_name, col):
        with col:
            if data.empty:
                st.warning(f"No data for {driver_name}")
                return

            team = data["team"].mode().iloc[0]
            skill = safe_float(data["skill_score"].mean())
            avg_weight = safe_float(data["avg_weight"].mean())
            avg_grip = safe_float(data["avg_grip"].mean())
            n_laps = int(data["n_laps"].sum())
            color = F1_BLUE if skill < 0 else F1_RED

            st.markdown(
                f"""
                <div class='dark-card' style='border-top:4px solid {color};'>
                    <h3 style='color:{color}; margin-bottom:8px;'>{driver_name}</h3>
                    <p style='color:#BFC7D5; margin-top:0;'>{team}</p>
                    <hr style='border-color:rgba(255,255,255,0.1);'>
                    <p><b>Skill Score:</b> {skill:+.4f}s</p>
                    <p><b>Avg Car Weight:</b> {avg_weight:.1f}kg</p>
                    <p><b>Avg Tire Grip:</b> {avg_grip:.4f}</p>
                    <p><b>Laps Analyzed:</b> {n_laps}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    c1, c2 = st.columns(2)
    render_driver_card(d1_data, driver_1, c1)
    render_driver_card(d2_data, driver_2, c2)

    if compare_circuit == "All Circuits":
        st.markdown("#### Performance by Circuit — Both Drivers")

        d1_by_circuit = driver_df[driver_df["driver"] == driver_1].groupby("circuit")["skill_score"].mean().reset_index()
        d1_by_circuit["driver"] = driver_1

        d2_by_circuit = driver_df[driver_df["driver"] == driver_2].groupby("circuit")["skill_score"].mean().reset_index()
        d2_by_circuit["driver"] = driver_2

        combined = pd.concat([d1_by_circuit, d2_by_circuit], ignore_index=True)

        if not combined.empty:
            fig_circuit_comp = px.bar(
                combined,
                x="circuit",
                y="skill_score",
                color="driver",
                barmode="group",
                color_discrete_map={driver_1: F1_RED, driver_2: F1_BLUE},
                labels={
                    "skill_score": "Skill Score (s from median)",
                    "circuit": "Circuit",
                    "driver": "Driver"
                }
            )
            fig_circuit_comp.add_hline(y=0, line_color=F1_WHITE, line_width=1.5, line_dash="dash")
            fig_circuit_comp.update_layout(
                plot_bgcolor="rgba(20,22,30,0.92)",
                paper_bgcolor="rgba(20,22,30,0)",
                font=dict(color="#F5F7FA"),
                height=350,
                xaxis=dict(tickangle=30)
            )

            st.markdown("<div class='content-panel'>", unsafe_allow_html=True)
            st.plotly_chart(fig_circuit_comp, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# MODEL INSIGHTS
# ============================================================

elif page == "🔬 Model Insights":
    st.markdown("<h2 class='section-header'>🔬 Model Insights</h2>", unsafe_allow_html=True)

    st.markdown(
        "<div class='insight-box'>"
        "🔬 This page explains <b>why the model can be trusted</b>. "
        "It combines holdout performance, experiment tracking, and SHAP-based "
        "feature explanations to verify that the model learned meaningful F1 physics "
        "rather than accidental correlations."
        "</div>",
        unsafe_allow_html=True
    )

    if model_meta:
        metrics = model_meta.get("metrics", {})
        p1, p2, p3, p4, p5 = st.columns(5)
        with p1:
            st.metric("Best Model", model_meta.get("model_name", "N/A"))
        with p2:
            rmse = metrics.get("rmse")
            st.metric("Test RMSE", f"{safe_float(rmse):.4f}s" if rmse is not None else "N/A")
        with p3:
            mae = metrics.get("mae")
            st.metric("Test MAE", f"{safe_float(mae):.4f}s" if mae is not None else "N/A")
        with p4:
            r2 = metrics.get("r2")
            st.metric("R² Score", f"{safe_float(r2):.4f}" if r2 is not None else "N/A")
        with p5:
            cv = metrics.get("cv_rmse_mean")
            if cv is None:
                cv = model_meta.get("cv_results", {}).get("cv_rmse_mean")
            st.metric("CV RMSE", f"{safe_float(cv):.4f}s" if cv is not None else "N/A")

    st.markdown(
        "<div class='warning-box'>"
        "⚠️ <b>Known limitation:</b> Monaco is systematically harder to model than other circuits. "
        "Its street-circuit layout, traffic sensitivity, and incident-driven variability make it "
        "less predictable from standard lap-level features. Monaco simulation outputs should therefore "
        "be interpreted with more caution than circuits such as Bahrain, Monza, or Spa."
        "</div>",
        unsafe_allow_html=True
    )

    if mlflow_data and mlflow_data.get("all_models"):
        models_df = pd.DataFrame(mlflow_data["all_models"])
        best_model_type = mlflow_data.get("best_model", {}).get("model_type")

        fig_comp = go.Figure()
        for _, row in models_df.iterrows():
            fig_comp.add_trace(go.Bar(
                x=[row["model_type"]],
                y=[safe_float(row.get("rmse"))],
                marker_color=F1_RED if row["model_type"] == best_model_type else "#BDBDBD",
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['model_type']}</b><br>"
                    f"RMSE: {safe_float(row.get('rmse')):.4f}s<br>"
                    f"R²: {safe_float(row.get('r2')):.4f}<extra></extra>"
                )
            ))
        fig_comp.add_hline(y=0.90, line_dash="dash", line_color="orange")
        fig_comp.update_layout(
            plot_bgcolor="rgba(20,22,30,0.92)",
            paper_bgcolor="rgba(20,22,30,0)",
            font=dict(color="#F5F7FA"),
            height=340,
            yaxis=dict(title="RMSE (seconds)")
        )

        st.markdown("<div class='content-panel'>", unsafe_allow_html=True)
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='insight-box'>"
        "📊 <b>Model selection insight:</b> Three models were trained and evaluated under temporal holdout "
        "validation (train on 2023–2024, test on 2025). LightGBM was selected because it delivered the lowest "
        "test RMSE while preserving strong interpretability through SHAP."
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='insight-box'>"
        "🧠 <b>How to read SHAP:</b> SHAP values show how much each feature contributes to a prediction in seconds. "
        "Higher mean absolute SHAP values indicate that the feature has a stronger average influence on lap-time estimates."
        "</div>",
        unsafe_allow_html=True
    )

    if shap_data and shap_data.get("global_feature_importance"):
        imp_df = pd.DataFrame(
            list(shap_data["global_feature_importance"].items()),
            columns=["Feature", "Mean |SHAP| (s)"]
        ).sort_values("Mean |SHAP| (s)", ascending=True)

        fig_shap = go.Figure()
        fig_shap.add_trace(go.Bar(
            x=imp_df["Mean |SHAP| (s)"],
            y=imp_df["Feature"],
            orientation="h",
            marker_color=F1_RED,
            hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}s<extra></extra>"
        ))
        fig_shap.update_layout(
            plot_bgcolor="rgba(20,22,30,0.92)",
            paper_bgcolor="rgba(20,22,30,0)",
            font=dict(color="#F5F7FA"),
            height=600
        )

        st.markdown("<div class='content-panel'>", unsafe_allow_html=True)
        st.plotly_chart(fig_shap, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("SHAP summary not found. Run the explainability pipeline to populate this section.")

    st.markdown(
        "<div class='insight-box'>"
        "🏎️ <b>Physics consistency check:</b> Weight-related features rank near the top, confirming that the model learned "
        "a realistic relationship between vehicle mass and lap time. This is important because the 2026 simulation depends "
        "heavily on feature-level transformations to car weight, combustion performance, and electric deployment."
        "</div>",
        unsafe_allow_html=True
    )

    plot_tab1, plot_tab2 = st.tabs(["SHAP Beeswarm", "Circuit Heatmap"])

    with plot_tab1:
            beeswarm_path = get_plot_path("shap_beeswarm.png")
            if beeswarm_path:
                # Changed use_container_width=True to use_column_width=True
                st.image(beeswarm_path, use_column_width=True)
                st.markdown(
                    "<div class='insight-box'>🔵 <b>Reading the beeswarm:</b> Each point is one lap prediction. "
                    "The horizontal spread shows SHAP contribution in seconds. Red indicates high feature values; "
                    "blue indicates low values.</div>",
                    unsafe_allow_html=True
                )
            else:
                st.info("SHAP beeswarm plot not found.")

    with plot_tab2:
        heatmap_path = get_plot_path("shap_circuit_heatmap.png")
        if heatmap_path:
            # Changed use_container_width=True to use_column_width=True
            st.image(heatmap_path, use_column_width=True)
        else:
            st.info("Circuit heatmap not found.")

    if not gold_df.empty:
        st.markdown("#### Training Data Coverage")
        pivot_df = gold_df[gold_df["data_split"].isin(["train", "test"])].groupby(
            ["circuit", "season"]
        )["n_laps"].sum().reset_index()

        if not pivot_df.empty:
            pivot_table = pivot_df.pivot(
                index="circuit", columns="season", values="n_laps"
            ).fillna(0).astype(int)

            st.dataframe(pivot_table, use_container_width=True)
            st.caption("Number of clean laps per circuit and season used in the Gold modeling dataset.")

# ============================================================
# VALIDATION
# ============================================================

elif page == "✅ Simulation Validation":
    st.markdown("<h2 class='section-header'>✅ Simulation Validation</h2>", unsafe_allow_html=True)

    st.markdown(
        "<div class='insight-box'>"
        "✅ This validation compares the <b>pre-2026 simulation outputs</b> against the first available "
        "real 2026 race data. It is the strongest realism check in the project because it evaluates whether "
        "a scenario model trained on historical data generalizes to a new regulation era."
        "</div>",
        unsafe_allow_html=True
    )

    if val_df.empty:
        st.warning("No validation data available.")
        st.stop()

    avg_error = val_df["prediction_error"].abs().mean() if "prediction_error" in val_df.columns else None
    avg_error_pct = None
    if "prediction_error_pct" in val_df.columns and val_df["prediction_error_pct"].notna().any():
        avg_error_pct = val_df["prediction_error_pct"].abs().mean()

    v1, v2, v3, v4 = st.columns(4)
    with v1:
        st.metric("Circuits Validated", val_df["circuit"].nunique())
    with v2:
        st.metric("Mean Prediction Error", f"{avg_error:.4f}s" if avg_error is not None else "N/A")
    with v3:
        st.metric("Mean Error %", f"{avg_error_pct:.2f}%" if avg_error_pct is not None else "N/A")
    with v4:
        st.metric("2026 Races Used", ", ".join(sorted(val_df["circuit"].unique())))

    if not val_df.empty and {"circuit", "simulated_change", "actual_2026", "prediction_error"}.issubset(val_df.columns):
        aus_row = val_df[val_df["circuit"] == "Australia"]
        chn_row = val_df[val_df["circuit"] == "China"]
        notes = []

        if not aus_row.empty:
            aus_err = safe_float(aus_row["prediction_error"].iloc[0])
            notes.append(f"Australia fallback validation error: {aus_err:+.4f}s")

        if not chn_row.empty:
            chn_err = safe_float(chn_row["prediction_error"].iloc[0])
            notes.append(f"China standard-pipeline validation error: {chn_err:+.4f}s")

        note_text = " | ".join(notes) if notes else "Validation rows loaded successfully."

        st.markdown(
            f"<div class='insight-box'>"
            f"📌 <b>Validation takeaway:</b> {note_text}. "
            f"The simulator performs better on Australia after fallback damping, while China remains the clearest example "
            f"of regime-shift error between historical training patterns and real 2026 race conditions."
            f"</div>",
            unsafe_allow_html=True
        )

    if {"simulated_change", "actual_2026"}.issubset(val_df.columns):
        fig_val = go.Figure()
        for _, row in val_df.iterrows():
            fig_val.add_trace(go.Bar(
                x=[f"{row['circuit']} Sim"],
                y=[row["simulated_change"]],
                marker_color=F1_BLUE,
                showlegend=False,
                hovertemplate=f"<b>{row['circuit']} Simulated</b><br>%{{y:+.4f}}s<extra></extra>"
            ))
            fig_val.add_trace(go.Bar(
                x=[f"{row['circuit']} Actual"],
                y=[row["actual_2026"]],
                marker_color=F1_RED,
                showlegend=False,
                hovertemplate=f"<b>{row['circuit']} Actual</b><br>%{{y:+.4f}}s<extra></extra>"
            ))
        fig_val.add_hline(y=0, line_color=F1_WHITE)
        fig_val.update_layout(
            barmode="group",
            plot_bgcolor="rgba(20,22,30,0.92)",
            paper_bgcolor="rgba(20,22,30,0)",
            font=dict(color="#F5F7FA"),
            height=380,
            yaxis=dict(title="Lap Time Delta Change (s)")
        )

        st.markdown("<div class='content-panel'>", unsafe_allow_html=True)
        st.plotly_chart(fig_val, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='warning-box'>"
        "⚠️ <b>Transparency note:</b> The simulation was designed using pre-2026 data only. "
        "Validation against Australia and China 2026 was added after race data became available. "
        "Any remaining mismatch reflects genuine uncertainty in forecasting a major regulation change."
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("#### Validation Results Table")

    val_display = val_df.copy()

    if "fallback_used" in val_display.columns:
        val_display["fallback_used"] = val_display["fallback_used"].map(
            lambda x: "Yes" if x is True else "No" if x is False else "N/A"
        )

    if "prediction_error_pct" in val_display.columns:
        val_display["prediction_error_pct"] = val_display["prediction_error_pct"].fillna("N/A")

    format_map = {}
    for col in val_display.select_dtypes(include="number").columns:
        if col == "prediction_error_pct":
            format_map[col] = "{:+.2f}%"
        elif col == "n_real_laps":
            format_map[col] = "{:.0f}"
        else:
            format_map[col] = "{:+.4f}"

    st.dataframe(
        val_display.style.format(format_map),
        use_container_width=True,
        hide_index=True
    )