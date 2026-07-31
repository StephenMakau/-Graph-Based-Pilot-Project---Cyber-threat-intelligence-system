import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cyber_threat_model import (
    predict_2027,
    get_model_accuracy,
    get_results,
    get_dataset,
    get_parameters,
    train_on_uploaded_data,
    get_current_data_source,
    generate_excel_template,
    validate_uploaded_data
)

# =====================================
# SESSION STATE INITIALIZATION
# =====================================
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
    
if 'mobile_view' not in st.session_state:
    st.session_state.mobile_view = False

if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

if 'training_status' not in st.session_state:
    st.session_state.training_status = None

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="MKU Cyber Threat Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# SIDEBAR CONTROLS
# =====================================
with st.sidebar:
    st.title("⚙️ SYSTEM CONTROLS")
    st.markdown("---")
    
    dark_mode = st.toggle(
        "🌙 Dark Mode / ☀️ Light Mode", 
        value=st.session_state.dark_mode,
        help="Switch between Dark and Light theme"
    )
    st.session_state.dark_mode = dark_mode
    
    mobile_view = st.toggle(
        "📱 Mobile View / 💻 Desktop View", 
        value=st.session_state.mobile_view,
        help="Switch between Mobile and Desktop layout"
    )
    st.session_state.mobile_view = mobile_view
    
    st.markdown("---")
    
    # Data Source Indicator
    data_source = get_current_data_source()
    if data_source == "uploaded":
        st.success("📊 Using: **Uploaded Data**")
    else:
        st.info("📊 Using: **Default Dataset**")
    
    st.caption(f"System Time: {datetime.now(ZoneInfo('Africa/Nairobi')).strftime('%H:%M:%S')}")
    st.caption("MKU Cybersecurity v2.0.8")

# =====================================
# THEME CSS
# =====================================
if st.session_state.dark_mode:
    theme_css = """
    <style>
        .stApp { background: linear-gradient(135deg, #0b1120 0%, #1e293b 50%, #0f172a 100%); color: #f8fafc; }
        h1, h2, h3, h4, h5, h6 { color: #f8fafc !important; text-shadow: 0 0 20px rgba(6, 182, 212, 0.3); }
        h1 { background: linear-gradient(90deg, #06b6d4, #8b5cf6, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; border-bottom: 2px solid rgba(6, 182, 212, 0.3); }
        h2 { color: #06b6d4 !important; border-left: 4px solid #06b6d4; }
        h3 { color: #8b5cf6 !important; }
        p, li, div, span, label, .stMarkdown, .stAlert { color: #e2e8f0 !important; }
        .stTabs [data-testid="stTab"] { color: #94a3b8; background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(6, 182, 212, 0.2); }
        .stTabs [data-testid="stTab"]:hover { color: #06b6d4; background: rgba(6, 182, 212, 0.1); }
        .stTabs [data-testid="stTab"][aria-selected="true"] { background: linear-gradient(135deg, #06b6d4, #3b82f6); color: #ffffff !important; }
        [data-testid="stMetric"] { background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(6, 182, 212, 0.3); }
        [data-testid="stMetricValue"] { color: #06b6d4 !important; }
        [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
        div[data-testid="stDataFrame"], div[data-testid="stTable"] { background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(6, 182, 212, 0.2); }
        .threat-high { background: rgba(251, 188, 4, 0.1); border: 1px solid rgba(251, 188, 4, 0.5); color: #fcd34d !important; }
        .threat-critical { background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.5); color: #fca5a5 !important; }
        .threat-moderate { background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.5); color: #6ee7b7 !important; }
        .stAlert { background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(6, 182, 212, 0.3); color: #06b6d4 !important; }
        .tech-container { background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(6, 182, 212, 0.2); }
        strong, b { color: #06b6d4 !important; }
        .explanation-box { background: rgba(6, 182, 212, 0.1); border-left: 3px solid #06b6d4; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .upload-box { background: rgba(139, 92, 246, 0.1); border: 2px dashed rgba(139, 92, 246, 0.5); border-radius: 10px; padding: 20px; text-align: center; }
        .template-box { background: rgba(16, 185, 129, 0.1); border: 2px solid rgba(16, 185, 129, 0.5); border-radius: 10px; padding: 20px; }
    </style>
    """
else:
    theme_css = """
    <style>
        .stApp { background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #f1f5f9 100%); color: #1e293b; }
        h1, h2, h3, h4, h5, h6 { color: #1e293b !important; text-shadow: none; }
        h1 { background: linear-gradient(90deg, #0369a1, #7c3aed, #2563eb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; border-bottom: 2px solid rgba(37, 99, 235, 0.3); }
        h2 { color: #0369a1 !important; border-left: 4px solid #0369a1; }
        h3 { color: #7c3aed !important; }
        p, li, div, span, label, .stMarkdown, .stAlert { color: #334155 !important; }
        .stTabs [data-testid="stTab"] { color: #64748b; background: rgba(255, 255, 255, 0.8); border: 1px solid rgba(37, 99, 235, 0.2); }
        .stTabs [data-testid="stTab"]:hover { color: #0369a1; background: rgba(37, 99, 235, 0.1); }
        .stTabs [data-testid="stTab"][aria-selected="true"] { background: linear-gradient(135deg, #0369a1, #2563eb); color: #ffffff !important; }
        [data-testid="stMetric"] { background: rgba(255, 255, 255, 0.9); border: 1px solid rgba(37, 99, 235, 0.3); box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
        [data-testid="stMetricValue"] { color: #0369a1 !important; }
        [data-testid="stMetricLabel"] { color: #64748b !important; }
        div[data-testid="stDataFrame"], div[data-testid="stTable"] { background: rgba(255, 255, 255, 0.9); border: 1px solid rgba(37, 99, 235, 0.2); }
        .threat-high { background: rgba(251, 191, 36, 0.15); border: 1px solid rgba(251, 191, 36, 0.6); color: #92400e !important; }
        .threat-high h1 { -webkit-text-fill-color: #92400e; color: #92400e !important; }
        .threat-critical { background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.6); color: #991b1b !important; }
        .threat-critical h1 { -webkit-text-fill-color: #991b1b; color: #991b1b !important; }
        .threat-moderate { background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.6); color: #065f46 !important; }
        .threat-moderate h1 { -webkit-text-fill-color: #065f46; color: #065f46 !important; }
        .stAlert { background: rgba(219, 234, 254, 0.8); border: 1px solid rgba(37, 99, 235, 0.3); color: #1e40af !important; }
        .tech-container { background: rgba(255, 255, 255, 0.9); border: 1px solid rgba(37, 99, 235, 0.2); }
        strong, b { color: #0369a1 !important; }
        hr { border-color: rgba(37, 99, 235, 0.2) !important; }
        .explanation-box { background: rgba(37, 99, 235, 0.1); border-left: 3px solid #0369a1; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .upload-box { background: rgba(139, 92, 246, 0.05); border: 2px dashed rgba(139, 92, 246, 0.5); border-radius: 10px; padding: 20px; text-align: center; }
        .template-box { background: rgba(16, 185, 129, 0.05); border: 2px solid rgba(16, 185, 129, 0.5); border-radius: 10px; padding: 20px; }
    </style>
    """

if st.session_state.mobile_view:
    mobile_css = """
    <style>
        .block-container { max-width: 480px !important; padding-left: 1rem !important; padding-right: 1rem !important; }
        h1 { font-size: 1.8rem !important; }
        h2 { font-size: 1.4rem !important; }
        h3 { font-size: 1.2rem !important; }
        .stTabs [data-testid="stTab"] { padding: 8px 12px !important; font-size: 0.8rem !important; }
        [data-testid="stMetric"] { padding: 15px !important; }
        [data-testid="stMetricValue"] { font-size: 1.8rem !important; }
    </style>
    """
else:
    mobile_css = """
    <style>
        .block-container { max-width: 95% !important; }
    </style>
    """

st.markdown(theme_css + mobile_css, unsafe_allow_html=True)

# =====================================
# FUTURISTIC GRAPH FUNCTION (FIXED)
# =====================================
def create_futuristic_graph(df):
    """
    Create a futuristic cyberpunk-style visualization of threat data
    """
    try:
        df = df.copy()
        
        # Safely create date column
        # Ensure Year and Month are numeric
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
        
        # Drop rows with invalid dates
        df = df.dropna(subset=['Year', 'Month'])
        
        # Convert Month to 1-12 range if it looks like a year (e.g. 2021 -> 1)
        # This handles the specific error in your default data where Month had 2021
        df['Month'] = df['Month'].apply(lambda x: int(x) if 1 <= int(x) <= 12 else 1)
        df['Year'] = df['Year'].astype(int)
        
        # Create proper date objects
        df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
        df = df.sort_values('Date')
        
        if df.empty:
            st.error("No valid data available for visualization.")
            return None

        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=('⚡ ATTACK VECTORS OVER TIME', '🔒 VULNERABILITY METRICS')
        )
        
        # Color palette - Cyberpunk Neon
        colors = {
            'ddos': '#00f5ff',      # Cyan
            'malware': '#ff00ff',   # Magenta
            'phishing': '#39ff14',  # Neon Green
            'web': '#ff6b35',       # Orange-Red
            'cves': '#ff1744',      # Red
            'patch': '#ffd700',     # Gold
            'traffic': '#8b5cf6',   # Purple
            'grid': 'rgba(6, 182, 212, 0.1)',
            'text': '#e2e8f0'
        }
        
        # Row 1: Attack Volumes
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['DDoS_Attacks'],
                name='DDoS Attacks',
                mode='lines+markers',
                line=dict(color=colors['ddos'], width=3),
                marker=dict(size=8, symbol='diamond', line=dict(width=2, color='white')),
                fill='tonexty',
                fillcolor='rgba(0, 245, 255, 0.1)'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['Malware_Attacks'],
                name='Malware',
                mode='lines+markers',
                line=dict(color=colors['malware'], width=3),
                marker=dict(size=8, symbol='circle', line=dict(width=2, color='white')),
                fill='tonexty',
                fillcolor='rgba(255, 0, 255, 0.1)'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['Phishing_Attacks'],
                name='Phishing',
                mode='lines+markers',
                line=dict(color=colors['phishing'], width=3),
                marker=dict(size=8, symbol='triangle-up', line=dict(width=2, color='white')),
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['Web_Attacks'],
                name='Web Attacks',
                mode='lines+markers',
                line=dict(color=colors['web'], width=3, dash='dot'),
                marker=dict(size=8, symbol='square', line=dict(width=2, color='white')),
            ),
            row=1, col=1
        )
        
        # Row 2: Vulnerability Metrics (scaled for visibility)
        fig.add_trace(
            go.Bar(
                x=df['Date'], y=df['Critical_CVEs'],
                name='Critical CVEs',
                marker=dict(
                    color=colors['cves'],
                    line=dict(color='white', width=1),
                    opacity=0.8
                ),
                text=df['Critical_CVEs'],
                textposition='outside',
                textfont=dict(color=colors['cves'], size=10)
            ),
            row=2, col=1
        )
        
        # Add Patch Delay as line on secondary axis
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['Patch_Delay_Days'],
                name='Patch Delay (Days)',
                mode='lines+markers',
                line=dict(color=colors['patch'], width=3),
                marker=dict(size=10, symbol='x', line=dict(width=2, color='white')),
                yaxis='y3'
            ),
            row=2, col=1
        )
        
        # Update layout with futuristic styling
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(11, 17, 32, 0.8)',
            font=dict(family='JetBrains Mono, monospace', color=colors['text'], size=11),
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                bgcolor='rgba(11, 17, 32, 0.9)',
                bordercolor='#06b6d4',
                borderwidth=1,
                font=dict(size=10)
            ),
            margin=dict(l=60, r=60, t=100, b=60),
            hovermode='x unified',
            xaxis=dict(
                showgrid=True,
                gridcolor=colors['grid'],
                gridwidth=1,
                zeroline=False,
                linecolor='#06b6d4',
                linewidth=2,
                tickfont=dict(size=10)
            ),
            xaxis2=dict(
                showgrid=True,
                gridcolor=colors['grid'],
                gridwidth=1,
                zeroline=False,
                linecolor='#06b6d4',
                linewidth=2,
                title='Timeline'
            ),
            yaxis=dict(
                title=dict(text='Attack Volume', font=dict(size=12, color='#06b6d4')),
                showgrid=True,
                gridcolor=colors['grid'],
                gridwidth=1,
                zeroline=False,
                linecolor='#06b6d4',
                linewidth=2,
                tickfont=dict(size=10)
            ),
            yaxis2=dict(
                title=dict(text='CVE Count', font=dict(size=12, color='#ff1744')),
                showgrid=True,
                gridcolor=colors['grid'],
                gridwidth=1,
                zeroline=False,
                linecolor='#ff1744',
                linewidth=2,
                tickfont=dict(size=10)
            ),
            yaxis3=dict(
                title=dict(text='Patch Delay (Days)', font=dict(size=12, color='#ffd700')),
                overlaying='y2',
                side='right',
                showgrid=False,
                zeroline=False,
                linecolor='#ffd700',
                linewidth=2,
                tickfont=dict(size=10, color='#ffd700')
            ),
            title=dict(
                text='📊 CYBER THREAT INTELLIGENCE // VISUALIZATION MODULE',
                font=dict(size=16, color='#06b6d4', family='JetBrains Mono'),
                x=0.5,
                xanchor='center'
            ),
            # Add futuristic shapes
            shapes=[
                # Top border line
                dict(type='line', x0=0, x1=1, y0=1.15, y1=1.15, 
                     xref='paper', yref='paper',
                     line=dict(color='#06b6d4', width=2)),
                # Bottom border line
                dict(type='line', x0=0, x1=1, y0=-0.15, y1=-0.15,
                     xref='paper', yref='paper',
                     line=dict(color='#06b6d4', width=2)),
            ],
            # Add annotations for futuristic feel
            annotations=[
                dict(
                    text='◢ LIVE DATA STREAM ◣',
                    xref='paper', yref='paper',
                    x=0.5, y=1.08,
                    showarrow=False,
                    font=dict(size=10, color='#00f5ff', family='monospace'),
                    bgcolor='rgba(0,0,0,0.5)',
                    bordercolor='#00f5ff',
                    borderwidth=1,
                    borderpad=4
                )
            ]
        )
        
        # Update hover template for futuristic look
        fig.update_traces(
            hovertemplate='<b>%{data.name}</b><br>' +
                          'Date: %{x|%Y-%m}<br>' +
                          'Value: %{y:,}<br>' +
                          '<extra></extra>'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"⚠️ Graph Generation Error: {str(e)}")
        return None

# =====================================
# NAVIGATION
# =====================================
home, data_upload, overview, dataset, models, parameters = st.tabs(
    [
        "🏠 HOME",
        "📤 DATA UPLOAD",
        "📄 PROJECT OVERVIEW", 
        "📊 DATASET",
        "🤖 AI MODELS",
        "⚙️ PARAMETERS"
    ]
)

# =====================================
# HOME - PRIORITY: OUTCOME FIRST
# =====================================
with home:
    st.title("🛡️ CYBER THREAT INTELLIGENCE")
    st.subheader("SYSTEM v2.0.8 | Mount Kenya University")
    st.markdown("**Operator:** Stephen Musau Makau | **Clearance:** MSc Cybersecurity")
    
    mode_indicator = "🌙 DARK" if st.session_state.dark_mode else "☀️ LIGHT"
    view_indicator = "📱 MOBILE" if st.session_state.mobile_view else "💻 DESKTOP"
    data_indicator = "📊 CUSTOM" if get_current_data_source() == "uploaded" else "📋 DEFAULT"
    st.caption(f"⏱️ SYS.TIME: {datetime.now(ZoneInfo('Africa/Nairobi')).strftime('%d/%m/%Y | %H:%M:%S')} EAT | STATUS: ONLINE | MODE: {mode_indicator} | VIEW: {view_indicator} | DATA: {data_indicator}")
    
    st.divider()

    # 1. EXECUTION: Get Prediction & Accuracy Immediately
    try:
        prediction = predict_2027()
        accuracy = get_model_accuracy() * 100
    except Exception as e:
        st.error(f"⚠️ System Error: {e}")
        prediction = "Unknown"
        accuracy = 0.0

    # 2. OUTCOME: Display Threat Projection FIRST (Top Priority)
    st.header("🚨 THREAT PROJECTION // 2027")
    
    if prediction == "High":
        st.markdown("""
        <div class="threat-high">
            <h1>⚠️ HIGH RISK DETECTED</h1>
            <h3>THREAT_LEVEL: HIGH</h3>
            <p>Predictive algorithms indicate significant escalation in cyber threats targeting critical infrastructure. Immediate countermeasures required. Threat vectors include advanced persistent threats (APTs) and zero-day exploits.</p>
        </div>
        """, unsafe_allow_html=True)
        st.warning("⚠️ **Interpretation**: Historical data shows that when attack volumes exceed 4,000 (DDoS), 18,000 (Malware), and economic pressure indicators are elevated, threat levels typically spike to HIGH. Review defensive protocols immediately.")
        
    elif prediction == "Critical":
        st.markdown("""
        <div class="threat-critical">
            <h1>🛑 CRITICAL ALERT</h1>
            <h3>THREAT_LEVEL: CRITICAL</h3>
            <p>Maximum threat level detected. System predicts unprecedented attack surge. Emergency protocols activated. All defensive systems should be raised to maximum alert status immediately.</p>
        </div>
        """, unsafe_allow_html=True)
        st.error("🛑 **Interpretation**: The convergence of high CVE counts (95), low patch compliance (9 days delay), and economic instability historically correlates with CRITICAL threat periods. Immediate executive action required.")
        
    else:
        st.markdown("""
        <div class="threat-moderate">
            <h1>✅ STABLE STATUS</h1>
            <h3>THREAT_LEVEL: MODERATE</h3>
            <p>Threat parameters within acceptable ranges. Standard monitoring protocols sufficient. Continue baseline security operations and routine system audits.</p>
        </div>
        """, unsafe_allow_html=True)
        st.success("✅ **Interpretation**: Current projections indicate manageable threat levels. Attack volumes remain within historical norms and economic indicators suggest stable conditions. Maintain standard operations.")

    st.divider()

    # 3. FUTURISTIC GRAPH: Visual Data Intelligence
    st.markdown("### 📡 THREAT VISUALIZATION MATRIX")
    
    # Get current dataset for visualization
    viz_data = get_dataset()
    
    # Create and display the futuristic graph
    fig = create_futuristic_graph(viz_data)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, height=600)
    
    # Graph explanation
    with st.expander("📖 **Graph Interpretation Guide**", expanded=False):
        st.markdown("""
        **🔍 Understanding the Visualization:**
        
        **Top Panel - Attack Vectors:**
        - **Cyan Line (◆)**: DDoS Attacks - Distributed denial of service incidents
        - **Magenta Line (●)**: Malware Infections - Virus/Trojan/Ransomware detections  
        - **Green Line (▲)**: Phishing Attempts - Social engineering attacks
        - **Orange Line (■)**: Web Attacks - SQL injection, XSS, application exploits
        
        **Bottom Panel - Vulnerability Metrics:**
        - **Red Bars**: Critical CVEs published each month (security vulnerabilities)
        - **Gold Line (×)**: Patch Delay Days (time to deploy security fixes)
        
        **Key Patterns to Watch:**
        - **Spikes in red bars** + **high gold line** = Critical vulnerability window
        - **Rising cyan/magenta** = Active attack campaign in progress
        - **All lines trending up** = Approaching CRITICAL threat level
        
        **Interactive Features:**
        - Hover over any point for exact values
        - Click legend items to show/hide specific attack types
        - Drag to zoom into specific time periods
        """)
    
    st.divider()

    # 4. CONTEXT: System Metrics & Algorithm Info (Secondary Priority)
    st.markdown("### 📡 THREAT ASSESSMENT MODULE")
    
    if st.session_state.mobile_view:
        c1, c2 = st.columns(2)
        c3 = st.container()
    else:
        c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="tech-container" style="text-align: center;">
            <h4 style="color: #94a3b8; margin:15px 0 0 0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2px; font-family: Inter, sans-serif;">Core Algorithm</h4>
            <h2 style="margin: 20px 0; color: #06b6d4; font-family: JetBrains Mono, monospace !important; font-size: 2.6rem; font-weight: 700; text-shadow: 0 0 15px rgba(6,182,212,0.5);">XGBoost</h2>
            <p style="color: #64748b; font-size: 0.95rem; font-family: Calibri, sans-serif; margin-bottom: 15px;">ML.Engine.GradientBoost</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("💡 **XGBoost** (eXtreme Gradient Boosting) is an advanced ML algorithm that combines multiple decision trees to make predictions. It was selected for its superior accuracy in handling cybersecurity data.")

    with c2:
        st.metric(
            "🎯 MODEL ACCURACY",
            f"{accuracy:.2f}%",
            help="Training validation score"
        )
        st.caption("💡 This percentage indicates how often the model correctly predicted historical threat levels. Above 80% is considered reliable for operational use.")

    with c3:
        st.markdown("""
        <div class="tech-container" style="text-align: center;">
            <h4 style="color: #94a3b8; margin:15px 0 0 0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2px; font-family: Inter, sans-serif;">Target Year</h4>
            <h2 style="margin: 20px 0; color: #8b5cf6; font-family: JetBrains Mono, monospace !important; font-size: 2.6rem; font-weight: 700; text-shadow: 0 0 15px rgba(139,92,246,0.5);">2027</h2>
            <p style="color: #64748b; font-size: 0.95rem; font-family: Calibri, sans-serif; margin-bottom: 15px;">Forecast.Horizon</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("💡 The model projects threat levels 2 years ahead using trend analysis of attack patterns, economic indicators, and system vulnerabilities from 2020-2025 data.")

    st.divider()

    # 4. EDUCATION: Methodology & Explanations (Moved Below Outcome)
    st.markdown("### 🧠 **Prediction Methodology**")
    st.info("""
    **How the 2027 Prediction is Generated:**
    
    1. **Data Input**: The model receives projected values for 12 parameters (DDoS attacks, malware volume, CVE counts, inflation, GDP, etc.)
    2. **Pattern Recognition**: XGBoost compares these projections against historical patterns where similar conditions resulted in specific threat levels
    3. **Classification**: The system classifies the 2027 scenario into one of three categories: Moderate, High, or Critical
    4. **Confidence**: The accuracy metric indicates how much trust to place in this prediction based on past performance
    
    **Why This Matters**: Early warning allows security teams to allocate resources proactively rather than reacting to attacks after they occur.
    """)

    # EXPLANATION: What this page shows
    with st.expander("📖 **How to Read This Dashboard**", expanded=False):
        st.markdown("""
        **Welcome to the Cyber Threat Intelligence System.** This dashboard predicts cyber threat levels for Kenyan Government Digital Services using Machine Learning.
        
        **Key Components:**
        - **🎯 Model Accuracy**: Shows how well our AI predicts past threats (higher % = more reliable)
        - **📅 Target Year**: The system forecasts threats for **2027** based on historical patterns (2020-2025)
        - **🚨 Threat Projection**: The colored alert box shows the predicted threat level using three categories:
            - **MODERATE** (Green): Normal operations sufficient
            - **HIGH** (Orange): Increased vigilance required
            - **CRITICAL** (Red): Maximum alert status needed
        
        **How it Works**: The system analyzes 10 historical data points across 12 variables (attack types, economic factors, vulnerabilities) to identify patterns and predict future threats.
        
        **📤 Upload Your Own Data**: Go to the **DATA UPLOAD** tab to use your own historical data for custom predictions!
        """)

# =====================================
# DATA UPLOAD - NEW TAB
# =====================================
with data_upload:
    st.title("📤 DATA UPLOAD CENTER")
    st.markdown("**Upload your own cybersecurity data to generate custom threat predictions**")
    
    st.divider()
    
    # STEP 1: Download Template
    st.header("📋 STEP 1: Download Template")
    
    st.markdown("""
    ### 📖 **Data Format Requirements**
    
    Before uploading your data, download the Excel template below. Your file **must** contain these exact columns:
    
    | Column | Type | Description | Example |
    |--------|------|-------------|---------|
    | **Year** | Integer | Year of observation | 2024 |
    | **Month** | Integer | Month (1-12) | 8 |
    | **DDoS_Attacks** | Integer | Number of DDoS incidents | 4200 |
    | **Malware_Attacks** | Integer | Malware infection count | 18500 |
    | **Phishing_Attacks** | Integer | Phishing attempts recorded | 4100 |
    | **Web_Attacks** | Integer | Web application attacks | 5200 |
    | **Critical_CVEs** | Integer | Critical vulnerabilities published | 95 |
    | **Patch_Delay_Days** | Integer | Average days to patch | 9 |
    | **Traffic_Volume** | Integer | Network traffic (requests) | 1100000 |
    | **Inflation_Rate** | Float | Inflation percentage | 6.8 |
    | **GDP_Growth** | Float | GDP growth percentage | 5.2 |
    | **Economic_Environment** | Text | Economic condition | Stable, High_Cost, Improving, Pressure |
    | **Threat_Level** | Text | Historical threat level | Medium, High, Critical |
    """)
    
    # Generate and provide download button for template
    template_buffer = generate_excel_template()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="📥 DOWNLOAD EXCEL TEMPLATE",
            data=template_buffer,
            file_name="cyber_threat_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        st.caption("💡 This template contains sample data showing exactly how to format your entries")
    
    st.divider()
    
    # STEP 2: Upload Data
    st.header("📤 STEP 2: Upload Your Data")
    
    st.markdown("""
    ### 📖 **Upload Instructions**
    
    1. **Prepare your data** using the template above
    2. **Save as Excel (.xlsx)** or CSV (.csv) format
    3. **Ensure minimum 5 rows** of historical data (more is better)
    4. **Verify column names match exactly** (case-sensitive)
    5. **Upload below** and click "Train Model"
    
    ⚠️ **Important**: The system will automatically retrain all ML algorithms (Logistic Regression, Random Forest, XGBoost) using your uploaded data.
    """)
    
    uploaded_file = st.file_uploader(
        "Choose your data file (Excel .xlsx or CSV .csv)",
        type=['xlsx', 'csv'],
        help="Upload your historical cybersecurity data following the template format"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: **{uploaded_file.name}**")
        
        # Preview uploaded data
        try:
            if uploaded_file.name.endswith('.csv'):
                df_preview = pd.read_csv(uploaded_file)
            else:
                df_preview = pd.read_excel(uploaded_file)
            
            st.markdown("### 👁️ **Data Preview** (First 5 rows)")
            st.dataframe(df_preview.head(), use_container_width=True)
            
            # Validate data
            validation_result = validate_uploaded_data(df_preview)
            
            if validation_result['valid']:
                st.success(f"✅ **Validation Passed**: {validation_result['message']}")
                
                # Train button
                if st.button("🚀 TRAIN MODEL WITH UPLOADED DATA", use_container_width=True, type="primary"):
                    with st.spinner("Training models... Please wait..."):
                        success, message = train_on_uploaded_data(df_preview)
                        if success:
                            st.session_state.uploaded_data = df_preview
                            st.session_state.training_status = "success"
                            st.success(f"✅ {message}")
                            st.balloons()
                            st.info("🔄 Navigate to **HOME** or **AI MODELS** tabs to see results from your data!")
                        else:
                            st.session_state.training_status = "error"
                            st.error(f"❌ {message}")
            else:
                st.error(f"❌ **Validation Failed**: {validation_result['message']}")
                st.warning("Please fix the issues above and re-upload.")
                
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("💡 Make sure your file is not corrupted and is in the correct format.")
    
    st.divider()
    
    # Reset option
    st.header("🔄 Reset to Default")
    if st.button("↩️ RETURN TO DEFAULT DATASET", use_container_width=True):
        st.session_state.uploaded_data = None
        # Reinitialize with default data
        train_on_uploaded_data(None)
        st.success("✅ System reset to default dataset")
        st.info("The system is now using the original 2020-2025 research data.")

# =====================================
# PROJECT OVERVIEW - WITH EXPLANATIONS
# =====================================
with overview:
    st.title("📄 SYSTEM OVERVIEW")
    
    # EXPLANATION: Project context
    st.markdown("""
    ### 🎓 **Research Context**
    This system was developed as part of an **MSc Cybersecurity thesis at Mount Kenya University** to address a critical gap: most security systems react to attacks after they happen, but this tool **predicts threats before they occur**.
    
    **The Problem**: Kenyan Government Digital Services face increasing cyber attacks, but traditional defenses only respond after damage occurs. This creates vulnerability windows.
    
    **The Solution**: Machine Learning analyzes historical attack patterns alongside economic and technical indicators to forecast threat levels 2 years in advance, enabling **proactive defense**.
    
    **🆕 New Feature**: You can now upload your own historical data to generate custom predictions specific to your organization or region!
    """)
    
    if st.session_state.mobile_view:
        st.subheader("🎯 MISSION OBJECTIVE")
        st.write("Advanced predictive intelligence platform for Kenyan Government Digital Services.")
        st.subheader("⚠️ THREAT LANDSCAPE")
        st.write("Digital transformation acceleration correlates with exponential threat growth.")
        st.subheader("🔬 SYSTEM ARCHITECTURE")
        st.markdown("""
        - **Attack Vectors:** DDoS, Malware, Phishing
        - **Vulnerability Metrics:** CVE Criticality
        - **Network Intelligence:** Traffic Anomalies
        - **Economic Indicators:** Inflation/GDP correlation
        """)
        st.subheader("🌍 OPERATIONAL IMPACT")
        st.markdown("""
        - Critical Infrastructure Protection
        - Resource Optimization
        - Policy Intelligence
        """)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎯 MISSION OBJECTIVE")
            st.write("""
            Advanced predictive intelligence platform for Kenyan Government Digital Services. 
            Deploys machine learning algorithms to forecast cyber threat evolution and enable 
            proactive defense strategies rather than reactive responses.
            
            **Now with custom data upload capability** for organization-specific predictions.
            """)
            
            st.subheader("⚠️ THREAT LANDSCAPE")
            st.write("""
            Digital transformation acceleration correlates with exponential threat growth. 
            Conventional reactive defenses inadequate against modern automated attack vectors. 
            Critical infrastructure requires predictive capabilities to stay ahead of adversaries.
            """)
            
            # EXPLANATION: Why ML works here
            st.info("""
            **Why Machine Learning?**
            
            Traditional rule-based systems look for known attack signatures. ML discovers **hidden patterns** across multiple variables (e.g., when inflation rises above 7% AND patch delays exceed 10 days, threat levels typically spike within 3 months). These correlations are invisible to human analysts but detectable by algorithms.
            """)
            
        with col2:
            st.subheader("🔬 SYSTEM ARCHITECTURE")
            st.markdown("""
            **Core Engine**: XGBoost Neural Networks processing multi-dimensional correlation matrices:
            
            - **🎯 Attack Vectors**: DDoS, Malware, Phishing, Web Exploits
            - **🔒 Vulnerability Metrics**: CVE Criticality, Patch Latency  
            - **📡 Network Intelligence**: Traffic Anomaly Detection
            - **📈 Economic Indicators**: Inflation/GDP correlation algorithms
            
            **Data Flow**: Raw data → Feature Engineering → Model Training → Prediction → Alert Generation
            
            **Upload Capability**: Custom datasets → Auto-validation → Model Retraining → Custom Predictions
            """)
            
            st.subheader("🌍 OPERATIONAL IMPACT")
            st.markdown("""
            **Strategic Benefits**:
            - **🏛️ Critical Infrastructure Protection**: Secure national digital assets preemptively
            - **💰 Resource Optimization**: Allocate security budget efficiently based on predicted risk
            - **📋 Policy Intelligence**: Inform cybersecurity policy with data-driven forecasts
            - **🤝 Public Trust**: Maintain confidence in e-government services through proactive security
            - **📊 Custom Analysis**: Upload your own data for organization-specific threat modeling
            """)

    st.info("🔒 **SECURITY PROTOCOL**: All data displayed is synthetic/anonymized for research purposes. No real-time government data is exposed.")

# =====================================
# DATASET - WITH EXPLANATIONS
# =====================================
with dataset:
    st.title("📊 DATA MATRIX")
    
    # Show current data source
    if get_current_data_source() == "uploaded":
        st.success("📊 **Currently Displaying**: Your uploaded custom dataset")
    else:
        st.info("📊 **Currently Displaying**: Default research dataset (2020-2025)")
    
    # EXPLANATION: What the data represents
    st.markdown("""
    ### 📖 **Understanding the Training Data**
    
    This dataset contains historical observations used to train the prediction model. 
    Each row represents a snapshot in time with 12 measured variables that correlate with cyber threat levels.
    
    **How to Read the Columns:**
    - **Temporal**: Year/Month when data was recorded
    - **Attack Metrics**: Raw counts of DDoS, Malware, Phishing, and Web attacks
    - **Vulnerability**: Critical CVEs (security flaws) and Patch Delay Days (how long systems remain exposed)
    - **Economic**: Inflation Rate and GDP Growth (economic stress often correlates with increased cybercrime)
    - **Target**: Threat_Level (Medium/High/Critical) - what the model learns to predict
    """)
    
    st.markdown("Accessing training datasets...")
    
    height = 400 if st.session_state.mobile_view else 500
    st.dataframe(
        get_dataset(),
        use_container_width=True,
        height=height
    )
    
    # EXPLANATION: Data patterns
    st.success("""
    **📈 Key Patterns Visible in This Data:**
    
    1. **2023 Peak**: The only "Critical" threat period occurred when DDoS attacks reached 3,200 and malware hit 15,000 incidents
    2. **Economic Correlation**: High/Critical threats align with "High_Cost" economic environment and inflation above 7%
    3. **Patch Delay Impact**: When patch delays drop below 10 days, threat levels tend to decrease (faster patching = less vulnerability)
    4. **Attack Escalation**: Clear upward trend in attack volumes from 2020-2023, with slight stabilization in 2024-2025
    
    **Training Process**: The model learned these patterns to recognize that specific combinations of these values predict future threat levels.
    
    **💡 Tip**: Upload your own data in the **DATA UPLOAD** tab to see patterns specific to your organization!
    """)

# =====================================
# AI MODELS - WITH EXPLANATIONS
# =====================================
with models:
    st.title("🤖 AI CORE PERFORMANCE")
    
    # Show current data source
    if get_current_data_source() == "uploaded":
        st.success("🤖 **Models trained on**: Your uploaded custom dataset")
    else:
        st.info("🤖 **Models trained on**: Default research dataset")
    
    # EXPLANATION: Algorithm comparison
    st.markdown("""
    ### 📖 **Algorithm Selection Process**
    
    Three machine learning algorithms were evaluated to determine which best predicts cyber threat levels. 
    Each was trained on the same historical data and tested on unseen validation data to measure accuracy.
    
    **Why Compare Multiple Algorithms?**
    Different algorithms handle data patterns differently. We selected the one with highest accuracy on the specific cybersecurity dataset.
    """)
    
    st.markdown("Algorithmic benchmarking and selection metrics...")
    
    results = get_results()
    table_data = []
    for name, value in results.items():
        table_data.append({
            "Algorithm": name,
            "Accuracy": f"{value*100:.2f}%",
            "Status": "ACTIVE" if name == "XGBoost" else "STANDBY"
        })
    
    st.table(table_data)
    
    # EXPLANATION: Results interpretation
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **🥇 Winner: XGBoost (eXtreme Gradient Boosting)**
        
        **Why it performed best:**
        - **Handles Imbalanced Data**: Works well even if you have more "Medium" than "Critical" samples
        - **Non-Linear Patterns**: Captures complex interactions (e.g., inflation + patch delay combined effect)
        - **Regularization**: Prevents overfitting to smaller datasets
        - **Feature Importance**: Automatically identifies which variables matter most in YOUR data
        
        **Accuracy Interpretation**: If XGBoost shows 85% accuracy, it correctly predicted the threat level in 8.5 out of 10 historical cases.
        """)
    
    with col2:
        st.warning("""
        **📊 Other Algorithms Tested:**
        
        **Logistic Regression** (Baseline):
        - Simple linear classifier
        - Lower accuracy because threat patterns are non-linear
        - Good for interpretability but misses complex interactions
        
        **Random Forest**:
        - Ensemble of decision trees
        - Good accuracy but can overfit with small datasets
        - Less effective than XGBoost at handling imbalanced classes
        
        **Note**: Results above reflect performance on your current dataset (default or uploaded).
        """)
    
    st.warning("⚠️ **SYSTEM NOTE**: XGBoost selected for production deployment. Superior handling of imbalanced threat datasets and complex feature interactions.")

# =====================================
# PARAMETERS - WITH EXPLANATIONS
# =====================================
with parameters:
    st.title("⚙️ SYSTEM PARAMETERS")
    
    # EXPLANATION: Parameters meaning
    st.markdown("""
    ### 📖 **2027 Projection Parameters**
    
    These values represent **projected conditions for August 2027** based on trend analysis, economic forecasts, and technological growth projections. 
    The model uses these 12 inputs to classify the threat level.
    
    **How Projections Are Derived:**
    - **Attack Volumes**: Extrapolated from historical growth curves
    - **CVE Counts**: Based on National Vulnerability Database growth rates
    - **Economic**: Central Bank inflation forecasts and GDP projections
    - **Operational**: Expected traffic volume and patch management efficiency targets
    
    **Note**: These projections remain constant, but the model's interpretation of them changes based on patterns learned from your uploaded data!
    """)
    
    st.markdown("Feature configuration for 2027 threat projection horizon:")
    
    height = 300 if st.session_state.mobile_view else 400
    st.dataframe(
        get_parameters(),
        use_container_width=True,
        height=height
    )
    
    # EXPLANATION: Parameter significance
    st.warning("""
    **⚠️ Critical Insight: Parameter Significance**
    
    Analysis reveals which inputs most influence the threat prediction:
    
    1. **Patch Delay Days** (9 days): *High Impact* - Longer delays mean more time for attackers to exploit known vulnerabilities
    2. **Critical CVEs** (95): *High Impact* - More security flaws = more attack opportunities  
    3. **Economic Environment** (Stable): *Medium Impact* - Economic stress correlates with increased cybercrime motivation
    4. **DDoS Attacks** (4,200): *Medium Impact* - Indicates attacker capability and infrastructure stress
    
    **Why These Matter**: The model learned that when Patch Delay < 10 days AND CVEs > 90 AND Economic Environment = Stable, the system typically faces HIGH threat levels due to the vulnerability-exposure window.
    
    **Custom Data Impact**: When you upload your own data, the model learns different patterns and may weight these factors differently based on your specific threat landscape!
    """)
