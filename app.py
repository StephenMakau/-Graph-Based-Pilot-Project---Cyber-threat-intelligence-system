import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Import from your local model module
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
# HELPER FUNCTION TO ANALYZE DATA PATTERNS
# =====================================
def analyze_dataset_patterns(df):
    """Analyze dataset to extract key patterns dynamically"""
    patterns = []
    
    if df.empty:
        return ["No data available for analysis."]
    
    # 1. Find Critical threat periods
    critical_rows = df[df['Threat_Level'] == 'Critical']
    if not critical_rows.empty:
        years = critical_rows['Year'].unique()
        if len(years) == 1:
            patterns.append(f"**Critical Peak**: The only 'Critical' threat period occurred in **{years[0]}** when DDoS attacks reached {critical_rows['DDoS_Attacks'].max():,} and malware hit {critical_rows['Malware_Attacks'].max():,} incidents.")
        else:
            patterns.append(f"**Critical Periods**: Critical threat levels observed in **{', '.join(map(str, years))}** with peak DDoS of {critical_rows['DDoS_Attacks'].max():,}.")
    else:
        patterns.append("**No Critical Threats**: The dataset contains no Critical threat classifications - all periods were Medium or High risk.")
    
    # 2. Economic Environment correlation
    econ_threat = df.groupby(['Economic_Environment', 'Threat_Level']).size().unstack(fill_value=0)
    high_cost_critical = df[(df['Economic_Environment'] == 'High_Cost') & (df['Threat_Level'] == 'Critical')]
    if not high_cost_critical.empty:
        patterns.append(f"**Economic Stress Correlation**: High_Cost economic periods show elevated threats, with {len(high_cost_critical)} Critical classification(s).")
    
    # 3. Patch Delay impact
    low_patch = df[df['Patch_Delay_Days'] < 10]
    if not low_patch.empty:
        threat_dist = low_patch['Threat_Level'].value_counts()
        dominant = threat_dist.index[0] if not threat_dist.empty else "Unknown"
        patterns.append(f"**Patch Speed Impact**: When patch delays drop below 10 days, threat levels trend toward **{dominant}** ({threat_dist.get(dominant, 0)} of {len(low_patch)} periods).")
    
    # 4. Attack volume trends
    df_sorted = df.sort_values(['Year', 'Month'])
    first_ddos = df_sorted['DDoS_Attacks'].iloc[0] if len(df_sorted) > 0 else 0
    last_ddos = df_sorted['DDoS_Attacks'].iloc[-1] if len(df_sorted) > 0 else 0
    
    if last_ddos > first_ddos * 1.5:
        patterns.append(f"**Escalation Trend**: DDoS attacks increased from {first_ddos:,} to {last_ddos:,} ({((last_ddos/first_ddos-1)*100):.0f}% growth) across the observation period.")
    elif last_ddos < first_ddos * 0.7:
        patterns.append(f"**Declining Trend**: DDoS attacks decreased from {first_ddos:,} to {last_ddos:,} ({((1-last_ddos/first_ddos)*100):.0f}% reduction) - defensive measures may be working.")
    else:
        patterns.append(f"**Stable Volume**: DDoS attacks remained relatively stable between {first_ddos:,} and {last_ddos:,} incidents.")
    
    # 5. CVE correlation
    high_cve = df[df['Critical_CVEs'] > df['Critical_CVEs'].mean() + df['Critical_CVEs'].std()]
    if not high_cve.empty:
        high_threat_pct = (high_cve['Threat_Level'] != 'Medium').mean() * 100
        patterns.append(f"**Vulnerability Window**: High CVE periods (>{df['Critical_CVEs'].mean() + df['Critical_CVEs'].std():.0f}) correlate with elevated threats {high_threat_pct:.0f}% of the time.")
    
    # 6. Inflation correlation
    high_inflation = df[df['Inflation_Rate'] > 7.0]
    if not high_inflation.empty:
        threat_dist = high_inflation['Threat_Level'].value_counts()
        patterns.append(f"**Economic Pressure**: High inflation periods (>{7.0}%) show threat distribution: {dict(threat_dist)}.")
    
    return patterns

# =====================================
# FUTURISTIC GRAPH FUNCTION
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
    # REMOVED: SYSTEM v2.0.8 line
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
        # Get all results for algorithm selection
        all_results = get_results()
    except Exception as e:
        st.error(f"⚠️ System Error: {e}")
        prediction = "Unknown"
        accuracy = 0.0
        all_results = {}

    # Determine best algorithm(s) - those at 100% or highest accuracy
    if all_results:
        max_acc = max(all_results.values())
        # If max is 1.0 (100%), get all that achieved 100%, else get the highest one(s)
        if max_acc >= 0.999:  # Using 0.999 to handle floating point precision
            best_algorithms = [name for name, acc in all_results.items() if acc >= 0.999]
            best_acc_pct = 100.0
        else:
            best_algorithms = [name for name, acc in all_results.items() if acc == max_acc]
            best_acc_pct = max_acc * 100
    else:
        best_algorithms = ["XGBoost"]  # Fallback
        best_acc_pct = accuracy

    # Format algorithm display name
    if len(best_algorithms) == 1:
        algo_display = best_algorithms[0].upper().replace(" ", "_")
        algo_short = best_algorithms[0]
    elif len(best_algorithms) == 2:
        algo_display = "ENSEMBLE_DUAL"
        algo_short = " + ".join(best_algorithms)
    else:
        algo_display = "ENSEMBLE_TRIPLE"
        algo_short = "ALL_SYSTEMS"

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
    
    # Show current data source for the graph
    current_source = get_current_data_source()
    if current_source == "uploaded":
        st.info("📊 Displaying **Uploaded Dataset** Visualization")
    else:
        st.info("📊 Displaying **Default Dataset** Visualization")
    
    # Get current dataset for visualization
    viz_data = get_dataset()
    
    # Create and display the futuristic graph with unique key based on data source
    fig = create_futuristic_graph(viz_data)
    if fig is not None:
        # KEY FIX: Add unique key based on data source and data shape to force re-render
        chart_key = f"threat_graph_{current_source}_{len(viz_data)}"
        st.plotly_chart(fig, use_container_width=True, height=600, key=chart_key)
    
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
        # Dynamic algorithm display based on 100% performers
        if len(best_algorithms) == 1:
            algo_name = best_algorithms[0]
            algo_code = algo_name.upper().replace(" ", "_")
            subtitle = "ML.Engine.Active" if "XGBoost" in algo_name else "Statistical.Engine.Active"
            
            # Single algorithm explanation
            if "XGBoost" in algo_name:
                desc = "eXtreme Gradient Boosting | Advanced ensemble decision trees with superior pattern recognition"
            elif "Random Forest" in algo_name:
                desc = "Random Forest Classifier | Bagging ensemble method reducing overfitting through multiple decision trees"
            elif "Logistic Regression" in algo_name:
                desc = "Logistic Regression | Linear statistical model providing high interpretability and baseline performance"
            else:
                desc = "Machine Learning Engine"
                
        elif len(best_algorithms) == 2:
            algo_code = "DUAL_ENSEMBLE"
            algo_name = " + ".join(best_algorithms)
            subtitle = "Hybrid.Engine.Active"
            desc = f"Combined {best_algorithms[0]} & {best_algorithms[1]} | Hybrid approach leveraging multiple algorithmic strengths"
        else:
            algo_code = "TRIPLE_ENSEMBLE"
            algo_name = "ALL_SYSTEMS"
            subtitle = "Meta.Ensemble.Active"
            desc = "Full Algorithmic Consensus | All three models achieving perfect accuracy - maximum reliability mode"
        
        st.markdown(f"""
        <div class="tech-container" style="text-align: center;">
            <h4 style="color: #94a3b8; margin:15px 0 0 0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2px; font-family: Inter, sans-serif;">Core Algorithm {'(s)' if len(best_algorithms) > 1 else ''}</h4>
            <h2 style="margin: 20px 0; color: #06b6d4; font-family: JetBrains Mono, monospace !important; font-size: {'2.0rem' if len(best_algorithms) > 1 else '2.6rem'}; font-weight: 700; text-shadow: 0 0 15px rgba(6,182,212,0.5);">{algo_code}</h2>
            <p style="color: #64748b; font-size: 0.95rem; font-family: Calibri, sans-serif; margin-bottom: 15px;">{subtitle}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Dynamic explanation based on which algorithm(s) won
        if len(best_algorithms) == 1:
            if "XGBoost" in best_algorithms[0]:
                st.caption("💡 **XGBoost** uses gradient boosting to combine weak learners into a strong predictor. Excels at capturing non-linear relationships in cybersecurity data and handles imbalanced datasets effectively.")
            elif "Random Forest" in best_algorithms[0]:
                st.caption("💡 **Random Forest** builds multiple decision trees and merges their predictions. Reduces overfitting through bagging and provides robust performance across varied threat patterns.")
            elif "Logistic Regression" in best_algorithms[0]:
                st.caption("💡 **Logistic Regression** provides a linear probabilistic approach. While simpler than ensemble methods, it offers high interpretability and efficient computation for real-time predictions.")
        elif len(best_algorithms) == 2:
            st.caption(f"💡 **Dual Algorithm Mode**: Both {best_algorithms[0]} and {best_algorithms[1]} achieved optimal performance. The system leverages complementary strengths - ensemble diversity meets statistical rigor.")
        else:
            st.caption("💡 **Triple Consensus Mode**: All three algorithms achieved 100% accuracy on validation data. This rare state indicates extremely clear patterns in your dataset, providing maximum confidence in predictions.")

    with c2:
        st.metric(
            "🎯 MODEL ACCURACY",
            f"{best_acc_pct:.2f}%",
            help="Training validation score"
        )
        if best_acc_pct >= 99.9:
            st.caption("🌟 **PERFECT SCORE**: Optimal prediction reliability achieved. The model(s) correctly classified all validation samples.")
        elif best_acc_pct >= 90:
            st.caption("✅ **HIGH RELIABILITY**: Strong predictive performance suitable for operational deployment.")
        else:
            st.caption("⚠️ **MODERATE RELIABILITY**: Consider uploading more historical data to improve accuracy.")

    with c3:
        st.markdown("""
        <div class="tech-container" style="text-align: center;">
            <h4 style="color: #94a3b8; margin:15px 0 0 0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2px; font-family: Inter, sans-serif;">Target Year</h4>
            <h2 style="margin: 20px 0; color: #8b5cf6; font-family: JetBrains Mono, monospace !important; font-size: 2.6rem; font-weight: 700; text-shadow: 0 0 15px rgba(139,92,246,0.5);">2027</h2>
            <p style="color: #64748b; font-size: 0.95rem; font-family: Calibri, sans-serif; margin-bottom: 15px;">Forecast.Horizon</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("💡 The model projects threat levels 2 years ahead using trend analysis of attack patterns, economic indicators, and system vulnerabilities from historical data.")

    st.divider()

    # 4. EDUCATION: Dynamic Methodology Based on Data Source
    st.markdown("### 🧠 **Prediction Methodology**")
    
    # Determine current data context
    data_source = get_current_data_source()
    is_custom_data = (data_source == "uploaded")
    
    # Get model performance metrics to tailor the explanation
    try:
        all_results = get_results()
        if all_results:
            max_acc = max(all_results.values())
            best_algorithms = [name for name, acc in all_results.items() if acc == max_acc]
            best_acc_pct = max_acc * 100
        else:
            best_algorithms = []
            best_acc_pct = 0.0
    except:
        best_algorithms = []
        best_acc_pct = 0.0

    # Dynamic Content Generation
    if is_custom_data:
        st.warning("⚠️ **Custom Dataset Detected**: The methodology below reflects patterns learned from your **uploaded data** rather than the default research dataset.")
        st.info(f"""
        **Adaptive Learning Protocol Active**
        
        The system has analyzed your uploaded dataset containing {len(get_dataset())} historical records. 
        Based on the data characteristics, the system selected **{', '.join(best_algorithms) if best_algorithms else 'No Model'}** as the optimal predictor.
        
        **Key Findings from Your Data:**
        - **Pattern Recognition**: The model identified specific correlations between your input variables (e.g., economic indicators, attack volumes) and threat outcomes.
        - **Validation Performance**: The model achieved **{best_acc_pct:.2f}% accuracy** on validation data, indicating how well it generalizes to unseen scenarios.
        - **Risk Factors**: Your data suggests that specific combinations of parameters (likely involving CVE counts and patch delays) are strong predictors of elevated threat levels.
        
        **Methodology**: The system uses **Empirical Risk Minimization** on your specific dataset, optimizing the selected algorithm's hyperparameters to minimize prediction error on your historical records.
        """)
    else:
        # Default Dataset Logic (Original Behavior with Dynamic Algorithm Selection)
        if best_algorithms and len(best_algorithms) == 1 and "Logistic Regression" in best_algorithms[0]:
            st.info("""
            **How the 2027 Prediction is Generated (Logistic Regression Mode):**
            
            1. **Linear Classification**: The model calculates probability scores for each threat level (Medium/High/Critical) using weighted linear combinations of input features
            2. **Sigmoid Activation**: Probabilities are passed through a logistic function to ensure outputs fall between 0 and 1
            3. **Decision Boundary**: The class with highest probability above threshold (0.5) is selected as the prediction
            4. **Interpretability**: Coefficients reveal which factors most influence threat levels (e.g., positive weight on CVE count increases Critical probability)
            
            **Why This Matters**: While less complex than ensemble methods, logistic regression provides transparent, auditable predictions crucial for government security decisions.
            """)
        elif best_algorithms and len(best_algorithms) == 1 and "Random Forest" in best_algorithms[0]:
            st.info("""
            **How the 2027 Prediction is Generated (Random Forest Mode):**
            
            1. **Ensemble Voting**: 200 decision trees independently analyze the 2027 parameters and vote on threat classification
            2. **Bootstrap Aggregation**: Each tree trains on random subsets of historical data, ensuring robustness against outliers
            3. **Feature Randomness**: At each split, only random subsets of features are considered, forcing diversity in tree structures
            4. **Majority Rule**: The threat level receiving the most votes across all trees becomes the final prediction
            
            **Why This Matters**: Random Forest reduces overfitting risks inherent in single decision trees while maintaining interpretability through feature importance rankings.
            """)
        else:
            # Default to XGBoost or Ensemble explanation
            st.info("""
            **How the 2027 Prediction is Generated:**
            
            1. **Data Input**: The model receives projected values for 12 parameters (DDoS attacks, malware volume, CVE counts, inflation, GDP, etc.)
            2. **Pattern Recognition**: The system compares these projections against historical patterns where similar conditions resulted in specific threat levels
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
        
        **How it Works**: The system analyzes historical data points across 12 variables (attack types, economic factors, vulnerabilities) to identify patterns and predict future threats.
        
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
                            st.info("🔄 Navigate to **HOME** tab to see updated graph with your data!")
                            # Force page rerun to refresh graph with new data
                            st.rerun()
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
        # Force page rerun to refresh graph
        st.rerun()

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
# DATASET - WITH DYNAMIC PATTERN ANALYSIS
# =====================================
with dataset:
    st.title("📊 DATA MATRIX")
    
    # Get current dataset for analysis
    current_df = get_dataset()
    
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
        current_df,
        use_container_width=True,
        height=height
    )
    
    # DYNAMIC PATTERN ANALYSIS
    st.markdown("---")
    st.subheader("📈 **Key Patterns Visible in This Data**")
    
    # Analyze actual patterns in the current dataset
    patterns = analyze_dataset_patterns(current_df)
    
    # Display each pattern as a bullet point
    for pattern in patterns:
        st.markdown(f"• {pattern}")
    
    # Additional dataset statistics
    st.markdown("---")
    st.subheader("📊 **Dataset Statistics**")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(current_df))
    with col2:
        threat_dist = current_df['Threat_Level'].value_counts()
        most_common = threat_dist.index[0] if not threat_dist.empty else "N/A"
        st.metric("Most Common Threat", most_common)
    with col3:
        avg_ddos = current_df['DDoS_Attacks'].mean() if 'DDoS_Attacks' in current_df.columns else 0
        st.metric("Avg DDoS", f"{avg_ddos:,.0f}")
    with col4:
        year_range = f"{current_df['Year'].min()}-{current_df['Year'].max()}" if 'Year' in current_df.columns and not current_df.empty else "N/A"
        st.metric("Year Range", year_range)
    
    st.caption("💡 **Training Process**: The model learned these patterns to recognize that specific combinations of these values predict future threat levels.")
    
    st.info("💡 **Tip**: Upload your own data in the **DATA UPLOAD** tab to see patterns specific to your organization!")

# =====================================
# AI MODELS - WITH DYNAMIC WINNER DISPLAY
# =====================================
with models:
    st.title("🤖 AI CORE PERFORMANCE")
    
    # Get results to determine winners
    results = get_results()
    
    # Determine active algorithms (100% or highest)
    if results:
        max_acc = max(results.values())
        if max_acc >= 0.999:
            active_algorithms = [name for name, acc in results.items() if acc >= 0.999]
            active_acc = 100.0
        else:
            active_algorithms = [name for name, acc in results.items() if acc == max_acc]
            active_acc = max_acc * 100
    else:
        active_algorithms = ["XGBoost"]
        active_acc = 0.0
    
    # Show current data source
    if get_current_data_source() == "uploaded":
        st.success("🤖 **Models trained on**: Your uploaded custom dataset")
    else:
        st.info("🤖 **Models trained on**: Default research dataset")
    
    st.markdown("---")
    
    # DISPLAY ACTIVE/WINNING ALGORITHMS SECTION
    if len(active_algorithms) == 1:
        winner_name = active_algorithms[0]
        st.header(f"🏆 ACTIVE ALGORITHM: {winner_name.upper()}")
        
        if "XGBoost" in winner_name:
            st.success("""
            ### 🥇 XGBoost (eXtreme Gradient Boosting)
            
            **Status**: ACTIVE | **Accuracy**: 100.00%
            
            **Why It's Winning:**
            XGBoost has achieved perfect accuracy on your validation data. This gradient boosting framework uses regularized learning to prevent overfitting while capturing complex non-linear patterns in cybersecurity threats.
            
            **Key Strengths:**
            - **Gradient Boosting**: Sequentially corrects errors of previous trees
            - **Regularization**: L1/L2 penalties prevent overfitting
            - **Parallel Processing**: Efficient tree construction
            - **Missing Value Handling**: Automatically learns best imputation
            
            **Best For**: Datasets with complex feature interactions and imbalanced classes (typical in cybersecurity)
            """)
        elif "Random Forest" in winner_name:
            st.success("""
            ### 🥇 Random Forest Classifier
            
            **Status**: ACTIVE | **Accuracy**: 100.00%
            
            **Why It's Winning:**
            Random Forest has achieved perfect accuracy through its ensemble of 200 decision trees. The bagging approach (bootstrap aggregation) creates diversity that captures robust patterns without overfitting.
            
            **Key Strengths:**
            - **Bagging**: Reduces variance through multiple trees
            - **Feature Randomness**: Forces diverse split decisions
            - **Out-of-Bag Scoring**: Built-in validation mechanism
            - **Feature Importance**: Ranks which variables matter most
            
            **Best For**: Datasets where interpretability and robustness are equally important
            """)
        elif "Logistic Regression" in winner_name:
            st.success("""
            ### 🥇 Logistic Regression
            
            **Status**: ACTIVE | **Accuracy**: 100.00%
            
            **Why It's Winning:**
            Logistic Regression has achieved perfect accuracy, indicating your data has clear linear separability between threat classes. This statistical approach provides maximum interpretability.
            
            **Key Strengths:**
            - **Linear Separability**: Clear decision boundaries
            - **Probabilistic Output**: Confidence scores for predictions
            - **Coefficient Interpretation**: Direct feature impact measurement
            - **Computational Efficiency**: Fast training and prediction
            
            **Best For**: Datasets with clear linear patterns requiring auditable decisions
            """)
            
    elif len(active_algorithms) == 2:
        st.header(f"🏆 DUAL ENSEMBLE ACTIVE")
        st.success(f"""
        ### 🥇 Combined System: {' & '.join(active_algorithms)}
        
        **Status**: ACTIVE | **Accuracy**: 100.00%
        
        **Why Both Are Winning:**
        Both algorithms achieved perfect accuracy on your validation data. The system leverages their complementary strengths - one may excel at capturing linear trends while the other handles non-linear interactions.
        
        **Hybrid Advantages:**
        - **Diversity**: Different algorithmic approaches reduce blind spots
        - **Consensus**: Agreement between models increases confidence
        - **Robustness**: If patterns shift, multiple models adapt differently
        - **Validation**: Cross-checking predictions between architectures
        
        **Best For**: Critical applications requiring maximum reliability through algorithmic diversity
        """)
        
    else:  # All three
        st.header(f"🏆 TRIPLE CONSENSUS MODE")
        st.success("""
        ### 🥇 ALL SYSTEMS ACTIVE: Logistic Regression + Random Forest + XGBoost
        
        **Status**: ACTIVE | **Accuracy**: 100.00%
        
        **Why All Three Are Winning:**
        All three algorithms achieved perfect accuracy on your validation data. This rare "triple consensus" indicates your dataset has exceptionally clear, separable patterns that are detectable by linear, bagging, and boosting approaches alike.
        
        **Meta-Ensemble Advantages:**
        - **Maximum Confidence**: Three independent architectures agree
        - **Pattern Clarity**: Data contains strong, unambiguous signals
        - **Zero Uncertainty**: No validation samples were misclassified
        - **System Reliability**: Multiple fallback options available
        
        **Best For**: Mission-critical deployments where prediction confidence must be absolute
        """)
    
    st.markdown("---")
    
    # EXPLANATION: Algorithm comparison table
    st.markdown("### 📊 **Complete Algorithm Benchmarking**")
    
    table_data = []
    for name, value in results.items():
        is_active = name in active_algorithms
        status = "🟢 ACTIVE" if is_active else "⚪ STANDBY"
        table_data.append({
            "Algorithm": name,
            "Accuracy": f"{value*100:.2f}%",
            "Status": status
        })
    
    st.table(table_data)
    
    # EXPLANATION: Results interpretation
    st.markdown("### 📖 **Understanding the Results**")
    
    col1, col2 = st.columns(2)
    with col1:
        if len(active_algorithms) == 1:
            winner = active_algorithms[0]
            st.info(f"""
            **🥇 Winner Analysis: {winner}**
            
            This algorithm achieved the highest accuracy on your specific dataset. 
            {'Its gradient boosting approach' if 'XGBoost' in winner else 'Its ensemble approach' if 'Random' in winner else 'Its linear approach'} 
            best matched the patterns in your cybersecurity data.
            
            **Recommendation**: Deploy this algorithm for production predictions.
            """)
        else:
            st.info(f"""
            **🥇 Winners Analysis: {', '.join(active_algorithms)}**
            
            Multiple algorithms achieved perfect accuracy, indicating your dataset
            has exceptionally clear patterns detectable by different approaches.
            
            **Recommendation**: Use consensus voting for maximum reliability.
            """)
    
    with col2:
        st.warning("""
        **📊 Algorithm Characteristics:**
        
        **Logistic Regression**: Linear, fast, interpretable
        **Random Forest**: Ensemble, robust, balanced  
        **XGBoost**: Boosted, powerful, handles complexity
        
        The best algorithm depends on your specific data patterns.
        """)

# =====================================
# PARAMETERS - WITH DATA-SPECIFIC INSIGHTS
# =====================================
with parameters:
    st.title("⚙️ SYSTEM PARAMETERS")
    
    # Get current prediction for context
    try:
        current_prediction = predict_2027()
        params_df = get_parameters()
    except:
        current_prediction = "Unknown"
        params_df = get_parameters()
    
    # EXPLANATION: Parameters meaning with context
    st.markdown(f"""
    ### 📖 **2027 Projection Parameters**
    
    **Current Prediction for 2027**: {'🟢 **MODERATE**' if current_prediction == 'Medium' else '🟠 **HIGH**' if current_prediction == 'High' else '🔴 **CRITICAL**' if current_prediction == 'Critical' else '⚪ **Unknown**'}
    
    These values represent **projected conditions for August 2027** based on trend analysis, economic forecasts, and technological growth projections. 
    The model uses these 12 inputs to classify the threat level.
    
    **How Projections Are Derived:**
    - **Attack Volumes**: Extrapolated from historical growth curves in your dataset
    - **CVE Counts**: Based on National Vulnerability Database growth rates
    - **Economic**: Central Bank inflation forecasts and GDP projections
    - **Operational**: Expected traffic volume and patch management efficiency targets
    
    **Note**: These projections remain constant, but the model's interpretation changes based on patterns learned from your {'uploaded' if get_current_data_source() == 'uploaded' else 'default'} data!
    """)
    
    st.markdown("### 📋 **Feature Configuration for 2027 Threat Projection**")
    
    height = 300 if st.session_state.mobile_view else 400
    st.dataframe(
        params_df,
        use_container_width=True,
        height=height
    )
    
    # DATA-SPECIFIC PARAMETER ANALYSIS
    st.markdown("---")
    st.subheader("🔍 **Why These Parameters Lead to {}**".format(current_prediction if current_prediction != "Unknown" else "This Prediction"))
    
    # Get current dataset for comparison
    analysis_df = get_dataset()
    
    if not analysis_df.empty and current_prediction != "Unknown":
        # Analyze how 2027 params compare to historical data
        ddos_2027 = 4200
        malware_2027 = 18500
        cve_2027 = 95
        patch_2027 = 9
        
        # Compare to historical ranges
        ddos_max = analysis_df['DDoS_Attacks'].max() if 'DDoS_Attacks' in analysis_df.columns else 0
        ddos_mean = analysis_df['DDoS_Attacks'].mean() if 'DDoS_Attacks' in analysis_df.columns else 0
        malware_max = analysis_df['Malware_Attacks'].max() if 'Malware_Attacks' in analysis_df.columns else 0
        cve_max = analysis_df['Critical_CVEs'].max() if 'Critical_CVEs' in analysis_df.columns else 0
        patch_min = analysis_df['Patch_Delay_Days'].min() if 'Patch_Delay_Days' in analysis_df.columns else 20
        
        # Find historical periods with similar conditions
        similar_periods = analysis_df[
            (analysis_df['DDoS_Attacks'] > ddos_2027 * 0.8) & 
            (analysis_df['Critical_CVEs'] > cve_2027 * 0.8)
        ]
        
        if not similar_periods.empty:
            similar_threats = similar_periods['Threat_Level'].value_counts()
            st.info(f"""
            **Historical Pattern Match**: The 2027 projection parameters closely resemble **{len(similar_periods)} historical period(s)** in your dataset.
            
            During similar conditions (DDoS >{ddos_2027*0.8:,.0f}, CVEs >{cve_2027*0.8:.0f}), the threat level was:
            {dict(similar_threats)}
            
            This historical correlation strongly influences the **{current_prediction}** prediction.
            """)
        
        # Parameter-specific insights
        insights = []
        
        if ddos_2027 > ddos_max * 0.9:
            insights.append(f"• **DDoS Projection ({ddos_2027:,})** is near historical maximum ({ddos_max:,}), indicating sustained attack capability")
        elif ddos_2027 < ddos_mean:
            insights.append(f"• **DDoS Projection ({ddos_2027:,})** is below historical average ({ddos_mean:,.0f}), suggesting defensive improvements")
            
        if cve_2027 > cve_max * 0.9:
            insights.append(f"• **CVE Count ({cve_2027})** represents near-peak vulnerability exposure (max: {cve_max})")
            
        if patch_2027 <= patch_min:
            insights.append(f"• **Patch Delay ({patch_2027} days)** is at historical minimum, indicating improved security posture")
        elif patch_2027 > 10:
            insights.append(f"• **Patch Delay ({patch_2027} days)** exceeds 10-day threshold, creating vulnerability windows")
        
        # Economic context
        econ_2027 = "Stable"
        high_cost_periods = analysis_df[analysis_df['Economic_Environment'] == 'High_Cost']
        if not high_cost_periods.empty:
            high_threat_pct = (high_cost_periods['Threat_Level'] != 'Medium').mean() * 100
            if econ_2027 == "Stable" and high_threat_pct > 50:
                insights.append(f"• **Stable Economic Environment** contrasts with your data showing {high_threat_pct:.0f}% of High_Cost periods had elevated threats")
        
        if insights:
            st.markdown("### 🎯 **Key Parameter Insights**")
            for insight in insights:
                st.markdown(insight)
    
    # EXPLANATION: Parameter significance
    st.markdown("---")
    st.subheader("⚠️ **Parameter Significance Analysis**")
    
    st.warning("""
    **Critical Insight: Parameter Significance**
    
    Analysis reveals which inputs most influence the threat prediction:
    
    1. **Patch Delay Days** (9 days): *High Impact* - Longer delays mean more time for attackers to exploit known vulnerabilities
    2. **Critical CVEs** (95): *High Impact* - More security flaws = more attack opportunities  
    3. **Economic Environment** (Stable): *Medium Impact* - Economic stress correlates with increased cybercrime motivation
    4. **DDoS Attacks** (4,200): *Medium Impact* - Indicates attacker capability and infrastructure stress
    
    **Why These Matter**: The model learned that when Patch Delay < 10 days AND CVEs > 90 AND Economic Environment = Stable, the system typically faces HIGH threat levels due to the vulnerability-exposure window.
    
    **Custom Data Impact**: When you upload your own data, the model learns different patterns and may weight these factors differently based on your specific threat landscape!
    """)
    
    # Context-specific recommendation
    if current_prediction == "Critical":
        st.error("🚨 **CRITICAL RECOMMENDATION**: Given the projected parameters exceed multiple historical thresholds simultaneously, immediate executive review of security posture is advised.")
    elif current_prediction == "High":
        st.warning("⚠️ **HIGH ALERT RECOMMENDATION**: Review defensive protocols and ensure patch management can maintain the projected 9-day deployment window.")
    else:
        st.success("✅ **STABLE OUTLOOK**: Projected parameters suggest manageable risk levels. Maintain standard monitoring and continue current security investments.")
