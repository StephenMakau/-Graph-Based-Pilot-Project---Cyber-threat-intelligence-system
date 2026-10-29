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

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "🏠 HOME"

# Get current year from browser/system
current_year = datetime.now().year

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="MKU Cyber Threat Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================
# CUSTOM CSS - FIXED NAVIGATION BAR
# =====================================
nav_css = f"""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Hide default sidebar toggle button completely */
    button[kind="header"] {{
        display: none !important;
    }}
    
    /* Fixed Navigation Bar - stays on top when scrolling */
    .nav-fixed {{
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 55px;
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border-bottom: 2px solid #06b6d4;
        z-index: 999999;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 15px;
        box-shadow: 0 4px 20px rgba(6, 182, 212, 0.3);
    }}
    
    /* Hamburger icon only - no text */
    .hamburger-only {{
        background: linear-gradient(135deg, #06b6d4, #3b82f6);
        border: none;
        border-radius: 8px;
        width: 40px;
        height: 40px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.5);
        font-size: 20px;
        color: white;
    }}
    
    .hamburger-only:hover {{
        transform: scale(1.1);
        box-shadow: 0 0 25px rgba(6, 182, 212, 0.8);
    }}
    
    /* Nav Title */
    .nav-title-text {{
        color: #06b6d4;
        font-size: 1rem;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(6, 182, 212, 0.5);
        margin-left: 10px;
    }}
    
    /* Push ALL content below nav */
    .main .block-container {{
        padding-top: 70px !important;
        max-width: 100% !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }}
    
    /* Compact sections */
    .compact-section {{
        margin-bottom: 10px !important;
    }}
    
    /* Hide radio button circles for nav */
    div[data-testid="stRadio"] > div {{
        flex-direction: row !important;
        gap: 4px !important;
        flex-wrap: wrap;
    }}
    
    div[data-testid="stRadio"] label {{
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.1), rgba(59, 130, 246, 0.1));
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 6px;
        padding: 6px 10px;
        color: #94a3b8;
        font-size: 0.72rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 0 !important;
    }}
    
    div[data-testid="stRadio"] label:hover {{
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.3), rgba(59, 130, 246, 0.3));
        color: #06b6d4;
        border-color: #06b6d4;
        transform: translateY(-1px);
    }}
    
    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {{
        display: none !important;
    }}
    
    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:nth-child(2) {{
        margin-left: 0 !important;
    }}
    
    /* Active/selected state */
    div[data-testid="stRadio"] label:has(input:checked) {{
        background: linear-gradient(135deg, #06b6d4, #3b82f6) !important;
        color: white !important;
        border-color: #06b6d4 !important;
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.6);
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    }}
    
    /* Compact metrics */
    [data-testid="stMetric"] {{
        padding: 10px !important;
    }}
    
    /* Reduce spacing */
    h1 {{ margin-bottom: 10px !important; font-size: 1.6rem !important; }}
    h2 {{ margin-bottom: 8px !important; font-size: 1.3rem !important; }}
    h3 {{ margin-bottom: 6px !important; font-size: 1.1rem !important; }}
    
    /* Mobile responsive */
    @media (max-width: 768px) {{
        .nav-title-text {{ font-size: 0.8rem; }}
    }}
</style>
"""

st.markdown(nav_css, unsafe_allow_html=True)

# =====================================
# FIXED NAVIGATION BAR - RENDERED FIRST
# =====================================
# Create container for fixed nav
nav_container = st.container()

with nav_container:
    cols = st.columns([1, 4, 8])
    
    with cols[0]:
        # Single hamburger icon button
        if st.button("☰", key="hamburger_toggle", help="System Controls", 
                    type="primary", use_container_width=False):
            st.session_state.sidebar_open = not st.session_state.sidebar_open
            # Use JavaScript to toggle sidebar
            st.components.v1.html("""
            <script>
                var sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
                var button = sidebar.querySelector('button[kind="header"]');
                if (button) button.click();
            </script>
            """, height=0)
    
    with cols[1]:
        st.markdown('<span class="nav-title-text">🛡️ MKU CYBER THREAT INTELLIGENCE</span>', unsafe_allow_html=True)
    
    with cols[2]:
        # Horizontal radio for navigation - compact
        tab_options = ["🏠 HOME", "📤 DATA UPLOAD", "📄 PROJECT OVERVIEW", "📊 DATASET", "🤖 AI MODELS", "⚙️ PARAMETERS"]
        
        selected_tab = st.radio(
            "Navigation",
            tab_options,
            index=tab_options.index(st.session_state.active_tab) if st.session_state.active_tab in tab_options else 0,
            horizontal=True,
            label_visibility="collapsed",
            key="nav_radio"
        )
        
        if selected_tab != st.session_state.active_tab:
            st.session_state.active_tab = selected_tab
            st.rerun()

# =====================================
# SIDEBAR CONTROLS - ONLY VIA HAMBURGER
# =====================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin: 15px 0;">
        <div style="display: inline-block; padding: 12px; background: linear-gradient(135deg, #06b6d4, #3b82f6); border-radius: 10px; box-shadow: 0 0 15px rgba(6, 182, 212, 0.5);">
            <div style="width: 25px; height: 3px; background: white; margin: 4px 0; border-radius: 2px;"></div>
            <div style="width: 25px; height: 3px; background: white; margin: 4px 0; border-radius: 2px;"></div>
            <div style="width: 25px; height: 3px; background: white; margin: 4px 0; border-radius: 2px;"></div>
        </div>
        <p style="color: #06b6d4; font-size: 0.85rem; margin-top: 10px; font-weight: 600;">⚙️ SYSTEM CONTROLS</p>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    data_source = get_current_data_source()
    if data_source == "uploaded":
        st.success("📊 Using: **Uploaded Data**")
    else:
        st.info("📊 Using: **Default Dataset**")
    
    st.caption(f"System Time: {datetime.now(ZoneInfo('Africa/Nairobi')).strftime('%H:%M:%S')}")

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
        .explanation-box { background: rgba(6, 182, 212, 0.1); border-left: 3px solid #06b6d4; padding: 12px; border-radius: 5px; margin: 8px 0; }
        .upload-box { background: rgba(139, 92, 246, 0.1); border: 2px dashed rgba(139, 92, 246, 0.5); border-radius: 10px; padding: 15px; text-align: center; }
        .template-box { background: rgba(16, 185, 129, 0.1); border: 2px solid rgba(16, 185, 129, 0.5); border-radius: 10px; padding: 15px; }
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
        .explanation-box { background: rgba(37, 99, 235, 0.1); border-left: 3px solid #0369a1; padding: 12px; border-radius: 5px; margin: 8px 0; }
        .upload-box { background: rgba(139, 92, 246, 0.05); border: 2px dashed rgba(139, 92, 246, 0.5); border-radius: 10px; padding: 15px; text-align: center; }
        .template-box { background: rgba(16, 185, 129, 0.05); border: 2px solid rgba(16, 185, 129, 0.5); border-radius: 10px; padding: 15px; }
    </style>
    """

if st.session_state.mobile_view:
    mobile_css = """
    <style>
        .block-container { max-width: 480px !important; padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
        h1 { font-size: 1.4rem !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1rem !important; }
        .stTabs [data-testid="stTab"] { padding: 6px 10px !important; font-size: 0.75rem !important; }
        [data-testid="stMetric"] { padding: 8px !important; }
        [data-testid="stMetricValue"] { font-size: 1.5rem !important; }
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
# HELPER FUNCTIONS (ALL ORIGINAL FUNCTIONALITY PRESERVED)
# =====================================
def analyze_dataset_patterns(df):
    """Analyze dataset to extract key patterns dynamically"""
    patterns = []
    
    if df.empty:
        return ["No data available for analysis."]
    
    critical_rows = df[df['Threat_Level'] == 'Critical']
    if not critical_rows.empty:
        years = critical_rows['Year'].unique()
        if len(years) == 1:
            patterns.append(f"**Critical Peak**: The only 'Critical' threat period occurred in **{years[0]}** when DDoS attacks reached {critical_rows['DDoS_Attacks'].max():,} and malware hit {critical_rows['Malware_Attacks'].max():,} incidents.")
        else:
            patterns.append(f"**Critical Periods**: Critical threat levels observed in **{', '.join(map(str, years))}** with peak DDoS of {critical_rows['DDoS_Attacks'].max():,}.")
    else:
        patterns.append("**No Critical Threats**: The dataset contains no Critical threat classifications - all periods were Medium or High risk.")
    
    high_cost_critical = df[(df['Economic_Environment'] == 'High_Cost') & (df['Threat_Level'] == 'Critical')]
    if not high_cost_critical.empty:
        patterns.append(f"**Economic Stress Correlation**: High_Cost economic periods show elevated threats, with {len(high_cost_critical)} Critical classification(s).")
    
    low_patch = df[df['Patch_Delay_Days'] < 10]
    if not low_patch.empty:
        threat_dist = low_patch['Threat_Level'].value_counts()
        dominant = threat_dist.index[0] if not threat_dist.empty else "Unknown"
        patterns.append(f"**Patch Speed Impact**: When patch delays drop below 10 days, threat levels trend toward **{dominant}** ({threat_dist.get(dominant, 0)} of {len(low_patch)} periods).")
    
    df_sorted = df.sort_values(['Year', 'Month'])
    first_ddos = df_sorted['DDoS_Attacks'].iloc[0] if len(df_sorted) > 0 else 0
    last_ddos = df_sorted['DDoS_Attacks'].iloc[-1] if len(df_sorted) > 0 else 0
    
    if last_ddos > first_ddos * 1.5:
        patterns.append(f"**Escalation Trend**: DDoS attacks increased from {first_ddos:,} to {last_ddos:,} ({((last_ddos/first_ddos-1)*100):.0f}% growth) across the observation period.")
    elif last_ddos < first_ddos * 0.7:
        patterns.append(f"**Declining Trend**: DDoS attacks decreased from {first_ddos:,} to {last_ddos:,} ({((1-last_ddos/first_ddos)*100):.0f}% reduction) - defensive measures may be working.")
    else:
        patterns.append(f"**Stable Volume**: DDoS attacks remained relatively stable between {first_ddos:,} and {last_ddos:,} incidents.")
    
    high_cve = df[df['Critical_CVEs'] > df['Critical_CVEs'].mean() + df['Critical_CVEs'].std()]
    if not high_cve.empty:
        high_threat_pct = (high_cve['Threat_Level'] != 'Medium').mean() * 100
        patterns.append(f"**Vulnerability Window**: High CVE periods (>{df['Critical_CVEs'].mean() + df['Critical_CVEs'].std():.0f}) correlate with elevated threats {high_threat_pct:.0f}% of the time.")
    
    high_inflation = df[df['Inflation_Rate'] > 7.0]
    if not high_inflation.empty:
        threat_dist = high_inflation['Threat_Level'].value_counts()
        patterns.append(f"**Economic Pressure**: High inflation periods (>{7.0}%) show threat distribution: {dict(threat_dist)}.")
    
    return patterns

def create_futuristic_graph(df):
    """Create a futuristic cyberpunk-style visualization of threat data"""
    try:
        df = df.copy()
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
        df = df.dropna(subset=['Year', 'Month'])
        df['Month'] = df['Month'].apply(lambda x: int(x) if 1 <= int(x) <= 12 else 1)
        df['Year'] = df['Year'].astype(int)
        df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
        df = df.sort_values('Date')
        
        if df.empty:
            st.error("No valid data available for visualization.")
            return None

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=('⚡ ATTACK VECTORS OVER TIME', '🔒 VULNERABILITY METRICS')
        )
        
        colors = {
            'ddos': '#00f5ff', 'malware': '#ff00ff', 'phishing': '#39ff14',
            'web': '#ff6b35', 'cves': '#ff1744', 'patch': '#ffd700',
            'traffic': '#8b5cf6', 'grid': 'rgba(6, 182, 212, 0.1)', 'text': '#e2e8f0'
        }
        
        fig.add_trace(go.Scatter(x=df['Date'], y=df['DDoS_Attacks'], name='DDoS Attacks', mode='lines+markers',
            line=dict(color=colors['ddos'], width=3), marker=dict(size=8, symbol='diamond', line=dict(width=2, color='white')),
            fill='tonexty', fillcolor='rgba(0, 245, 255, 0.1)'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Malware_Attacks'], name='Malware', mode='lines+markers',
            line=dict(color=colors['malware'], width=3), marker=dict(size=8, symbol='circle', line=dict(width=2, color='white')),
            fill='tonexty', fillcolor='rgba(255, 0, 255, 0.1)'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Phishing_Attacks'], name='Phishing', mode='lines+markers',
            line=dict(color=colors['phishing'], width=3), marker=dict(size=8, symbol='triangle-up', line=dict(width=2, color='white'))), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Web_Attacks'], name='Web Attacks', mode='lines+markers',
            line=dict(color=colors['web'], width=3, dash='dot'), marker=dict(size=8, symbol='square', line=dict(width=2, color='white'))), row=1, col=1)
        
        fig.add_trace(go.Bar(x=df['Date'], y=df['Critical_CVEs'], name='Critical CVEs',
            marker=dict(color=colors['cves'], line=dict(color='white', width=1), opacity=0.8),
            text=df['Critical_CVEs'], textposition='outside', textfont=dict(color=colors['cves'], size=10)), row=2, col=1)
        
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Patch_Delay_Days'], name='Patch Delay (Days)', mode='lines+markers',
            line=dict(color=colors['patch'], width=3), marker=dict(size=10, symbol='x', line=dict(width=2, color='white')), yaxis='y3'), row=2, col=1)
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(11, 17, 32, 0.8)',
            font=dict(family='JetBrains Mono, monospace', color=colors['text'], size=11),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                bgcolor='rgba(11, 17, 32, 0.9)', bordercolor='#06b6d4', borderwidth=1, font=dict(size=10)),
            margin=dict(l=60, r=60, t=100, b=60),
            hovermode='x unified',
            xaxis=dict(showgrid=True, gridcolor=colors['grid'], gridwidth=1, zeroline=False, linecolor='#06b6d4', linewidth=2, tickfont=dict(size=10)),
            xaxis2=dict(showgrid=True, gridcolor=colors['grid'], gridwidth=1, zeroline=False, linecolor='#06b6d4', linewidth=2, title='Timeline'),
            yaxis=dict(title=dict(text='Attack Volume', font=dict(size=12, color='#06b6d4')), showgrid=True, gridcolor=colors['grid'], gridwidth=1, zeroline=False, linecolor='#06b6d4', linewidth=2, tickfont=dict(size=10)),
            yaxis2=dict(title=dict(text='CVE Count', font=dict(size=12, color='#ff1744')), showgrid=True, gridcolor=colors['grid'], gridwidth=1, zeroline=False, linecolor='#ff1744', linewidth=2, tickfont=dict(size=10)),
            yaxis3=dict(title=dict(text='Patch Delay (Days)', font=dict(size=12, color='#ffd700')), overlaying='y2', side='right', showgrid=False, zeroline=False, linecolor='#ffd700', linewidth=2, tickfont=dict(size=10, color='#ffd700')),
            title=dict(text='📊 CYBER THREAT INTELLIGENCE // VISUALIZATION MODULE', font=dict(size=16, color='#06b6d4', family='JetBrains Mono'), x=0.5, xanchor='center'),
            shapes=[dict(type='line', x0=0, x1=1, y0=1.15, y1=1.15, xref='paper', yref='paper', line=dict(color='#06b6d4', width=2)),
                   dict(type='line', x0=0, x1=1, y0=-0.15, y1=-0.15, xref='paper', yref='paper', line=dict(color='#06b6d4', width=2))],
            annotations=[dict(text='◢ LIVE DATA STREAM ◣', xref='paper', yref='paper', x=0.5, y=1.08, showarrow=False,
                font=dict(size=10, color='#00f5ff', family='monospace'), bgcolor='rgba(0,0,0,0.5)', bordercolor='#00f5ff', borderwidth=1, borderpad=4)]
        )
        
        fig.update_traces(hovertemplate='<b>%{data.name}</b><br>Date: %{x|%Y-%m}<br>Value: %{y:,}<br><extra></extra>')
        
        return fig
        
    except Exception as e:
        st.error(f"⚠️ Graph Generation Error: {str(e)}")
        return None

# =====================================
# TAB CONTENT FUNCTIONS - COMPACT LAYOUT
# =====================================
def render_home():
    # Compact header
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
        <span style="font-size: 1.5rem;">🛡️</span>
        <div>
            <h1 style="margin: 0; font-size: 1.4rem;">CYBER THREAT INTELLIGENCE</h1>
            <p style="margin: 0; font-size: 0.8rem; color: #94a3b8;">Operator: Stephen Musau Makau | MSc Cybersecurity</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    mode_indicator = "🌙 DARK" if st.session_state.dark_mode else "☀️ LIGHT"
    view_indicator = "📱 MOBILE" if st.session_state.mobile_view else "💻 DESKTOP"
    data_indicator = "📊 CUSTOM" if get_current_data_source() == "uploaded" else "📋 DEFAULT"
    st.caption(f"⏱️ {datetime.now(ZoneInfo('Africa/Nairobi')).strftime('%d/%m/%Y %H:%M')} EAT | {mode_indicator} | {view_indicator} | {data_indicator}")

    try:
        prediction = predict_2027()
        accuracy = get_model_accuracy() * 100
        all_results = get_results()
    except Exception as e:
        st.error(f"⚠️ System Error: {e}")
        prediction = "Unknown"
        accuracy = 0.0
        all_results = {}

    if all_results:
        max_acc = max(all_results.values())
        if max_acc >= 0.999:
            best_algorithms = [name for name, acc in all_results.items() if acc >= 0.999]
            best_acc_pct = 100.0
        else:
            best_algorithms = [name for name, acc in all_results.items() if acc == max_acc]
            best_acc_pct = max_acc * 100
    else:
        best_algorithms = ["XGBoost"]
        best_acc_pct = accuracy

    # THREAT PROJECTION - Compact
    st.markdown("### 🚨 THREAT PROJECTION // 2027")
    
    if prediction == "High":
        st.markdown("""
        <div class="threat-high" style="padding: 15px; border-radius: 8px; margin: 8px 0;">
            <h2 style="margin: 0; font-size: 1.3rem;">⚠️ HIGH RISK DETECTED</h2>
            <p style="margin: 5px 0 0 0; font-size: 0.9rem;">Significant escalation predicted. Immediate countermeasures required.</p>
        </div>
        """, unsafe_allow_html=True)
        st.warning("⚠️ Attack volumes >4,000 DDoS + economic pressure = HIGH threat historically.")
        
    elif prediction == "Critical":
        st.markdown("""
        <div class="threat-critical" style="padding: 15px; border-radius: 8px; margin: 8px 0;">
            <h2 style="margin: 0; font-size: 1.3rem;">🛑 CRITICAL ALERT</h2>
            <p style="margin: 5px 0 0 0; font-size: 0.9rem;">Maximum threat level. Emergency protocols activated.</p>
        </div>
        """, unsafe_allow_html=True)
        st.error("🛑 High CVEs (95) + low patch compliance (9 days) + instability = CRITICAL historically.")
        
    else:
        st.markdown("""
        <div class="threat-moderate" style="padding: 15px; border-radius: 8px; margin: 8px 0;">
            <h2 style="margin: 0; font-size: 1.3rem;">✅ STABLE STATUS</h2>
            <p style="margin: 5px 0 0 0; font-size: 0.9rem;">Parameters within acceptable ranges. Standard monitoring sufficient.</p>
        </div>
        """, unsafe_allow_html=True)
        st.success("✅ Manageable threat levels. Maintain standard operations.")

    # VISUALIZATION
    st.markdown("### 📡 THREAT VISUALIZATION")
    
    viz_data = get_dataset()
    fig = create_futuristic_graph(viz_data)
    if fig is not None:
        chart_key = f"threat_graph_{get_current_data_source()}_{len(viz_data)}"
        st.plotly_chart(fig, use_container_width=True, height=500, key=chart_key)

    # METRICS - Compact 3-column
    st.markdown("### 📡 SYSTEM METRICS")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        algo_code = best_algorithms[0].upper().replace(" ", "_") if len(best_algorithms) == 1 else "ENSEMBLE"
        st.metric("Core Algorithm", algo_code, f"{best_acc_pct:.0f}% Acc")
    with c2:
        st.metric("Target Year", "2027", "Forecast")
    with c3:
        st.metric("Data Source", get_current_data_source().upper(), f"{len(viz_data)} Records")

    # Methodology - Collapsed by default
    with st.expander("🧠 Prediction Methodology", expanded=False):
        st.info("""
        1. **Data Input**: 12 parameters (attacks, CVEs, economic indicators)
        2. **Pattern Recognition**: Compare projections to historical patterns
        3. **Classification**: Categorize as Moderate/High/Critical
        4. **Confidence**: Based on model validation accuracy
        """)

def render_data_upload():
    st.title("📤 DATA UPLOAD")
    
    # Two columns side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Download Template")
        st.markdown("""
        Required columns: Year, Month, DDoS_Attacks, Malware_Attacks, Phishing_Attacks, 
        Web_Attacks, Critical_CVEs, Patch_Delay_Days, Traffic_Volume, Inflation_Rate, 
        GDP_Growth, Economic_Environment, Threat_Level
        """)
        template_buffer = generate_excel_template()
        st.download_button("📥 DOWNLOAD TEMPLATE", template_buffer, "cyber_threat_template.xlsx",
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    with col2:
        st.markdown("### 📤 Upload Your Data")
        uploaded_file = st.file_uploader("Choose file (.xlsx or .csv)", type=['xlsx', 'csv'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_preview = pd.read_csv(uploaded_file)
                else:
                    df_preview = pd.read_excel(uploaded_file)
                
                validation_result = validate_uploaded_data(df_preview)
                
                if validation_result['valid']:
                    st.success("✅ Valid")
                    if st.button("🚀 TRAIN MODEL", type="primary"):
                        with st.spinner("Training..."):
                            success, message = train_on_uploaded_data(df_preview)
                            if success:
                                st.session_state.uploaded_data = df_preview
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
                else:
                    st.error(f"❌ {validation_result['message']}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    if st.button("↩️ RESET TO DEFAULT", use_container_width=True):
        st.session_state.uploaded_data = None
        train_on_uploaded_data(None)
        st.success("✅ Reset complete")
        st.rerun()

def render_overview():
    st.title("📄 PROJECT OVERVIEW")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎓 Research Context
        **MSc Cybersecurity thesis at Mount Kenya University**
        
        **Problem**: Traditional defenses only react after attacks occur.
        
        **Solution**: ML predicts threats 2 years in advance for proactive defense.
        
        **New**: Upload your own data for custom predictions!
        """)
        
        st.markdown("""
        ### ⚠️ Threat Landscape
        Digital transformation correlates with exponential threat growth. 
        Conventional reactive defenses are inadequate against modern automated attacks.
        """)
    
    with col2:
        st.markdown("""
        ### 🔬 System Architecture
        - **Attack Vectors**: DDoS, Malware, Phishing, Web Exploits
        - **Vulnerability**: CVE Criticality, Patch Latency
        - **Network**: Traffic Anomaly Detection
        - **Economic**: Inflation/GDP correlation
        
        **Data Flow**: Raw → Feature Engineering → Training → Prediction → Alert
        """)
        
        st.markdown("""
        ### 🌍 Operational Impact
        - 🏛️ Critical Infrastructure Protection
        - 💰 Resource Optimization
        - 📋 Policy Intelligence
        - 🤝 Public Trust Maintenance
        - 📊 Custom Organization Analysis
        """)

def render_dataset():
    st.title("📊 DATA MATRIX")
    
    current_df = get_dataset()
    
    if get_current_data_source() == "uploaded":
        st.success("📊 Your uploaded custom dataset")
    else:
        st.info("📊 Default research dataset (2020-2025)")
    
    st.dataframe(current_df, use_container_width=True, height=350)
    
    # Patterns and stats in columns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📈 Key Patterns")
        patterns = analyze_dataset_patterns(current_df)
        for pattern in patterns[:3]:  # Show top 3 only
            st.markdown(f"• {pattern}")
    
    with col2:
        st.markdown("### 📊 Statistics")
        c1, c2 = st.columns(2)
        c1.metric("Records", len(current_df))
        c2.metric("Year Range", f"{current_df['Year'].min():.0f}-{current_df['Year'].max():.0f}" if 'Year' in current_df.columns else "N/A")

def render_models():
    st.title("🤖 AI CORE PERFORMANCE")
    
    results = get_results()
    
    if results:
        max_acc = max(results.values())
        active_algorithms = [name for name, acc in results.items() if acc == max_acc]
        active_acc = max_acc * 100
    else:
        active_algorithms = ["XGBoost"]
        active_acc = 0.0
    
    # Winner announcement
    if len(active_algorithms) == 1:
        st.success(f"🏆 ACTIVE: {active_algorithms[0]} | Accuracy: {active_acc:.2f}%")
    else:
        st.success(f"🏆 ENSEMBLE: {' + '.join(active_algorithms)} | Accuracy: {active_acc:.2f}%")
    
    # Benchmark table
    st.markdown("### 📊 Algorithm Benchmarking")
    table_data = [{"Algorithm": name, "Accuracy": f"{v*100:.2f}%", "Status": "🟢 ACTIVE" if name in active_algorithms else "⚪ STANDBY"} 
                  for name, v in results.items()]
    st.table(table_data)
    
    # Algorithm descriptions
    with st.expander("📖 Algorithm Details", expanded=False):
        st.markdown("""
        **Logistic Regression**: Linear, fast, interpretable. Best for clear linear separability.
        
        **Random Forest**: Ensemble of 200 trees. Robust, balanced, reduces overfitting.
        
        **XGBoost**: Gradient boosting with regularization. Handles complex non-linear patterns.
        """)

def render_parameters():
    st.title("⚙️ SYSTEM PARAMETERS")
    
    try:
        current_prediction = predict_2027()
        params_df = get_parameters()
    except:
        current_prediction = "Unknown"
        params_df = get_parameters()
    
    st.markdown(f"**2027 Prediction**: {'🟢 MODERATE' if current_prediction == 'Medium' else '🟠 HIGH' if current_prediction == 'High' else '🔴 CRITICAL' if current_prediction == 'Critical' else '⚪ Unknown'}")
    
    st.dataframe(params_df, use_container_width=True, height=300)
    
    # Parameter significance
    with st.expander("⚠️ Parameter Significance", expanded=False):
        st.warning("""
        **High Impact**:
        1. **Patch Delay Days** (9) - Longer delays = more exploit time
        2. **Critical CVEs** (95) - More flaws = more opportunities
        
        **Medium Impact**:
        3. **Economic Environment** - Stress correlates with cybercrime
        4. **DDoS Attacks** (4,200) - Indicates attacker capability
        """)

# =====================================
# MAIN RENDERING BASED ON ACTIVE TAB
# =====================================
tab_functions = {
    "🏠 HOME": render_home,
    "📤 DATA UPLOAD": render_data_upload,
    "📄 PROJECT OVERVIEW": render_overview,
    "📊 DATASET": render_dataset,
    "🤖 AI MODELS": render_models,
    "⚙️ PARAMETERS": render_parameters
}

active_tab = st.session_state.active_tab
if active_tab in tab_functions:
    tab_functions[active_tab]()
else:
    render_home()

# =====================================
# FOOTER - DYNAMIC YEAR
# =====================================
st.divider()
st.markdown(f"""
<div style="text-align: center; color: #64748b; padding: 15px;">
    <p style="margin: 0; font-size: 0.8rem;">
        🛡️ MKU Cyber Threat Intelligence | © {current_year} Mount Kenya University
    </p>
    <p style="margin: 5px 0 0 0; font-size: 0.7rem;">
        Data source: {get_current_data_source()} | Built with Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
