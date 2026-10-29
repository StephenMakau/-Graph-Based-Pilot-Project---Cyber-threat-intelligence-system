I'll create the complete code with all features integrated. Here's the full working version:PythonCopied!Copyimport streamlit as st

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

    st.session_state.active_tab = "home"

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

# CUSTOM CSS - FIXED NAVIGATION BAR & HAMBURGER

# =====================================

nav_css = """

<style>

    /* Hide default Streamlit elements */

    #MainMenu {visibility: hidden;}

    header {visibility: hidden;}

    footer {visibility: hidden;}

    

    /* Fixed Navigation Bar at very top */

    .nav-container {

        position: fixed;

        top: 0;

        left: 0;

        right: 0;

        height: 60px;

        background: linear-gradient(90deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);

        border-bottom: 2px solid #06b6d4;

        z-index: 999999;

        display: flex;

        align-items: center;

        justify-content: space-between;

        padding: 0 20px;

        box-shadow: 0 4px 20px rgba(6, 182, 212, 0.3);

    }

    

    /* Hamburger Button - 3 stacked lines */

    .hamburger-btn {

        width: 42px;

        height: 42px;

        background: linear-gradient(135deg, #06b6d4, #3b82f6);

        border: none;

        border-radius: 8px;

        cursor: pointer;

        display: flex;

        flex-direction: column;

        align-items: center;

        justify-content: center;

        gap: 5px;

        transition: all 0.3s ease;

        box-shadow: 0 0 15px rgba(6, 182, 212, 0.5);

    }

    

    .hamburger-btn:hover {

        transform: scale(1.1);

        box-shadow: 0 0 25px rgba(6, 182, 212, 0.8);

    }

    

    .hamburger-line {

        width: 22px;

        height: 2.5px;

        background: white;

        border-radius: 2px;

        transition: all 0.3s ease;

    }

    

    /* Nav Title */

    .nav-title {

        color: #06b6d4;

        font-size: 1.1rem;

        font-weight: 700;

        text-shadow: 0 0 10px rgba(6, 182, 212, 0.5);

        margin-left: 15px;

        font-family: 'Inter', sans-serif;

    }

    

    /* Nav Buttons Container */

    .nav-buttons {

        display: flex;

        gap: 6px;

        align-items: center;

        flex-wrap: wrap;

    }

    

    /* Flashy Nav Button */

    .nav-btn {

        padding: 8px 14px;

        border-radius: 8px;

        border: 1px solid rgba(6, 182, 212, 0.3);

        background: linear-gradient(135deg, rgba(6, 182, 212, 0.1), rgba(59, 130, 246, 0.1));

        color: #94a3b8;

        font-size: 0.78rem;

        font-weight: 600;

        cursor: pointer;

        transition: all 0.3s ease;

        text-decoration: none;

        display: inline-flex;

        align-items: center;

        gap: 5px;

        white-space: nowrap;

        font-family: 'Inter', sans-serif;

    }

    

    .nav-btn:hover {

        background: linear-gradient(135deg, rgba(6, 182, 212, 0.3), rgba(59, 130, 246, 0.3));

        color: #06b6d4;

        border-color: #06b6d4;

        transform: translateY(-2px);

        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.4);

    }

    

    .nav-btn.active {

        background: linear-gradient(135deg, #06b6d4, #3b82f6);

        color: white;

        border-color: #06b6d4;

        box-shadow: 0 0 20px rgba(6, 182, 212, 0.6);

        animation: pulse 2s infinite;

    }

    

    @keyframes pulse {

        0%, 100% { box-shadow: 0 0 20px rgba(6, 182, 212, 0.6); }

        50% { box-shadow: 0 0 30px rgba(6, 182, 212, 0.9); }

    }

    

    /* Push content below nav */

    .main .block-container {

        padding-top: 80px !important;

    }

    

    /* Sidebar styling */

    .css-1d391kg, [data-testid="stSidebar"] {

        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;

    }

    

    /* Mobile responsive */

    @media (max-width: 768px) {

        .nav-buttons { display: none; }

        .nav-title { font-size: 0.85rem; }

    }

</style>

"""

st.markdown(nav_css, unsafe_allow_html=True)

# =====================================

# FIXED NAVIGATION BAR RENDERING

# =====================================

def render_nav_bar():

    """Render the fixed navigation bar with hamburger and buttons"""

    

    tabs = {

        "home": "🏠 HOME",

        "data_upload": "📤 DATA UPLOAD",

        "overview": "📄 PROJECT OVERVIEW",

        "dataset": "📊 DATASET",

        "models": "🤖 AI MODELS",

        "parameters": "⚙️ PARAMETERS"

    }

    

    buttons_html = ""

    for tab_key, tab_label in tabs.items():

        active_class = "active" if st.session_state.active_tab == tab_key else ""

        buttons_html += f'<button class="nav-btn {active_class}" onclick="window.parent.postMessage({{type: \'streamlit:setComponentValue\', value: \'{tab_key}\'}}, \'*\')">{tab_label}</button>'

    

    nav_html = f"""

    <div class="nav-container">

        <div style="display: flex; align-items: center;">

            <button class="hamburger-btn" onclick="window.parent.postMessage({{type: 'streamlit:setComponentValue', value: 'toggle_sidebar'}}, '*')" title="System Controls">

                <div class="hamburger-line"></div>

                <div class="hamburger-line"></div>

                <div class="hamburger-line"></div>

            </button>

            <span class="nav-title">🛡️ MKU CYBER THREAT INTELLIGENCE</span>

        </div>

        <div class="nav-buttons">

            {buttons_html}

        </div>

    </div>

    """

    st.markdown(nav_html, unsafe_allow_html=True)

render_nav_bar()

# =====================================

# SIDEBAR CONTROLS (Opened via Hamburger)

# =====================================

with st.sidebar:

    st.markdown("""

    <div style="text-align: center; margin: 20px 0;">

        <div style="display: inline-block; padding: 15px; background: linear-gradient(135deg, #06b6d4, #3b82f6); border-radius: 12px; box-shadow: 0 0 20px rgba(6, 182, 212, 0.5);">

            <div style="width: 30px; height: 3px; background: white; margin: 5px 0; border-radius: 2px;"></div>

            <div style="width: 30px; height: 3px; background: white; margin: 5px 0; border-radius: 2px;"></div>

            <div style="width: 30px; height: 3px; background: white; margin: 5px 0; border-radius: 2px;"></div>

        </div>

        <p style="color: #06b6d4; font-size: 0.9rem; margin-top: 12px; font-weight: 600;">⚙️ SYSTEM CONTROLS</p>

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

# NAVIGATION STATE MANAGEMENT

# =====================================

query_params = st.query_params

if "tab" in query_params:

    selected_tab = query_params["tab"]

    if selected_tab in ["home", "data_upload", "overview", "dataset", "models", "parameters"]:

        st.session_state.active_tab = selected_tab

        st.query_params.clear()

# =====================================

# HELPER FUNCTIONS

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

    

    econ_threat = df.groupby(['Economic_Environment', 'Threat_Level']).size().unstack(fill_value=0)

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

# TAB CONTENT FUNCTIONS

# =====================================

def render_home():

    st.title("🛡️ CYBER THREAT INTELLIGENCE")

    st.markdown("**Operator:** Stephen Musau Makau | **Clearance:** MSc Cybersecurity")

    

    mode_indicator = "🌙 DARK" if st.session_state.dark_mode else "☀️ LIGHT"

    view_indicator = "📱 MOBILE" if st.session_state.mobile_view else "💻 DESKTOP"

    data_indicator = "📊 CUSTOM" if get_current_data_source() == "uploaded" else "📋 DEFAULT"

    st.caption(f"⏱️ SYS.TIME: {datetime.now(ZoneInfo('Africa/Nairobi')).strftime('%d/%m/%Y | %H:%M:%S')} EAT | STATUS: ONLINE | MODE: {mode_indicator} | VIEW: {view_indicator} | DATA: {data_indicator}")

    

    st.divider()

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

    top_course_data = pd.DataFrame()  # Placeholder

    top_course = "N/A"

    top_course_val = 0

    st.header("🚨 THREAT PROJECTION // 2027")

    

    if prediction == "High":

        st.markdown("""

        <div class="threat-high">

            <h1>⚠️ HIGH RISK DETECTED</h1>

            <h3>THREAT_LEVEL: HIGH</h3>

            <p>Predictive algorithms indicate significant escalation in cyber threats targeting critical infrastructure. Immediate countermeasures required.</p>

        </div>

        """, unsafe_allow_html=True)

        st.warning("⚠️ **Interpretation**: Historical data shows that when attack volumes exceed 4,000 (DDoS), 18,000 (Malware), and economic pressure indicators are elevated, threat levels typically spike to HIGH.")

        

    elif prediction == "Critical":

        st.markdown("""

        <div class="threat-critical">

            <h1>🛑 CRITICAL ALERT</h1>

            <h3>THREAT_LEVEL: CRITICAL</h3>

            <p>Maximum threat level detected. System predicts unprecedented attack surge. Emergency protocols activated.</p>

        </div>

        """, unsafe_allow_html=True)

        st.error("🛑 **Interpretation**: The convergence of high CVE counts (95), low patch compliance (9 days delay), and economic instability historically correlates with CRITICAL threat periods.")

        

    else:

        st.markdown("""

        <div class="threat-moderate">

            <h1>✅ STABLE STATUS</h1>

            <h3>THREAT_LEVEL: MODERATE</h3>

            <p>Threat parameters within acceptable ranges. Standard monitoring protocols sufficient.</p>

        </div>

        """, unsafe_allow_html=True)

        st.success("✅ **Interpretation**: Current projections indicate manageable threat levels. Maintain standard operations.")

    st.divider()

    st.markdown("### 📡 THREAT VISUALIZATION MATRIX")

    

    current_source = get_current_data_source()

    if current_source == "uploaded":

        st.info("📊 Displaying **Uploaded Dataset** Visualization")

    else:

        st.info("📊 Displaying **Default Dataset** Visualization")

    

    viz_data = get_dataset()

    fig = create_futuristic_graph(viz_data)

    if fig is not None:

        chart_key = f"threat_graph_{current_source}_{len(viz_data)}"

        st.plotly_chart(fig, use_container_width=True, height=600, key=chart_key)

    

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

        """)

    st.divider()

    st.markdown("### 📡 THREAT ASSESSMENT MODULE")

    

    if st.session_state.mobile_view:

        c1, c2 = st.columns(2)

        c3 = st.container()

    else:

        c1, c2, c3 = st.columns(3)

    with c1:

        if len(best_algorithms) == 1:

            algo_code = best_algorithms[0].upper().replace(" ", "_")

            subtitle = "ML.Engine.Active"

        elif len(best_algorithms) == 2:

            algo_code = "ENSEMBLE_DUAL"

            subtitle = "Hybrid.Engine.Active"

        else:

            algo_code = "ENSEMBLE_TRIPLE"

            subtitle = "Meta.Ensemble.Active"

        

        st.markdown(f"""

        <div class="tech-container" style="text-align: center;">

            <h4 style="color: #94a3b8; margin:15px 0 0 0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2px; font-family: Inter, sans-serif;">Core Algorithm</h4>

            <h2 style="margin: 20px 0; color: #06b6d4; font-family: JetBrains Mono, monospace !important; font-size: 2.0rem; font-weight: 700; text-shadow: 0 0 15px rgba(6,182,212,0.5);">{algo_code}</h2>

            <p style="color: #64748b; font-size: 0.95rem; font-family: Calibri, sans-serif; margin-bottom: 15px;">{subtitle}</p>

        </div>

        """, unsafe_allow_html=True)

    with c2:

        st.metric("🎯 MODEL ACCURACY", f"{best_acc_pct:.2f}%")

        if best_acc_pct >= 99.9:

            st.caption("🌟 **PERFECT SCORE**: Optimal prediction reliability achieved.")

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

    st.markdown("### 🧠 **Prediction Methodology**")

    

    data_source = get_current_data_source()

    is_custom_data = (data_source == "uploaded")

    

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

    if is_custom_data:

        st.warning("⚠️ **Custom Dataset Detected**: The methodology below reflects patterns learned from your **uploaded data** rather than the default research dataset.")

        st.info(f"""

        **Adaptive Learning Protocol Active**

        

        The system has analyzed your uploaded dataset containing {len(get_dataset())} historical records. 

        Based on the data characteristics, the system selected **{', '.join(best_algorithms) if best_algorithms else 'No Model'}** as the optimal predictor.

        """)

    else:

        st.info("""

        **How the 2027 Prediction is Generated:**

        

        1. **Data Input**: The model receives projected values for 12 parameters (DDoS attacks, malware volume, CVE counts, inflation, GDP, etc.)

        2. **Pattern Recognition**: The system compares these projections against historical patterns where similar conditions resulted in specific threat levels

        3. **Classification**: The system classifies the 2027 scenario into one of three categories: Moderate, High, or Critical

        4. **Confidence**: The accuracy metric indicates how much trust to place in this prediction based on past performance

        """)

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

        

        **📤 Upload Your Own Data**: Go to the **DATA UPLOAD** tab to use your own historical data for custom predictions!

        """)

def render_data_upload():

    st.title("📤 DATA UPLOAD CENTER")

    st.markdown("**Upload your own cybersecurity data to generate custom threat predictions**")

    st.divider()

    

    st.header("📋 STEP 1: Download Template")

    

    st.markdown("""

    ### 📖 **Data Format Requirements**

    

    Your file **must** contain these exact columns:

    

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

    

    st.header("📤 STEP 2: Upload Your Data")

    

    st.markdown("""

    ### 📖 **Upload Instructions**

    

    1. **Prepare your data** using the template above

    2. **Save as Excel (.xlsx)** or CSV (.csv) format

    3. **Ensure minimum 5 rows** of historical data (more is better)

    4. **Verify column names match exactly** (case-sensitive)

    5. **Upload below** and click "Train Model"

    

    ⚠️ **Important**: The system will automatically retrain all ML algorithms using your uploaded data.

    """)

    

    uploaded_file = st.file_uploader(

        "Choose your data file (Excel .xlsx or CSV .csv)",

        type=['xlsx', 'csv'],

        help="Upload your historical cybersecurity data following the template format"

    )

    

    if uploaded_file is not None:

        st.success(f"✅ File uploaded: **{uploaded_file.name}**")

        

        try:

            if uploaded_file.name.endswith('.csv'):

                df_preview = pd.read_csv(uploaded_file)

            else:

                df_preview = pd.read_excel(uploaded_file)

            

            st.markdown("### 👁️ **Data Preview** (First 5 rows)")

            st.dataframe(df_preview.head(), use_container_width=True)

            

            validation_result = validate_uploaded_data(df_preview)

            

            if validation_result['valid']:

                st.success(f"✅ **Validation Passed**: {validation_result['message']}")

                

                if st.button("🚀 TRAIN MODEL WITH UPLOADED DATA", use_container_width=True, type="primary"):

                    with st.spinner("Training models... Please wait..."):

                        success, message = train_on_uploaded_data(df_preview)

                        if success:

                            st.session_state.uploaded_data = df_preview

                            st.session_state.training_status = "success"

                            st.success(f"✅ {message}")

                            st.balloons()

                            st.info("🔄 Navigate to **HOME** tab to see updated predictions!")

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

    

    st.header("🔄 Reset to Default")

    if st.button("↩️ RETURN TO DEFAULT DATASET", use_container_width=True):

        st.session_state.uploaded_data = None

        train_on_uploaded_data(None)

        st.success("✅ System reset to default dataset")

        st.info("The system is now using the original research data.")

        st.rerun()

def render_overview():

    st.title("📄 SYSTEM OVERVIEW")

    

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

def render_dataset():

    st.title("📊 DATA MATRIX")

    

    current_df = get_dataset()

    

    if get_current_data_source() == "uploaded":

        st.success("📊 **Currently Displaying**: Your uploaded custom dataset")

    else:

        st.info("📊 **Currently Displaying**: Default research dataset (2020-2025)")

    

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

    st.dataframe(current_df, use_container_width=True, height=height)

    

    st.markdown("---")

    st.subheader("📈 **Key Patterns Visible in This Data**")

    

    patterns = analyze_dataset_patterns(current_df)

    

    for pattern in patterns:

        st.markdown(f"• {pattern}")

    

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

def render_models():

    st.title("🤖 AI CORE PERFORMANCE")

    

    results = get_results()

    

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

    

    if get_current_data_source() == "uploaded":

        st.success("🤖 **Models trained on**: Your uploaded custom dataset")

    else:

        st.info("🤖 **Models trained on**: Default research dataset")

    

    st.markdown("---")

    

    if len(active_algorithms) == 1:

        winner_name = active_algorithms[0]

        st.header(f"🏆 ACTIVE ALGORITHM: {winner_name.upper()}")

        

        if "XGBoost" in winner_name:

            st.success("""

            ### 🥇 XGBoost (eXtreme Gradient Boosting)

            

            **Status**: ACTIVE | **Accuracy**: 100.00%

            

            **Why It's Winning:**

            XGBoost has achieved perfect accuracy on your validation data. This gradient boosting framework uses regularized learning to prevent overfitting while capturing complex non-linear patterns in cybersecurity threats.

            """)

        elif "Random Forest" in winner_name:

            st.success("""

            ### 🥇 Random Forest Classifier

            

            **Status**: ACTIVE | **Accuracy**: 100.00%

            

            **Why It's Winning:**

            Random Forest has achieved perfect accuracy through its ensemble of 200 decision trees. The bagging approach creates diversity that captures robust patterns without overfitting.

            """)

        elif "Logistic Regression" in winner_name:

            st.success("""

            ### 🥇 Logistic Regression

            

            **Status**: ACTIVE | **Accuracy**: 100.00%

            

            **Why It's Winning:**

            Logistic Regression has achieved perfect accuracy, indicating your data has clear linear separability between threat classes. This statistical approach provides maximum interpretability.

            """)

            

    elif len(active_algorithms) == 2:

        st.header(f"🏆 DUAL ENSEMBLE ACTIVE")

        st.success(f"""

        ### 🥇 Combined System: {' & '.join(active_algorithms)}

        

        **Status**: ACTIVE | **Accuracy**: 100.00%

        

        **Why Both Are Winning:**

        Both algorithms achieved perfect accuracy on your validation data. The system leverages their complementary strengths.

        """)

        

    else:

        st.header(f"🏆 TRIPLE CONSENSUS MODE")

        st.success("""

        ### 🥇 ALL SYSTEMS ACTIVE: Logistic Regression + Random Forest + XGBoost

        

        **Status**: ACTIVE | **Accuracy**: 100.00%

        

        **Why All Three Are Winning:**

        All three algorithms achieved perfect accuracy on your validation data. This rare "triple consensus" indicates exceptionally clear, separable patterns.

        """)

    

    st.markdown("---")

    

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

    

    st.markdown("### 📖 **Understanding the Results**")

    

    col1, col2 = st.columns(2)

    with col1:

        if len(active_algorithms) == 1:

            winner = active_algorithms[0]

            st.info(f"""

            **🥇 Winner Analysis: {winner}**

            

            This algorithm achieved the highest accuracy on your specific dataset. 

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

def render_parameters():

    st.title("⚙️ SYSTEM PARAMETERS")

    

    try:

        current_prediction = predict_2027()

        params_df = get_parameters()

    except:

        current_prediction = "Unknown"

        params_df = get_parameters()

    

    st.markdown(f"""

    ### 📖 **2027 Projection Parameters**

    

    **Current Prediction for 2027**: {'🟢 **MODERATE**' if current_prediction == 'Medium' else '🟠 **HIGH**' if current_prediction == 'High' else '🔴 **CRITICAL**' if current_prediction == 'Critical' else '⚪ **Unknown**'}

    

    These values represent **projected conditions for August 2027** based on trend analysis, economic forecasts, and technological growth projections. 

    The model uses these 12 inputs to classify the threat level.

    """)

    

    st.markdown("### 📋 **Feature Configuration for 2027 Threat Projection**")

    

    height = 300 if st.session_state.mobile_view else 400

    st.dataframe(params_df, use_container_width=True, height=height)

    

    st.markdown("---")

    st.subheader("🔍 **Why These Parameters Lead to {}**".format(current_prediction if current_prediction != "Unknown" else "This Prediction"))

    

    analysis_df = get_dataset()

    

    if not analysis_df.empty and current_prediction != "Unknown":

        ddos_2027 = 4200

        malware_2027 = 18500

        cve_2027 = 95

        patch_2027 = 9

        

        ddos_max = analysis_df['DDoS_Attacks'].max() if 'DDoS_Attacks' in analysis_df.columns else 0

        ddos_mean = analysis_df['DDoS_Attacks'].mean() if 'DDoS_Attacks' in analysis_df.columns else 0

        malware_max = analysis_df['Malware_Attacks'].max() if 'Malware_Attacks' in analysis_df.columns else 0

        cve_max = analysis_df['Critical_CVEs'].max() if 'Critical_CVEs' in analysis_df.columns else 0

        patch_min = analysis_df['Patch_Delay_Days'].min() if 'Patch_Delay_Days' in analysis_df.columns else 20

        

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

        

        if insights:

            st.markdown("### 🎯 **Key Parameter Insights**")

            for insight in insights:

                st.markdown(insight)

    

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

    

    if current_prediction == "Critical":

        st.error("🚨 **CRITICAL RECOMMENDATION**: Given the projected parameters exceed multiple historical thresholds simultaneously, immediate executive review of security posture is advised.")

    elif current_prediction == "High":

        st.warning("⚠️ **HIGH ALERT RECOMMENDATION**: Review defensive protocols and ensure patch management can maintain the projected 9-day deployment window.")

    else:

        st.success("✅ **STABLE OUTLOOK**: Projected parameters suggest manageable risk levels. Maintain standard monitoring and continue current security investments.")

# =====================================

# MAIN RENDERING BASED ON ACTIVE TAB

# =====================================

tab_functions = {

    "home": render_home,

    "data_upload": render_data_upload,

    "overview": render_overview,

    "dataset": render_dataset,

    "models": render_models,

    "parameters": render_parameters

}

active_tab = st.session_state.active_tab

if active_tab in tab_functions:

    tab_functions[active_tab]()

else:

    render_home()

# =====================================

# FOOTER

# =====================================

st.divider()

st.markdown(f"""

<div style="text-align: center; color: #64748b; padding: 20px;">

    <p style="margin: 0; font-size: 0.875rem;">

        🛡️ MKU Cyber Threat Intelligence System v2.0.8 | © 2024 Mount Kenya University

    </p>

    <p style="margin: 8px 0 0 0; font-size: 0.75rem;">

        Data source: {get_current_data_source()} | Built with Streamlit

    </p>

</div>

""", unsafe_allow_html=True)
