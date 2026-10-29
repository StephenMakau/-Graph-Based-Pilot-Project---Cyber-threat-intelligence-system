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
    st.session_state.dark_mode = False  # LIGHT MODE AS DEFAULT
    
if 'mobile_view' not in st.session_state:
    st.session_state.mobile_view = False

if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

if 'training_status' not in st.session_state:
    st.session_state.training_status = None

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "home"

if 'sidebar_open' not in st.session_state:
    st.session_state.sidebar_open = False

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
# CUSTOM CSS - COMPACT & SCROLLABLE
# =====================================
def get_css():
    if st.session_state.dark_mode:
        return """
        <style>
            .stApp { background: linear-gradient(135deg, #0b1120 0%, #1e293b 50%, #0f172a 100%); color: #f8fafc; }
            h1, h2, h3, h4, h5, h6 { color: #f8fafc !important; }
            h1 { font-size: 1.4rem !important; margin-bottom: 8px !important; background: linear-gradient(90deg, #06b6d4, #8b5cf6, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            h2 { font-size: 1.15rem !important; color: #06b6d4 !important; border-left: 3px solid #06b6d4; padding-left: 8px; margin: 12px 0 6px 0 !important; }
            h3 { font-size: 1rem !important; color: #8b5cf6 !important; margin: 10px 0 5px 0 !important; }
            p, li, div, span, label, .stMarkdown, .stAlert { color: #e2e8f0 !important; font-size: 0.9rem !important; line-height: 1.4 !important; }
            .stTabs [data-testid="stTab"] { color: #94a3b8; background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(6, 182, 212, 0.2); }
            .stTabs [data-testid="stTab"][aria-selected="true"] { background: linear-gradient(135deg, #06b6d4, #3b82f6); color: #ffffff !important; }
            [data-testid="stMetric"] { background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(6, 182, 212, 0.3); padding: 8px !important; }
            [data-testid="stMetricValue"] { color: #06b6d4 !important; font-size: 1.3rem !important; }
            [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.7rem !important; }
            div[data-testid="stDataFrame"], div[data-testid="stTable"] { background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(6, 182, 212, 0.2); }
            .threat-high { background: rgba(251, 188, 4, 0.1); border: 1px solid rgba(251, 188, 4, 0.5); border-radius: 8px; padding: 12px; margin: 8px 0; }
            .threat-critical { background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.5); border-radius: 8px; padding: 12px; margin: 8px 0; }
            .threat-moderate { background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.5); border-radius: 8px; padding: 12px; margin: 8px 0; }
            .stAlert { background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(6, 182, 212, 0.3); color: #06b6d4 !important; padding: 10px !important; }
            .tech-container { background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(6, 182, 212, 0.2); border-radius: 8px; padding: 12px; }
            strong, b { color: #06b6d4 !important; }
            .explanation-box { background: rgba(6, 182, 212, 0.1); border-left: 3px solid #06b6d4; padding: 10px; border-radius: 5px; margin: 8px 0; }
            .upload-box { background: rgba(139, 92, 246, 0.1); border: 2px dashed rgba(139, 92, 246, 0.5); border-radius: 8px; padding: 15px; text-align: center; }
            .template-box { background: rgba(16, 185, 129, 0.1); border: 2px solid rgba(16, 185, 129, 0.5); border-radius: 8px; padding: 15px; }
            /* Compact spacing */
            .block-container { padding-top: 70px !important; padding-bottom: 10px !important; }
            div[data-testid="stVerticalBlock"] > div { gap: 0.3rem !important; }
            hr { margin: 10px 0 !important; border-color: rgba(6, 182, 212, 0.2) !important; }
            /* Sidebar narrow */
            [data-testid="stSidebar"] { width: 280px !important; background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important; }
            /* Nav bar fixed */
            .nav-fixed { position: fixed; top: 0; left: 0; right: 0; height: 50px; background: linear-gradient(90deg, #0f172a 0%, #1e293b 50%, #0f172a 100%); border-bottom: 2px solid #06b6d4; z-index: 999999; display: flex; align-items: center; justify-content: space-between; padding: 0 15px; box-shadow: 0 4px 20px rgba(6, 182, 212, 0.3); }
            /* Table compact */
            .stTable td, .stTable th { padding: 6px !important; font-size: 0.85rem !important; }
            /* Expander compact */
            .streamlit-expanderHeader { padding: 8px 12px !important; font-size: 0.9rem !important; }
            .streamlit-expanderContent { padding: 10px 12px !important; }
        </style>
        """
    else:
        return """
        <style>
            .stApp { background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #f1f5f9 100%); color: #1e293b; }
            h1, h2, h3, h4, h5, h6 { color: #1e293b !important; }
            h1 { font-size: 1.4rem !important; margin-bottom: 8px !important; background: linear-gradient(90deg, #0369a1, #7c3aed, #2563eb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            h2 { font-size: 1.15rem !important; color: #0369a1 !important; border-left: 3px solid #0369a1; padding-left: 8px; margin: 12px 0 6px 0 !important; }
            h3 { font-size: 1rem !important; color: #7c3aed !important; margin: 10px 0 5px 0 !important; }
            p, li, div, span, label, .stMarkdown, .stAlert { color: #334155 !important; font-size: 0.9rem !important; line-height: 1.4 !important; }
            .stTabs [data-testid="stTab"] { color: #64748b; background: rgba(255, 255, 255, 0.8); border: 1px solid rgba(37, 99, 235, 0.2); }
            .stTabs [data-testid="stTab"][aria-selected="true"] { background: linear-gradient(135deg, #0369a1, #2563eb); color: #ffffff !important; }
            [data-testid="stMetric"] { background: rgba(255, 255, 255, 0.9); border: 1px solid rgba(37, 99, 235, 0.3); box-shadow: 0 2px 8px rgba(0,0,0,0.08); padding: 8px !important; }
            [data-testid="stMetricValue"] { color: #0369a1 !important; font-size: 1.3rem !important; }
            [data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.7rem !important; }
            div[data-testid="stDataFrame"], div[data-testid="stTable"] { background: rgba(255, 255, 255, 0.9); border: 1px solid rgba(37, 99, 235, 0.2); }
            .threat-high { background: rgba(251, 191, 36, 0.15); border: 1px solid rgba(251, 191, 36, 0.6); border-radius: 8px; padding: 12px; margin: 8px 0; }
            .threat-critical { background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.6); border-radius: 8px; padding: 12px; margin: 8px 0; }
            .threat-moderate { background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.6); border-radius: 8px; padding: 12px; margin: 8px 0; }
            .stAlert { background: rgba(219, 234, 254, 0.8); border: 1px solid rgba(37, 99, 235, 0.3); color: #1e40af !important; padding: 10px !important; }
            .tech-container { background: rgba(255, 255, 255, 0.9); border: 1px solid rgba(37, 99, 235, 0.2); border-radius: 8px; padding: 12px; }
            strong, b { color: #0369a1 !important; }
            .explanation-box { background: rgba(37, 99, 235, 0.1); border-left: 3px solid #0369a1; padding: 10px; border-radius: 5px; margin: 8px 0; }
            .upload-box { background: rgba(139, 92, 246, 0.05); border: 2px dashed rgba(139, 92, 246, 0.5); border-radius: 8px; padding: 15px; text-align: center; }
            .template-box { background: rgba(16, 185, 129, 0.05); border: 2px solid rgba(16, 185, 129, 0.5); border-radius: 8px; padding: 15px; }
            /* Compact spacing */
            .block-container { padding-top: 70px !important; padding-bottom: 10px !important; }
            div[data-testid="stVerticalBlock"] > div { gap: 0.3rem !important; }
            hr { margin: 10px 0 !important; border-color: rgba(37, 99, 235, 0.2) !important; }
            /* Sidebar narrow */
            [data-testid="stSidebar"] { width: 280px !important; background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%) !important; }
            /* Nav bar fixed */
            .nav-fixed { position: fixed; top: 0; left: 0; right: 0; height: 50px; background: linear-gradient(90deg, #f8fafc 0%, #e2e8f0 50%, #f1f5f9 100%); border-bottom: 2px solid #0369a1; z-index: 999999; display: flex; align-items: center; justify-content: space-between; padding: 0 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            /* Table compact */
            .stTable td, .stTable th { padding: 6px !important; font-size: 0.85rem !important; }
            /* Expander compact */
            .streamlit-expanderHeader { padding: 8px 12px !important; font-size: 0.9rem !important; }
            .streamlit-expanderContent { padding: 10px 12px !important; }
        </style>
        """

st.markdown(get_css(), unsafe_allow_html=True)

# =====================================
# FIXED NAVIGATION BAR WITH WORKING HAMBURGER
# =====================================
# Create columns that simulate a fixed nav bar
nav_cols = st.columns([1, 5, 10])

with nav_cols[0]:
    # Working hamburger button
    if st.button("☰", key="hamburger", help="Open Settings", type="secondary"):
        st.session_state.sidebar_open = True
        # Force sidebar to open
        st.components.v1.html("""
        <script>
            var sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
            if (sidebar) {
                var button = sidebar.querySelector('button[kind="header"]');
                if (button) button.click();
            }
        </script>
        """, height=0)

with nav_cols[1]:
    title_color = "#06b6d4" if st.session_state.dark_mode else "#0369a1"
    st.markdown(f'<span style="color: {title_color}; font-size: 1rem; font-weight: 700;">🛡️ MKU CYBER THREAT INTELLIGENCE</span>', unsafe_allow_html=True)

with nav_cols[2]:
    # Navigation tabs using radio
    tab_options = {
        "home": "🏠 HOME",
        "data_upload": "📤 UPLOAD",
        "overview": "📄 OVERVIEW",
        "dataset": "📊 DATA",
        "models": "🤖 MODELS",
        "parameters": "⚙️ PARAMS"
    }
    
    selected = st.radio(
        "Nav",
        list(tab_options.keys()),
        format_func=lambda x: tab_options[x],
        index=list(tab_options.keys()).index(st.session_state.active_tab) if st.session_state.active_tab in tab_options else 0,
        horizontal=True,
        label_visibility="collapsed",
        key="nav_radio_main"
    )
    
    if selected != st.session_state.active_tab:
        st.session_state.active_tab = selected
        st.rerun()

# =====================================
# SIDEBAR - NARROW WITH WORKING TOGGLES
# =====================================
with st.sidebar:
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 15px;">
        <div style="display: inline-block; padding: 10px; background: linear-gradient(135deg, #06b6d4, #3b82f6); border-radius: 8px;">
            <div style="width: 20px; height: 2px; background: white; margin: 3px 0;"></div>
            <div style="width: 20px; height: 2px; background: white; margin: 3px 0;"></div>
            <div style="width: 20px; height: 2px; background: white; margin: 3px 0;"></div>
        </div>
        <p style="color: #06b6d4; font-size: 0.8rem; margin-top: 8px; font-weight: 600;">⚙️ SETTINGS</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dark/Light mode toggle
    new_dark_mode = st.toggle(
        "🌙 Dark Mode" if not st.session_state.dark_mode else "☀️ Light Mode",
        value=st.session_state.dark_mode
    )
    if new_dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = new_dark_mode
        st.rerun()
    
    st.caption("Switch between dark and light theme")
    
    st.markdown("---")
    
    # Mobile/Desktop toggle
    new_mobile = st.toggle(
        "📱 Mobile View",
        value=st.session_state.mobile_view
    )
    if new_mobile != st.session_state.mobile_view:
        st.session_state.mobile_view = new_mobile
        st.rerun()
    
    st.caption("Compact layout for small screens")
    
    st.markdown("---")
    
    # Data source indicator
    data_source = get_current_data_source()
    if data_source == "uploaded":
        st.success("📊 Using: **Custom Data**")
    else:
        st.info("📊 Using: **Default Data**")
    
    st.caption(f"Time: {datetime.now(ZoneInfo('Africa/Nairobi')).strftime('%H:%M')}")
    
    # Close button
    if st.button("✕ Close", use_container_width=True):
        st.session_state.sidebar_open = False
        st.rerun()

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
            patterns.append(f"**Critical Peak**: Only 'Critical' period in **{years[0]}** with DDoS {critical_rows['DDoS_Attacks'].max():,}, Malware {critical_rows['Malware_Attacks'].max():,}")
        else:
            patterns.append(f"**Critical Periods**: In **{', '.join(map(str, years))}** with peak DDoS {critical_rows['DDoS_Attacks'].max():,}")
    else:
        patterns.append("**No Critical Threats**: Dataset has no Critical classifications")
    
    high_cost_critical = df[(df['Economic_Environment'] == 'High_Cost') & (df['Threat_Level'] == 'Critical')]
    if not high_cost_critical.empty:
        patterns.append(f"**Economic Stress**: High_Cost periods show {len(high_cost_critical)} Critical classification(s)")
    
    low_patch = df[df['Patch_Delay_Days'] < 10]
    if not low_patch.empty:
        threat_dist = low_patch['Threat_Level'].value_counts()
        dominant = threat_dist.index[0] if not threat_dist.empty else "Unknown"
        patterns.append(f"**Patch Speed**: Delays <10 days trend toward **{dominant}** ({threat_dist.get(dominant, 0)}/{len(low_patch)} periods)")
    
    df_sorted = df.sort_values(['Year', 'Month'])
    first_ddos = df_sorted['DDoS_Attacks'].iloc[0] if len(df_sorted) > 0 else 0
    last_ddos = df_sorted['DDoS_Attacks'].iloc[-1] if len(df_sorted) > 0 else 0
    
    if last_ddos > first_ddos * 1.5:
        patterns.append(f"**Escalation**: DDoS grew {first_ddos:,} → {last_ddos:,} (+{((last_ddos/first_ddos-1)*100):.0f}%)")
    elif last_ddos < first_ddos * 0.7:
        patterns.append(f"**Declining**: DDoS fell {first_ddos:,} → {last_ddos:,} (-{((1-last_ddos/first_ddos)*100):.0f}%)")
    else:
        patterns.append(f"**Stable**: DDoS between {first_ddos:,} - {last_ddos:,}")
    
    high_cve = df[df['Critical_CVEs'] > df['Critical_CVEs'].mean() + df['Critical_CVEs'].std()]
    if not high_cve.empty:
        high_threat_pct = (high_cve['Threat_Level'] != 'Medium').mean() * 100
        patterns.append(f"**Vulnerability Window**: High CVEs correlate with elevated threats {high_threat_pct:.0f}% of time")
    
    high_inflation = df[df['Inflation_Rate'] > 7.0]
    if not high_inflation.empty:
        threat_dist = high_inflation['Threat_Level'].value_counts()
        patterns.append(f"**Economic Pressure**: High inflation shows: {dict(threat_dist)}")
    
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
            return None

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=('⚡ ATTACK VECTORS', '🔒 VULNERABILITIES')
        )
        
        colors = {
            'ddos': '#00f5ff', 'malware': '#ff00ff', 'phishing': '#39ff14',
            'web': '#ff6b35', 'cves': '#ff1744', 'patch': '#ffd700',
            'grid': 'rgba(6, 182, 212, 0.1)', 'text': '#e2e8f0'
        }
        
        fig.add_trace(go.Scatter(x=df['Date'], y=df['DDoS_Attacks'], name='DDoS', mode='lines+markers',
            line=dict(color=colors['ddos'], width=2), marker=dict(size=6, symbol='diamond'),
            fill='tonexty', fillcolor='rgba(0, 245, 255, 0.1)'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Malware_Attacks'], name='Malware', mode='lines+markers',
            line=dict(color=colors['malware'], width=2), marker=dict(size=6, symbol='circle'),
            fill='tonexty', fillcolor='rgba(255, 0, 255, 0.1)'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Phishing_Attacks'], name='Phishing', mode='lines+markers',
            line=dict(color=colors['phishing'], width=2), marker=dict(size=6, symbol='triangle-up')), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Web_Attacks'], name='Web', mode='lines+markers',
            line=dict(color=colors['web'], width=2, dash='dot'), marker=dict(size=6, symbol='square')), row=1, col=1)
        
        fig.add_trace(go.Bar(x=df['Date'], y=df['Critical_CVEs'], name='CVEs',
            marker=dict(color=colors['cves'], opacity=0.8), text=df['Critical_CVEs'], textposition='outside'), row=2, col=1)
        
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Patch_Delay_Days'], name='Patch Delay', mode='lines+markers',
            line=dict(color=colors['patch'], width=2), marker=dict(size=8, symbol='x'), yaxis='y3'), row=2, col=1)
        
        fig.update_layout(
            template='plotly_dark' if st.session_state.dark_mode else 'plotly_white',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(11, 17, 32, 0.8)' if st.session_state.dark_mode else 'rgba(255,255,255,0.9)',
            font=dict(family='Inter, sans-serif', color=colors['text'] if st.session_state.dark_mode else '#334155', size=10),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(0,0,0,0.5)' if st.session_state.dark_mode else 'rgba(255,255,255,0.9)', bordercolor='#06b6d4', borderwidth=1, font=dict(size=9)),
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='x unified',
            height=450,
            title=dict(text='📊 THREAT INTELLIGENCE VISUALIZATION', font=dict(size=14, color='#06b6d4' if st.session_state.dark_mode else '#0369a1'), x=0.5),
            xaxis=dict(showgrid=True, gridcolor=colors['grid'], zeroline=False),
            xaxis2=dict(showgrid=True, gridcolor=colors['grid'], title='Timeline'),
            yaxis=dict(title='Attack Volume', showgrid=True, gridcolor=colors['grid']),
            yaxis2=dict(title='CVE Count', showgrid=True, gridcolor=colors['grid']),
            yaxis3=dict(title='Patch Delay (Days)', overlaying='y2', side='right', showgrid=False)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"⚠️ Graph Error: {str(e)}")
        return None

# =====================================
# TAB CONTENT FUNCTIONS
# =====================================
def render_home():
    # Compact header
    cols = st.columns([10, 2])
    with cols[0]:
        st.markdown("## 🛡️ CYBER THREAT INTELLIGENCE")
        st.caption(f"**Stephen Musau Makau** | MSc Cybersecurity | ⏱️ {datetime.now(ZoneInfo('Africa/Nairobi')).strftime('%d/%m %H:%M')} EAT")
    with cols[1]:
        st.caption(f"{'🌙' if st.session_state.dark_mode else '☀️'} {'📱' if st.session_state.mobile_view else '💻'}")
    
    # Get prediction
    try:
        prediction = predict_2027()
        all_results = get_results()
        max_acc = max(all_results.values()) if all_results else 0
        best_algorithms = [n for n, a in all_results.items() if a == max_acc] if all_results else ["XGBoost"]
        best_acc_pct = max_acc * 100
    except:
        prediction = "Unknown"
        best_algorithms = ["XGBoost"]
        best_acc_pct = 0.0

    # THREAT PROJECTION
    st.markdown("### 🚨 THREAT PROJECTION // 2027")
    
    if prediction == "High":
        st.markdown(f"""
        <div class="threat-{'high' if st.session_state.dark_mode else 'high'}">
            <h4 style="margin: 0;">⚠️ HIGH RISK DETECTED</h4>
            <p style="margin: 5px 0 0 0; font-size: 0.85rem;">Significant escalation predicted. Immediate countermeasures required. Threat vectors include APTs and zero-day exploits.</p>
        </div>
        """, unsafe_allow_html=True)
        st.warning("⚠️ **Why**: Attack volumes >4,000 DDoS + economic pressure = HIGH threat historically. Review defensive protocols.")
        
    elif prediction == "Critical":
        st.markdown(f"""
        <div class="threat-{'critical' if st.session_state.dark_mode else 'critical'}">
            <h4 style="margin: 0;">🛑 CRITICAL ALERT</h4>
            <p style="margin: 5px 0 0 0; font-size: 0.85rem;">Maximum threat level. Emergency protocols activated. All defensive systems to maximum alert.</p>
        </div>
        """, unsafe_allow_html=True)
        st.error("🛑 **Why**: High CVEs (95) + low patch compliance (9 days) + instability = CRITICAL historically.")
        
    else:
        st.markdown(f"""
        <div class="threat-{'moderate' if st.session_state.dark_mode else 'moderate'}">
            <h4 style="margin: 0;">✅ STABLE STATUS</h4>
            <p style="margin: 5px 0 0 0; font-size: 0.85rem;">Parameters within acceptable ranges. Standard monitoring protocols sufficient.</p>
        </div>
        """, unsafe_allow_html=True)
        st.success("✅ **Why**: Manageable threat levels. Attack volumes within norms. Maintain standard operations.")

    # VISUALIZATION
    st.markdown("### 📡 THREAT VISUALIZATION")
    
    viz_data = get_dataset()
    fig = create_futuristic_graph(viz_data)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key=f"viz_{get_current_data_source()}_{len(viz_data)}")

    with st.expander("📖 Graph Guide", expanded=False):
        st.markdown("""
        **Top Panel**: Cyan=DDoS, Magenta=Malware, Green=Phishing, Orange=Web Attacks  
        **Bottom Panel**: Red bars=Critical CVEs, Gold line=Patch Delay Days  
        **Watch for**: Red spikes + high gold line = Critical vulnerability window
        """)

    # METRICS
    st.markdown("### 📡 SYSTEM METRICS")
    c1, c2, c3 = st.columns(3)
    with c1:
        algo = best_algorithms[0].upper().replace(" ", "_") if len(best_algorithms) == 1 else "ENSEMBLE"
        st.metric("Algorithm", algo, f"{best_acc_pct:.0f}%")
        st.caption("Core ML engine")
    with c2:
        st.metric("Target", "2027", "Forecast")
        st.caption("2-year horizon")
    with c3:
        st.metric("Source", get_current_data_source().upper(), f"{len(viz_data)} recs")
        st.caption("Training data")

    # METHODOLOGY
    with st.expander("🧠 Prediction Methodology", expanded=False):
        is_custom = get_current_data_source() == "uploaded"
        if is_custom:
            st.info(f"""
            **Custom Dataset Mode**
            
            Analyzed {len(viz_data)} records. Selected **{', '.join(best_algorithms)}** as optimal predictor.
            
            **Key Findings**:
            - Pattern recognition on your specific variables
            - **{best_acc_pct:.2f}%** validation accuracy
            - CVE counts and patch delays are key predictors
            
            Uses Empirical Risk Minimization on your dataset.
            """)
        else:
            st.info("""
            **How Prediction Works**:
            
            1. **Input**: 12 parameters (attacks, CVEs, economic indicators)
            2. **Pattern Match**: Compare to historical conditions
            3. **Classification**: Categorize as Moderate/High/Critical
            4. **Confidence**: Based on model validation accuracy
            
            Early warning enables proactive defense vs reactive response.
            """)

def render_data_upload():
    st.markdown("## 📤 DATA UPLOAD")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Download Template")
        st.markdown("""
        Required: Year, Month, DDoS_Attacks, Malware_Attacks, Phishing_Attacks, 
        Web_Attacks, Critical_CVEs, Patch_Delay_Days, Traffic_Volume, 
        Inflation_Rate, GDP_Growth, Economic_Environment, Threat_Level
        """)
        template = generate_excel_template()
        st.download_button("📥 DOWNLOAD TEMPLATE", template, "template.xlsx", 
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    with col2:
        st.markdown("### 📤 Upload Data")
        uploaded = st.file_uploader("Choose file (.xlsx or .csv)", type=['xlsx', 'csv'])
        
        if uploaded:
            try:
                df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
                result = validate_uploaded_data(df)
                
                if result['valid']:
                    st.success("✅ Valid")
                    if st.button("🚀 TRAIN MODEL", type="primary"):
                        with st.spinner("Training..."):
                            success, msg = train_on_uploaded_data(df)
                            if success:
                                st.session_state.uploaded_data = df
                                st.success(msg)
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(msg)
                else:
                    st.error(f"❌ {result['message']}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    if st.button("↩️ RESET TO DEFAULT", use_container_width=True):
        train_on_uploaded_data(None)
        st.session_state.uploaded_data = None
        st.success("✅ Reset complete")
        st.rerun()

def render_overview():
    st.markdown("## 📄 PROJECT OVERVIEW")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎓 Research Context
        **MSc Cybersecurity thesis at Mount Kenya University**
        
        **Problem**: Traditional defenses only react after attacks occur, creating vulnerability windows.
        
        **Solution**: ML predicts threats 2 years in advance for **proactive defense**.
        
        **New**: Upload your own data for custom predictions!
        
        ### ⚠️ Threat Landscape
        Digital transformation correlates with exponential threat growth. 
        Conventional reactive defenses are inadequate against modern automated attacks.
        """)
        
        st.info("""
        **Why Machine Learning?**
        
        Traditional systems look for known signatures. ML discovers **hidden patterns** across variables (e.g., when inflation >7% AND patch delays >10 days, threats spike within 3 months).
        """)
    
    with col2:
        st.markdown("""
        ### 🔬 System Architecture
        - **Attack Vectors**: DDoS, Malware, Phishing, Web Exploits
        - **Vulnerability**: CVE Criticality, Patch Latency
        - **Network**: Traffic Anomaly Detection
        - **Economic**: Inflation/GDP correlation
        
        **Data Flow**: Raw → Feature Engineering → Training → Prediction → Alert
        
        ### 🌍 Operational Impact
        - 🏛️ Critical Infrastructure Protection
        - 💰 Resource Optimization
        - 📋 Policy Intelligence
        - 🤝 Public Trust Maintenance
        - 📊 Custom Organization Analysis
        """)

    st.info("🔒 **SECURITY PROTOCOL**: All data is synthetic/anonymized for research. No real-time government data exposed.")

def render_dataset():
    st.markdown("## 📊 DATA MATRIX")
    
    df = get_dataset()
    
    if get_current_data_source() == "uploaded":
        st.success("📊 Your uploaded custom dataset")
    else:
        st.info("📊 Default research dataset (2020-2025)")
    
    st.markdown("""
    **Columns**: Year, Month, DDoS_Attacks, Malware_Attacks, Phishing_Attacks, 
    Web_Attacks, Critical_CVEs, Patch_Delay_Days, Traffic_Volume, 
    Inflation_Rate, GDP_Growth, Economic_Environment, Threat_Level
    """)
    
    st.dataframe(df, use_container_width=True, height=300)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📈 Key Patterns")
        for p in analyze_dataset_patterns(df)[:4]:
            st.markdown(f"• {p}")
    with col2:
        st.markdown("### 📊 Statistics")
        c1, c2 = st.columns(2)
        c1.metric("Records", len(df))
        yr_range = f"{df['Year'].min():.0f}-{df['Year'].max():.0f}" if 'Year' in df.columns and len(df) > 0 else "N/A"
        c2.metric("Years", yr_range)

def render_models():
    st.markdown("## 🤖 AI CORE PERFORMANCE")
    
    results = get_results()
    if results:
        max_acc = max(results.values())
        active = [n for n, a in results.items() if a == max_acc]
        acc_pct = max_acc * 100
    else:
        active = ["XGBoost"]
        acc_pct = 0.0
    
    # Winner banner
    if len(active) == 1:
        st.success(f"🏆 ACTIVE: {active[0]} | {acc_pct:.2f}% Accuracy")
    else:
        st.success(f"🏆 ENSEMBLE: {' + '.join(active)} | {acc_pct:.2f}% Accuracy")
    
    # Benchmark table
    st.markdown("### 📊 Algorithm Benchmarking")
    st.table([{"Algorithm": n, "Accuracy": f"{a*100:.2f}%", "Status": "🟢 ACTIVE" if n in active else "⚪ STANDBY"} 
              for n, a in results.items()])
    
    # Winner details
    with st.expander("🏆 Winner Details", expanded=True):
        if len(active) == 1:
            w = active[0]
            if "XGBoost" in w:
                st.markdown("""
                **XGBoost** (eXtreme Gradient Boosting)
                
                Uses gradient boosting to combine weak learners. Excels at non-linear relationships and imbalanced datasets. Features: L1/L2 regularization, parallel processing, missing value handling.
                """)
            elif "Random Forest" in w:
                st.markdown("""
                **Random Forest** (200 trees ensemble)
                
                Bagging approach reduces overfitting. Built-in validation via out-of-bag scoring. Feature importance rankings available.
                """)
            else:
                st.markdown("""
                **Logistic Regression** (Linear classifier)
                
                Maximum interpretability with probabilistic outputs. Coefficients show feature impact directly. Fast and auditable.
                """)
        else:
            st.markdown(f"""
            **Multiple Winners**: {', '.join(active)}
            
            All achieved perfect accuracy. System uses consensus voting for maximum reliability.
            """)

def render_parameters():
    st.markdown("## ⚙️ SYSTEM PARAMETERS")
    
    try:
        pred = predict_2027()
        params = get_parameters()
    except:
        pred = "Unknown"
        params = get_parameters()
    
    st.markdown(f"**2027 Prediction**: {'🟢 MODERATE' if pred == 'Medium' else '🟠 HIGH' if pred == 'High' else '🔴 CRITICAL' if pred == 'Critical' else '⚪ Unknown'}")
    
    st.dataframe(params, use_container_width=True, height=250)
    
    with st.expander("⚠️ Parameter Significance", expanded=False):
        st.warning("""
        **High Impact**:
        1. **Patch Delay Days** (9) - Longer delays = more exploit time
        2. **Critical CVEs** (95) - More flaws = more opportunities
        
        **Medium Impact**:
        3. **Economic Environment** - Stress correlates with cybercrime
        4. **DDoS Attacks** (4,200) - Indicates attacker capability
        
        **Pattern**: Patch Delay < 10 days AND CVEs > 90 typically faces HIGH threat levels.
        """)

# =====================================
# MAIN RENDERING
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
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {'#64748b' if st.session_state.dark_mode else '#64748b'}; padding: 10px;">
    <p style="margin: 0; font-size: 0.75rem;">
        🛡️ MKU Cyber Threat Intelligence | © {current_year} Mount Kenya University | {get_current_data_source().title()} Data
    </p>
</div>
""", unsafe_allow_html=True)
