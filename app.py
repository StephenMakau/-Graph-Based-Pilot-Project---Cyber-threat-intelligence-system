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
                title='Attack Volume',
                showgrid=True,
                gridcolor=colors['grid'],
                gridwidth=1,
                zeroline=False,
                linecolor='#06b6d4',
                linewidth=2,
                tickfont=dict(size=10),
                titlefont=dict(size=12, color='#06b6d4)
            ),
            yaxis2=dict(
                title='CVE Count',
                showgrid=True,
                gridcolor=colors['grid'],
                gridwidth=1,
                zeroline=False,
                linecolor='#ff1744',
                linewidth=2,
                tickfont=dict(size=10),
                titlefont=dict(size=12, color='#ff1744)
            ),
            yaxis3=dict(
                title='Patch Delay (Days)',
                overlaying='y2',
                side='right',
                showgrid=False,
                zeroline=False,
                linecolor='#ffd700',
                linewidth=2,
                tickfont=dict(size=10, color='#ffd700),
                titlefont=dict(size=12, color='#ffd700)
            ),
            title=dict(
                text='📊 CYBER THREAT INTELLIGENCE // VISUALIZATION MODULE',
                font=dict(size=16, color='#06b6d4', family='JetBrains Mono'),
                x=0.5,
                xanchor='center
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
    data_indicator = "📊 CUSTOM
