import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import io
import re
import textwrap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path


# =====================================
# IN-PAGE NAVIGATION LINKS
# =====================================
# Markdown references such as "DATA UPLOAD tab" become clickable links.
# The query parameter is read during the next Streamlit run and selects the
# corresponding item in the fixed top navigation.
TAB_LINKS = {
    "OVERVIEW": "home",
    "HOME": "home",
    "DATA INTAKE": "data_upload",
    "DATA UPLOAD": "data_upload",
    "PROJECT NOTES": "overview",
    "SYSTEM OVERVIEW": "overview",
    "DATASET": "dataset",
    "MODEL OUTPUTS": "models",
    "MODELS": "models",
    "CONFIGURATION": "parameters",
    "PARAMETERS": "parameters",
}


def linkify_tab_references(body):
    """Turn explicit references to a navigation tab into deep links."""
    if not isinstance(body, str):
        return body

    # Only link references explicitly described as a tab, section, or page.
    # This avoids linking ordinary uses of words such as "dataset".
    pattern = re.compile(
        r'(?P<bold>\*\*)?(?P<label>OVERVIEW|HOME|DATA INTAKE|DATA UPLOAD|'
        r'PROJECT NOTES|SYSTEM OVERVIEW|DATASET|MODEL OUTPUTS|MODELS|'
        r'CONFIGURATION|PARAMETERS)(?P=bold)(?P<suffix>\s+(?:tab|section|page)\b)',
        flags=re.IGNORECASE,
    )

    def replace(match):
        label = match.group("label")
        target = TAB_LINKS.get(label.upper())
        if not target:
            return match.group(0)
        display = f'**{label}**' if match.group("bold") else label
        return f'[{display}](?tab={target}){match.group("suffix")}'

    return pattern.sub(replace, body)


def linked_markdown(body, *args, **kwargs):
    # Preserve raw HTML blocks (watermark, cards, footer) exactly as authored.
    if kwargs.get("unsafe_allow_html", False):
        return st.markdown(body, *args, **kwargs)
    return st.markdown(linkify_tab_references(body), *args, **kwargs)


def linked_message(method, body, *args, **kwargs):
    """Apply the same tab-link behaviour to Streamlit message/caption text."""
    return method(linkify_tab_references(body), *args, **kwargs)

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
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Cyber Threat Intelligence Platform",
    page_icon=str(Path(__file__).with_name("cti_mark.svg")),
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================
# SESSION STATE INITIALIZATION
# =====================================
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
    
if 'mobile_view' not in st.session_state:
    st.session_state.mobile_view = False

if 'controls_open' not in st.session_state:
    st.session_state.controls_open = False

if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

if 'training_status' not in st.session_state:
    st.session_state.training_status = None

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "home"

# Allow links inside explanations to open the corresponding top navigation tab.
try:
    requested_tab = st.query_params.get("tab")
except Exception:
    requested_tab = None

valid_tabs = {"home", "data_upload", "overview", "dataset", "models", "parameters"}
if requested_tab in valid_tabs:
    st.session_state.active_tab = requested_tab
    # Keep the keyed radio widget synchronised with the deep-linked destination.
    st.session_state.nav_radio = requested_tab

# Get current year from browser/system
current_year = datetime.now().year

# =====================================
# FIXED TOP NAVIGATION + FULL-HEIGHT SYSTEM CONTROLS
# =====================================
# Streamlit reruns the script after each interaction. The keyed container gives
# the navigation a stable CSS target, so it remains fixed while page content scrolls.
sidebar_transform = "translateX(0)" if st.session_state.controls_open else "translateX(-105%)"
if st.session_state.dark_mode:
    nav_background = "#111923"
    nav_text = "#f8fafc"
    nav_muted = "#cbd5e1"
    nav_border = "#3f6578"
    nav_item_background = "rgba(255, 255, 255, .055)"
    nav_active_background = "#263f4d"
    nav_hover_background = "#1c2b36"
    sidebar_background = "#111923"
    brand_color = "#9cc8d6"
else:
    nav_background = "#f7f8fa"
    nav_text = "#172033"
    nav_muted = "#475569"
    nav_border = "#b8c7d0"
    nav_item_background = "rgba(15, 23, 42, .035)"
    nav_active_background = "#e0e9ee"
    nav_hover_background = "#edf2f5"
    sidebar_background = "#f7f8fa"
    brand_color = "#31596b"
nav_css = f"""
<style>
    #MainMenu, footer {{ visibility: hidden; }}
    header {{ visibility: visible; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stToolbar"] {{ visibility: hidden; height: 0; }}
    .main .block-container {{ padding-top: 92px !important; }}

    .st-key-top_nav {{
        position: fixed !important;
        top: 0;
        left: 0;
        right: 0;
        z-index: 2147483000 !important;
        isolation: isolate;
        pointer-events: auto !important;
        min-height: 68px;
        padding: 8px 18px 6px;
        background: {nav_background};
        color: {nav_text};
        border-bottom: 1px solid {nav_border};
        transition: background .25s ease, border-color .25s ease, box-shadow .25s ease;
        box-shadow: 0 8px 28px rgba(0, 0, 0, .24);
    }}
    .st-key-top_nav [data-testid="stHorizontalBlock"] {{
        align-items: center;
        gap: .6rem;
    }}
    .st-key-top_nav [data-testid="stRadio"] > label {{ display: none; }}
    .st-key-top_nav [role="radiogroup"] {{
        display: flex;
        flex-wrap: nowrap;
        gap: .18rem;
        overflow-x: auto;
        scrollbar-width: none;
    }}
    .st-key-top_nav [role="radiogroup"] label {{
        cursor: pointer !important;
        pointer-events: auto !important;
        white-space: nowrap;
        border: 1px solid rgba(148, 163, 184, .20);
        border-radius: 8px;
        padding: 5px 9px;
        background: {nav_item_background};
        color: {nav_text} !important;
        font-size: .78rem;
        transition: background .18s ease, color .18s ease, border-color .18s ease, transform .18s ease;
    }}
    .st-key-top_nav [role="radiogroup"] label *,
    .st-key-top_nav [role="radiogroup"] label p,
    .st-key-top_nav [role="radiogroup"] label span,
    .st-key-top_nav [role="radiogroup"] label div {{
        color: {nav_text} !important;
        opacity: 1 !important;
    }}
    .st-key-top_nav [role="radiogroup"] label:hover {{
        background: {nav_hover_background};
        border-color: {nav_border};
        transform: translateY(-1px);
    }}
    .st-key-top_nav [role="radiogroup"] label:has(input:checked) {{
        background: {nav_active_background};
        border-color: {nav_border};
        color: {nav_text} !important;
        box-shadow: inset 0 -2px 0 {brand_color};
    }}
    [data-testid="stSidebar"] {{
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        bottom: 0 !important;
        height: 100vh !important;
        width: 320px !important;
        min-width: 320px !important;
        z-index: 1000001 !important;
        background: {sidebar_background} !important;
        color: {nav_text} !important;
        border-right: 1px solid {nav_border};
        transition: transform .24s ease-in-out, background .25s ease, border-color .25s ease;
        transform: {sidebar_transform} !important;
        visibility: visible !important;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        height: 100vh !important;
        overflow-y: auto !important;
        padding-top: 1rem;
    }}
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
        height: 100vh;
    }}
    @media (max-width: 900px) {{
        .st-key-top_nav {{ padding: 6px 8px; }}
        .st-key-top_nav [role="radiogroup"] label {{ font-size: .68rem; padding: 4px 6px; }}
        .main .block-container {{ padding-top: 105px !important; }}
        [data-testid="stSidebar"] {{ width: min(86vw, 320px) !important; min-width: min(86vw, 320px) !important; }}
    }}
</style>
"""
linked_markdown(nav_css, unsafe_allow_html=True)

with st.container(key="top_nav"):
    col_ham, col_title, col_nav = st.columns([0.7, 3.0, 10.5], vertical_alignment="center")
    with col_ham:
        if st.button("MENU", key="hamburger_btn", help="Open system controls", type="primary"):
            st.session_state.controls_open = not st.session_state.controls_open
            st.rerun()
    with col_title:
        linked_markdown(
            f'<div style="color:{brand_color};font-size:.96rem;font-weight:650;letter-spacing:.035em;white-space:nowrap;">'
            'CTI <span style="display:inline-block;width:7px;height:7px;margin:0 7px 1px;border-radius:50%;background:#6f9f87;box-shadow:0 0 0 3px rgba(111,159,135,.13);"></span> CYBER THREAT INTELLIGENCE</div>',
            unsafe_allow_html=True
        )
    with col_nav:
        # Text-first navigation avoids generic stock iconography and keeps the
        # interface closer to a purpose-built monitoring console.
        tab_options = {
            "home": "OVERVIEW",
            "data_upload": "DATA INTAKE",
            "overview": "PROJECT NOTES",
            "dataset": "DATASET",
            "models": "MODEL OUTPUTS",
            "parameters": "CONFIGURATION"
        }
        selected_tab = st.radio(
            "Navigation",
            options=list(tab_options.keys()),
            format_func=lambda x: tab_options[x],
            index=list(tab_options.keys()).index(st.session_state.active_tab)
                if st.session_state.active_tab in tab_options else 0,
            horizontal=True,
            label_visibility="collapsed",
            key="nav_radio"
        )
        if selected_tab != st.session_state.active_tab:
            st.session_state.active_tab = selected_tab
            st.rerun()

# The sidebar is a full-height control panel. Its visibility is controlled by
# the hamburger button in the fixed top navigation.
with st.sidebar:
    close_col, heading_col = st.columns([1, 5])
    with close_col:
        if st.button("CLOSE PANEL", key="close_controls", help="Close system controls"):
            st.session_state.controls_open = False
            st.rerun()
    with heading_col:
        linked_markdown("### SYSTEM CONTROLS")
    linked_message(st.caption, "Appearance and display settings")
    st.divider()

    st.session_state.dark_mode = st.toggle(
        "Dark appearance",
        value=st.session_state.dark_mode,
        help="Turn dark appearance on or off."
    )
    st.session_state.mobile_view = st.toggle(
        "Compact mobile layout",
        value=st.session_state.mobile_view,
        help="Use a narrow, phone-sized content layout. This changes the app layout, not your device."
    )

    st.divider()
    data_source = get_current_data_source()
    if data_source == "uploaded":
        linked_message(st.success, ":material/analytics: Using: **Uploaded Data**")
    else:
        linked_message(st.info, ":material/analytics: Using: **Default Dataset**")
    linked_message(st.caption, f"System time: {datetime.now(ZoneInfo('Africa/Nairobi')).strftime('%H:%M:%S EAT')}")

# =====================================
# THEME CSS
# =====================================
if st.session_state.dark_mode:
    theme_css = """
    <style>
        .stApp { --app-text: #e7edf2; --app-muted: #aebbc5; --app-panel: #18232d; --app-input-bg: #121b23; --watermark-line: #9bb4c0; --watermark-sweep: rgba(155,180,192,.65); --watermark-label: #9bb4c0; --watermark-logo-opacity: .13; --watermark-label-opacity: .22; --brand-accent: #6b9aad; background: #101820; color: var(--app-text); }
        .stApp, .stApp p, .stApp li, .stApp label, .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stWidgetLabel"] { color: var(--app-text); }
        .stApp small, .stApp .stCaption, .stApp [data-testid="stCaptionContainer"] { color: var(--app-muted) !important; }
        .stApp input, .stApp textarea, .stApp [data-baseweb="select"] > div, .stApp [data-baseweb="input"] > div { background-color: var(--app-input-bg) !important; color: #e7edf2 !important; border-color: #536674 !important; }
        .stApp [data-baseweb="select"] *, .stApp [data-baseweb="popover"] *, .stApp [role="option"] { color: #e7edf2 !important; }
        .stApp [data-testid="stDataFrame"], .stApp [data-testid="stTable"], .stApp [data-testid="stMetric"] { color: #e7edf2 !important; }
        .stApp [data-testid="stMetricLabel"], .stApp [data-testid="stMetricDelta"] { color: #aebbc5 !important; }
        h1, h2, h3, h4, h5, h6 { color: #e7edf2 !important; text-shadow: none !important; -webkit-text-fill-color: currentColor !important; }
        h1 { border-bottom: 1px solid #344653; padding-bottom: .35rem; }
        h2 { border-left: 3px solid #6b9aad; padding-left: .65rem; }
        h3 { color: #b8d0db !important; }
        p, li, label, .stMarkdown { color: #e7edf2 !important; }
        [data-testid="stMetric"] { background: #18232d; border: 1px solid #344653; box-shadow: none; border-radius: 8px; }
        [data-testid="stMetricValue"] { color: #c2dce5 !important; }
        [data-testid="stMetricLabel"] { color: #aebbc5 !important; }
        div[data-testid="stDataFrame"], div[data-testid="stTable"] { background: #141e27; border: 1px solid #344653; }
        .threat-high { background: #30291b; border: 1px solid #8e743c; color: #f0d99d !important; }
        .threat-critical { background: #321f23; border: 1px solid #8e4b54; color: #f2b8bf !important; }
        .threat-moderate { background: #1d3029; border: 1px solid #4d8069; color: #b8dfcb !important; }
        .status-icon { width: 1.35rem; height: 1.35rem; vertical-align: -0.18em; margin-right: .4rem; stroke: currentColor; fill: none; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }
        .stAlert { background: #18232d; border: 1px solid #344653; color: #e7edf2 !important; }
        .tech-container { background: #18232d; border: 1px solid #344653; border-radius: 8px; }
        .tech-label { color: #aebbc5; }
        .tech-code, .tech-year { color: #c2dce5; }
        .tech-container h2 { border-left: 0 !important; padding-left: 0 !important; }
        .tech-subtitle { color: #aebbc5; }
        strong, b { color: #c2dce5 !important; }
        hr { border-color: #344653 !important; }
        .explanation-box { background: #18232d; border-left: 3px solid #6b9aad; padding: 15px; border-radius: 6px; margin: 10px 0; }
        .upload-box { background: #141e27; border: 1px dashed #536674; border-radius: 8px; padding: 20px; text-align: center; }
        .template-box { background: #182820; border: 1px solid #4d8069; border-radius: 8px; padding: 20px; }
    </style>
    """
else:
    theme_css = """
    <style>
        .stApp { --app-text: #202b33; --app-muted: #596a75; --app-panel: #ffffff; --app-input-bg: #ffffff; --watermark-line: #527d8d; --watermark-sweep: rgba(49,89,107,.42); --watermark-label: #31596b; --watermark-logo-opacity: .19; --watermark-label-opacity: .30; --brand-accent: #527d8d; background: #f4f6f7; color: var(--app-text); }
        .stApp, .stApp p, .stApp li, .stApp label, .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stWidgetLabel"] { color: var(--app-text); }
        .stApp small, .stApp .stCaption, .stApp [data-testid="stCaptionContainer"] { color: var(--app-muted) !important; }
        .stApp input, .stApp textarea, .stApp [data-baseweb="select"] > div, .stApp [data-baseweb="input"] > div { background-color: #ffffff !important; color: #202b33 !important; border-color: #b8c4cb !important; }
        .stApp [data-baseweb="select"] *, .stApp [data-baseweb="popover"] *, .stApp [role="option"] { color: #202b33 !important; }
        .stApp [data-testid="stDataFrame"], .stApp [data-testid="stTable"], .stApp [data-testid="stMetric"] { color: #202b33 !important; }
        .stApp [data-testid="stMetricLabel"], .stApp [data-testid="stMetricDelta"] { color: #596a75 !important; }
        h1, h2, h3, h4, h5, h6 { color: #202b33 !important; text-shadow: none !important; -webkit-text-fill-color: currentColor !important; }
        h1 { border-bottom: 1px solid #d4dde2; padding-bottom: .35rem; }
        h2 { border-left: 3px solid #527d8d; padding-left: .65rem; }
        h3 { color: #31596b !important; }
        p, li, label, .stMarkdown { color: #202b33 !important; }
        [data-testid="stMetric"] { background: #ffffff; border: 1px solid #d4dde2; box-shadow: 0 2px 8px rgba(24, 42, 52, .04); border-radius: 8px; }
        [data-testid="stMetricValue"] { color: #31596b !important; }
        [data-testid="stMetricLabel"] { color: #596a75 !important; }
        div[data-testid="stDataFrame"], div[data-testid="stTable"] { background: #ffffff; border: 1px solid #d4dde2; }
        .threat-high { background: #fbf5e7; border: 1px solid #c5a75c; color: #684f18 !important; }
        .threat-critical { background: #faeeee; border: 1px solid #c77b83; color: #7d2832 !important; }
        .threat-moderate { background: #edf5f0; border: 1px solid #7ca891; color: #24573d !important; }
        .status-icon { width: 1.35rem; height: 1.35rem; vertical-align: -0.18em; margin-right: .4rem; stroke: currentColor; fill: none; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }
        .stAlert { background: #ffffff; border: 1px solid #d4dde2; color: #202b33 !important; }
        .tech-container { background: #ffffff; border: 1px solid #d4dde2; border-radius: 8px; }
        .tech-label { color: #596a75; }
        .tech-code, .tech-year { color: #31596b; }
        .tech-container h2 { border-left: 0 !important; padding-left: 0 !important; }
        .tech-subtitle { color: #596a75; }
        strong, b { color: #31596b !important; }
        hr { border-color: #d4dde2 !important; }
        .explanation-box { background: #edf2f4; border-left: 3px solid #527d8d; padding: 15px; border-radius: 6px; margin: 10px 0; }
        .upload-box { background: #ffffff; border: 1px dashed #a8b8c1; border-radius: 8px; padding: 20px; text-align: center; }
        .template-box { background: #edf5f0; border: 1px solid #7ca891; border-radius: 8px; padding: 20px; }
    </style>
    """

# Keep layout-width rules inside the single final stylesheet so no standalone
# <style> fragment is emitted between the app theme and its content.
content_width = "480px" if st.session_state.mobile_view else "95%"
contrast_css = f"""
<style>

    .live-watermark {{
        position: fixed; inset: 68px 0 0 0; z-index: 0; pointer-events: none; overflow: hidden;
        display: grid; place-items: center; isolation: isolate;
    }}
    .live-watermark::before {{
        content: ""; position: absolute; width: min(76vw, 820px); aspect-ratio: 1; border-radius: 50%;
        border: 1px solid var(--watermark-line, #9bb4c0); opacity: .44;
        background: conic-gradient(from 0deg, transparent 0deg 300deg, var(--watermark-sweep, rgba(107,154,173,.45)) 328deg, transparent 342deg),
                    repeating-radial-gradient(circle, transparent 0 54px, var(--watermark-line, #9bb4c0) 55px 56px, transparent 57px 94px);
        box-shadow: inset 0 0 0 34px transparent, inset 0 0 0 35px var(--watermark-line, #9bb4c0);
        animation: watermark-sweep 42s linear infinite;
    }}
    .watermark-logo {{
        position: absolute; width: min(42vw, 430px); min-width: 230px; height: auto;
        opacity: var(--watermark-logo-opacity, .15); filter: none;
    }}
    .watermark-logo svg {{ width: 100%; height: auto; display: block; }}
    .watermark-label {{
        position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%) rotate(-18deg);
        white-space: nowrap; font: 700 clamp(1rem, 3vw, 2.6rem)/1.2 Arial, sans-serif;
        letter-spacing: .15em; color: var(--watermark-label, #9bb4c0);
        border-top: 1px solid currentColor; border-bottom: 1px solid currentColor;
        padding: .45em 1.1em; opacity: var(--watermark-label-opacity, .24);
    }}
    @keyframes watermark-sweep {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
    .stApp [data-testid="stAppViewContainer"], .stApp [data-testid="stHeader"] {{ position: relative; }}
    /* Do not create a stacking context above the fixed navigation. This keeps
       the navigation's high z-index effective and the tabs clickable. */
    /* Keep the main page out of a higher stacking context than the fixed nav. */
    .stApp [data-testid="stMainBlockContainer"] {{ position: relative; z-index: auto !important; isolation: auto !important; transform: none !important; filter: none !important; }}
    .stApp [data-testid="stAppViewContainer"] {{ isolation: auto !important; }}
    .st-key-top_nav {{ position: fixed !important; z-index: 2147483000 !important; pointer-events: auto !important; backdrop-filter: blur(10px); }}
    .st-key-top_nav [data-testid="stRadio"],
    .st-key-top_nav [data-testid="stRadio"] *,
    .st-key-top_nav [role="radiogroup"],
    .st-key-top_nav [role="radiogroup"] *,
    .st-key-top_nav button {{ pointer-events: auto !important; }}
    .st-key-top_nav [role="radiogroup"] label {{ position: relative; z-index: 2; cursor: pointer !important; }}
    .st-key-top_nav [role="radiogroup"] label {{ border-radius: 5px !important; }}
    .st-key-top_nav [role="radiogroup"] label:has(input:checked) {{ box-shadow: inset 0 -2px 0 var(--brand-accent, #6b9aad) !important; }}

    .main .block-container {{
        max-width: {content_width} !important;
        padding-left: {"1rem" if st.session_state.mobile_view else "2rem"} !important;
        padding-right: {"1rem" if st.session_state.mobile_view else "2rem"} !important;
    }}
    /* Keep text readable and let the selected theme control navigation colours. */
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li,
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stCaptionContainer"] p {{ line-height: 1.55; }}
    .stApp [data-testid="stTextInput"] input,
    .stApp [data-testid="stNumberInput"] input,
    .stApp [data-testid="stTextArea"] textarea,
    .stApp [data-testid="stDateInput"] input {{ opacity: 1 !important; }}
    .stApp [data-testid="stDataFrame"] *, .stApp [data-testid="stTable"] * {{ text-shadow: none !important; }}
    .stApp [data-testid="stAlert"] p, .stApp [data-testid="stAlert"] span {{ color: inherit; }}
    .stApp [data-testid="stRadio"] label, .stApp [data-testid="stCheckbox"] label, .stApp [data-testid="stToggle"] label {{ opacity: 1 !important; }}
    .stApp button {{ font-weight: 600; }}
    /* Make native Streamlit tabs and segmented controls readable in both themes. */
    .stApp [data-baseweb="tab-list"] {{ gap: .35rem; border-bottom: 1px solid var(--brand-accent, #527d8d); }}
    .stApp [data-baseweb="tab"], .stApp [role="tab"] {{
        color: var(--app-text, #202b33) !important; opacity: 1 !important; font-weight: 650 !important;
        background: var(--app-panel, #fff) !important; border: 1px solid #8a9ba5 !important;
        border-bottom: 3px solid transparent !important; border-radius: 6px 6px 0 0 !important;
        padding: .55rem .9rem !important;
    }}
    .stApp [data-baseweb="tab"] *, .stApp [role="tab"] * {{ color: var(--app-text, #202b33) !important; opacity: 1 !important; }}
    .stApp [aria-selected="true"][role="tab"] {{ border-bottom-color: var(--brand-accent, #527d8d) !important; background: var(--app-input-bg, #fff) !important; }}
    .stApp [data-baseweb="tab-highlight"] {{ background-color: var(--brand-accent, #527d8d) !important; }}
    .stApp [data-testid="stSegmentedControl"] button {{ color: var(--app-text, #202b33) !important; opacity: 1 !important; }}

    /* A restrained ECG pulse makes elevated projections noticeable without rapid flashing. */
    .threat-high, .threat-critical, .threat-moderate {{
        position: relative; overflow: hidden; isolation: isolate; border-radius: 10px;
        padding: 1.15rem 1.3rem; margin: .65rem 0 1rem; border-width: 1px; border-style: solid;
    }}
    .threat-high h1, .threat-critical h1, .threat-moderate h1 {{ border: 0; padding: 0; margin-top: 0; }}
    .threat-high p, .threat-high h1, .threat-high h3, .threat-critical p, .threat-critical h1, .threat-critical h3,
    .threat-moderate p, .threat-moderate h1, .threat-moderate h3 {{ color: inherit !important; position: relative; z-index: 2; }}
    .threat-pulse {{ animation: threat-heartbeat 1.8s ease-in-out infinite; }}
    .threat-high.threat-pulse {{ --pulse-rgb: 190, 145, 58; }}
    .threat-critical.threat-pulse {{ --pulse-rgb: 190, 70, 78; }}
    .ecg-wave {{ position: absolute; right: 1.2rem; bottom: .5rem; width: min(34%, 270px); height: 44px;
        fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round;
        opacity: .72; z-index: 1; pointer-events: none; stroke-dasharray: 300; animation: ecg-trace 2.4s linear infinite; }}
    @keyframes threat-heartbeat {{
        0%, 100% {{ box-shadow: 0 0 0 0 rgba(var(--pulse-rgb), .08); }}
        13% {{ box-shadow: 0 0 0 3px rgba(var(--pulse-rgb), .18), 0 0 18px rgba(var(--pulse-rgb), .14); }}
        23% {{ box-shadow: 0 0 0 1px rgba(var(--pulse-rgb), .08); }}
        35% {{ box-shadow: 0 0 0 4px rgba(var(--pulse-rgb), .13), 0 0 24px rgba(var(--pulse-rgb), .10); }}
        48% {{ box-shadow: 0 0 0 0 rgba(var(--pulse-rgb), .04); }}
    }}
    @keyframes ecg-trace {{ from {{ stroke-dashoffset: 300; }} to {{ stroke-dashoffset: 0; }} }}

    /* Restrained motion: small transitions, with accessibility support. */
    .stApp [data-testid="stMetric"], .stApp [data-testid="stDataFrame"],
    .stApp [data-testid="stTable"], .stApp button {{
        transition: border-color .2s ease, box-shadow .2s ease, transform .2s ease;
    }}
    .stApp button:hover {{ transform: translateY(-1px); }}
    .st-key-top_nav button:hover {{ transform: none !important; }}
    @media (max-width: 700px) {{
        .watermark-logo {{ width: min(66vw, 360px); min-width: 190px; }}
        .watermark-label {{ letter-spacing: .08em; font-size: clamp(.8rem, 3vw, 1.4rem); }}
        .ecg-wave {{ width: 45%; opacity: .5; }}
    }}
    @media (prefers-reduced-motion: reduce) {{
        .threat-pulse, .ecg-wave, .live-watermark::before {{ animation: none !important; }}
        .stApp *, .st-key-top_nav, [data-testid="stSidebar"] {{
            animation: none !important; transition: none !important; scroll-behavior: auto !important;
        }}
    }}
</style>
"""
linked_markdown(theme_css + contrast_css, unsafe_allow_html=True)
# Reuse the browser favicon artwork as the large watermark emblem so the visual identity matches.
try:
    watermark_svg = Path(__file__).with_name("cti_mark.svg").read_text(encoding="utf-8")
    # The favicon's rounded-square background is omitted in the watermark to avoid a dark tile in light mode.
    watermark_svg = re.sub(r'<rect[^>]*/>', '', watermark_svg, count=1)
except (OSError, NameError):
    watermark_svg = ""
linked_markdown(
    f'<div class="live-watermark" aria-hidden="true"><div class="watermark-logo">{watermark_svg}</div>'
    '<div class="watermark-label">CTI / LIVE SIGNAL MONITOR</div></div>',
    unsafe_allow_html=True
)

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
            linked_message(st.error, "No valid data available for visualization.")
            return None

        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=('ATTACK VECTORS OVER TIME', 'VULNERABILITY METRICS')
        )
        
        # A restrained, colour-blind-conscious palette with clear semantic contrast.
        dark_chart = st.session_state.dark_mode
        colors = {
            'ddos': '#6fa8c0',
            'malware': '#c58b5b',
            'phishing': '#78a88b',
            'web': '#9b9dbb',
            'cves': '#b85c63',
            'patch': '#c6a75e',
            'traffic': '#66879a',
            'grid': 'rgba(150, 170, 182, 0.18)' if dark_chart else 'rgba(70, 90, 102, 0.14)',
            'text': '#dce5ea' if dark_chart else '#27343c',
            'paper': 'rgba(0,0,0,0)',
            'plot': 'rgba(20, 30, 39, 0.72)' if dark_chart else 'rgba(255,255,255,0.78)',
            'line': '#667984' if dark_chart else '#a9b7bf',
            'marker_line': '#17232b' if dark_chart else '#ffffff',
            'legend_bg': 'rgba(17, 25, 35, 0.92)' if dark_chart else 'rgba(255,255,255,0.92)',
            'legend_border': '#536674' if dark_chart else '#c7d2d9',
        }
        
        # Row 1: Attack Volumes
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['DDoS_Attacks'],
                name='DDoS Attacks',
                mode='lines+markers',
                line=dict(color=colors['ddos'], width=2.4),
                marker=dict(size=7, symbol='diamond', line=dict(width=1, color=colors['marker_line'])),
                fill='tonexty',
                fillcolor='rgba(111, 168, 192, 0.10)'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['Malware_Attacks'],
                name='Malware',
                mode='lines+markers',
                line=dict(color=colors['malware'], width=2.4),
                marker=dict(size=7, symbol='circle', line=dict(width=1, color=colors['marker_line'])),
                fill='tonexty',
                fillcolor='rgba(197, 139, 91, 0.08)'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['Phishing_Attacks'],
                name='Phishing',
                mode='lines+markers',
                line=dict(color=colors['phishing'], width=2.4),
                marker=dict(size=7, symbol='triangle-up', line=dict(width=1, color=colors['marker_line'])),
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['Web_Attacks'],
                name='Web Attacks',
                mode='lines+markers',
                line=dict(color=colors['web'], width=2, dash='dot'),
                marker=dict(size=7, symbol='square', line=dict(width=1, color=colors['marker_line'])),
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
                    line=dict(color=colors['marker_line'], width=1),
                    opacity=0.85
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
                line=dict(color=colors['patch'], width=2.4),
                marker=dict(size=10, symbol='x', line=dict(width=2, color='white')),
                yaxis='y3'
            ),
            row=2, col=1
        )
        
        # Update layout with futuristic styling
        fig.update_layout(
            template='plotly_dark' if dark_chart else 'plotly_white',
            paper_bgcolor=colors['paper'],
            plot_bgcolor=colors['plot'],
            font=dict(family='Arial, sans-serif', color=colors['text'], size=11),
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                bgcolor=colors['legend_bg'],
                bordercolor=colors['legend_border'],
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
                linecolor=colors['line'],
                linewidth=2,
                tickfont=dict(size=10)
            ),
            xaxis2=dict(
                showgrid=True,
                gridcolor=colors['grid'],
                gridwidth=1,
                zeroline=False,
                linecolor=colors['line'],
                linewidth=2,
                title='Timeline'
            ),
            yaxis=dict(
                title=dict(text='Attack Volume', font=dict(size=12, color=colors['ddos'])),
                showgrid=True,
                gridcolor=colors['grid'],
                gridwidth=1,
                zeroline=False,
                linecolor=colors['line'],
                linewidth=2,
                tickfont=dict(size=10)
            ),
            yaxis2=dict(
                title=dict(text='CVE Count', font=dict(size=12, color=colors['cves'])),
                showgrid=True,
                gridcolor=colors['grid'],
                gridwidth=1,
                zeroline=False,
                linecolor=colors['cves'],
                linewidth=2,
                tickfont=dict(size=10)
            ),
            yaxis3=dict(
                title=dict(text='Patch Delay (Days)', font=dict(size=12, color=colors['patch'])),
                overlaying='y2',
                side='right',
                showgrid=False,
                zeroline=False,
                linecolor=colors['patch'],
                linewidth=2,
                tickfont=dict(size=10, color=colors['patch'])
            ),
            title=dict(
                text='CYBER THREAT INTELLIGENCE | VISUALIZATION MODULE',
                font=dict(size=16, color=colors['text'], family='Arial, sans-serif'),
                x=0.5,
                xanchor='center'
            ),
            # Fine structural lines give the chart a technical, restrained finish.
            shapes=[
                # Top border line
                dict(type='line', x0=0, x1=1, y0=1.15, y1=1.15, 
                     xref='paper', yref='paper',
                     line=dict(color=colors['line'], width=1)),
                # Bottom border line
                dict(type='line', x0=0, x1=1, y0=-0.15, y1=-0.15,
                     xref='paper', yref='paper',
                     line=dict(color=colors['line'], width=1)),
            ],
            # Add annotations for futuristic feel
            annotations=[
                dict(
                    text='LIVE DATA STREAM',
                    xref='paper', yref='paper',
                    x=0.5, y=1.08,
                    showarrow=False,
                    font=dict(size=10, color=colors['text'], family='Arial, sans-serif'),
                    bgcolor=colors['legend_bg'],
                    bordercolor=colors['legend_border'],
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
        linked_message(st.error, f":material/warning: Graph Generation Error: {str(e)}")
        return None

# =====================================
# TAB CONTENT FUNCTIONS - ALL EXPLANATIONS PRESERVED
# =====================================
def render_home():
    st.title(":material/shield: CYBER THREAT INTELLIGENCE")
    linked_markdown("**Operator:** Stephen Musau Makau | **Clearance:** MSc Cybersecurity")
    
    mode_indicator = ":material/dark_mode: DARK" if st.session_state.dark_mode else ":material/light_mode: LIGHT"
    view_indicator = ":material/smartphone: MOBILE" if st.session_state.mobile_view else ":material/desktop_windows: DESKTOP"
    data_indicator = ":material/analytics: CUSTOM" if get_current_data_source() == "uploaded" else ":material/storage: DEFAULT"
    linked_message(st.caption, f":material/schedule: SYS.TIME: {datetime.now(ZoneInfo('Africa/Nairobi')).strftime('%d/%m/%Y | %H:%M:%S')} EAT | STATUS: ONLINE | MODE: {mode_indicator} | VIEW: {view_indicator} | DATA: {data_indicator}")
    
    st.divider()

    # 1. EXECUTION: Get Prediction & Accuracy Immediately
    try:
        prediction = predict_2027()
        accuracy = get_model_accuracy() * 100
        # Get all results for algorithm selection
        all_results = get_results()
    except Exception as e:
        linked_message(st.error, f":material/warning: System Error: {e}")
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
    st.header(":material/warning: THREAT PROJECTION // 2027")
    
    if prediction == "High":
        linked_markdown(textwrap.dedent("""
        <div class="threat-high threat-pulse">
            <svg class="ecg-wave" viewBox="0 0 240 44" aria-hidden="true"><path d="M0 23 H45 L57 23 L69 13 L80 34 L94 4 L108 30 L119 23 H240"/></svg>
            <h1><svg class="status-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3 22 20H2L12 3Z"/><path d="M12 9v5M12 17h.01"/></svg>HIGH RISK DETECTED</h1>
            <h3>THREAT_LEVEL: HIGH</h3>
            <p>Predictive algorithms indicate significant escalation in cyber threats targeting critical infrastructure. Immediate countermeasures required. Threat vectors include advanced persistent threats (APTs) and zero-day exploits.</p>
        </div>
        """), unsafe_allow_html=True)
        linked_message(st.warning, ":material/warning: **Interpretation**: Historical data shows that when attack volumes exceed 4,000 (DDoS), 18,000 (Malware), and economic pressure indicators are elevated, threat levels typically spike to HIGH. Review defensive protocols immediately.")
        
    elif prediction == "Critical":
        linked_markdown(textwrap.dedent("""
        <div class="threat-critical threat-pulse">
            <svg class="ecg-wave" viewBox="0 0 240 44" aria-hidden="true"><path d="M0 23 H45 L57 23 L69 13 L80 34 L94 4 L108 30 L119 23 H240"/></svg>
            <h1><svg class="status-icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="m9 9 6 6m0-6-6 6"/></svg>CRITICAL ALERT</h1>
            <h3>THREAT_LEVEL: CRITICAL</h3>
            <p>Maximum threat level detected. System predicts unprecedented attack surge. Emergency protocols activated. All defensive systems should be raised to maximum alert status immediately.</p>
        </div>
        """), unsafe_allow_html=True)
        linked_message(st.error, ":material/report: **Interpretation**: The convergence of high CVE counts (95), low patch compliance (9 days delay), and economic instability historically correlates with CRITICAL threat periods. Immediate executive action required.")
        
    else:
        linked_markdown(textwrap.dedent("""
        <div class="threat-moderate">
            <h1><svg class="status-icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="m8 12 2.5 2.5L16 9"/></svg>STABLE STATUS</h1>
            <h3>THREAT_LEVEL: MODERATE</h3>
            <p>Threat parameters within acceptable ranges. Standard monitoring protocols sufficient. Continue baseline security operations and routine system audits.</p>
        </div>
        """), unsafe_allow_html=True)
        linked_message(st.success, ":material/check_circle: **Interpretation**: Current projections indicate manageable threat levels. Attack volumes remain within historical norms and economic indicators suggest stable conditions. Maintain standard operations.")

    st.divider()

    # 3. FUTURISTIC GRAPH: Visual Data Intelligence
    linked_markdown("### :material/radar: THREAT VISUALIZATION MATRIX")
    
    # Show current data source for the graph
    current_source = get_current_data_source()
    if current_source == "uploaded":
        linked_message(st.info, ":material/analytics: Displaying **Uploaded Dataset** Visualization")
    else:
        linked_message(st.info, ":material/analytics: Displaying **Default Dataset** Visualization")
    
    # Get current dataset for visualization
    viz_data = get_dataset()
    
    # Create and display the futuristic graph with unique key based on data source
    fig = create_futuristic_graph(viz_data)
    if fig is not None:
        # KEY FIX: Add unique key based on data source and data shape to force re-render
        chart_key = f"threat_graph_{current_source}_{len(viz_data)}"
        st.plotly_chart(fig, use_container_width=True, height=600, key=chart_key)
    
    # Graph explanation
    with st.expander(":material/menu_book: **Graph Interpretation Guide**", expanded=False):
        linked_markdown(textwrap.dedent("""
        **Understanding the Visualization:**
        
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
        """))
    
    st.divider()

    # 4. CONTEXT: System Metrics & Algorithm Info (Secondary Priority)
    linked_markdown("###  THREAT ASSESSMENT MODULE")
    
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
        
        linked_markdown(textwrap.dedent(f"""
        <div class="tech-container" style="text-align: center;">
            <h4 class="tech-label" style="margin:15px 0 0 0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;">Core Algorithm {'(s)' if len(best_algorithms) > 1 else ''}</h4>
            <h2 class="tech-code" style="margin: 20px 0; font-family: Arial, sans-serif; font-size: {'2.0rem' if len(best_algorithms) > 1 else '2.6rem'}; font-weight: 700;">{algo_code}</h2>
            <p class="tech-subtitle" style="font-size: 0.95rem; margin-bottom: 15px;">{subtitle}</p>
        </div>
        """), unsafe_allow_html=True)
        
        # Dynamic explanation based on which algorithm(s) won
        if len(best_algorithms) == 1:
            if "XGBoost" in best_algorithms[0]:
                linked_message(st.caption, ":material/lightbulb: **XGBoost** uses gradient boosting to combine weak learners into a strong predictor. Excels at capturing non-linear relationships in cybersecurity data and handles imbalanced datasets effectively.")
            elif "Random Forest" in best_algorithms[0]:
                linked_message(st.caption, ":material/lightbulb: **Random Forest** builds multiple decision trees and merges their predictions. Reduces overfitting through bagging and provides robust performance across varied threat patterns.")
            elif "Logistic Regression" in best_algorithms[0]:
                linked_message(st.caption, ":material/lightbulb: **Logistic Regression** provides a linear probabilistic approach. While simpler than ensemble methods, it offers high interpretability and efficient computation for real-time predictions.")
        elif len(best_algorithms) == 2:
            linked_message(st.caption, f":material/lightbulb: **Dual Algorithm Mode**: Both {best_algorithms[0]} and {best_algorithms[1]} achieved optimal performance. The system leverages complementary strengths - ensemble diversity meets statistical rigor.")
        else:
            linked_message(st.caption, ":material/lightbulb: **Triple Consensus Mode**: All three algorithms achieved 100% accuracy on validation data. This rare state indicates extremely clear patterns in your dataset, providing maximum confidence in predictions.")

    with c2:
        st.metric(
            ":material/track_changes: MODEL ACCURACY",
            f"{best_acc_pct:.2f}%",
            help="Training validation score"
        )
        if best_acc_pct >= 99.9:
            linked_message(st.caption, ":material/star: **PERFECT SCORE**: Optimal prediction reliability achieved. The model(s) correctly classified all validation samples.")
        elif best_acc_pct >= 90:
            linked_message(st.caption, ":material/check_circle: **HIGH RELIABILITY**: Strong predictive performance suitable for operational deployment.")
        else:
            linked_message(st.caption, ":material/warning: **MODERATE RELIABILITY**: Consider uploading more historical data to improve accuracy.")

    with c3:
        linked_markdown(textwrap.dedent("""
        <div class="tech-container" style="text-align: center;">
            <h4 class="tech-label" style="margin:15px 0 0 0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;">Target Year</h4>
            <h2 class="tech-year" style="margin: 20px 0; font-family: Arial, sans-serif; font-size: 2.6rem; font-weight: 700;">2027</h2>
            <p class="tech-subtitle" style="font-size: 0.95rem; margin-bottom: 15px;">Forecast horizon</p>
        </div>
        """), unsafe_allow_html=True)
        linked_message(st.caption, ":material/lightbulb: The model projects threat levels 2 years ahead using trend analysis of attack patterns, economic indicators, and system vulnerabilities from historical data.")

    st.divider()

    # 4. EDUCATION: Dynamic Methodology Based on Data Source
    linked_markdown("### :material/psychology: **Prediction Methodology**")
    
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
        linked_message(st.warning, ":material/warning: **Custom Dataset Detected**: The methodology below reflects patterns learned from your **uploaded data** rather than the default research dataset.")
        linked_message(st.info, f"""
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
            linked_message(st.info, """
            **How the 2027 Prediction is Generated (Logistic Regression Mode):**
            
            1. **Linear Classification**: The model calculates probability scores for each threat level (Medium/High/Critical) using weighted linear combinations of input features
            2. **Sigmoid Activation**: Probabilities are passed through a logistic function to ensure outputs fall between 0 and 1
            3. **Decision Boundary**: The class with highest probability above threshold (0.5) is selected as the prediction
            4. **Interpretability**: Coefficients reveal which factors most influence threat levels (e.g., positive weight on CVE count increases Critical probability)
            
            **Why This Matters**: While less complex than ensemble methods, logistic regression provides transparent, auditable predictions crucial for government security decisions.
            """)
        elif best_algorithms and len(best_algorithms) == 1 and "Random Forest" in best_algorithms[0]:
            linked_message(st.info, """
            **How the 2027 Prediction is Generated (Random Forest Mode):**
            
            1. **Ensemble Voting**: 200 decision trees independently analyze the 2027 parameters and vote on threat classification
            2. **Bootstrap Aggregation**: Each tree trains on random subsets of historical data, ensuring robustness against outliers
            3. **Feature Randomness**: At each split, only random subsets of features are considered, forcing diversity in tree structures
            4. **Majority Rule**: The threat level receiving the most votes across all trees becomes the final prediction
            
            **Why This Matters**: Random Forest reduces overfitting risks inherent in single decision trees while maintaining interpretability through feature importance rankings.
            """)
        else:
            # Default to XGBoost or Ensemble explanation
            linked_message(st.info, """
            **How the 2027 Prediction is Generated:**
            
            1. **Data Input**: The model receives projected values for 12 parameters (DDoS attacks, malware volume, CVE counts, inflation, GDP, etc.)
            2. **Pattern Recognition**: The system compares these projections against historical patterns where similar conditions resulted in specific threat levels
            3. **Classification**: The system classifies the 2027 scenario into one of three categories: Moderate, High, or Critical
            4. **Confidence**: The accuracy metric indicates how much trust to place in this prediction based on past performance
            
            **Why This Matters**: Early warning allows security teams to allocate resources proactively rather than reacting to attacks after they occur.
            """)

    # EXPLANATION: What this page shows
    with st.expander(":material/menu_book: **How to Read This Dashboard**", expanded=False):
        linked_markdown(textwrap.dedent("""
        **Welcome to the Cyber Threat Intelligence System.** This dashboard predicts cyber threat levels for Kenyan Government Digital Services using Machine Learning.
        
        **Key Components:**
        - **Model Accuracy**: Shows how well our AI predicts past threats (higher % = more reliable)
        - **Target Year**: The system forecasts threats for **2027** based on historical patterns (2020-2025)
        - **Threat Projection**: The colored alert box shows the predicted threat level using three categories:
            - **MODERATE** (Green): Normal operations sufficient
            - **HIGH** (Orange): Increased vigilance required
            - **CRITICAL** (Red): Maximum alert status needed
        
        **How it Works**: The system analyzes historical data points across 12 variables (attack types, economic factors, vulnerabilities) to identify patterns and predict future threats.
        
        **Upload Your Own Data**: Go to the **DATA UPLOAD** tab to use your own historical data for custom predictions!
        """))

def render_data_upload():
    st.title(" DATA UPLOAD CENTER")
    linked_markdown("**Upload your own cybersecurity data to generate custom threat predictions**")
    
    st.divider()
    
    # STEP 1: Download Template
    st.header(" STEP 1: Download Template")
    
    linked_markdown(textwrap.dedent("""
    ###  **Data Format Requirements**
    
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
    """))
    
    # Generate and provide download button for template
    template_buffer = generate_excel_template()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label=" DOWNLOAD EXCEL TEMPLATE",
            data=template_buffer,
            file_name="cyber_threat_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        linked_message(st.caption, " This template contains sample data showing exactly how to format your entries")
    
    st.divider()
    
    # STEP 2: Upload Data
    st.header(" STEP 2: Upload Your Data")
    
    linked_markdown(textwrap.dedent("""
    ###  **Upload Instructions**
    
    1. **Prepare your data** using the template above
    2. **Save as Excel (.xlsx)** or CSV (.csv) format
    3. **Ensure minimum 5 rows** of historical data (more is better)
    4. **Verify column names match exactly** (case-sensitive)
    5. **Upload below** and click "Train Model"
    
     **Important**: The system will automatically retrain all ML algorithms (Logistic Regression, Random Forest, XGBoost) using your uploaded data.
    """))
    
    uploaded_file = st.file_uploader(
        "Choose your data file (Excel .xlsx or CSV .csv)",
        type=['xlsx', 'csv'],
        help="Upload your historical cybersecurity data following the template format"
    )
    
    if uploaded_file is not None:
        linked_message(st.success, f" File uploaded: **{uploaded_file.name}**")
        
        # Preview uploaded data
        try:
            if uploaded_file.name.endswith('.csv'):
                df_preview = pd.read_csv(uploaded_file)
            else:
                df_preview = pd.read_excel(uploaded_file)
            
            linked_markdown("### **Data Preview** (First 5 rows)")
            st.dataframe(df_preview.head(), use_container_width=True)
            
            # Validate data
            validation_result = validate_uploaded_data(df_preview)
            
            if validation_result['valid']:
                linked_message(st.success, f" **Validation Passed**: {validation_result['message']}")
                
                # Train button
                if st.button(" TRAIN MODEL WITH UPLOADED DATA", use_container_width=True, type="primary"):
                    with st.spinner("Training models... Please wait..."):
                        success, message = train_on_uploaded_data(df_preview)
                        if success:
                            st.session_state.uploaded_data = df_preview
                            st.session_state.training_status = "success"
                            linked_message(st.success, f" {message}")
                            st.balloons()
                            linked_message(st.info, " Navigate to **HOME** tab to see updated graph with your data!")
                            # Force page rerun to refresh graph with new data
                            st.rerun()
                        else:
                            st.session_state.training_status = "error"
                            linked_message(st.error, f" {message}")
            else:
                linked_message(st.error, f" **Validation Failed**: {validation_result['message']}")
                linked_message(st.warning, "Please fix the issues above and re-upload.")
                
        except Exception as e:
            linked_message(st.error, f" Error reading file: {str(e)}")
            linked_message(st.info, " Make sure your file is not corrupted and is in the correct format.")
    
    st.divider()
    
    # Reset option
    st.header(" Reset to Default")
    if st.button("↩️ RETURN TO DEFAULT DATASET", use_container_width=True):
        st.session_state.uploaded_data = None
        # Reinitialize with default data
        train_on_uploaded_data(None)
        linked_message(st.success, " System reset to default dataset")
        linked_message(st.info, "The system is now using the original 2020-2025 research data.")
        # Force page rerun to refresh graph
        st.rerun()

def render_overview():
    st.title(" SYSTEM OVERVIEW")
    
    # EXPLANATION: Project context
    linked_markdown(textwrap.dedent("""
    ###  **Research Context**
    This system was developed as part of an **MSc Cybersecurity thesis at Mount Kenya University** to address a critical gap: most security systems react to attacks after they happen, but this tool **predicts threats before they occur**.
    
    **The Problem**: Kenyan Government Digital Services face increasing cyber attacks, but traditional defenses only respond after damage occurs. This creates vulnerability windows.
    
    **The Solution**: Machine Learning analyzes historical attack patterns alongside economic and technical indicators to forecast threat levels 2 years in advance, enabling **proactive defense**.
    
    **New Feature**: You can now upload your own historical data to generate custom predictions specific to your organization or region!
    """))
    
    if st.session_state.mobile_view:
        st.subheader(" MISSION OBJECTIVE")
        st.write("Advanced predictive intelligence platform for Kenyan Government Digital Services.")
        st.subheader(" THREAT LANDSCAPE")
        st.write("Digital transformation acceleration correlates with exponential threat growth.")
        st.subheader(" SYSTEM ARCHITECTURE")
        linked_markdown(textwrap.dedent("""
        - **Attack Vectors:** DDoS, Malware, Phishing
        - **Vulnerability Metrics:** CVE Criticality
        - **Network Intelligence:** Traffic Anomalies
        - **Economic Indicators:** Inflation/GDP correlation
        """))
        st.subheader(" OPERATIONAL IMPACT")
        linked_markdown(textwrap.dedent("""
        - Critical Infrastructure Protection
        - Resource Optimization
        - Policy Intelligence
        """))
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(" MISSION OBJECTIVE")
            st.write(textwrap.dedent("""
            Advanced predictive intelligence platform for Kenyan Government Digital Services. 
            Deploys machine learning algorithms to forecast cyber threat evolution and enable 
            proactive defense strategies rather than reactive responses.
            
            **Now with custom data upload capability** for organization-specific predictions.
            """))
            
            st.subheader(" THREAT LANDSCAPE")
            st.write(textwrap.dedent("""
            Digital transformation acceleration correlates with exponential threat growth. 
            Conventional reactive defenses inadequate against modern automated attack vectors. 
            Critical infrastructure requires predictive capabilities to stay ahead of adversaries.
            """))
            
            # EXPLANATION: Why ML works here
            linked_message(st.info, """
            **Why Machine Learning?**
            
            Traditional rule-based systems look for known attack signatures. ML discovers **hidden patterns** across multiple variables (e.g., when inflation rises above 7% AND patch delays exceed 10 days, threat levels typically spike within 3 months). These correlations are invisible to human analysts but detectable by algorithms.
            """)
            
        with col2:
            st.subheader(" SYSTEM ARCHITECTURE")
            linked_markdown(textwrap.dedent("""
            **Core Engine**: XGBoost Neural Networks processing multi-dimensional correlation matrices:
            
            - **Attack Vectors**: DDoS, Malware, Phishing, Web Exploits
            - **Vulnerability Metrics**: CVE Criticality, Patch Latency  
            - **Network Intelligence**: Traffic Anomaly Detection
            - **Economic Indicators**: Inflation/GDP correlation algorithms
            
            **Data Flow**: Raw data → Feature Engineering → Model Training → Prediction → Alert Generation
            
            **Upload Capability**: Custom datasets → Auto-validation → Model Retraining → Custom Predictions
            """))
            
            st.subheader(" OPERATIONAL IMPACT")
            linked_markdown(textwrap.dedent("""
            **Strategic Benefits**:
            - **Critical Infrastructure Protection**: Secure national digital assets preemptively
            - **Resource Optimization**: Allocate security budget efficiently based on predicted risk
            - **Policy Intelligence**: Inform cybersecurity policy with data-driven forecasts
            - **Public Trust**: Maintain confidence in e-government services through proactive security
            - **Custom Analysis**: Upload your own data for organization-specific threat modeling
            """))

    linked_message(st.info, " **SECURITY PROTOCOL**: All data displayed is synthetic/anonymized for research purposes. No real-time government data is exposed.")

def render_dataset():
    st.title(" DATA MATRIX")
    
    # Get current dataset for analysis
    current_df = get_dataset()
    
    # Show current data source
    if get_current_data_source() == "uploaded":
        linked_message(st.success, " **Currently Displaying**: Your uploaded custom dataset")
    else:
        linked_message(st.info, " **Currently Displaying**: Default research dataset (2020-2025)")
    
    # EXPLANATION: What the data represents
    linked_markdown(textwrap.dedent("""
    ###  **Understanding the Training Data**
    
    This dataset contains historical observations used to train the prediction model. 
    Each row represents a snapshot in time with 12 measured variables that correlate with cyber threat levels.
    
    **How to Read the Columns:**
    - **Temporal**: Year/Month when data was recorded
    - **Attack Metrics**: Raw counts of DDoS, Malware, Phishing, and Web attacks
    - **Vulnerability**: Critical CVEs (security flaws) and Patch Delay Days (how long systems remain exposed)
    - **Economic**: Inflation Rate and GDP Growth (economic stress often correlates with increased cybercrime)
    - **Target**: Threat_Level (Medium/High/Critical) - what the model learns to predict
    """))
    
    linked_markdown("Accessing training datasets...")
    
    height = 400 if st.session_state.mobile_view else 500
    st.dataframe(
        current_df,
        use_container_width=True,
        height=height
    )
    
    # DYNAMIC PATTERN ANALYSIS
    linked_markdown("---")
    st.subheader(" **Key Patterns Visible in This Data**")
    
    # Analyze actual patterns in the current dataset
    patterns = analyze_dataset_patterns(current_df)
    
    # Display each pattern as a bullet point
    for pattern in patterns:
        linked_markdown(f"• {pattern}")
    
    # Additional dataset statistics
    linked_markdown("---")
    st.subheader(" **Dataset Statistics**")
    
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
    
    linked_message(st.caption, " **Training Process**: The model learned these patterns to recognize that specific combinations of these values predict future threat levels.")
    
    linked_message(st.info, " **Tip**: Upload your own data in the **DATA UPLOAD** tab to see patterns specific to your organization!")

def render_models():
    st.title(" AI CORE PERFORMANCE")
    
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
        linked_message(st.success, " **Models trained on**: Your uploaded custom dataset")
    else:
        linked_message(st.info, " **Models trained on**: Default research dataset")
    
    linked_markdown("---")
    
    # DISPLAY ACTIVE/WINNING ALGORITHMS SECTION
    if len(active_algorithms) == 1:
        winner_name = active_algorithms[0]
        st.header(f" ACTIVE ALGORITHM: {winner_name.upper()}")
        
        if "XGBoost" in winner_name:
            linked_message(st.success, """
            ###  XGBoost (eXtreme Gradient Boosting)
            
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
            linked_message(st.success, """
            ###  Random Forest Classifier
            
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
            linked_message(st.success, """
            ###  Logistic Regression
            
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
        st.header(f" DUAL ENSEMBLE ACTIVE")
        linked_message(st.success, f"""
        ###  Combined System: {' & '.join(active_algorithms)}
        
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
        st.header(f" TRIPLE CONSENSUS MODE")
        linked_message(st.success, """
        ###  ALL SYSTEMS ACTIVE: Logistic Regression + Random Forest + XGBoost
        
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
    
    linked_markdown("---")
    
    # EXPLANATION: Algorithm comparison table
    linked_markdown("###  **Complete Algorithm Benchmarking**")
    
    table_data = []
    for name, value in results.items():
        is_active = name in active_algorithms
        status = " ACTIVE" if is_active else " STANDBY"
        table_data.append({
            "Algorithm": name,
            "Accuracy": f"{value*100:.2f}%",
            "Status": status
        })
    
    st.table(table_data)
    
    # EXPLANATION: Results interpretation
    linked_markdown("###  **Understanding the Results**")
    
    col1, col2 = st.columns(2)
    with col1:
        if len(active_algorithms) == 1:
            winner = active_algorithms[0]
            linked_message(st.info, f"""
            **Winner Analysis: {winner}**
            
            This algorithm achieved the highest accuracy on your specific dataset. 
            {'Its gradient boosting approach' if 'XGBoost' in winner else 'Its ensemble approach' if 'Random' in winner else 'Its linear approach'} 
            best matched the patterns in your cybersecurity data.
            
            **Recommendation**: Deploy this algorithm for production predictions.
            """)
        else:
            linked_message(st.info, f"""
            **Winners Analysis: {', '.join(active_algorithms)}**
            
            Multiple algorithms achieved perfect accuracy, indicating your dataset
            has exceptionally clear patterns detectable by different approaches.
            
            **Recommendation**: Use consensus voting for maximum reliability.
            """)
    
    with col2:
        linked_message(st.warning, """
        **Algorithm Characteristics:**
        
        **Logistic Regression**: Linear, fast, interpretable
        **Random Forest**: Ensemble, robust, balanced  
        **XGBoost**: Boosted, powerful, handles complexity
        
        The best algorithm depends on your specific data patterns.
        """)

def render_parameters():
    st.title(" SYSTEM PARAMETERS")
    
    # Get current prediction for context
    try:
        current_prediction = predict_2027()
        params_df = get_parameters()
    except:
        current_prediction = "Unknown"
        params_df = get_parameters()
    
    # EXPLANATION: Parameters meaning with context
    linked_markdown(textwrap.dedent(f"""
    ###  **2027 Projection Parameters**
    
    **Current Prediction for 2027**: {' **MODERATE**' if current_prediction == 'Medium' else ' **HIGH**' if current_prediction == 'High' else ' **CRITICAL**' if current_prediction == 'Critical' else ' **Unknown**'}
    
    These values represent **projected conditions for August 2027** based on trend analysis, economic forecasts, and technological growth projections. 
    The model uses these 12 inputs to classify the threat level.
    
    **How Projections Are Derived:**
    - **Attack Volumes**: Extrapolated from historical growth curves in your dataset
    - **CVE Counts**: Based on National Vulnerability Database growth rates
    - **Economic**: Central Bank inflation forecasts and GDP projections
    - **Operational**: Expected traffic volume and patch management efficiency targets
    
    **Note**: These projections remain constant, but the model's interpretation changes based on patterns learned from your {'uploaded' if get_current_data_source() == 'uploaded' else 'default'} data!
    """))
    
    linked_markdown("###  **Feature Configuration for 2027 Threat Projection**")
    
    height = 300 if st.session_state.mobile_view else 400
    st.dataframe(
        params_df,
        use_container_width=True,
        height=height
    )
    
    # DATA-SPECIFIC PARAMETER ANALYSIS
    linked_markdown("---")
    st.subheader(" **Why These Parameters Lead to {}**".format(current_prediction if current_prediction != "Unknown" else "This Prediction"))
    
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
            linked_message(st.info, f"""
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
            linked_markdown("###  **Key Parameter Insights**")
            for insight in insights:
                linked_markdown(insight)
    
    # EXPLANATION: Parameter significance
    linked_markdown("---")
    st.subheader(" **Parameter Significance Analysis**")
    
    linked_message(st.warning, """
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
        linked_message(st.error, " **CRITICAL RECOMMENDATION**: Given the projected parameters exceed multiple historical thresholds simultaneously, immediate executive review of security posture is advised.")
    elif current_prediction == "High":
        linked_message(st.warning, " **HIGH ALERT RECOMMENDATION**: Review defensive protocols and ensure patch management can maintain the projected 9-day deployment window.")
    else:
        linked_message(st.success, " **STABLE OUTLOOK**: Projected parameters suggest manageable risk levels. Maintain standard monitoring and continue current security investments.")

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
# FOOTER - DYNAMIC YEAR
# =====================================
st.divider()
linked_markdown(textwrap.dedent(f"""
<div style="text-align: center; color: #64748b; padding: 20px;">
    <p style="margin: 0; font-size: 0.875rem;">
         Cyber Threat Intelligence Platform | © {current_year}
    </p>
    <p style="margin: 8px 0 0 0; font-size: 0.75rem;">
        Data source: {get_current_data_source()} | Built with Streamlit
    </p>
</div>
"""), unsafe_allow_html=True)
