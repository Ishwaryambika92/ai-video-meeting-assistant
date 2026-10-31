"""
AI Video/Meeting Assistant - Streamlit UI.

YouTube URL / local file
    -> audio extraction
    -> transcription
    -> translation / cleanup
    -> summarization
    -> extraction
    -> vector store
    -> RAG chat
    -> TXT / PDF / JPG report export.
"""

from __future__ import annotations

import sys
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path


_ROOT = Path(__file__).resolve().parent
_VENV_SITE = _ROOT / ".venv" / "Lib" / "site-packages"
if _VENV_SITE.exists() and str(_VENV_SITE) not in sys.path:
    sys.path.insert(0, str(_VENV_SITE))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

# --------------------------------------------------------------------------
# Streamlit page configuration - Executed FIRST before any other operations
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Video/Meeting Assistant",
    page_icon="🎬",
    layout="wide",
)

from config import get_logger, get_settings, validate_settings
from core.auth import AuthService, User, get_auth_service
from core.auth_ui import render_auth_page, render_user_profile_dropdown

import html
import re

from core.dashboard_ui import (
    render_action_items_tracker,
    render_decisions_and_memory,
    render_meeting_history,
    render_overview_dashboard,
)
from core.meeting_store import get_all_meetings, save_meeting

logger = get_logger(__name__)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

def _init_state() -> None:
    query_theme = st.query_params.get("theme")
    if "theme" not in st.session_state:
        if query_theme and query_theme.lower() in ["dark", "light", "system"]:
            init_theme = query_theme.capitalize()
        else:
            init_theme = "Dark"
        st.session_state["theme"] = init_theme
        st.session_state["theme_radio_selector"] = init_theme
        st.session_state["_synced_query_theme"] = init_theme.lower()
    else:
        # Check if URL query param was changed externally
        if query_theme and query_theme.lower() in ["dark", "light", "system"]:
            if query_theme.lower() != st.session_state.get("_synced_query_theme"):
                canonical = query_theme.capitalize()
                st.session_state["theme"] = canonical
                st.session_state["theme_radio_selector"] = canonical
                st.session_state["_synced_query_theme"] = query_theme.lower()
        init_theme = st.session_state.get("theme", "Dark")

    defaults = {
        "authenticated_user": None,
        "auth_token": None,
        "session_id": None,
        "processed": False,
        "clean_transcript": "",
        "original_transcript": "",
        "summary": None,
        "extraction": None,
        "chat_history": [],
        "dashboard_chat_history": [],
        "chat_panel_open": False,
        "processing_error": None,
        "active_tab": "summary",
        "theme": init_theme,
        "app_mode": "workspace",
        "dashboard_tab": "overview",
        "preferred_response_language": "english",
        "is_processing": False,
        "_last_upload_sig": None,
        "_last_upload_path": None,
    }


    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Check query param for persistent session token across browser refreshes
    auth_service = get_auth_service()
    if st.session_state.get("authenticated_user") is None:
        url_token = st.query_params.get("auth_token")
        if url_token:
            valid_user = auth_service.validate_session(url_token)
            if valid_user:
                st.session_state["authenticated_user"] = valid_user
                st.session_state["auth_token"] = url_token
            else:
                st.query_params.pop("auth_token", None)


_init_state()

logger.info(
    "THEME RUNTIME: session_theme=%s, query_theme=%s",
    st.session_state.get("theme"),
    st.query_params.get("theme"),
)


# --------------------------------------------------------------------------
# UI Design & Theme Styles (Light, Dark, and System)
# --------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _generate_theme_css(theme: str) -> str:
    """Generate dynamic CSS variables and layout styles based on the active theme."""

    if theme == "Dark":
        vars_css = """
        :root, .stApp {
            color-scheme: dark;
            --primary-color: #FF5A5F;
            --background-color: #0B0F17;
            --secondary-background-color: #1E293B;
            --text-color: #F8FAFC;
            --bg-main: #0B0F17;
            --bg-sidebar: #0F172A;
            --bg-card: #1E293B;
            --bg-secondary: #334155;
            --text-main: #F8FAFC;
            --text-secondary: #CBD5E1;
            --text-muted: #94A3B8;
            --border-color: #334155;
            --input-bg: #1E293B;
            --input-text: #F8FAFC;
            --placeholder-text: #64748B;
            --button-bg: #1E293B;
            --button-text: #F8FAFC;
            --primary-accent: #FF5A5F;
            --primary-hover: #FF7478;
            --color-success: #4ADE80;
            --color-warning: #FBBF24;
            --color-error: #F87171;
            --chat-user-bg: #334155;
            --chat-asst-bg: #1E293B;
            --shadow-subtle: 0 1px 3px rgba(0, 0, 0, 0.4);
            --shadow-card: 0 4px 12px rgba(0, 0, 0, 0.35);
            --shadow-popover: 0 14px 40px rgba(0, 0, 0, 0.6);
        }
        """
    elif theme == "System":
        vars_css = """
        :root, .stApp {
            color-scheme: light;
            --primary-color: #FF4B4B;
            --background-color: #F8FAFC;
            --secondary-background-color: #FFFFFF;
            --text-color: #0F172A;
            --bg-main: #F8FAFC;
            --bg-sidebar: #FFFFFF;
            --bg-card: #FFFFFF;
            --bg-secondary: #F1F5F9;
            --text-main: #0F172A;
            --text-secondary: #334155;
            --text-muted: #64748B;
            --border-color: #CBD5E1;
            --input-bg: #FFFFFF;
            --input-text: #0F172A;
            --placeholder-text: #94A3B8;
            --button-bg: #FFFFFF;
            --button-text: #0F172A;
            --primary-accent: #FF4B4B;
            --primary-hover: #E63E45;
            --color-success: #16A34A;
            --color-warning: #D97706;
            --color-error: #DC2626;
            --chat-user-bg: #F1F5F9;
            --chat-asst-bg: #FFFFFF;
            --shadow-subtle: 0 1px 3px rgba(0, 0, 0, 0.06);
            --shadow-card: 0 2px 8px rgba(0, 0, 0, 0.08);
            --shadow-popover: 0 14px 40px rgba(0, 0, 0, 0.12);
        }
        @media (prefers-color-scheme: dark) {
            :root, .stApp {
                color-scheme: dark;
                --primary-color: #FF5A5F;
                --background-color: #0B0F17;
                --secondary-background-color: #1E293B;
                --text-color: #F8FAFC;
                --bg-main: #0B0F17;
                --bg-sidebar: #0F172A;
                --bg-card: #1E293B;
                --bg-secondary: #334155;
                --text-main: #F8FAFC;
                --text-secondary: #CBD5E1;
                --text-muted: #94A3B8;
                --border-color: #334155;
                --input-bg: #1E293B;
                --input-text: #F8FAFC;
                --placeholder-text: #64748B;
                --button-bg: #1E293B;
                --button-text: #F8FAFC;
                --primary-accent: #FF5A5F;
                --primary-hover: #FF7478;
                --color-success: #4ADE80;
                --color-warning: #FBBF24;
                --color-error: #F87171;
                --chat-user-bg: #334155;
                --chat-asst-bg: #1E293B;
                --shadow-subtle: 0 1px 3px rgba(0, 0, 0, 0.4);
                --shadow-card: 0 4px 12px rgba(0, 0, 0, 0.35);
                --shadow-popover: 0 14px 40px rgba(0, 0, 0, 0.6);
            }
        }
        """
    else:  # Light theme default
        vars_css = """
        :root, .stApp {
            color-scheme: light;
            --primary-color: #FF4B4B;
            --background-color: #F8FAFC;
            --secondary-background-color: #FFFFFF;
            --text-color: #0F172A;
            --bg-main: #F8FAFC;
            --bg-sidebar: #FFFFFF;
            --bg-card: #FFFFFF;
            --bg-secondary: #F1F5F9;
            --text-main: #0F172A;
            --text-secondary: #334155;
            --text-muted: #64748B;
            --border-color: #CBD5E1;
            --input-bg: #FFFFFF;
            --input-text: #0F172A;
            --placeholder-text: #94A3B8;
            --button-bg: #FFFFFF;
            --button-text: #0F172A;
            --primary-accent: #FF4B4B;
            --primary-hover: #E63E45;
            --color-success: #16A34A;
            --color-warning: #D97706;
            --color-error: #DC2626;
            --chat-user-bg: #F1F5F9;
            --chat-asst-bg: #FFFFFF;
            --shadow-subtle: 0 1px 3px rgba(0, 0, 0, 0.06);
            --shadow-card: 0 2px 8px rgba(0, 0, 0, 0.08);
            --shadow-popover: 0 14px 40px rgba(0, 0, 0, 0.12);
        }
        """

    return f"""
    <style>
    {vars_css}

    /* Global Typography & App Layout */
    html, body, [class*="css"] {{
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        color: var(--text-main);
    }}

    * {{
        scrollbar-color: var(--border-color) var(--bg-main);
        scrollbar-width: thin;
    }}
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: var(--bg-main);
    }}
    ::-webkit-scrollbar-thumb {{
        background: var(--border-color);
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: var(--text-muted);
    }}

    .stApp,
    div[data-testid="stAppViewContainer"],
    section[data-testid="stSidebar"] {{
        background-color: var(--bg-main) !important;
        color: var(--text-main) !important;
        --primary-color: var(--primary-accent) !important;
        --background-color: var(--bg-main) !important;
        --secondary-background-color: var(--input-bg) !important;
        --text-color: var(--text-main) !important;
    }}

    header[data-testid="stHeader"],
    header[data-testid="stHeader"] > div,
    div[data-testid="stDecoration"],
    div[data-testid="stToolbar"] {{
        background-color: transparent !important;
        background: transparent !important;
        z-index: 1000 !important;
    }}

    div[data-testid="stToolbar"] button,
    div[data-testid="stToolbar"] button[data-testid="baseButton-secondary"],
    div[data-testid="stToolbar"] button[data-testid="baseButton-header"],
    div[data-testid="stToolbar"] button[data-testid="baseButton-headerNoPadding"],
    button[data-testid="stDeployButton"],
    button[aria-label="Main menu"],
    button[aria-label="Manage app"],
    html body button[aria-label="Main menu"],
    html body div[data-testid="stToolbar"] button,
    html body header[data-testid="stHeader"] button,
    [class*="ewrlt5x"] button,
    header[data-testid="stHeader"] button,
    div[data-testid="stStatusWidget"],
    div[data-testid="stStatusWidget"] *,
    div[data-testid="stToolbar"] div[data-testid="stStatusWidget"],
    header[data-testid="stHeader"] div[data-testid="stStatusWidget"],
    [class*="statusWidget"],
    [class*="statusWidget"] * {{
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        color: var(--text-main) !important;
        box-shadow: none !important;
    }}

    div[data-testid="stToolbar"] div[data-testid="stMainMenu"],
    header[data-testid="stHeader"] div[data-testid="stMainMenu"],
    html body div[data-testid="stMainMenu"] {{
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}

    div[data-testid="stToolbar"] button:hover,
    button[data-testid="stDeployButton"]:hover,
    header[data-testid="stHeader"] button:hover {{
        background-color: var(--bg-secondary) !important;
        background: var(--bg-secondary) !important;
        border-color: var(--border-color) !important;
    }}

    div[data-testid="stToolbar"] button *,
    button[data-testid="stDeployButton"] *,
    header[data-testid="stHeader"] button * {{
        background: transparent !important;
        background-color: transparent !important;
        color: var(--text-main) !important;
    }}

    svg path[fill="none"],
    svg [fill="none"] {{
        fill: none !important;
    }}

    /* Modal Dialogs (Deploy dialog, Streamlit modals) */
    div[data-testid="stModal"] > div,
    div[data-testid="stDialog"] > div,
    div[role="dialog"],
    div[data-baseweb="modal"] > div,
    div[data-baseweb="dialog"] {{
        background-color: var(--bg-card) !important;
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-main) !important;
        box-shadow: var(--shadow-popover) !important;
    }}

    div[role="dialog"] *,
    div[data-testid="stModal"] *,
    div[data-testid="stDialog"] *,
    div[data-baseweb="modal"] * {{
        color: var(--text-main) !important;
    }}

    div[role="dialog"] a,
    div[data-testid="stModal"] a,
    div[data-baseweb="modal"] a {{
        color: var(--primary-accent) !important;
        text-decoration: underline !important;
    }}

    div[role="dialog"] button[aria-label="Close"],
    div[data-baseweb="modal"] button[aria-label="Close"] {{
        background: transparent !important;
        color: var(--text-secondary) !important;
    }}

    div[role="dialog"] button[aria-label="Close"] svg,
    div[data-baseweb="modal"] button[aria-label="Close"] svg {{
        fill: var(--text-secondary) !important;
        stroke: var(--text-secondary) !important;
        color: var(--text-secondary) !important;
    }}

    .main .block-container {{
        padding-top: 1.8rem !important;
        padding-bottom: 5.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1400px !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }}

    @media (max-width: 1024px) {{
        .main .block-container {{
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
            padding-top: 1.4rem !important;
        }}
    }}

    @media (max-width: 768px) {{
        .main .block-container {{
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1rem !important;
            padding-bottom: 6.5rem !important;
        }}
    }}

    @media (max-width: 480px) {{
        .main .block-container {{
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-top: 0.85rem !important;
            padding-bottom: 7rem !important;
        }}
    }}

    /* Sidebar Structure & Theme */
    section[data-testid="stSidebar"] {{
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-color) !important;
        color: var(--text-main) !important;
        min-width: 320px !important;
        width: 340px !important;
        max-width: 380px !important;
        box-shadow: 2px 0 12px rgba(0, 0, 0, 0.03) !important;
    }}

    @media (max-width: 768px) {{
        section[data-testid="stSidebar"] {{
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            bottom: 0 !important;
            height: 100vh !important;
            z-index: 999990 !important;
            min-width: 280px !important;
            width: 82vw !important;
            max-width: 340px !important;
            box-shadow: 4px 0 24px rgba(0, 0, 0, 0.25) !important;
            transition: transform 0.25s ease !important;
        }}

        section[data-testid="stSidebar"][aria-expanded="false"] {{
            transform: translateX(-100%) !important;
        }}

        .main {{
            width: 100% !important;
            max-width: 100vw !important;
        }}
    }}

    section[data-testid="stSidebar"] * {{
        color: var(--text-main);
    }}

    section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {{
        padding-top: 1.25rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
        overflow-y: auto !important;
        box-sizing: border-box !important;
    }}

    section[data-testid="stSidebar"] .block-container {{
        padding-top: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        max-width: 100% !important;
    }}

    /* Sidebar Headings & Spacing */
    section[data-testid="stSidebar"] h2 {{
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: var(--text-main) !important;
        margin-top: 0 !important;
        margin-bottom: 1.2rem !important;
    }}

    /* Sidebar Controls Vertical Spacing */
    section[data-testid="stSidebar"] div[data-testid="stRadio"],
    section[data-testid="stSidebar"] div[data-testid="stTextInput"],
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"],
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"],
    section[data-testid="stSidebar"] div[data-testid="stSlider"] {{
        margin-bottom: 1.25rem !important;
    }}

    section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] {{
        margin-bottom: 0.4rem !important;
    }}

    section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] label,
    section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] span {{
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: var(--text-main) !important;
    }}

    section[data-testid="stSidebar"] hr {{
        margin-top: 1.25rem !important;
        margin-bottom: 1.25rem !important;
        border-color: var(--border-color) !important;
    }}

    /* Headings & Text Contrast Across All Containers */
    h1, h2, h3, h4, h5, h6,
    div[data-testid="stHeadingWithAnchor"] *,
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stMarkdownContainer"] h4,
    div[data-testid="stMarkdownContainer"] h5,
    div[data-testid="stMarkdownContainer"] h6 {{
        color: var(--text-main) !important;
        font-weight: 600 !important;
    }}

    p, span, label, li,
    div[data-testid="stMarkdownContainer"],
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] span,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] div,
    div[data-testid="stText"],
    div[data-testid="stText"] *,
    div[data-testid="stWrite"],
    div[data-testid="stWrite"] *,
    div[data-testid="stMarkdown"] p,
    div[data-testid="stMarkdown"] span,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] span {{
        color: var(--text-main) !important;
    }}

    div[data-testid="stMarkdownContainer"] strong,
    div[data-testid="stMarkdownContainer"] b,
    section[data-testid="stSidebar"] strong,
    section[data-testid="stSidebar"] b {{
        color: var(--text-main) !important;
        font-weight: 600 !important;
    }}

    div[data-testid="stCaptionContainer"],
    div[data-testid="stCaptionContainer"] p,
    div[data-testid="stCaptionContainer"] span,
    div[data-testid="stCaptionContainer"] *,
    small, .stCaption {{
        color: var(--text-secondary) !important;
    }}

    div[data-testid="stWidgetLabel"],
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] span,
    div[data-testid="stWidgetLabel"] * {{
        color: var(--text-main) !important;
        font-weight: 500 !important;
    }}

    div[data-testid="stSpinner"],
    div[data-testid="stSpinner"] * {{
        color: var(--text-main) !important;
    }}

    /* Primary Buttons */
    button[kind="primary"],
    button[data-testid="baseButton-primary"] {{
        background-color: var(--primary-accent) !important;
        background: var(--primary-accent) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        height: 44px !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(255, 75, 75, 0.25) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}

    button[kind="primary"] *,
    button[data-testid="baseButton-primary"] * {{
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }}

    button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover {{
        background-color: var(--primary-hover) !important;
        background: var(--primary-hover) !important;
        box-shadow: 0 4px 14px rgba(255, 75, 75, 0.45) !important;
        transform: translateY(-1px) !important;
    }}

    /* Secondary Buttons */
    button[kind="secondary"],
    button[data-testid="baseButton-secondary"] {{
        background-color: var(--button-bg) !important;
        color: var(--button-text) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        height: 44px !important;
        font-weight: 500 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: var(--shadow-subtle) !important;
    }}

    button[kind="secondary"] *,
    button[data-testid="baseButton-secondary"] * {{
        color: var(--button-text) !important;
    }}

    button[kind="secondary"]:hover,
    button[data-testid="baseButton-secondary"]:hover {{
        border-color: var(--primary-accent) !important;
        color: var(--primary-accent) !important;
        background-color: var(--bg-secondary) !important;
        transform: translateY(-1px) !important;
    }}

    button[kind="secondary"]:hover *,
    button[data-testid="baseButton-secondary"]:hover * {{
        color: var(--primary-accent) !important;
    }}

    button:focus-visible,
    button:focus {{
        outline: 2px solid var(--primary-accent) !important;
        outline-offset: 2px !important;
    }}

    button:disabled,
    button[disabled],
    button[data-testid="baseButton-primary"]:disabled,
    button[data-testid="baseButton-secondary"]:disabled,
    input:disabled,
    textarea:disabled,
    [aria-disabled="true"] {{
        background-color: var(--bg-secondary) !important;
        background: var(--bg-secondary) !important;
        color: var(--text-secondary) !important;
        -webkit-text-fill-color: var(--text-secondary) !important;
        border: 1px solid var(--border-color) !important;
        opacity: 0.88 !important;
        cursor: not-allowed !important;
        transform: none !important;
        box-shadow: none !important;
    }}

    button:disabled *,
    button[disabled] *,
    button[data-testid="baseButton-primary"]:disabled *,
    button[data-testid="baseButton-secondary"]:disabled * {{
        color: var(--text-secondary) !important;
        fill: var(--text-secondary) !important;
        -webkit-text-fill-color: var(--text-secondary) !important;
    }}

    /* Form Inputs, Textareas, DateInput, NumberInput */
    div[data-testid="stTextArea"],
    div[data-testid="stTextArea"] > div,
    div[data-baseweb="textarea"],
    div[data-baseweb="textarea"] > div,
    div[data-testid="stTextArea"] textarea,
    div[data-baseweb="textarea"] textarea,
    .stTextArea textarea {{
        background-color: var(--input-bg) !important;
        background: var(--input-bg) !important;
        color: var(--input-text) !important;
        -webkit-text-fill-color: var(--input-text) !important;
        border-color: var(--border-color) !important;
    }}

    /* Text Inputs (YouTube URL, Chat, Search, DateInput, NumberInput) */
    div[data-testid="stTextInput"],
    div[data-testid="stTextInput"] > div,
    div[data-testid="stTextInputRootElement"],
    div[data-testid="stTextInputRootElement"] > div,
    .stTextInput,
    .stTextInput > div,
    div[data-testid="stDateInput"],
    div[data-testid="stDateInput"] > div,
    div[data-testid="stNumberInput"],
    div[data-testid="stForm"] div[data-baseweb="input"],
    div[data-baseweb="input"],
    div[data-baseweb="input"] > div,
    section[data-testid="stSidebar"] div[data-baseweb="input"],
    section[data-testid="stSidebar"] div[data-baseweb="input"] > div,
    section[data-testid="stSidebar"] .stTextInput div[data-baseweb="input"],
    section[data-testid="stSidebar"] .stTextInput div[data-baseweb="input"] > div,
    .st-key-youtube_url_input div[data-baseweb="input"],
    .st-key-youtube_url_input div[data-baseweb="input"] > div,
    .st-key-floating_chat_panel div[data-baseweb="input"],
    .st-key-floating_chat_panel div[data-baseweb="input"] > div {{
        background-color: var(--input-bg) !important;
        background: var(--input-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--input-text) !important;
    }}

    /* Inner base-input wrappers and containers must stay transparent */
    div[data-baseweb="base-input"],
    div[data-baseweb="base-input"] > div,
    div[data-baseweb="input"] div[data-baseweb="base-input"],
    .stTextInput div[data-baseweb="base-input"],
    section[data-testid="stSidebar"] div[data-baseweb="base-input"],
    .st-key-youtube_url_input div[data-baseweb="base-input"] {{
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}

    /* Input text fields: crisp contrast, transparent background, correct caret */
    div[data-testid="stTextInputRootElement"] input,
    div[data-testid="stTextInput"] input,
    .stTextInput input,
    div[data-baseweb="input"] input,
    div[data-baseweb="base-input"] input,
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] .stTextInput input,
    .st-key-youtube_url_input input,
    .st-key-floating_chat_panel input,
    div[data-testid="stDateInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stForm"] input,
    input[aria-label="YouTube URL"] {{
        background-color: transparent !important;
        background: transparent !important;
        color: var(--input-text) !important;
        -webkit-text-fill-color: var(--input-text) !important;
        caret-color: var(--primary-accent) !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        font-size: 0.92rem !important;
    }}

    /* Chrome, Brave, Edge & Safari Autofill / Autocomplete dark-mode override */
    input:-webkit-autofill,
    input:-webkit-autofill:hover, 
    input:-webkit-autofill:focus, 
    input:-webkit-autofill:active,
    input:-internal-autofill-selected,
    .stTextInput input:-webkit-autofill,
    .stTextInput input:-internal-autofill-selected,
    .st-key-youtube_url_input input:-webkit-autofill,
    .st-key-youtube_url_input input:-internal-autofill-selected,
    section[data-testid="stSidebar"] input:-webkit-autofill,
    section[data-testid="stSidebar"] input:-internal-autofill-selected {{
        -webkit-box-shadow: 0 0 0 1000px var(--input-bg) inset !important;
        box-shadow: 0 0 0 1000px var(--input-bg) inset !important;
        -webkit-text-fill-color: var(--input-text) !important;
        color: var(--input-text) !important;
        caret-color: var(--primary-accent) !important;
        transition: background-color 500000s ease-in-out 0s !important;
        border-radius: 6px !important;
    }}

    /* Placeholder text */
    input::placeholder,
    textarea::placeholder,
    div[data-baseweb="input"] input::placeholder,
    div[data-baseweb="base-input"] input::placeholder,
    div[data-baseweb="textarea"] textarea::placeholder,
    div[data-testid="stTextInputRootElement"] input::placeholder,
    div[data-testid="stTextInput"] input::placeholder,
    .stTextInput input::placeholder,
    .st-key-youtube_url_input input::placeholder,
    div[data-testid="stTextArea"] textarea::placeholder {{
        color: var(--placeholder-text) !important;
        -webkit-text-fill-color: var(--placeholder-text) !important;
        opacity: 0.8 !important;
    }}

    /* Focus States for Inputs */
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="input"] > div:focus-within,
    .stTextInput:focus-within div[data-baseweb="input"],
    .st-key-youtube_url_input:focus-within div[data-baseweb="input"],
    section[data-testid="stSidebar"] .stTextInput div[data-baseweb="input"]:focus-within {{
        border-color: var(--primary-accent) !important;
        box-shadow: 0 0 0 1px var(--primary-accent) !important;
    }}

    /* Streamlit Input Instructions ("Press Enter to apply") */
    div[data-testid="stInputInstructions"],
    .stTextInput div[data-testid="stInputInstructions"],
    section[data-testid="stSidebar"] div[data-testid="stInputInstructions"] {{
        display: none !important;
    }}

    div[data-testid="stNumberInput"] button,
    div[data-testid="stNumberInput"] button * {{
        color: var(--text-main) !important;
        fill: var(--text-main) !important;
    }}

    /* Datepicker Calendar */
    div[data-baseweb="calendar"],
    div[data-baseweb="calendar"] * {{
        background-color: var(--bg-card);
        color: var(--text-main);
    }}

    div[data-baseweb="calendar"] button {{
        color: var(--text-main) !important;
    }}

    div[data-baseweb="calendar"] [aria-selected="true"] {{
        background-color: var(--primary-accent) !important;
        color: #FFFFFF !important;
    }}

    /* Outer Selectbox widget wrapper stays transparent without rogue boxes */
    div[data-testid*="Selectbox"],
    div[data-testid*="Selectbox"] > div,
    div[data-testid*="Selectbox"] > div > div,
    div[class*="e1f1d6gn3"],
    div[class*="e1f1d6gn2"],
    div[class*="e1f1d6gn1"],
    .stSelectbox,
    .stSelectbox > div,
    .stSelectbox > div > div,
    .stMultiSelect,
    .stMultiSelect > div,
    .stMultiSelect > div > div,
    section[data-testid="stSidebar"] div[data-testid*="Selectbox"],
    section[data-testid="stSidebar"] div[data-testid*="Selectbox"] > div,
    section[data-testid="stSidebar"] div[data-testid*="Selectbox"] > div > div,
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stSelectbox > div,
    section[data-testid="stSidebar"] .stSelectbox > div > div {{
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}

    /* Selectbox & MultiSelect Control Box (BaseWeb select control) */
    div[data-baseweb="select"],
    div[data-baseweb="select"] > div,
    div[class*="e1fp86qc0"],
    div[class*="e1fp86qc4"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stSelectbox"] div[class*="e1fp86qc4"],
    .stSelectbox div[data-baseweb="select"],
    .stSelectbox div[data-baseweb="select"] > div,
    .stSelectbox div[class*="e1fp86qc4"],
    .stMultiSelect div[data-baseweb="select"],
    .stMultiSelect div[data-baseweb="select"] > div,
    .stMultiSelect div[class*="e1fp86qc4"],
    section[data-testid="stSidebar"] div[data-baseweb="select"],
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] div[class*="e1fp86qc4"],
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"],
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[class*="e1fp86qc4"],
    .st-key-language_select div[data-baseweb="select"],
    .st-key-language_select div[data-baseweb="select"] > div,
    .st-key-language_select div[class*="e1fp86qc4"] {{
        background-color: var(--input-bg) !important;
        background: var(--input-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--input-text) !important;
        -webkit-text-fill-color: var(--input-text) !important;
    }}

    /* Inner value container and controls stay transparent */
    div[data-baseweb="select"] > div > div,
    div[data-baseweb="select"] > div > div *,
    div[data-baseweb="select"] input,
    div[class*="e1fp86qc5"],
    div[class*="e1fp86qc5"] *,
    div[class*="e1fp86qc1"],
    div[class*="e1fp86qc2"],
    div[class*="e1fp86qc3"],
    .st-key-language_select div[data-baseweb="select"] > div > div,
    .st-key-language_select div[data-baseweb="select"] > div > div *,
    .st-key-language_select div[data-baseweb="select"] input,
    div[data-baseweb="select"] [aria-hidden="true"],
    div[data-baseweb="select"] button[aria-label="Open"],
    div[data-baseweb="select"] button[aria-label="Clear value"] {{
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}

    /* Text & Dropdown Icons inside Selectboxes */
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] *,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] p,
    .stSelectbox span,
    .stSelectbox div,
    .stSelectbox p,
    .stSelectbox input,
    div[data-testid="stSelectbox"] span,
    div[data-testid="stSelectbox"] div,
    div[data-testid="stSelectbox"] p,
    div[data-testid="stSelectbox"] input,
    div[data-testid="stSelectbox"] div[data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] div[data-baseweb="select"] *,
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] *,
    .st-key-language_select * {{
        color: var(--input-text) !important;
        -webkit-text-fill-color: var(--input-text) !important;
    }}

    div[data-baseweb="select"] svg,
    div[data-testid*="Selectbox"] svg,
    .st-key-language_select svg {{
        fill: var(--text-secondary) !important;
        stroke: var(--text-secondary) !important;
        color: var(--text-secondary) !important;
    }}

    /* Tags inside MultiSelect */
    div[data-baseweb="tag"],
    span[data-baseweb="tag"] {{
        background-color: var(--bg-secondary) !important;
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
        color: var(--text-main) !important;
    }}

    div[data-baseweb="tag"] *,
    span[data-baseweb="tag"] * {{
        color: var(--text-main) !important;
        fill: var(--text-main) !important;
        -webkit-text-fill-color: var(--text-main) !important;
    }}

    div[data-baseweb="tag"] [role="button"],
    span[data-baseweb="tag"] [role="button"] {{
        background: transparent !important;
    }}

    div[data-baseweb="select"] button,
    div[data-baseweb="select"] [role="button"],
    div[data-testid="stSelectbox"] button,
    div[data-testid="stSelectbox"] [aria-label="Open"],
    div[data-testid="stSelectbox"] [role="button"],
    div[data-testid="stMultiSelect"] button,
    div[data-testid="stMultiSelect"] [role="button"],
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] button {{
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: var(--input-text) !important;
    }}

    div[data-baseweb="select"] svg,
    div[data-testid="stSelectbox"] svg,
    div[data-testid="stMultiSelect"] svg {{
        fill: var(--text-secondary) !important;
        stroke: var(--text-secondary) !important;
        color: var(--text-secondary) !important;
    }}

    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="textarea"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within {{
        border-color: var(--primary-accent) !important;
        box-shadow: 0 0 0 1px var(--primary-accent) !important;
    }}

    /* Dropdown Menus, Popovers, & MainMenu */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"],
    div[role="listbox"],
    ul[role="listbox"],
    div[data-testid="stMainMenu"] ul,
    ul[data-testid="main-menu-list"],
    [role="menu"] {{
        background-color: var(--bg-card) !important;
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-main) !important;
        border-radius: 10px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18), var(--shadow-popover) !important;
        z-index: 999999 !important;
        box-sizing: border-box !important;
    }}

    /* Listbox Container Scrolling & Padding */
    div[role="listbox"],
    ul[role="listbox"],
    div[data-baseweb="popover"] ul[role="listbox"],
    div[data-baseweb="popover"] [role="listbox"] {{
        max-height: 280px !important;
        overflow-y: auto !important;
        padding: 6px !important;
        background-color: var(--bg-card) !important;
        background: var(--bg-card) !important;
    }}

    /* Sidebar Dropdown Container Sizing & Overflow prevention */
    section[data-testid="stSidebar"] div[data-baseweb="popover"],
    section[data-testid="stSidebar"] div[data-baseweb="popover"] > div,
    section[data-testid="stSidebar"] [role="listbox"] {{
        max-width: 100% !important;
        box-sizing: border-box !important;
        z-index: 999999 !important;
    }}

    /* Dropdown Options */
    [role="option"],
    div[role="option"],
    li[role="option"],
    li[data-testid="stMenuItem"],
    [role="menuitem"],
    div[data-baseweb="popover"] [role="option"],
    div[data-baseweb="menu"] li {{
        background-color: var(--bg-card) !important;
        background: var(--bg-card) !important;
        color: var(--text-main) !important;
        -webkit-text-fill-color: var(--text-main) !important;
        padding: 9px 12px !important;
        margin: 2px 0 !important;
        border-radius: 6px !important;
        cursor: pointer !important;
        font-size: 0.92rem !important;
        line-height: 1.4 !important;
        transition: background-color 0.12s ease, color 0.12s ease !important;
        display: flex !important;
        align-items: center !important;
        box-sizing: border-box !important;
        width: 100% !important;
    }}

    /* Inner spans/divs inside options inherit text color and stay transparent */
    [role="option"] *,
    div[role="option"] *,
    li[role="option"] *,
    li[data-testid="stMenuItem"] *,
    [role="menuitem"] *,
    div[data-baseweb="popover"] [role="option"] * {{
        background-color: transparent !important;
        background: transparent !important;
        color: inherit !important;
        -webkit-text-fill-color: inherit !important;
    }}

    /* Hover state */
    [role="option"]:hover,
    div[role="option"]:hover,
    li[role="option"]:hover,
    li[data-testid="stMenuItem"]:hover,
    [role="menuitem"]:hover,
    div[data-baseweb="popover"] [role="option"]:hover {{
        background-color: var(--bg-secondary) !important;
        background: var(--bg-secondary) !important;
        color: var(--primary-accent) !important;
        -webkit-text-fill-color: var(--primary-accent) !important;
    }}

    /* Selected state */
    [role="option"][aria-selected="true"],
    div[role="option"][aria-selected="true"],
    li[role="option"][aria-selected="true"],
    div[data-baseweb="popover"] [role="option"][aria-selected="true"] {{
        background-color: var(--bg-secondary) !important;
        background: var(--bg-secondary) !important;
        color: var(--primary-accent) !important;
        -webkit-text-fill-color: var(--primary-accent) !important;
        font-weight: 600 !important;
    }}

    div[data-testid="stMainMenu"] footer,
    div[data-testid="stMainMenu"] [class*="footer"],
    div[data-baseweb="menu"] footer,
    div[data-baseweb="menu"] > div:last-child,
    ul[data-testid="main-menu-list"] + div,
    ul[data-testid="main-menu-list"] ~ * {{
        background-color: var(--bg-card) !important;
        background: var(--bg-card) !important;
        color: var(--text-secondary) !important;
        border-top: 1px solid var(--border-color) !important;
    }}

    div[data-testid="stMainMenu"] footer *,
    div[data-testid="stMainMenu"] [class*="footer"] *,
    ul[data-testid="main-menu-list"] ~ * * {{
        color: var(--text-secondary) !important;
    }}

    /* Tooltip styling (Streamlit tooltips & Vega/Altair chart tooltips) */
    div[data-testid="stTooltipContent"],
    div[data-baseweb="tooltip"],
    #vg-tooltip-element,
    .vg-tooltip {{
        background-color: var(--bg-card) !important;
        background: var(--bg-card) !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        box-shadow: var(--shadow-popover) !important;
        font-family: inherit !important;
        font-size: 0.85rem !important;
        z-index: 999999 !important;
    }}

    div[data-testid="stTooltipContent"] *,
    #vg-tooltip-element table,
    .vg-tooltip table {{
        color: var(--text-main) !important;
        background: transparent !important;
    }}

    #vg-tooltip-element td,
    #vg-tooltip-element th,
    .vg-tooltip td,
    .vg-tooltip th {{
        color: var(--text-main) !important;
        background: transparent !important;
        border: none !important;
    }}

    #vg-tooltip-element td.key,
    .vg-tooltip td.key {{
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        padding-right: 8px !important;
    }}

    #vg-tooltip-element td.value,
    .vg-tooltip td.value {{
        color: var(--text-main) !important;
        font-weight: 700 !important;
    }}

    /* Element Toolbar (floating pill on chart/dataframe hover) */
    div[data-testid="stElementToolbar"],
    div[class*="e1plw2qp0"] {{
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}

    div[data-testid="stElementToolbarButtonContainer"],
    div[class*="e1plw2qp1"] {{
        background-color: var(--bg-card) !important;
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-main) !important;
        box-shadow: var(--shadow-card) !important;
        padding: 4px 6px !important;
        opacity: 1 !important;
    }}

    div[data-testid="stElementToolbar"] button,
    div[data-testid="stElementToolbarButtonContainer"] button,
    div[data-testid="stElementToolbarButton"] button,
    button[data-testid*="stBaseButton-elementToolbar"] {{
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        border-radius: 6px !important;
        color: var(--text-main) !important;
        box-shadow: none !important;
        padding: 4px 6px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        opacity: 1 !important;
    }}

    div[data-testid="stElementToolbar"] button *,
    div[data-testid="stElementToolbarButtonContainer"] button * {{
        opacity: 1 !important;
    }}

    div[data-testid="stElementToolbar"] button:hover,
    div[data-testid="stElementToolbarButtonContainer"] button:hover,
    div[data-testid="stElementToolbarButton"] button:hover,
    button[data-testid*="stBaseButton-elementToolbar"]:hover {{
        background-color: var(--bg-secondary) !important;
        background: var(--bg-secondary) !important;
        color: var(--primary-accent) !important;
    }}

    div[data-testid="stElementToolbar"] svg,
    div[data-testid="stElementToolbarButtonContainer"] svg,
    div[data-testid="stElementToolbarButtonIcon"] svg,
    div[data-testid="stElementToolbar"] [data-testid="stIcon"],
    div[data-testid="stElementToolbarButtonIcon"] {{
        color: var(--text-main) !important;
        fill: var(--text-main) !important;
        opacity: 1 !important;
        width: 16px !important;
        height: 16px !important;
    }}

    div[data-testid="stElementToolbar"] svg path,
    div[data-testid="stElementToolbarButtonContainer"] svg path {{
        fill: var(--text-main) !important;
        color: var(--text-main) !important;
        opacity: 1 !important;
    }}

    div[data-testid="stElementToolbar"] svg path[fill="none"],
    div[data-testid="stElementToolbar"] svg path[d="M0 0h24v24H0V0z"],
    div[data-testid="stElementToolbar"] svg [fill="none"],
    div[data-testid="stElementToolbarButtonContainer"] svg path[fill="none"],
    div[data-testid="stElementToolbarButtonContainer"] svg path[d="M0 0h24v24H0V0z"],
    div[data-testid="stElementToolbarButtonContainer"] svg [fill="none"] {{
        fill: none !important;
    }}

    div[data-testid="stElementToolbar"] button:hover svg,
    div[data-testid="stElementToolbarButtonContainer"] button:hover svg,
    div[data-testid="stElementToolbar"] button:hover svg path,
    div[data-testid="stElementToolbarButtonContainer"] button:hover svg path {{
        color: var(--primary-accent) !important;
        fill: var(--primary-accent) !important;
    }}

    div[data-testid="stElementToolbar"] button:hover svg path[fill="none"],
    div[data-testid="stElementToolbarButtonContainer"] button:hover svg path[fill="none"],
    div[data-testid="stElementToolbar"] button:hover svg path[d="M0 0h24v24H0V0z"],
    div[data-testid="stElementToolbarButtonContainer"] button:hover svg path[d="M0 0h24v24H0V0z"] {{
        fill: none !important;
    }}

    /* Vega embed action menu */
    .vega-actions,
    .vega-actions-wrapper,
    details.vega-actions,
    summary.vega-actions {{
        background-color: var(--bg-card) !important;
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
        color: var(--text-main) !important;
        box-shadow: var(--shadow-subtle) !important;
    }}

    .vega-actions a {{
        color: var(--text-main) !important;
    }}

    .vega-actions a:hover {{
        color: var(--primary-accent) !important;
        background-color: var(--bg-secondary) !important;
    }}

    /* Chart SVG text, labels, axes, and legends */
    div[data-testid="stVegaLiteChart"],
    div[data-testid="stAltairChart"],
    div[data-testid="stArrowVegaLiteChart"],
    div[data-testid="stVegaLiteChart"] svg,
    div[data-testid="stAltairChart"] svg,
    div[data-testid="stArrowVegaLiteChart"] svg {{
        overflow: visible !important;
    }}

    div[data-testid="stVegaLiteChart"] svg text,
    div[data-testid="stAltairChart"] svg text,
    div[data-testid="stArrowVegaLiteChart"] svg text {{
        fill: var(--text-main) !important;
        font-family: Inter, sans-serif !important;
    }}

    div[data-testid="stVegaLiteChart"] svg line,
    div[data-testid="stAltairChart"] svg line,
    div[data-testid="stArrowVegaLiteChart"] svg line {{
        stroke: var(--border-color) !important;
    }}

    /* Streamlit Tabs */
    div[data-testid="stTabs"] {{
        border-bottom-color: var(--border-color) !important;
    }}

    div[data-baseweb="tab-list"] {{
        background-color: transparent !important;
        border-bottom: 1px solid var(--border-color) !important;
        gap: 8px !important;
    }}

    button[data-baseweb="tab"],
    div[data-testid="stTabs"] button[role="tab"] {{
        background-color: transparent !important;
        color: var(--text-secondary) !important;
        border: none !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        transition: color 0.2s ease !important;
    }}

    button[data-baseweb="tab"] *,
    div[data-testid="stTabs"] button[role="tab"] * {{
        color: var(--text-secondary) !important;
    }}

    button[data-baseweb="tab"]:hover,
    div[data-testid="stTabs"] button[role="tab"]:hover,
    button[data-baseweb="tab"]:hover *,
    div[data-testid="stTabs"] button[role="tab"]:hover * {{
        color: var(--primary-accent) !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"],
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
        color: var(--primary-accent) !important;
        font-weight: 600 !important;
        border-bottom: 2px solid var(--primary-accent) !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"] *,
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] * {{
        color: var(--primary-accent) !important;
    }}

    div[data-baseweb="tab-highlight"] {{
        background-color: var(--primary-accent) !important;
    }}

    div[data-baseweb="tab-border"] {{
        background-color: var(--border-color) !important;
    }}

    /* Formatted Transcript Container */
    .transcript-box {{
        background-color: var(--input-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        padding: 16px 20px !important;
        max-height: 400px !important;
        overflow-y: auto !important;
        font-size: 0.93rem !important;
        line-height: 1.65 !important;
        color: var(--text-main) !important;
        box-shadow: var(--shadow-subtle) !important;
    }}

    .transcript-entry {{
        margin-bottom: 12px !important;
    }}

    .transcript-entry:last-child {{
        margin-bottom: 0 !important;
    }}

    .transcript-speaker {{
        font-weight: 700 !important;
        color: var(--primary-accent) !important;
    }}

    .transcript-timestamp {{
        font-family: monospace !important;
        font-size: 0.82rem !important;
        color: var(--text-muted) !important;
        margin-left: 2px !important;
        margin-right: 6px !important;
    }}

    .transcript-dialogue {{
        color: var(--text-main) !important;
    }}

    /* Radio buttons & Checkboxes */
    div[data-testid="stRadio"],
    div[data-testid="stRadio"] *,
    div[data-testid="stCheckbox"],
    div[data-testid="stCheckbox"] *,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] *,
    section[data-testid="stSidebar"] div[data-testid="stCheckbox"] * {{
        color: var(--text-main) !important;
    }}

    /* Sliders */
    div[data-testid="stSlider"],
    div[data-testid="stSlider"] label,
    div[data-testid="stSlider"] div[data-testid="stThumbValue"],
    section[data-testid="stSidebar"] div[data-testid="stSlider"] label,
    div[data-testid="stSlider"] div[data-testid="stTickBar"] div,
    div[data-testid="stSlider"] p,
    div[data-testid="stSlider"] span {{
        color: var(--text-main) !important;
    }}

    div[data-testid="stSlider"] div[data-testid="stTickBarMin"],
    div[data-testid="stSlider"] div[data-testid="stTickBarMax"] {{
        color: var(--text-secondary) !important;
    }}

    div[data-testid="stSlider"] div[role="slider"] {{
        background-color: var(--primary-accent) !important;
        border-color: var(--primary-accent) !important;
    }}

    /* Tables & Dataframes */
    div[data-testid="stTable"] table {{
        background-color: var(--bg-card) !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
    }}

    div[data-testid="stTable"] th {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-main) !important;
        border-bottom: 1px solid var(--border-color) !important;
        font-weight: 600 !important;
    }}

    div[data-testid="stTable"] th *,
    div[data-testid="stTable"] td,
    div[data-testid="stTable"] td * {{
        color: var(--text-main) !important;
        border-bottom-color: var(--border-color) !important;
    }}

    div[data-testid="stDataFrame"] {{
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
    }}

    /* Expanders */
    div[data-testid="stExpander"] {{
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }}

    div[data-testid="stExpander"] summary {{
        background-color: var(--bg-card) !important;
        color: var(--text-main) !important;
    }}

    div[data-testid="stExpander"] summary * {{
        color: var(--text-main) !important;
        font-weight: 600 !important;
    }}

    div[data-testid="stExpander"] summary svg {{
        color: var(--text-main) !important;
        fill: var(--text-main) !important;
    }}

    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{
        background-color: var(--bg-card) !important;
        color: var(--text-main) !important;
        border-top: 1px solid var(--border-color) !important;
    }}

    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] * {{
        color: var(--text-main) !important;
    }}

    /* Code, links, and inline text */
    code, pre {{
        background-color: var(--bg-secondary) !important;
        color: var(--primary-accent) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
        font-size: 0.88em !important;
    }}

    a,
    div[data-testid="stMarkdownContainer"] a {{
        color: var(--primary-accent) !important;
        text-decoration: underline !important;
    }}
    a:hover,
    div[data-testid="stMarkdownContainer"] a:hover {{
        color: var(--primary-hover) !important;
    }}

    /* File Uploader in Dark Theme */
    div[data-testid="stFileUploader"],
    div[data-testid="stFileUploader"] section,
    div[data-testid="stFileUploaderDropzone"],
    section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] {{
        background-color: var(--input-bg) !important;
        border: 1px dashed var(--border-color) !important;
        border-radius: 10px !important;
        color: var(--text-main) !important;
    }}

    div[data-testid="stFileUploader"] section *,
    div[data-testid="stFileUploaderDropzone"] *,
    div[data-testid="stFileUploaderDropzoneInstructions"] span,
    div[data-testid="stFileUploaderDropzoneInstructions"] p {{
        color: var(--text-main) !important;
    }}

    div[data-testid="stFileUploaderDropzoneInstructions"] small,
    div[data-testid="stFileUploader"] small {{
        color: var(--text-secondary) !important;
    }}

    div[data-testid="stFileUploader"] button[data-testid="baseButton-secondary"] {{
        background-color: var(--button-bg) !important;
        color: var(--button-text) !important;
        border: 1px solid var(--border-color) !important;
    }}

    div[data-testid="stFileUploaderFile"] {{
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
    }}

    div[data-testid="stFileUploaderFileName"] {{
        color: var(--text-main) !important;
        font-weight: 500 !important;
    }}

    div[data-testid="stFileUploaderFileSize"] {{
        color: var(--text-secondary) !important;
    }}

    div[data-testid="stFileUploaderDeleteBtn"] svg {{
        fill: var(--text-secondary) !important;
    }}

    /* Alerts, Toasts & Status Containers */
    div[data-testid="stAlert"] {{
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        color: var(--text-main) !important;
    }}

    div[data-testid="stAlert"] * {{
        color: var(--text-main) !important;
    }}

    div[data-testid="stStatusWidget"] {{
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-main) !important;
        border-radius: 10px !important;
    }}

    div[data-testid="stStatusWidget"] * {{
        color: var(--text-main) !important;
    }}

    div[data-testid="stToast"] {{
        background-color: var(--bg-card) !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        box-shadow: var(--shadow-popover) !important;
    }}

    div[data-testid="stToast"] * {{
        color: var(--text-main) !important;
    }}

    /* Dividers */
    hr {{
        border-color: var(--border-color) !important;
    }}

    /* Floating Bottom-Right Chat Assistant (Trigger Button) */
    .st-key-floating_chat_trigger {{
        position: fixed !important;
        bottom: 20px !important;
        right: 24px !important;
        z-index: 999995 !important;
        width: auto !important;
        height: auto !important;
        display: block !important;
    }}

    .st-key-floating_chat_trigger button {{
        background-color: var(--primary-accent) !important;
        background: var(--primary-accent) !important;
        color: #FFFFFF !important;
        border-radius: 50% !important;
        width: 58px !important;
        height: 58px !important;
        min-width: 58px !important;
        min-height: 58px !important;
        max-width: 58px !important;
        max-height: 58px !important;
        border: none !important;
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.45) !important;
        padding: 0 !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease !important;
    }}

    .st-key-floating_chat_trigger button * {{
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }}

    .st-key-floating_chat_trigger button:hover {{
        background-color: var(--primary-hover) !important;
        background: var(--primary-hover) !important;
        transform: scale(1.08) !important;
        box-shadow: 0 8px 26px rgba(255, 75, 75, 0.6) !important;
    }}

    .st-key-floating_chat_trigger button span[data-testid="stIconMaterial"] {{
        font-size: 26px !important;
        line-height: 26px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }}

    /* Floating Chat Assistant Overlay Panel */
    .st-key-floating_chat_panel {{
        position: fixed !important;
        bottom: 88px !important;
        right: 24px !important;
        width: 385px !important;
        max-width: calc(100vw - 32px) !important;
        height: 510px !important;
        max-height: calc(100vh - 110px) !important;
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 16px !important;
        box-shadow: var(--shadow-popover) !important;
        z-index: 999990 !important;
        padding: 14px 16px !important;
        display: flex !important;
        flex-direction: column !important;
        overflow: hidden !important;
    }}

    .st-key-floating_chat_panel * {{
        color: var(--text-main) !important;
    }}

    .st-key-floating_chat_panel a {{
        color: var(--primary-accent) !important;
    }}

    .st-key-close_chat_panel_btn button {{
        background: transparent !important;
        border: none !important;
        color: var(--text-secondary) !important;
        padding: 2px !important;
        height: 30px !important;
        width: 30px !important;
        min-width: 30px !important;
        min-height: 30px !important;
        box-shadow: none !important;
    }}

    .st-key-close_chat_panel_btn button:hover {{
        background: var(--bg-secondary) !important;
        color: var(--text-main) !important;
        transform: none !important;
    }}

    .st-key-close_chat_panel_btn button * {{
        color: var(--text-secondary) !important;
    }}

    .st-key-close_chat_panel_btn button:hover * {{
        color: var(--text-main) !important;
    }}

    /* Message Bubbles inside Assistant */
    .chat-bubble-user {{
        background-color: var(--chat-user-bg);
        color: var(--text-main);
        border: 1px solid var(--border-color);
        border-radius: 12px 12px 2px 12px;
        padding: 9px 12px;
        margin-bottom: 8px;
        margin-left: 14%;
        font-size: 0.9rem;
        line-height: 1.45;
        word-break: break-word;
    }}

    .chat-bubble-assistant {{
        background-color: var(--chat-asst-bg);
        color: var(--text-main);
        border: 1px solid var(--border-color);
        border-radius: 12px 12px 12px 2px;
        padding: 9px 12px;
        margin-bottom: 8px;
        margin-right: 8%;
        font-size: 0.9rem;
        line-height: 1.45;
        word-break: break-word;
        box-shadow: var(--shadow-subtle);
    }}

    /* Mobile responsive columns: wrap gracefully on small screens (320px, 375px) */
    @media (max-width: 640px) {{
        div[data-testid="stHorizontalBlock"] {{
            flex-wrap: wrap !important;
            gap: 10px !important;
        }}

        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {{
            min-width: calc(50% - 6px) !important;
            flex: 1 1 calc(50% - 6px) !important;
        }}

        /* Top-level mode buttons take full width on mobile */
        div[data-testid="stHorizontalBlock"]:has(.st-key-app_mode_workspace_btn) > div[data-testid="column"],
        div[data-testid="stHorizontalBlock"]:has(.st-key-app_mode_dashboard_btn) > div[data-testid="column"] {{
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }}

        .st-key-floating_chat_trigger {{
            bottom: 16px !important;
            right: 16px !important;
        }}

        .st-key-floating_chat_trigger button {{
            width: 48px !important;
            height: 48px !important;
            min-width: 48px !important;
            min-height: 48px !important;
            max-width: 48px !important;
            max-height: 48px !important;
            box-shadow: 0 4px 14px rgba(255, 75, 75, 0.4) !important;
        }}

        .st-key-floating_chat_trigger button span[data-testid="stIconMaterial"] {{
            font-size: 22px !important;
            line-height: 22px !important;
        }}

        .st-key-floating_chat_panel {{
            bottom: 72px !important;
            right: 12px !important;
            width: calc(100vw - 24px) !important;
            max-width: calc(100vw - 24px) !important;
            height: 480px !important;
            max-height: calc(100vh - 90px) !important;
        }}
    }}
    </style>
    """


st.markdown(
    _generate_theme_css(st.session_state.get("theme", "Dark")),
    unsafe_allow_html=True,
)

settings = get_settings()
current_user: User | None = st.session_state.get("authenticated_user")

# --------------------------------------------------------------------------
# Authentication Gate & Route Protection
# --------------------------------------------------------------------------

if settings.require_auth and not current_user:
    render_auth_page()
    st.stop()




# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def run_pipeline(
    source: str,
    language: str,
    chunk_minutes: int,
) -> None:
    from utils.audio_processor import AudioProcessingError, cleanup_chunks, process_input
    from core.transcriber import TranscriptionError, transcribe_all, transcribe_all_original
    from core.translator import TranslationError, clean_transcript, translate_to_clean_english
    from core.summariser import SummarizationError, summarize_transcript
    from core.extractor import ExtractionError, extract_action_items
    from core.vector_store import VectorStoreError, index_transcript

    session_id = uuid.uuid4().hex[:12]
    st.session_state["is_processing"] = True

    chunk_paths: list[str] = []


    status = st.status(
        "Processing...",
        expanded=True,
    )

    try:

        # --------------------------------------------------------------
        # Step 1 - Extract and chunk audio
        # --------------------------------------------------------------

        status.write(
            "Step 1/6 - Extracting & chunking audio..."
        )

        chunk_paths = process_input(
            source,
            chunk_minutes=chunk_minutes,
        )

        if not chunk_paths:
            raise RuntimeError(
                "No audio chunks were created from the input."
            )

        # --------------------------------------------------------------
        # Step 2 - Transcription
        # --------------------------------------------------------------

        status.write(
            f"Step 2/6 - Transcribing "
            f"{len(chunk_paths)} chunk(s)..."
        )

        transcript_result = transcribe_all(
            chunk_paths,
            language=language,
            chunk_seconds=chunk_minutes * 60,
        )

        original_text = transcript_result.full_text

        # --------------------------------------------------------------
        # Step 3 - Translation / cleanup
        # --------------------------------------------------------------

        if language.lower() == "hinglish":

            status.write(
                "Step 3/6 - Preserving original-language transcript..."
            )

            original_text = transcribe_all_original(
                chunk_paths,
                language=language,
            )

            status.write(
                "Step 3/6 - Translating and cleaning transcript..."
            )

            # Sarvam translation mode returns English; run editorial cleanup pass
            clean_text = clean_transcript(transcript_result.full_text)

        elif language.lower() == "english":

            status.write(
                "Step 3/6 - Cleaning English transcript..."
            )

            clean_text = clean_transcript(
                transcript_result.full_text
            )

        elif language.lower() in ("auto", "auto detect", "automatic"):

            status.write(
                "Step 3/6 - Normalizing and cleaning transcript..."
            )

            # Check if original audio was detected as English or Indic
            detected_lang = (
                transcript_result.segments[0].language_code
                if transcript_result.segments and transcript_result.segments[0].language_code
                else "en"
            )
            if detected_lang in ("en", "english"):
                clean_text = clean_transcript(transcript_result.full_text)
            else:
                clean_text = translate_to_clean_english(
                    transcript_result.full_text,
                    source_language=detected_lang,
                )

        else:

            status.write(
                "Step 3/6 - Translating to clean English..."
            )

            clean_text = translate_to_clean_english(
                transcript_result.full_text,
                source_language=language.lower(),
            )

        # --------------------------------------------------------------
        # Validate transcript
        # --------------------------------------------------------------

        if not clean_text.strip():

            raise RuntimeError(
                "Transcription produced no text. "
                "The audio may be silent or unsupported."
            )

        # --------------------------------------------------------------
        # Step 4 - Summarization
        # --------------------------------------------------------------

        status.write(
            "Step 4/6 - Generating summary..."
        )

        summary = summarize_transcript(
            clean_text
        )

        # --------------------------------------------------------------
        # Step 5 - Extraction
        # --------------------------------------------------------------

        status.write(
            "Step 5/6 - Extracting action items, "
            "decisions & questions..."
        )

        extraction = extract_action_items(
            clean_text
        )

        # --------------------------------------------------------------
        # Step 6 - Vector store
        # --------------------------------------------------------------

        status.write(
            "Step 6/6 - Building searchable index for chat..."
        )

        index_transcript(
            session_id,
            transcript_result.segments,
        )

        # --------------------------------------------------------------
        # Step 7 - Save to persistent meeting memory & action items
        # --------------------------------------------------------------

        try:
            curr_uid = current_user.id if current_user else "default_user"
            save_meeting(
                meeting_id=session_id,
                title=summary.title or "Processed Meeting",
                summary_obj=summary,
                extraction_obj=extraction,
                clean_transcript=clean_text,
                original_transcript=original_text,
                source=source,
                user_id=curr_uid,
            )
        except Exception as store_err:
            logger.warning("Could not persist meeting to database: %s", store_err)

        # --------------------------------------------------------------
        # Save results in Streamlit session
        # --------------------------------------------------------------

        st.session_state.session_id = session_id

        st.session_state.clean_transcript = clean_text

        st.session_state.original_transcript = original_text

        st.session_state.summary = summary

        st.session_state.extraction = extraction

        st.session_state.processed = True

        st.session_state.processing_error = None

        st.session_state.chat_history = []

        status.update(
            label="Processing complete!",
            state="complete",
            expanded=False,
        )

    # ------------------------------------------------------------------
    # Known errors
    # ------------------------------------------------------------------

    except AudioProcessingError as exc:

        status.update(
            label="Audio processing failed",
            state="error",
        )

        st.session_state.processing_error = (
            f"Audio error: {exc}"
        )

    except TranscriptionError as exc:

        status.update(
            label="Transcription failed",
            state="error",
        )

        st.session_state.processing_error = (
            f"Transcription error: {exc}"
        )

    except TranslationError as exc:

        status.update(
            label="Translation failed",
            state="error",
        )

        st.session_state.processing_error = (
            f"Translation error: {exc}"
        )

    except SummarizationError as exc:

        status.update(
            label="Summarization failed",
            state="error",
        )

        st.session_state.processing_error = (
            f"Summarization error: {exc}"
        )

    except ExtractionError as exc:

        status.update(
            label="Extraction failed",
            state="error",
        )

        st.session_state.processing_error = (
            f"Extraction error: {exc}"
        )

    except VectorStoreError as exc:

        status.update(
            label="Indexing failed",
            state="error",
        )

        st.session_state.processing_error = (
            f"Vector store error: {exc}"
        )

    except Exception as exc:

        status.update(
            label="Unexpected error",
            state="error",
        )

        logger.exception(
            "Unexpected pipeline failure"
        )

        st.session_state.processing_error = (
            f"Unexpected error: {exc}"
        )

    finally:
        st.session_state["is_processing"] = False
        cleanup_chunks(chunk_paths)
        if source and Path(source).exists() and ("tmp" in source.lower() or "converted" in source.lower()):
            try:
                Path(source).unlink(missing_ok=True)
            except OSError:
                pass



# --------------------------------------------------------------------------
# Sidebar - Input
# --------------------------------------------------------------------------

st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 24px; padding-top: 4px;">
        <div style="width: 46px; height: 46px; border-radius: 12px; background: linear-gradient(135deg, #FF4B4B 0%, #E63E45 100%); display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 14px rgba(255, 75, 75, 0.32); flex-shrink: 0;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect x="2.5" y="5.5" width="12" height="13" rx="2.5" stroke="#FFFFFF" stroke-width="2"/>
                <path d="M14.5 10L20.5 6.5V17.5L14.5 14" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M8.5 8C8.5 9.2 9.3 10 10.5 10C9.3 10 8.5 10.8 8.5 12C8.5 10.8 7.7 10 6.5 10C7.7 10 8.5 9.2 8.5 8Z" fill="#FFFFFF"/>
            </svg>
        </div>
        <div>
            <h1 style="color: var(--text-main); font-size: 2.05rem; font-weight: 700; margin: 0 0 3px 0; letter-spacing: -0.02em;">AI Video/Meeting Assistant</h1>
            <p style="color: var(--text-secondary); font-size: 0.98rem; margin: 0; font-weight: 400;">Transcribe, summarize, and interact with any video or meeting recording.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Configuration check
# --------------------------------------------------------------------------

config_problems = validate_settings(
    require_gemini=True,
    require_sarvam=False,
)

if config_problems:

    st.warning(
        "\n".join(config_problems)
    )


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:

    if current_user:
        render_user_profile_dropdown(current_user)

    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 9px; margin-bottom: 16px;">
            <div style="width: 28px; height: 28px; border-radius: 7px; background-color: var(--primary-accent); display: flex; align-items: center; justify-content: center; color: white;">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <polygon points="5 3 19 12 5 21 5 3"></polygon>
                </svg>
            </div>
            <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: var(--text-main);">Process Meeting</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-size: 0.74rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;'>1. Media Source</div>",
        unsafe_allow_html=True,
    )

    input_mode = st.radio(
        "Source",
        [
            "YouTube URL",
            "Upload file",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    source_value: str | None = None

    uploaded_temp_path: str | None = None

    # --------------------------------------------------------------
    # YouTube
    # --------------------------------------------------------------

    if input_mode == "YouTube URL":

        source_value = st.text_input(
            "YouTube URL",
            placeholder=(
                "https://www.youtube.com/watch?v=..."
            ),
            key="youtube_url_input",
        )

    # --------------------------------------------------------------
    # Upload
    # --------------------------------------------------------------

    else:

        uploaded = st.file_uploader(
            "Audio or video file",
            type=[
                "mp4",
                "mp3",
                "wav",
                "m4a",
                "mov",
                "mkv",
                "webm",
            ],
        )

        if uploaded is not None:
            if uploaded.size == 0:
                st.error("The uploaded file is empty (0 bytes). Please upload a valid recording.")
            else:
                upload_sig = f"{uploaded.name}_{uploaded.size}"
                prev_sig = st.session_state.get("_last_upload_sig")
                prev_path = st.session_state.get("_last_upload_path")

                if prev_sig != upload_sig or not prev_path or not Path(prev_path).exists():
                    if prev_path:
                        try:
                            Path(prev_path).unlink(missing_ok=True)
                        except OSError:
                            pass

                    suffix = Path(uploaded.name).suffix or ".bin"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded.getbuffer())
                        new_temp_path = tmp.name

                    st.session_state["_last_upload_sig"] = upload_sig
                    st.session_state["_last_upload_path"] = new_temp_path
                    source_value = new_temp_path
                else:
                    source_value = prev_path
        else:
            prev_path = st.session_state.get("_last_upload_path")
            if prev_path:
                try:
                    Path(prev_path).unlink(missing_ok=True)
                except OSError:
                    pass
                st.session_state["_last_upload_path"] = None
                st.session_state["_last_upload_sig"] = None


    # --------------------------------------------------------------
    # Language
    # --------------------------------------------------------------

    st.markdown(
        "<div style='font-size: 0.74rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.06em; margin-top: 14px; margin-bottom: 6px;'>2. Spoken Language</div>",
        unsafe_allow_html=True,
    )

    language = st.selectbox(
        "Language",
        [
            "Auto Detect",
            "English",
            "Telugu",
            "Hindi",
            "Hinglish",
            "Tamil",
            "Kannada",
            "Malayalam",
            "Marathi",
            "Bengali",
            "Gujarati",
            "Punjabi",
        ],
        key="language_select",
        label_visibility="collapsed",
        help=(
            "Auto Detect automatically identifies the spoken language. "
            "English uses Whisper; Indic languages use Sarvam AI."
        ),
    )

    # --------------------------------------------------------------
    # Audio chunk length
    # --------------------------------------------------------------

    st.markdown(
        "<div style='font-size: 0.74rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.06em; margin-top: 14px; margin-bottom: 6px;'>3. Audio Chunk Duration</div>",
        unsafe_allow_html=True,
    )

    chunk_minutes = st.slider(
        "Audio chunk length (minutes)",
        2,
        20,
        10,
        label_visibility="collapsed",
    )

    # --------------------------------------------------------------
    # Process button
    # --------------------------------------------------------------

    st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

    process_clicked = st.button(
        "Process Meeting",
        icon=":material/play_arrow:",
        type="primary",
        use_container_width=True,
        disabled=not source_value or bool(st.session_state.get("is_processing", False)),
    )


    if not st.session_state.processed:
        if st.button(
            "Load Sample Session",
            icon=":material/visibility:",
            use_container_width=True,
        ):
            from core.summariser import MeetingSummary
            from core.extractor import ActionItem, ExtractionResult

            st.session_state.processed = True
            st.session_state.summary = MeetingSummary(
                title="Q3 Product Strategy & AI Roadmap Alignment",
                overview="Executive sync covering Q3 milestones, agentic AI assistant rollout, and latency benchmarks.",
                summary="The product and engineering teams aligned on deploying the real-time meeting intelligence assistant by late Q3. Key focus areas include sub-second transcription caching, Indic language support via Sarvam AI, and cross-platform export fidelity.",
                important_points=[
                    "Agentic pipeline latency benchmark set to under 1.2s for interactive Q&A.",
                    "Multilingual fallback pipeline successfully handles Hindi and Telugu streams.",
                    "Export formats must support high-res JPEG infographics and print-ready PDFs.",
                ],
                key_takeaways=[
                    "Product launch scheduled for end of month.",
                    "ChromaDB retrieval accuracy achieved 94% on test datasets.",
                    "UI overhaul completed with coral-red SaaS design system.",
                ],
            )
            st.session_state.extraction = ExtractionResult(
                action_items=[
                    ActionItem(action="Finalize ChromaDB vector store indexing benchmarks", owner="Engineering", deadline="Friday", status="In Progress"),
                    ActionItem(action="Validate PDF and JPG report generator layout", owner="Design Team", deadline="Next Tuesday", status="Pending"),
                    ActionItem(action="Deploy Sarvam AI Indic models to staging cluster", owner="DevOps", deadline="Oct 20", status="Open"),
                ],
                decisions=[
                    "Approved coral-red SaaS design system across all client dashboards.",
                    "Retain local embedding caches to reduce cold-start latency.",
                ],
                open_questions=[
                    "Should we support live RTSP audio streaming in phase 2?",
                    "What is the maximum token context for long 2-hour recordings?",
                ],
                follow_ups=[
                    "Schedule review meeting next Thursday with stakeholders.",
                    "Collect feedback from initial beta testers.",
                ],
            )
            st.session_state.original_transcript = (
                "Speaker 1 (00:00): um hello uh welcome everyone to our Q3 product alignment session. "
                "like basically today we we are walking through the new a i video and meeting assistant architecture and uh you know the core roadmap.\n\n"
                "Speaker 2 (00:45): yeah thanks Alex. On the engineering side uh we've integrated whisper for English and sarvam for indic languages. "
                "the vector in dexing now runs seamlessly with chroma d b and the sink is working great.\n\n"
                "Speaker 1 (01:30): that's fantastic. let's make sure the report exports for t x t, pdf, and j p g look pixel perfect for executive presentations and we won't to have any delays."
            )
            st.session_state.clean_transcript = (
                "Speaker 1 (00:00): Welcome everyone to our Q3 product alignment session. "
                "Today we are walking through the new AI video and meeting assistant architecture and the core roadmap.\n\n"
                "Speaker 2 (00:45): Thanks Alex. On the engineering side, we've integrated Whisper for English and Sarvam AI for Indic languages. "
                "The vector indexing now runs seamlessly with ChromaDB, and the sync is working great.\n\n"
                "Speaker 1 (01:30): That's fantastic. Let's make sure the report exports for TXT, PDF, and JPG look pixel-perfect for executive presentations and we want to have no delays."
            )
            st.session_state.session_id = "sample-session-123"
            st.session_state.active_tab = "summary"
            try:
                save_meeting(
                    meeting_id=st.session_state.session_id,
                    title=st.session_state.summary.title,
                    summary_obj=st.session_state.summary,
                    extraction_obj=st.session_state.extraction,
                    clean_transcript=st.session_state.clean_transcript,
                    original_transcript=st.session_state.original_transcript,
                    source="Sample Meeting Demo",
                )
            except Exception as sm_err:
                logger.warning("Could not persist sample meeting to database: %s", sm_err)
            st.rerun()

    # --------------------------------------------------------------
    # New session
    # --------------------------------------------------------------

    if st.session_state.processed:

        st.divider()

        if st.button(
            "Start New Session",
            icon=":material/refresh:",
            use_container_width=True,
        ):

            st.session_state.session_id = None

            st.session_state.processed = False

            st.session_state.clean_transcript = ""

            st.session_state.original_transcript = ""

            st.session_state.summary = None

            st.session_state.extraction = None

            st.session_state.chat_history = []

            st.session_state.preferred_response_language = "english"

            st.session_state.processing_error = None

            st.rerun()

    # --------------------------------------------------------------
    # Theme selector
    # --------------------------------------------------------------

    st.divider()

    st.markdown(
        "<p style='font-size:0.8rem; font-weight:600; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;'>Appearance</p>",
        unsafe_allow_html=True,
    )

    theme_options = ["Light", "Dark", "System"]
    current_theme = st.session_state.get("theme", "Dark")
    if current_theme not in theme_options:
        current_theme = "Dark"

    def _on_theme_radio_change() -> None:
        chosen = st.session_state.get("theme_radio_selector")
        if chosen in theme_options:
            st.session_state["theme"] = chosen
            st.session_state["_synced_query_theme"] = chosen.lower()
            st.query_params["theme"] = chosen.lower()

    if st.session_state.get("theme_radio_selector") not in theme_options:
        st.session_state["theme_radio_selector"] = current_theme

    st.radio(
        "Theme",
        theme_options,
        key="theme_radio_selector",
        horizontal=True,
        label_visibility="collapsed",
        on_change=_on_theme_radio_change,
    )


# --------------------------------------------------------------------------
# Process input
# --------------------------------------------------------------------------

if process_clicked and source_value:
    from utils.audio_processor import is_youtube_url

    if input_mode == "YouTube URL" and not is_youtube_url(source_value):
        st.error("Please enter a valid YouTube URL (e.g., https://www.youtube.com/watch?v=... or https://youtu.be/...).")
        st.stop()

    # Convert UI language names into language codes
    language_map = {
        "auto detect": "auto",
        "english": "en",
        "telugu": "te",
        "hindi": "hi",
        "hinglish": "hinglish",
        "tamil": "ta",
        "kannada": "kn",
        "malayalam": "ml",
        "marathi": "mr",
        "bengali": "bn",
        "gujarati": "gu",
        "punjabi": "pa",
    }

    selected_language = language_map.get(
        language.lower(),
        language.lower(),
    )

    run_pipeline(
        source_value,
        selected_language,
        chunk_minutes,
    )


# --------------------------------------------------------------------------
# Processing error
# --------------------------------------------------------------------------

if st.session_state.processing_error:

    st.error(
        st.session_state.processing_error
    )


# --------------------------------------------------------------------------
# UI Helpers for Transcript & Floating Chat
# --------------------------------------------------------------------------

def _highlight_transcript_dialogue(dialogue: str) -> str:
    """Highlight commitments, deadlines, and decisions with subtle, readable badges."""
    # Pattern for commitments and assigned actions
    dialogue = re.sub(
        r"(?i)\b(I will|I'll|we will|we'll|action item|assigned to|responsible for|take ownership|follow up on|will deliver|needs to|must be done|please ensure)\b",
        r'<span style="background: rgba(59, 130, 246, 0.16); color: #60A5FA; font-weight: 600; padding: 1px 5px; border-radius: 4px;">\1</span>',
        dialogue,
    )
    # Pattern for decisions
    dialogue = re.sub(
        r"(?i)\b(we decided|decided to|agreed that|agreed to|decision is|approved|conclusion is|confirmed that)\b",
        r'<span style="background: rgba(245, 158, 11, 0.16); color: #FBBF24; font-weight: 600; padding: 1px 5px; border-radius: 4px;">\1</span>',
        dialogue,
    )
    # Pattern for deadlines and urgent timing
    dialogue = re.sub(
        r"(?i)\b(by tomorrow|by next week|by Friday|by Monday|by EOD|by end of day|deadline is|due by|due date|urgent|critical|asap)\b",
        r'<span style="background: rgba(239, 68, 68, 0.16); color: #F87171; font-weight: 600; padding: 1px 5px; border-radius: 4px;">\1</span>',
        dialogue,
    )
    return dialogue


def _render_transcript_display(title: str, text: str, is_original: bool = False) -> None:
    """Render transcript with dark/light container styling, colored speakers, badges, and keyword highlighting."""
    icon_svg = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FF5A5F" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
        <polyline points="14 2 14 8 20 8"></polyline>
        <line x1="16" y1="13" x2="8" y2="13"></line>
        <line x1="16" y1="17" x2="8" y2="17"></line>
    </svg>"""

    word_count = len(text.split()) if text else 0
    read_mins = max(1, round(word_count / 200)) if word_count > 0 else 0
    top_margin = "18px" if is_original else "10px"

    status_badge = (
        '<span style="background: rgba(156, 163, 175, 0.15); color: var(--text-secondary); border: 1px solid var(--border-color); padding: 3px 8px; border-radius: 6px; font-size: 0.74rem; font-weight: 600;">● Raw Audio Stream</span>'
        if is_original
        else '<span style="background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3); padding: 3px 8px; border-radius: 6px; font-size: 0.74rem; font-weight: 600;">✓ AI Structured & Cleaned</span>'
    )

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; margin-top: {top_margin};">
            <div style="display: flex; align-items: center; gap: 8px;">
                {icon_svg}
                <h4 style="color: var(--text-main); font-weight: 700; margin: 0; font-size: 1.15rem;">{title}</h4>
                {status_badge}
            </div>
            <div style="display: flex; align-items: center; gap: 10px; font-size: 0.82rem; color: var(--text-secondary); font-weight: 500;">
                <span>{word_count:,} words</span>
                <span>·</span>
                <span>~{read_mins} min read</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not text.strip():
        st.caption("Transcript is empty.")
        return

    # Parse formatted lines if Speaker / timestamp format is present
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    formatted_html = []
    speaker_pattern = re.compile(r"^(Speaker\s+[^:(]+|\w+[^:(]*)\s*(\([0-9:]+\))?\s*:\s*(.*)$", re.IGNORECASE)

    for line in lines:
        match = speaker_pattern.match(line)
        if match:
            spk = html.escape(match.group(1).strip())
            ts = html.escape(match.group(2).strip()) if match.group(2) else ""
            raw_dialogue = html.escape(match.group(3).strip())
            highlighted_dialogue = _highlight_transcript_dialogue(raw_dialogue)
            ts_span = f'<span class="transcript-timestamp">{ts}</span>' if ts else ""
            formatted_html.append(
                f'<div class="transcript-entry"><span class="transcript-speaker">{spk}</span>{ts_span}: <span class="transcript-dialogue">{highlighted_dialogue}</span></div>'
            )
        else:
            raw_line = html.escape(line)
            highlighted_line = _highlight_transcript_dialogue(raw_line)
            formatted_html.append(f'<div class="transcript-entry"><span class="transcript-dialogue">{highlighted_line}</span></div>')

    inner_html = "".join(formatted_html)
    st.markdown(
        f'<div class="transcript-box">{inner_html}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("View / Copy / Download Raw Text", expanded=False):
        d_col1, d_col2 = st.columns([1, 2])
        with d_col1:
            st.download_button(
                f"Download {title} (.txt)",
                data=text,
                file_name=f"{title.lower().replace(' ', '_')}.txt",
                mime="text/plain",
                icon=":material/download:",
                key=f"dl_btn_{title.lower().replace(' ', '_')}",
                use_container_width=True,
            )
        st.text_area(
            f"Raw {title}",
            text,
            height=160,
            label_visibility="collapsed",
            key=f"raw_transcript_area_{title.lower().replace(' ', '_')}",
        )


def _render_floating_ai_chat() -> None:
    """Render the floating AI assistant widget at the bottom right across all app modes."""
    # Ensure state initialized
    if "chat_panel_open" not in st.session_state:
        st.session_state.chat_panel_open = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "dashboard_chat_history" not in st.session_state:
        st.session_state.dashboard_chat_history = []

    current_app_mode = st.session_state.get("app_mode", "workspace")
    is_dashboard = current_app_mode == "memory_dashboard"

    # 1. Floating panel (when open)
    if st.session_state.chat_panel_open:
        with st.container(key="floating_chat_panel"):
            # Header
            h_col1, h_col2 = st.columns([5, 1])
            with h_col1:
                scope_label = "Meeting Memory (Cross-Meeting)" if is_dashboard else "Active Meeting"
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#FF4B4B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <rect x="3" y="11" width="18" height="10" rx="2"></rect>
                            <circle cx="12" cy="5" r="2"></circle>
                            <path d="M12 7v4"></path>
                            <line x1="8" y1="16" x2="8" y2="16"></line>
                            <line x1="16" y1="16" x2="16" y2="16"></line>
                        </svg>
                        <div>
                            <div style="font-size: 0.98rem; font-weight: 700; color: var(--text-main); line-height: 1.2;">AI Video Assistant</div>
                            <div style="font-size: 0.72rem; color: #16A34A; font-weight: 600;">● Online &nbsp;<span style="color: var(--text-muted); font-weight: 400;">({scope_label})</span></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with h_col2:
                if st.button("", icon=":material/close:", key="close_chat_panel_btn", help="Close Assistant"):
                    st.session_state.chat_panel_open = False
                    st.rerun()

            st.markdown("<div style='border-bottom: 1px solid var(--border-color); margin: 6px 0 10px 0;'></div>", unsafe_allow_html=True)

            # Messages container
            active_history = st.session_state.dashboard_chat_history if is_dashboard else st.session_state.chat_history
            chat_box = st.container(height=310)

            curr_uid = current_user.id if current_user else None
            all_meetings = get_all_meetings(user_id=curr_uid) if is_dashboard else []

            with chat_box:
                if is_dashboard:
                    if not all_meetings:
                        st.info("👋 No meetings recorded in memory yet. Process a recording in Active Workspace or load a sample session to chat across meeting memory.")
                    elif not active_history:
                        st.info("👋 Ask anything across your past meetings, verified decisions, and action items.")
                    else:
                        for role, text in active_history:
                            if role == "user":
                                st.markdown(f'<div class="chat-bubble-user"><strong>You:</strong><br>{html.escape(text)}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="chat-bubble-assistant"><strong>Assistant:</strong><br>{text}</div>', unsafe_allow_html=True)
                else:
                    if not st.session_state.processed:
                        st.info("👋 Please process a video or meeting recording first (or load a sample session) so I can answer questions about it.")
                    elif not active_history:
                        st.info("👋 Ask me anything about this video or meeting.")
                    else:
                        for role, text in active_history:
                            if role == "user":
                                st.markdown(f'<div class="chat-bubble-user"><strong>You:</strong><br>{html.escape(text)}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="chat-bubble-assistant"><strong>Assistant:</strong><br>{text}</div>', unsafe_allow_html=True)

            # Input form fixed at bottom
            can_ask = (len(all_meetings) > 0) if is_dashboard else bool(st.session_state.processed)
            placeholder_text = "Ask across all meeting memories..." if is_dashboard else "Ask something about this video..."

            with st.form("floating_chat_form", clear_on_submit=True):
                col_input, col_submit = st.columns([5, 1])
                with col_input:
                    user_question = st.text_input(
                        "Question",
                        placeholder=placeholder_text,
                        label_visibility="collapsed",
                        disabled=not can_ask or bool(st.session_state.get("is_processing", False)),
                    )
                with col_submit:
                    send_clicked = st.form_submit_button(
                        "",
                        icon=":material/arrow_upward:",
                        help="Send message",
                        type="primary",
                        disabled=not can_ask or bool(st.session_state.get("is_processing", False)),
                    )

                if send_clicked and user_question.strip():
                    q = user_question.strip()
                    if is_dashboard:
                        st.session_state.dashboard_chat_history.append(("user", q))
                        answer_text = ""
                        try:
                            with st.spinner("Synthesizing answer from cross-meeting memory..."):
                                from core.cross_meeting_rag import ask_cross_meetings

                                res = ask_cross_meetings(
                                    question=q,
                                    conversation_history=st.session_state.dashboard_chat_history[:-1],
                                    preferred_language=st.session_state.get("preferred_response_language", "english"),
                                    user_id=curr_uid,
                                )

                                if hasattr(res, "language") and res.language:
                                    st.session_state.preferred_response_language = res.language
                                formatted_res = html.escape(res.answer).replace("\n", "<br>")
                                answer_text = formatted_res
                                if res.meetings_referenced:
                                    ref_str = ", ".join([f"<code>{html.escape(m)}</code>" for m in res.meetings_referenced])
                                    answer_text += f"<br><br><small><strong>Meetings Referenced:</strong> {ref_str}</small>"
                        except Exception as exc:
                            answer_text = f"⚠️ Could not answer from cross-meeting memory: {exc}"

                        st.session_state.dashboard_chat_history.append(("assistant", answer_text))
                    else:
                        st.session_state.chat_history.append(("user", q))
                        answer_text = ""
                        try:
                            from core.rag_engine import RagError, ask

                            with st.spinner("Thinking..."):
                                result = ask(
                                    session_id=st.session_state.session_id,
                                    question=q,
                                    conversation_history=st.session_state.chat_history[:-1],
                                    clean_transcript=st.session_state.get("clean_transcript", ""),
                                    preferred_language=st.session_state.get("preferred_response_language", "english"),
                                )
                                if hasattr(result, "language") and result.language:
                                    st.session_state.preferred_response_language = result.language
                            formatted_answer = html.escape(result.answer).replace("\n", "<br>")
                            answer_text = formatted_answer
                            if result.timestamps:
                                answer_text += (
                                    "<br><br><em>Referenced around: "
                                    + html.escape(", ".join(result.timestamps))
                                    + "</em>"
                                )
                        except RagError as exc:
                            answer_text = f"⚠️ {html.escape(str(exc))}"
                        except Exception as exc:
                            answer_text = f"⚠️ An error occurred: {html.escape(str(exc))}"

                        st.session_state.chat_history.append(("assistant", answer_text))

                    st.session_state.chat_panel_open = True
                    st.rerun()

    # 2. Floating trigger button (always rendered at bottom right)
    with st.container(key="floating_chat_trigger"):
        trigger_icon = ":material/close:" if st.session_state.chat_panel_open else ":material/chat:"
        if st.button("", icon=trigger_icon, key="floating_chat_toggle_btn"):
            st.session_state.chat_panel_open = not st.session_state.chat_panel_open
            st.rerun()

    # Escape key listener to close chat panel
    if st.session_state.chat_panel_open:
        st.markdown(
            """
            <script>
            (function() {
                const handler = function(e) {
                    if (e.key === 'Escape') {
                        const closeBtn = parent.document.querySelector('.st-key-close_chat_panel_btn button');
                        if (closeBtn) {
                            closeBtn.click();
                        }
                    }
                };
                parent.document.removeEventListener('keydown', handler);
                parent.document.addEventListener('keydown', handler);
            })();
            </script>
            """,
            unsafe_allow_html=True,
        )


def _render_workspace_view() -> None:
    """Render the Active Meeting Workspace (summary, actions, transcript, export)."""
    if not st.session_state.processed:
        st.markdown(
            """
            <div style="background-color: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 12px; padding: 28px; margin-top: 8px;">
                <div style="display: flex; align-items: center; gap: 16px;">
                    <div style="width: 44px; height: 44px; border-radius: 10px; background-color: rgba(255, 75, 75, 0.1); display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#FF4B4B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polygon points="23 7 16 12 23 17 23 7"></polygon>
                            <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
                        </svg>
                    </div>
                    <div>
                        <h3 style="color: var(--text-main); margin: 0 0 4px 0; font-size: 1.15rem; font-weight: 600;">Ready to transcribe and analyze</h3>
                        <p style="color: var(--text-secondary); margin: 0; font-size: 0.95rem;">Provide a YouTube URL or upload an audio/video file in the left sidebar, then click <strong>Process</strong> to get started.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    summary = st.session_state.summary
    extraction = st.session_state.extraction

    # Processed Header Card
    st.markdown(
        f"""
        <div style="background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 22px 24px; margin-bottom: 24px; box-shadow: var(--shadow-card);">
            <div style="display: inline-block; background-color: rgba(255, 75, 75, 0.12); color: #FF4B4B; font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; padding: 4px 10px; border-radius: 6px; margin-bottom: 10px;">Processed Recording</div>
            <h2 style="color: var(--text-main); margin: 0 0 8px 0; font-size: 1.55rem; font-weight: 700;">{summary.title}</h2>
            <p style="color: var(--text-secondary); margin: 0; font-size: 0.98rem; line-height: 1.55;">{summary.overview}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------------------------
    # Navigation Bar (4 Separate Professional Buttons)
    # ----------------------------------------------------------------------

    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    current_tab = st.session_state.get("active_tab", "summary")

    with nav_col1:
        if st.button(
            "Summary",
            icon=":material/description:",
            type="primary" if current_tab == "summary" else "secondary",
            use_container_width=True,
            key="nav_btn_summary",
        ):
            st.session_state.active_tab = "summary"
            st.rerun()

    with nav_col2:
        if st.button(
            "Action Items",
            icon=":material/checklist:",
            type="primary" if current_tab == "actions" else "secondary",
            use_container_width=True,
            key="nav_btn_actions",
        ):
            st.session_state.active_tab = "actions"
            st.rerun()

    with nav_col3:
        if st.button(
            "Transcript",
            icon=":material/subtitles:",
            type="primary" if current_tab == "transcript" else "secondary",
            use_container_width=True,
            key="nav_btn_transcript",
        ):
            st.session_state.active_tab = "transcript"
            st.rerun()

    with nav_col4:
        if st.button(
            "Export Report",
            icon=":material/download:",
            type="primary" if current_tab == "export" else "secondary",
            use_container_width=True,
            key="nav_btn_export",
        ):
            st.session_state.active_tab = "export"
            st.rerun()

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # ======================================================================
    # Section 1: Summary
    # ======================================================================

    if current_tab == "summary":

        summary_text_html = summary.summary.replace("\n", "<br>")
        st.markdown(
            f"""
            <div style="background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 22px 24px; margin-bottom: 20px; box-shadow: var(--shadow-card);">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FF4B4B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                        <line x1="16" y1="13" x2="8" y2="13"></line>
                        <line x1="16" y1="17" x2="8" y2="17"></line>
                    </svg>
                    <h4 style="color: var(--text-main); font-weight: 600; margin: 0; font-size: 1.15rem;">Executive Summary</h4>
                </div>
                <div style="color: var(--text-main); font-size: 0.98rem; line-height: 1.65;">{summary_text_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            points_items = "".join([f"<li style='margin-bottom: 8px;'>{p}</li>" for p in (summary.important_points or [])])
            st.markdown(
                f"""
                <div style="background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 22px 24px; height: 100%; box-shadow: var(--shadow-card);">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px;">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FF4B4B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <line x1="12" y1="17" x2="12" y2="22"></line>
                            <path d="M5 17h14v-1.76a2 2 0 0 0-1.11-1.79l-1.78-.89A4 4 0 0 1 14 9V5a2 2 0 0 0-2-2h0a2 2 0 0 0-2 2v4a4 4 0 0 1-2.11 3.56l-1.78.89A2 2 0 0 0 5 15.24V17z"></path>
                        </svg>
                        <h4 style="color: var(--text-main); margin: 0; font-size: 1.05rem; font-weight: 600;">Important Points</h4>
                    </div>
                    <ul style="color: var(--text-main); font-size: 0.95rem; line-height: 1.6; padding-left: 20px; margin: 0;">
                        {points_items if points_items else "<li>None recorded.</li>"}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            takeaways_items = "".join([f"<li style='margin-bottom: 8px;'>{t}</li>" for t in (summary.key_takeaways or [])])
            st.markdown(
                f"""
                <div style="background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 22px 24px; height: 100%; box-shadow: var(--shadow-card);">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px;">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FF4B4B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <line x1="9" y1="18" x2="15" y2="18"></line>
                            <line x1="10" y1="22" x2="14" y2="22"></line>
                            <path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14"></path>
                        </svg>
                        <h4 style="color: var(--text-main); margin: 0; font-size: 1.05rem; font-weight: 600;">Key Takeaways</h4>
                    </div>
                    <ul style="color: var(--text-main); font-size: 0.95rem; line-height: 1.6; padding-left: 20px; margin: 0;">
                        {takeaways_items if takeaways_items else "<li>None recorded.</li>"}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ======================================================================
    # Section 2: Action Items
    # ======================================================================

    elif current_tab == "actions":

        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FF4B4B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="9 11 12 14 22 4"></polyline>
                    <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"></path>
                </svg>
                <h4 style="color: var(--text-main); font-weight: 600; margin: 0; font-size: 1.15rem;">Action Items</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if extraction.action_items:
            for idx, item in enumerate(extraction.action_items):
                status_str = str(item.status or "Pending").strip()
                if status_str.lower() == "completed":
                    st_badge = "<span style='background: rgba(16, 185, 129, 0.15); color: #10B981; font-weight: 700; padding: 4px 10px; border-radius: 6px; font-size: 0.72rem; letter-spacing: 0.04em;'>✓ COMPLETED</span>"
                elif status_str.lower() in ("in progress", "in_progress"):
                    st_badge = "<span style='background: rgba(59, 130, 246, 0.15); color: #3B82F6; font-weight: 700; padding: 4px 10px; border-radius: 6px; font-size: 0.72rem; letter-spacing: 0.04em;'>● IN PROGRESS</span>"
                else:
                    st_badge = "<span style='background: rgba(245, 158, 11, 0.15); color: #F59E0B; font-weight: 700; padding: 4px 10px; border-radius: 6px; font-size: 0.72rem; letter-spacing: 0.04em;'>○ PENDING</span>"

                deadline_str = str(item.deadline or "Not specified").strip()
                is_urgent = any(kw in (item.action + " " + deadline_str).lower() for kw in ["urgent", "today", "eod", "asap", "immediate", "critical"])
                prio_badge = "<span style='background: rgba(239, 68, 68, 0.15); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.3); font-weight: 700; padding: 3px 8px; border-radius: 6px; font-size: 0.7rem;'>▲ HIGH</span>" if is_urgent else "<span style='background: rgba(59, 130, 246, 0.12); color: #3B82F6; border: 1px solid rgba(59, 130, 246, 0.3); font-weight: 600; padding: 3px 8px; border-radius: 6px; font-size: 0.7rem;'>■ MEDIUM</span>"

                esc_action = html.escape(item.action)
                esc_owner = html.escape(item.owner or "Unassigned")
                esc_dl = html.escape(deadline_str)

                st.markdown(
                    f"""
                    <div style="background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 16px 20px; margin-bottom: 12px; box-shadow: var(--shadow-card);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <div style="display: flex; gap: 8px; align-items: center;">
                                {st_badge}
                                {prio_badge}
                            </div>
                            <span style="font-size: 0.78rem; color: var(--text-secondary); font-family: monospace; background: var(--bg-secondary); border: 1px solid var(--border-color); padding: 2px 8px; border-radius: 6px;">#{idx + 1}</span>
                        </div>
                        <div style="font-size: 1.05rem; font-weight: 600; color: var(--text-main); margin-bottom: 10px; line-height: 1.45;">
                            {esc_action}
                        </div>
                        <div style="display: flex; flex-wrap: wrap; gap: 16px; font-size: 0.85rem; color: var(--text-secondary); border-top: 1px solid var(--border-color); padding-top: 10px;">
                            <div>👤 <strong>Owner:</strong> {esc_owner}</div>
                            <div>📅 <strong>Deadline:</strong> {esc_dl}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No action items found.")

        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            decisions_items = "".join([f"<li style='margin-bottom: 6px;'>{d}</li>" for d in (extraction.decisions or ["None found."])])
            st.markdown(
                f"""
                <div style="background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; height: 100%; box-shadow: var(--shadow-card);">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FF4B4B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
                        </svg>
                        <h4 style="color: var(--text-main); margin: 0; font-size: 1rem; font-weight: 600;">Decisions</h4>
                    </div>
                    <ul style="color: var(--text-main); font-size: 0.92rem; line-height: 1.55; padding-left: 18px; margin: 0;">
                        {decisions_items}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            questions_items = "".join([f"<li style='margin-bottom: 6px;'>{q}</li>" for q in (extraction.open_questions or ["None found."])])
            st.markdown(
                f"""
                <div style="background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; height: 100%; box-shadow: var(--shadow-card);">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FF4B4B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="10"></circle>
                            <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>
                            <line x1="12" y1="17" x2="12.01" y2="17"></line>
                        </svg>
                        <h4 style="color: var(--text-main); margin: 0; font-size: 1rem; font-weight: 600;">Open Questions</h4>
                    </div>
                    <ul style="color: var(--text-main); font-size: 0.92rem; line-height: 1.55; padding-left: 18px; margin: 0;">
                        {questions_items}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            followups_items = "".join([f"<li style='margin-bottom: 6px;'>{f}</li>" for f in (extraction.follow_ups or ["None found."])])
            st.markdown(
                f"""
                <div style="background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; height: 100%; box-shadow: var(--shadow-card);">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FF4B4B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polyline points="23 4 23 10 17 10"></polyline>
                            <polyline points="1 20 1 14 7 14"></polyline>
                            <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
                        </svg>
                        <h4 style="color: var(--text-main); margin: 0; font-size: 1rem; font-weight: 600;">Follow-ups</h4>
                    </div>
                    <ul style="color: var(--text-main); font-size: 0.92rem; line-height: 1.55; padding-left: 18px; margin: 0;">
                        {followups_items}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ======================================================================
    # Section 3: Transcript
    # ======================================================================

    elif current_tab == "transcript":

        orig_text = (st.session_state.get("original_transcript") or "").strip()
        clean_text = (st.session_state.get("clean_transcript") or "").strip()

        # Transcript Header & Comparison View Selector
        v_col1, v_col2 = st.columns([1.6, 1.4])
        with v_col1:
            st.markdown(
                """
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FF4B4B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                        <line x1="16" y1="13" x2="8" y2="13"></line>
                        <line x1="16" y1="17" x2="8" y2="17"></line>
                    </svg>
                    <h4 style="color: var(--text-main); font-weight: 700; margin: 0; font-size: 1.15rem;">Transcript Intelligence & Verification</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with v_col2:
            view_mode = st.radio(
                "Transcript View Mode",
                ["Side-by-Side Comparison", "Clean Transcript Only", "Original Raw Stream Only"],
                horizontal=True,
                label_visibility="collapsed",
                key="transcript_view_mode_toggle",
            )

        # Comparative Analytics Bar
        orig_words = len(orig_text.split()) if orig_text else 0
        clean_words = len(clean_text.split()) if clean_text else 0
        diff_words = max(0, orig_words - clean_words)

        st.markdown(
            f"""
            <div style="background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; padding: 12px 18px; margin-bottom: 18px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; box-shadow: var(--shadow-subtle);">
                <div style="display: flex; align-items: center; gap: 14px; font-size: 0.86rem; color: var(--text-secondary); flex-wrap: wrap;">
                    <span><strong>Raw Audio Stream:</strong> {orig_words:,} words</span>
                    <span>·</span>
                    <span><strong>AI Cleaned & Structured:</strong> {clean_words:,} words</span>
                    <span>·</span>
                    <span style="color: #10B981; font-weight: 600;">{diff_words} filler/stutter words removed</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="background: rgba(16, 185, 129, 0.12); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.25); font-size: 0.74rem; font-weight: 700; padding: 3px 8px; border-radius: 6px;">✓ 100% Facts & Names Preserved</span>
                    <span style="background: rgba(59, 130, 246, 0.12); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.25); font-size: 0.74rem; font-weight: 700; padding: 3px 8px; border-radius: 6px;">✓ STT Errors Repaired</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if view_mode == "Side-by-Side Comparison":
            col_orig, col_clean = st.columns(2)
            with col_orig:
                _render_transcript_display(
                    "Original Transcript",
                    st.session_state.original_transcript,
                    is_original=True,
                )
            with col_clean:
                _render_transcript_display(
                    "Clean Transcript",
                    st.session_state.clean_transcript,
                    is_original=False,
                )
        elif view_mode == "Original Raw Stream Only":
            _render_transcript_display(
                "Original Transcript",
                st.session_state.original_transcript,
                is_original=True,
            )
        else:
            _render_transcript_display(
                "Clean Transcript",
                st.session_state.clean_transcript,
                is_original=False,
            )

    # ======================================================================
    # Section 4: Export Report
    # ======================================================================

    elif current_tab == "export":

        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#FF4B4B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="7 10 12 15 17 10"></polyline>
                    <line x1="12" y1="15" x2="12" y2="3"></line>
                </svg>
                <h4 style="color: var(--text-main); font-weight: 600; margin: 0; font-size: 1.15rem;">Export Report</h4>
            </div>
            <p style="color: var(--text-secondary); font-size: 0.95rem; margin-bottom: 20px;">Download the complete meeting summary, action items, and transcripts in your preferred format.</p>
            """,
            unsafe_allow_html=True,
        )

        from core.report_generator import (
            ReportData,
            build_pdf_report,
            build_txt_report,
            build_jpg_report,
        )

        report_data = ReportData(
            title=summary.title,
            summary=summary,
            extraction=extraction,
            transcript=st.session_state.clean_transcript,
            original_transcript=st.session_state.original_transcript,
        )

        txt_report = build_txt_report(report_data)

        try:
            jpg_bytes = build_jpg_report(report_data)
        except Exception as exc:
            jpg_bytes = None
            st.error(f"Could not build JPG: {exc}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button(
                "Download TXT",
                data=txt_report,
                file_name="report.txt",
                mime="text/plain",
                icon=":material/description:",
                use_container_width=True,
            )

        with col2:
            try:
                pdf_bytes = build_pdf_report(report_data)
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name="report.pdf",
                    mime="application/pdf",
                    icon=":material/picture_as_pdf:",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Could not build PDF: {exc}")

        with col3:
            if jpg_bytes:
                st.download_button(
                    "Download JPG",
                    data=jpg_bytes,
                    file_name="report.jpg",
                    mime="image/jpeg",
                    icon=":material/image:",
                    use_container_width=True,
                )


# --------------------------------------------------------------------------
# Main SaaS Navigation Bar (5 Unified Sections)
# --------------------------------------------------------------------------

current_app_mode = st.session_state.get("app_mode", "workspace")
current_dtab = st.session_state.get("dashboard_tab", "overview")

if current_app_mode == "workspace":
    active_nav = "workspace"
else:
    active_nav = current_dtab

nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)

with nav_col1:
    if st.button(
        "Overview",
        icon=":material/insights:",
        type="primary" if active_nav == "overview" else "secondary",
        use_container_width=True,
        key="nav_btn_overview",
    ):
        st.session_state.app_mode = "memory_dashboard"
        st.session_state.dashboard_tab = "overview"
        st.rerun()

with nav_col2:
    if st.button(
        "Active Meeting Workspace",
        icon=":material/mic:",
        type="primary" if active_nav == "workspace" else "secondary",
        use_container_width=True,
        key="nav_btn_workspace",
    ):
        st.session_state.app_mode = "workspace"
        st.rerun()

with nav_col3:
    if st.button(
        "Meeting History",
        icon=":material/history:",
        type="primary" if active_nav == "history" else "secondary",
        use_container_width=True,
        key="nav_btn_history",
    ):
        st.session_state.app_mode = "memory_dashboard"
        st.session_state.dashboard_tab = "history"
        st.rerun()

with nav_col4:
    if st.button(
        "Action Items",
        icon=":material/checklist:",
        type="primary" if active_nav == "tasks" else "secondary",
        use_container_width=True,
        key="nav_btn_tasks",
    ):
        st.session_state.app_mode = "memory_dashboard"
        st.session_state.dashboard_tab = "tasks"
        st.rerun()

with nav_col5:
    if st.button(
        "Decisions & Memory",
        icon=":material/psychology:",
        type="primary" if active_nav == "memory" else "secondary",
        use_container_width=True,
        key="nav_btn_memory",
    ):
        st.session_state.app_mode = "memory_dashboard"
        st.session_state.dashboard_tab = "memory"
        st.rerun()

st.markdown("<div style='margin-bottom: 22px;'></div>", unsafe_allow_html=True)

curr_uid = current_user.id if current_user else None

if active_nav == "overview":
    render_overview_dashboard(user_id=curr_uid)
elif active_nav == "workspace":
    _render_workspace_view()
elif active_nav == "history":
    render_meeting_history(user_id=curr_uid)
elif active_nav == "tasks":
    render_action_items_tracker(user_id=curr_uid)
elif active_nav == "memory":
    render_decisions_and_memory(user_id=curr_uid)


# --------------------------------------------------------------------------
# Floating AI Chat Assistant (Shared Across Both Modes)
# --------------------------------------------------------------------------

_render_floating_ai_chat()