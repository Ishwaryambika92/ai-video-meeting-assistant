"""
AI Video/Meeting Assistant - Authentication UI Components.

Provides production-quality, responsive UI for:
1. Login with email & password, password visibility toggle, and remember me.
2. Sign Up with email, password strength meter, confirmation, and terms agreement.
3. Forgot Password & Password Reset workflows.
4. User Profile card & Logout controls in the application sidebar.
"""

from __future__ import annotations

import html
import re
from typing import Callable

import streamlit as st

from config import get_logger, get_settings
from core.auth import AuthService, User, get_auth_service

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Helpers & Styling
# -----------------------------------------------------------------------------

def _render_auth_header() -> None:
    """Render the branded hero header for authentication screens."""
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 28px; padding-top: 10px;">
            <div style="display: inline-flex; align-items: center; justify-content: center; width: 56px; height: 56px; border-radius: 16px; background: linear-gradient(135deg, #FF4B4B 0%, #E63E45 100%); box-shadow: 0 8px 24px rgba(255, 75, 75, 0.35); margin-bottom: 16px;">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="2.5" y="5.5" width="12" height="13" rx="2.5" stroke="#FFFFFF" stroke-width="2"/>
                    <path d="M14.5 10L20.5 6.5V17.5L14.5 14" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M8.5 8C8.5 9.2 9.3 10 10.5 10C9.3 10 8.5 10.8 8.5 12C8.5 10.8 7.7 10 6.5 10C7.7 10 8.5 9.2 8.5 8Z" fill="#FFFFFF"/>
                </svg>
            </div>
            <h2 style="margin: 0 0 6px 0; font-size: 1.85rem; font-weight: 800; letter-spacing: -0.02em; color: var(--text-main);">
                AI Video & Meeting Assistant
            </h2>
            <p style="margin: 0; font-size: 0.95rem; color: var(--text-secondary); max-width: 420px; display: inline-block;">
                Enterprise-grade intelligence for meeting transcription, action tracking, and multi-modal memory.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _calculate_password_strength(password: str) -> tuple[int, str, str]:
    """
    Calculate password strength score from 0-4.
    Returns: (score, label, color_hex)
    """
    if not password:
        return (0, "Empty", "var(--text-muted)")
    score = 0
    if len(password) >= 8:
        score += 1
    if re.search(r"[A-Z]", password) and re.search(r"[a-z]", password):
        score += 1
    if re.search(r"\d", password):
        score += 1
    if re.search(r"[@$!%*?&_\-#^~+=/\\|<>]", password):
        score += 1

    if score <= 1:
        return (score, "Weak (min 8 chars with mixed cases)", "#EF4444")
    elif score == 2:
        return (score, "Fair (add numbers & symbols)", "#F59E0B")
    elif score == 3:
        return (score, "Good (add special characters)", "#3B82F6")
    else:
        return (score, "Strong password", "#10B981")


# -----------------------------------------------------------------------------
# Main Auth Page Controller
# -----------------------------------------------------------------------------

def render_auth_page() -> None:
    """
    Render the main authentication container with tabbed or stateful switching
    between Login, Sign Up, and Forgot Password modes.
    """
    auth_service = get_auth_service()

    if "auth_view_mode" not in st.session_state:
        st.session_state["auth_view_mode"] = "login"

    # Center card container using responsive Streamlit columns
    left_space, card_col, right_space = st.columns([1, 1.4, 1])

    with card_col:
        with st.container(border=True):
            _render_auth_header()

            # Engine mode badge
            engine_badge = "Supabase Cloud Auth" if auth_service.is_supabase else "Production Cryptographic Vault (bcrypt)"
            badge_bg = "rgba(16, 185, 129, 0.12)" if auth_service.is_supabase else "rgba(59, 130, 246, 0.12)"
            badge_color = "#10B981" if auth_service.is_supabase else "#3B82F6"
            st.markdown(
                f"""
                <div style="text-align: center; margin-bottom: 22px;">
                    <span style="display: inline-flex; align-items: center; gap: 6px; font-size: 0.76rem; font-weight: 600; padding: 4px 12px; border-radius: 9999px; background-color: {badge_bg}; color: {badge_color}; border: 1px solid {badge_color}33;">
                        <span style="width: 6px; height: 6px; border-radius: 50%; background-color: {badge_color};"></span>
                        Security Engine: {engine_badge}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Display any flashed auth error messages
            if "auth_error" in st.session_state:
                st.error(st.session_state.pop("auth_error"))

            # Render corresponding form
            view_mode = st.session_state.get("auth_view_mode", "login")

            if view_mode == "login":
                _render_login_form(auth_service)
            elif view_mode == "signup":
                _render_signup_form(auth_service)
            elif view_mode == "forgot_password":
                _render_forgot_password_form(auth_service)
            elif view_mode == "reset_password":
                _render_reset_password_form(auth_service)


# -----------------------------------------------------------------------------
# 1. Login View
# -----------------------------------------------------------------------------

def _render_login_form(auth_service: AuthService) -> None:
    """Render modern email & password login interface."""
    st.markdown(
        """
        <div style="margin-bottom: 18px;">
            <h3 style="margin: 0 0 4px 0; font-size: 1.35rem; font-weight: 700; color: var(--text-main);">Welcome Back</h3>
            <p style="margin: 0; font-size: 0.88rem; color: var(--text-secondary);">Enter your credentials to access your workspaces and meeting history.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show password toggle state
    if "login_show_pwd" not in st.session_state:
        st.session_state["login_show_pwd"] = False

    with st.form("form_login", clear_on_submit=False):
        email = st.text_input(
            "Email Address",
            placeholder="you@company.com",
            key="login_email_input",
        )

        pwd_type = "default" if st.session_state.get("login_show_pwd") else "password"
        password = st.text_input(
            "Password",
            type=pwd_type,
            placeholder="Enter your password",
            key="login_pwd_input",
        )

        col_check1, col_check2 = st.columns([1, 1])
        with col_check1:
            remember_me = st.checkbox("Remember me", value=True, key="login_remember_me")
        with col_check2:
            show_pwd = st.checkbox("Show password", value=st.session_state["login_show_pwd"], key="login_show_pwd_toggle")
            if show_pwd != st.session_state["login_show_pwd"]:
                st.session_state["login_show_pwd"] = show_pwd

        submit_login = st.form_submit_button(
            "Sign In",
            type="primary",
            use_container_width=True,
        )

    # Handle Login Submission
    if submit_login:
        if not email or not password:
            st.error("Please enter both email and password.")
        else:
            with st.spinner("Authenticating securely..."):
                result = auth_service.login(email=email.strip(), password=password)

            if result.success and result.user:
                st.session_state["authenticated_user"] = result.user
                if result.session_token and remember_me:
                    st.query_params["auth_token"] = result.session_token
                    st.session_state["auth_token"] = result.session_token
                st.toast(f"Welcome back, {result.user.name or result.user.email}!", icon="🎉")
                st.rerun()
            else:
                st.error(f"Authentication Failed: {result.error_message}")

    # Navigation Links (Forgot Password & Sign Up)
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    link_col1, link_col2 = st.columns(2)
    with link_col1:
        if st.button("Forgot Password?", key="btn_to_forgot_pwd", use_container_width=True):
            st.session_state["auth_view_mode"] = "forgot_password"
            st.rerun()

    with link_col2:
        if st.button("Create an Account", key="btn_to_signup", type="secondary", use_container_width=True):
            st.session_state["auth_view_mode"] = "signup"
            st.rerun()


# -----------------------------------------------------------------------------
# 2. Sign Up View
# -----------------------------------------------------------------------------

def _render_signup_form(auth_service: AuthService) -> None:
    """Render comprehensive user registration form with validation and strength indicator."""
    st.markdown(
        """
        <div style="margin-bottom: 18px;">
            <h3 style="margin: 0 0 4px 0; font-size: 1.35rem; font-weight: 700; color: var(--text-main);">Create Your Account</h3>
            <p style="margin: 0; font-size: 0.88rem; color: var(--text-secondary);">Start transcribing, tracking decisions, and querying your video meetings.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "signup_show_pwd" not in st.session_state:
        st.session_state["signup_show_pwd"] = False

    with st.form("form_signup", clear_on_submit=False):
        name = st.text_input(
            "Full Name",
            placeholder="Jane Doe",
            key="signup_name_input",
        )

        email = st.text_input(
            "Work Email Address",
            placeholder="jane@company.com",
            key="signup_email_input",
        )

        pwd_type = "default" if st.session_state.get("signup_show_pwd") else "password"
        password = st.text_input(
            "Password",
            type=pwd_type,
            placeholder="At least 8 characters (mixed case, numbers, symbols)",
            key="signup_pwd_input",
        )

        password_confirm = st.text_input(
            "Confirm Password",
            type=pwd_type,
            placeholder="Re-enter your password",
            key="signup_pwd_confirm_input",
        )

        col_opts1, col_opts2 = st.columns([1, 1])
        with col_opts1:
            show_pwd = st.checkbox("Show passwords", value=st.session_state["signup_show_pwd"], key="signup_show_pwd_toggle")
            if show_pwd != st.session_state["signup_show_pwd"]:
                st.session_state["signup_show_pwd"] = show_pwd
        with col_opts2:
            terms_agreed = st.checkbox("I agree to Terms & Privacy", value=True, key="signup_terms_agree")

        submit_signup = st.form_submit_button(
            "Complete Registration",
            type="primary",
            use_container_width=True,
        )

    # Real-time Password Strength Meter preview
    if password:
        score, label, color = _calculate_password_strength(password)
        bar_width = f"{score * 25}%"
        st.markdown(
            f"""
            <div style="margin: -6px 0 16px 0;">
                <div style="display: flex; justify-content: space-between; font-size: 0.78rem; font-weight: 600; color: var(--text-secondary); margin-bottom: 4px;">
                    <span>Password Strength</span>
                    <span style="color: {color};">{label}</span>
                </div>
                <div style="width: 100%; height: 5px; background-color: var(--border-color); border-radius: 9999px; overflow: hidden;">
                    <div style="width: {bar_width}; height: 100%; background-color: {color}; transition: width 0.3s ease;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Handle Sign Up Submission
    if submit_signup:
        if not terms_agreed:
            st.error("You must agree to the Terms of Service to create an account.")
        elif not name.strip():
            st.error("Please enter your name.")
        elif not email.strip():
            st.error("Please enter your email address.")
        elif password != password_confirm:
            st.error("Passwords do not match. Please ensure both fields are identical.")
        else:
            with st.spinner("Creating secure account..."):
                result = auth_service.signup(
                    email=email.strip(),
                    password=password,
                    name=name.strip(),
                )

            if result.success and result.user:
                st.session_state["authenticated_user"] = result.user
                if result.session_token:
                    st.query_params["auth_token"] = result.session_token
                    st.session_state["auth_token"] = result.session_token
                st.success("Account created successfully! Welcome to Video Agent.")
                st.toast("Registration complete!", icon="🚀")
                st.rerun()
            else:
                st.error(f"Sign Up Failed: {result.error_message}")

    # Back to Login Navigation
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    if st.button("Already have an account? Sign In", key="btn_back_to_login", use_container_width=True):
        st.session_state["auth_view_mode"] = "login"
        st.rerun()


# -----------------------------------------------------------------------------
# 3. Forgot Password View
# -----------------------------------------------------------------------------

def _render_forgot_password_form(auth_service: AuthService) -> None:
    """Render password reset request form."""
    st.markdown(
        """
        <div style="margin-bottom: 18px;">
            <h3 style="margin: 0 0 4px 0; font-size: 1.35rem; font-weight: 700; color: var(--text-main);">Reset Your Password</h3>
            <p style="margin: 0; font-size: 0.88rem; color: var(--text-secondary);">Enter your account email to receive a password recovery token.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("form_forgot_password", clear_on_submit=False):
        email = st.text_input(
            "Account Email",
            placeholder="you@company.com",
            key="forgot_email_input",
        )
        submit_forgot = st.form_submit_button(
            "Send Reset Instructions",
            type="primary",
            use_container_width=True,
        )

    if submit_forgot:
        if not email.strip():
            st.error("Please enter your registered email address.")
        else:
            with st.spinner("Generating password reset request..."):
                result = auth_service.request_password_reset(email.strip())

            if result.success:
                st.success("Password reset request created successfully!")
                if result.reset_token:
                    st.session_state["active_reset_token"] = result.reset_token
                    st.info(
                        f"🔒 **Recovery Token Generated**: `{result.reset_token}`\n\n"
                        "Copy this token and use it below to choose a new password."
                    )
                    st.session_state["auth_view_mode"] = "reset_password"
                    st.rerun()
            else:
                st.error(result.error_message or "Could not process password reset.")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("Have a reset token?", key="btn_go_to_reset_token", use_container_width=True):
            st.session_state["auth_view_mode"] = "reset_password"
            st.rerun()
    with col_nav2:
        if st.button("Back to Login", key="btn_forgot_to_login", use_container_width=True):
            st.session_state["auth_view_mode"] = "login"
            st.rerun()


# -----------------------------------------------------------------------------
# 4. Reset Password View
# -----------------------------------------------------------------------------

def _render_reset_password_form(auth_service: AuthService) -> None:
    """Render password reset confirmation form."""
    st.markdown(
        """
        <div style="margin-bottom: 18px;">
            <h3 style="margin: 0 0 4px 0; font-size: 1.35rem; font-weight: 700; color: var(--text-main);">Choose New Password</h3>
            <p style="margin: 0; font-size: 0.88rem; color: var(--text-secondary);">Enter the recovery token and specify your new secure password.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_token = st.session_state.get("active_reset_token", "")

    with st.form("form_reset_password", clear_on_submit=False):
        token = st.text_input(
            "Reset Recovery Token",
            value=default_token,
            placeholder="Paste your 64-character token here",
            key="reset_token_input",
        )

        new_password = st.text_input(
            "New Password",
            type="password",
            placeholder="At least 8 characters (mixed case, numbers, symbols)",
            key="reset_new_pwd_input",
        )

        new_password_confirm = st.text_input(
            "Confirm New Password",
            type="password",
            placeholder="Re-enter your new password",
            key="reset_new_pwd_confirm_input",
        )

        submit_reset = st.form_submit_button(
            "Update Password",
            type="primary",
            use_container_width=True,
        )

    if submit_reset:
        clean_token = (token or "").strip()
        if not clean_token:
            st.error("Please provide the recovery token.")
        elif not new_password:
            st.error("Please enter a new password.")
        elif new_password != new_password_confirm:
            st.error("Passwords do not match. Please verify both fields.")
        else:
            with st.spinner("Updating password securely..."):
                result = auth_service.confirm_password_reset(clean_token, new_password)

            if result.success:
                st.success("Your password has been updated successfully! You can now log in.")
                st.session_state["auth_view_mode"] = "login"
                st.session_state.pop("active_reset_token", None)
                st.rerun()
            else:
                st.error(f"Reset Failed: {result.error_message}")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    if st.button("Cancel & Return to Login", key="btn_cancel_reset", use_container_width=True):
        st.session_state["auth_view_mode"] = "login"
        st.rerun()


# -----------------------------------------------------------------------------
# 5. User Profile Menu & Sidebar Card
# -----------------------------------------------------------------------------

def render_user_profile_dropdown(current_user: User) -> None:
    """
    Render a polished profile card and logout trigger in the sidebar.
    Displays user initials, display name, email, auth provider badge, and logout button.
    """
    auth_service = get_auth_service()

    # Determine initials for user avatar
    display_name = current_user.name.strip() if current_user.name else current_user.email.split("@")[0]
    initials = "".join([part[0].upper() for part in display_name.split()[:2]]) or "U"
    provider_label = "Supabase Auth" if current_user.auth_provider == "supabase" else "Local Vault"

    st.sidebar.markdown(
        f"""
        <div style="background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 12px 14px; margin-bottom: 18px; box-shadow: var(--shadow-subtle);">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="width: 38px; height: 38px; border-radius: 50%; background: linear-gradient(135deg, #FF4B4B 0%, #E63E45 100%); color: #FFFFFF; font-weight: 700; font-size: 0.95rem; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 8px rgba(255, 75, 75, 0.3); flex-shrink: 0;">
                    {html.escape(initials)}
                </div>
                <div style="overflow: hidden; flex: 1;">
                    <div style="font-weight: 600; font-size: 0.88rem; color: var(--text-main); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                        {html.escape(display_name)}
                    </div>
                    <div style="font-size: 0.76rem; color: var(--text-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                        {html.escape(current_user.email)}
                    </div>
                </div>
            </div>
            <div style="margin-top: 8px; display: flex; align-items: center; justify-content: space-between; font-size: 0.72rem; color: var(--text-muted); border-top: 1px solid var(--border-color); padding-top: 6px;">
                <span>Provider: <strong>{provider_label}</strong></span>
                <span style="color: #10B981; font-weight: 600;">● Active</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.sidebar.button(
        "Sign Out",
        icon=":material/logout:",
        use_container_width=True,
        key="btn_sidebar_logout",
        help="Log out and terminate active session",
    ):
        active_token = st.session_state.get("auth_token") or st.query_params.get("auth_token")
        if active_token:
            auth_service.logout(active_token)

        st.session_state.pop("authenticated_user", None)
        st.session_state.pop("auth_token", None)
        st.query_params.pop("auth_token", None)
        st.toast("Signed out safely.", icon="👋")
        st.rerun()
