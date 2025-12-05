"""
Streamlit authentication helpers based on bcrypt password hashes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import bcrypt
import streamlit as st

from auth_store import UserRecord, get_user

SESSION_FLAG = "auth.is_authenticated"
SESSION_USERNAME = "auth.username"
SESSION_ERROR = "auth.error"

LOGGER = logging.getLogger(__name__)


def _init_session_state() -> None:
    st.session_state.setdefault(SESSION_FLAG, False)
    st.session_state.setdefault(SESSION_USERNAME, None)


def hash_password(plain_text: str) -> str:
    """Generate a salted bcrypt hash."""
    if not plain_text:
        raise ValueError("Password cannot be empty.")
    return bcrypt.hashpw(plain_text.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_text: str, hashed: str) -> bool:
    """Validate a password against its bcrypt hash."""
    if not plain_text or not hashed:
        return False
    try:
        return bcrypt.checkpw(plain_text.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        LOGGER.warning("Invalid bcrypt hash encountered for comparison.")
        return False


@dataclass
class AuthResult:
    success: bool
    message: Optional[str] = None


def authenticate(username: str, password: str) -> AuthResult:
    """Check credentials and update the session state if valid."""
    _init_session_state()
    username = (username or "").strip()
    if not username or not password:
        return AuthResult(False, "请输入用户名和密码。")

    record: Optional[UserRecord] = get_user(username)
    if record is None:
        LOGGER.info("Login attempt for unknown user '%s'", username)
        return AuthResult(False, "用户名或密码错误。")

    if not verify_password(password, record.password_hash):
        LOGGER.info("Password mismatch for user '%s'", username)
        return AuthResult(False, "用户名或密码错误。")

    st.session_state[SESSION_FLAG] = True
    st.session_state[SESSION_USERNAME] = username
    return AuthResult(True, None)


def logout() -> None:
    """Clear the authentication session state."""
    _init_session_state()
    st.session_state[SESSION_FLAG] = False
    st.session_state[SESSION_USERNAME] = None


def is_authenticated() -> bool:
    """Expose the current auth flag."""
    _init_session_state()
    return bool(st.session_state.get(SESSION_FLAG))


def current_user() -> Optional[str]:
    """Get the logged-in username."""
    _init_session_state()
    return st.session_state.get(SESSION_USERNAME)


def render_login(title: str = "🔐 登录", subtitle: str = "请输入账号和密码") -> None:
    """Render a login form and handle submission feedback."""
    _init_session_state()
    error_message = st.session_state.pop(SESSION_ERROR, None)
    st.title(title)
    st.caption(subtitle)

    with st.form("auth-login-form", clear_on_submit=False):
        username = st.text_input("用户名", value=current_user() or "")
        password = st.text_input("密码", type="password")
        submitted = st.form_submit_button("登录", use_container_width=True)

    if submitted:
        result = authenticate(username, password)
        if result.success:
            st.success("登录成功，即将进入系统…")
            st.rerun()
        else:
            st.session_state[SESSION_ERROR] = result.message or "登录失败。"
            st.rerun()

    if error_message:
        st.error(error_message)


def enforce_login() -> None:
    """
    Ensure the current session is authenticated.

    When the user is not authenticated the login form is rendered and
    execution stops so the rest of the page body does not render.
    """
    if not is_authenticated():
        render_login()
        st.stop()


def render_logout_button(label: str = "退出登录", *, container=None) -> None:
    """Display a logout button inside the provided container."""
    if not is_authenticated():
        return

    target = container if container is not None else st.sidebar
    if target.button(label, use_container_width=True, key="auth-logout-btn"):
        logout()
        st.success("已退出登录。")
        st.rerun()


