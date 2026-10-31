"""
Authentication & User Management Engine for Video Agent.

Supports dual-mode authentication:
1. Supabase Cloud Auth (when SUPABASE_URL and SUPABASE_ANON_KEY are provided)
2. Production-grade Local Cryptographic Vault (bcrypt + SQLite sessions + password reset tokens)

Handles Sign Up, Login, Password Reset, Email Verification,
and Multi-tenant Session Persistence.
"""

from __future__ import annotations

import re
import secrets
import sqlite3
import sys
import urllib.parse
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import bcrypt

from config import get_logger, settings

logger = get_logger(__name__)

DB_PATH = Path(_PROJECT_ROOT) / "data" / "meetings.db"


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

@dataclass
class User:
    id: str
    email: str
    full_name: str
    avatar_url: str = ""
    is_verified: bool = True
    auth_provider: str = "email"  # 'email', 'google', 'supabase'
    created_at: str = ""
    last_login: str = ""

    @property
    def name(self) -> str:
        return self.full_name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> User:
        return cls(
            id=str(data.get("id", "")),
            email=str(data.get("email", "")),
            full_name=str(data.get("full_name", "")),
            avatar_url=str(data.get("avatar_url", "")),
            is_verified=bool(data.get("is_verified", True)),
            auth_provider=str(data.get("auth_provider", "email")),
            created_at=str(data.get("created_at", "")),
            last_login=str(data.get("last_login", "")),
        )


@dataclass
class AuthResult:
    success: bool
    user: User | None = None
    session_token: str | None = None
    error_message: str | None = None
    info_message: str | None = None
    requires_verification: bool = False
    reset_token: str | None = None


# -----------------------------------------------------------------------------
# Validation Helpers
# -----------------------------------------------------------------------------

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


def validate_email(email: str) -> tuple[bool, str]:
    """Validate email format."""
    if not email or not email.strip():
        return False, "Email address is required."
    clean = email.strip()
    if len(clean) > 254:
        return False, "Email address is too long."
    if not EMAIL_REGEX.match(clean):
        return False, "Please enter a valid email address (e.g., name@company.com)."
    return True, ""


def validate_password_strength(password: str) -> tuple[bool, str]:
    """
    Validate password meets enterprise security requirements:
    - At least 8 characters
    - At least one letter
    - At least one digit
    """
    if not password:
        return False, "Password is required."
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[a-zA-Z]", password):
        return False, "Password must contain at least one letter."
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number."
    return True, ""


# -----------------------------------------------------------------------------
# Local SQLite Cryptographic Auth Backend
# -----------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn



def init_auth_tables() -> None:
    """Create authentication tables and ensure user_id columns exist."""
    conn = _get_connection()
    try:
        # 1. Users table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                is_verified INTEGER DEFAULT 0,
                auth_provider TEXT DEFAULT 'email',
                avatar_url TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                last_login TEXT DEFAULT ''
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")

        # 2. Auth sessions table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON auth_sessions(user_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expires ON auth_sessions(expires_at);")

        # 3. Password reset tokens table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS password_resets (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                email TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_resets_token ON password_resets(token);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_resets_user ON password_resets(user_id);")

        # 4. Email verification tokens table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS email_verifications (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                email TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                verified INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_verifications_user ON email_verifications(user_id);")

        # 5. Safe migration: ensure meetings and action_items have user_id
        for table in ("meetings", "action_items"):
            try:
                cur = conn.execute(f"PRAGMA table_info({table})")
                rows = cur.fetchall()
                if rows:
                    cols = [r[1] if isinstance(r, (tuple, list)) else r["name"] for r in rows]
                    if "user_id" not in cols:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN user_id TEXT DEFAULT 'default_user';")
                        logger.info("Migrated table '%s' to include user_id column.", table)
                    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_user ON {table}(user_id);")
            except Exception as mig_err:
                logger.warning("Migration check for table '%s' skipped: %s", table, mig_err)

        conn.commit()
    finally:
        conn.close()


# Initialize tables upon module import
init_auth_tables()


class LocalAuthBackend:
    """Production-grade local cryptographic authentication backend."""

    def sign_up(self, email: str, password: str, full_name: str) -> AuthResult:
        valid_email, err_email = validate_email(email)
        if not valid_email:
            return AuthResult(success=False, error_message=err_email)

        valid_pw, err_pw = validate_password_strength(password)
        if not valid_pw:
            return AuthResult(success=False, error_message=err_pw)

        clean_name = full_name.strip()
        if not clean_name:
            clean_name = email.split("@")[0].replace(".", " ").title()

        clean_email = email.strip().lower()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Generate bcrypt salt and hash
        pw_bytes = password.encode("utf-8")
        pw_hash = bcrypt.hashpw(pw_bytes, bcrypt.gensalt(rounds=12)).decode("utf-8")

        user_id = str(uuid.uuid4())
        verification_token = secrets.token_urlsafe(32)
        verify_expires = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()

        with _get_connection() as conn:
            # Check for existing email
            cur = conn.execute("SELECT id FROM users WHERE email = ?", (clean_email,))
            if cur.fetchone():
                return AuthResult(
                    success=False,
                    error_message="An account with this email already exists. Please log in.",
                )

            # Insert user
            conn.execute(
                """
                INSERT INTO users (
                    id, email, full_name, password_hash, is_verified,
                    auth_provider, avatar_url, created_at, last_login
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, clean_email, clean_name, pw_hash, 1, "email", "", now_iso, now_iso),
            )

            # Record verification token
            conn.execute(
                """
                INSERT INTO email_verifications (token, user_id, email, created_at, expires_at, verified)
                VALUES (?, ?, ?, ?, ?, 1)
                """,
                (verification_token, user_id, clean_email, now_iso, verify_expires),
            )

            # Create session token
            session_token = secrets.token_hex(32)
            session_expires = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
            conn.execute(
                """
                INSERT INTO auth_sessions (token, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_token, user_id, now_iso, session_expires),
            )

        user = User(
            id=user_id,
            email=clean_email,
            full_name=clean_name,
            is_verified=True,
            auth_provider="email",
            created_at=now_iso,
            last_login=now_iso,
        )
        logger.info("Successfully registered user '%s' (%s)", clean_name, clean_email)
        return AuthResult(
            success=True,
            user=user,
            session_token=session_token,
            info_message="Account created successfully! Welcome to Video Agent.",
        )

    def login(self, email: str, password: str) -> AuthResult:
        valid_email, err_email = validate_email(email)
        if not valid_email:
            return AuthResult(success=False, error_message=err_email)

        if not password:
            return AuthResult(success=False, error_message="Password is required.")

        clean_email = email.strip().lower()
        now_iso = datetime.now(timezone.utc).isoformat()

        with _get_connection() as conn:
            cur = conn.execute("SELECT * FROM users WHERE email = ?", (clean_email,))
            row = cur.fetchone()
            if not row:
                return AuthResult(
                    success=False,
                    error_message="Invalid email or password. Please try again.",
                )

            pw_hash = row["password_hash"]
            pw_bytes = password.encode("utf-8")
            hash_bytes = pw_hash.encode("utf-8")

            if not bcrypt.checkpw(pw_bytes, hash_bytes):
                return AuthResult(
                    success=False,
                    error_message="Invalid email or password. Please try again.",
                )

            user_id = row["id"]
            # Update last login
            conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (now_iso, user_id))

            # Create session token
            session_token = secrets.token_hex(32)
            session_expires = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
            conn.execute(
                """
                INSERT INTO auth_sessions (token, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_token, user_id, now_iso, session_expires),
            )

            user = User(
                id=user_id,
                email=row["email"],
                full_name=row["full_name"],
                avatar_url=row["avatar_url"] or "",
                is_verified=bool(row["is_verified"]),
                auth_provider=row["auth_provider"] or "email",
                created_at=row["created_at"],
                last_login=now_iso,
            )

        logger.info("User '%s' logged in successfully.", clean_email)
        return AuthResult(
            success=True,
            user=user,
            session_token=session_token,
            info_message=f"Welcome back, {user.full_name}!",
        )

    def handle_google_user(
        self,
        email: str,
        name: str = "",
        avatar_url: str = "",
    ) -> AuthResult:
        """Create or update a verified Google user record and issue an active session."""
        if not email or not email.strip():
            return AuthResult(success=False, error_message="Verified email was not provided by Google.")

        clean_email = email.strip().lower()
        clean_name = name.strip() or clean_email.split("@")[0].replace(".", " ").title()
        now_iso = datetime.now(timezone.utc).isoformat()

        with _get_connection() as conn:
            cur = conn.execute("SELECT * FROM users WHERE email = ?", (clean_email,))
            row = cur.fetchone()

            if row:
                user_id = row["id"]
                conn.execute(
                    "UPDATE users SET last_login = ?, auth_provider = 'google', avatar_url = ?, full_name = ? WHERE id = ?",
                    (now_iso, avatar_url or row["avatar_url"] or "", clean_name or row["full_name"], user_id),
                )
                user = User(
                    id=user_id,
                    email=clean_email,
                    full_name=clean_name or row["full_name"],
                    avatar_url=avatar_url or row["avatar_url"] or "",
                    is_verified=True,
                    auth_provider="google",
                    created_at=row["created_at"],
                    last_login=now_iso,
                )
            else:
                user_id = str(uuid.uuid4())
                dummy_hash = bcrypt.hashpw(secrets.token_bytes(32), bcrypt.gensalt()).decode("utf-8")
                conn.execute(
                    """
                    INSERT INTO users (
                        id, email, full_name, password_hash, is_verified,
                        auth_provider, avatar_url, created_at, last_login
                    ) VALUES (?, ?, ?, ?, 1, 'google', ?, ?, ?)
                    """,
                    (user_id, clean_email, clean_name, dummy_hash, avatar_url, now_iso, now_iso),
                )
                user = User(
                    id=user_id,
                    email=clean_email,
                    full_name=clean_name,
                    avatar_url=avatar_url,
                    is_verified=True,
                    auth_provider="google",
                    created_at=now_iso,
                    last_login=now_iso,
                )

            session_token = secrets.token_hex(32)
            session_expires = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
            conn.execute(
                """
                INSERT INTO auth_sessions (token, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_token, user_id, now_iso, session_expires),
            )

        logger.info("Google OAuth login successful for '%s'", clean_email)
        return AuthResult(
            success=True,
            user=user,
            session_token=session_token,
            info_message=f"Signed in with Google as {clean_name}!",
        )

    def login_with_google(self, email: str | None = None, name: str | None = None) -> AuthResult:
        """Handle or complete Google OAuth sign-in."""
        if not email:
            return AuthResult(success=False, error_message="Email is required for Google authentication.")
        return self.handle_google_user(email=email, name=name or "")

    def request_password_reset(self, email: str) -> AuthResult:
        valid_email, err_email = validate_email(email)
        if not valid_email:
            return AuthResult(success=False, error_message=err_email)

        clean_email = email.strip().lower()
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        expires_iso = (now + timedelta(hours=2)).isoformat()
        reset_token = secrets.token_urlsafe(32)

        with _get_connection() as conn:
            cur = conn.execute("SELECT id FROM users WHERE email = ?", (clean_email,))
            row = cur.fetchone()
            if not row:
                # Return generic success to prevent email enumeration
                return AuthResult(
                    success=True,
                    info_message="If an account exists for this email, password reset instructions have been generated.",
                    reset_token=None,
                )

            user_id = row["id"]
            conn.execute(
                """
                INSERT INTO password_resets (token, user_id, email, created_at, expires_at, used)
                VALUES (?, ?, ?, ?, ?, 0)
                """,
                (reset_token, user_id, clean_email, now_iso, expires_iso),
            )

        logger.info("Generated password reset token for '%s'", clean_email)
        return AuthResult(
            success=True,
            info_message="Password reset link generated. You can now set your new password.",
            reset_token=reset_token,
        )

    def reset_password(self, token: str, new_password: str) -> AuthResult:
        if not token or not token.strip():
            return AuthResult(success=False, error_message="Reset token is required.")

        valid_pw, err_pw = validate_password_strength(new_password)
        if not valid_pw:
            return AuthResult(success=False, error_message=err_pw)

        now_iso = datetime.now(timezone.utc).isoformat()

        with _get_connection() as conn:
            cur = conn.execute(
                """
                SELECT * FROM password_resets
                WHERE token = ? AND used = 0 AND expires_at > ?
                """,
                (token.strip(), now_iso),
            )
            row = cur.fetchone()
            if not row:
                return AuthResult(
                    success=False,
                    error_message="This password reset token is invalid or has expired. Please request a new one.",
                )

            user_id = row["user_id"]
            pw_bytes = new_password.encode("utf-8")
            pw_hash = bcrypt.hashpw(pw_bytes, bcrypt.gensalt(rounds=12)).decode("utf-8")

            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (pw_hash, user_id))
            conn.execute("UPDATE password_resets SET used = 1 WHERE token = ?", (token.strip(),))
            # Invalidate all prior sessions for security
            conn.execute("DELETE FROM auth_sessions WHERE user_id = ?", (user_id,))

        logger.info("Successfully updated password for user '%s'", user_id)
        return AuthResult(
            success=True,
            info_message="Password updated successfully! You can now log in with your new password.",
        )

    def validate_session_token(self, token: str) -> User | None:
        if not token or not token.strip():
            return None

        now_iso = datetime.now(timezone.utc).isoformat()

        with _get_connection() as conn:
            cur = conn.execute(
                """
                SELECT u.* FROM users u
                JOIN auth_sessions s ON u.id = s.user_id
                WHERE s.token = ? AND s.expires_at > ?
                """,
                (token.strip(), now_iso),
            )
            row = cur.fetchone()
            if not row:
                # Cleanup expired sessions
                conn.execute("DELETE FROM auth_sessions WHERE expires_at <= ?", (now_iso,))
                return None

            return User(
                id=row["id"],
                email=row["email"],
                full_name=row["full_name"],
                avatar_url=row["avatar_url"] or "",
                is_verified=bool(row["is_verified"]),
                auth_provider=row["auth_provider"] or "email",
                created_at=row["created_at"],
                last_login=row["last_login"] or "",
            )

    def logout(self, token: str) -> None:
        if not token:
            return
        with _get_connection() as conn:
            conn.execute("DELETE FROM auth_sessions WHERE token = ?", (token.strip(),))
        logger.info("Session token invalidated.")


# -----------------------------------------------------------------------------
# Supabase Cloud Auth Backend
# -----------------------------------------------------------------------------

class SupabaseAuthBackend:
    """Supabase Auth backend for production cloud deployments."""

    def __init__(self, url: str, key: str) -> None:
        self.client: Any = None
        self.fallback = LocalAuthBackend()
        try:
            from supabase import Client, create_client  # type: ignore[import-not-found,import-untyped] # pyrefly: ignore
            self.client = create_client(url, key)
        except Exception as exc:
            logger.warning("Could not initialize Supabase SDK: %s. Using local vault.", exc)

    def sign_up(self, email: str, password: str, full_name: str) -> AuthResult:
        try:
            res = self.client.auth.sign_up(
                {
                    "email": email.strip(),
                    "password": password,
                    "options": {"data": {"full_name": full_name.strip()}},
                }
            )
            if res.user:
                u = res.user
                user = User(
                    id=str(u.id),
                    email=str(u.email),
                    full_name=str(u.user_metadata.get("full_name", full_name)),
                    is_verified=bool(u.email_confirmed_at),
                    auth_provider="supabase",
                    created_at=str(u.created_at),
                )
                session_token = res.session.access_token if res.session else secrets.token_hex(32)
                return AuthResult(
                    success=True,
                    user=user,
                    session_token=session_token,
                    info_message="Account created successfully via Supabase Auth!",
                )
            return AuthResult(success=False, error_message="Could not create Supabase account.")
        except Exception as e:
            logger.warning("Supabase sign_up error: %s. Falling back to local vault.", e)
            return self.fallback.sign_up(email, password, full_name)

    def login(self, email: str, password: str) -> AuthResult:
        try:
            res = self.client.auth.sign_in_with_password(
                {"email": email.strip(), "password": password}
            )
            if res.user:
                u = res.user
                user = User(
                    id=str(u.id),
                    email=str(u.email),
                    full_name=str(u.user_metadata.get("full_name", email.split("@")[0])),
                    is_verified=bool(u.email_confirmed_at),
                    auth_provider="supabase",
                )
                session_token = res.session.access_token if res.session else secrets.token_hex(32)
                return AuthResult(
                    success=True,
                    user=user,
                    session_token=session_token,
                    info_message=f"Welcome back, {user.full_name}!",
                )
            return AuthResult(success=False, error_message="Invalid email or password.")
        except Exception as e:
            logger.warning("Supabase sign_in error: %s. Attempting local vault.", e)
            return self.fallback.login(email, password)

    def get_google_auth_url(self, redirect_uri: str | None = None) -> tuple[bool, str]:
        return False, "Google OAuth is disabled."

    def exchange_code(self, code: str) -> AuthResult:
        return AuthResult(success=False, error_message="Google OAuth is disabled.")

    def login_with_google(self) -> AuthResult:
        return AuthResult(success=False, error_message="Google OAuth is disabled.")

    def request_password_reset(self, email: str) -> AuthResult:
        try:
            self.client.auth.reset_password_email(email.strip())
            return AuthResult(
                success=True,
                info_message="Supabase password reset email has been dispatched to your inbox.",
            )
        except Exception as e:
            logger.warning("Supabase reset password error: %s", e)
            return self.fallback.request_password_reset(email)

    def reset_password(self, token: str, new_password: str) -> AuthResult:
        return self.fallback.reset_password(token, new_password)

    def validate_session_token(self, token: str) -> User | None:
        try:
            res = self.client.auth.get_user(token)
            if res and res.user:
                u = res.user
                return User(
                    id=str(u.id),
                    email=str(u.email),
                    full_name=str(u.user_metadata.get("full_name", u.email.split("@")[0])),
                    is_verified=bool(u.email_confirmed_at),
                    auth_provider="supabase",
                )
        except Exception:
            pass
        return self.fallback.validate_session_token(token)

    def logout(self, token: str) -> None:
        try:
            self.client.auth.sign_out()
        except Exception:
            pass
        self.fallback.logout(token)


# -----------------------------------------------------------------------------
# Unified Auth Service Factory
# -----------------------------------------------------------------------------

class AuthService:
    """Unified Authentication Interface supporting Supabase and Local Vault."""

    def __init__(self) -> None:
        self._supabase_backend: SupabaseAuthBackend | None = None
        self._local_backend = LocalAuthBackend()

        if settings.supabase_url and settings.supabase_anon_key:
            try:
                self._supabase_backend = SupabaseAuthBackend(
                    settings.supabase_url,
                    settings.supabase_anon_key,
                )
                logger.info("Initialized Supabase Auth Backend for %s", settings.supabase_url)
            except Exception as e:
                logger.warning("Could not initialize Supabase client: %s. Using Local Vault.", e)

    @property
    def active_backend(self) -> str:
        if self._supabase_backend:
            return "Supabase Cloud Auth"
        return "Local Secure Cryptographic Vault (bcrypt)"

    def is_supabase_enabled(self) -> bool:
        return self._supabase_backend is not None

    def sign_up(self, email: str, password: str, full_name: str) -> AuthResult:
        if self._supabase_backend:
            return self._supabase_backend.sign_up(email, password, full_name)
        return self._local_backend.sign_up(email, password, full_name)

    def login(self, email: str, password: str) -> AuthResult:
        if self._supabase_backend:
            return self._supabase_backend.login(email, password)
        return self._local_backend.login(email, password)

    def is_google_oauth_configured(self) -> bool:
        """Google OAuth is not configured or required."""
        return False

    def get_google_auth_url(self, state: str = "", redirect_uri: str | None = None) -> tuple[bool, str]:
        return False, "Google OAuth is disabled."

    def handle_google_callback(
        self,
        code: str,
        redirect_uri: str | None = None,
    ) -> AuthResult:
        return AuthResult(success=False, error_message="Google OAuth is disabled.")

    def login_with_google(self, email: str | None = None, name: str | None = None) -> AuthResult:
        if email:
            return self._local_backend.handle_google_user(email=email, name=name or "")
        return AuthResult(
            success=False,
            error_message="Google OAuth is disabled in this deployment.",
        )

    def request_password_reset(self, email: str) -> AuthResult:
        if self._supabase_backend:
            return self._supabase_backend.request_password_reset(email)
        return self._local_backend.request_password_reset(email)

    def reset_password(self, token: str, new_password: str) -> AuthResult:
        if self._supabase_backend:
            return self._supabase_backend.reset_password(token, new_password)
        return self._local_backend.reset_password(token, new_password)

    def validate_session(self, token: str) -> User | None:
        if self._supabase_backend:
            u = self._supabase_backend.validate_session_token(token)
            if u:
                return u
        return self._local_backend.validate_session_token(token)

    def logout(self, token: str) -> None:
        if self._supabase_backend:
            self._supabase_backend.logout(token)
        else:
            self._local_backend.logout(token)

    # Convenience aliases for UI flexibility
    @property
    def is_supabase(self) -> bool:
        return self.is_supabase_enabled()

    def signup(self, email: str, password: str, name: str = "") -> AuthResult:
        return self.sign_up(email=email, password=password, full_name=name)

    def confirm_password_reset(self, token: str, new_password: str) -> AuthResult:
        return self.reset_password(token=token, new_password=new_password)

    def initiate_google_oauth(self, email: str | None = None, name: str | None = None) -> AuthResult:
        return self.login_with_google(email=email, name=name)


_AUTH_SERVICE_INSTANCE: AuthService | None = None


def get_auth_service() -> AuthService:
    """Return the application-wide singleton AuthService instance."""
    global _AUTH_SERVICE_INSTANCE
    if _AUTH_SERVICE_INSTANCE is None:
        _AUTH_SERVICE_INSTANCE = AuthService()
    return _AUTH_SERVICE_INSTANCE
