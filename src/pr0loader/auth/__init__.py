"""
Authentication module for pr0loader.

Provides:
- Browser cookie extraction (Firefox, Chrome, Edge, Brave)
- Interactive login with captcha
- Secure credential storage
"""

from pr0loader.auth.cookies import (
    AuthCredentials,
    CookieExtractor,
    FirefoxCookieExtractor,
    ChromeCookieExtractor,
    EdgeCookieExtractor,
    BraveCookieExtractor,
    get_all_extractors,
    extract_cookies_from_browsers,
)
from pr0loader.auth.login import (
    LoginClient,
    CaptchaChallenge,
    interactive_login_terminal,
)
from pr0loader.auth.storage import (
    CredentialStore,
    get_credential_store,
)
from pr0loader.auth.manager import (
    AuthManager,
    get_auth_manager,
)

__all__ = [
    # Credentials
    "AuthCredentials",
    # Cookie extraction
    "CookieExtractor",
    "FirefoxCookieExtractor",
    "ChromeCookieExtractor",
    "EdgeCookieExtractor",
    "BraveCookieExtractor",
    "get_all_extractors",
    "extract_cookies_from_browsers",
    # Login
    "LoginClient",
    "CaptchaChallenge",
    "interactive_login_terminal",
    # Storage
    "CredentialStore",
    "get_credential_store",
    # Manager
    "AuthManager",
    "get_auth_manager",
]

