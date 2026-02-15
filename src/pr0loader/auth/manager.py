"""
Authentication manager - orchestrates all auth methods.
"""

import logging
from typing import Optional, Tuple

from pr0loader.auth.cookies import AuthCredentials, extract_cookies_from_browsers, get_all_extractors
from pr0loader.auth.login import LoginClient, interactive_login_terminal
from pr0loader.auth.storage import get_credential_store

logger = logging.getLogger(__name__)


class AuthManager:
    """
    Manages authentication for pr0loader.

    Tries authentication methods in order:
    1. Stored credentials (keyring/file)
    2. Browser cookie extraction
    3. Interactive login
    """

    def __init__(self):
        self.store = get_credential_store()
        self.login_client = LoginClient()
        self._credentials: Optional[AuthCredentials] = None

    @property
    def credentials(self) -> Optional[AuthCredentials]:
        """Get current credentials."""
        return self._credentials

    @property
    def is_authenticated(self) -> bool:
        """Check if we have valid credentials."""
        return self._credentials is not None and self._credentials.is_valid()

    def get_credentials(self, auto_login: bool = True) -> Optional[AuthCredentials]:
        """
        Get credentials, trying all available methods.

        Args:
            auto_login: If True, try browser extraction and interactive login.

        Returns:
            AuthCredentials if successful, None otherwise.
        """
        # 1. Try stored credentials
        logger.info("Checking for stored credentials...")
        creds = self.store.load()
        if creds and creds.is_valid():
            if self._verify_credentials(creds):
                logger.info(f"Using stored credentials (user: {creds.username})")
                self._credentials = creds
                return creds
            else:
                logger.warning("Stored credentials are invalid/expired")

        if not auto_login:
            return None

        # 2. Try browser extraction
        logger.info("Trying browser cookie extraction...")
        creds = extract_cookies_from_browsers()
        if creds and creds.is_valid():
            if self._verify_credentials(creds):
                logger.info(f"Using browser credentials (user: {creds.username})")
                # Save for future use
                self.store.save(creds)
                self._credentials = creds
                return creds
            else:
                logger.warning("Browser credentials are invalid/expired")

        return None

    def login_interactive(self) -> Optional[AuthCredentials]:
        """
        Perform interactive login.

        Returns:
            AuthCredentials if successful, None otherwise.
        """
        creds = interactive_login_terminal()
        if creds and creds.is_valid():
            # Save credentials
            self.store.save(creds)
            self._credentials = creds
            return creds
        return None

    def login_with_credentials(
        self,
        username: str,
        password: str,
        captcha_token: str,
        captcha_solution: str,
    ) -> Tuple[bool, Optional[AuthCredentials], str]:
        """
        Login with provided credentials.

        Returns:
            Tuple of (success, credentials, error_message)
        """
        success, creds, error = self.login_client.login(
            username=username,
            password=password,
            captcha_token=captcha_token,
            captcha_solution=captcha_solution,
        )

        if success and creds:
            self.store.save(creds)
            self._credentials = creds

        return success, creds, error

    def extract_from_browser(self, browser: Optional[str] = None) -> Optional[AuthCredentials]:
        """
        Extract credentials from a specific browser or all browsers.

        Args:
            browser: Browser name (firefox, chrome, edge, brave) or None for all.

        Returns:
            AuthCredentials if successful, None otherwise.
        """
        if browser:
            browser = browser.lower()
            extractors = get_all_extractors()

            for extractor in extractors:
                if extractor.get_name().lower() == browser:
                    if extractor.is_available():
                        creds = extractor.extract_cookies()
                        if creds and creds.is_valid():
                            if self._verify_credentials(creds):
                                self.store.save(creds)
                                self._credentials = creds
                                return creds
                    else:
                        logger.warning(f"{browser} not found or not accessible")
                    break
            return None
        else:
            return extract_cookies_from_browsers()

    def logout(self) -> bool:
        """
        Clear stored credentials.

        Returns:
            True if successful.
        """
        self._credentials = None
        return self.store.delete()

    def get_status(self) -> dict:
        """
        Get current authentication status.

        Returns:
            Dictionary with status information.
        """
        # Check stored credentials
        stored = self.store.load()

        # Check available browsers
        browsers = []
        for extractor in get_all_extractors():
            if extractor.is_available():
                browsers.append(extractor.get_name())

        status = {
            "authenticated": self.is_authenticated,
            "stored_credentials": stored is not None and stored.is_valid(),
            "available_browsers": browsers,
        }

        if self._credentials:
            status["username"] = self._credentials.username
            status["user_id"] = self._credentials.user_id
            status["verified"] = self._credentials.is_verified
        elif stored:
            status["username"] = stored.username
            status["user_id"] = stored.user_id
            status["verified"] = stored.is_verified

        return status

    def _verify_credentials(self, credentials: AuthCredentials) -> bool:
        """Verify that credentials work with the API."""
        return self.login_client.verify_credentials(credentials)


# Global auth manager instance
_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Get the global auth manager instance."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager

