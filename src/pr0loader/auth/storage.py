"""
Secure credential storage using system keyring.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from pr0loader.auth.cookies import AuthCredentials

logger = logging.getLogger(__name__)

SERVICE_NAME = "pr0loader"
ACCOUNT_NAME = "pr0gramm"


class CredentialStore:
    """Secure storage for pr0gramm credentials."""

    def __init__(self, auth_dir: Optional[Path] = None):
        """
        Initialize credential store.

        Args:
            auth_dir: Directory for file-based credential storage.
                     If None, uses settings.auth_dir or default.
        """
        self._auth_dir = auth_dir
        self._keyring_available = self._check_keyring()

    def _check_keyring(self) -> bool:
        """Check if keyring is available."""
        try:
            import keyring
            # Test if keyring backend is working
            keyring.get_keyring()
            return True
        except ImportError:
            logger.debug("keyring module not installed")
            return False
        except Exception as e:
            logger.debug(f"keyring not available: {e}")
            return False

    def save(self, credentials: AuthCredentials) -> bool:
        """
        Save credentials securely.

        Args:
            credentials: Credentials to save.

        Returns:
            True if saved successfully.
        """
        if self._keyring_available:
            return self._save_keyring(credentials)
        else:
            return self._save_file(credentials)

    def load(self) -> Optional[AuthCredentials]:
        """
        Load saved credentials.

        Returns:
            AuthCredentials if found, None otherwise.
        """
        if self._keyring_available:
            creds = self._load_keyring()
            if creds:
                return creds

        # Fallback to file
        return self._load_file()

    def delete(self) -> bool:
        """
        Delete saved credentials.

        Returns:
            True if deleted successfully.
        """
        success = True

        if self._keyring_available:
            success = self._delete_keyring() and success

        success = self._delete_file() and success

        return success

    def _save_keyring(self, credentials: AuthCredentials) -> bool:
        """Save to system keyring."""
        try:
            import keyring

            data = json.dumps(credentials.to_dict())
            keyring.set_password(SERVICE_NAME, ACCOUNT_NAME, data)
            logger.info("Credentials saved to system keyring")
            return True

        except Exception as e:
            logger.warning(f"Failed to save to keyring: {e}")
            return False

    def _load_keyring(self) -> Optional[AuthCredentials]:
        """Load from system keyring."""
        try:
            import keyring

            data = keyring.get_password(SERVICE_NAME, ACCOUNT_NAME)
            if data:
                cred_dict = json.loads(data)
                return AuthCredentials.from_dict(cred_dict)

        except Exception as e:
            logger.debug(f"Failed to load from keyring: {e}")

        return None

    def _delete_keyring(self) -> bool:
        """Delete from system keyring."""
        try:
            import keyring

            keyring.delete_password(SERVICE_NAME, ACCOUNT_NAME)
            return True

        except Exception as e:
            logger.debug(f"Failed to delete from keyring: {e}")
            return False

    def _get_auth_dir(self) -> Path:
        """Get the auth directory path."""
        if self._auth_dir:
            return self._auth_dir

        # Try to get from settings
        try:
            from pr0loader.config import load_settings
            settings = load_settings()
            return settings.auth_dir
        except Exception:
            pass

        # Fallback to user config directory
        import os
        if os.name == "nt":
            local_app_data = os.environ.get("LOCALAPPDATA")
            if local_app_data:
                return Path(local_app_data) / "pr0loader" / "auth"

        return Path.home() / ".local" / "share" / "pr0loader" / "auth"

    def _get_credentials_file(self) -> Path:
        """Get path to credentials file (fallback)."""
        auth_dir = self._get_auth_dir()
        auth_dir.mkdir(parents=True, exist_ok=True)
        return auth_dir / "credentials.json"

    def _save_file(self, credentials: AuthCredentials) -> bool:
        """Save to file (fallback when keyring unavailable)."""
        try:
            cred_file = self._get_credentials_file()

            with open(cred_file, "w") as f:
                json.dump(credentials.to_dict(), f)

            # Set restrictive permissions on Unix
            try:
                import os
                os.chmod(cred_file, 0o600)
            except:
                pass

            logger.info(f"Credentials saved to {cred_file}")
            logger.warning("Using file storage - install 'keyring' for better security")
            return True

        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
            return False

    def _load_file(self) -> Optional[AuthCredentials]:
        """Load from file (fallback)."""
        try:
            cred_file = self._get_credentials_file()

            if cred_file.exists():
                with open(cred_file, "r") as f:
                    cred_dict = json.load(f)
                return AuthCredentials.from_dict(cred_dict)

        except Exception as e:
            logger.debug(f"Failed to load credentials file: {e}")

        return None

    def _delete_file(self) -> bool:
        """Delete credentials file."""
        try:
            cred_file = self._get_credentials_file()
            if cred_file.exists():
                cred_file.unlink()
            return True
        except Exception as e:
            logger.debug(f"Failed to delete credentials file: {e}")
            return False


# Global store instance
_store: Optional[CredentialStore] = None


def get_credential_store() -> CredentialStore:
    """Get the global credential store instance."""
    global _store
    if _store is None:
        _store = CredentialStore()
    return _store

