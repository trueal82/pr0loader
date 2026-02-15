"""
Authentication module for pr0loader.

Provides multiple authentication methods:
1. Browser cookie extraction (Firefox, Chrome, Edge)
2. Interactive login with captcha
3. Secure credential storage using system keyring
"""

import json
import logging
import os
import shutil
import sqlite3
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

logger = logging.getLogger(__name__)


class AuthCredentials:
    """Container for pr0gramm authentication credentials."""

    def __init__(self, pp: str, me: str):
        self.pp = pp
        self.me = me
        self._me_data: Optional[dict] = None

    @property
    def me_data(self) -> dict:
        """Parse the ME cookie JSON data."""
        if self._me_data is None:
            try:
                decoded = unquote(self.me)
                self._me_data = json.loads(decoded)
            except (json.JSONDecodeError, Exception):
                self._me_data = {}
        return self._me_data

    @property
    def username(self) -> Optional[str]:
        """Get username from ME cookie."""
        return self.me_data.get("n")

    @property
    def user_id(self) -> Optional[int]:
        """Get user ID from ME cookie."""
        return self.me_data.get("id")

    @property
    def is_verified(self) -> bool:
        """Check if account is age-verified (can access NSFW)."""
        # 'a' field typically indicates verification status
        return self.me_data.get("a", 0) >= 1

    def is_valid(self) -> bool:
        """Check if credentials appear valid (non-empty)."""
        return bool(self.pp and self.me)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {"pp": self.pp, "me": self.me}

    @classmethod
    def from_dict(cls, data: dict) -> "AuthCredentials":
        """Create from dictionary."""
        return cls(pp=data.get("pp", ""), me=data.get("me", ""))

    def __repr__(self):
        username = self.username or "unknown"
        return f"AuthCredentials(user={username}, verified={self.is_verified})"


class CookieExtractor(ABC):
    """Abstract base class for browser cookie extractors."""

    DOMAIN = ".pr0gramm.com"
    REQUIRED_COOKIES = ["pp", "me"]

    @abstractmethod
    def get_name(self) -> str:
        """Return the browser name."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this browser is installed."""
        pass

    @abstractmethod
    def extract_cookies(self) -> Optional[AuthCredentials]:
        """Extract pr0gramm cookies from the browser."""
        pass


class FirefoxCookieExtractor(CookieExtractor):
    """Extract cookies from Firefox browser."""

    def get_name(self) -> str:
        return "Firefox"

    def _get_profile_paths(self) -> list[Path]:
        """Get Firefox profile directories."""
        paths = []

        if os.name == "nt":  # Windows
            base = Path(os.environ.get("APPDATA", "")) / "Mozilla" / "Firefox" / "Profiles"
        elif os.name == "posix":
            if "darwin" in os.uname().sysname.lower():  # macOS
                base = Path.home() / "Library" / "Application Support" / "Firefox" / "Profiles"
            else:  # Linux
                base = Path.home() / ".mozilla" / "firefox"
        else:
            return paths

        if base.exists():
            # Find all profile directories
            for item in base.iterdir():
                if item.is_dir():
                    cookie_file = item / "cookies.sqlite"
                    if cookie_file.exists():
                        paths.append(item)

        return paths

    def is_available(self) -> bool:
        """Check if Firefox profiles exist."""
        return len(self._get_profile_paths()) > 0

    def extract_cookies(self) -> Optional[AuthCredentials]:
        """Extract pr0gramm cookies from Firefox."""
        profiles = self._get_profile_paths()

        for profile in profiles:
            cookies = self._extract_from_profile(profile)
            if cookies and cookies.is_valid():
                logger.info(f"Found valid cookies in Firefox profile: {profile.name}")
                return cookies

        return None

    def _extract_from_profile(self, profile_path: Path) -> Optional[AuthCredentials]:
        """Extract cookies from a specific Firefox profile."""
        cookie_file = profile_path / "cookies.sqlite"

        if not cookie_file.exists():
            return None

        # Copy database to temp file (Firefox may have it locked)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite") as tmp:
            tmp_path = Path(tmp.name)

        try:
            shutil.copy2(cookie_file, tmp_path)

            conn = sqlite3.connect(tmp_path)
            cursor = conn.cursor()

            # Firefox cookie table structure
            cursor.execute("""
                SELECT name, value FROM moz_cookies 
                WHERE host LIKE ? AND name IN (?, ?)
            """, (f"%{self.DOMAIN}", "pp", "me"))

            cookies = {row[0].lower(): row[1] for row in cursor.fetchall()}
            conn.close()

            if "pp" in cookies and "me" in cookies:
                return AuthCredentials(pp=cookies["pp"], me=cookies["me"])

        except Exception as e:
            logger.debug(f"Error extracting Firefox cookies: {e}")
        finally:
            tmp_path.unlink(missing_ok=True)

        return None


class ChromeCookieExtractor(CookieExtractor):
    """Extract cookies from Chrome/Chromium browsers."""

    def __init__(self, browser_name: str = "Chrome"):
        self.browser_name = browser_name

    def get_name(self) -> str:
        return self.browser_name

    def _get_cookie_paths(self) -> list[Path]:
        """Get Chrome cookie database paths."""
        paths = []

        if os.name == "nt":  # Windows
            local_app_data = Path(os.environ.get("LOCALAPPDATA", ""))

            browser_paths = {
                "Chrome": local_app_data / "Google" / "Chrome" / "User Data",
                "Edge": local_app_data / "Microsoft" / "Edge" / "User Data",
                "Brave": local_app_data / "BraveSoftware" / "Brave-Browser" / "User Data",
                "Chromium": local_app_data / "Chromium" / "User Data",
            }

            base = browser_paths.get(self.browser_name)
            if base and base.exists():
                # Check Default profile
                for profile in ["Default", "Profile 1", "Profile 2"]:
                    # Try both old and new cookie locations
                    for cookie_path in [
                        base / profile / "Network" / "Cookies",
                        base / profile / "Cookies",
                    ]:
                        if cookie_path.exists():
                            paths.append(cookie_path)

        elif os.name == "posix":
            home = Path.home()

            if "darwin" in os.uname().sysname.lower():  # macOS
                browser_paths = {
                    "Chrome": home / "Library" / "Application Support" / "Google" / "Chrome",
                    "Edge": home / "Library" / "Application Support" / "Microsoft Edge",
                    "Brave": home / "Library" / "Application Support" / "BraveSoftware" / "Brave-Browser",
                }
            else:  # Linux
                browser_paths = {
                    "Chrome": home / ".config" / "google-chrome",
                    "Chromium": home / ".config" / "chromium",
                    "Edge": home / ".config" / "microsoft-edge",
                    "Brave": home / ".config" / "BraveSoftware" / "Brave-Browser",
                }

            base = browser_paths.get(self.browser_name)
            if base and base.exists():
                for profile in ["Default", "Profile 1"]:
                    cookie_path = base / profile / "Cookies"
                    if cookie_path.exists():
                        paths.append(cookie_path)

        return paths

    def _get_encryption_key(self) -> Optional[bytes]:
        """Get the encryption key for Chrome cookies."""
        if os.name == "nt":
            return self._get_windows_key()
        elif os.name == "posix":
            return self._get_posix_key()
        return None

    def _get_windows_key(self) -> Optional[bytes]:
        """Get Chrome encryption key on Windows using DPAPI."""
        try:
            import base64
            import win32crypt

            if os.name != "nt":
                return None

            local_app_data = Path(os.environ.get("LOCALAPPDATA", ""))

            browser_paths = {
                "Chrome": local_app_data / "Google" / "Chrome" / "User Data",
                "Edge": local_app_data / "Microsoft" / "Edge" / "User Data",
                "Brave": local_app_data / "BraveSoftware" / "Brave-Browser" / "User Data",
            }

            base = browser_paths.get(self.browser_name)
            if not base:
                return None

            local_state_path = base / "Local State"
            if not local_state_path.exists():
                return None

            with open(local_state_path, "r", encoding="utf-8") as f:
                local_state = json.load(f)

            encrypted_key = base64.b64decode(
                local_state["os_crypt"]["encrypted_key"]
            )

            # Remove 'DPAPI' prefix
            encrypted_key = encrypted_key[5:]

            # Decrypt with DPAPI
            key = win32crypt.CryptUnprotectData(encrypted_key, None, None, None, 0)[1]
            return key

        except ImportError:
            logger.debug("win32crypt not available, cannot decrypt Chrome cookies")
            return None
        except Exception as e:
            logger.debug(f"Error getting Windows encryption key: {e}")
            return None

    def _get_posix_key(self) -> Optional[bytes]:
        """Get Chrome encryption key on Linux/macOS."""
        try:
            import secretstorage

            # Linux: Get key from Secret Service
            connection = secretstorage.dbus_init()
            collection = secretstorage.get_default_collection(connection)

            for item in collection.get_all_items():
                if item.get_label() == f"{self.browser_name} Safe Storage":
                    return item.get_secret()

            # Fallback: default key
            return b"peanuts"

        except ImportError:
            # Fallback for systems without secretstorage
            return b"peanuts"
        except Exception as e:
            logger.debug(f"Error getting POSIX encryption key: {e}")
            return b"peanuts"

    def _decrypt_value(self, encrypted_value: bytes, key: bytes) -> Optional[str]:
        """Decrypt a Chrome cookie value."""
        try:
            if os.name == "nt":
                return self._decrypt_windows(encrypted_value, key)
            else:
                return self._decrypt_posix(encrypted_value, key)
        except Exception as e:
            logger.debug(f"Decryption error: {e}")
            return None

    def _decrypt_windows(self, encrypted_value: bytes, key: bytes) -> Optional[str]:
        """Decrypt cookie on Windows."""
        try:
            from Crypto.Cipher import AES

            # Check for v10/v11 prefix
            if encrypted_value[:3] == b"v10" or encrypted_value[:3] == b"v11":
                # AES-GCM decryption
                nonce = encrypted_value[3:15]
                ciphertext = encrypted_value[15:-16]
                tag = encrypted_value[-16:]

                cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
                decrypted = cipher.decrypt_and_verify(ciphertext, tag)
                return decrypted.decode("utf-8")
            else:
                # Old DPAPI encryption
                import win32crypt
                decrypted = win32crypt.CryptUnprotectData(encrypted_value, None, None, None, 0)[1]
                return decrypted.decode("utf-8")

        except ImportError:
            logger.debug("PyCryptodome not available for AES decryption")
            return None
        except Exception as e:
            logger.debug(f"Windows decryption error: {e}")
            return None

    def _decrypt_posix(self, encrypted_value: bytes, key: bytes) -> Optional[str]:
        """Decrypt cookie on Linux/macOS."""
        try:
            from Crypto.Cipher import AES
            from Crypto.Protocol.KDF import PBKDF2

            # Check for v10/v11 prefix
            if encrypted_value[:3] in (b"v10", b"v11"):
                # Derive key using PBKDF2
                salt = b"saltysalt"
                iterations = 1 if os.uname().sysname == "Darwin" else 1
                derived_key = PBKDF2(key, salt, dkLen=16, count=iterations)

                # AES-CBC decryption
                iv = b" " * 16
                cipher = AES.new(derived_key, AES.MODE_CBC, iv)

                encrypted_data = encrypted_value[3:]
                decrypted = cipher.decrypt(encrypted_data)

                # Remove PKCS7 padding
                padding_len = decrypted[-1]
                decrypted = decrypted[:-padding_len]

                return decrypted.decode("utf-8")
            else:
                # Unencrypted or unknown format
                return encrypted_value.decode("utf-8", errors="ignore")

        except ImportError:
            logger.debug("PyCryptodome not available")
            return None
        except Exception as e:
            logger.debug(f"POSIX decryption error: {e}")
            return None

    def is_available(self) -> bool:
        """Check if Chrome cookie database exists."""
        return len(self._get_cookie_paths()) > 0

    def extract_cookies(self) -> Optional[AuthCredentials]:
        """Extract pr0gramm cookies from Chrome."""
        cookie_paths = self._get_cookie_paths()
        key = self._get_encryption_key()

        for cookie_path in cookie_paths:
            cookies = self._extract_from_db(cookie_path, key)
            if cookies and cookies.is_valid():
                logger.info(f"Found valid cookies in {self.browser_name}: {cookie_path}")
                return cookies

        return None

    def _extract_from_db(self, cookie_path: Path, key: Optional[bytes]) -> Optional[AuthCredentials]:
        """Extract cookies from a Chrome cookie database."""
        # Copy database to temp file (Chrome may have it locked)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite") as tmp:
            tmp_path = Path(tmp.name)

        try:
            shutil.copy2(cookie_path, tmp_path)

            conn = sqlite3.connect(tmp_path)
            cursor = conn.cursor()

            # Chrome cookie table structure
            cursor.execute("""
                SELECT name, encrypted_value, value FROM cookies 
                WHERE host_key LIKE ? AND name IN (?, ?)
            """, (f"%{self.DOMAIN}", "pp", "me"))

            cookies = {}
            for name, encrypted_value, plain_value in cursor.fetchall():
                name = name.lower()

                # Try plain value first (older Chrome versions)
                if plain_value:
                    cookies[name] = plain_value
                elif encrypted_value and key:
                    decrypted = self._decrypt_value(encrypted_value, key)
                    if decrypted:
                        cookies[name] = decrypted

            conn.close()

            if "pp" in cookies and "me" in cookies:
                return AuthCredentials(pp=cookies["pp"], me=cookies["me"])

        except Exception as e:
            logger.debug(f"Error extracting {self.browser_name} cookies: {e}")
        finally:
            tmp_path.unlink(missing_ok=True)

        return None


class EdgeCookieExtractor(ChromeCookieExtractor):
    """Extract cookies from Microsoft Edge (Chromium-based)."""

    def __init__(self):
        super().__init__(browser_name="Edge")


class BraveCookieExtractor(ChromeCookieExtractor):
    """Extract cookies from Brave browser."""

    def __init__(self):
        super().__init__(browser_name="Brave")


def get_all_extractors() -> list[CookieExtractor]:
    """Get all available cookie extractors."""
    return [
        FirefoxCookieExtractor(),
        ChromeCookieExtractor("Chrome"),
        EdgeCookieExtractor(),
        BraveCookieExtractor(),
    ]


def extract_cookies_from_browsers() -> Optional[AuthCredentials]:
    """
    Try to extract pr0gramm cookies from all installed browsers.

    Returns:
        AuthCredentials if found, None otherwise.
    """
    extractors = get_all_extractors()

    for extractor in extractors:
        if extractor.is_available():
            logger.info(f"Trying to extract cookies from {extractor.get_name()}...")
            try:
                credentials = extractor.extract_cookies()
                if credentials and credentials.is_valid():
                    logger.info(f"Successfully extracted cookies from {extractor.get_name()}")
                    return credentials
            except Exception as e:
                logger.debug(f"Error with {extractor.get_name()}: {e}")

    return None

