"""
Interactive login with captcha support.
"""

import base64
import io
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import requests

from pr0loader.auth.cookies import AuthCredentials

logger = logging.getLogger(__name__)

# API URLs
API_URL = "https://pr0gramm.com/api"
BASE_URL = "https://pr0gramm.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
    "Referer": "https://pr0gramm.com/",
    "Origin": "https://pr0gramm.com",
}


class CaptchaChallenge:
    """Represents a captcha challenge from pr0gramm."""

    def __init__(self, token: str, image_data: bytes):
        self.token = token
        self.image_data = image_data

    def save_image(self, path: Path) -> Path:
        """Save captcha image to file."""
        path.write_bytes(self.image_data)
        return path

    def get_image_base64(self) -> str:
        """Get captcha image as base64 string."""
        return base64.b64encode(self.image_data).decode("utf-8")

    def display_in_terminal(self):
        """Display captcha as ASCII art in terminal (best effort)."""
        try:
            from PIL import Image

            # Load image
            img = Image.open(io.BytesIO(self.image_data))

            # Resize for terminal display
            width = 60
            aspect_ratio = img.height / img.width
            height = int(width * aspect_ratio * 0.5)  # 0.5 because chars are taller than wide
            img = img.resize((width, height))

            # Convert to grayscale
            img = img.convert("L")

            # ASCII characters from dark to light
            chars = " .:-=+*#%@"

            # Convert to ASCII
            pixels = list(img.getdata())
            ascii_art = ""
            for i, pixel in enumerate(pixels):
                if i % width == 0 and i > 0:
                    ascii_art += "\n"
                char_idx = int(pixel / 256 * len(chars))
                char_idx = min(char_idx, len(chars) - 1)
                ascii_art += chars[char_idx]

            print(ascii_art)

        except ImportError:
            print("[Captcha image cannot be displayed - PIL not available]")
            print(f"[Image saved to temp file or use Gradio UI for visual captcha]")


class LoginClient:
    """Client for pr0gramm login flow."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def get_captcha(self) -> Optional[CaptchaChallenge]:
        """
        Get a new captcha challenge.

        Returns:
            CaptchaChallenge or None if failed.
        """
        try:
            resp = self.session.get(f"{API_URL}/user/captcha", timeout=30)
            resp.raise_for_status()

            data = resp.json()

            token = data.get("token")
            captcha_data = data.get("captcha", "")

            if not token:
                logger.error("No captcha token in response")
                return None

            # Captcha is base64 encoded image
            if captcha_data.startswith("data:image"):
                # Remove data URL prefix
                captcha_data = captcha_data.split(",", 1)[1]

            image_data = base64.b64decode(captcha_data)

            return CaptchaChallenge(token=token, image_data=image_data)

        except Exception as e:
            logger.error(f"Failed to get captcha: {e}")
            return None

    def login(
        self,
        username: str,
        password: str,
        captcha_token: str,
        captcha_solution: str,
    ) -> Tuple[bool, Optional[AuthCredentials], str]:
        """
        Perform login with credentials and solved captcha.

        Args:
            username: pr0gramm username
            password: pr0gramm password
            captcha_token: Token from get_captcha()
            captcha_solution: User's solution to the captcha

        Returns:
            Tuple of (success, credentials, error_message)
        """
        try:
            data = {
                "name": username,
                "password": password,
                "token": captcha_token,
                "captcha": captcha_solution,
            }

            resp = self.session.post(
                f"{API_URL}/user/login",
                data=data,
                timeout=30,
            )

            result = resp.json()

            if result.get("success"):
                # Extract cookies from session
                pp = self.session.cookies.get("pp", "")
                me = self.session.cookies.get("me", "")

                if pp and me:
                    credentials = AuthCredentials(pp=pp, me=me)
                    return True, credentials, ""
                else:
                    return False, None, "Login succeeded but cookies not received"

            else:
                error = result.get("error", "Unknown error")
                ban_info = result.get("ban")

                if ban_info:
                    reason = ban_info.get("reason", "Unknown")
                    until = ban_info.get("until", "Unknown")
                    return False, None, f"Account banned: {reason} (until {until})"

                if error == "invalidLogin":
                    return False, None, "Invalid username or password"
                elif error == "invalidCaptcha":
                    return False, None, "Invalid captcha solution"
                elif error == "rateLimited":
                    return False, None, "Rate limited - try again later"
                else:
                    return False, None, f"Login failed: {error}"

        except requests.RequestException as e:
            return False, None, f"Network error: {e}"
        except Exception as e:
            return False, None, f"Error: {e}"

    def verify_credentials(self, credentials: AuthCredentials) -> bool:
        """
        Verify that credentials are valid by making an authenticated request.

        Args:
            credentials: Credentials to verify

        Returns:
            True if credentials are valid.
        """
        try:
            session = requests.Session()
            session.headers.update(HEADERS)
            session.cookies.set("pp", credentials.pp, domain=".pr0gramm.com")
            session.cookies.set("me", credentials.me, domain=".pr0gramm.com")

            resp = session.get(f"{API_URL}/user/info", timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                return data.get("account") is not None

            return False

        except Exception as e:
            logger.debug(f"Credential verification failed: {e}")
            return False


def interactive_login_terminal() -> Optional[AuthCredentials]:
    """
    Perform interactive login in terminal.

    Returns:
        AuthCredentials if successful, None otherwise.
    """
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.panel import Panel

    console = Console()
    client = LoginClient()

    console.print(Panel(
        "[bold]pr0gramm Login[/bold]\n\n"
        "You'll need to solve a captcha to log in.",
        title="Authentication",
    ))

    max_attempts = 3

    for attempt in range(max_attempts):
        # Get captcha
        console.print("\n[cyan]Fetching captcha...[/cyan]")
        captcha = client.get_captcha()

        if not captcha:
            console.print("[red]Failed to get captcha[/red]")
            continue

        # Save captcha to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            captcha.save_image(tmp_path)

        console.print(f"\n[green]Captcha saved to: {tmp_path}[/green]")
        console.print("[dim]Open this file to view the captcha[/dim]\n")

        # Try ASCII display
        console.print("[dim]ASCII preview (may not be accurate):[/dim]")
        captcha.display_in_terminal()
        console.print()

        # Get credentials
        username = Prompt.ask("Username")
        password = Prompt.ask("Password", password=True)
        captcha_solution = Prompt.ask("Captcha solution")

        # Clean up temp file
        tmp_path.unlink(missing_ok=True)

        # Attempt login
        console.print("\n[cyan]Logging in...[/cyan]")
        success, credentials, error = client.login(
            username=username,
            password=password,
            captcha_token=captcha.token,
            captcha_solution=captcha_solution,
        )

        if success and credentials:
            console.print(f"[green]Login successful![/green]")
            console.print(f"[dim]Logged in as: {credentials.username}[/dim]")
            return credentials
        else:
            console.print(f"[red]{error}[/red]")

            if attempt < max_attempts - 1:
                console.print(f"[yellow]Attempt {attempt + 1}/{max_attempts} failed. Trying again...[/yellow]")

    console.print("[red]Login failed after maximum attempts[/red]")
    return None

