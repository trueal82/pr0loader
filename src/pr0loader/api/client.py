"""HTTP client for pr0gramm API with rate limiting and backoff."""

import logging
import time
from pathlib import Path
from typing import Optional

import requests

from pr0loader.config import Settings
from pr0loader.models import Item, ItemsResponse, ItemInfoResponse, Tag, Comment
from pr0loader.utils.backoff import BackoffStrategy, fibonacci_backoff

logger = logging.getLogger(__name__)


class APIClient:
    """HTTP client for pr0gramm API."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = self._create_session()
        self.backoff = BackoffStrategy(
            strategy=lambda a, m: fibonacci_backoff(a, m),
            max_seconds=settings.max_backoff_seconds
        )

    def _create_session(self) -> requests.Session:
        """Create and configure HTTP session."""
        session = requests.Session()
        session.cookies.update({
            "me": self.settings.me,
            "pp": self.settings.pp,
        })
        return session

    def _request(self, url: str) -> dict:
        """Make an HTTP request with retry logic."""
        tries = 0

        while tries < self.settings.max_retries:
            tries += 1
            try:
                logger.debug(f"Request {tries}/{self.settings.max_retries}: {url}")
                response = self.session.get(url, timeout=self.settings.request_timeout)
                response.raise_for_status()
                self.backoff.reset()
                return response.json()

            except requests.HTTPError as e:
                status_code = e.response.status_code if e.response else None

                if status_code == 429:
                    wait_time = self.backoff.next()
                    api_suggested = e.response.headers.get("Retry-After", "unknown")
                    logger.warning(
                        f"Rate limited (API suggests {api_suggested}s). "
                        f"Using Fibonacci backoff: {wait_time}s (attempt #{self.backoff.attempt})"
                    )
                    time.sleep(wait_time)
                elif status_code and 400 <= status_code < 500:
                    logger.error(f"Client error {status_code}: {e.response.reason}")
                    raise
                else:
                    logger.error(f"HTTP error: {e}")
                    time.sleep(2 ** min(tries, 6))

            except requests.RequestException as e:
                logger.error(f"Network error: {e}")
                time.sleep(2 ** min(tries, 6))

        raise Exception(f"Failed to fetch {url} after {self.settings.max_retries} attempts")

    def get_items(self, older_than: Optional[int] = None) -> ItemsResponse:
        """Fetch items from the API."""
        url = f"{self.settings.api_base_url}/items/get?flags={self.settings.content_flags}"
        if older_than:
            url += f"&older={older_than}"

        data = self._request(url)
        return ItemsResponse.model_validate(data)

    def get_item_info(self, item_id: int) -> ItemInfoResponse:
        """Fetch detailed info for an item."""
        url = (
            f"{self.settings.api_base_url}/items/info"
            f"?itemId={item_id}&flags={self.settings.content_flags}"
        )
        data = self._request(url)
        return ItemInfoResponse.model_validate(data)

    def get_highest_id(self) -> int:
        """Get the highest item ID from the API."""
        response = self.get_items()
        if response.items:
            return response.items[0].id
        raise Exception("No items found in API response")

    def download_media(self, filename: str, destination: Path) -> int:
        """
        Download a media file.

        Returns:
            Number of bytes downloaded.
        """
        url = f"{self.settings.media_base_url}/{filename}"
        destination.parent.mkdir(parents=True, exist_ok=True)

        tries = 0
        while tries < self.settings.max_retries:
            tries += 1
            try:
                response = self.session.get(
                    url,
                    timeout=self.settings.request_timeout,
                    stream=True
                )
                response.raise_for_status()

                size = 0
                with open(destination, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            size += len(chunk)

                self.backoff.reset()
                return size

            except requests.HTTPError as e:
                status_code = e.response.status_code if e.response else None

                if status_code == 429:
                    wait_time = self.backoff.next()
                    logger.warning(f"Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                elif status_code and 400 <= status_code < 500:
                    logger.error(f"Client error downloading {filename}: {status_code}")
                    raise
                else:
                    time.sleep(2 ** min(tries, 6))

            except requests.RequestException as e:
                logger.error(f"Network error downloading {filename}: {e}")
                time.sleep(2 ** min(tries, 6))

        raise Exception(f"Failed to download {filename} after {self.settings.max_retries} attempts")

    def get_remote_file_size(self, filename: str) -> Optional[int]:
        """
        Get the size of a remote file using HEAD request.

        Returns:
            File size in bytes, or None if unavailable.
        """
        url = f"{self.settings.media_base_url}/{filename}"

        try:
            response = self.session.head(url, timeout=self.settings.request_timeout)
            response.raise_for_status()

            content_length = response.headers.get("Content-Length")
            if content_length:
                return int(content_length)
            return None

        except requests.RequestException as e:
            logger.debug(f"HEAD request failed for {filename}: {e}")
            return None

    def needs_download(self, filename: str, local_path: Path) -> bool:
        """
        Check if a file needs to be downloaded.

        Compares local file size with remote file size via HEAD request.

        Returns:
            True if file should be downloaded (missing or size mismatch).
        """
        if not local_path.exists():
            return True

        local_size = local_path.stat().st_size
        remote_size = self.get_remote_file_size(filename)

        if remote_size is None:
            # Can't verify, assume OK if local file exists
            logger.debug(f"Cannot verify {filename}, keeping local file")
            return False

        if local_size != remote_size:
            logger.info(f"Size mismatch for {filename}: local={local_size}, remote={remote_size}")
            return True

        return False


