"""Backoff strategies for rate limiting."""

from typing import Callable


def fibonacci_backoff(attempt: int, max_seconds: int = 300) -> int:
    """
    Calculate Fibonacci-based backoff time for rate limiting.

    Sequence: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233...

    Args:
        attempt: The attempt number (1-based).
        max_seconds: Maximum backoff time in seconds.

    Returns:
        The number of seconds to wait, capped at max_seconds.
    """
    if attempt <= 0:
        return 1

    a, b = 1, 1
    for _ in range(attempt - 1):
        a, b = b, a + b

    return min(a, max_seconds)


def exponential_backoff(attempt: int, base: int = 2, max_seconds: int = 300) -> int:
    """
    Calculate exponential backoff time.

    Args:
        attempt: The attempt number (1-based).
        base: Base for exponential calculation.
        max_seconds: Maximum backoff time in seconds.

    Returns:
        The number of seconds to wait, capped at max_seconds.
    """
    return min(base ** attempt, max_seconds)


class BackoffStrategy:
    """Backoff strategy manager for retries."""

    def __init__(
        self,
        strategy: Callable[[int], int] = fibonacci_backoff,
        max_seconds: int = 300
    ):
        self.strategy = strategy
        self.max_seconds = max_seconds
        self._attempt = 0

    def next(self) -> int:
        """Get the next backoff time and increment attempt counter."""
        self._attempt += 1
        return self.strategy(self._attempt, self.max_seconds)

    def reset(self):
        """Reset the attempt counter."""
        self._attempt = 0

    @property
    def attempt(self) -> int:
        """Current attempt number."""
        return self._attempt

