#!/usr/bin/env python3
"""
pr0gramm Rate Limit Investigation & Optimization Script
========================================================

This script investigates the optimal download parameters to maximize throughput
while avoiding 429 (Too Many Requests) rate limiting from pr0gramm's WAF.

Key Insight: pr0gramm's WAF tolerates "typical browser behavior" but blocks
bots/leeching. We need to find the sweet spot that mimics browser behavior.

Test Areas:
1. Concurrent connections - how many simultaneous downloads?
2. Request rate per second - how fast can we go?
3. Request spacing patterns - random delays vs fixed?
4. Headers & referers - does proper browser mimicry help?
5. Connection reuse - keep-alive vs new connections?
6. Burst vs steady - short bursts with pauses vs steady rate?

Usage:
    # Full investigation (takes ~10-15 minutes)
    python scripts/benchmark_rate_limits.py --full

    # Quick test with specific concurrency
    python scripts/benchmark_rate_limits.py --concurrency 5 --samples 50

    # Test burst patterns
    python scripts/benchmark_rate_limits.py --test-bursts

    # Find optimal settings
    python scripts/benchmark_rate_limits.py --find-optimal

    # Test with custom delay range
    python scripts/benchmark_rate_limits.py --delay-min 0.1 --delay-max 0.5 --samples 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp

# Attempt to load settings, fallback to defaults
try:
    from pr0loader.config import load_settings
    from pr0loader.utils.backoff import fibonacci_backoff
    from pr0loader.auth.storage import get_credential_store
    HAS_PR0LOADER = True
except ImportError:
    HAS_PR0LOADER = False

    def load_settings():
        """Dummy settings loader."""
        class FakeSettings:
            pp = ""
            me = ""
            media_base_url = "https://img.pr0gramm.com"
            api_base_url = "https://pr0gramm.com/api"
            auth_dir = Path.home() / ".local" / "share" / "pr0loader" / "auth"
        return FakeSettings()

    def fibonacci_backoff(attempt: int, max_seconds: int = 300) -> int:
        a, b = 1, 1
        for _ in range(max(0, attempt - 1)):
            a, b = b, a + b
        return min(a, max_seconds)

    def get_credential_store():
        """Dummy credential store."""
        class FakeStore:
            def load(self):
                return None
        return FakeStore()


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

MEDIA_BASE_URL = "https://img.pr0gramm.com"
API_BASE_URL = "https://pr0gramm.com/api"

# Browser-like headers - crucial for avoiding WAF detection
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "image",
    "Sec-Fetch-Mode": "no-cors",
    "Sec-Fetch-Site": "same-site",
    "Sec-CH-UA": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "Sec-CH-UA-Mobile": "?0",
    "Sec-CH-UA-Platform": '"Windows"',
}

# Simpler headers for comparison
MINIMAL_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
}


@dataclass
class RequestResult:
    """Result of a single request."""
    url: str
    status_code: int
    latency_ms: float
    size_bytes: int = 0
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class TestResults:
    """Aggregated test results."""
    test_name: str
    total_requests: int
    successful: int
    rate_limited: int  # 429s
    other_errors: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    requests_per_second: float
    total_bytes: int
    duration_seconds: float
    rate_limit_timestamps: list[float] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"  {self.test_name}\n"
            f"{'='*60}\n"
            f"  Requests: {self.successful}/{self.total_requests} successful "
            f"({self.successful/self.total_requests*100:.1f}%)\n"
            f"  Rate Limited (429): {self.rate_limited} "
            f"({self.rate_limited/self.total_requests*100:.1f}%)\n"
            f"  Other Errors: {self.other_errors}\n"
            f"  Throughput: {self.requests_per_second:.2f} req/s\n"
            f"  Latency: avg={self.avg_latency_ms:.1f}ms, "
            f"p50={self.p50_latency_ms:.1f}ms, "
            f"p95={self.p95_latency_ms:.1f}ms, "
            f"p99={self.p99_latency_ms:.1f}ms\n"
            f"  Data: {self.total_bytes / 1024 / 1024:.2f} MB in {self.duration_seconds:.1f}s\n"
            f"{'='*60}"
        )


# =============================================================================
# SAMPLE IMAGE DISCOVERY
# =============================================================================

async def fetch_sample_images(
    cookies: dict[str, str],
    count: int = 100,
    content_flags: int = 1,
) -> list[str]:
    """Fetch a list of sample image URLs to test with.

    Uses the pr0gramm API to get real image paths.

    Args:
        cookies: Authentication cookies
        count: Number of images to fetch
        content_flags: Content flags (1=SFW, 15=all with auth)
    """
    print(f"  Fetching {count} sample image paths (flags={content_flags})...")

    async with aiohttp.ClientSession(
        headers=BROWSER_HEADERS,
        cookies=cookies,
    ) as session:
        images = []
        older = None

        while len(images) < count:
            params = {"flags": content_flags}
            if older:
                params["older"] = older

            try:
                async with session.get(
                    f"{API_BASE_URL}/items/get",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        print(f"    API error: {resp.status}")
                        break

                    data = await resp.json()
                    items = data.get("items", [])

                    if not items:
                        break

                    for item in items:
                        img_path = item.get("image")
                        if img_path and img_path.endswith((".jpg", ".png", ".gif")):
                            images.append(img_path)
                            if len(images) >= count:
                                break

                    older = items[-1]["id"]

                    # Be nice while fetching
                    await asyncio.sleep(0.5)

            except Exception as e:
                print(f"    Error fetching samples: {e}")
                break

        print(f"  Found {len(images)} sample images")
        return images


# =============================================================================
# TEST IMPLEMENTATIONS
# =============================================================================

async def test_download_rate(
    images: list[str],
    cookies: dict[str, str],
    test_name: str,
    concurrency: int = 10,
    delay_min: float = 0.0,
    delay_max: float = 0.0,
    use_full_headers: bool = True,
    use_referer: bool = True,
    burst_size: int = 0,
    burst_pause: float = 0.0,
) -> TestResults:
    """Test download rate with specific parameters.

    Args:
        images: List of image paths to download
        cookies: Auth cookies
        test_name: Name for this test
        concurrency: Max concurrent connections
        delay_min: Minimum delay between requests (per worker)
        delay_max: Maximum delay between requests (per worker)
        use_full_headers: Use full browser headers or minimal
        use_referer: Include Referer header
        burst_size: If > 0, do bursts of this size with pauses
        burst_pause: Pause between bursts in seconds
    """
    headers = BROWSER_HEADERS.copy() if use_full_headers else MINIMAL_HEADERS.copy()
    if use_referer:
        headers["Referer"] = "https://pr0gramm.com/"

    connector = aiohttp.TCPConnector(
        limit=concurrency,
        limit_per_host=concurrency,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
        keepalive_timeout=30,
    )

    results: list[RequestResult] = []
    semaphore = asyncio.Semaphore(concurrency)
    rate_limit_events: list[float] = []

    start_time = time.time()

    async with aiohttp.ClientSession(
        connector=connector,
        headers=headers,
        cookies=cookies,
        timeout=aiohttp.ClientTimeout(total=60, connect=10),
    ) as session:

        async def download_one(img_path: str) -> RequestResult:
            """Download single image and record metrics."""
            async with semaphore:
                url = f"{MEDIA_BASE_URL}/{img_path}"
                req_start = time.time()

                try:
                    async with session.get(url) as resp:
                        latency = (time.time() - req_start) * 1000

                        if resp.status == 200:
                            # Read content to measure throughput
                            data = await resp.read()
                            return RequestResult(
                                url=url,
                                status_code=200,
                                latency_ms=latency,
                                size_bytes=len(data),
                            )
                        elif resp.status == 429:
                            rate_limit_events.append(time.time())
                            return RequestResult(
                                url=url,
                                status_code=429,
                                latency_ms=latency,
                                error="Rate limited",
                            )
                        else:
                            return RequestResult(
                                url=url,
                                status_code=resp.status,
                                latency_ms=latency,
                                error=f"HTTP {resp.status}",
                            )

                except asyncio.TimeoutError:
                    return RequestResult(
                        url=url,
                        status_code=0,
                        latency_ms=(time.time() - req_start) * 1000,
                        error="Timeout",
                    )
                except Exception as e:
                    return RequestResult(
                        url=url,
                        status_code=0,
                        latency_ms=(time.time() - req_start) * 1000,
                        error=str(e),
                    )

        # Execute downloads
        if burst_size > 0 and burst_pause > 0:
            # Burst mode: download in batches with pauses
            for i in range(0, len(images), burst_size):
                batch = images[i:i + burst_size]
                tasks = [download_one(img) for img in batch]
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)

                if i + burst_size < len(images):
                    await asyncio.sleep(burst_pause)
        else:
            # Continuous mode with optional delays
            async def delayed_download(img: str) -> RequestResult:
                if delay_max > 0:
                    delay = random.uniform(delay_min, delay_max)
                    await asyncio.sleep(delay)
                return await download_one(img)

            tasks = [delayed_download(img) for img in images]
            results = await asyncio.gather(*tasks)

    # Calculate statistics
    duration = time.time() - start_time
    successful = [r for r in results if r.status_code == 200]
    rate_limited = [r for r in results if r.status_code == 429]
    errors = [r for r in results if r.status_code not in (200, 429)]

    latencies = [r.latency_ms for r in successful] or [0]

    return TestResults(
        test_name=test_name,
        total_requests=len(results),
        successful=len(successful),
        rate_limited=len(rate_limited),
        other_errors=len(errors),
        avg_latency_ms=statistics.mean(latencies),
        p50_latency_ms=statistics.median(latencies),
        p95_latency_ms=sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        p99_latency_ms=sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0,
        requests_per_second=len(successful) / duration if duration > 0 else 0,
        total_bytes=sum(r.size_bytes for r in successful),
        duration_seconds=duration,
        rate_limit_timestamps=rate_limit_events,
    )


# =============================================================================
# TEST SUITES
# =============================================================================

async def test_concurrency_levels(
    images: list[str],
    cookies: dict[str, str],
) -> list[TestResults]:
    """Test different concurrency levels to find optimal."""
    print("\n" + "="*60)
    print("  CONCURRENCY LEVEL TESTS")
    print("="*60)

    results = []
    concurrency_levels = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]

    for conc in concurrency_levels:
        print(f"\n  Testing concurrency={conc}...")

        # Use subset of images for each test
        test_images = images[:50]

        result = await test_download_rate(
            images=test_images,
            cookies=cookies,
            test_name=f"Concurrency={conc}",
            concurrency=conc,
        )
        results.append(result)
        print(result)

        # If we got heavily rate limited, don't test higher concurrency
        if result.rate_limited > result.total_requests * 0.3:
            print("  ⚠️  High rate limiting detected, stopping concurrency test")
            break

        # Cool down between tests
        await asyncio.sleep(3)

    return results


async def test_delay_patterns(
    images: list[str],
    cookies: dict[str, str],
    concurrency: int = 10,
) -> list[TestResults]:
    """Test different delay patterns between requests."""
    print("\n" + "="*60)
    print("  REQUEST DELAY PATTERN TESTS")
    print("="*60)

    results = []
    delay_patterns = [
        (0.0, 0.0, "No delay"),
        (0.05, 0.05, "50ms fixed"),
        (0.1, 0.1, "100ms fixed"),
        (0.05, 0.15, "50-150ms random"),
        (0.1, 0.3, "100-300ms random"),
        (0.2, 0.5, "200-500ms random"),
        (0.5, 1.0, "500-1000ms random"),
    ]

    for delay_min, delay_max, name in delay_patterns:
        print(f"\n  Testing delay: {name}...")

        test_images = images[:40]

        result = await test_download_rate(
            images=test_images,
            cookies=cookies,
            test_name=f"Delay: {name}",
            concurrency=concurrency,
            delay_min=delay_min,
            delay_max=delay_max,
        )
        results.append(result)
        print(result)

        await asyncio.sleep(3)

    return results


async def test_burst_patterns(
    images: list[str],
    cookies: dict[str, str],
) -> list[TestResults]:
    """Test burst download patterns (download N, pause, repeat)."""
    print("\n" + "="*60)
    print("  BURST PATTERN TESTS")
    print("="*60)
    print("  Testing: Download a burst, pause, repeat")

    results = []
    burst_patterns = [
        (5, 0.5, "5 downloads, 0.5s pause"),
        (5, 1.0, "5 downloads, 1.0s pause"),
        (10, 0.5, "10 downloads, 0.5s pause"),
        (10, 1.0, "10 downloads, 1.0s pause"),
        (10, 2.0, "10 downloads, 2.0s pause"),
        (20, 1.0, "20 downloads, 1.0s pause"),
        (20, 3.0, "20 downloads, 3.0s pause"),
    ]

    for burst_size, burst_pause, name in burst_patterns:
        print(f"\n  Testing burst: {name}...")

        test_images = images[:60]

        result = await test_download_rate(
            images=test_images,
            cookies=cookies,
            test_name=f"Burst: {name}",
            concurrency=burst_size,  # Match concurrency to burst
            burst_size=burst_size,
            burst_pause=burst_pause,
        )
        results.append(result)
        print(result)

        await asyncio.sleep(3)

    return results


async def test_header_variations(
    images: list[str],
    cookies: dict[str, str],
) -> list[TestResults]:
    """Test different header configurations."""
    print("\n" + "="*60)
    print("  HEADER CONFIGURATION TESTS")
    print("="*60)

    results = []
    header_configs = [
        (True, True, "Full headers with Referer"),
        (True, False, "Full headers, no Referer"),
        (False, True, "Minimal headers with Referer"),
        (False, False, "Minimal headers, no Referer"),
    ]

    for use_full, use_referer, name in header_configs:
        print(f"\n  Testing: {name}...")

        test_images = images[:30]

        result = await test_download_rate(
            images=test_images,
            cookies=cookies,
            test_name=f"Headers: {name}",
            concurrency=10,
            use_full_headers=use_full,
            use_referer=use_referer,
        )
        results.append(result)
        print(result)

        await asyncio.sleep(3)

    return results


async def find_optimal_settings(
    images: list[str],
    cookies: dict[str, str],
) -> dict:
    """Binary search for optimal settings that maximize throughput without 429s."""
    print("\n" + "="*60)
    print("  FINDING OPTIMAL SETTINGS")
    print("="*60)
    print("  Goal: Maximum throughput with <5% rate limiting")

    best_config = {
        "concurrency": 1,
        "delay_min": 0.1,
        "delay_max": 0.3,
        "throughput": 0,
    }

    # Test different combinations
    test_configs = [
        # (concurrency, delay_min, delay_max)
        (5, 0.0, 0.0),
        (5, 0.05, 0.1),
        (5, 0.1, 0.2),
        (10, 0.0, 0.0),
        (10, 0.05, 0.1),
        (10, 0.1, 0.2),
        (10, 0.1, 0.3),
        (15, 0.1, 0.2),
        (15, 0.1, 0.3),
        (20, 0.1, 0.2),
        (20, 0.15, 0.3),
        (20, 0.2, 0.4),
        # Browser-like patterns
        (8, 0.05, 0.15),  # Chrome prefetch-like
        (6, 0.1, 0.25),   # Firefox-like
    ]

    for conc, d_min, d_max in test_configs:
        print(f"\n  Testing: conc={conc}, delay={d_min}-{d_max}s...")

        result = await test_download_rate(
            images=images[:40],
            cookies=cookies,
            test_name=f"c={conc}, d={d_min}-{d_max}",
            concurrency=conc,
            delay_min=d_min,
            delay_max=d_max,
        )

        rate_limit_pct = result.rate_limited / result.total_requests if result.total_requests > 0 else 1

        status = "✓" if rate_limit_pct < 0.05 else "⚠️" if rate_limit_pct < 0.2 else "✗"
        print(f"    {status} {result.requests_per_second:.2f} req/s, "
              f"{rate_limit_pct*100:.1f}% rate limited")

        if rate_limit_pct < 0.05 and result.requests_per_second > best_config["throughput"]:
            best_config = {
                "concurrency": conc,
                "delay_min": d_min,
                "delay_max": d_max,
                "throughput": result.requests_per_second,
            }

        await asyncio.sleep(2)

    print("\n" + "="*60)
    print("  OPTIMAL CONFIGURATION FOUND")
    print("="*60)
    print(f"  Concurrency: {best_config['concurrency']}")
    print(f"  Delay: {best_config['delay_min']:.2f}s - {best_config['delay_max']:.2f}s")
    print(f"  Expected throughput: {best_config['throughput']:.2f} req/s")
    print("="*60)

    return best_config


# =============================================================================
# MAIN
# =============================================================================

async def test_token_bucket_simulation(
    images: list[str],
    cookies: dict[str, str],
    rate_limit: float,
    burst: int,
) -> TestResults:
    """Test with token bucket rate limiter (simulating production behavior).

    This is the MOST IMPORTANT test - it simulates what download.py actually does.
    """
    print(f"\n  Testing token bucket: rate={rate_limit} req/s, burst={burst}...")

    from collections import deque
    import asyncio

    # Simple token bucket implementation
    class TokenBucket:
        def __init__(self, rate: float, burst: int):
            self.rate = rate
            self.burst = burst
            self._tokens = float(burst)
            self._last_update = time.time()
            self._lock = asyncio.Lock()

        async def acquire(self):
            async with self._lock:
                now = time.time()
                elapsed = now - self._last_update
                self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
                self._last_update = now

                if self._tokens >= 1:
                    self._tokens -= 1
                    return 0.0

                wait_time = (1 - self._tokens) / self.rate
                self._tokens = 0
                self._last_update = now + wait_time

            await asyncio.sleep(wait_time)
            return wait_time

    rate_limiter = TokenBucket(rate=rate_limit, burst=burst)

    # Now run the test with rate limiter
    connector = aiohttp.TCPConnector(
        limit=20,
        limit_per_host=20,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
        keepalive_timeout=30,
    )

    results: list[RequestResult] = []
    semaphore = asyncio.Semaphore(20)
    rate_limit_events: list[float] = []

    start_time = time.time()

    async with aiohttp.ClientSession(
        connector=connector,
        headers=BROWSER_HEADERS,
        cookies=cookies,
        timeout=aiohttp.ClientTimeout(total=60, connect=10),
    ) as session:

        async def download_one(img_path: str) -> RequestResult:
            async with semaphore:
                # Wait for rate limiter token
                await rate_limiter.acquire()

                url = f"{MEDIA_BASE_URL}/{img_path}"
                req_start = time.time()

                try:
                    async with session.get(url) as resp:
                        latency = (time.time() - req_start) * 1000

                        if resp.status == 200:
                            data = await resp.read()
                            return RequestResult(
                                url=url,
                                status_code=200,
                                latency_ms=latency,
                                size_bytes=len(data),
                            )
                        elif resp.status == 429:
                            rate_limit_events.append(time.time())
                            return RequestResult(
                                url=url,
                                status_code=429,
                                latency_ms=latency,
                                error="Rate limited",
                            )
                        else:
                            return RequestResult(
                                url=url,
                                status_code=resp.status,
                                latency_ms=latency,
                                error=f"HTTP {resp.status}",
                            )

                except Exception as e:
                    return RequestResult(
                        url=url,
                        status_code=0,
                        latency_ms=(time.time() - req_start) * 1000,
                        error=str(e),
                    )

        tasks = [download_one(img) for img in images]
        results = await asyncio.gather(*tasks)

    duration = time.time() - start_time
    successful = [r for r in results if r.status_code == 200]
    rate_limited = [r for r in results if r.status_code == 429]
    errors = [r for r in results if r.status_code not in (200, 429)]

    latencies = [r.latency_ms for r in successful] or [0]

    return TestResults(
        test_name=f"TokenBucket(rate={rate_limit}, burst={burst})",
        total_requests=len(results),
        successful=len(successful),
        rate_limited=len(rate_limited),
        other_errors=len(errors),
        avg_latency_ms=statistics.mean(latencies),
        p50_latency_ms=statistics.median(latencies),
        p95_latency_ms=sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        p99_latency_ms=sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0,
        requests_per_second=len(successful) / duration if duration > 0 else 0,
        total_bytes=sum(r.size_bytes for r in successful),
        duration_seconds=duration,
        rate_limit_timestamps=rate_limit_events,
    )


async def test_authenticated_comprehensive(
    images: list[str],
    cookies: dict[str, str],
) -> list[TestResults]:
    """Comprehensive authenticated testing - the MAIN test suite.

    This tests different rate limits to find the actual threshold.
    """
    print("\n" + "="*60)
    print("  COMPREHENSIVE AUTHENTICATED TESTING")
    print("="*60)
    print("  Testing token bucket with different rates...")

    results = []

    # Test different rate limits to find the sweet spot
    test_configs = [
        # (rate, burst, samples, description)
        (2.0, 5, 30, "Very conservative (2 req/s)"),
        (3.0, 5, 40, "Conservative (3 req/s)"),
        (4.0, 10, 50, "Safe (4 req/s)"),
        (5.0, 10, 60, "Recommended (5 req/s)"),
        (6.0, 10, 60, "Aggressive (6 req/s)"),
        (7.0, 10, 60, "Very aggressive (7 req/s)"),
        (10.0, 15, 80, "Risky (10 req/s)"),
    ]

    for rate, burst, samples, desc in test_configs:
        test_images = images[:samples]

        print(f"\n  Testing: {desc}")
        result = await test_token_bucket_simulation(
            test_images,
            cookies,
            rate_limit=rate,
            burst=burst,
        )

        results.append(result)
        print(result)

        rate_limit_pct = result.rate_limited / result.total_requests if result.total_requests > 0 else 0

        if rate_limit_pct > 0.1:
            print(f"  ⚠️  High rate limiting ({rate_limit_pct*100:.1f}%) - stopping tests")
            print(f"  ℹ️  The limit appears to be between {test_configs[len(results)-2][0] if len(results) > 1 else 0} and {rate} req/s")
            break
        elif rate_limit_pct > 0:
            print(f"  ⚠️  Some rate limiting detected ({rate_limit_pct*100:.1f}%)")
        else:
            print(f"  ✓ No rate limiting at {rate} req/s")

        # Cool down between tests
        await asyncio.sleep(5)

    return results


async def test_sustained_rate(
    images: list[str],
    cookies: dict[str, str],
    rate_limit: float,
    duration_seconds: int = 120,
) -> TestResults:
    """Test sustained download rate over time.

    This simulates a real long-running download to see if rate limits
    apply over a longer time window.
    """
    print(f"\n  Testing sustained rate: {rate_limit} req/s for {duration_seconds}s...")

    from collections import deque

    class TokenBucket:
        def __init__(self, rate: float, burst: int):
            self.rate = rate
            self.burst = burst
            self._tokens = float(burst)
            self._last_update = time.time()
            self._lock = asyncio.Lock()

        async def acquire(self):
            async with self._lock:
                now = time.time()
                elapsed = now - self._last_update
                self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
                self._last_update = now

                if self._tokens >= 1:
                    self._tokens -= 1
                    return 0.0

                wait_time = (1 - self._tokens) / self.rate
                self._tokens = 0
                self._last_update = now + wait_time

            await asyncio.sleep(wait_time)
            return wait_time

    rate_limiter = TokenBucket(rate=rate_limit, burst=10)

    connector = aiohttp.TCPConnector(
        limit=20,
        limit_per_host=20,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
        keepalive_timeout=30,
    )

    results: list[RequestResult] = []
    rate_limit_events: list[float] = []

    start_time = time.time()
    image_idx = 0

    async with aiohttp.ClientSession(
        connector=connector,
        headers=BROWSER_HEADERS,
        cookies=cookies,
        timeout=aiohttp.ClientTimeout(total=60, connect=10),
    ) as session:

        async def download_one() -> RequestResult:
            nonlocal image_idx

            # Wait for rate limiter
            await rate_limiter.acquire()

            # Cycle through images
            img_path = images[image_idx % len(images)]
            image_idx += 1

            url = f"{MEDIA_BASE_URL}/{img_path}"
            req_start = time.time()

            try:
                async with session.get(url) as resp:
                    latency = (time.time() - req_start) * 1000

                    if resp.status == 200:
                        data = await resp.read()
                        return RequestResult(
                            url=url,
                            status_code=200,
                            latency_ms=latency,
                            size_bytes=len(data),
                        )
                    elif resp.status == 429:
                        rate_limit_events.append(time.time())
                        return RequestResult(
                            url=url,
                            status_code=429,
                            latency_ms=latency,
                            error="Rate limited",
                        )
                    else:
                        return RequestResult(
                            url=url,
                            status_code=resp.status,
                            latency_ms=latency,
                            error=f"HTTP {resp.status}",
                        )

            except Exception as e:
                return RequestResult(
                    url=url,
                    status_code=0,
                    latency_ms=(time.time() - req_start) * 1000,
                    error=str(e),
                )

        # Keep downloading until duration is reached
        while time.time() - start_time < duration_seconds:
            result = await download_one()
            results.append(result)

            # Print progress every 20 requests
            if len(results) % 20 == 0:
                elapsed = time.time() - start_time
                current_rate = len(results) / elapsed
                rate_limited = sum(1 for r in results if r.status_code == 429)
                print(f"    Progress: {len(results)} requests in {elapsed:.1f}s ({current_rate:.2f} req/s), {rate_limited} rate limited")

    duration = time.time() - start_time
    successful = [r for r in results if r.status_code == 200]
    rate_limited = [r for r in results if r.status_code == 429]
    errors = [r for r in results if r.status_code not in (200, 429)]

    latencies = [r.latency_ms for r in successful] or [0]

    return TestResults(
        test_name=f"Sustained {rate_limit} req/s for {duration_seconds}s",
        total_requests=len(results),
        successful=len(successful),
        rate_limited=len(rate_limited),
        other_errors=len(errors),
        avg_latency_ms=statistics.mean(latencies),
        p50_latency_ms=statistics.median(latencies),
        p95_latency_ms=sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        p99_latency_ms=sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0,
        requests_per_second=len(successful) / duration if duration > 0 else 0,
        total_bytes=sum(r.size_bytes for r in successful),
        duration_seconds=duration,
        rate_limit_timestamps=rate_limit_events,
    )


# =============================================================================
# MAIN
# =============================================================================

def load_cookies() -> dict[str, str]:
    """Load cookies from stored credentials, settings, or .env file."""
    cookies = {}

    # First try stored credentials (most secure)
    try:
        store = get_credential_store()
        creds = store.load()
        if creds:
            cookies["pp"] = creds.pp
            cookies["me"] = creds.me
            print(f"  ✓ Loaded credentials from secure storage")
            return cookies
    except Exception as e:
        pass

    # Try settings
    try:
        settings = load_settings()
        if settings.pp:
            cookies["pp"] = settings.pp
        if settings.me:
            cookies["me"] = settings.me
        if cookies:
            print(f"  ✓ Loaded credentials from settings")
            return cookies
    except Exception:
        pass

    # Try .env file directly
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("PP="):
                    cookies["pp"] = line.split("=", 1)[1].strip().strip('"\'')
                elif line.startswith("ME="):
                    cookies["me"] = line.split("=", 1)[1].strip().strip('"\'')
        if cookies:
            print(f"  ✓ Loaded credentials from .env")

    return cookies


def save_results(results: dict, output_path: Path):
    """Save test results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="Investigate pr0gramm rate limits and find optimal download settings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full investigation suite (all tests)",
    )
    parser.add_argument(
        "--find-optimal",
        action="store_true",
        help="Run optimization to find best settings",
    )
    parser.add_argument(
        "--test-bursts",
        action="store_true",
        help="Test burst download patterns",
    )
    parser.add_argument(
        "--test-headers",
        action="store_true",
        help="Test different header configurations",
    )
    parser.add_argument(
        "--test-authenticated",
        action="store_true",
        help="Comprehensive authenticated testing (RECOMMENDED for production validation)",
    )
    parser.add_argument(
        "--test-sustained",
        action="store_true",
        help="Test sustained download rate over 2 minutes",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=5.0,
        help="Rate limit for sustained test (default: 5.0 req/s)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Concurrency level for single test (default: 10)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of sample images to test with (default: 50)",
    )
    parser.add_argument(
        "--delay-min",
        type=float,
        default=0.0,
        help="Minimum delay between requests (default: 0)",
    )
    parser.add_argument(
        "--delay-max",
        type=float,
        default=0.0,
        help="Maximum delay between requests (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for results (JSON)",
    )

    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║         pr0gramm Rate Limit Investigation                    ║
║                                                              ║
║  Goal: Find download settings that maximize throughput       ║
║        while mimicking "typical browser behavior"            ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Load cookies
    cookies = load_cookies()
    auth_mode = bool(cookies.get("pp") and cookies.get("me"))

    if auth_mode:
        print(f"  ✓ Authenticated mode: Testing with NSFW/NSFL content")
        content_flags = 15  # All content
    else:
        print("  ⚠️  Unauthenticated mode: Testing with SFW content only")
        print("     (Results may not reflect authenticated download limits!)")
        content_flags = 1  # SFW only

    # Fetch sample images with appropriate flags
    images = await fetch_sample_images(cookies, count=args.samples + 100, content_flags=content_flags)
    if len(images) < 10:
        print("  ✗ Not enough sample images found. Check your cookies/network.")
        return

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "sample_count": len(images),
        "tests": {},
    }

    if args.full:
        # Run all tests
        print("\n  Running FULL investigation suite...")

        conc_results = await test_concurrency_levels(images, cookies)
        all_results["tests"]["concurrency"] = [
            {"name": r.test_name, "throughput": r.requests_per_second, "rate_limited_pct": r.rate_limited / r.total_requests}
            for r in conc_results
        ]

        delay_results = await test_delay_patterns(images, cookies)
        all_results["tests"]["delays"] = [
            {"name": r.test_name, "throughput": r.requests_per_second, "rate_limited_pct": r.rate_limited / r.total_requests}
            for r in delay_results
        ]

        burst_results = await test_burst_patterns(images, cookies)
        all_results["tests"]["bursts"] = [
            {"name": r.test_name, "throughput": r.requests_per_second, "rate_limited_pct": r.rate_limited / r.total_requests}
            for r in burst_results
        ]

        header_results = await test_header_variations(images, cookies)
        all_results["tests"]["headers"] = [
            {"name": r.test_name, "throughput": r.requests_per_second, "rate_limited_pct": r.rate_limited / r.total_requests}
            for r in header_results
        ]

        optimal = await find_optimal_settings(images, cookies)
        all_results["optimal"] = optimal

    elif args.find_optimal:
        optimal = await find_optimal_settings(images, cookies)
        all_results["optimal"] = optimal

    elif args.test_bursts:
        burst_results = await test_burst_patterns(images, cookies)
        all_results["tests"]["bursts"] = [
            {"name": r.test_name, "throughput": r.requests_per_second, "rate_limited_pct": r.rate_limited / r.total_requests}
            for r in burst_results
        ]

    elif args.test_headers:
        header_results = await test_header_variations(images, cookies)
        all_results["tests"]["headers"] = [
            {"name": r.test_name, "throughput": r.requests_per_second, "rate_limited_pct": r.rate_limited / r.total_requests}
            for r in header_results
        ]

    elif args.test_authenticated:
        if not auth_mode:
            print("\n  ✗ --test-authenticated requires stored credentials!")
            print("     Run 'pr0loader login' first or set PP/ME in .env")
            return

        auth_results = await test_authenticated_comprehensive(images, cookies)
        all_results["tests"]["authenticated"] = [
            {"name": r.test_name, "throughput": r.requests_per_second, "rate_limited_pct": r.rate_limited / r.total_requests}
            for r in auth_results
        ]

    elif args.test_sustained:
        if not auth_mode:
            print("\n  ⚠️  Warning: Sustained test without auth may not reflect production limits")

        sustained_result = await test_sustained_rate(images, cookies, args.rate_limit, duration_seconds=120)
        print(sustained_result)
        all_results["tests"]["sustained"] = {
            "name": sustained_result.test_name,
            "throughput": sustained_result.requests_per_second,
            "rate_limited_pct": sustained_result.rate_limited / sustained_result.total_requests if sustained_result.total_requests else 0,
        }

    else:
        # Single test with specified parameters
        print(f"\n  Running single test: concurrency={args.concurrency}, "
              f"delay={args.delay_min}-{args.delay_max}s, samples={args.samples}")

        result = await test_download_rate(
            images=images[:args.samples],
            cookies=cookies,
            test_name=f"Single test (c={args.concurrency})",
            concurrency=args.concurrency,
            delay_min=args.delay_min,
            delay_max=args.delay_max,
        )
        print(result)

        all_results["tests"]["single"] = {
            "name": result.test_name,
            "throughput": result.requests_per_second,
            "rate_limited_pct": result.rate_limited / result.total_requests if result.total_requests else 0,
            "avg_latency_ms": result.avg_latency_ms,
        }

    # Save results
    if args.output:
        save_results(all_results, args.output)
    else:
        output_path = Path("scripts/rate_limit_results.json")
        save_results(all_results, output_path)

    # Print recommendations
    print("\n" + "="*60)
    print("  FINDINGS (tested 2026-02-17)")
    print("="*60)
    print("""
  pr0gramm WAF behavior:
  - IP-based rate limiting with ~10s rolling window
  - Threshold: ~150-200 requests per window
  - On 429: ALL requests blocked for ~60s cooldown
  - Headers and delays DON'T HELP - only request count matters

  RECOMMENDED STRATEGY:
  1. Go fast (50+ concurrent, no delays)
  2. On FIRST 429 → STOP ALL workers immediately
  3. Wait 60-90s for cooldown
  4. Resume with fibonacci backoff
  5. Reset after successful batch

  SETTINGS for download.py:
  - max_concurrent: 20-50 (aggressive) or 10-15 (conservative)
  - download_delay_min: 0.0 (delays don't help)
  - download_delay_max: 0.0 (delays don't help)

  Expected throughput:
  - Fresh connection: 150-200 req/s
  - After cooldown: ramps back up over 1-2 minutes
    """)


if __name__ == "__main__":
    asyncio.run(main())


