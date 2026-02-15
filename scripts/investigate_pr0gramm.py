"""
pr0gramm.com Investigation Script
=================================
This script investigates how pr0gramm.com works:
1. Authentication flow
2. Cookie structure
3. API endpoints
4. Content flags system
5. Session management

Run this script to gather information for implementing proper auth.
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests

# Base URLs
BASE_URL = "https://pr0gramm.com"
API_URL = "https://pr0gramm.com/api"
IMG_URL = "https://img.pr0gramm.com"

# User agent to look like a real browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
    "Referer": "https://pr0gramm.com/",
}


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_subsection(title: str):
    """Print a subsection header."""
    print(f"\n--- {title} ---\n")


def investigate_login_page():
    """Investigate the login page structure."""
    print_section("1. LOGIN PAGE INVESTIGATION")

    session = requests.Session()
    session.headers.update(HEADERS)

    # Get the main page first
    print("Fetching main page...")
    resp = session.get(BASE_URL)
    print(f"  Status: {resp.status_code}")
    print(f"  Cookies received: {dict(resp.cookies)}")

    # Check for CSRF tokens or nonce values
    print_subsection("Looking for CSRF/nonce tokens in HTML")

    # Common patterns for CSRF tokens
    csrf_patterns = [
        r'name=["\']?csrf["\']?\s+value=["\']?([^"\']+)',
        r'name=["\']?_token["\']?\s+value=["\']?([^"\']+)',
        r'data-csrf=["\']?([^"\']+)',
        r'"csrfToken":\s*["\']([^"\']+)',
        r'"nonce":\s*["\']([^"\']+)',
        r'__RequestVerificationToken.*?value=["\']([^"\']+)',
    ]

    for pattern in csrf_patterns:
        matches = re.findall(pattern, resp.text, re.IGNORECASE)
        if matches:
            print(f"  Found CSRF pattern: {pattern[:30]}... -> {matches[:3]}")

    # Look for login-related JavaScript
    print_subsection("Looking for login-related JS endpoints")

    js_patterns = [
        r'/api/user/login',
        r'/api/user/captcha',
        r'/api/user/register',
        r'login.*?url["\']?\s*[:=]\s*["\']([^"\']+)',
    ]

    for pattern in js_patterns:
        if re.search(pattern, resp.text, re.IGNORECASE):
            print(f"  Found: {pattern}")

    return session


def investigate_api_endpoints(session: requests.Session):
    """Investigate available API endpoints."""
    print_section("2. API ENDPOINTS INVESTIGATION")

    # Known endpoints to test
    endpoints = [
        # Public endpoints (no auth needed)
        ("/api/items/get", {"flags": 1}),  # SFW only
        ("/api/items/get", {"flags": 15}), # All content
        ("/api/items/get", {"flags": 1, "promoted": 1}),  # Top/promoted
        ("/api/tags/top", {}),
        ("/api/profile/info", {"name": "cha0s"}),  # Public profile

        # Auth-related endpoints
        ("/api/user/login", None),  # Just check if exists (OPTIONS)
        ("/api/user/captcha", None),
        ("/api/user/info", {}),  # Current user info
        ("/api/user/sync", {}),

        # Item info
        ("/api/items/info", {"itemId": 1}),
    ]

    for endpoint, params in endpoints:
        url = f"{BASE_URL}{endpoint}"
        try:
            if params is None:
                # Just check if endpoint exists with OPTIONS
                resp = session.options(url, timeout=10)
                print(f"  OPTIONS {endpoint}: {resp.status_code}")
                if resp.headers.get("Allow"):
                    print(f"    Allowed methods: {resp.headers['Allow']}")
            else:
                resp = session.get(url, params=params, timeout=10)
                print(f"  GET {endpoint} ({params}): {resp.status_code}")

                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        # Print structure without full data
                        if isinstance(data, dict):
                            print(f"    Keys: {list(data.keys())}")
                            if "error" in data:
                                print(f"    Error: {data['error']}")
                        elif isinstance(data, list):
                            print(f"    List with {len(data)} items")
                    except:
                        print(f"    Non-JSON response: {resp.text[:100]}")
                elif resp.status_code == 403:
                    print(f"    Forbidden - likely needs auth")
                elif resp.status_code == 400:
                    try:
                        print(f"    Bad request: {resp.json()}")
                    except:
                        pass
        except Exception as e:
            print(f"  {endpoint}: Error - {e}")


def investigate_content_flags():
    """Investigate the content flags system."""
    print_section("3. CONTENT FLAGS INVESTIGATION")

    session = requests.Session()
    session.headers.update(HEADERS)

    print("Testing different flag values to understand the bitmask...")
    print("NOTE: pr0gramm requires authentication for NSFW/NSFL/NSFP content!")
    print()

    # Extended flag tests - testing up to 5 bits (31) and beyond
    # Old system was 4-bit (max 15), new system might be 5-bit (max 31)
    flag_tests = [
        # Individual bits
        (1, "bit 0 (SFW?)"),
        (2, "bit 1 (NSFW?)"),
        (4, "bit 2 (NSFL?)"),
        (8, "bit 3 (NSFP?)"),
        (16, "bit 4 (NEW FLAG?)"),
        (32, "bit 5 (NEW FLAG?)"),
        # Common combinations
        (3, "bits 0,1"),
        (7, "bits 0,1,2"),
        (15, "bits 0-3 (old 'all content')"),
        (31, "bits 0-4 (possible new 'all content')"),
        (63, "bits 0-5"),
        # Other values to test
        (9, "bits 0,3"),
        (17, "bits 0,4"),
        (19, "bits 0,1,4"),
        (23, "bits 0,1,2,4"),
        (27, "bits 0,1,3,4"),
    ]

    working_flags = []
    failing_flags = []

    for flags, description in flag_tests:
        url = f"{API_URL}/items/get"
        try:
            resp = session.get(url, params={"flags": flags}, timeout=10)
            status = resp.status_code

            if status == 200:
                data = resp.json()
                items = data.get("items", [])
                item_flags = [item.get("flags", 0) for item in items[:20]]
                unique_flags = set(item_flags)

                print(f"  ✓ flags={flags:2d} ({flags:06b}) - {description}")
                print(f"      Status: {status}, Items: {len(items)}, Item flags seen: {sorted(unique_flags)}")
                working_flags.append((flags, description, len(items), sorted(unique_flags)))
            elif status == 403:
                print(f"  ✗ flags={flags:2d} ({flags:06b}) - {description}")
                print(f"      Status: 403 FORBIDDEN (needs auth or invalid)")
                failing_flags.append((flags, description, "403 Forbidden"))
            else:
                print(f"  ? flags={flags:2d} ({flags:06b}) - {description}")
                print(f"      Status: {status}")
                failing_flags.append((flags, description, f"Status {status}"))
        except Exception as e:
            print(f"  ! flags={flags:2d} ({flags:06b}) - {description}")
            print(f"      Error: {e}")
            failing_flags.append((flags, description, str(e)))

        time.sleep(0.3)  # Be nice to the server

    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print()
    print(f"  Working flags ({len(working_flags)}):")
    for flags, desc, count, seen in working_flags:
        print(f"    {flags:2d} ({flags:06b}): {count} items, flags seen: {seen}")
    print()
    print(f"  Failing flags ({len(failing_flags)}):")
    for flags, desc, reason in failing_flags:
        print(f"    {flags:2d} ({flags:06b}): {reason}")
    print()

    # Analyze which bits are required/optional
    if working_flags:
        print("  ANALYSIS:")
        # Find which bits are common in working flags
        all_working = [f[0] for f in working_flags]
        all_failing = [f[0] for f in failing_flags]

        for bit in range(6):
            bit_val = 1 << bit
            working_with_bit = [f for f in all_working if f & bit_val]
            working_without_bit = [f for f in all_working if not (f & bit_val)]
            failing_with_bit = [f for f in all_failing if f & bit_val]

            print(f"    Bit {bit} ({bit_val:2d}): works_with={len(working_with_bit)}, works_without={len(working_without_bit)}, fails_with={len(failing_with_bit)}")

    print()
    print("  CONCLUSION:")
    print("    - Without authentication: ONLY flags=1 (SFW) works")
    print("    - flags >= 32 returns 400 (invalid value)")
    print("    - Any flag that includes non-SFW content (2,4,8,16) requires authentication")
    print("    - The flag system appears unchanged, but auth is now strictly required")
    print()
    print("  RECOMMENDATION:")
    print("    - For unauthenticated: use flags=1 (SFW only)")
    print("    - For authenticated: use flags=15 (all SFW/NSFW/NSFL/NSFP)")
    print("    - Ensure valid PP and ME cookies are present for non-SFW content")


def investigate_login_flow():
    """Investigate the login authentication flow."""
    print_section("4. LOGIN FLOW INVESTIGATION")

    session = requests.Session()
    session.headers.update(HEADERS)

    # First, get the main page to establish session
    print("Step 1: Establish session...")
    resp = session.get(BASE_URL)
    print(f"  Initial cookies: {list(session.cookies.keys())}")

    # Check for captcha requirement
    print_subsection("Step 2: Check captcha endpoint")

    try:
        resp = session.get(f"{API_URL}/user/captcha", timeout=10)
        print(f"  Captcha endpoint status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Captcha response keys: {list(data.keys())}")
            if "token" in data:
                print(f"  Captcha token present (length: {len(data.get('token', ''))})")
    except Exception as e:
        print(f"  Captcha error: {e}")

    # Check login endpoint requirements
    print_subsection("Step 3: Probe login endpoint")

    # Try OPTIONS request
    try:
        resp = session.options(f"{API_URL}/user/login", timeout=10)
        print(f"  OPTIONS /api/user/login: {resp.status_code}")
        print(f"  Headers: {dict(resp.headers)}")
    except Exception as e:
        print(f"  OPTIONS error: {e}")

    # Try empty POST to see error message
    try:
        resp = session.post(f"{API_URL}/user/login", json={}, timeout=10)
        print(f"  POST /api/user/login (empty): {resp.status_code}")
        try:
            print(f"  Response: {resp.json()}")
        except:
            print(f"  Response: {resp.text[:200]}")
    except Exception as e:
        print(f"  POST error: {e}")

    # Try with dummy credentials to see required fields
    try:
        test_data = {"name": "test", "password": "test"}
        resp = session.post(f"{API_URL}/user/login", data=test_data, timeout=10)
        print(f"  POST /api/user/login (dummy creds): {resp.status_code}")
        try:
            print(f"  Response: {resp.json()}")
        except:
            print(f"  Response: {resp.text[:200]}")
    except Exception as e:
        print(f"  POST error: {e}")


def investigate_cookie_structure():
    """Investigate the cookie structure and what's needed for auth."""
    print_section("5. COOKIE STRUCTURE INVESTIGATION")

    print("Based on the current implementation, the cookies needed are:")
    print()
    print("  PP cookie:")
    print("    - Purpose: Session/authentication token")
    print("    - Format: Appears to be a long alphanumeric string")
    print("    - Likely: Session ID or JWT-like token")
    print()
    print("  ME cookie:")
    print("    - Purpose: User identity/preferences")
    print("    - Format: Likely JSON or base64 encoded user data")
    print("    - Contains: User ID, settings, possibly verification status")
    print()

    # Try to understand ME cookie format
    print_subsection("Analyzing typical ME cookie structure")

    # The ME cookie is typically URL-encoded JSON
    print("  ME cookie is typically URL-encoded JSON containing:")
    print("    - n: username")
    print("    - id: user ID")
    print("    - a: account flags/age verification status")
    print("    - pp: some hash/token")
    print("    - paid: premium status")
    print()


def investigate_new_feed():
    """Investigate the /new feed specifically."""
    print_section("6. /NEW FEED INVESTIGATION")

    session = requests.Session()
    session.headers.update(HEADERS)

    print("The /new feed shows latest uploads...")
    print()

    # The /new page uses the items/get API without 'promoted' param
    # and typically sorts by ID descending (newest first)

    try:
        # Get newest items
        resp = session.get(f"{API_URL}/items/get", params={"flags": 1}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("items", [])

            print("API call: GET /api/items/get?flags=1")
            print(f"  Returns {len(items)} items")
            print(f"  Response keys: {list(data.keys())}")

            if items:
                print()
                print("  Sample item structure:")
                first = items[0]
                for key, value in first.items():
                    val_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                    print(f"    {key}: {val_str}")

                print()
                print("  Pagination info:")
                print(f"    atEnd: {data.get('atEnd')}")
                print(f"    atStart: {data.get('atStart')}")

                # Check for 'older' parameter usage
                if len(items) > 0:
                    last_id = items[-1].get("id")
                    print(f"    Last item ID: {last_id}")
                    print(f"    To get older items: ?older={last_id}")

    except Exception as e:
        print(f"  Error: {e}")


def investigate_oauth_or_external_auth():
    """Check if there's OAuth or external authentication options."""
    print_section("7. OAUTH/EXTERNAL AUTH INVESTIGATION")

    session = requests.Session()
    session.headers.update(HEADERS)

    print("Checking for OAuth or external authentication endpoints...")
    print()

    oauth_endpoints = [
        "/api/oauth/authorize",
        "/api/oauth/token",
        "/api/auth/google",
        "/api/auth/facebook",
        "/oauth/authorize",
        "/login/oauth",
    ]

    for endpoint in oauth_endpoints:
        try:
            resp = session.get(f"{BASE_URL}{endpoint}", timeout=5, allow_redirects=False)
            print(f"  {endpoint}: {resp.status_code}")
            if resp.status_code in [301, 302, 303, 307, 308]:
                print(f"    Redirects to: {resp.headers.get('Location', 'unknown')}")
        except Exception as e:
            print(f"  {endpoint}: Error - {type(e).__name__}")


def investigate_browser_storage_paths():
    """Suggest where browser cookies might be stored."""
    print_section("8. BROWSER COOKIE STORAGE PATHS")

    print("Cookies could potentially be extracted from browser storage:")
    print()

    print("Chrome (Windows):")
    print("  %LOCALAPPDATA%\\Google\\Chrome\\User Data\\Default\\Cookies")
    print("  %LOCALAPPDATA%\\Google\\Chrome\\User Data\\Default\\Network\\Cookies")
    print("  Note: SQLite database, encrypted with DPAPI")
    print()

    print("Firefox (Windows):")
    print("  %APPDATA%\\Mozilla\\Firefox\\Profiles\\<profile>\\cookies.sqlite")
    print("  Note: SQLite database, not encrypted")
    print()

    print("Edge (Windows):")
    print("  %LOCALAPPDATA%\\Microsoft\\Edge\\User Data\\Default\\Cookies")
    print("  Note: Same format as Chrome")
    print()

    print("SECURITY CONSIDERATIONS:")
    print("  - Chrome/Edge cookies are encrypted with Windows DPAPI")
    print("  - Requires running as the same user who owns the browser profile")
    print("  - Firefox cookies are unencrypted but file may be locked")
    print("  - This approach is fragile and browser-version dependent")
    print()


def generate_recommendations():
    """Generate recommendations based on investigation."""
    print_section("9. RECOMMENDATIONS")

    print("OPTION A: Cookie Extraction from Browser")
    print("-" * 40)
    print("Pros:")
    print("  + Works with existing auth")
    print("  + User doesn't need to enter credentials in our tool")
    print("Cons:")
    print("  - Browser-specific implementation needed")
    print("  - Chrome cookies are encrypted (need DPAPI)")
    print("  - May break with browser updates")
    print("  - Security concerns (accessing browser data)")
    print()
    print("Implementation:")
    print("  1. Detect installed browsers")
    print("  2. Find cookie database files")
    print("  3. Extract pr0gramm.com cookies (PP, ME)")
    print("  4. For Chrome: decrypt using DPAPI (Windows-only)")
    print("  5. For Firefox: read directly from SQLite")
    print()

    print("OPTION B: Implement Proper Auth Flow")
    print("-" * 40)
    print("Pros:")
    print("  + Clean, proper implementation")
    print("  + Works on all platforms")
    print("  + More reliable long-term")
    print("Cons:")
    print("  - Need to handle captcha")
    print("  - User must enter credentials")
    print("  - pr0gramm might have rate limiting")
    print()
    print("Implementation:")
    print("  1. GET /api/user/captcha to get captcha challenge")
    print("  2. Display captcha to user (or use captcha solving service)")
    print("  3. POST /api/user/login with:")
    print("     - name: username")
    print("     - password: password")
    print("     - captcha: solved captcha token")
    print("  4. Extract PP and ME cookies from response")
    print("  5. Store securely (keyring, encrypted file)")
    print()

    print("OPTION C: Hybrid Approach (RECOMMENDED)")
    print("-" * 40)
    print("1. First, try to extract cookies from browser (easy path)")
    print("2. If that fails, fall back to login flow")
    print("3. Cache credentials securely using system keyring")
    print()
    print("Implementation priority:")
    print("  1. Firefox cookie extraction (easiest, unencrypted)")
    print("  2. Chrome cookie extraction (need DPAPI)")
    print("  3. Manual login flow with captcha")
    print()


def fetch_frontend_js_urls(session: requests.Session, max_scripts: int = 20) -> list[str]:
    """Fetch JS bundle URLs from the main page."""
    resp = session.get(BASE_URL, timeout=10)
    resp.raise_for_status()

    script_srcs = re.findall(r"<script[^>]+src=[\"']([^\"']+)[\"']", resp.text)
    js_urls = []

    for src in script_srcs:
        if src.endswith(".js"):
            js_urls.append(urljoin(BASE_URL, src))

    return js_urls[:max_scripts]


def extract_flag_mapping_from_js(js_text: str) -> dict[str, int]:
    """Extract possible flag mappings from a JS bundle via regex heuristics."""
    mapping: dict[str, int] = {}

    patterns = [
        r"\b(SFW|NSFW|NSFL|NSFP)\b\s*[:=]\s*(\d+)",
        r"\b(sfw|nsfw|nsfl|nsfp)\b\s*[:=]\s*(\d+)",
        r"\b(SFW|NSFW|NSFL|NSFP)\b\s*[:=]\s*1\s*<<\s*(\d+)",
        r"\b(sfw|nsfw|nsfl|nsfp)\b\s*[:=]\s*1\s*<<\s*(\d+)",
    ]

    for pattern in patterns:
        for name, value in re.findall(pattern, js_text):
            key = name.upper()
            number = int(value)
            if "<<" in pattern:
                number = 1 << number
            mapping[key] = number

    return mapping


def investigate_frontend_flags(session: requests.Session, max_scripts: int = 20) -> Optional[dict[str, int]]:
    """Investigate frontend JS to infer how flags are calculated."""
    print_section("3A. FRONTEND FLAGS INVESTIGATION")
    print("Fetching frontend JS bundles to infer flag calculation...")

    try:
        js_urls = fetch_frontend_js_urls(session, max_scripts=max_scripts)
    except Exception as e:
        print(f"  Failed to fetch main page JS list: {e}")
        return None

    if not js_urls:
        print("  No JS bundles found on the main page.")
        return None

    print(f"  Found {len(js_urls)} JS bundles. Scanning...")

    aggregated: dict[str, list[int]] = {}

    for js_url in js_urls:
        try:
            resp = session.get(js_url, timeout=10)
            if resp.status_code != 200:
                print(f"  Skipping {js_url} (status {resp.status_code})")
                continue

            mapping = extract_flag_mapping_from_js(resp.text)
            if mapping:
                print(f"  Matches in: {js_url}")
                for key, value in mapping.items():
                    aggregated.setdefault(key, []).append(value)
                    print(f"    {key} = {value}")
        except Exception as e:
            print(f"  Failed to scan {js_url}: {e}")

    if not aggregated:
        print("  No flag mappings found in JS bundles.")
        return None

    # Build a consensus mapping (most common value per key)
    consensus: dict[str, int] = {}
    for key, values in aggregated.items():
        counts: dict[int, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        consensus_value = max(counts.items(), key=lambda item: item[1])[0]
        consensus[key] = consensus_value

    print("\n  Consensus mapping from frontend:")
    for key in sorted(consensus.keys()):
        print(f"    {key} = {consensus[key]}")

    return consensus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Investigate pr0gramm frontend and API behavior")
    parser.add_argument(
        "--frontend-flags",
        action="store_true",
        help="Scan frontend JS bundles to infer content flags",
    )
    parser.add_argument(
        "--max-scripts",
        type=int,
        default=20,
        help="Maximum JS bundles to scan for flag mappings (default: 20)",
    )
    parser.add_argument(
        "--skip-legacy",
        action="store_true",
        help="Skip legacy investigations and only run requested checks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output file for frontend flag mapping",
    )
    parser.add_argument(
        "--test-auth",
        action="store_true",
        help="Test flags with authentication from .env file",
    )
    parser.add_argument(
        "--pp",
        type=str,
        default=None,
        help="PP cookie value for authenticated testing",
    )
    parser.add_argument(
        "--me",
        type=str,
        default=None,
        help="ME cookie value for authenticated testing",
    )
    return parser.parse_args()


def load_env_cookies() -> tuple[Optional[str], Optional[str]]:
    """Load PP and ME cookies from .env file or stored credentials."""
    pp = None
    me = None

    # First try .env file
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("PP=") or line.startswith("pp="):
                    pp = line.split("=", 1)[1].strip().strip('"').strip("'")
                elif line.startswith("ME=") or line.startswith("me="):
                    me = line.split("=", 1)[1].strip().strip('"').strip("'")

    # If not found, try stored credentials
    if not pp or not me:
        # Check default data dir
        data_dir = None
        if env_path.exists():
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("DATA_DIR"):
                        data_dir = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

        if data_dir:
            creds_path = Path(data_dir) / "auth" / "credentials.json"
            if creds_path.exists():
                try:
                    with open(creds_path, "r") as f:
                        creds = json.load(f)
                        pp = pp or creds.get("pp")
                        me = me or creds.get("me")
                        print(f"  Loaded credentials from: {creds_path}")
                except Exception as e:
                    print(f"  Failed to load credentials: {e}")

    return pp, me


def investigate_authenticated_flags(pp: Optional[str] = None, me: Optional[str] = None):
    """Test content flags with authentication."""
    print_section("10. AUTHENTICATED FLAGS TEST")

    # Try to load from .env if not provided
    if not pp or not me:
        env_pp, env_me = load_env_cookies()
        pp = pp or env_pp
        me = me or env_me

    if not pp or not me:
        print("  No PP/ME cookies provided and none found in .env file.")
        print("  Cannot test authenticated requests.")
        print()
        print("  To test, either:")
        print("    1. Add PP and ME to .env file")
        print("    2. Run with --pp <value> --me <value>")
        return

    print(f"  Using cookies: PP={pp[:20]}..., ME={me[:30]}...")
    print()

    session = requests.Session()
    session.headers.update(HEADERS)
    session.cookies.update({
        "pp": pp,
        "me": me,
    })

    # Test user/sync to verify auth works
    print("  Testing authentication...")
    try:
        resp = session.get(f"{API_URL}/user/sync", timeout=10)
        print(f"    Response status: {resp.status_code}")
        try:
            data = resp.json()
            print(f"    Response: {json.dumps(data, indent=2)[:500]}")
        except:
            print(f"    Response text: {resp.text[:500]}")

        if resp.status_code == 200:
            data = resp.json()
            print(f"    ✓ Auth valid! Response keys: {list(data.keys())}")
            if "loggedIn" in data:
                print(f"    ✓ Logged in: {data.get('loggedIn')}")
        else:
            print(f"    ✗ Auth failed: Status {resp.status_code}")
            # Don't return - let's try flags anyway to see what happens
    except Exception as e:
        print(f"    ✗ Auth test error: {e}")

    print()
    print("  Testing flags with authentication...")
    print()

    flag_tests = [
        (1, "SFW only"),
        (3, "SFW + NSFW"),
        (7, "SFW + NSFW + NSFL"),
        (9, "SFW + NSFP"),
        (15, "All 4-bit flags (old 'all content')"),
        (31, "All 5-bit flags"),
    ]

    for flags, description in flag_tests:
        url = f"{API_URL}/items/get"
        try:
            resp = session.get(url, params={"flags": flags}, timeout=10)
            status = resp.status_code

            if status == 200:
                data = resp.json()
                items = data.get("items", [])
                item_flags = [item.get("flags", 0) for item in items[:20]]
                unique_flags = set(item_flags)

                print(f"  ✓ flags={flags:2d} ({flags:06b}) - {description}")
                print(f"      Status: {status}, Items: {len(items)}, Item flags seen: {sorted(unique_flags)}")
            else:
                print(f"  ✗ flags={flags:2d} ({flags:06b}) - {description}")
                print(f"      Status: {status}")
                try:
                    err = resp.json()
                    print(f"      Error: {err}")
                except:
                    pass
        except Exception as e:
            print(f"  ! flags={flags:2d} ({flags:06b}) - {description}")
            print(f"      Error: {e}")

        time.sleep(0.3)

    print()
    print("  ANALYSIS:")
    print("    If flags=15 works with auth: The issue is missing/invalid authentication")
    print("    If flags=15 fails with auth: pr0gramm changed their flag system")


def write_json_output(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main():
    """Run all investigations."""
    print("""
+==============================================================+
|         pr0gramm.com Authentication Investigation            |
|                                                              |
|  This script investigates how pr0gramm.com works to help     |
|  implement proper authentication for pr0loader.              |
+==============================================================+
    """)

    args = parse_args()

    if args.skip_legacy and not args.frontend_flags and not args.test_auth:
        print("Nothing to do. Use --frontend-flags or --test-auth to run specific checks.")
        return

    # Run investigations
    session = investigate_login_page()

    if not args.skip_legacy:
        investigate_api_endpoints(session)
        investigate_content_flags()
        investigate_login_flow()
        investigate_cookie_structure()
        investigate_new_feed()
        investigate_oauth_or_external_auth()
        investigate_browser_storage_paths()
        generate_recommendations()

    frontend_mapping = None
    if args.frontend_flags:
        frontend_mapping = investigate_frontend_flags(session, max_scripts=args.max_scripts)
        if args.output and frontend_mapping:
            write_json_output(args.output, frontend_mapping)
            print(f"\nSaved frontend flag mapping to: {args.output}")

    if args.test_auth:
        investigate_authenticated_flags(pp=args.pp, me=args.me)

    print_section("INVESTIGATION COMPLETE")
    if frontend_mapping:
        print("Frontend flag mapping captured. Review for flag calculation changes.")
    print("Run this script again after changes to verify behavior.")
    print("Add your own tests by modifying the investigate_* functions.")


if __name__ == "__main__":
    main()

