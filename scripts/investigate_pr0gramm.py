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

import requests
import json
import re
from urllib.parse import urljoin, urlparse, parse_qs
from pathlib import Path
import time

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
    print()

    # Flags appear to be a bitmask:
    # Bit 0 (1): SFW (Safe for Work)
    # Bit 1 (2): NSFW (Not Safe for Work)
    # Bit 2 (4): NSFL (Not Safe for Life - gore, etc.)
    # Bit 3 (8): NSFP (Not Safe for Public - political)
    #
    # So: flags=15 (1111 binary) = all content
    #     flags=1  (0001 binary) = SFW only
    #     flags=3  (0011 binary) = SFW + NSFW
    #     etc.

    flag_tests = [
        (1, "SFW only (bit 0)"),
        (2, "NSFW only (bit 1)"),
        (4, "NSFL only (bit 2)"),
        (8, "NSFP only (bit 3)"),
        (3, "SFW + NSFW (bits 0,1)"),
        (7, "SFW + NSFW + NSFL (bits 0,1,2)"),
        (9, "SFW + NSFP (bits 0,3)"),
        (15, "All content (bits 0,1,2,3)"),
    ]

    for flags, description in flag_tests:
        url = f"{API_URL}/items/get"
        try:
            resp = session.get(url, params={"flags": flags}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("items", [])

                # Analyze the flags of returned items
                item_flags = [item.get("flags", 0) for item in items[:20]]
                unique_flags = set(item_flags)

                print(f"  flags={flags:2d} ({flags:04b}) - {description}")
                print(f"    Items returned: {len(items)}")
                print(f"    Item flags seen: {sorted(unique_flags)}")

                # Check first item details
                if items:
                    first = items[0]
                    print(f"    First item: id={first.get('id')}, flags={first.get('flags')}, image={first.get('image', '')[:30]}")
            else:
                print(f"  flags={flags}: Status {resp.status_code}")
        except Exception as e:
            print(f"  flags={flags}: Error - {e}")

        time.sleep(0.5)  # Be nice to the server

    print()
    print("CONCLUSION:")
    print("  Content flags are a 4-bit bitmask:")
    print("    Bit 0 (1): SFW  - Safe for Work")
    print("    Bit 1 (2): NSFW - Not Safe for Work")
    print("    Bit 2 (4): NSFL - Not Safe for Life")
    print("    Bit 3 (8): NSFP - Not Safe for Public (political)")
    print("  ")
    print("  flags=15 (1111) = Show ALL content types")
    print("  User must be logged in to access NSFW/NSFL/NSFP content")


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

    # Run investigations
    session = investigate_login_page()
    investigate_api_endpoints(session)
    investigate_content_flags()
    investigate_login_flow()
    investigate_cookie_structure()
    investigate_new_feed()
    investigate_oauth_or_external_auth()
    investigate_browser_storage_paths()
    generate_recommendations()

    print_section("INVESTIGATION COMPLETE")
    print("Run this script again after changes to verify behavior.")
    print("Add your own tests by modifying the investigate_* functions.")


if __name__ == "__main__":
    main()


