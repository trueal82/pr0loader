#!/usr/bin/env python3
"""Quick test to demonstrate new functionality."""

import sys
import subprocess
import time

print("="*60)
print("Testing New Features")
print("="*60)

# Test 1: Verify imports
print("\n1. Testing imports...")
result = subprocess.run(
    ["uv", "run", "python", "-c",
     "from pr0loader.pipeline.download import AsyncDownloader; "
     "from scripts.benchmark_rate_limits import test_authenticated_comprehensive; "
     "print('✓ All imports OK')"],
    capture_output=True,
    text=True,
    cwd="/home/trueal/dev/pr0loader"
)
print(result.stdout)
if result.returncode != 0:
    print(f"✗ Import failed: {result.stderr}")
    sys.exit(1)

# Test 2: Show new test options
print("\n2. New benchmark test options:")
result = subprocess.run(
    ["uv", "run", "python", "scripts/benchmark_rate_limits.py", "--help"],
    capture_output=True,
    text=True,
    cwd="/home/trueal/dev/pr0loader"
)
lines = result.stdout.split('\n')
for line in lines:
    if 'test-authenticated' in line or 'test-sustained' in line or 'rate-limit' in line:
        print(f"  {line.strip()}")

# Test 3: Check credential loading
print("\n3. Testing credential loading...")
result = subprocess.run(
    ["uv", "run", "python", "-c",
     "from scripts.benchmark_rate_limits import load_cookies; "
     "cookies = load_cookies(); "
     "print(f'✓ Credentials loaded: {bool(cookies)}')"],
    capture_output=True,
    text=True,
    cwd="/home/trueal/dev/pr0loader"
)
print(result.stdout)

print("\n"+"="*60)
print("Summary")
print("="*60)
print("""
✓ Graceful shutdown added to download.py
  - Single Ctrl+C now stops gracefully
  - Partial results are saved and reported

✓ Comprehensive authenticated testing added
  - Use: uv run python scripts/benchmark_rate_limits.py --test-authenticated
  - Tests rates from 2-10 req/s to find threshold

✓ Sustained rate testing added
  - Use: uv run python scripts/benchmark_rate_limits.py --test-sustained --rate-limit 5.0
  - Validates limits over 2 minute window

✓ Credential loading improved
  - Automatically loads from keyring/file storage
  - Tests with proper authenticated content (flags=15)

NEXT STEPS:
1. Run authenticated testing before modifying download.py:
   uv run python scripts/benchmark_rate_limits.py --test-authenticated

2. Test graceful shutdown:
   pr0loader download
   <Press Ctrl+C after a few seconds>

See scripts/CHANGES_2026-02-17.md for full details.
""")

