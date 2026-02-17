# pr0loader Scripts

This folder contains development, testing, and investigation scripts for pr0loader.
These are **not** part of the main application but are useful for debugging, benchmarking, and understanding pr0gramm's behavior.

> **For AI Agents:** See [AGENTS.md](./AGENTS.md) for detailed findings, learnings, and how to use these scripts effectively.

## Quick Reference

| Script | Purpose | Run Command |
|--------|---------|-------------|
| `benchmark_rate_limits.py` | Find optimal download settings | `uv run python scripts/benchmark_rate_limits.py --test-authenticated` |
| `test_new_features.py` | Verify implementation features | `uv run python scripts/test_new_features.py` |
| `benchmark_fs_checks.py` | Compare filesystem scan strategies | `uv run python scripts/benchmark_fs_checks.py --method all` |
| `investigate_pr0gramm.py` | Investigate pr0gramm API/auth | `uv run python scripts/investigate_pr0gramm.py` |
| `test_head_backoff.py` | Test HEAD request backoff logic | `uv run python scripts/test_head_backoff.py` |
| `test_logging_buffer.py` | Verify log buffering works | `uv run python scripts/test_logging_buffer.py` |
| `demo_queue_monitor.py` | Preview queue visualization | `uv run python scripts/demo_queue_monitor.py` |

**See [AGENTS.md](./AGENTS.md) for detailed implementation findings and learnings.**

---

## Script Details

### 🚀 `benchmark_rate_limits.py` - Rate Limit Investigation

**Purpose:** Investigate pr0gramm's rate limiting (WAF) behavior and find optimal download settings that maximize throughput without triggering 429 errors.

**Background:** pr0gramm's WAF tolerates "typical browser behavior" but blocks bots/leeching. This script helps find the sweet spot.

**What it tests:**
- Different concurrency levels (1-50 simultaneous connections)
- Request delay patterns (fixed vs random delays)
- Burst patterns (download N files, pause, repeat)
- Header configurations (full browser headers vs minimal)
- Optimal settings that maximize throughput with <5% rate limiting

**Usage:**
```bash
# Full investigation suite (takes ~15 minutes)
uv run python scripts/benchmark_rate_limits.py --full

# Quick find optimal settings
uv run python scripts/benchmark_rate_limits.py --find-optimal

# Test specific concurrency level
uv run python scripts/benchmark_rate_limits.py --concurrency 10 --samples 50

# Test burst patterns only
uv run python scripts/benchmark_rate_limits.py --test-bursts

# Custom delay range test
uv run python scripts/benchmark_rate_limits.py --delay-min 0.1 --delay-max 0.3 --samples 100

# Save results to file
uv run python scripts/benchmark_rate_limits.py --full --output results.json
```

**Output:** JSON file with test results and recommendations for optimal settings.

---

### 📊 `benchmark_fs_checks.py` - Filesystem Check Strategies

**Purpose:** Compare different strategies for checking which files need to be downloaded (exists on disk vs in database).

**Strategies tested:**
1. **per_item_stat** - Check each file with `Path.exists()` (slow for millions of files)
2. **set_diff** - Build sets of DB paths and FS paths, then set difference (fast, used in production)
3. **dir_batched** - Group by directory and `listdir` once per directory

**Usage:**
```bash
# Test all strategies
uv run python scripts/benchmark_fs_checks.py --method all --max-items 200000

# Test specific strategy
uv run python scripts/benchmark_fs_checks.py --method set_diff --max-files 2000000

# Full run (no limits - use with caution on large datasets)
uv run python scripts/benchmark_fs_checks.py --method all
```

**When to use:** When optimizing the download pipeline's "which files to download" phase, especially with large datasets (millions of items).

---

### 🔍 `investigate_pr0gramm.py` - API & Auth Investigation

**Purpose:** Investigate how pr0gramm.com works - authentication flow, API endpoints, content flags, and session management.

**What it investigates:**
1. Login page structure and CSRF tokens
2. API endpoints (public and authenticated)
3. Content flags system (SFW/NSFW/NSFL/NSFP bitmask)
4. Login authentication flow
5. Cookie structure (PP and ME cookies)
6. OAuth/external auth options
7. Browser cookie storage paths
8. Frontend JavaScript flag mappings

**Usage:**
```bash
# Full investigation
uv run python scripts/investigate_pr0gramm.py

# Test authentication with stored cookies
uv run python scripts/investigate_pr0gramm.py --test-auth

# Test with specific cookies
uv run python scripts/investigate_pr0gramm.py --test-auth --pp "YOUR_PP_COOKIE" --me "YOUR_ME_COOKIE"

# Scan frontend JS for flag definitions
uv run python scripts/investigate_pr0gramm.py --frontend-flags

# Skip legacy tests, only run specific checks
uv run python scripts/investigate_pr0gramm.py --skip-legacy --test-auth
```

**When to use:** When debugging authentication issues or when pr0gramm changes their API/flags.

---

### 🔄 `test_head_backoff.py` - Backoff Strategy Test

**Purpose:** Smoke test for the HEAD request backoff strategy used when verifying file sizes.

**What it tests:**
- Simulates 429 rate limit responses
- Verifies Fibonacci backoff timing works correctly
- Ensures retry logic eventually succeeds

**Usage:**
```bash
uv run python scripts/test_head_backoff.py
```

**Expected output:** `OK: size=12345, calls=3, elapsed=X.XXs`

---

### 📝 `test_logging_buffer.py` - Log Buffer Test

**Purpose:** Verify that log buffering works correctly for the Rich Live display during downloads.

**What it tests:**
- Custom logging handler captures messages
- Buffer size limits work
- Log levels are respected

**Usage:**
```bash
uv run python scripts/test_logging_buffer.py
```

**Expected output:** Shows buffered messages and confirms buffering is working.

---

### 📺 `demo_queue_monitor.py` - Queue Visualization Demo

**Purpose:** Preview what the ASCII queue status visualization looks like at different fill levels.

**What it shows:**
- Queue bar rendering at various fill percentages
- Color coding (green/yellow/red zones)
- Layout of the status display

**Usage:**
```bash
uv run python scripts/demo_queue_monitor.py
```

**Note:** This is a visual demo only - it doesn't interact with any real queues.

---

## Adding New Scripts

When adding new scripts to this folder:

1. Add a docstring at the top explaining the script's purpose
2. Include usage examples in the docstring
3. Update this README with an entry for the new script
4. Use `uv run python scripts/yourscript.py` as the run command

## Requirements

All scripts use the pr0loader project's dependencies. Run from the project root:

```bash
# Install dependencies
uv sync

# Run any script
uv run python scripts/<script_name>.py
```

Some scripts require valid authentication cookies (PP/ME) in your `.env` file:

```env
PP=your_pp_cookie_value
ME=your_me_cookie_value
```


