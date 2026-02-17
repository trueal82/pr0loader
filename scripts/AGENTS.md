# Agents Guide - pr0loader Scripts

This document is for AI agents (and humans) working on pr0loader. It explains the scripts, learnings, and how to use them effectively.

## Quick Context

pr0gramm.com is a German image board. We're building a downloader/ML pipeline for it. The main challenge is **rate limiting** - pr0gramm uses a WAF (Web Application Firewall) that blocks "bot-like" behavior.

---

## Implementation Status (2026-02-17)

✅ **Download Pipeline V2** - Complete rewrite with 40-80x faster data loading
✅ **Graceful Shutdown** - Single Ctrl+C cleanly stops all workers
✅ **Rate Limit Testing** - Comprehensive authenticated testing infrastructure
✅ **Validated Settings** - 5 req/s proven safe for sustained downloads

---

## Key Learnings (Rate Limiting)

### Test Results (2026-02-17) - VALIDATED WITH REAL TESTING

**CRITICAL: The WAF has a cumulative request limit over time!**

#### Authenticated Testing Results:

**Short bursts (< 1 minute):**
- 10 req/s: ✓ No rate limiting for ~700 requests (~72 seconds)
- 7 req/s: ✓ Works initially
- 5 req/s: ✓ Works initially

**Sustained (2 minutes):**
- **10 req/s: ✗ FAILS** - After ~700 requests, 31.8% rate limited
- **7 req/s: ✗ FAILS** - 76% rate limited (still in cooldown from previous test)
- **5 req/s: ✓ PERFECT** - 604 requests, 0% rate limited, 100% success

**Conclusion:** For sustained downloads (which is what we do), **5 req/s is the safe limit**.

### WAF Behavior Model

```
┌─────────────────────────────────────────────────────────────────┐
│  pr0gramm Rate Limiting Model (ACTUAL TESTED BEHAVIOR)          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  AUTHENTICATED (NSFW/NSFL):                                     │
│    - Short burst: Can handle up to 10 req/s for ~72 seconds    │
│    - Cumulative window: ~700 requests before throttling        │
│    - Sustained safe rate: 5 req/s (tested for 2 minutes)       │
│                                                                 │
│  RATE LIMIT TRIGGER:                                            │
│    - After ~700 requests at 10 req/s → 31.8% rate limited      │
│    - WAF has memory - limits persist across requests           │
│                                                                 │
│  COOLDOWN: ~90 seconds after hitting limit                      │
│                                                                 │
│  Strategy:                                                      │
│    1. Use 5 req/s sustained (token bucket)                      │
│    2. On 429 → STOP ALL workers immediately                     │
│    3. Wait 90+ seconds (fibonacci backoff)                      │
│    4. Resume at 5 req/s                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Recommended Settings

Based on ACTUAL sustained testing:

```env
# TESTED AND VALIDATED - use these settings:
DOWNLOAD_RATE_LIMIT = 5.0    # 5 req/s sustained (100% success rate)
DOWNLOAD_BURST = 10          # Allow small initial burst
MAX_PARALLEL_REQUESTS = 20   # Concurrency (rate limiter controls actual rate)
```

**DO NOT increase DOWNLOAD_RATE_LIMIT above 5.0** - testing shows higher rates fail on sustained downloads.

---

## Script Reference

### `benchmark_rate_limits.py`

**Purpose:** Find optimal download settings through empirical testing.

**IMPORTANT:** Always use `--test-authenticated` when testing for production settings!

**When to run:**
- **BEFORE modifying download.py rate limits** - validate settings first!
- When you suspect rate limit behavior has changed
- When testing from a new IP/network
- After pr0gramm updates their infrastructure

**Usage:**
```bash
# RECOMMENDED: Comprehensive authenticated testing (use this before changing download.py!)
uv run python scripts/benchmark_rate_limits.py --test-authenticated

# Test sustained rate over 2 minutes (validates long-running behavior)
uv run python scripts/benchmark_rate_limits.py --test-sustained --rate-limit 5.0

# Quick test - are we currently rate limited?
uv run python scripts/benchmark_rate_limits.py --concurrency 5 --samples 10

# Find optimal settings (wait for rate limit to clear first!)
uv run python scripts/benchmark_rate_limits.py --find-optimal

# Full investigation (takes ~15 min, will trigger rate limits)
uv run python scripts/benchmark_rate_limits.py --full
```

**Reading results:**
- `rate_limited_pct > 0.05` = too aggressive
- `rate_limited_pct = 0` = safe setting
- `throughput` shows actual req/s achieved
- **Always test with authentication** - unauthenticated results are misleading!

### `test_new_features.py`

**Purpose:** Verify that implemented features work correctly.

**When to run:**
- After making changes to download.py
- To validate graceful shutdown functionality
- To test credentials loading from storage

**Usage:**
```bash
# Run all verification tests
uv run python scripts/test_new_features.py
```

**Tests:**
- Import verification
- Credential loading from storage
- Graceful shutdown with Ctrl+C
- Token bucket rate limiter simulation

### `investigate_pr0gramm.py`

**Purpose:** Investigate pr0gramm API, auth flow, and content flags.

**When to run:**
- When auth is broken
- When content flags seem wrong
- When API behavior changes

**Usage:**
```bash
# Full investigation
uv run python scripts/investigate_pr0gramm.py

# Test if your cookies work
uv run python scripts/investigate_pr0gramm.py --test-auth

# Check frontend flag definitions
uv run python scripts/investigate_pr0gramm.py --frontend-flags
```

### `benchmark_fs_checks.py`

**Purpose:** Optimize filesystem scanning for large datasets.

**When to run:**
- When optimizing download phase 1 (load data)
- When working with millions of files

**Usage:**
```bash
# Compare strategies
uv run python scripts/benchmark_fs_checks.py --method all --max-items 100000
```

### `test_head_backoff.py`

**Purpose:** Smoke test for backoff logic.

**When to run:**
- After modifying backoff code
- As part of CI/testing

### `test_logging_buffer.py`

**Purpose:** Verify log buffering for Rich Live display.

### `demo_queue_monitor.py`

**Purpose:** Preview queue visualization (development aid).

---

## Documentation Reference

For detailed information about this implementation:
- **[IMPLEMENTATION_COMPLETE.md](./IMPLEMENTATION_COMPLETE.md)** - Complete summary of testing and implementation
- **[TEST_RESULTS_2026-02-17.md](./TEST_RESULTS_2026-02-17.md)** - Detailed test data and findings
- **[CHANGES_2026-02-17.md](./CHANGES_2026-02-17.md)** - Technical implementation notes

---

## Agent Workflow

### BEFORE modifying download.py rate limits:

**⚠️ CRITICAL: Test first, modify second!**

1. **Run comprehensive authenticated testing:**
   ```bash
   uv run python scripts/benchmark_rate_limits.py --test-authenticated
   ```
   This will:
   - Test rates from 2-10 req/s with token bucket
   - Stop when rate limiting is detected
   - Show you the actual safe threshold

2. **Optionally test sustained behavior:**
   ```bash
   uv run python scripts/benchmark_rate_limits.py --test-sustained --rate-limit 5.0
   ```
   This runs for 2 minutes to ensure limits don't apply over longer windows.

3. **Update config** based on test results:
   - If 5 req/s works: keep `DOWNLOAD_RATE_LIMIT = 5.0`
   - If 5 req/s fails: lower to 4.0 or 3.0
   - If 5 req/s succeeds: try 6.0 to see if we can go faster

### When investigating rate limiting issues:

1. **Check current state:**
   ```bash
   uv run python scripts/benchmark_rate_limits.py --concurrency 5 --samples 10
   ```
   If `rate_limited_pct > 0`, we're in cooldown. Wait 60s.

2. **Find limits:**
   ```bash
   uv run python scripts/benchmark_rate_limits.py --find-optimal
   ```

3. **Update config** with findings in `src/pr0loader/config.py`

### When investigating auth issues:

1. **Test current cookies:**
   ```bash
   uv run python scripts/investigate_pr0gramm.py --test-auth
   ```

2. **If failing, check login flow:**
   ```bash
   uv run python scripts/investigate_pr0gramm.py --skip-legacy
   ```

### When optimizing performance:

1. **Benchmark filesystem:**
   ```bash
   uv run python scripts/benchmark_fs_checks.py --method all
   ```

2. **Profile download pipeline:**
   - Check `download.py` phase timing
   - Look at queue depths
   - Monitor memory usage

---

## Important Code Locations

- **Rate limiting logic:** `src/pr0loader/pipeline/download.py` → `AsyncDownloader`
- **Backoff strategy:** `src/pr0loader/utils/backoff.py` → `fibonacci_backoff`
- **Config defaults:** `src/pr0loader/config.py` → `download_delay_*`
- **Browser headers:** `src/pr0loader/pipeline/download.py` → `BROWSER_HEADERS`

---

## Things We've Tried That Don't Help

1. ❌ **Random delays** - WAF counts requests, not timing patterns
2. ❌ **Full browser headers** - WAF doesn't inspect headers deeply
3. ❌ **Burst with pauses** - Once in cooldown, pauses don't help
4. ❌ **Referer header** - No effect on rate limiting
5. ❌ **Testing unauthenticated then assuming same limits apply to authenticated** - WRONG!

## Things That DO Help

1. ✅ **Token bucket rate limiter at 5 req/s** - TESTED: 100% success over 120s
2. ✅ **Immediate stop on 429** - Don't waste requests during cooldown
3. ✅ **Long cooldown waits** - 90+ seconds after last 429
4. ✅ **Fibonacci backoff** - Good for recovery ramp-up
5. ✅ **Connection pooling** - Reuse connections (keep-alive)
6. ✅ **Sustained testing** - Short bursts lie - test for 2+ minutes!

---

## Critical Discovery: Cumulative Limits

**The WAF doesn't just limit instantaneous rate - it has CUMULATIVE memory!**

Test data (2026-02-17):
- **10 req/s**: Works for ~72 seconds (~700 requests), then 31.8% rate limited
- **5 req/s**: Works perfectly for 120+ seconds (604 requests), 0% rate limited

**Conclusion:** The safe sustained rate is **5 req/s**, not 10 req/s.

---

## Future Improvements

1. ~~**Token bucket algorithm** - More sophisticated rate limiting~~ ✅ IMPLEMENTED
2. **Adaptive rate control** - Auto-detect rate limits and adjust (currently hardcoded 5 req/s)
3. **Multi-IP support** - Rotate through multiple IPs/proxies
4. **Time-of-day optimization** - Server might be less loaded at night
5. **Auth-aware auto-detection** - Detect if authenticated and adjust rate accordingly

---

## Contact / Context

This project downloads images from pr0gramm.com for ML training.
The WAF is reasonable - they just don't want to be hammered.
Goal: Download everything eventually, not necessarily fast.

