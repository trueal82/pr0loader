# Content Flags Fix - Investigation & Resolution

## Issue

After logging in, running `pr0loader fetch` with `flags=15` was returning a 403 Forbidden error.

## Investigation Results

Using `investigate_pr0gramm.py`, we discovered:

### 1. Flag System Is Unchanged

The pr0gramm content flags system still works as expected:
- `flags=1` (bit 0): SFW content
- `flags=2` (bit 1): NSFW content
- `flags=4` (bit 2): NSFL content
- `flags=8` (bit 3): NSFP content
- `flags=15`: All content types (SFW + NSFW + NSFL + NSFP)

### 2. New Flag Discovered

There's a **new bit 4 (value 16)** that wasn't previously documented:
- `flags=31` (bits 0-4) returns items with flags including `16`
- This appears to be a new content category

### 3. Authentication Is Strictly Required

Without valid authentication cookies (PP and ME):
- `flags=1` (SFW only): ✓ Works
- `flags > 1` (any non-SFW): ✗ Returns 403 Forbidden

With valid authentication:
- `flags=1` through `flags=31`: ✓ All work

### 4. The Real Bug

The stored credentials in `credentials.json` were being **verified** but not **loaded into the Settings object** before creating the API client.

**Before (broken):**
```python
def check_auth_for_content_flags(settings, headless):
    # ...
    creds = auth.store.load()
    if creds and creds.is_valid():
        return True  # Just returns True, doesn't load credentials!
```

**After (fixed):**
```python
def check_auth_for_content_flags(settings, headless):
    # ...
    creds = auth.store.load()
    if creds and creds.is_valid():
        # IMPORTANT: Load credentials into settings so APIClient can use them
        settings.pp = creds.pp
        settings.me = creds.me
        print_info(f"Using stored credentials for user: {creds.username}")
        return True
```

## Fix Applied

**File:** `src/pr0loader/cli.py`

**Function:** `check_auth_for_content_flags()`

**Change:** Now loads stored credentials into the `settings` object so they're available when `APIClient` is created.

## Testing

After the fix:
```
$ pr0loader fetch --start-from 100
ℹ Using stored credentials for user: pr0tag0nist
📥 Fetch Metadata
ℹ Fetching items from ID 100 down to 1
✓ Fetch complete! (87 items processed)
```

## Summary

| Issue | Cause | Fix |
|-------|-------|-----|
| 403 on flags=15 | Credentials not loaded into settings | Load `creds.pp` and `creds.me` into settings object |
| Flag system changed? | No - auth was just missing | N/A |
| New flag bit 16? | Yes - appears to be new content type | Consider updating `CONTENT_FLAGS` default to 31 |

## Recommendations

1. **Current config works fine** - `flags=15` covers all traditional content types
2. **Optional:** Update default `CONTENT_FLAGS` to `31` to include the new flag bit 16
3. **The investigate_pr0gramm.py script** is useful for debugging auth/flag issues

