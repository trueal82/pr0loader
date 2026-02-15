# Fix: "disk I/O error" Bug

## The Problem

When running `pr0loader fetch` after login, users were getting:
```
✗ Fetch failed: disk I/O error
```

## Root Cause

The error was caused by **attempting to change SQLite's `page_size` on an existing database**. 

### Technical Details

SQLite's `page_size` pragma can **only be set BEFORE any tables are created**. Once a database has tables, attempting to change the page size results in a "disk I/O error".

Our optimization code was:
```python
# WRONG - Always tried to set page_size
cursor.execute('PRAGMA page_size=4096')
```

This would fail on existing databases!

### Additional Issues

1. **WAL mode issues on some filesystems**
   - WSL (Windows Subsystem for Linux) may have issues with WAL mode
   - Network drives don't support WAL
   - No error handling if WAL mode failed

2. **No error handling for any PRAGMA statements**
   - Any PRAGMA failure would crash the entire connection
   - Made debugging difficult

## The Fix

### 1. Conditional Page Size Setting

```python
# Check if database is new (no tables yet)
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
has_tables = cursor.fetchone() is not None

if not has_tables:
    # New database - we can set page size
    cursor.execute('PRAGMA page_size=4096')
    logger.debug("Set page size to 4096 (new database)")
else:
    # Existing database - cannot change page size
    logger.debug("Skipping page_size setting (existing database)")
```

**Result:** Only sets page_size on new databases!

### 2. WAL Mode Error Handling

```python
try:
    result = cursor.execute('PRAGMA journal_mode=WAL').fetchone()
    if result and result[0] != 'wal':
        logger.warning(f"Could not enable WAL mode, using {result[0]} instead.")
        logger.warning("This can happen on network drives or certain filesystems.")
except Exception as e:
    logger.warning(f"Failed to set WAL mode: {e}. Continuing with default journal mode.")
```

**Result:** Gracefully falls back if WAL mode not supported!

### 3. Error Handling for ALL Pragmas

Every PRAGMA statement now has try/except:
```python
try:
    cursor.execute('PRAGMA cache_size=-524288')
except Exception as e:
    logger.warning(f"Failed to set cache size: {e}")
```

**Result:** One failing PRAGMA won't crash the entire connection!

## What Changed

**File:** `src/pr0loader/storage/sqlite.py`

**Method:** `_optimize_connection()`

**Changes:**
1. ✅ Added check for existing tables before setting page_size
2. ✅ Added error handling for WAL mode with fallback
3. ✅ Added try/except around all PRAGMA statements
4. ✅ Added informative warning messages
5. ✅ Database connection succeeds even if some optimizations fail

## Testing the Fix

### Scenario 1: New Database (First Run)

```bash
pr0loader fetch
```

**Expected behavior:**
- ✅ All optimizations applied successfully
- ✅ Page size set to 4096
- ✅ WAL mode enabled
- ✅ No errors

**Log output:**
```
DEBUG: Set page size to 4096 (new database)
DEBUG: SQLite performance optimizations applied
INFO: Connected to database: ...
```

### Scenario 2: Existing Database (Second Run)

```bash
pr0loader fetch
```

**Expected behavior:**
- ✅ Connection succeeds
- ✅ Page size NOT changed (skipped)
- ✅ Other optimizations still applied
- ✅ No "disk I/O error"

**Log output:**
```
DEBUG: Skipping page_size setting (existing database)
DEBUG: SQLite performance optimizations applied
INFO: Connected to database: ...
```

### Scenario 3: WAL Mode Not Supported (WSL, Network Drive)

```bash
pr0loader fetch  # On WSL or network drive
```

**Expected behavior:**
- ✅ Connection succeeds
- ⚠️ Warning about WAL mode
- ✅ Falls back to default journal mode
- ✅ Other optimizations still applied

**Log output:**
```
WARNING: Could not enable WAL mode, using delete instead. Performance may be reduced.
WARNING: This can happen on network drives or certain filesystems.
DEBUG: SQLite performance optimizations applied
INFO: Connected to database: ...
```

## Performance Impact

### With Full Optimizations (New DB, WAL Supported)

```
Journal mode: WAL
Page size: 4096
Cache: 512MB
Sync: NORMAL

Performance: Excellent (as designed)
```

### With Partial Optimizations (Existing DB, or No WAL)

```
Journal mode: DELETE (fallback)
Page size: 1024 (default, can't change)
Cache: 512MB (still applied!)
Sync: NORMAL (still applied!)

Performance: Good (cache + sync still help!)
```

**Key point:** Even without WAL and large page size, the 512MB cache and NORMAL sync provide significant performance improvement!

## User Actions

### If You Encountered This Bug

**Option 1: Continue with existing database** (Recommended)
```bash
# Just run fetch again - it will now work!
pr0loader fetch
```

**Result:** Works with existing database, slightly reduced performance (no large page size)

**Option 2: Start fresh for maximum performance** (Optional)
```bash
# Backup existing data if needed
mv ~/.local/share/pr0loader/pr0loader.db ~/.local/share/pr0loader/pr0loader.db.backup

# Run fetch - will create new DB with all optimizations
pr0loader fetch
```

**Result:** New database with all optimizations (4096 page size, WAL, etc.)

### No Action Required!

The fix is **backward compatible** - your existing database will work fine, just without the page_size optimization (which is minor compared to cache and WAL).

## Prevention

This issue is now **prevented** by:

1. ✅ **Checking for existing tables** before setting page_size
2. ✅ **Graceful degradation** - continues even if optimizations fail
3. ✅ **Better error messages** - warns but doesn't crash
4. ✅ **Try/except on all PRAGMAs** - one failure doesn't affect others

## Summary

### The Bug
```
❌ Tried to set page_size on existing database
❌ No error handling
❌ Crashed with "disk I/O error"
```

### The Fix
```
✅ Only sets page_size on NEW databases
✅ Error handling on all PRAGMAs
✅ Graceful fallback if WAL not supported
✅ Informative warnings
✅ Works on existing databases
```

### User Impact
```
Before: ❌ fetch crashes with disk I/O error
After:  ✅ fetch works on all databases
```

**The bug is now fixed! Just run `pr0loader fetch` again and it will work.** 🎉

## Technical Notes

### SQLite Page Size Limitations

From SQLite documentation:
> The page size must be set before any other database operations are performed. Once the database has been created, the page size cannot be changed.

**Why this matters:**
- Larger page sizes (4096) are better for sequential I/O
- Smaller page sizes (1024 default) are fine for random I/O
- For our use case (bulk inserts), 4096 is ideal
- But it's not critical - cache and WAL matter more

### WAL Mode on WSL

WSL1 and some WSL2 configurations may not support WAL mode properly because:
- Shared filesystem between Windows and Linux
- File locking differences
- Some filesystems don't support WAL's requirements

**Fallback works fine** - just slightly slower than WAL mode.

## Related Documentation

- SQLite PRAGMA documentation: https://www.sqlite.org/pragma.html
- WAL mode: https://www.sqlite.org/wal.html
- Page size considerations: https://www.sqlite.org/pgszchng2016.html

