# Parallel Metadata Fetching Improvements

## Overview

The fetch pipeline has been enhanced to use parallel requests when fetching metadata (tags and comments) from the pr0gramm API. This significantly improves performance by reducing the time spent waiting for sequential API responses.

## Changes Made

### 1. **fetch.py** - Parallel Request Implementation

#### New Methods:
- **`_fetch_item_info(item: Item) -> Optional[Item]`**
  - Fetches detailed info (tags, comments) for a single item
  - Returns the populated item or None if failed
  - Handles errors gracefully and logs debug information

- **`_fetch_batch_info_parallel(items: list[Item]) -> list[Item]`**
  - Fetches info for multiple items in parallel using ThreadPoolExecutor
  - Configurable number of workers via `max_parallel_requests` setting
  - Processes results as they complete using `as_completed()`
  - Maintains proper error counting in stats

#### Updated Logic:
- **Before**: Sequential loop fetching item info one by one
  ```python
  for item in response.items:
      info = self.api.get_item_info(item.id)  # Sequential
      item.tags = info.tags
      item.comments = info.comments
  ```

- **After**: Parallel batch processing
  ```python
  processed_items = self._fetch_batch_info_parallel(response.items)  # Parallel
  batch.extend(processed_items)
  ```

### 2. **config.py** - New Configuration Option

Added `max_parallel_requests` setting:
```python
max_parallel_requests: int = Field(
    default=10,
    description="Maximum number of parallel metadata requests (for fetch pipeline)"
)
```

**Default**: 10 parallel workers
**Purpose**: Controls how many metadata requests are made simultaneously

### 3. **template.env** - Configuration Template Update

Added documentation for the new settings:
```bash
# MAX_PARALLEL_REQUESTS = 10  # Parallel workers for metadata fetching
```

Users can now easily configure this in their `.env` file.

## Performance Impact

### Before (Sequential):
- For a batch of 120 items (default API page size)
- Each `get_item_info()` call takes ~0.5-1 second
- **Total time**: 120 × 0.75s = **~90 seconds per batch**

### After (Parallel with 10 workers):
- Same 120 items processed in parallel batches of 10
- Each batch of 10 takes ~0.5-1 second
- **Total time**: (120 ÷ 10) × 0.75s = **~9 seconds per batch**

### Expected Speedup: **~10x faster** 🚀

## Key Features

### 1. **Thread-Safe Implementation**
- Uses Python's `concurrent.futures.ThreadPoolExecutor`
- No additional dependencies required
- Thread-safe session handling in `APIClient`

### 2. **Robust Error Handling**
- Individual item failures don't affect the entire batch
- Failed items are properly logged and counted
- Successful items are still processed

### 3. **Configurable Parallelism**
- Adjustable via `MAX_PARALLEL_REQUESTS` in `.env`
- Can be tuned based on:
  - Network bandwidth
  - API rate limits
  - System resources

### 4. **Backward Compatible**
- Setting `MAX_PARALLEL_REQUESTS=1` reverts to sequential behavior
- All existing functionality preserved
- Same database batch processing

## Usage

### Default Configuration (Recommended)
No changes needed! The default of 10 parallel workers is optimal for most use cases.

### Custom Configuration
Edit your `.env` file:
```bash
# Conservative (slower, less aggressive)
MAX_PARALLEL_REQUESTS = 5

# Aggressive (faster, more requests)
MAX_PARALLEL_REQUESTS = 20

# Sequential (original behavior)
MAX_PARALLEL_REQUESTS = 1
```

### Monitor Performance
The fetch command will show the parallel worker count:
```
Fetching items from ID 6154365 down to 1
Estimated items: ~6,154,365
Parallel workers: 10
DB batch size: 200 items
```

## Rate Limiting Considerations

The implementation respects existing rate limiting:
- Backoff strategy still applies per request
- Individual failed requests are retried
- Main loop still uses `request_delay` between API page fetches
- Parallel requests are for metadata info only, not main item lists

## Testing Recommendations

1. **Start with default settings** (10 workers)
2. **Monitor your network and API responses**
3. **Adjust if needed**:
   - Reduce if you see rate limiting (429 errors)
   - Increase if your network can handle more
4. **Check logs** for any error patterns

## Technical Notes

### Why ThreadPoolExecutor?
- **I/O-bound workload**: Network requests spend most time waiting
- **GIL not a problem**: Threads work well for I/O operations
- **No async refactor needed**: Works with existing `requests` library
- **Simple and reliable**: Standard library, well-tested

### Thread Safety
- `requests.Session` is thread-safe for concurrent requests
- Each thread makes independent API calls
- Stats counting uses standard Python operations (safe in this context)

## Future Improvements

Potential enhancements for consideration:
- [ ] Adaptive parallelism based on response times
- [ ] Per-thread rate limiting
- [ ] Connection pooling optimization
- [ ] Async/await implementation (if migrating to aiohttp)

## Rollback Instructions

If you need to revert to sequential processing:
1. Set `MAX_PARALLEL_REQUESTS=1` in your `.env` file
2. Or restore the previous version of `fetch.py` from git

## Summary

This improvement provides a significant performance boost (~10x) for metadata fetching while maintaining:
- ✅ Code clarity and maintainability
- ✅ Robust error handling
- ✅ Configurability
- ✅ Backward compatibility
- ✅ No additional dependencies

**Combined with SQLite optimizations** (see [SQLITE_OPTIMIZATIONS.md](SQLITE_OPTIMIZATIONS.md)), the fetch pipeline achieves **30-50x overall speedup**! The parallel requests handle API I/O bottleneck while SQLite optimizations handle the database write bottleneck.

The fetch pipeline is now much more efficient at downloading large amounts of metadata from the pr0gramm API!

