# pr0loader Developer Documentation

**For developers, maintainers, and AI assistants (Claude, etc.)**

This document explains not just HOW the code works, but WHY certain decisions were made.
It captures the evolution of design decisions, performance learnings, and architectural patterns.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Pipeline Stages](#pipeline-stages)
3. [Performance Optimizations](#performance-optimizations)
4. [Data-Driven Design Philosophy](#data-driven-design-philosophy)
5. [SQLite Considerations](#sqlite-considerations)
6. [Image Preprocessing](#image-preprocessing)
7. [Tag Processing Logic](#tag-processing-logic)
8. [Common Pitfalls](#common-pitfalls)
9. [For AI Assistants](#for-ai-assistants)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              pr0loader Pipeline                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────┐     ┌──────────┐     ┌─────────┐     ┌─────────┐             │
│   │  FETCH  │ ──► │ DOWNLOAD │ ──► │ PREPARE │ ──► │  TRAIN  │             │
│   └─────────┘     └──────────┘     └─────────┘     └─────────┘             │
│        │               │                │               │                   │
│        ▼               ▼                ▼               ▼                   │
│   ┌─────────┐     ┌──────────┐     ┌─────────┐     ┌─────────┐             │
│   │ SQLite  │     │  Media   │     │ Parquet │     │  Model  │             │
│   │   DB    │     │  Files   │     │  File   │     │ .keras  │             │
│   └─────────┘     └──────────┘     └─────────┘     └─────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

                           SYNC = FETCH + DOWNLOAD
```

### Key Files

| File | Purpose |
|------|---------|
| `cli.py` | Typer-based CLI with interactive menu |
| `config.py` | Pydantic settings, loads from `.env` |
| `pipeline/fetch.py` | API metadata fetching |
| `pipeline/download.py` | Media file downloading |
| `pipeline/prepare.py` | Dataset preparation (pandas + multiprocessing) |
| `pipeline/train.py` | TensorFlow/Keras model training |
| `storage/sqlite.py` | SQLite storage with WAL mode |

---

## Pipeline Stages

### Stage 1: FETCH

**Purpose:** Download metadata (items, tags) from pr0gramm API

**Key Decisions:**

1. **Parallel Requests with Semaphore**
   - Uses `asyncio` + `aiohttp` for concurrent API calls
   - Semaphore limits concurrent requests (default: 5)
   - WHY: pr0gramm rate-limits aggressive clients

2. **Fibonacci Backoff**
   - On 503/rate-limit: wait Fibonacci sequence (1, 1, 2, 3, 5, 8...)
   - WHY: More gradual than exponential, gentler on the server

3. **Flags Calculation**
   - User sets content preferences (SFW, NSFW, NSFL, NSFP)
   - Flags calculated as: `sum(1 << bit for enabled)`
   - IMPORTANT: flags=15 may cause 403 if account doesn't have all flags unlocked

4. **Range Detection**
   - Detects gaps in the database
   - Fetches missing ranges first, then newest
   - WHY: Ensures complete dataset even after interruptions

```
Flags Bits:
┌─────┬──────┬──────┬──────┐
│ SFW │ NSFW │ NSFL │ NSFP │
│  1  │  2   │  4   │  8   │
└─────┴──────┴──────┴──────┘

Example: SFW + NSFW = 1 + 2 = 3
```

### Stage 2: DOWNLOAD

**Purpose:** Download actual media files (images/videos)

**Key Decisions:**

1. **HEAD Request Verification**
   - Before downloading, sends HEAD request
   - Compares Content-Length with local file size
   - WHY: Avoids re-downloading existing files

2. **Images Only by Default**
   - Videos are large and slow to process
   - Use `--include-videos` to download videos too

3. **Connection Pool**
   - Reuses HTTP connections
   - Pool size matches concurrency level
   - WARNING: "Connection pool full" message indicates pool size mismatch

### Stage 3: PREPARE

**Purpose:** Create training-ready Parquet file with embedded images

**This is the most complex stage. See detailed sections below.**

**Key Outputs:**
- `{timestamp}_dataset.parquet` - Training data with embedded images
- `{timestamp}_dataset.meta.json` - Metadata with computed thresholds

### Stage 4: TRAIN

**Purpose:** Train ResNet50-based multi-label classifier

**Key Decisions:**

1. **No Image Preprocessing in Train**
   - Images come pre-processed from PREPARE
   - Already: resized, RGB→BGR, ImageNet mean subtracted
   - WHY: Prepare once, train many times

2. **Frozen Base Model**
   - ResNet50 weights frozen initially
   - Only classification head is trained
   - WHY: Faster training, prevents overfitting on small datasets

3. **Multi-label Classification**
   - Uses sigmoid activation (not softmax)
   - Binary cross-entropy loss
   - Each image can have multiple tags

---

## Performance Optimizations

### The GIL Problem

**Problem:** Python's Global Interpreter Lock (GIL) prevents true parallelism

```
WRONG (GIL blocks parallelism):
┌──────────────────────────────────────┐
│     ThreadPoolExecutor               │
│  ┌──────┐ ┌──────┐ ┌──────┐         │
│  │ T1   │ │ T2   │ │ T3   │  ◄─ GIL │
│  │ wait │ │ work │ │ wait │         │
│  └──────┘ └──────┘ └──────┘         │
└──────────────────────────────────────┘

RIGHT (bypass GIL):
┌──────────────────────────────────────┐
│     multiprocessing.Pool             │
│  ┌──────┐ ┌──────┐ ┌──────┐         │
│  │ P1   │ │ P2   │ │ P3   │  ◄─ Separate interpreters │
│  │ work │ │ work │ │ work │         │
│  └──────┘ └──────┘ └──────┘         │
└──────────────────────────────────────┘
```

**Solution in prepare.py:**
- Use `multiprocessing.Pool` instead of `ThreadPoolExecutor`
- Module-level function `_preprocess_image_for_resnet()` (can't be a method)
- Each worker is a separate Python process

### SQLite Optimizations

**Problem:** Default SQLite is optimized for safety, not speed

**Applied PRAGMA Settings:**

```sql
PRAGMA journal_mode=WAL;      -- Write-Ahead Logging (3-5x faster writes)
PRAGMA synchronous=NORMAL;    -- Balanced safety/speed
PRAGMA cache_size=-524288;    -- 512MB cache (default is ~2MB)
PRAGMA mmap_size=1073741824;  -- 1GB memory-mapped I/O
PRAGMA temp_store=MEMORY;     -- Temp tables in RAM
```

**WAL Mode Explained:**

```
Without WAL (DELETE mode):
┌────────┐  write  ┌────────┐  commit  ┌────────┐
│ Memory │ ──────► │ Journal│ ───────► │  DB    │
└────────┘         └────────┘          └────────┘
                   (creates file)      (blocks reads)

With WAL:
┌────────┐  write  ┌────────┐  checkpoint  ┌────────┐
│ Memory │ ──────► │  WAL   │ ──────────► │  DB    │
└────────┘         └────────┘             └────────┘
                   (append-only)     (readers not blocked)
```

**Concurrent Access Warning:**
- Running PREPARE while FETCH is active = slow reads
- WAL helps but doesn't eliminate lock contention
- Added detection in prepare.py: `_check_concurrent_access()`

### Pandas Over SQLite

**Problem:** SQLite JSON functions are extremely slow

```python
# SLOW (SQLite processes JSON):
SELECT * FROM items WHERE json_array_length(tags_data) >= 5

# FAST (Pandas processes JSON):
df = pd.read_sql('SELECT * FROM items', conn)
df['tags'] = df['tags_data'].apply(json.loads)
df = df[df['tags'].apply(len) >= 5]
```

**Result:** 10-20x faster filtering with 64GB RAM available

---

## Data-Driven Design Philosophy

### NO HARDCODED THRESHOLDS

**Old approach (bad):**
```python
# Hardcoded trash tag patterns
TRASH_PATTERNS = [r'^repost$', r'^gif$', r'^video$', ...]
HIGH_CONFIDENCE_THRESHOLD = 0.3
MIN_TAG_FREQUENCY = 5
```

**New approach (good):**
```python
# All thresholds computed from actual data
high_conf_threshold = df['confidence'].quantile(0.75)
min_corpus_freq = max(3, int(df['frequency'].quantile(0.25)))
trash_tags = identify_from_statistics(tag_stats)
```

### Trash Tag Detection Criteria

All criteria are data-driven:

| Criterion | Calculation | Rationale |
|-----------|-------------|-----------|
| High doc frequency | > 95th percentile | Meta-tags appear everywhere |
| Low IDF + low conf | IDF < 10th pctl AND conf < median | Common but untrusted |
| High conf variance | std > 2× mean std | Users disagree |
| Singleton + low conf | doc_freq=1 AND conf < median | One-off noise |

### Confidence Merging

**Problem:** Same tag may appear multiple times with different capitalizations

```
"Feet" -> confidence 0.3
"feet" -> confidence 0.4
"FEET" -> confidence 0.2
```

**Solution:** Normalize to lowercase, SUM confidences (not max)

```python
# WHY SUM, not MAX:
# - Each tag occurrence represents a user vote
# - If both "Feet" and "feet" exist, that's 2 people who think it's relevant
# - Combined confidence should be higher than either individual
normalized[tag_lower] = normalized.get(tag_lower, 0) + conf
```

---

## Image Preprocessing

### ResNet50 Requirements

ResNet50 expects images preprocessed as:

1. Size: 224×224 pixels
2. Color order: BGR (not RGB!)
3. Zero-centered by ImageNet mean: `[103.939, 116.779, 123.68]`

```python
# Complete preprocessing in prepare.py:
arr = np.array(img, dtype=np.float32)  # RGB, 0-255
arr = arr[..., ::-1]                    # RGB → BGR
arr -= IMAGENET_MEAN                    # Zero-center
# Result: float32, shape (224, 224, 3), values roughly [-128, +128]
```

### Why Embed in Parquet?

**Problem:** Training reads images repeatedly

```
Without embedding (6M images × 5 epochs):
┌─────────┐     ┌──────────────┐
│ Train   │ ──► │ Filesystem   │  × 30 MILLION file reads!
│ Epoch 1 │     │ (HDD/RAID5)  │
│ ...     │     │              │
│ Epoch 5 │     │              │
└─────────┘     └──────────────┘

With embedding:
┌─────────┐     ┌──────────────┐     ┌─────────┐
│ Prepare │ ──► │ Parquet      │ ──► │ Train   │
│ (once)  │     │ (sequential) │     │ (RAM)   │
└─────────┘     └──────────────┘     └─────────┘
                 6M × 1 read          All in memory
```

**Storage Format:**

```python
# Each image stored as raw bytes (tobytes())
image_data = arr.tobytes()  # 224 × 224 × 3 × 4 = 602,112 bytes per image

# Reconstructed in train:
arr = np.frombuffer(image_data, dtype=np.float32).reshape(224, 224, 3)
```

---

## Tag Processing Logic

### Content Flags vs Tags

**Content flags** are metadata about content rating:
- `sfw`, `nsfw`, `nsfl`, `nsfp`
- These are NOT descriptive tags
- Extracted into separate boolean columns

```python
CONTENT_FLAGS = frozenset({'nsfw', 'nsfl', 'nsfp', 'sfw'})

# In processing:
if tag_lower == 'nsfw':
    is_nsfw = True  # Store as metadata
elif tag_lower not in CONTENT_FLAGS:
    normalized[tag_lower] = ...  # Process as content tag
```

### Minimum Tags Requirement

**WHY 5 tags minimum?**
- Multi-label classifier needs multiple labels per sample
- Single-tag images don't help model learn correlations
- 5 is a balance: keeps most data, ensures meaningful labels

**Trash Tag Rescue:**

```python
# If good tags < min_tags, add back trash tags with highest confidence
if len(valid_tags) < min_tags:
    needed = min_tags - len(valid_tags)
    valid_tags.extend(trash_tags_sorted[:needed])
```

---

## Common Pitfalls

### 1. Flags 403 Error

**Symptom:** `Fetch failed: disk I/O error` or 403 responses

**Cause:** Requesting content flags user doesn't have access to

**Fix:** Check account settings on pr0gramm.com, or use lower flags value

### 2. Connection Pool Full Warning

**Symptom:** `Connection pool is full, discarding connection`

**Cause:** Pool size doesn't match concurrency level

**Fix:** Match pool size to number of concurrent workers

### 3. Slow SQLite Reads

**Symptom:** PREPARE takes forever on first step

**Cause:** FETCH running concurrently OR slow storage

**Fix:**
- Wait for FETCH to complete: `pr0loader prepare --wait`
- Or use SSD instead of HDD

### 4. Out of Memory in PREPARE

**Symptom:** Process killed during image embedding

**Cause:** Too many images × 600KB each

**Fix:** Use dev mode for testing: `settings.dev_mode = True`

### 5. TensorFlow Not Using GPU

**Symptom:** Training very slow, GPU at 0%

**Cause:** CUDA/cuDNN not installed or version mismatch

**Fix:** Check `pr0loader info` output, verify CUDA installation

---

## For AI Assistants

### Key Context for Future Claude Instances

When working on this codebase, remember:

1. **Performance is Critical**
   - Target: 6 million images
   - Every loop matters
   - Prefer numpy/pandas over Python loops
   - Use multiprocessing for CPU-bound tasks (not threading!)

2. **Data-Driven, Not Hardcoded**
   - NO magic numbers in tag filtering
   - All thresholds from percentiles/statistics
   - Dataset may change, code must adapt

3. **Separation of Concerns**
   - PREPARE does ALL image preprocessing
   - TRAIN does NO image manipulation
   - This is intentional for "prepare once, train many"

4. **SQLite Quirks**
   - WAL mode is enabled but concurrent writes still slow reads
   - Don't use `json_array_length()` or similar - do it in pandas
   - Large cache (512MB) is set - uses RAM

5. **Testing Philosophy**
   - `--dev` mode limits data for fast iteration
   - Always test with small dataset first
   - Full runs take hours - don't do them casually

### File Locations Quick Reference

```
src/pr0loader/
├── cli.py           # ALL CLI commands defined here
├── config.py        # Settings class, .env loading
├── models.py        # Pydantic models for API data
├── pipeline/
│   ├── fetch.py     # API fetching (async)
│   ├── download.py  # Media downloading
│   ├── prepare.py   # Dataset prep (pandas + multiprocessing)
│   ├── train.py     # Model training (TensorFlow)
│   └── sync.py      # Combines fetch + download
├── storage/
│   └── sqlite.py    # Database operations, WAL mode
└── utils/
    ├── console.py   # Rich console helpers
    └── backoff.py   # Fibonacci backoff
```

### Common Modifications

**Adding a new CLI option:**
1. Add parameter to function in `cli.py`
2. Add to Settings class in `config.py` if persistent
3. Update help text

**Changing tag filtering:**
1. Edit `_analyze_tags_datadriven()` in `prepare.py`
2. Thresholds should be computed, not hardcoded
3. Document reasoning in `trash_tag_reasons`

**Changing image preprocessing:**
1. Edit `_preprocess_image_for_resnet()` in `prepare.py`
2. Update metadata in `_save_metadata()`
3. Verify train.py reads correctly

---

## Metadata File Format

Every Parquet file has a `.meta.json` companion:

```json
{
  "created": "2026-02-16T00:00:00",
  "total_items": 500000,
  "items_skipped": 50000,
  "min_tags": 5,
  "unique_tags": 100000,
  
  "thresholds": {
    "high_confidence": 0.342,
    "min_corpus_frequency": 3
  },
  
  "images_embedded": true,
  "image_size": [224, 224],
  "image_format": {
    "dtype": "float32",
    "shape": [224, 224, 3],
    "preprocessing": "resnet50",
    "color_order": "BGR",
    "imagenet_mean": [103.939, 116.779, 123.68]
  },
  
  "trash_tags_sample": {
    "repost": "high_doc_freq (>5.0%)",
    "gif": "high_doc_freq (>5.0%)"
  }
}
```

---

## Version History

- **v2.0.0** - Complete rewrite with pandas, multiprocessing, embedded images
- **v1.x** - Original implementation with row-by-row processing

---

*Last updated: February 2026*
*Maintained by: Human + Claude collaboration*

