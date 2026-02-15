# pr0loader — Developer & Architecture Documentation

Audience: **software architects**, **Python developers**, **data scientists/ML engineers**, and **LLM assistants**.

This document is both:
- a **human-readable architecture guide** (why + tradeoffs)
- a **machine-usable spec** (contracts + invariants + “do-not-break” rules)

> Design intent: **Prepare once, train many times** — with **data-driven** thresholds so behavior adapts as the dataset grows.

---

## Contents

1. [Executive summary](#0-executive-summary-tldr)
2. [System architecture](#1-system-architecture)
3. [Repository map](#2-repository-map-where-to-look)
4. [Pipeline stages](#3-pipeline-stages-what--why)
5. [Performance model](#4-performance-model-what-makes-it-fast)
6. [Tag logic](#5-tag-logic-data-driven-no-static-lists)
7. [Image preprocessing](#6-image-preprocessing-done-in-prepare-never-in-train)
8. [Operational guidance](#7-operational-guidance)
9. [Troubleshooting](#8-common-pitfalls--troubleshooting)
10. [LLM / future maintainer notes](#9-llm--future-maintainer-notes)
11. [Design glossary](#appendix-b-design-glossary)

---

## 0) Executive summary (TL;DR)

### What you get

- `fetch` → **SQLite** metadata (high volume writes)
- `download` → **media/** files (high volume IO)
- `prepare` → **Parquet + .meta.json** (expensive analytics + preprocessing, done once)
- `train` → **model** (repeated often; should be cheap relative to prepare)

### Three rules that keep this fast

1. **No SQLite JSON functions** (`json_array_length`, `json_extract`, …)
   - Parse JSON in pandas/numpy.
2. **No Python threads for CPU-heavy work**
   - Use `multiprocessing` to bypass the GIL.
3. **No static tag assumptions**
   - Thresholds and “trash tags” are learned from current data distributions.

---

## 1) System architecture

### 1.1 Pipeline overview

```
                 (network)             (disk)                 (RAM+CPU)              (GPU/CPU)

   ┌─────────┐      ┌─────────┐        ┌──────────┐           ┌─────────┐            ┌─────────┐
   │ pr0 API  │ ◄──► │  FETCH  │ ─────► │  SQLite  │ ────────► │ PREPARE │ ─────────► │  TRAIN  │
   └─────────┘      └─────────┘        └──────────┘           └─────────┘            └─────────┘
                         │                   ▲                     │
                         │                   │                     │
                         ▼                   │                     ▼
                    ┌─────────┐              │                ┌──────────┐
                    │ DOWNLOAD│ ─────────────┘                │ Parquet   │
                    └─────────┘                               │ + .meta   │
                         │                                    └──────────┘
                         ▼
                    ┌─────────┐
                    │ media/  │
                    └─────────┘

   sync = fetch + download
```

### 1.2 Key invariants (contracts)

These are **non-negotiable**. Stages rely on them.

#### Prepare output contract (`*.parquet` + `*.meta.json`)

**Required columns**:
- `id: int`
- `image: str` (relative path, kept for traceability/debug)
- `tags: list[str]` (length >= `min_tags`)
- `confidences: list[float]` (same length and order as `tags`)
- `is_nsfw/is_nsfl/is_nsfp: bool`
- `image_data: bytes` (raw float32 bytes)

**Image decoding contract**:
- `arr = np.frombuffer(image_data, np.float32).reshape(H, W, 3)`
- channel order is **BGR**
- already **ResNet50-preprocessed** (ImageNet mean subtracted)

#### Train contract

- Uses `image_data` directly.
- Must not decode/resize/normalize (no `tf.io.read_file`, no `decode_jpeg`, no `preprocess_input`).
- If `image_data` is missing, training **fails fast** with a clear message.

---

## 2) Repository map (where to look)

```
src/pr0loader/
├── cli.py                 # CLI commands
├── config.py              # Settings (.env / defaults)
├── models.py              # Typed models / stats
├── storage/
│   └── sqlite.py          # SQLite connect + PRAGMAs
└── pipeline/
    ├── fetch.py            # Metadata ingest
    ├── download.py         # Media download
    ├── prepare.py          # Analytics + dataset build (pandas + multiprocessing)
    ├── train.py            # Training (expects embedded images)
    └── sync.py             # Orchestrates fetch+download
```

---

## 3) Pipeline stages (what + why)

This section uses a consistent template:
- **Purpose** (what it does)
- **Why** (reason it exists)
- **Outputs**
- **Performance notes**

### 3.1 `fetch` — metadata ingest

**Purpose:** Pull item metadata from the pr0 API and store it in SQLite.

**Why:**
- Decouples network ingestion from dataset building.
- Makes ingestion resumable (gap filling, incremental updates).

**Outputs:** SQLite rows in `items` (and related tables).

**Performance notes:**
- Parallel HTTP helps, but SQLite writes often dominate.
- Some content flag combinations can yield **403** depending on account permissions.

Flags (conceptual):

```
Bits:  1      2      4      8
     SFW   NSFW   NSFL   NSFP

flags = sum(enabled_bits)
```

### 3.2 `download` — media ingest

**Purpose:** Download images/videos into `media/`.

**Why:**
- Keeps `prepare` / `train` offline and repeatable.

**Outputs:** filesystem objects (images/videos).

**Performance notes:**
- Uses verification to avoid redownload.
- Connection pool warnings usually mean concurrency > pool size.

### 3.3 `prepare` — dataset creation (the heavy lifter)

**Purpose:** Build a **training-ready** Parquet with embedded, preprocessed images.

**Why:**
- Expensive work happens once.
- Training becomes repeatable and fast.

**Outputs:**
- `output/*.parquet`
- `output/*.meta.json`

**Performance notes:**
- Avoid SQLite JSON functions; use pandas.
- Use multiprocessing for CPU-bound image preprocessing.

**Core steps:**
1. Read minimal columns from SQLite.
2. Parse tags JSON in Python.
3. Compute data-driven tag thresholds + trash tags.
4. Build per-item tag lists with `min_tags` invariant.
5. Preprocess images (resize + ResNet50 transform) and embed.

### 3.4 `train` — model training

**Purpose:** Train the ML model from prepared Parquet.

**Why:**
- Training is expected to run many times (hyperparams, experiments).
- If train does decoding/resizing/normalization, it becomes IO-bound and wastes CPU.

**Outputs:**
- `models/*.keras` (and optional mappings)

---

## 4) Performance model (what makes it fast)

### 4.1 SQLite: WAL + pragmatic IO settings

SQLite is used because it’s simple and portable.

We rely on:
- WAL to reduce writer/reader contention
- bigger cache/mmap for fewer HDD seeks

**Caveat:** Concurrent `fetch` (writes) + `prepare` (reads) is still slower.

### 4.2 Don’t fight the GIL

- Threads don’t help CPU-heavy Python loops.
- `multiprocessing` gives real parallelism.

Rule of thumb:
- **IO-bound** → threads can be OK
- **CPU-bound** → processes

### 4.3 Pandas over SQLite JSON

We moved *all* JSON analysis into pandas/numpy because:
- SQLite JSON functions scale poorly in practice (especially on HDD)
- pandas can utilize vectorized C/NumPy operations

---

## 5) Tag logic (data-driven, no static lists)

### 5.1 Principles

1. **Normalize tags to lowercase**
2. **Merge confidence by SUM** (votes accumulate)
3. **Trash detection is statistical**
4. **Keep samples label-rich** (`min_tags`) to support multi-label learning

### 5.2 “Trash tags” are learned

Trash is not a hardcoded list.
We derive it from the dataset using metrics like:
- document frequency
- IDF
- confidence distribution
- confidence variance

Output artifacts:
- `trash_tags` set
- `trash_tag_reasons` mapping (debuggable)

### 5.3 `min_tags` invariant

If strict filtering would drop below `min_tags`, we re-add the highest-confidence trash tags.
This is a deliberate compromise:
- preserves tag-learnability
- avoids throwing away too much data

---

## 6) Image preprocessing (done in `prepare`, never in `train`)

### 6.1 Target (ResNet50-ready) format

Transform:
1. decode image → RGB
2. resize → `image_size` (typically 224×224)
3. `float32`
4. RGB → BGR
5. subtract ImageNet mean `[103.939, 116.779, 123.68]`

Result: `float32` array shaped `(H, W, 3)`.

### 6.2 Storage: bytes in Parquet

We store raw bytes:

```
image_data = arr.tobytes()

# reconstruct
arr = np.frombuffer(image_data, np.float32).reshape(H, W, 3)
```

Why bytes:
- Arrow/Parquet handle bytes reliably
- avoids nested ndarray conversion issues

### 6.3 Scale note

Raw float32 images are large at millions of items.
Future enhancements may include:
- sharded Parquet datasets
- streaming training
- caching embeddings instead of pixels

---

## 7) Operational guidance

### 7.1 Recommended run order

1) `sync`
2) `prepare`
3) iterate on `train`

### 7.2 Avoid fetch+prepare overlap

- use `pr0loader prepare --wait`
- or schedule prepare separately

---

## 8) Common pitfalls & troubleshooting

- **403 on fetch**: flags/account permissions.
- **prepare load is slow**: fetch running or slow HDD seek.
- **CPU not utilized**: likely IO-bound or accidentally threaded CPU work.
- **train slow**: dataset missing `image_data` or train is preprocessing (should not).

---

## 9) LLM / future-maintainer notes

### 9.1 Do-not-break list

- `prepare` always embeds images and writes `.meta.json`.
- `train` never preprocesses images.
- tag thresholds remain data-driven.

### 9.2 Change checklist

Before merging changes, ask:
1. Did I introduce static heuristics?
2. Did I reintroduce SQLite JSON evaluation?
3. Did I add CPU-heavy threading?
4. Did I change the Parquet contract without updating train + metadata?

---

## Appendix A) Dataset metadata (`*.meta.json`)

`.meta.json` exists so humans and tools can answer:
- which preprocessing is used
- which thresholds were computed

Example:

```json
{
  "images_embedded": true,
  "image_size": [224, 224],
  "image_format": {
    "dtype": "float32",
    "shape": [224, 224, 3],
    "preprocessing": "resnet50",
    "color_order": "BGR",
    "imagenet_mean": [103.939, 116.779, 123.68]
  },
  "thresholds": {
    "high_confidence": 0.342,
    "min_corpus_frequency": 3
  }
}
```

---

## Appendix B) Design glossary

A glossary for architects/data scientists (and for LLM context).

### Dataset / tagging terms

- **Tag**: a (usually human-added) label for an item.
- **Confidence**: numeric score/vote strength from user interactions (+/-). In pr0loader we treat it as *vote mass*.
- **Lowercasing / normalization**: canonicalize tag names to reduce duplicates (`Feet` → `feet`).
- **Frequency**: how often a tag occurs across all tag assignments (total occurrences).
- **Document frequency (DF)**: number of distinct items containing the tag.
- **DF ratio**: `DF / total_items` (share of dataset containing that tag).
- **IDF (inverse document frequency)**: `log(total_items / DF)`. Higher = rarer = more “informative”.
- **Multi-label**: one image can have multiple tags (not mutually exclusive).
- **Multi-hot vector**: label encoding where each class has a 0/1 bit.

### Model / ML terms

- **ResNet50 preprocessing**: specific transform expected by the ImageNet-pretrained ResNet50:
  - RGB → BGR
  - subtract ImageNet mean `[103.939, 116.779, 123.68]`
- **Frozen base model**: use pretrained CNN weights as fixed feature extractor; train only the classifier head.
- **Sigmoid head**: independent probability per tag (required for multi-label).
- **Binary cross entropy**: typical loss for multi-label classification.

### Systems / performance terms

- **GIL (Global Interpreter Lock)**: prevents true parallel CPU execution in one Python process.
- **Threading vs multiprocessing**: threads share one interpreter (GIL); processes run truly parallel.
- **WAL (Write-Ahead Logging)**: SQLite journaling mode enabling better concurrent read/write behavior.
- **IO-bound vs CPU-bound**:
  - IO-bound: waiting on disk/network → threads can help.
  - CPU-bound: heavy compute → processes (or vectorized/native code).

---

*Last updated: 2026-02*
