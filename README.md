# pr0loader 🚀

A high-performance CLI toolchain for fetching, processing, and training ML models on pr0gramm data.

**Goal:** Train a model that predicts the 5 most likely tags for any given image.

## ✨ Features

- 📥 **Fetch** - Download metadata from pr0gramm API with smart rate limiting
- 📁 **Download** - Batch download media files with parallel loading and async I/O (40-80x faster!)
- 📊 **Prepare** - Generate training datasets with embedded, preprocessed images
- 🧠 **Train** - Train a ResNet50-based tag prediction model
- 🔮 **Predict** - Predict tags for new images
- 🎨 **Beautiful CLI** - Rich progress bars and colored output

## 🚀 Quick Start

### Requirements

- Python 3.10, 3.11, or 3.12 (TensorFlow does not support 3.13+ yet)
- For GPU training: NVIDIA GPU with CUDA 11.8+

### Installation

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh   # Linux/Mac
# Windows PowerShell: irm https://astral.sh/uv/install.ps1 | iex

# Clone and install
git clone https://github.com/yourusername/pr0loader.git
cd pr0loader
uv venv && source .venv/bin/activate
uv pip install -e ".[ml]"

# Run setup wizard
pr0loader
```

## 📖 Usage

### Interactive Mode

```bash
pr0loader              # Launch interactive menu
pr0loader setup        # Run setup wizard
```

### Full Pipeline

```bash
# Sync all data (fetch metadata + download images)
pr0loader sync

# Prepare training dataset
pr0loader prepare

# Train model
pr0loader train output/20240915_dataset.parquet

# Predict tags
pr0loader predict image.jpg
```

### Individual Commands

```bash
# Fetch metadata only
pr0loader fetch
pr0loader fetch --full           # Re-fetch everything

# Download media (now 40-80x faster!)
pr0loader download
pr0loader download --include-videos

# Prepare dataset
pr0loader prepare --output dataset.parquet
pr0loader prepare --min-tags 3   # Adjust minimum tags

# Train model  
pr0loader train dataset.parquet --epochs 10

# Predict
pr0loader predict image.jpg --json
```

### API Server

```bash
# Start REST API
pr0loader api --host 0.0.0.0 --port 8000

# Start Web UI
pr0loader ui

# Both together
pr0loader serve
```

## 🔑 Authentication

```bash
# Auto-detect from browser (recommended)
pr0loader login --auto

# Or from specific browser
pr0loader login --browser firefox

# Check status
pr0loader auth-status
```

## ⚙️ Configuration

All data stored under:
- **Windows:** `%LOCALAPPDATA%\pr0loader`
- **Linux/Mac:** `~/.local/share/pr0loader`

```
pr0loader-data/
├── pr0loader.db      # SQLite database
├── media/            # Downloaded images
├── output/           # Training datasets
└── models/           # Trained models
```

Override with `.env` file or `DATA_DIR` environment variable.

## 🕥️ Headless Mode

For CI/CD or scripts:

```bash
pr0loader --headless sync
pr0loader --headless --verbose fetch
```

## ⚡ Performance Highlights

- **Download Pipeline (V2):** 40-80x faster data loading phase with parallel DB + FS scanning
- **Training:** Images pre-embedded and preprocessed for fast repeated training
- **Scale:** Optimized for millions of images with parallel processing
- **Memory:** Efficient Parquet storage with embedded float32 image data

See [DOCUMENTATION.md](DOCUMENTATION.md) for technical details and architecture deep-dives.

## 🧪 Benchmarks

Measure filesystem existence-check strategies (per-item stat, set diff, dir-batched):

```bash
python3 scripts/benchmark_fs_checks.py --method all --max-items 100000
```

## 🔧 Development

```bash
# Install with dev dependencies
uv pip install -e ".[all,dev]"

# Run tests
pytest

# Format code
black src/
```

## 📚 Documentation

- **[README.md](README.md)** (this file) - User guide and quick start
- **[DOCUMENTATION.md](DOCUMENTATION.md)** - Technical architecture, design decisions, and implementation details

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.


