# pr0loader 🚀

A beautiful CLI toolchain for fetching, processing, and training ML models on pr0gramm data.

**Goal:** Train a model that predicts the 5 most likely tags for any given image.

## ✨ Features

- 📥 **Fetch** - Download metadata from pr0gramm API with smart rate limiting (Fibonacci backoff)
- 📁 **Download** - Batch download media files with progress tracking
- 📊 **Prepare** - Generate training datasets from collected data
- 🧠 **Train** - Train a ResNet50-based tag prediction model
- 🔮 **Predict** - Predict tags for new images
- 🎨 **Beautiful CLI** - Rich progress bars and colored output (or `--headless` for CI/scripts)

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pr0loader.git
cd pr0loader

# Install in development mode
pip install -e .

# For ML features (training/prediction), install with extras:
pip install -e ".[ml]"
```

## ⚙️ Configuration

Copy `template.env` to `.env` and configure your settings:

```bash
cp template.env .env
```

Required settings:
- `PP` - Your pr0gramm PP cookie
- `ME` - Your pr0gramm ME cookie
- `FILESYSTEM_PREFIX` - Directory to store downloaded media

## 📖 Usage

### Interactive Mode

Simply run `pr0loader` without any arguments to launch the interactive menu:

```bash
pr0loader
```

This will show a menu where you can:
- Select which command to run
- Configure all options interactively
- Review settings before execution

**Works in screen/tmux sessions!** The menu automatically detects terminal capabilities.

### Show Help

```bash
pr0loader --help
```

### View System Info

```bash
pr0loader info
```

### 🔄 Full Sync (Recommended)

The `sync` command fetches metadata AND downloads assets in one efficient run:

```bash
# Sync everything (incremental - only new items)
pr0loader sync

# Full sync from scratch
pr0loader sync --full

# Sync from specific ID
pr0loader sync --start-from 1000000

# Metadata only (no downloads)
pr0loader sync --metadata-only

# Include videos (by default, only images are downloaded)
pr0loader sync --include-videos

# Skip file verification (faster, but won't detect corrupted files)
pr0loader sync --no-verify
```

**Note:** Metadata is always fetched for ALL items (images + videos), but by default only images are downloaded. Use `--include-videos` to also download video files.

**Smart file verification:** Existing files are checked via HEAD request - only re-downloaded if the file size differs from the remote.

### Complete Pipeline

Run all stages (fetch → download → prepare → train):

```bash
pr0loader run-all
```

Or run individual stages:

### 1. Fetch Metadata

```bash
# Fetch new items
pr0loader fetch

# Full re-fetch
pr0loader fetch --full

# Start from specific ID
pr0loader fetch --start-from 1000000
```

### 2. Download Media

```bash
# Download images only (default)
pr0loader download

# Also download videos
pr0loader download --include-videos
```

### 3. Prepare Dataset

```bash
# Generate training CSV
pr0loader prepare

# Custom output path
pr0loader prepare --output ./my_dataset.csv

# Adjust minimum tags
pr0loader prepare --min-tags 3
```

### 4. Train Model

```bash
# Train with default settings
pr0loader train ./output/20240915_dataset.csv

# Custom training parameters
pr0loader train dataset.csv --epochs 10 --batch-size 64

# Development mode (small subset)
pr0loader train dataset.csv --dev
```

### 5. Predict Tags

```bash
# Predict tags for an image
pr0loader predict image.jpg

# Multiple images
pr0loader predict image1.jpg image2.jpg image3.jpg

# JSON output (for APIs/scripts)
pr0loader predict image.jpg --json

# Custom model
pr0loader predict image.jpg --model ./my_model.keras
```

## 🖥️ Headless Mode

For CI/CD or scripts, use `--headless` to disable fancy output:

```bash
pr0loader --headless fetch
pr0loader --headless run-all
```

## 📁 Project Structure

```
pr0loader/
├── src/pr0loader/
│   ├── cli.py              # CLI entry point
│   ├── config.py           # Pydantic settings
│   ├── models.py           # Data models
│   ├── api/                # API client
│   ├── storage/            # SQLite storage
│   ├── pipeline/           # Pipeline stages
│   └── utils/              # Utilities
├── pyproject.toml          # Package config
├── requirements.txt        # Dependencies
└── template.env            # Example config
```

## 🔧 Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.


