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

**Requirements:**
- Python 3.10, 3.11, or 3.12 (TensorFlow doesn't support 3.13+ yet)
- For GPU training: NVIDIA GPU with CUDA 11.8 or 12.x

### Quick Start (Recommended: using uv)

[uv](https://docs.astral.sh/uv/) is a blazingly fast Python package manager (10-100x faster than pip!)

```bash
# Install uv (if not already installed)
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/pr0loader.git
cd pr0loader

# uv automatically uses Python 3.12 (from .python-version file)
# Create venv and install with ML features
uv venv
uv pip install -e ".[ml]"

# Activate the environment
.venv\Scripts\activate         # Windows
source .venv/bin/activate      # Linux/Mac
```

### Alternative: Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/pr0loader.git
cd pr0loader

# Create virtual environment with Python 3.12
py -3.12 -m venv .venv         # Windows
python3.12 -m venv .venv       # Linux/Mac

# Activate virtual environment
.venv\Scripts\activate         # Windows
source .venv/bin/activate      # Linux/Mac

# Install with ML features
pip install -e ".[ml]"
```

### GPU Support (Recommended for Training)

For faster training with NVIDIA GPUs, TensorFlow 2.15+ automatically uses CUDA when available:

**Requirements:**
- NVIDIA GPU with CUDA Compute Capability 3.5 or higher
- CUDA Toolkit 11.8 or 12.x
- cuDNN 8.6+

**Installation:**

```bash
# 1. Install CUDA Toolkit from NVIDIA
# Download from: https://developer.nvidia.com/cuda-downloads

# 2. Install pr0loader with ML extras
uv pip install -e ".[ml]"
# or: pip install -e ".[ml]"

# 3. Verify GPU is detected
pr0loader --version
# When running train/e2e-test, you should see:
# "GPU available: True"
```

**Note:** TensorFlow 2.10+ has built-in GPU support. No need to install `tensorflow-gpu` separately.

If GPU is not detected, training will still work but use CPU (much slower).

## ⚙️ Configuration

### First-Time Setup

Run pr0loader without arguments to start the **setup wizard**:

```bash
pr0loader
```

The wizard will guide you through:
1. **Data directory** - Where to store all pr0loader files
2. **Authentication** - Auto-login from browser or enter cookies manually
3. **Content preferences** - Which content types to download (SFW, NSFW, etc.)
4. **Advanced settings** - Training parameters (optional)

You can also run the setup explicitly:

```bash
pr0loader setup          # Run setup wizard
pr0loader setup --force  # Overwrite existing configuration
```

### Data Directory

All pr0loader data is stored under a single base directory:

| OS | Default Location |
|----|------------------|
| Windows | `%LOCALAPPDATA%\pr0loader` |
| Linux | `~/.local/share/pr0loader` |
| macOS | `~/.local/share/pr0loader` |

You can override this in `.env`:

```bash
DATA_DIR = ~/my-pr0loader-data
```

### Directory Structure

```
{DATA_DIR}/
├── pr0loader.db      # SQLite database (metadata, tags)
├── media/            # Downloaded images/videos
├── output/           # Generated datasets
├── models/           # Trained models
├── checkpoints/      # Training checkpoints
└── auth/             # Credentials (if not using keyring)
```

All paths can be individually overridden in `.env` if needed.

**Authentication is now handled automatically!** See the Authentication section below.

## 🔐 Authentication

pr0loader now supports automatic authentication - no more manual cookie extraction!

### Quick Start

```bash
# Auto-detect from installed browsers (recommended)
pr0loader login --auto

# Or extract from a specific browser
pr0loader login --browser firefox
pr0loader login --browser chrome

# Or interactive login with captcha
pr0loader login --interactive

# Check current status
pr0loader auth-status

# Logout / clear credentials
pr0loader logout
```

### Authentication Methods

| Method | Command | Description |
|--------|---------|-------------|
| Auto-detect | `--auto` | Tries all installed browsers |
| Firefox | `--browser firefox` | Extract from Firefox (easiest) |
| Chrome | `--browser chrome` | Extract from Chrome (needs DPAPI) |
| Edge | `--browser edge` | Extract from Edge |
| Brave | `--browser brave` | Extract from Brave |
| Interactive | `--interactive` | Login with username/password + captcha |

### How It Works

1. **Browser extraction**: Reads cookies from your browser's database
   - Firefox: Unencrypted SQLite, easiest method
   - Chrome/Edge: Encrypted with Windows DPAPI, requires `pycryptodome`

2. **Interactive login**: 
   - Fetches captcha from pr0gramm
   - Displays captcha (saved as temp file + ASCII preview)
   - You enter username, password, and captcha solution
   - Credentials stored securely in system keyring

3. **Secure storage**: Credentials are stored using:
   - System keyring (Windows Credential Manager, macOS Keychain, etc.)
   - Falls back to encrypted file if keyring unavailable

### Install Auth Dependencies

```bash
pip install pr0loader[auth]
# or
pip install keyring pycryptodome pillow
```

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

### 6. Inference API Server

Start a REST API for tag prediction:

```bash
# Start API server on default port 8000
pr0loader api

# Custom host and port
pr0loader api --host 0.0.0.0 --port 8080

# With custom model
pr0loader api --model ./my_model.keras
```

#### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Health check |
| GET | `/docs` | Interactive API documentation (Swagger) |
| POST | `/predict` | Predict tags for single image |
| POST | `/predict/batch` | Predict tags for multiple images |

#### API Usage Examples

```bash
# Single image prediction
curl -X POST "http://localhost:8000/predict" \
    -F "file=@image.jpg" \
    -F "top_k=5"

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
    -F "files=@image1.jpg" \
    -F "files=@image2.jpg"
```

### 7. Gradio Web UI

Launch an interactive web interface for testing:

```bash
# Start Gradio UI (uses local model)
pr0loader ui

# Connect to remote API server
pr0loader ui --api-url http://localhost:8000

# Create public shareable link
pr0loader ui --share
```

### 8. Combined Server (API + UI)

Start both the API server and Gradio UI together:

```bash
# Start both servers
pr0loader serve

# Custom ports
pr0loader serve --api-port 8000 --ui-port 7860
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

### Using uv (Recommended)

```bash
# Create venv with Python 3.12 (auto-detected from .python-version)
uv venv

# Activate venv
.venv\Scripts\activate         # Windows
source .venv/bin/activate      # Linux/Mac

# Install with all dependencies including dev tools
uv pip install -e ".[all,dev]"

# Add a new dependency
uv pip install package-name

# Update dependencies
uv pip install --upgrade -e ".[all,dev]"

# Run tests
pytest

# Format code
black src/
```

### Using pip

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
```

### Quick Reference: uv vs pip

| Task | uv | pip |
|------|----|----|
| Create venv | `uv venv` | `python -m venv .venv` |
| Install package | `uv pip install pkg` | `pip install pkg` |
| Install from pyproject.toml | `uv pip install -e ".[ml]"` | `pip install -e ".[ml]"` |
| Update package | `uv pip install --upgrade pkg` | `pip install --upgrade pkg` |
| List installed | `uv pip list` | `pip list` |
| Uninstall | `uv pip uninstall pkg` | `pip uninstall pkg` |

**Why uv?** 10-100x faster, better dependency resolution, automatic Python version detection.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.


