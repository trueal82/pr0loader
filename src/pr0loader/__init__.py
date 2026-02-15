"""
pr0loader - A toolchain for fetching, processing, and training ML models on pr0gramm data.

Pipeline stages:
1. fetch    - Download metadata from pr0gramm API
2. download - Download media files (images/videos)
3. prepare  - Prepare dataset for ML training (filter, clean, export CSV)
4. train    - Train a tag prediction model
5. predict  - Predict tags for new images
"""

__version__ = "2.0.0"
__author__ = "pr0loader team"

