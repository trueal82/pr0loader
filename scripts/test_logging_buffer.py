#!/usr/bin/env python3
"""Test script to verify log buffering is working."""

import asyncio
import logging
from collections import deque

# Simulate the buffer
_LOG_BUFFER = deque(maxlen=10)

class _BufferingHandler(logging.Handler):
    """Custom logging handler that buffers ALL messages for Live display."""
    def emit(self, record):
        try:
            msg = self.format(record)
            _LOG_BUFFER.append(msg)
            print(f"[HANDLER] Buffered: {msg}")
        except Exception as e:
            print(f"[HANDLER ERROR] {e}")
            self.handleError(record)

# Set up logger
logger = logging.getLogger('test_download')

# Remove all handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add buffering handler
handler = _BufferingHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger.propagate = False

print(f"Logger level: {logger.level} ({logging.getLevelName(logger.level)})")
print(f"Logger effective level: {logger.getEffectiveLevel()}")
print(f"Logger has handlers: {len(logger.handlers)}")
print(f"Is DEBUG enabled: {logger.isEnabledFor(logging.DEBUG)}")
print()

# Test logging
print("=== Testing logger.debug() ===")
logger.debug("Test message 1")
logger.debug("Test message 2")
logger.debug("SKIP(exists_no_verify) 2024/02/file.jpg")

print(f"\nBuffer contents ({len(_LOG_BUFFER)} items):")
for i, msg in enumerate(_LOG_BUFFER):
    print(f"  [{i}] {msg}")

print("\n=== RESULT ===")
if len(_LOG_BUFFER) > 0:
    print("✓ Buffering is working!")
else:
    print("✗ Buffer is empty - logging not working")

