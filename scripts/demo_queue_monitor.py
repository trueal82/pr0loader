#!/usr/bin/env python3
"""Demo of the queue monitor visualization for download pipeline.

Shows what the ASCII queue status looks like with different fill levels.
"""

from pr0loader.pipeline.download import DownloadPipeline
from pr0loader.config import load_settings

settings = load_settings()
pipeline = DownloadPipeline(settings)

# Demo different queue fill levels
demo_states = [
    (234, 10000, 45, 1000, "Light load"),
    (5234, 10000, 450, 1000, "Moderate load"),
    (9500, 10000, 850, 1000, "Heavy load"),
    (10000, 10000, 1000, 1000, "Saturated"),
]

print("=" * 65)
print("QUEUE MONITOR VISUALIZATION DEMO")
print("=" * 65)
print()

for check_size, check_max, dl_size, dl_max, label in demo_states:
    check_pct = (check_size / check_max * 100)
    dl_pct = (dl_size / dl_max * 100)

    check_bar = pipeline._format_queue_bar(check_size, check_max)
    dl_bar = pipeline._format_queue_bar(dl_size, dl_max)

    print(f"STATE: {label}")
    print("┌─────────────────────────────────────────────────────────┐")
    print("│ QUEUE STATUS                                            │")
    print("├─────────────────────────────────────────────────────────┤")
    print(f"│ Check    {check_bar} {check_size:>6,}/{check_max:<6,} ({check_pct:>3.0f}%)     │")
    print(f"│ Download {dl_bar} {dl_size:>6,}/{dl_max:<6,} ({dl_pct:>3.0f}%)     │")
    print("├─────────────────────────────────────────────────────────┤")
    print(f"│ ✓ 12,345 skip │ ⬇ 234 dl (10.2 MB) │ ✗ 2 fail         │")
    print("└─────────────────────────────────────────────────────────┘")
    print()

print("BAR LEGEND:")
print("  █ = 0-50% full (green zone)")
print("  ▓ = 50-80% full (yellow zone)")
print("  ▒ = 80-100% full (red zone)")
print("  ░ = empty")
print()
print("This visualization appears every 3 seconds in --verbose mode")
print("showing real-time queue depths and download progress.")

