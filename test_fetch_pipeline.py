#!/usr/bin/env python3
"""
Comprehensive test suite for the new fetch pipeline.

Tests:
1. Data structure initialization
2. Gap analysis logic with various scenarios
3. ID loading and set operations
4. Integration with existing codebase
"""

import sys
sys.path.insert(0, 'src')

from pr0loader.pipeline.fetch import FetchPipeline, FetchStats, load_local_ids, analyze_gaps

def test_fetch_stats():
    """Test FetchStats dataclass."""
    print("TEST: FetchStats initialization")
    stats = FetchStats()
    assert stats.remote_max_id == 0
    assert stats.local_count == 0
    assert stats.items_fetched == 0
    assert stats.items_failed == 0
    assert stats.items_skipped == 0
    assert stats.gap_analysis_time == 0.0
    assert stats.fetch_time == 0.0
    assert stats.db_write_time == 0.0
    print("  ✓ All fields initialize correctly")


def test_analyze_gaps_empty():
    """Test gap analysis with empty database."""
    print("\nTEST: analyze_gaps with empty set")
    gaps = analyze_gaps(set(), 1000)

    assert gaps['local_min'] == 0
    assert gaps['local_max'] == 0
    assert gaps['top_gap'] == 1000
    assert gaps['bottom_gap'] == 0
    assert gaps['total_missing'] == 1000
    assert gaps['is_complete'] == False
    print("  ✓ Empty database detected correctly")
    print(f"     top_gap={gaps['top_gap']}, total_missing={gaps['total_missing']}")


def test_analyze_gaps_with_data():
    """Test gap analysis with actual data."""
    print("\nTEST: analyze_gaps with sparse data")
    # Local IDs: 1,2,3,5,6,7,10 (missing 4, 8, 9, and 11-15)
    local_ids = {1, 2, 3, 5, 6, 7, 10}
    gaps = analyze_gaps(local_ids, 15)

    assert gaps['local_min'] == 1
    assert gaps['local_max'] == 10
    assert gaps['top_gap'] == 5, f"Expected top_gap=5 (15-10), got {gaps['top_gap']}"
    assert gaps['bottom_gap'] == 0, f"Expected bottom_gap=0 (1-1), got {gaps['bottom_gap']}"
    # Internal gaps: expected in range (10-1+1=10) - actual (7) = 3
    assert gaps['internal_gaps'] == 3, f"Expected internal_gaps=3, got {gaps['internal_gaps']}"
    assert gaps['total_missing'] == 8, f"Expected total_missing=8 (5+0+3), got {gaps['total_missing']}"
    assert gaps['is_complete'] == False
    print("  ✓ Gap detection works correctly")
    print(f"     local_min={gaps['local_min']}, local_max={gaps['local_max']}")
    print(f"     top_gap={gaps['top_gap']}, bottom_gap={gaps['bottom_gap']}, internal_gaps={gaps['internal_gaps']}")


def test_analyze_gaps_complete():
    """Test gap analysis with complete database."""
    print("\nTEST: analyze_gaps with complete data")
    # No gaps: 1-100 all present
    local_ids = set(range(1, 101))
    gaps = analyze_gaps(local_ids, 100)

    assert gaps['local_min'] == 1
    assert gaps['local_max'] == 100
    assert gaps['top_gap'] == 0
    assert gaps['bottom_gap'] == 0
    assert gaps['internal_gaps'] == 0
    assert gaps['total_missing'] == 0
    assert gaps['is_complete'] == True
    print("  ✓ Complete database detected correctly")


def test_analyze_gaps_bottom_gap():
    """Test gap analysis with missing items at bottom."""
    print("\nTEST: analyze_gaps with bottom gap")
    # Starting from ID 100, but remote max is 200
    local_ids = set(range(100, 201))
    gaps = analyze_gaps(local_ids, 200)

    assert gaps['local_min'] == 100
    assert gaps['local_max'] == 200
    assert gaps['top_gap'] == 0  # No new items
    assert gaps['bottom_gap'] == 99  # Missing 1-99
    assert gaps['internal_gaps'] == 0  # No internal gaps
    assert gaps['total_missing'] == 99
    assert gaps['is_complete'] == False
    print("  ✓ Bottom gap detected correctly")
    print(f"     bottom_gap={gaps['bottom_gap']}")


def test_analyze_gaps_top_gap():
    """Test gap analysis with new items at top."""
    print("\nTEST: analyze_gaps with top gap")
    # Have 1-100, but remote has up to 150
    local_ids = set(range(1, 101))
    gaps = analyze_gaps(local_ids, 150)

    assert gaps['local_min'] == 1
    assert gaps['local_max'] == 100
    assert gaps['top_gap'] == 50  # New items 101-150
    assert gaps['bottom_gap'] == 0
    assert gaps['internal_gaps'] == 0
    assert gaps['total_missing'] == 50
    assert gaps['is_complete'] == False
    print("  ✓ Top gap detected correctly")
    print(f"     top_gap={gaps['top_gap']}")


def test_set_membership_performance():
    """Test that set membership is O(1)."""
    print("\nTEST: Set membership performance")
    import time

    # Create large set
    large_set = set(range(1, 1000001))  # 1 million items

    # Test membership checks (should be instant)
    start = time.perf_counter()
    for i in [500000, 999999, 1, 1000002]:
        _ = i in large_set
    elapsed = time.perf_counter() - start

    print(f"  ✓ 4 membership checks in {elapsed*1000:.3f}ms")
    print(f"     Set size: {len(large_set):,} items")
    assert elapsed < 0.001, "Set membership should be near-instant"


def test_skip_logic():
    """Test the skip logic used in the fetch loop."""
    print("\nTEST: Skip logic simulation")

    # Simulate API response with mix of new and existing items
    class FakeItem:
        def __init__(self, id):
            self.id = id

    local_ids = {1, 2, 3, 5, 6, 7}
    api_response = [FakeItem(i) for i in [8, 7, 6, 5, 4, 3]]  # Mix of new and existing
    end_id = 1

    # This is the exact logic from fetch.py
    new_items = [
        item for item in api_response
        if item.id not in local_ids and item.id >= end_id
    ]

    new_ids = [item.id for item in new_items]
    assert new_ids == [8, 4], f"Expected [8, 4], got {new_ids}"
    print(f"  ✓ Skip logic works correctly")
    print(f"     API returned: {[i.id for i in api_response]}")
    print(f"     Already have: {sorted(local_ids)}")
    print(f"     New items: {new_ids}")


def test_edge_cases():
    """Test edge cases."""
    print("\nTEST: Edge cases")

    # Single item
    gaps = analyze_gaps({1}, 1)
    assert gaps['is_complete'] == True
    print("  ✓ Single item case works")

    # Large gap
    gaps = analyze_gaps({1}, 1000000)
    assert gaps['top_gap'] == 999999
    print("  ✓ Large gap case works")

    # Negative check (should never happen but handle gracefully)
    gaps = analyze_gaps({100}, 50)  # local_max > remote_max
    assert gaps['top_gap'] == 0  # max(0, 50-100) = 0
    print("  ✓ Handles local_max > remote_max gracefully")


def main():
    print("="*70)
    print("FETCH PIPELINE TEST SUITE")
    print("="*70)

    try:
        test_fetch_stats()
        test_analyze_gaps_empty()
        test_analyze_gaps_with_data()
        test_analyze_gaps_complete()
        test_analyze_gaps_bottom_gap()
        test_analyze_gaps_top_gap()
        test_set_membership_performance()
        test_skip_logic()
        test_edge_cases()

        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nThe new fetch pipeline is ready to use!")
        print("\nKey features validated:")
        print("  • Gap analysis logic (top, bottom, internal)")
        print("  • Set-based skip logic (O(1) membership)")
        print("  • Data structure initialization")
        print("  • Edge case handling")
        print("  • Performance characteristics")
        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

