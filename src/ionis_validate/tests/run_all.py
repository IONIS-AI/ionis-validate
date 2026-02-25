#!/usr/bin/env python3
"""
run_all.py — IONIS V22-gamma Complete Test Suite Orchestrator

Runs all test groups and produces a consolidated summary.

Test Groups:
  KI7MT Override: 18 operator-grounded tests (4 gates)
  TST-900: Band x Time Discrimination (11 tests)

Total: 29 tests

Usage:
  python run_all.py
  ionis-validate test
"""

import subprocess
import sys
import os
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Test modules in execution order
TEST_MODULES = [
    ("KI7MT", "Override Validation", "test_ki7mt_override.py", 18),
    ("TST-900", "Band x Time Discrimination", "test_tst900_band_time.py", 11),
]


def run_test_module(script_name: str) -> tuple[bool, str]:
    """Run a test module and capture output."""
    script_path = os.path.join(SCRIPT_DIR, script_name)

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per module
        )
        output = result.stdout + result.stderr
        passed = result.returncode == 0
        return passed, output
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: Test module exceeded 5 minute limit"
    except Exception as e:
        return False, f"ERROR: {e}"


def main():
    print("=" * 70)
    print("  IONIS V22-gamma + PhysicsOverrideLayer — Complete Test Suite")
    print("=" * 70)
    print(f"\n  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model: IONIS V22-gamma")
    print(f"  Checkpoint: ionis_v22_gamma.safetensors")
    print()

    total_tests = sum(count for _, _, _, count in TEST_MODULES)
    print(f"  Running {len(TEST_MODULES)} test groups ({total_tests} total tests)...")
    print()

    results = []
    all_outputs = []

    for group_id, group_name, script, test_count in TEST_MODULES:
        print(f"  [{group_id}] {group_name} ({test_count} tests)...", end=" ", flush=True)

        passed, output = run_test_module(script)
        results.append((group_id, group_name, test_count, passed))
        all_outputs.append((group_id, output))

        status = "PASS" if passed else "FAIL"
        print(status)

    # Summary
    print()
    print("=" * 70)
    print("  TEST SUITE SUMMARY")
    print("=" * 70)
    print()

    passed_groups = sum(1 for _, _, _, p in results if p)
    passed_tests = sum(count for _, _, count, p in results if p)

    print(f"  {'Group':<10s}  {'Description':<30s}  {'Tests':>6s}  {'Status':>8s}")
    print(f"  {'-'*60}")

    for group_id, group_name, test_count, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {group_id:<10s}  {group_name:<30s}  {test_count:>6d}  {status:>8s}")

    print(f"  {'-'*60}")
    print(f"  {'TOTAL':<10s}  {'':<30s}  {total_tests:>6d}  {passed_tests}/{total_tests}")
    print()

    if passed_groups == len(TEST_MODULES):
        print("  " + "=" * 50)
        print("  ALL TEST GROUPS PASSED")
        print("  " + "=" * 50)
        print()
        print("  IONIS V22-gamma + PhysicsOverrideLayer validated.")
        print("  Model is production-ready.")
        print()
        return 0
    else:
        failed_groups = [g for g, _, _, p in results if not p]
        print("  " + "=" * 50)
        print(f"  {len(failed_groups)} TEST GROUP(S) FAILED: {', '.join(failed_groups)}")
        print("  " + "=" * 50)
        print()
        print("  Review individual test output for details.")
        print()

        # Show failed group outputs
        for group_id, output in all_outputs:
            if any(g == group_id and not p for g, _, _, p in results):
                print(f"\n  --- {group_id} Output ---")
                print(output)

        return 1


if __name__ == "__main__":
    sys.exit(main())
