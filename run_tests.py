#!/usr/bin/env python
"""
Test Runner Script

This script executes all tests for the E-commerce Recommendation System
and generates a simple report of the results.
"""

import os
import sys
import pytest
import time
from pathlib import Path

def run_tests():
    """Run all tests and report results"""
    print("=" * 80)
    print("E-commerce Recommendation System - Test Suite")
    print("=" * 80)
    
    # Get the project root directory
    root_dir = Path(__file__).resolve().parent
    
    # Check if the test directory exists
    test_dir = root_dir / "tests"
    if not test_dir.exists() or not test_dir.is_dir():
        print(f"Error: Test directory not found at {test_dir}")
        return 1
    
    # Start timing
    start_time = time.time()
    
    # Run tests with pytest
    print("\nRunning tests...")
    result = pytest.main(["-v", str(test_dir)])
    
    # End timing
    duration = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"Test execution completed in {duration:.2f} seconds")
    
    if result == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed. Please check the output above for details.")
    
    print("\nTest modules:")
    for test_file in test_dir.glob("test_*.py"):
        print(f"  - {test_file.name}")
    
    print("\nTo run a specific test module:")
    print(f"  python -m pytest tests/test_file_name.py -v")
    
    return result

if __name__ == "__main__":
    sys.exit(run_tests()) 