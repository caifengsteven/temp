#!/usr/bin/env python3
"""
Test runner for HSTECH Index Estimation System.

This script runs all tests and generates a coverage report.
"""

import sys
import subprocess
from pathlib import Path


def run_tests():
    """Run all tests with coverage reporting."""
    
    print("HSTECH Index Estimation System - Test Runner")
    print("=" * 50)
    
    # Add src to Python path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    try:
        # Run tests with pytest and coverage
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/",
            "-v",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--tb=short"
        ]
        
        print("Running tests with coverage...")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 50)
        
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode == 0:
            print("\n" + "=" * 50)
            print("✓ All tests passed!")
            print("✓ Coverage report generated in htmlcov/")
            print("=" * 50)
        else:
            print("\n" + "=" * 50)
            print("✗ Some tests failed!")
            print("=" * 50)
            sys.exit(1)
            
    except FileNotFoundError:
        print("Error: pytest not found. Please install test dependencies:")
        print("pip install pytest pytest-cov")
        sys.exit(1)
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)


def run_quick_tests():
    """Run tests without coverage for quick feedback."""
    
    print("HSTECH Index Estimation System - Quick Test Run")
    print("=" * 50)
    
    # Add src to Python path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    try:
        cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]
        
        print("Running quick tests...")
        print("-" * 50)
        
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode == 0:
            print("\n✓ All tests passed!")
        else:
            print("\n✗ Some tests failed!")
            sys.exit(1)
            
    except FileNotFoundError:
        print("Error: pytest not found. Please install test dependencies:")
        print("pip install pytest")
        sys.exit(1)


def run_specific_test(test_file):
    """Run a specific test file."""
    
    print(f"Running specific test: {test_file}")
    print("-" * 50)
    
    # Add src to Python path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    try:
        cmd = [sys.executable, "-m", "pytest", f"tests/{test_file}", "-v"]
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode != 0:
            sys.exit(1)
            
    except FileNotFoundError:
        print("Error: pytest not found. Please install pytest")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            run_quick_tests()
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python run_tests.py           # Run all tests with coverage")
            print("  python run_tests.py --quick   # Run tests without coverage")
            print("  python run_tests.py test_models.py  # Run specific test file")
        elif sys.argv[1].startswith("test_"):
            run_specific_test(sys.argv[1])
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
            sys.exit(1)
    else:
        run_tests()
