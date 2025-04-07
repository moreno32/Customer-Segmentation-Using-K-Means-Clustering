#!/usr/bin/env python
"""
Application Runner Script

This script launches the E-commerce Recommendation System Streamlit application
with proper error handling and checks.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_app():
    """Run the Streamlit application"""
    print("=" * 80)
    print("E-commerce Recommendation System - Application Runner")
    print("=" * 80)
    
    # Get the project root directory
    root_dir = Path(__file__).resolve().parent
    app_path = root_dir / "app" / "main.py"
    
    # Check if the app file exists
    if not app_path.exists():
        print(f"Error: Application file not found at {app_path}")
        return 1
    
    # Check if required directories exist, create if not
    for dir_name in ["data", "data/raw", "data/processed"]:
        dir_path = root_dir / dir_name
        if not dir_path.exists():
            print(f"Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"Using Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("Error: Streamlit not found. Please install required dependencies:")
        print("pip install -r requirements.txt")
        return 1
    
    print("\nStarting the application...")
    print("=" * 80)
    print("URL: http://localhost:8501")
    print("=" * 80)
    print("\nPress Ctrl+C to stop the application")
    
    # Run the Streamlit app
    try:
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.headless", "true"]
        process = subprocess.run(cmd)
        return process.returncode
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
        return 0
    except Exception as e:
        print(f"\nError launching application: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(run_app()) 