#!/usr/bin/env python
"""
Environment Setup Script

This script creates a virtual environment and installs the required dependencies
for the E-commerce Recommendation System project.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def setup_environment():
    """Set up a virtual environment and install dependencies"""
    print("=" * 80)
    print("E-commerce Recommendation System - Environment Setup")
    print("=" * 80)
    
    # Get the project root directory
    root_dir = Path(__file__).resolve().parent
    requirements_file = root_dir / "requirements.txt"
    
    # Check if requirements file exists
    if not requirements_file.exists():
        print(f"Error: Requirements file not found at {requirements_file}")
        return 1
    
    # Create a virtual environment
    venv_dir = root_dir / "venv"
    if venv_dir.exists():
        print(f"Virtual environment already exists at {venv_dir}")
        use_existing = input("Use existing environment? (y/n) [y]: ").strip().lower() or 'y'
        if use_existing != 'y':
            print("Creating a new virtual environment...")
            try:
                import shutil
                shutil.rmtree(venv_dir)
            except Exception as e:
                print(f"Error removing existing environment: {str(e)}")
                return 1
        else:
            print("Using existing virtual environment.")
    
    # Create virtual environment if it doesn't exist
    if not venv_dir.exists():
        print("Creating virtual environment...")
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {str(e)}")
            return 1
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return 1
    
    # Determine path to pip based on OS
    if platform.system() == "Windows":
        pip_path = venv_dir / "Scripts" / "pip"
        activate_path = venv_dir / "Scripts" / "activate"
    else:
        pip_path = venv_dir / "bin" / "pip"
        activate_path = venv_dir / "bin" / "activate"
    
    # Install dependencies
    print("\nInstalling dependencies...")
    try:
        subprocess.run([str(pip_path), "install", "-U", "pip"], check=True)
        subprocess.run([str(pip_path), "install", "-r", str(requirements_file)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {str(e)}")
        return 1
    
    print("\n" + "=" * 80)
    print("Environment setup complete!")
    print("=" * 80)
    
    print("\nTo activate the virtual environment:")
    if platform.system() == "Windows":
        print(f"    {venv_dir}\\Scripts\\activate")
    else:
        print(f"    source {venv_dir}/bin/activate")
    
    print("\nTo run the application:")
    print("    python run_app.py")
    
    print("\nTo run the tests:")
    print("    python run_tests.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(setup_environment()) 