#!/usr/bin/env python3
"""
Setup script for Medical Image Classification project.
Creates necessary directories and checks environment.
"""

import os
import sys
from pathlib import Path
import subprocess


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_success(text):
    """Print success message."""
    print(f"✅ {text}")


def print_error(text):
    """Print error message."""
    print(f"❌ {text}")


def print_info(text):
    """Print info message."""
    print(f"ℹ️  {text}")


def check_python_version():
    """Check if Python version is compatible."""
    print_header("Checking Python Version")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    print(f"Current Python version: {version_str}")

    if version.major >= 3 and version.minor >= 8:
        print_success("Python version is compatible")
        return True
    else:
        print_error(f"Python 3.8+ required, found {version_str}")
        return False


def create_directories():
    """Create necessary project directories."""
    print_header("Creating Project Directories")

    directories = [
        "data/raw",
        "data/processed",
        "data/splits",
        "models/architectures",
        "models/pretrained",
        "models/saved_models",
        "results/visualizations",
        "results/evaluation",
        "results/predictions",
        "results/explanations",
        "logs",
    ]

    created_count = 0
    existing_count = 0

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print_success(f"Created: {directory}")
            created_count += 1

            # Create .gitkeep file
            gitkeep = dir_path / ".gitkeep"
            gitkeep.touch()
        else:
            existing_count += 1

    print_info(f"Created {created_count} directories, {existing_count} already existed")
    return True


def check_dependencies():
    """Check if dependencies are installed."""
    print_header("Checking Dependencies")

    required_packages = [
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "matplotlib",
        "opencv-python",
        "pillow",
        "scikit-learn",
        "pyyaml",
        "streamlit",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print_success(f"{package} is installed")
        except ImportError:
            print_error(f"{package} is NOT installed")
            missing_packages.append(package)

    if missing_packages:
        print_info(f"\nMissing {len(missing_packages)} packages")
        print_info("Run: pip install -r requirements.txt")
        return False
    else:
        print_success("All dependencies are installed")
        return True


def check_gpu():
    """Check GPU availability."""
    print_header("Checking GPU Availability")

    try:
        import torch

        if torch.cuda.is_available():
            print_success("CUDA is available")
            print_info(f"GPU Device: {torch.cuda.get_device_name()}")
            print_info(f"CUDA Version: {torch.version.cuda}")
            return True
        else:
            print_info("CUDA is NOT available")
            print_info("Training will use CPU (slower)")
            return False
    except ImportError:
        print_error("PyTorch not installed")
        return False


def verify_config():
    """Verify configuration file exists."""
    print_header("Verifying Configuration")

    config_path = Path("config/config.yaml")

    if config_path.exists():
        print_success("Configuration file found")
        return True
    else:
        print_error("Configuration file not found")
        print_info("Check: config/config.yaml")
        return False


def display_next_steps():
    """Display next steps for user."""
    print_header("Setup Complete!")

    print("\n📌 Next Steps:\n")

    print("1. Download Dataset:")
    print(
        "   URL: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"
    )
    print("   Extract to: data/raw/chest_xray/\n")

    print("2. Explore Data:")
    print("   jupyter notebook notebooks/01_data_exploration.ipynb\n")

    print("3. Preprocess Data:")
    print("   python src/data_preprocessing.py\n")

    print("4. Train Model:")
    print("   python src/train.py --model resnet50\n")

    print("5. Evaluate Model:")
    print("   python src/evaluate.py\n")

    print("6. Run Web App:")
    print("   streamlit run app.py\n")

    print("📚 Documentation:")
    print("   - README.md - Project overview")
    print("   - GETTING_STARTED.md - Detailed setup guide")
    print("   - QUICK_REFERENCE.md - Command reference\n")


def main():
    """Main setup function."""
    print("\n" + "🏥 " + "Medical Image Classification - Setup Script" + " 🏥")

    # Run all checks
    checks = {
        "Python Version": check_python_version(),
        "Directories": create_directories(),
        "Configuration": verify_config(),
    }

    # Check dependencies (don't block if missing)
    print_info("\nChecking optional components...")
    check_dependencies()
    check_gpu()

    # Summary
    print_header("Setup Summary")

    all_passed = all(checks.values())

    for check_name, result in checks.items():
        if result:
            print_success(f"{check_name}: OK")
        else:
            print_error(f"{check_name}: FAILED")

    print("\n")

    if all_passed:
        display_next_steps()
        return 0
    else:
        print_error("Some checks failed. Please resolve the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
