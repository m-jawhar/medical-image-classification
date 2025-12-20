#!/usr/bin/env python3
"""
Data Preparation Helper Script
Checks if dataset exists and guides user through setup process.
"""

import os
from pathlib import Path
import sys


def check_dataset():
    """Check if the dataset has been downloaded."""
    dataset_path = Path("data/raw/chest_xray")

    print("=" * 60)
    print("🏥 Medical Image Classification - Data Setup")
    print("=" * 60)

    # Check if raw data exists
    if not dataset_path.exists():
        print("\n❌ Dataset NOT found!")
        print(f"   Expected location: {dataset_path.absolute()}")
        print("\n📥 Please download the dataset:")
        print(
            "   1. Visit: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"
        )
        print("   2. Download the ZIP file (~2GB)")
        print(f"   3. Extract to: {dataset_path.absolute()}")
        print("\n📁 Expected structure after extraction:")
        print("   data/raw/chest_xray/")
        print("   ├── train/")
        print("   │   ├── NORMAL/")
        print("   │   └── PNEUMONIA/")
        print("   ├── test/")
        print("   │   ├── NORMAL/")
        print("   │   └── PNEUMONIA/")
        print("   └── val/")
        print("       ├── NORMAL/")
        print("       └── PNEUMONIA/")
        return False

    # Check subdirectories
    train_dir = dataset_path / "train"
    test_dir = dataset_path / "test"
    val_dir = dataset_path / "val"

    if not all([train_dir.exists(), test_dir.exists(), val_dir.exists()]):
        print("\n⚠️  Dataset directory exists but structure is incomplete!")
        print(f"   Location: {dataset_path.absolute()}")
        print("\n   Missing directories:")
        if not train_dir.exists():
            print("   ❌ train/")
        if not test_dir.exists():
            print("   ❌ test/")
        if not val_dir.exists():
            print("   ❌ val/")
        return False

    # Count images
    train_normal = (
        list((train_dir / "NORMAL").glob("*.jpeg"))
        if (train_dir / "NORMAL").exists()
        else []
    )
    train_pneumonia = (
        list((train_dir / "PNEUMONIA").glob("*.jpeg"))
        if (train_dir / "PNEUMONIA").exists()
        else []
    )

    if not train_normal and not train_pneumonia:
        print("\n⚠️  Dataset directories exist but no images found!")
        print("   Please ensure you've extracted the images correctly.")
        return False

    print("\n✅ Dataset found!")
    print(f"   Location: {dataset_path.absolute()}")
    print(f"\n📊 Dataset Statistics:")
    print(f"   Training NORMAL images: {len(train_normal)}")
    print(f"   Training PNEUMONIA images: {len(train_pneumonia)}")

    # Check if splits exist
    splits_path = Path("data/splits")
    if splits_path.exists() and (splits_path / "train.csv").exists():
        print(f"\n✅ Data splits already created!")
        print(f"   Location: {splits_path.absolute()}")
        print("\n🚀 You're ready to train!")
        print("   Run: python src/train.py --model resnet50")
        return True
    else:
        print(f"\n⏳ Data splits NOT created yet")
        print(f"\n📝 Next step: Create data splits")
        print("   Option 1: Run preprocessing script")
        print("      python src/data_preprocessing.py")
        print("\n   Option 2: Create splits manually (quick start)")
        print("      python prepare_data.py --create-splits")
        return False


def create_quick_splits():
    """Create quick CSV splits from the dataset."""
    import pandas as pd
    from pathlib import Path

    print("\n🔄 Creating data splits...")

    dataset_path = Path("data/raw/chest_xray")
    splits_path = Path("data/splits")
    splits_path.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split_name in ["train", "test", "val"]:
        split_dir = dataset_path / split_name

        if not split_dir.exists():
            print(f"   ⚠️  Skipping {split_name} (directory not found)")
            continue

        data = []

        # Process NORMAL images
        normal_dir = split_dir / "NORMAL"
        if normal_dir.exists():
            for img_path in normal_dir.glob("*.jpeg"):
                data.append(
                    {
                        "image_path": str(img_path.absolute()),
                        "label": 0,  # NORMAL
                        "class": "NORMAL",
                    }
                )

        # Process PNEUMONIA images
        pneumonia_dir = split_dir / "PNEUMONIA"
        if pneumonia_dir.exists():
            for img_path in pneumonia_dir.glob("*.jpeg"):
                data.append(
                    {
                        "image_path": str(img_path.absolute()),
                        "label": 1,  # PNEUMONIA
                        "class": "PNEUMONIA",
                    }
                )

        # Save to CSV
        if data:
            df = pd.DataFrame(data)
            output_file = splits_path / f"{split_name}.csv"
            df.to_csv(output_file, index=False)
            print(f"   ✅ Created {split_name}.csv ({len(df)} images)")
        else:
            print(f"   ⚠️  No images found for {split_name}")

    print("\n✅ Data splits created successfully!")
    print(f"   Location: {splits_path.absolute()}")
    print("\n🚀 You're ready to train!")
    print("   Run: python src/train.py --model resnet50")


if __name__ == "__main__":
    if "--create-splits" in sys.argv:
        if check_dataset():
            print("\n✅ Splits already exist. Nothing to do.")
        else:
            create_quick_splits()
    else:
        check_dataset()
