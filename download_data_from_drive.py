"""
Simple script to copy data from mounted Google Drive
Usage: Mount Drive in Colab, then run this script
"""

import shutil
from pathlib import Path
import sys

def copy_from_drive(source_path: str, dest_path: str = "data"):
    """Copy data folder from mounted Google Drive"""
    
    source = Path(source_path)
    dest = Path(dest_path)
    
    if not source.exists():
        print(f"❌ Source not found: {source}")
        print("\nMake sure:")
        print("1. You've mounted Google Drive in Colab")
        print("2. The path is correct")
        print("\nExample paths:")
        print("  /content/drive/MyDrive/LLM_Data/data")
        print("  /content/drive/MyDrive/fineweb")
        return False
    
    print(f"Copying from: {source}")
    print(f"Copying to: {dest}")
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy entire directory
    if source.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest)
        print(f"\n✓ Copied directory successfully!")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        print(f"\n✓ Copied file successfully!")
    
    # Show size
    if dest.is_dir():
        total_size = sum(f.stat().st_size for f in dest.rglob('*') if f.is_file())
        print(f"Total size: {total_size / 1e9:.2f} GB")
    else:
        print(f"File size: {dest.stat().st_size / 1e9:.2f} GB")
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path in Google Drive (e.g., /content/drive/MyDrive/LLM_Data/data)")
    parser.add_argument("--dest", default="data", help="Destination folder (default: data)")
    
    args = parser.parse_args()
    success = copy_from_drive(args.source, args.dest)
    sys.exit(0 if success else 1)

