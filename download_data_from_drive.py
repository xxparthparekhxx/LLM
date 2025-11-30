"""
Simple script to download data folder from Google Drive
Usage: python download_data_from_drive.py --gdrive-id YOUR_FOLDER_ID
"""

import argparse
import subprocess
import sys
import os

def download_from_drive(gdrive_id: str, output_path: str = "data"):
    """Download folder/file from Google Drive"""
    
    # Install gdown if not available
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
        import gdown
    
    print(f"Downloading from Google Drive (ID: {gdrive_id})...")
    print(f"Output: {output_path}")
    
    # Download folder
    url = f"https://drive.google.com/drive/folders/{gdrive_id}"
    gdown.download_folder(url, output=output_path, quiet=False, use_cookies=False)
    
    print(f"\n✓ Download complete: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdrive-id", required=True, help="Google Drive folder ID")
    parser.add_argument("--output", default="data", help="Output folder (default: data)")
    
    args = parser.parse_args()
    download_from_drive(args.gdrive_id, args.output)
