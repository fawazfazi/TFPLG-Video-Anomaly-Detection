"""
Setup Script — Run this FIRST before anything else
===================================================
Creates folder structure and installs requirements
"""

import os
import subprocess
import sys

def create_folders():
    folders = [
        "data/videos/Abuse",
        "data/videos/Arrest", 
        "data/videos/Assault",
        "data/videos/Burglary",
        "data/videos/Fighting",
        "data/videos/normal",
        "data/annotations",
        "features",
        "results/baseline",
        "results/improved",
        "results/comparison"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"[OK] Created: {folder}")

def install_requirements():
    packages = [
        "torch",
        "torchvision", 
        "opencv-python",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "Pillow",
        "ftfy",
        "regex"
    ]
    print("\n[INFO] Installing required packages...")
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
        print(f"[OK] Installed: {pkg}")
    
    # Install CLIP separately
    print("[INFO] Installing CLIP...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "git+https://github.com/openai/CLIP.git", "-q"
    ])
    print("[OK] Installed: CLIP")

if __name__ == "__main__":
    print("=" * 50)
    print("Setting up TFPLG Project")
    print("=" * 50)
    
    print("\n[STEP 1] Creating folder structure...")
    create_folders()
    
    print("\n[STEP 2] Installing packages...")
    install_requirements()
    
    print("\n" + "=" * 50)
    print("SETUP COMPLETE!")
    print("=" * 50)
    print("""
NEXT STEPS:
─────────────────────────────────────────────────
1. Place mini dataset videos in:
   data/videos/Abuse/     ← .mp4 files
   data/videos/Arrest/
   data/videos/Assault/
   data/videos/Burglary/
   data/videos/Fighting/
   data/videos/normal/

2. Place UCA JSON files in:
   data/annotations/UCFCrime_Train.json
   data/annotations/UCFCrime_Test.json
   data/annotations/UCFCrime_Val.json

3. Then run scripts in order:
   python 01_extract_features.py
   python 02_baseline.py
   python 03_improved.py
   python 04_compare.py
─────────────────────────────────────────────────
""")
