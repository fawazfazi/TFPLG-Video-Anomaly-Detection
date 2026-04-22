"""
Script 1: Extract CLIP Features from Video Frames
===================================================
This script:
- Reads videos from the mini dataset
- Extracts frames every N frames
- Extracts CLIP visual features from each frame
- Saves features as .npy files for later use
"""

import os
import cv2
import numpy as np
import torch
import clip
from tqdm import tqdm
import json

# ─── CONFIG ───────────────────────────────────────────────────────────────────
VIDEO_DIR   = "data/videos"          # folder with Abuse/, Arrest/, etc.
FEATURE_DIR = "features"             # where .npy files will be saved
FRAME_SKIP  = 10                     # extract 1 frame every 10 frames (adjust if slow)
IMG_SIZE    = 224                    # CLIP input size
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"  # auto-detect GPU
# ──────────────────────────────────────────────────────────────────────────────

def load_clip_model():
    print(f"[INFO] Loading CLIP model (ViT-B/16) on {DEVICE}...")
    model, preprocess = clip.load("ViT-B/16", device=DEVICE)
    model.eval()
    print("[INFO] CLIP model loaded successfully!")
    return model, preprocess


def extract_frames(video_path, frame_skip=10):
    """Extract frames from a video file."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[WARNING] Could not open video: {video_path}")
        return frames
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        frame_idx += 1
    
    cap.release()
    return frames


def extract_clip_features(frames, model, preprocess):
    """Extract CLIP features from a list of frames."""
    from PIL import Image
    
    features = []
    with torch.no_grad():
        for i, frame in enumerate(frames):
            pil_img = Image.fromarray(frame)
            img_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
            feat = model.encode_image(img_tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)  # normalize
            features.append(feat.cpu().numpy())
            if (i + 1) % 10 == 0:
                print(f"    Extracted {i + 1}/{len(frames)} frames...")
    
    if len(features) == 0:
        return None
    
    return np.vstack(features)  # shape: (num_frames, 512)


def main():
    os.makedirs(FEATURE_DIR, exist_ok=True)
    
    # Load CLIP
    model, preprocess = load_clip_model()
    
    # Get all categories
    categories = [d for d in os.listdir(VIDEO_DIR) 
                  if os.path.isdir(os.path.join(VIDEO_DIR, d))]
    print(f"[INFO] Found categories: {categories}")
    
    video_info = {}  # store metadata
    
    for category in sorted(categories):
        cat_dir = os.path.join(VIDEO_DIR, category)
        feat_cat_dir = os.path.join(FEATURE_DIR, category)
        os.makedirs(feat_cat_dir, exist_ok=True)
        
        video_files = [f for f in os.listdir(cat_dir) 
                       if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
        
        print(f"\n[INFO] Processing category: {category} ({len(video_files)} videos)")
        
        for video_file in tqdm(video_files, desc=category):
            video_path = os.path.join(cat_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            feat_path  = os.path.join(feat_cat_dir, f"{video_name}.npy")
            
            # Skip if already extracted
            if os.path.exists(feat_path):
                print(f"  [SKIP] Already extracted: {video_name}")
                continue
            
            # Extract frames
            frames = extract_frames(video_path, frame_skip=FRAME_SKIP)
            
            if len(frames) == 0:
                print(f"  [WARNING] No frames extracted from: {video_name}")
                continue
            
            # Extract CLIP features
            features = extract_clip_features(frames, model, preprocess)
            
            if features is None:
                continue
            
            # Save features
            np.save(feat_path, features)
            
            # Store metadata
            video_info[video_name] = {
                "category": category,
                "num_frames": len(frames),
                "feature_shape": list(features.shape),
                "feature_path": feat_path,
                "label": 0 if category.lower() == "normal" else 1
            }
            
            print(f"  [SAVED] {video_name}: {features.shape}")
    
    # Save metadata
    meta_path = os.path.join(FEATURE_DIR, "video_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(video_info, f, indent=2)
    
    print(f"\n[DONE] Features extracted for {len(video_info)} videos.")
    print(f"[DONE] Metadata saved to: {meta_path}")
    print(f"[DONE] Features saved to: {FEATURE_DIR}/")


if __name__ == "__main__":
    main()
