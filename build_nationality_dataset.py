# build_nationality_dataset.py
"""
Build a nationality dataset from FairFace downloads.

Usage:
    python build_nationality_dataset.py \
        --fairface_dir fairface \
        --out_dir data/nationality \
        --max_per_class 3000

Inputs expected in fairface_dir:
    - train/                  (folder of images)
    - val/                    (folder of images)
    - train_labels.csv
    - val_labels.csv

The script maps FairFace race -> your classes:
    'Indian'  <- FairFace race == "Indian"
    'United States' <- FairFace race == "White"
    'African' <- FairFace race == "Black"
    'Other'   <- all other races

It crops faces using bounding box info from the CSV (supports several bbox column name variants).
"""

import os
import argparse
import pandas as pd
from PIL import Image
import shutil
from collections import defaultdict
import math

# tqdm is optional (nice progress bar). If missing, script still runs.
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x: x

# Mapping FairFace race -> your nationality buckets
RACE_TO_CLASS = {
    "Indian": "Indian",
    "White": "United States",
    "Black": "African",
    # everything else -> Other
}

# Common bbox column name groups to try:
BBOX_SETS = [
    # x,y,w,h
    ("x", "y", "w", "h"),
    ("left", "top", "width", "height"),
    ("face_x", "face_y", "face_w", "face_h"),
    # x1,y1,x2,y2
    ("x1", "y1", "x2", "y2"),
    ("xmin", "ymin", "xmax", "ymax"),
]

IMAGE_COLUMNS = ["file", "image", "img", "filename", "path"]


def find_image_column(df):
    for col in IMAGE_COLUMNS:
        if col in df.columns:
            return col
    # try 'original_image' or 'file_name'
    for col in df.columns:
        if "file" in col.lower() or "image" in col.lower() or "img" in col.lower():
            return col
    raise ValueError("Could not find image filename column in CSV. Columns: " + ", ".join(df.columns))


def find_bbox_columns(df):
    for s in BBOX_SETS:
        if all(c in df.columns for c in s):
            return s
    # attempt to infer from any numeric columns that look like bbox
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 4:
        # take first 4 numeric columns (risky)
        return tuple(numeric_cols[:4])
    return None


def map_race_to_class(race_value):
    if pd.isna(race_value):
        return "Other"
    r = str(race_value).strip()
    return RACE_TO_CLASS.get(r, "Other")


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def crop_bbox_from_image(img: Image.Image, bbox, is_xywh=True, normalized=False):
    """
    bbox: (x,y,w,h) if is_xywh True OR (x1,y1,x2,y2) if is_xywh False
    normalized: whether coords are 0..1 (fractions)
    Returns a PIL.Image crop (clamped to image bounds).
    """
    w_img, h_img = img.size
    if is_xywh:
        x, y, w, h = bbox
        if normalized:
            x = x * w_img
            y = y * h_img
            w = w * w_img
            h = h * h_img
        # convert to x1,y1,x2,y2
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))
    else:
        x1, y1, x2, y2 = bbox
        if normalized:
            x1 = int(round(x1 * w_img))
            y1 = int(round(y1 * h_img))
            x2 = int(round(x2 * w_img))
            y2 = int(round(y2 * h_img))
        else:
            x1 = int(round(x1))
            y1 = int(round(y1))
            x2 = int(round(x2))
            y2 = int(round(y2))

    # clamp
    x1 = max(0, min(x1, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    x2 = max(0, min(x2, w_img))
    y2 = max(0, min(y2, h_img))

    # if box degenerate, expand a little
    if x2 <= x1 or y2 <= y1:
        # return center crop fallback (square)
        side = min(w_img, h_img)
        cx, cy = w_img // 2, h_img // 2
        half = side // 4
        return img.crop((cx - half, cy - half, cx + half, cy + half))

    return img.crop((x1, y1, x2, y2))


def process_csv(df, images_root, out_dir, class_counts, max_per_class=None, verbose=False):
    """
    df: labels DataFrame
    images_root: where train/ or val/ images live
    out_dir: base output dir (contains class subdirs)
    class_counts: dict to keep counts across calls
    """
    # find columns
    image_col = find_image_column(df)
    bbox_cols = find_bbox_columns(df)  # may be None
    has_bbox = bbox_cols is not None

    if verbose:
        print(f"Using image column: {image_col}")
        print(f"Using bbox columns: {bbox_cols}")

    for idx in tqdm(range(len(df))):
        row = df.iloc[idx]
        img_name = str(row[image_col]).strip()
        # file could be just filename or a path (some csvs list relative paths)
        img_path = os.path.join(images_root, img_name)
        if not os.path.exists(img_path):
            # try stripping folders; maybe csv contains subfolders or absolute paths
            base = os.path.basename(img_name)
            alt = os.path.join(images_root, base)
            if os.path.exists(alt):
                img_path = alt
            else:
                # cannot find image -> skip
                continue

        # map race to class
        race_val = row.get("race") if "race" in df.columns else row.get("ethnicity") if "ethnicity" in df.columns else None
        cls = map_race_to_class(race_val)

        # check max_per_class
        if max_per_class is not None and class_counts[cls] >= max_per_class:
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # corrupted image? skip
            continue

        # try to crop face
        crop = None
        if has_bbox:
            c0, c1, c2, c3 = bbox_cols
            try:
                v0 = row[c0]; v1 = row[c1]; v2 = row[c2]; v3 = row[c3]
                # if any NaN, fallback
                if pd.isna(v0) or pd.isna(v1) or pd.isna(v2) or pd.isna(v3):
                    crop = None
                else:
                    # detect whether bbox is xywh or x1y1x2y2 by heuristic:
                    is_xywh = True
                    # if names are x1,x2 etc then treat as x1y1x2y2
                    if ("x1" in c0.lower() or "xmin" in c0.lower() or "xmax" in c0.lower()):
                        # this could be x1,y1,x2,y2 or xmin,ymin,xmax,ymax
                        is_xywh = False
                    # decide normalized vs absolute: if any value <=1 assume normalized (0..1)
                    normalized = False
                    try:
                        numeric_vals = list(map(float, [v0, v1, v2, v3]))
                        if all(abs(v) <= 1.01 for v in numeric_vals):
                            normalized = True
                    except Exception:
                        normalized = False
                    if is_xywh:
                        bbox = (float(v0), float(v1), float(v2), float(v3))
                        crop = crop_bbox_from_image(img, bbox, is_xywh=True, normalized=normalized)
                    else:
                        bbox = (float(v0), float(v1), float(v2), float(v3))
                        crop = crop_bbox_from_image(img, bbox, is_xywh=False, normalized=normalized)
            except Exception:
                crop = None

        # If no crop from bbox, fallback to center square crop (best-effort)
        if crop is None:
            w_img, h_img = img.size
            side = int(min(w_img, h_img) * 0.7)
            cx, cy = w_img // 2, h_img // 2
            half = side // 2
            x1 = max(0, cx - half); y1 = max(0, cy - half)
            x2 = min(w_img, cx + half); y2 = min(h_img, cy + half)
            crop = img.crop((x1, y1, x2, y2))

        # save crop to class folder
        cls_dir = os.path.join(out_dir, cls)
        ensure_dir(cls_dir)
        # filename pattern: originalname_idx.jpg to avoid collisions
        base = os.path.splitext(os.path.basename(img_name))[0]
        out_name = f"{base}_{idx}.jpg"
        out_path = os.path.join(cls_dir, out_name)
        try:
            crop.save(out_path, format="JPEG", quality=90)
            class_counts[cls] += 1
        except Exception:
            # skip if cannot save
            continue


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def main(args):
    fairface_dir = args.fairface_dir
    out_dir = args.out_dir
    max_per_class = args.max_per_class

    ensure_dir(out_dir)
    # read CSVs
    train_csv = os.path.join(fairface_dir, "train_labels.csv")
    val_csv = os.path.join(fairface_dir, "val_labels.csv")
    train_images = os.path.join(fairface_dir, "train")
    val_images = os.path.join(fairface_dir, "val")

    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise FileNotFoundError("Could not find train_labels.csv or val_labels.csv in " + fairface_dir)

    if not os.path.exists(train_images) or not os.path.exists(val_images):
        raise FileNotFoundError("Could not find train/ or val/ image folders in " + fairface_dir)

    print("Reading CSVs...")
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    # create output class dirs
    classes = ["Indian", "United States", "African", "Other"]
    for c in classes:
        ensure_dir(os.path.join(out_dir, c))

    class_counts = defaultdict(int)
    print("Processing train CSV...")
    process_csv(df_train, train_images, out_dir, class_counts, max_per_class=max_per_class, verbose=False)

    print("Processing val CSV...")
    process_csv(df_val, val_images, out_dir, class_counts, max_per_class=max_per_class, verbose=False)

    print("\nDone. Class counts:")
    for c in classes:
        print(f"  {c}: {class_counts[c]} images")

    # optionally: create a small test split inside data/nationality/test by sampling 5% from each class
    if args.create_test_split:
        test_dir = os.path.join(out_dir, "test")
        ensure_dir(test_dir)
        for c in classes:
            src_dir = os.path.join(out_dir, c)
            dest_dir = os.path.join(test_dir, c)
            ensure_dir(dest_dir)
            files = [f for f in os.listdir(src_dir) if f.lower().endswith(".jpg")]
            files.sort()
            n = max(1, int(len(files) * 0.05))
            for i, fn in enumerate(files[:n]):
                shutil.copy(os.path.join(src_dir, fn), os.path.join(dest_dir, fn))
        print("Created small test split under:", test_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fairface_dir", type=str, default="fairface",
                   help="Path to downloaded FairFace folder (contains train/, val/, train_labels.csv, val_labels.csv)")
    p.add_argument("--out_dir", type=str, default="data/nationality",
                   help="Where to write the nationality dataset (class subfolders will be created)")
    p.add_argument("--max_per_class", type=int, default=None,
                   help="Maximum images per class (useful for quick testing). Default: None (no limit)")
    p.add_argument("--create_test_split", action="store_true",
                   help="Create a small test split (5% sample per class) under <out_dir>/test")
    args = p.parse_args()
    main(args)
