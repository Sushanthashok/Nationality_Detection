# prepare_emotion.py (minimal)
import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm

CSV = "data/fer2013.csv"   # adjust

out_dir = "data/emotion_images"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(CSV)
for i, row in tqdm(df.iterrows(), total=len(df)):
    pixels = np.fromstring(row['pixels'], dtype=int, sep=' ')
    img = pixels.reshape(48, 48).astype('uint8')
    # convert to RGB and resize to 224
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    label = int(row['emotion'])
    # make class dir
    cls_dir = os.path.join(out_dir, str(label))
    os.makedirs(cls_dir, exist_ok=True)
    filepath = os.path.join(cls_dir, f"{i}.jpg")
    cv2.imwrite(filepath, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
