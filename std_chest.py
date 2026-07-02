import os
import glob
import numpy as np
import cv2
import torch
import torchxrayvision as xrv
from tqdm import tqdm

# ==========================
# 設定
# ==========================
IMAGE_DIR = r"E:\ChestXray\Normal"   # 健常者画像フォルダ
SAVE_PATH = "reference_vector.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# モデル
# ==========================
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model = model.to(DEVICE)
model.eval()

# ==========================
# 画像一覧
# ==========================
image_paths = []

for ext in ["*.png", "*.jpg", "*.jpeg"]:
    image_paths += glob.glob(os.path.join(IMAGE_DIR, ext))

print(f"Images : {len(image_paths)}")

embeddings = []

# ==========================
# Feature Extraction
# ==========================
with torch.no_grad():

    for path in tqdm(image_paths):

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # float32
        img = img.astype(np.float32)

        # TorchXRayVision preprocessing
        img = xrv.datasets.normalize(img, 255)

        # 224×224
        img = cv2.resize(img, (224,224))

        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        img = img.to(DEVICE)

        ###############################
        # Feature Map
        ###############################
        feat = model.features(img)

        ###############################
        # Global Average Pooling
        ###############################
        feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1,1))

        feat = feat.view(-1)

        embeddings.append(
            feat.cpu().numpy()
        )

embeddings = np.stack(embeddings)

print("Embedding shape :", embeddings.shape)

# ==========================
# Reference Vector
# ==========================
reference = embeddings.mean(axis=0)

print(reference.shape)

# 保存
np.save(SAVE_PATH, reference)

print("Saved :", SAVE_PATH)