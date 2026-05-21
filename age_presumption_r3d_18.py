import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models.video as models
import os



# =========================
# 設定
# =========================
IMAGE_DIR = "E:/mri_age/dataset/3D_T1_skstp/train"
CSV_PATH = "E:/mri_age/dataset/data_train.csv"
TARGET_SHAPE = (128, 128, 128)
BATCH_SIZE = 8 #16
EPOCHS = 40
lr = 1e-4
wd =  1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


file_map = {}

for fname in os.listdir(IMAGE_DIR):
    if fname.startswith("IXI"):
        id_str = fname[3:6]  # "002"
        file_map[id_str] = fname



# =========================
# 前処理関数
# =========================
def load_nifti(path):
    from nibabel.orientations import aff2axcodes
    #print(aff2axcodes(img.affine))
    img = nib.load(path)                 # Nifti1Image
    img = nib.as_closest_canonical(img)  # 向き補正（重要）
    data = img.get_fdata()               # numpy.ndarrayに変換
    return data.astype(np.float32)       # ✅ ここでastype


def zscore(img):
    brain = img[img > 0]

    if len(brain) == 0:
        return np.zeros_like(img)

    mean = brain.mean()
    std = brain.std()

    if std < 1e-6:
        return img - mean

    return (img - mean) / std


def resize_3d(img, target_shape):
    img = np.nan_to_num(img)  # ★追加

    factors = [t/s for t, s in zip(target_shape, img.shape)]
    return zoom(img, factors, order=1)




def preprocess_image(path):
    img = load_nifti(path)
    img = zscore(img)
    img = resize_3d(img, TARGET_SHAPE)
    return img


# =========================
# Dataset
# =========================
class MultiModalDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        self.df = self.df.dropna(subset=["HEIGHT", "WEIGHT", "AGE"])

        self.df = self.df[self.df["IXI_ID"].apply(self._has_image)].reset_index(drop=True)

        # ⭐これが必要（重要）
        self.num_cols = ["HEIGHT", "WEIGHT", "BMI"] #["AGE", "HEIGHT", "WEIGHT", "BMI"]

        self.df["BMI"] = self.df["WEIGHT"] / ((self.df["HEIGHT"]/100)**2 + 1e-6)

        self.means = self.df[self.num_cols].mean()
        self.stds = self.df[self.num_cols].std() + 1e-6

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ===== MRI =====
        id_str = f"{int(row['IXI_ID']):03d}"
        img_path = os.path.join(IMAGE_DIR, file_map[id_str])
        #img_path = os.path.join(IMAGE_DIR, f"{row['ID']}.nii.gz")

        img = preprocess_image(img_path)

        if np.isnan(img).any():
            print("NaN in image!", img_path)

        if np.isinf(img).any():
            print("Inf in image!", img_path)

        img = np.expand_dims(img, 0)  # (1, D, H, W)
        img = torch.tensor(img, dtype=torch.float32)

        # ===== Tabular =====
        num = row[self.num_cols].values.astype(np.float32)
        num = (num - self.means.values) / self.stds.values

        sex = [1, 0] if row["SEX_ID (1=m, 2=f)"] == 1 else [0, 1]

        tab = np.concatenate([num, sex])
        tab = torch.tensor(tab, dtype=torch.float32)

        # ===== Target =====
        age = torch.tensor([row["AGE"]], dtype=torch.float32)

        return img, tab, age


    def _has_image(self, ix):
        id_str = f"{int(ix):03d}"
        return id_str in file_map
    

# =========================
# モデル
# =========================
class MRIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.r3d_18(pretrained=False)
        #MONAI DenseNet121
        #SwinUNETR

        self.backbone.stem[0] = nn.Conv3d(
            in_channels=1,
            out_channels=64,
            kernel_size=(3,7,7),
            stride=(1,2,2),
            padding=(1,3,3),
            bias=False
        )


        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)


class TabularModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            #nn.BatchNorm1d(128),
            nn.Linear(128, 64)
        )


    def forward(self, x):
        return self.net(x)


class MultiModalModel(nn.Module):
    def __init__(self, tab_dim):
        super().__init__()
        self.mri = MRIModel()
        self.tab = TabularModel(tab_dim)

        self.head = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, img, tab):
        img_feat = self.mri(img)
        tab_feat = self.tab(tab)

        x = torch.cat([img_feat, tab_feat], dim=1)
        return self.head(x)


# =========================
# 学習ループ
# =========================
def train():
    df = pd.read_csv(CSV_PATH)

    # 患者単位シャッフル
    df = df.sample(frac=1).reset_index(drop=True)

    split = int(len(df) * 0.8)
    train_df = df[:split]
    val_df = df[split:]

    train_ds = MultiModalDataset(train_df)
    val_ds = MultiModalDataset(val_df)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    tab_dim = len(train_ds[0][1])

    model = MultiModalModel(tab_dim).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.L1Loss()
    # ⭐ここに追加
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )    

    best_val_loss = float("inf")


    start_epoch = 0

    if os.path.exists("checkpoints/last.pth"):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, "checkpoints/last.pth")


    for epoch in range(start_epoch, EPOCHS):
        model.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        train_loss = 0

        for img, tab, age in pbar:
            img, tab, age = img.to(DEVICE), tab.to(DEVICE), age.to(DEVICE)

            pred = model(img, tab)

            if torch.isnan(pred).any():
                print("❌ NaN in pred")
                break

            loss = criterion(pred, age)

            if torch.isnan(loss):
                print("❌ NaN in loss")
                break

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # ===== validation =====
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for img, tab, age in val_loader:
                img, tab, age = img.to(DEVICE), tab.to(DEVICE), age.to(DEVICE)

                pred = model(img, tab)
                loss = criterion(pred, age)

                val_loss += loss.item()

        # ⭐平均にする（重要）
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: train={train_loss:.3f}, val={avg_val_loss:.3f}")

        # ⭐scheduler
        scheduler.step(avg_val_loss)

        # ===== checkpoint保存（ここが正しい位置）=====
        os.makedirs("checkpoints", exist_ok=True)

        # 毎epoch保存
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": avg_val_loss
        }, f"checkpoints/epoch_{epoch}.pth")

        # ベスト保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": avg_val_loss
            }, "checkpoints/best.pth")
            print("✅ Saved BEST model")

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_epoch = checkpoint["epoch"] + 1

    print(f"Resumed from epoch {start_epoch}")

    return model, optimizer, start_epoch


# =========================
# テスト用Dataset
# =========================
class TestDataset_age(Dataset):
    def __init__(self, df, image_dir):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir

        # BMI
        self.df["BMI"] = self.df["WEIGHT"] / (self.df["HEIGHT"]/100)**2

        self.num_cols = ["AGE", "HEIGHT", "WEIGHT", "BMI"]
        self.means = self.df[self.num_cols].mean()
        self.stds = self.df[self.num_cols].std() + 1e-6

        # file map
        self.file_map = {}
        for fname in os.listdir(self.image_dir):
            if fname.startswith("IXI"):
                id_str = fname[3:6]
                self.file_map[id_str] = fname

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ===== MRI =====
        id_str = f"{int(row['IXI_ID']):03d}"
        img_path = os.path.join(self.image_dir, self.file_map[id_str])

        img = preprocess_image(img_path)
        img = np.expand_dims(img, 0)
        img = torch.tensor(img, dtype=torch.float32)

        # ===== Tabular =====
        num = row[self.num_cols].values.astype(np.float32)
        num = (num - self.means.values) / self.stds.values

        sex = [1, 0] if row["SEX_ID (1=m, 2=f)"] == 1 else [0, 1]

        tab = np.concatenate([num, sex])
        tab = torch.tensor(tab, dtype=torch.float32)

        # ===== Target =====
        age = torch.tensor([row["AGE"]], dtype=torch.float32)

        return img, tab, age, id_str


# =========================
# テスト関数
# =========================
def test_age():
    TEST_IMAGE_DIR = "E:/mri_age/dataset/3D_T1_skstp/test"
    TEST_CSV_PATH = "E:/mri_age/dataset/data_test.csv"

    df = pd.read_csv(TEST_CSV_PATH)

    test_ds = TestDataset_age(df, TEST_IMAGE_DIR)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    tab_dim = len(test_ds[0][1])

    model = MultiModalModel(tab_dim).to(DEVICE)
    checkpoint = torch.load("./checkpoints/age_h_w_bmi_16/best.pth", map_location=DEVICE)

    model.load_state_dict(
        checkpoint["model_state"],
        strict=False
    )

    #model.load_state_dict(checkpoint["model_state"])
    model.eval()

    preds = []
    gts = []

    print("\n===== Test Results =====")



    with torch.no_grad():
        for img, tab, age, id_str in tqdm(test_loader, desc="Testing"):
            img = img.to(DEVICE)
            tab = tab.to(DEVICE)

            pred = model(img, tab)

            pred_age = pred.item()
            true_age = age.item()

            preds.append(pred_age)
            gts.append(true_age)

            print(f"ID: {id_str[0]} | Pred: {pred_age:.2f} | GT: {true_age:.2f}")

    # ===== MAE =====
    preds = np.array(preds)
    gts = np.array(gts)

    mae = np.mean(np.abs(preds - gts))

    print("\n===== Summary =====")
    print(f"MAE: {mae:.3f}")



# =========================
# テスト用Dataset
# =========================
class TestDataset(Dataset):
    def __init__(self, df, image_dir):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir

        # ===== clean =====
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        self.df = self.df.dropna(subset=["HEIGHT", "WEIGHT", "AGE"])

        # ===== BMI =====
        self.df["BMI"] = self.df["WEIGHT"] / (
            (self.df["HEIGHT"] / 100) ** 2 + 1e-6
        )

        # ===== trainと一致させる（重要）=====
        self.num_cols = ["HEIGHT", "WEIGHT", "BMI"]

        self.means = self.df[self.num_cols].mean()
        self.stds = self.df[self.num_cols].std() + 1e-6

        # ===== file map =====
        self.file_map = {}

        for fname in os.listdir(self.image_dir):
            if fname.startswith("IXI"):
                id_str = fname[3:6]
                self.file_map[id_str] = fname

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ===== MRI =====
        id_str = f"{int(row['IXI_ID']):03d}"

        img_path = os.path.join(
            self.image_dir,
            self.file_map[id_str]
        )

        img = preprocess_image(img_path)

        img = np.expand_dims(img, 0)

        img = np.nan_to_num(img)

        img = torch.tensor(img, dtype=torch.float32)

        # ===== Tabular =====
        num = row[self.num_cols].values.astype(np.float32)

        num = np.nan_to_num(num)

        num = (num - self.means.values) / self.stds.values

        sex = [1, 0] if row["SEX_ID (1=m, 2=f)"] == 1 else [0, 1]

        tab = np.concatenate([num, sex])

        tab = np.nan_to_num(tab)

        tab = torch.tensor(tab, dtype=torch.float32)

        # ===== Target =====
        age = torch.tensor([row["AGE"]], dtype=torch.float32)

        return img, tab, age, id_str


# =========================
# テスト関数
# =========================
def test():

    TEST_IMAGE_DIR = "E:/mri_age/dataset/3D_T1_skstp/test"
    TEST_CSV_PATH = "E:/mri_age/dataset/data_test.csv"

    df = pd.read_csv(TEST_CSV_PATH)

    test_ds = TestDataset(df, TEST_IMAGE_DIR)

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False
    )

    tab_dim = len(test_ds[0][1])

    model = MultiModalModel(tab_dim).to(DEVICE)

    checkpoint = torch.load(
        "./checkpoints/best.pth",
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    preds = []
    gts = []

    print("\n===== Test Results =====")

    with torch.no_grad():

        for img, tab, age, id_str in tqdm(
            test_loader,
            desc="Testing"
        ):

            img = img.to(DEVICE)
            tab = tab.to(DEVICE)

            pred = model(img, tab)

            pred_age = pred.item()
            true_age = age.item()

            preds.append(pred_age)
            gts.append(true_age)

            print(
                f"ID: {id_str[0]} | "
                f"Pred: {pred_age:.2f} | "
                f"GT: {true_age:.2f}"
            )

    # ===== MAE =====
    preds = np.array(preds)
    gts = np.array(gts)

    mae = np.mean(np.abs(preds - gts))

    print("\n===== Summary =====")
    print(f"MAE: {mae:.3f}")







if __name__ == "__main__":
    #train()
    #test()
    test_age()







