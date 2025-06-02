import os
import tarfile
import torch
import numpy as np
from io import BytesIO
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import fsspec
from sklearn.metrics import average_precision_score

def main():
    DATA_DIR = os.environ.get("DATA_DIR", "gs://thesishugosokolowskikatzer/")
    CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "gs://thesishugosokolowskikatzer/checkpoints/")
    LOG_PATH = os.environ.get("LOG_PATH", os.path.join(CHECKPOINT_DIR, "training_log.csv"))

    BATCH_SIZE = 28
    EPOCHS = 15
    IMG_SIZE = 256
    NUM_WORKERS = 0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fs = fsspec.filesystem("gcs")

    class CrossEntropyDiceLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.ce = nn.CrossEntropyLoss()

        def forward(self, pred, target):
            ce_loss = self.ce(pred, target)
            probs = torch.softmax(pred, dim=1)[:, 1]
            target_float = (target == 1).float()
            inter = (probs * target_float).sum()
            union = probs.sum() + target_float.sum()
            dice_loss = 1 - (2 * inter + 1e-6) / (union + 1e-6)
            return ce_loss + dice_loss

    class TarXBDDataset(Dataset):
        def __init__(self, tar_path, augment=False):
            self.tar_path = tar_path
            self.augment = augment
            self.valid_samples = []

            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Resize(IMG_SIZE, IMG_SIZE),
                A.Normalize(mean=[0.0]*6, std=[1.0]*6),
                ToTensorV2()
            ])

            try:
                self.tar = tarfile.open(tar_path, 'r')
                members = {m.name: m for m in self.tar.getmembers()}
                for name in sorted(members):
                    if name.endswith('.img.npy'):
                        mask_name = name.replace('.img.npy', '.mask.npy')
                        if mask_name in members:
                            try:
                                img_data = self.tar.extractfile(members[name]).read()
                                mask_data = self.tar.extractfile(members[mask_name]).read()
                                img_array = np.load(BytesIO(img_data), allow_pickle=False)
                                mask_array = np.load(BytesIO(mask_data), allow_pickle=False)
                                mask_array = (mask_array > 0).astype(np.uint8)
                                if img_array.ndim != 3 or img_array.shape[0] != 6:
                                    continue
                                if mask_array.ndim != 2:
                                    continue
                                self.valid_samples.append((members[name], members[mask_name]))
                            except:
                                continue
            except Exception as e:
                raise RuntimeError(f"âŒ Could not open TAR file {tar_path}: {e}")

        def __len__(self):
            return len(self.valid_samples)

        def __getitem__(self, idx):
            for attempt in range(10):
                img_member, mask_member = self.valid_samples[idx]
                try:
                    img_data = self.tar.extractfile(img_member).read()
                    mask_data = self.tar.extractfile(mask_member).read()
                    img_array = np.load(BytesIO(img_data), allow_pickle=False)
                    mask_array = np.load(BytesIO(mask_data), allow_pickle=False)
                    mask_array = (mask_array > 0).astype(np.uint8)

                    if self.augment:
                        transformed = self.transform(image=img_array.transpose(1, 2, 0), mask=mask_array)
                        image = transformed['image']
                        mask = transformed['mask'].long()
                    else:
                        image = torch.from_numpy(img_array).float()
                        mask = torch.from_numpy(mask_array).long()

                    return image, mask
                except:
                    idx = (idx + 1) % len(self.valid_samples)
            raise RuntimeError(f"âŒ Too many failed attempts to load data starting from index {idx}")

    def load_tar_datasets(base_dir, split="train", augment=False):
        tar_paths = fs.glob(f"{base_dir.rstrip('/')}/{split}-*.tar")
        print(f"Found {len(tar_paths)} {split} .tar files in GCS")
        local_paths = []
        for gcs_path in tqdm(tar_paths, desc=f"Downloading {split} .tar files"):
            local_path = f"/tmp/{os.path.basename(gcs_path)}"
            fs.get(gcs_path, local_path)
            local_paths.append(local_path)
        datasets = [TarXBDDataset(path, augment=augment) for path in local_paths]
        return ConcatDataset(datasets)

    train_dataset = load_tar_datasets(DATA_DIR, "train", augment=True)
    val_dataset = load_tar_datasets(DATA_DIR, "val", augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        def forward(self, x):
            return self.block(x)

    class UNetPP(nn.Module):
        def __init__(self, in_channels=6, out_channels=2):
            super().__init__()
            filters = [64, 128, 256, 512]
            self.conv00 = ConvBlock(in_channels, filters[0])
            self.pool0 = nn.MaxPool2d(2)
            self.conv10 = ConvBlock(filters[0], filters[1])
            self.pool1 = nn.MaxPool2d(2)
            self.conv20 = ConvBlock(filters[1], filters[2])
            self.pool2 = nn.MaxPool2d(2)
            self.conv30 = ConvBlock(filters[2], filters[3])
            self.up01 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
            self.conv01 = ConvBlock(filters[0]*2, filters[0])
            self.up11 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
            self.conv11 = ConvBlock(filters[1]*2, filters[1])
            self.up02 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
            self.conv02 = ConvBlock(filters[0]*3, filters[0])
            self.up21 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
            self.conv21 = ConvBlock(filters[2]*2, filters[2])
            self.up12 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
            self.conv12 = ConvBlock(filters[1]*3, filters[1])
            self.up03 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
            self.conv03 = ConvBlock(filters[0]*4, filters[0])
            self.final = nn.Conv2d(filters[0], out_channels, 1)
        def forward(self, x):
            x00 = self.conv00(x)
            x10 = self.conv10(self.pool0(x00))
            x20 = self.conv20(self.pool1(x10))
            x30 = self.conv30(self.pool2(x20))
            x01 = self.conv01(torch.cat([x00, self.up01(x10)], dim=1))
            x11 = self.conv11(torch.cat([x10, self.up11(x20)], dim=1))
            x02 = self.conv02(torch.cat([x00, x01, self.up02(x11)], dim=1))
            x21 = self.conv21(torch.cat([x20, self.up21(x30)], dim=1))
            x12 = self.conv12(torch.cat([x10, x11, self.up12(x21)], dim=1))
            x03 = self.conv03(torch.cat([x00, x01, x02, self.up03(x12)], dim=1))
            return self.final(x03)

    def compute_iou(pred, target, threshold=0.5):
        probs = torch.softmax(pred, dim=1)[:, 1]
        pred_mask = (probs > threshold).float()
        intersection = (pred_mask * target.float()).sum()
        union = pred_mask.sum() + target.sum() - intersection
        return ((intersection + 1e-6) / (union + 1e-6)).item()

    def soft_dice_score(pred, target, smooth=1e-6):
        probs = torch.softmax(pred, dim=1)[:, 1]
        inter = (probs * target.float()).sum()
        union = probs.sum() + target.sum()
        return ((2 * inter + smooth) / (union + smooth)).item()

    def pixel_correlation(pred, target):
        probs = torch.softmax(pred, dim=1)[:, 1].flatten()
        target = target.float().flatten()
        if target.std() == 0:
            return 0.0
        return torch.corrcoef(torch.stack([probs, target]))[0, 1].item()

    def pr_auc(pred, target):
        probs = torch.softmax(pred, dim=1)[:, 1].cpu().flatten().numpy()
        target = target.cpu().flatten().numpy()
        if target.sum() == 0:
            return 0.0
        return average_precision_score(target, probs)

    scaler = GradScaler()
    model = UNetPP().to(DEVICE)
    loss_fn = CrossEntropyDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_soft_dice': [], 'val_corr': [], 'val_auc': []}
    print("ðŸš€ Starting training...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = loss_fn(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                print(f"ðŸ§ª Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        model.eval()
        val_loss = 0
        iou_total = 0
        dice_total = 0
        corr_total = 0
        auc_total = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                with autocast():
                    outputs = model(images)
                    loss = loss_fn(outputs, masks)
                val_loss += loss.item()
                iou_total += compute_iou(outputs, masks)
                dice_total += soft_dice_score(outputs, masks)
                corr_total += pixel_correlation(outputs, masks)
                auc_total += pr_auc(outputs, masks)

        n_batches = len(val_loader)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / n_batches
        avg_iou = iou_total / n_batches
        avg_dice = dice_total / n_batches
        avg_corr = corr_total / n_batches
        avg_auc = auc_total / n_batches

        scheduler.step(avg_val_loss)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_iou'].append(avg_iou)
        history['val_soft_dice'].append(avg_dice)
        history['val_corr'].append(avg_corr)
        history['val_auc'].append(avg_auc)

        print(f"\nðŸ“ˆ Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - IoU: {avg_iou:.4f}")
        print(f"ðŸ“Š Soft Dice: {avg_dice:.4f}, Corr: {avg_corr:.4f}, PR-AUC: {avg_auc:.4f}")

        local_ckpt = f"/tmp/unetpp_state_epoch{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_iou': avg_iou
        }, local_ckpt)
        remote_ckpt = f"{CHECKPOINT_DIR.rstrip('/')}/unetpp_state_epoch{epoch+1}.pth"
        fs.put(local_ckpt, remote_ckpt)

    log_df = pd.DataFrame(history)
    local_log = "/tmp/training_log.csv"
    log_df.to_csv(local_log, index=False)
    fs.put(local_log, LOG_PATH)

    best_epoch = int(np.argmax(history['val_iou']))
    print(f"\nðŸ Best Epoch: {best_epoch + 1}")
    print(f"ðŸ”¹ Train Loss: {history['train_loss'][best_epoch]:.4f}")
    print(f"ðŸ”¹ Val Loss: {history['val_loss'][best_epoch]:.4f}")
    print(f"ðŸ”¹ Val IoU: {history['val_iou'][best_epoch]:.4f}")
    print("âœ… Training complete. Final model state saved.")

if __name__ == "__main__":
    main()