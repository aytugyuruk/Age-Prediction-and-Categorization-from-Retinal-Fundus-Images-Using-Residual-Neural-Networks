"""
Training script for retinal fundus age prediction with ResNet backbones.

Reproduces the ResNet-18/34/50/101/152 experiments from:
  Yuruk, M.A. and Memis, A. "Age Prediction and Categorization from Retinal
  Fundus Images Using Residual Neural Networks." IISEC 2026, pp. 628-633.
  DOI: 10.1109/IISEC69317.2026.11418414

Select the model with --model, and the preprocessing variant with
--train-csv/--val-csv (point them at the filtered or non-filtered split files).
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

MODEL_BUILDERS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, "small"),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, "small"),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, "large"),
    "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1, "large"),
    "resnet152": (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1, "large"),
}


def get_lds_weights(df, age_col="patient_age", sigma=2):
    """Label Distribution Smoothing weights (Yang et al., ICML 2021)."""
    value_counts = df[age_col].value_counts().sort_index()
    min_age = int(df[age_col].min())
    max_age = int(df[age_col].max())

    counts = np.zeros(max_age - min_age + 1)
    for age, count in value_counts.items():
        if age - min_age < len(counts):
            counts[int(age - min_age)] = count

    smoothed_counts = gaussian_filter1d(counts, sigma=sigma)
    weights = 1.0 / (smoothed_counts + 1e-5)
    weights = weights / weights.mean()

    return {age: weights[int(age - min_age)] for age in range(min_age, max_age + 1)}


class FundusAgeDataset(Dataset):
    def __init__(self, df, transform, mean_age, std_age, weights_dict=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.mean_age = mean_age
        self.std_age = std_age
        self.weights_dict = weights_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["img_path"]).convert("RGB")
        image = self.transform(image)

        age = row["patient_age"]
        norm_age = (age - self.mean_age) / self.std_age

        if self.weights_dict is None:
            weight = 1.0
        else:
            weight = self.weights_dict.get(int(age), 1.0)

        return (
            image,
            torch.tensor(norm_age, dtype=torch.float32),
            torch.tensor(age, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        )


def build_model(model_name: str) -> nn.Module:
    builder, weights, head_size = MODEL_BUILDERS[model_name]
    model = builder(weights=weights)
    for param in model.parameters():
        param.requires_grad = True

    if head_size == "small":
        model.fc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
    else:
        model.fc = nn.Sequential(
            nn.Linear(2048, 512), nn.BatchNorm1d(512, momentum=0.1),
            nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.BatchNorm1d(128, momentum=0.1),
            nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
    return model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(MODEL_BUILDERS))
    p.add_argument("--train-csv", default="csvFiles/train/trainFilteredImages.csv")
    p.add_argument("--val-csv", default="csvFiles/validation/validationFilteredImages.csv")
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42,
                    help="Seed for this run. Note: the checkpoints released with the paper were "
                         "trained without seeding, so exact bitwise reproduction of those specific "
                         "weights is not guaranteed -- but the reported metrics are reproducible "
                         "from the released checkpoint via evaluate.py.")
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    mean_age = train_df["patient_age"].mean()
    std_age = train_df["patient_age"].std()

    print(f"\nData: Train={len(train_df)}, Val={len(val_df)}")
    print(f"Age: Mean={mean_age:.2f}, Std={std_age:.2f}, "
          f"Range=[{train_df['patient_age'].min()}, {train_df['patient_age'].max()}]")

    print("\nComputing LDS (Label Distribution Smoothing) weights...")
    lds_weights_dict = get_lds_weights(train_df, age_col="patient_age", sigma=2)
    print("LDS weights ready.")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.1, 0.1, 0.05, 0.02),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = FundusAgeDataset(train_df, train_transform, mean_age, std_age, weights_dict=lds_weights_dict)
    val_ds = FundusAgeDataset(val_df, eval_transform, mean_age, std_age, weights_dict=None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=False)

    print(f"\nBuilding model: {args.model}")
    model = build_model(args.model).to(DEVICE)
    print(f"Model loaded on device: {DEVICE}")

    criterion = nn.SmoothL1Loss(beta=1.0, reduction="none")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7,
    )

    print(f"\n{'='*60}\nStarting training ({args.epochs} epochs total)\n{'='*60}")

    best_val_loss = float("inf")
    best_val_mae = float("inf")
    best_epoch = 0

    history = {
        "train_loss": [], "val_loss": [],
        "train_mae": [], "val_mae": [],
    }

    ckpt_path = os.path.join(args.output_dir, f"best_{args.model}.pth")

    for epoch in range(args.epochs):
        print(f"\n{'='*60}\nEpoch {epoch+1}/{args.epochs}\n{'='*60}")

        # --- training ---
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []

        for images, norm_ages, orig_ages, weights in train_loader:
            images = images.to(DEVICE)
            orig_ages = orig_ages.to(DEVICE)
            weights = weights.to(DEVICE)
            norm_ages = norm_ages.to(DEVICE)

            outputs = model(images).view(-1)
            loss_per_sample = criterion(outputs, norm_ages)
            loss = (loss_per_sample * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.detach().cpu() * std_age + mean_age
            train_preds.extend(preds.numpy())
            train_labels.extend(orig_ages.cpu().numpy())

        train_loss = train_loss / len(train_loader.dataset)
        train_mae = mean_absolute_error(train_labels, train_preds)

        # --- validation ---
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for images, norm_ages, orig_ages, _ in val_loader:
                images = images.to(DEVICE)
                orig_ages = orig_ages.to(DEVICE)

                outputs = model(images).view(-1)
                loss = criterion(outputs, norm_ages.to(DEVICE)).mean()

                val_loss += loss.item() * images.size(0)
                preds = outputs.detach().cpu() * std_age + mean_age
                val_preds.extend(preds.numpy())
                val_labels.extend(orig_ages.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_mae = mean_absolute_error(val_labels, val_preds)

        history["train_loss"].append(train_loss)
        history["train_mae"].append(train_mae)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        print(f"Train - Loss: {train_loss:.4f}, MAE: {train_mae:.2f}y (LDS weighted)")
        print(f"Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.2f}y")

        scheduler.step(val_loss)
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae

        if val_loss < best_val_loss:
            print(f"New best val loss: {val_loss:.4f} (val MAE at this epoch: {val_mae:.2f}y). Saving checkpoint.")
            best_val_loss = val_loss
            best_epoch = epoch + 1

            torch.save({
                "epoch": epoch + 1,
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_mae": val_mae,
                "mean_age": mean_age,
                "std_age": std_age,
                "history": history,
                "config": {
                    "BATCH_SIZE": args.batch_size,
                    "EPOCHS": args.epochs,
                    "LR": args.lr,
                    "WEIGHT_DECAY": args.weight_decay,
                    "SEED": args.seed,
                },
            }, ckpt_path)
        else:
            print(f"Val loss ({val_loss:.4f}) did not improve on best ({best_val_loss:.4f}). Not saved.")

    print(f"\n{'='*60}\nTraining finished ({args.epochs} epochs).\n"
          f"Best val loss: {best_val_loss:.4f} (epoch {best_epoch})\n{'='*60}")
    print(f"Best checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
