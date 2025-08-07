import warnings
import torch
from torch.utils.data import DataLoader 
from merlin import Merlin
from merlin.data.ctRate_dataloader import CTReportDataset

import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import csv

from merlin.train_utils import build_prompts, clip_loss, encode_prompts, predict_pathologies

# -----------------------
# Parse Arguments
# -----------------------
parser = argparse.ArgumentParser(description="Train Merlin on CT-RATE")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
parser.add_argument("--val_every", type=int, default=1, help="Validate every N epochs")
parser.add_argument("--temperature", type=float, default=0.07, help="Contrastive loss temperature")
args = parser.parse_args()


warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

# load dataset
ctrate_train_dataset = CTReportDataset(
    data_folder='/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_noflip_processed_train_images',
    report_csv='/cluster/projects/mcintoshgroup/publicData/CT-RATE/dataset/radiology_text_reports/train_reports.csv'
)
ctrate_val_dataset = CTReportDataset(
    data_folder='/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_noflip_processed_val_images',
    report_csv='/cluster/projects/mcintoshgroup/publicData/CT-RATE/dataset/radiology_text_reports/train_reports.csv' # TODO: need to replace with valid_reports.csv 
)

# load dataloader
train_loader = DataLoader(ctrate_train_dataset, batch_size=2, shuffle=True, num_workers=1)
val_loader = DataLoader(ctrate_val_dataset, batch_size=2, shuffle=False, num_workers=1)


# -----------------------
# Model & optimizer
# -----------------------
model = Merlin().to(device)
args.temperature = model.model.logit_scale
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Medical material	Arterial wall calcification	Cardiomegaly	Pericardial effusion	Coronary artery wall calcification	Hiatal hernia	Lymphadenopathy	Emphysema	Atelectasis	Lung nodule	Lung opacity	Pulmonary fibrotic sequela	Pleural effusion	Mosaic attenuation pattern	Peribronchial thickening	Consolidation	Bronchiectasis	Interlobular septal thickening

# list of labels
PATHOLOGIES = [
    'Medical material',
    'Arterial wall calcification',
    'Cardiomegaly',
    'Pericardial effusion',
    'Coronary artery wall calcification',
    'Hiatal hernia',
    'Lymphadenopathy',
    'Emphysema',
    'Atelectasis',
    'Lung nodule',
    'Lung opacity',
    'Pulmonary fibrotic sequela',
    'Pleural effusion',
    'Mosaic attenuation pattern',
    'Consolidation',
    'Bronchiectasis',
    'Interlobular septal thickening'
]

prompts = build_prompts(PATHOLOGIES)
txt_feats_norm = encode_prompts(model, prompts, device)

# -----------------------
# Training loop
# -----------------------
for epoch in range(1, args.epochs + 1):
    # ---- Train ----
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        img, txt = batch['image'].to(device, non_blocking=True), batch['text']
        optimizer.zero_grad()
        img_feats, txt_feats = model(img, txt)  # [B, 512], [B, 512]
        loss = clip_loss(img_feats, txt_feats, args.temperature)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)

    # ---- Validate ----
    if epoch % args.val_every == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                img, txt = batch['image'].to(device, non_blocking=True), batch['text']
                img_feats, txt_feats = model(img, txt)
                loss = clip_loss(img_feats, txt_feats, args.temperature)
                val_loss += loss.item()

        # TODO: store the labels according to some order into the csv file
        # Later we use the it to compare the ground truth
        val_loss /= len(val_loader)

        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # -----------------------
        # Run prompt-based multi-label predictions on the full val set
        # -----------------------
        predict_pathologies(
            model=model,
            val_loader=val_loader,
            pathologies=PATHOLOGIES,
            txt_feats_norm=txt_feats_norm,
            temperature=args.temperature,
            threshold=args.pred_threshold,
            out_csv=args.out_csv,
            device=device,
            id_key=args.id_key
        )

