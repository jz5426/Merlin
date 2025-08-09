import warnings
import torch
from torch.utils.data import DataLoader 
from merlin import Merlin
from merlin.data.ctRate_dataloader import CTReportDataset
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import os

from merlin.train_utils import build_prompts, clip_loss, count_params, encode_prompts, predict_pathologies, PATHOLOGIES

# -----------------------
# Parse Arguments
# -----------------------
parser = argparse.ArgumentParser(description="Train Merlin on CT-RATE")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
parser.add_argument("--val_every", type=int, default=1, help="Validate every N epochs")
parser.add_argument("--temperature", type=float, default=0.07, help="Contrastive loss temperature")
parser.add_argument("--num_workers", type=int, default=10, help="number of workers")
parser.add_argument("--out_csv", type=str, default='./ctrate_zeroshot/results.csv', help="zero-shot result storage path")
parser.add_argument("--ckpt_path", type=str, default='/cluster/projects/mcintoshgroup/publicData/merlin_checkpoint/ctrate_finetuned/ctrate_ckpt.pth', help="zero-shot result storage path")
args = parser.parse_args()

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

# load dataset
# NOTE: use the preprocessed files of /cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_noflip, 
# which include .h5 files; from the preprocess_data.py
ctrate_train_dataset = CTReportDataset(
    # data_folder='/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_noflip_processed_train_images',
    data_folder='/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_nii_fp16_noflip_merlin_preprocessed_train/',
    report_csv='/cluster/projects/mcintoshgroup/publicData/CT-RATE/dataset/radiology_text_reports/train_reports.csv'
)
ctrate_val_dataset = CTReportDataset(
    # data_folder='/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_noflip_processed_val_images',
    data_folder='/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_nii_fp16_noflip_merlin_preprocessed_val/',
    report_csv='/cluster/projects/mcintoshgroup/publicData/CT-RATE/dataset/radiology_text_reports/train_reports.csv' # TODO: need to replace with valid_reports.csv 
)

# load dataloader
train_loader = DataLoader(ctrate_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_loader = DataLoader(ctrate_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


# -----------------------
# Model & optimizer
# -----------------------
model = Merlin().to(device)
count_params(model)
args.temperature = model.model.logit_scale
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# print arguments
for k, v in vars(args).items():
    print(f"{k}: {v}")

prompts = build_prompts(PATHOLOGIES)

# -----------------------
# Training loop
# -----------------------
best_val_loss = float("inf")
for epoch in range(1, args.epochs + 1):

    # ---- Train ----
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
        img, txt = batch['image'].to(device, non_blocking=True), batch['text']
        optimizer.zero_grad()
        img_feats_norm, txt_feats_norm = model(img, txt)  # [B, 512], [B, 512]
        loss = clip_loss(img_feats_norm, txt_feats_norm, args.temperature)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)

    # ---- Validate ----
    if epoch % args.val_every == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                img, txt = batch['image'].to(device, non_blocking=True), batch['text']
                img_feats_norm, txt_feats_norm = model(img, txt)
                loss = clip_loss(img_feats_norm, txt_feats_norm, args.temperature)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # save the best checkpoint during the training trajectory
        os.makedirs(os.path.dirname(args.ckpt_path), exist_ok=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.ckpt_path)
            print(f"[Checkpoint] Best model updated (val_loss={val_loss:.4f}) → {args.ckpt_path}")
            
            # -----------------------
            # Run prompt-based multi-label predictions on the full val set
            # -----------------------
            txt_feats_norm = encode_prompts(model.model, prompts, device)
            predict_pathologies(
                model=model.model,
                val_loader=val_loader,
                pathologies=PATHOLOGIES,
                txt_feats_norm=txt_feats_norm,
                temperature=args.temperature,
                out_csv=args.out_csv,
                device=device,
            )
