
import torch
import torch
from torch import nn
import csv
import os


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

def count_params(model):
    # Count all parameters
    total_params = sum(p.numel() for p in model.parameters())
    # Count only trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

# -----------------------
# Loss function (CLIP-style)
# -----------------------
def clip_loss(img_feats, txt_feats, temperature):
    # NOTE: already normalized during forward pass
    # img_feats = nn.functional.normalize(img_feats, dim=-1)
    # txt_feats = nn.functional.normalize(txt_feats, dim=-1)

    logits_per_image = (img_feats @ txt_feats.t()) / temperature
    logits_per_text = logits_per_image.t()
    targets = torch.arange(img_feats.size(0), device=img_feats.device) # define target index for each row in logits_per_image or logits_per_text matrix.
    loss_i2t = nn.functional.cross_entropy(logits_per_image, targets)
    loss_t2i = nn.functional.cross_entropy(logits_per_text, targets)
    return (loss_i2t + loss_t2i) / 2

# -----------------------
# Utilities for prompts
# -----------------------
def build_prompts(pathologies):
    """
    pathologies: a list of pathology names
    """
    pos = [f"{p}" for p in pathologies]
    neg = [f"No {p}" for p in pathologies]
    
    # Interleave as [pos0, neg0, pos1, neg1, ...] for easy indexing
    prompts = []

    # pair up the positive and negative prompt for the same disease
    for p_pos, p_neg in zip(pos, neg):
        prompts.extend([p_pos, p_neg])
    return prompts  # length = 2 * P

@torch.no_grad()
def encode_prompts(model, prompts, device):
    """
    Returns normalized text features of shape [2P, D].
    Assumes model.encode_text exists. If not, replace with your model's text-encode call.
    """
    txt_feats = model.encode_text(prompts).to(device)
    txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
    return txt_feats


@torch.no_grad()
def predict_pathologies(model, val_loader, pathologies, txt_feats_norm, temperature, out_csv, device, id_key="image_id"):
    """
    For each image in val_loader, compute probability for each pathology:
        p = softmax([sim_pos, sim_neg]/T)[0]
    Save predictions and probabilities to CSV.
    """
    print(f"Zero shot validation...")

    model.eval()
    results = []

    # Precompute helpful indices
    # For pathology k: pos_idx = 2*k, neg_idx = 2*k+1
    # used to index the similarity matrix for each image down below.
    idx_pairs = [(2*k, 2*k+1) for k in range(len(pathologies))]

    row_idx = 0
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch['image'].to(device)

            # get the image_ids, the volume names, the main identifier
            assert id_key in batch
            ids = [str(x) for x in batch[id_key]]

            # Encode and normalized images
            image_features = model.encode_image(imgs) 
            img_feats_norm = image_features / image_features.norm(dim=-1, keepdim=True)

            # Similarities to all prompts: [B, 2P]
            sims = img_feats_norm @ txt_feats_norm.t()

            # For each sample, compute per-pathology prob & pred and save as a dictionary
            for b, sample_id in enumerate(ids):
                row = {"VolumeName": sample_id}
                for k, (pos_i, neg_i) in enumerate(idx_pairs):
                    logits = torch.stack([sims[b, pos_i], sims[b, neg_i]]) / temperature
                    probs = torch.softmax(logits, dim=0)
                    p_pos = probs[0].item() # index 0 is positive prompt
                    pred = int(p_pos >= 0.5)
                    pname = pathologies[k]

                    # store the probability and prediciton mainly for computing the metrics score down the road
                    row[f"{'_'.join(pname.split())}_prob"] = round(p_pos, 6)
                    row[f"{'_'.join(pname.split())}_pred"] = pred
                results.append(row)
            row_idx += len(ids)

    # overwrite the CSV file.
    fieldnames = ["VolumeName"] + [f"{'_'.join(p.split())}_prob" for p in pathologies] + [f"{'_'.join(p.split())}_pred" for p in pathologies]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"[Saved] {len(results)} rows to {out_csv}")
