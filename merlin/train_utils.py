
import torch
import torch
from torch import nn
import csv
import os

# -----------------------
# Loss function (CLIP-style)
# -----------------------
def clip_loss(img_feats, txt_feats, temperature):
    img_feats = nn.functional.normalize(img_feats, dim=-1)
    txt_feats = nn.functional.normalize(txt_feats, dim=-1)
    logits_per_image = (img_feats @ txt_feats.t()) / temperature
    logits_per_text = logits_per_image.t()
    targets = torch.arange(img_feats.size(0), device=img_feats.device)
    loss_i2t = nn.functional.cross_entropy(logits_per_image, targets)
    loss_t2i = nn.functional.cross_entropy(logits_per_text, targets)
    return (loss_i2t + loss_t2i) / 2

# -----------------------
# Utilities for prompts
# -----------------------
def build_prompts(pathologies):
    pos = [f"There is a {p}" for p in pathologies]
    neg = [f"There is no {p}" for p in pathologies]
    # Interleave as [pos0, neg0, pos1, neg1, ...] for easy indexing
    prompts = []
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
    return txt_feats


@torch.no_grad()
def predict_pathologies(model, val_loader, pathologies, txt_feats_norm, temperature, out_csv, device, id_key="image_id"):
    """
    For each image in val_loader, compute probability for each pathology:
        p = softmax([sim_pos, sim_neg]/T)[0]
    Save predictions and probabilities to CSV.
    """
    model.eval()
    results = []

    # Precompute helpful indices
    # For pathology k: pos_idx = 2*k, neg_idx = 2*k+1
    idx_pairs = [(2*k, 2*k+1) for k in range(len(pathologies))]

    row_idx = 0
    for batch in val_loader:
        imgs = batch['image'].to(device, non_blocking=True)

        assert id_key in batch
        # Grab identifier if present; otherwise create a running index
        ids = batch[id_key]
        # ensure list of strings
        if torch.is_tensor(ids): ids = ids.cpu().tolist()
        ids = [str(x) for x in ids]

        # Encode images
        img_feats_norm = model.encode_image(imgs) 

        # Similarities to all prompts: [B, 2P]
        sims = img_feats_norm @ txt_feats_norm.t()

        # For each sample, compute per-pathology prob & pred
        for b, sample_id in enumerate(ids):
            row = {"id": sample_id}
            for k, (pos_i, neg_i) in enumerate(idx_pairs):
                logits = torch.stack([sims[b, pos_i], sims[b, neg_i]]) / temperature
                probs = torch.softmax(logits, dim=0)
                p_pos = probs[0].item() # index 0 is positive prompt
                pred = int(p_pos >= 0.5)
                pname = pathologies[k]
                row[f"{pname}_prob"] = round(p_pos, 6)
                row[f"{pname}_pred"] = pred
            results.append(row)
        row_idx += len(ids)

    # Write CSV
    fieldnames = ["id"] + [f"{p}_prob" for p in pathologies] + [f"{p}_pred" for p in pathologies]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"[Saved] {len(results)} rows to {out_csv}")
