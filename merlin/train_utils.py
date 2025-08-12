
import torch
from torch import nn
import csv
import os
from PIL import Image
import numpy as np

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
    # TODO: should i keep the temperature fixed to 0.07 instead of learnable?
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


def save_middle_slices_normalized(ct_tensor, save_dir='/cluster/home/t135419uhn/Merlin/visualize_transformed_ct_slices/', convention="radiological", prefix="ct"):
    """
    Save middle slices from a normalized CT tensor in RAS orientation.
    Args:
        ct_tensor: torch.Tensor, shape [1, X, Y, Z] (RAS orientation)
        save_dir: output folder
        prefix: filename prefix
        convention: "radiological" or "neurological"
    """
    os.makedirs(save_dir, exist_ok=True)

    vol = ct_tensor.squeeze(0)  # [X,Y,Z]
    assert vol.ndim == 3, "Expected [X,Y,Z] after squeezing channel."

    X, Y, Z = vol.shape
    mid_x, mid_y, mid_z = X//2, Y//2, Z//2

    axial    = vol[:, :, mid_z].cpu().numpy()   # (X,Y)
    coronal  = vol[:, mid_y, :].cpu().numpy()   # (X,Z)
    sagittal = vol[mid_x, :, :].cpu().numpy()   # (Y,Z)

    # Arrange for display
    axial_img    = axial.T # [Y, X]
    coronal_img  = coronal.T # [Z, X]
    sagittal_img = sagittal.T # [Z, Y]

    coronal_img = np.flipud(coronal_img)
    sagittal_img = np.flipud(sagittal_img)

    # if convention.lower().startswith("radio"):
    #     axial_img    = np.fliplr(axial_img)
    #     coronal_img  = np.fliplr(coronal_img)

    # Robust percentile normalization to [0,255]
    def to_uint8(img, p_lo=1.0, p_hi=99.0, z_clip=2.5, eps=1e-6):
        """Robust slice scaling -> uint8 with multiple fallbacks."""
        a = np.asarray(img, dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

        nz = a[a != 0]  # ignore background
        use = nz if nz.size > 0 else a

        # 1) percentile on non-zero
        lo, hi = np.percentile(use, p_lo), np.percentile(use, p_hi)
        if hi - lo < eps:
            # 2) z-score window on non-zero
            m, s = use.mean(), use.std()
            lo, hi = m - z_clip * s, m + z_clip * s

        if hi - lo < eps:
            # 3) min-max on non-zero
            lo, hi = use.min(), use.max()

        # final guard
        if hi - lo < eps:
            # give it a tiny range so we don't divide by 0
            hi = lo + 1.0

        # scale
        out = (a - lo) / (hi - lo)
        out = np.clip(out, 0, 1)
        out = (out * 255.0).astype(np.uint8)

        return out

    axial_u8    = to_uint8(axial_img)
    coronal_u8  = to_uint8(coronal_img)
    sagittal_u8 = to_uint8(sagittal_img)

    def save_png(path, arr):
        # plt.imsave(path, arr, cmap="gray")
        pil_image = Image.fromarray(arr, 'L')
        pil_image.save(path)

    save_png(os.path.join(save_dir, f"{prefix}_axial.png"),    axial_u8)
    save_png(os.path.join(save_dir, f"{prefix}_coronal.png"),  coronal_u8)
    save_png(os.path.join(save_dir, f"{prefix}_sagittal.png"), sagittal_u8)

def saving_ckpt(model, loss, args):
    if args.is_saving_ckpt:
        # save the best checkpoint during the training trajectory
        os.makedirs(os.path.dirname(args.ckpt_path), exist_ok=True)

        torch.save(model.state_dict(), args.ckpt_path)
        print(f"[Checkpoint] Best model updated (val_loss={loss:.4f}) → {args.ckpt_path}")
        return

    print(f"[Checkpoint] NOT SAVING IT")
