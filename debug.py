
"""
trying to see which preprocessed .nii.gz file is corrupted that causes the LoadImaged error.
"""

from monai.transforms import LoadImaged
from monai.data import ITKReader
from tqdm import tqdm
from merlin.data.ctRate_dataloader import CTRateReportDataset

import argparse

# -----------------------
# Parse Arguments
# -----------------------
parser = argparse.ArgumentParser(description="Debug script")
parser.add_argument("--data_split", type=str, default='train', help="train or val")
args = parser.parse_args()

assert args.data_split in ['train', 'val']

loader = LoadImaged(keys=["image"], reader=ITKReader(image_only=False))
ctrate_dataset = CTRateReportDataset(
    data_folder=f'/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_nii_fp16_noflip_merlin_preprocessed_{args.data_split}/',
    report_csv='/cluster/projects/mcintoshgroup/publicData/CT-RATE/dataset/radiology_text_reports/train_reports.csv'
)

for p in tqdm(ctrate_dataset.datalist, desc=f"Files", leave=False):
    path = p[0]
    try:
        loader({"image": str(path)})
    except Exception as e:
        print("❌ Bad file:", path, "|", e, flush=True)

