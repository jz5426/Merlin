"""
similar to the demo.py file but as a training script
"""

"""
Download Merlin and test the model on sample data that is downloaded from huggingface
"""

import os
import warnings
import torch

from merlin.data import download_sample_data
# from merlin.data import DataLoader
from torch.utils.data import DataLoader 

from merlin import Merlin
from merlin.data.ctRate_dataloader import CTReportDataset


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

# load model
model = Merlin().to(device)

# training loop
for batch in train_loader:
    # TODO: double check image shape.
    img, text = batch['image'].to(device), batch['text']

# model.eval()
# model.cuda()

# data_dir = os.path.join(os.path.dirname(__file__), "abct_data")
# cache_dir = data_dir.replace("abct_data", "abct_data_cache")

# datalist = [
#     {
#         "image": download_sample_data(
#             data_dir
#         ),  # function returns local path to nifti file
#         "text": "Lower thorax: A small low-attenuating fluid structure is noted in the right cardiophrenic angle in keeping with a tiny pericardial cyst."
#         "Liver and biliary tree: Normal. Gallbladder: Normal. Spleen: Normal. Pancreas: Normal. Adrenal glands: Normal. "
#         "Kidneys and ureters: Symmetric enhancement and excretion of the bilateral kidneys, with no striated nephrogram to suggest pyelonephritis. "
#         "Urothelial enhancement bilaterally, consistent with urinary tract infection. No renal/ureteral calculi. No hydronephrosis. "
#         "Gastrointestinal tract: Normal. Normal gas-filled appendix. Peritoneal cavity: No free fluid. "
#         "Bladder: Marked urothelial enhancement consistent with cystitis. Uterus and ovaries: Normal. "
#         "Vasculature: Patent. Lymph nodes: Normal. Abdominal wall: Normal. "
#         "Musculoskeletal: Degenerative change of the spine.",
#     },
# ]

# # TODO: replace with ct-rate dataloader
# dataloader = DataLoader(
#     datalist=datalist,
#     cache_dir=cache_dir,
#     batchsize=8,
#     shuffle=True,
#     num_workers=0,
# )

# # TODO: replace with training loop
# for batch in dataloader:
#     outputs = model(batch["image"].to(device), batch["text"])
#     print("\n================== Output Shapes ==================")
#     print(f"Contrastive image embeddings shape: {outputs[0].shape}")
#     print(f"Phenotype predictions shape: {outputs[1].shape}")
#     print(f"Contrastive text embeddings shape: {outputs[2].shape}")

# # TODO: replace with evaluation loop and save the prediction results just like fvlm code
# ## Get the Image Embeddings
# model = Merlin(ImageEmbedding=True)
# model.eval()
# model.cuda()

# for batch in dataloader:
#     outputs = model(
#         batch["image"].to(device),
#     )
#     print("\n================== Output Shapes ==================")
#     print(
#         f"Image embeddings shape (Can be used for downstream tasks): {outputs[0].shape}"
#     )
