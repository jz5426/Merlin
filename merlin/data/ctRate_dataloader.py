import os
import glob
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import tqdm
import pandas as pd
import nibabel as nib

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
    CenterSpatialCropd,
)


### eval code reference
# /cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_noflip_processed_val_images
# NOTE: checkout the Datafolder class in eval.py and the collect_performance.py for evaluation of the fine-tuned model.

### Training code reference
# '/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_noflip_processed_train_images'
# NOTE: checkout caption_datasets.py in fvlm repository.

def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.Tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array

class CTReportDataset(Dataset):
    def __init__(self, data_folder, report_csv, min_slices=20, resize_dim=500):
        self.split = 'train' if 'train' in data_folder.lower() else 'val'

        # NOTE: must use this transformation from Merlin
        self.merlin_transform = Compose( # this is a set of deterministic transform functions
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"), # TODO: check this
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")), # TODO: check this.
                ScaleIntensityRanged(
                    keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
                ),
                SpatialPadd(keys=["image"], spatial_size=[224, 224, 160]),
                CenterSpatialCropd(
                    roi_size=[224, 224, 160],
                    keys=["image"],
                ),
                ToTensord(keys=["image"]),
            ]
        )
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.accession_to_text = None
        self.paths=[]
        self.accession_to_text = self.load_accession_text(report_csv)            
        self.datalist = self.prepare_samples()
        print('number of files ', len(self.datalist))

        self.count = 0

    def load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['VolumeName']] = row["Findings_EN"], row['Impressions_EN']
        # each key is a tuple, tuple[0] is findings and tuple[1] is impression
        return accession_to_text

    def nii_img_to_tensor(self, path):

        transformed_tensor = self.merlin_transform({'image': path})
        img_tensor = transformed_tensor["image"]

        return img_tensor

    def prepare_samples(self):
        datalist = []
        for patient_folder in tqdm.tqdm(glob.glob(os.path.join(self.data_folder, '*'))):
            for accession_folder in glob.glob(os.path.join(patient_folder, '*')):
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))
                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]
                    if accession_number not in self.accession_to_text:
                        continue
                    # TODO: if does not work, use the CT-CLIP implementation here
                    # report text
                    findings_impressions = self.accession_to_text[accession_number] # [0] is findings and [1] is impression

                    # combine the findings and impression sections
                    input_text_concat = ""
                    for text in findings_impressions:
                        if text == "Not given.":
                            text=""
                        input_text_concat = input_text_concat + str(text)

                    datalist.append((nii_file, input_text_concat))
                    self.paths.append(nii_file)
        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        nii_file, input_text = self.datalist[index]
        video_tensor = self.nii_img_to_tensor(nii_file)
        input_text = str(input_text)
        input_text = input_text.replace('"', '')
        input_text = input_text.replace('\'', '')
        input_text = input_text.replace('(', '')
        input_text = input_text.replace(')', '')

        data = {
            'image_id': os.path.basename(nii_file),
            'image': video_tensor,
            'text': input_text
        }
        return data

