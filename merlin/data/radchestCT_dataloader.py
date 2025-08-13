# CT_Scan_Metadata_Complete_35747: the scale and orientation information can be found in this metadata file. note that the 

# inference only dataloader

import os
import glob
from torch.utils.data import Dataset
import torch.nn.functional as F
import tqdm
import pandas as pd
from nibabel.orientations import aff2axcodes
from monai.data import NibabelReader
from monai.data import ITKReader

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    EnsureTyped,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
    CenterSpatialCropd,
)

from merlin.train_utils import save_middle_slices_normalized

class RadchestCTInferenceDataloader(Dataset):
    def __init__(self, data_folder, report_csv, min_slices=20):
        self.split = 'train' if 'train' in data_folder.lower() else 'val'

        # NOTE: must use this transformation from Merlin
        self.merlin_transform = Compose( # this is a set of deterministic transform functions
            [
                LoadImaged(keys=["image"], reader=ITKReader(image_only=False)),
                EnsureTyped(keys="image", track_meta=True), # manually added
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
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

        # NOTE: for testing only
        self.debug_transform = Compose( # this is a set of deterministic transform functions
            [
                LoadImaged(keys=["image"], reader=ITKReader(image_only=False)),
                EnsureTyped(keys="image", track_meta=True), # manually added
                EnsureChannelFirstd(keys=["image"]),
                # Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
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

    def nii_to_tensor(self, path):
        # non-debug implementation
        transformed_tensor = self.merlin_transform({'image': path})
        img_tensor = transformed_tensor["image"]
        assert aff2axcodes(img_tensor.meta["affine"]) == ('R', 'A', 'S')

        # NOTE: project a slice and visualize.
      #   save_middle_slices_normalized(
      #       self.debug_transform({'image': path})["image"],
      #       save_dir='/cluster/home/t135419uhn/Merlin/visualize_transformed_ct_slices_LPS')
        
        # relative to LPS slices, the RAS slices should be
        # axial slice: 180 degree rotation
        # coronal slice: x flip
        # sagittal slice: y flip
        # It computes the flips/rotations needed so the voxel axes 
        # now match the desired mapping while still being correct in the reference frame.
      #   save_middle_slices_normalized(
      #       img_tensor,
      #       save_dir='/cluster/home/t135419uhn/Merlin/visualize_transformed_ct_slices_RAS')
        
        return img_tensor

        # DEBUG ONLY
        # tx = Compose([
        #     LoadImaged(keys="image", image_only=False, reader=NibabelReader),
        #     EnsureTyped(keys="image", track_meta=True),
        # ])

        # d = tx({"image": path})
        # print("BEFORE(meta):", aff2axcodes(d["image"].meta["affine"]))

        # orient = Orientationd(keys="image", axcodes="RAS")
        # d = orient(d)  # <-- reassign!

        # print("AFTER(meta):", aff2axcodes(d["image"].meta["affine"]))
        # print('HI')

    def prepare_samples(self):
      # TODO: only need to contain images
        datalist = []
        for patient_folder in tqdm.tqdm(glob.glob(os.path.join(self.data_folder, '*'))):
            for accession_folder in glob.glob(os.path.join(patient_folder, '*')):
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))
                # nii_files = glob.glob(os.path.join(accession_folder, '*.h5'))
                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]
                    # accession_number = accession_number.replace('.h5', '.nii.gz')
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
        video_tensor = self.nii_to_tensor(nii_file)

        data = {
            'image_id': os.path.basename(nii_file),
            'image': video_tensor,
            # no need to have labels
        }
        return data

