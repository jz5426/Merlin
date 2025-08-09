"""
script is copied from fvlm.
"""

import ast
import concurrent.futures
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import tqdm
import h5py

def process_row(row):
    # Set up directory parameters
    VolumeName = row["VolumeName"]
    dir1 = VolumeName.rsplit("_", 1)[0]
    dir2 = VolumeName.rsplit("_", 2)[0]
    #TODO: organize the files in the following fomrat if necessary.
    # filepath = os.path.join(data_root, f"{split}", dir2, dir1, VolumeName)

    # transform from "train_1_a" to "train_1a" NOTE:TODO: this is just temporary changes, the file structure should follows exactly the same as the one from metadata file ideally.
    dir1 = dir1[::-1].replace("_", "", 1)[::-1] 

    # Handle compound extensions like .nii.gz
    base, ext = os.path.splitext(VolumeName)
    if ext == ".gz":
        base, _ = os.path.splitext(base)  # strip .nii as well
    VolumeName = base + ".h5"
    filepath = os.path.join(f'/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_noflip', dir2, dir1, VolumeName)
    dirpath = os.path.dirname(filepath)
    dirpath = dirpath.replace(f"/CTRATE_Volumes_raw_h5_fp16_noflip/", f"/CTRATE_Volumes_raw_nii_fp16_noflip_merlin_preprocessed/")

    # skip the file from csv if not exists in the data directory (intentional)
    if not os.path.exists(filepath):
        print('File not exists in the given directory => Skipping ', filepath)
        return 

    if os.path.exists(os.path.join(dirpath, os.path.basename(filepath).replace('.h5', '.nii.gz'))):
        return

    # Read Image
    if filepath.endswith('h5'):
        with h5py.File(filepath, "r") as f:
            image_np = f["ct"][:].astype(np.float32) # a numpy array (512, 512, 303), which is the original shape, by convention it is x,y,z
            image_np = np.transpose(image_np, (2, 1, 0)) # become z, y, x for sitk.
        # NOTE: This assumes axis order of the input is z, y, x (transposed alreadya this point)
        image = sitk.GetImageFromArray(image_np)  

    # Set Spacing
    (x, y), z = map(float, ast.literal_eval(row["XYSpacing"])), row["ZSpacing"]
    image.SetSpacing((x, y, z))

    # Set Origin
    image.SetOrigin(ast.literal_eval(row["ImagePositionPatient"]))

    # Set Direction
    orientation = ast.literal_eval(row["ImageOrientationPatient"])
    row_cosine, col_cosine = orientation[:3], orientation[3:6]
    z_cosine = np.cross(row_cosine, col_cosine).tolist()
    image.SetDirection(row_cosine + col_cosine + z_cosine)

    # Fix Rescale
    RescaleIntercept = row["RescaleIntercept"]
    RescaleSlope = row["RescaleSlope"]
    adjusted_hu = image * RescaleSlope + RescaleIntercept

    # Convert the image to int16
    adjusted_hu = sitk.Cast(adjusted_hu, sitk.sitkInt16)

    # Write Image
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    # replace
    base, ext = os.path.splitext(filepath)
    base += '.nii.gz'
    sitk.WriteImage(adjusted_hu, os.path.join(dirpath, os.path.basename(base))) # note that the array shape should be still z, y, x

    # Read the NIfTI file
    # img = sitk.ReadImage(os.path.join(dirpath, os.path.basename(base)))
    # arr = sitk.GetArrayFromImage(img)
    # print('finish processing ', os.path.join(dirpath, os.path.basename(base)))

if __name__ == "__main__":


    # split = 'train'
    # d = "validation" if split == "val" else "train"
    
    data_root = Path("/cluster/projects/mcintoshgroup/publicData/CT-RATE/dataset/")
    # metadata = pd.read_csv(os.path.join(data_root, f"metadata/{d}_metadata.csv")) # TODO: uncomment this when the val splits actually comes from the val_metadata
    metadata = pd.read_csv(os.path.join(data_root, f"metadata/train_metadata.csv"))
    rows = [row[1] for row in metadata.iterrows()]

    # filter rows
    filtered_rows = []
    for row in rows:
        VolumeName = row["VolumeName"]
        dir1 = VolumeName.rsplit("_", 1)[0]
        dir2 = VolumeName.rsplit("_", 2)[0]
        #TODO: organize the files in the following fomrat if necessary.
        # filepath = os.path.join(data_root, f"{split}", dir2, dir1, VolumeName)

        # transform from "train_1_a" to "train_1a" NOTE:TODO: this is just temporary changes, the file structure should follows exactly the same as the one from metadata file ideally.
        dir1 = dir1[::-1].replace("_", "", 1)[::-1] 

        # Handle compound extensions like .nii.gz
        base, ext = os.path.splitext(VolumeName)
        if ext == ".gz":
            base, _ = os.path.splitext(base)  # strip .nii as well
        VolumeName = base + ".h5"
        filepath = os.path.join(f'/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_noflip', dir2, dir1, VolumeName)
        dirpath = os.path.dirname(filepath)
        dirpath = dirpath.replace(f"/CTRATE_Volumes_raw_h5_fp16_noflip/", f"/CTRATE_Volumes_raw_nii_fp16_noflip_merlin_preprocessed/")

        # skip the file from csv if not exists in the .h5 data directory (intentional)
        if not os.path.exists(filepath):
            continue
        filtered_rows.append(row)
    print(f'number of remaining rows: {len(filtered_rows)}')

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm.tqdm(executor.map(process_row, filtered_rows), total=len(filtered_rows)))

    # process_row(rows[0])
    print('finished preprocess_data.py script for Merlin data')
