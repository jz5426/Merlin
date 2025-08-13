
import ast
import concurrent.futures
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import tqdm
import h5py

from pathlib import Path
import shutil
from typing import Dict, List
import random

# NOTE: check https://github.com/rachellea/ct-volume-preprocessing for the preprocessing code.
def process_row(row):
	# get the file name from the csv file
	VolumeName = row["VolumeAcc_DEID"] # with .npz extension
	data_filepath = os.path.join(f'/cluster/projects/mcintoshgroup/publicData/RADChestCT/radchest_ct_npz', VolumeName)
	dirpath = os.path.dirname(data_filepath)
	dirpath = dirpath.replace(f"/radchest_ct_npz/", f"/radchest_ct_preprocessed_nii/")

	# NOT EVERY FILE FROM THE CSV EXISTS IN THE DATA DIRECTORY: skip the file from csv if not exists in the original data directory (intentional)
	if not os.path.exists(data_filepath):
		print('File not exists in the given directory => Skipping ', data_filepath)
		return 

	# skip the file if it is already preprocessed in the /cluster/projects/mcintoshgroup/publicData/RADChestCT/radchest_ct_npz/radchest_ct_preprocessed_nii directory
	if os.path.exists(os.path.join(dirpath, os.path.basename(data_filepath).replace('.npz', '.nii.gz'))):
		return

	# Read Image
	if data_filepath.endswith('npz'):
		data = np.load(data_filepath)
		# TODO: check if the metadata file inside the npz file otherwise use the metadata file.
		# # List all arrays stored in the file
		# print(data.files)

		# # Access each array by its key
		# for key in data.files:
		# 	print(f"{key}: {data[key]}")

		# TODO: transpose like previous work?

		# with h5py.File(data_filepath, "r") as f:
		# 	image_np = f["ct"][:].astype(np.float32) # a numpy array (512, 512, 303), which is the original shape, by convention it is x,y,z
		# 	image_np = np.transpose(image_np, (2, 1, 0)) # become z, y, x for sitk.

		# NOTE: This assumes axis order of the input is z, y, x (transposed alreadya this point)
		image = sitk.GetImageFromArray(image_np)  

	# Set Spacing
	x, y, z = map(float, ast.literal_eval(row["final_spacing"])), map(float, ast.literal_eval(row["final_spacing"])), map(float, ast.literal_eval(row["final_spacing"]))
	image.SetSpacing((x, y, z))

	# Set Origin
	image.SetOrigin((0.0, 0.0, 0.0)) # NOTE: no information available on the metadata file, but this default should be fine for voxel-based ML.

	# Set Direction
	orientation = ast.literal_eval(row["orig_orientation"])
	row_cosine, col_cosine = orientation[:3], orientation[3:6]
	z_cosine = np.cross(row_cosine, col_cosine).tolist()
	image.SetDirection(row_cosine + col_cosine + z_cosine)

	# Fix Rescale
	RescaleIntercept = row["orig_inter"]
	RescaleSlope = row["orig_slope"]
	adjusted_hu = image * RescaleSlope + RescaleIntercept

	# Convert the image to int16
	adjusted_hu = sitk.Cast(adjusted_hu, sitk.sitkInt16)

	# Write Image
	Path(dirpath).mkdir(parents=True, exist_ok=True)

	# replace
	base, ext = os.path.splitext(data_filepath)
	base += '.nii.gz'
	sitk.WriteImage(adjusted_hu, os.path.join(dirpath, os.path.basename(base))) # note that the array shape should be still z, y, x

	# Read the NIfTI file
	# img = sitk.ReadImage(os.path.join(dirpath, os.path.basename(base)))
	# arr = sitk.GetArrayFromImage(img)
	# print('finish processing ', os.path.join(dirpath, os.path.basename(base)))

def process_npz_to_nii_main():
	# NOTE: to save safe, can only the filter the test split in radchest ct
    data_root = Path("/cluster/projects/mcintoshgroup/publicData/RADChestCT/")
    metadata = pd.read_csv(os.path.join(data_root, f"radchest_ct_npz/CT_Scan_Metadata_Complete_35747.csv"))
    rows = [row[1] for row in metadata.iterrows()]

    # filter rows
    filtered_rows = []
    for row in rows:
        VolumeName = row["VolumeAcc_DEID"] # including extension .nii.gz

        filepath = os.path.join(f'/cluster/projects/mcintoshgroup/publicData/RADChestCT/radchest_ct_npz', VolumeName)
        dirpath = os.path.dirname(filepath)

		# NOT EVERY FILE FROM THE CSV EXISTS IN THE DATA DIRECTORY: skip the file from csv if not exists in the original data directory (intentional)
        if not os.path.exists(filepath):
            continue
        filtered_rows.append(row)
    print(f'number of remaining rows: {len(filtered_rows)}')

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm.tqdm(executor.map(process_row, filtered_rows), total=len(filtered_rows)))

    # process_row(rows[0])
    print('finished preprocess_data.py script for Merlin data')


if __name__ == '__main__':
      process_npz_to_nii_main()