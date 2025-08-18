def read_nii_data(file_path):
    """
    Read NIfTI file data.

    Args:
    file_path (str): Path to the NIfTI file.

    Returns:
    np.ndarray: NIfTI file data.
    """
    try:
        nii_img = nib.load(file_path)
        nii_data = nii_img.get_fdata()
        return nii_data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def save_raw_file(file_path, phase='valid'):
    img_data = read_nii_data(file_path)
    if img_data is None:
        print(f"Read {file_path} unsuccessful. Passing")
        return
    # img_data = np.flip(np.rot90(img_data, 1, (0, 1)), axis=(0, 1))

    save_folder = r'M:\CTRATE_Volumes_raw_h5_fp16_valid_from_windows' #save folder for preprocessed
    file_name = os.path.basename(file_path)
    folder_path_new = os.path.join(save_folder, f"{phase}_" + file_name.split("_")[1], f"{phase}_" + file_name.split("_")[1] + file_name.split("_")[2]) #folder name for train or validation
    os.makedirs(folder_path_new, exist_ok=True)
    file_name = file_name.split(".")[0]+".h5"  # .pt file
    save_path = os.path.join(folder_path_new, file_name)
    save_to_hdf5(save_path, img_data.astype(np.float16))