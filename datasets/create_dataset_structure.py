import os

# Slot type prefixes must match validate_standard_dataset_structure.py exactly
# (e.g. slot 3 is "conf", not "config"; slot 8 results subdirs are 01_predictions/02_metrics).
level_1_structure = ["0_raw", "1_BIDS", "2_nnUNet", "3_conf", "4_splits", "5_scripts", "6_checkpoints", "7_analysis", "8_results", "9_tests"]
# 2_nnUNet required subdirs (validator: REQUIRED_NNUNET_SUBDIRS)
level_2_nnunet_structure = ["raw", "preprocessed"]
level_2_scripts_structure = ["00_utils", "01_create_splits", "02_nnunet", "03_preprocess", "04_train", "05_predict", "06_evaluate"]
# 8_results subdirs:
#   01_predictions/{nnUNet,auglab,...}/{run_id}/fold{k}/{contrast}/*.nii.gz
#   02_metrics/{category}_{run_id}/fold{k}/  +  02_00_aggregated_metrics.md
level_2_results_structure = ["01_predictions", "02_metrics"]

def create_dataset_structure(dataset_name):
    base_path = dataset_name
    for level in level_1_structure:
        os.makedirs(os.path.join(base_path, level + "_" + dataset_name), exist_ok=True)
    for level in level_2_nnunet_structure:
        os.makedirs(os.path.join(base_path, "2_nnUNet_" + dataset_name, level), exist_ok=True)
    for level in level_2_scripts_structure:
        os.makedirs(os.path.join(base_path, "5_scripts_" + dataset_name, level), exist_ok=True)
    for level in level_2_results_structure:
        os.makedirs(os.path.join(base_path, "8_results_" + dataset_name, level), exist_ok=True)

if __name__ == "__main__":
    dataset_name = input("Enter the name of the dataset: ")
    create_dataset_structure(dataset_name)
    print(f"Dataset structure for '{dataset_name}' created successfully at {os.path.join('.', dataset_name)}.")