import os

level_1_structure = ["0_raw", "1_BIDS", "2_nnUNet", "3_config", "4_splits", "5_scripts", "6_checkpoints", "7_analysis", "8_results", "9_tests"]
level_2_scripts_structure = ["00_utils", "01_create_splits", "02_nnunet", "03_preprocess", "04_train", "05_predict", "06_evaluate"]

def create_dataset_structure(dataset_name):
    base_path = os.path.join("datasets", dataset_name)
    for level in level_1_structure:
        os.makedirs(os.path.join(base_path, level), exist_ok=True)
    for level in level_2_scripts_structure:
        os.makedirs(os.path.join(base_path, "5_scripts", level), exist_ok=True)
        
    
