import sys
import os
import json
import numpy as np

# Add SynthSeg to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SynthSeg')))

from SynthSeg.brain_generator import BrainGenerator
from ext.lab2im import utils

def main():
    split_file = "splits/brats_subject_split.json"
    with open(split_file, "r") as f:
        data = json.load(f)
    
    train_subjects = data["train_subjects"]
    in_dir = "data/Task01_BrainTumour/labelsTr"
    
    out_img_dir = "data/v21_synthseg_offline/images"
    out_lbl_dir = "data/v21_synthseg_offline/labels"
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    
    for subj in train_subjects:
        label_map_path = os.path.join(in_dir, f"{subj}.nii.gz")
        
        # Configure BrainGenerator to use the sparse BraTS labels (Classes 0, 1, 2, 3)
        bg = BrainGenerator(labels_dir=label_map_path, generation_classes=np.array([0, 1, 2, 3]))
        
        for i in range(5):
            print(f"Generating image {i} for {subj}")
            im, lab = bg.generate_brain()
            
            img_out_path = os.path.join(out_img_dir, f"{subj}_synth_{i}.nii.gz")
            lab_out_path = os.path.join(out_lbl_dir, f"{subj}_synth_{i}.nii.gz")
            
            utils.save_volume(im, bg.aff, bg.header, img_out_path)
            utils.save_volume(lab, bg.aff, bg.header, lab_out_path)

if __name__ == "__main__":
    main()
