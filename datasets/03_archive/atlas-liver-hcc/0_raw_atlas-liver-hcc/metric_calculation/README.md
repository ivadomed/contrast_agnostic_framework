## File organisation

calculate_metrics.py is the main file and will allow you to calculate the metrics used in the ATLAS challenge, values per images can be stored in a json file or average values in a csv file. The metrics are defined in the metrics.py file.
Images from the label file should be named "lbxx.nii.gz" and have a corresponding images in the segmentation file named "imxx.nii.gz".

## Run the code

Install python dependencies

`python -m pip install -r requirements.txt`

Calculate the default metrics and store the output in a json or csv file

`python calculate_metrics.py  --segmentation_folder /path/to/the/model/segmentation/ --label_folder /path/to/the/ground/truth/ --csv_output_file /path/to/the/output/csv/file --json_output_file /path/to/the/output/json/file`