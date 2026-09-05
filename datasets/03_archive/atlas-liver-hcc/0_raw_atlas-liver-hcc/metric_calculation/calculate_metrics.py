import os
import csv
import argparse
from evaluator import aggregate_scores

def calcultate_metrics(segmentation_folder, label_folder, author, status, csv_output_file=None, json_output_file=None, num_threads=8):

    """"
    param segmentation_folder: folder with the segmentations under nifty format
    param label_folder: folder with the grond truth images under nifty format
    param output_file: path to the output csv file with average performances
    param json_output_file: path to the json output file with performances per image
    param num_threads: number of cpu threads to parallelize the computations
    return:
    """
    pred_gt_tuples = []
    for i, p in enumerate(os.listdir(label_folder)):
        if p.endswith('nii.gz'):
            file = os.path.join(label_folder, p)
            pred_gt_tuples.append([os.path.join(segmentation_folder, 'im' + p[2:]),file])

    scores = aggregate_scores(pred_gt_tuples, labels=[[1, 2], 2],
                         json_output_file=json_output_file, num_threads=num_threads)
    if csv_output_file != None:
        evaluation_metrics = {
            "Authors": author,
            "Liver ASD (mm³)": round(scores["mean"]["[1, 2]"]["Avg. Symmetric Surface Distance"], 1),
            "Liver Dice (%)": round(scores["mean"]["[1, 2]"]["Dice"] * 100, 1),
            "Liver Hausdorff Distance (mm³)": round(scores["mean"]["[1, 2]"]["Hausdorff Distance"], 1),
            "Liver Surface Dice (%)": round(scores["mean"]["[1, 2]"]["Surface Dice"] * 100, 1),
            "Tumor ASD (mm³)": round(scores["mean"]["2"]["Avg. Symmetric Surface Distance"], 1),
            "Tumor Dice (%)": round(scores["mean"]["2"]["Dice"] * 100, 1),
            "Tumor Hausdorff Distance (mm³)": round(scores["mean"]["2"]["Hausdorff Distance"], 1),
            "Tumor Surface Dice (%)": round(scores["mean"]["2"]["Surface Dice"] * 100, 1),
            "RMSE on Tumor Burden (%)": round(scores["mean"]["RMSE on Tumor Burden"] * 100, 1),
            "Status": status
                              }

        # Check if file exists, if not, write header
        if not os.path.isfile(csv_output_file):
            with open(csv_output_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=evaluation_metrics.keys())
                writer.writeheader()

        with open(csv_output_file, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=evaluation_metrics.keys())
            writer.writerow(evaluation_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Metrics calculation')
    parser.add_argument('--segmentation_folder', default="/path/to/the/segmentation/folder", type=str)
    parser.add_argument('--label_folder', default="/path/to/the/label/folder/labelsTr", type=str)
    parser.add_argument('--author', default="Author 1", type=str)
    parser.add_argument('--status', default="Docker container submitted", type=str)
    parser.add_argument('--csv_output_file', default="/path/to/the/output/csv/file.csv", type=str)
    parser.add_argument('--json_output_file', default="/path/to/the/output/json/file.json", type=str)
    parser.add_argument('--num_threads', default=8, type=int)

    args = parser.parse_args()
    calcultate_metrics(args.segmentation_folder, args.label_folder, args.author, args.status, csv_output_file=args.csv_output_file, json_output_file=args.json_output_file, num_threads=args.num_threads)