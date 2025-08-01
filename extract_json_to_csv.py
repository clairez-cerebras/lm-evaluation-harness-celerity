#!/usr/bin/env python3
import os
import json
import csv

# filepath: /mnt/local/shared/clairez/LD/lm-evaluation-harness/run_scripts/extract_json_to_csv.py
base_dir_path = "/cb/home/clairez/ws/depth_mup/lm-evaluation-harness/output"


for model_dir in os.listdir(base_dir_path):
    print(f"*** Processing {model_dir} ***")

    model_dir_path = os.path.join(base_dir_path, model_dir)
    try:
        model_dir_sub = os.listdir(model_dir_path)[0]
    except:
        print(f"=== No subdirectory found in {model_dir_path} ===")
        continue
    model_dir_sub_path = os.path.join(model_dir_path, model_dir_sub)

    
    csv_rows = []
    csv_rows.append(["model_name", model_dir])

    json_file_list = [f for f in os.listdir(model_dir_sub_path) if f.endswith('.json')]
    if len(json_file_list) == 0:
        print(f"=== No JSON files found in {model_dir_sub_path} ===")
        continue
    
    for file in json_file_list:
        file_path = os.path.join(model_dir_sub_path, file)
        with open(file_path, "r") as f:
            data = json.load(f)
            f.close()
        results = data["results"]
        for task in results.keys():
            if "mmlu_" in task:
                continue
            for k, v in results[task].items():
                if "alias" in k:
                    continue
                if "acc" in k:
                    v *= 100
                csv_rows.append([task, k, v])
    
    output_csv = os.path.join(model_dir_sub_path, "results_summary.csv")
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for row in csv_rows:
            writer.writerow(row)

    print(f"=== Done with {model_dir_sub_path} ===")