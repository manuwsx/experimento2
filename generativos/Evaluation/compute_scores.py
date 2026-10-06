import sys
import os
import argparse
import json
import torch
sys.path.append('.')
from utils.eval_helper import compute_score 

parser = argparse.ArgumentParser()
parser.add_argument("--samples", help="Path to generated samples .pth")
parser.add_argument("--ref", help="Path to reference .pth")
parser.add_argument("--output_json", help="Path to output JSON file")

args = parser.parse_args()

results = compute_score(args.samples, ref_name=args.ref, norm_box=True)

if args.output_json:
    output_data = {}
    
    # Load the existing data if the file aalready exists
    if os.path.exists(args.output_json):
        try:
            with open(args.output_json, 'r') as f:
                output_data = json.load(f)
        except json.JSONDecodeError:
            output_data = {}

    clean_results = {}
    
    clean_results["path_samples"] = args.samples
    clean_results["path_ref"] = args.ref

    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            try:
                clean_results[k] = v.item()
            except:
                clean_results[k] = v.tolist()
        elif isinstance(v, (float, int, str)):
            clean_results[k] = v
        elif isinstance(v, dict):
            clean_results[k] = v
            
    output_data[args.samples] = clean_results

    # Save the updated json file
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Results saved at {args.output_json}")