import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import glob
import re


def process_files(outputs_folder, table_name):

    ml_files = sorted(glob.glob(os.path.join(outputs_folder, "ml_ds_k*.txt")))

    def extract_k(filename):
        match = re.search(r'k(\d+)', filename)
        return match.group(1) if match else None

    k_values = sorted(set(extract_k(f) for f in ml_files))
    summary_rows = []

    for k in k_values:
        ml_path = os.path.join(outputs_folder, f"ml_ds_k{k}.txt")
        re_path = os.path.join(outputs_folder, f"re_ds_k{k}.txt")

        df_ml = pd.read_csv(ml_path)
        df_re = pd.read_csv(re_path)

        df = pd.concat([df_ml, df_re], ignore_index=True)

        stickiness_avg = df["stickiness"].mean()
        iterations_total = len(df)
        lock_fails_total = df["lock_fails"].sum()
        contention = iterations_total * 2 / (iterations_total * 2 + lock_fails_total)
        rank_error_total = df["rank_error"].sum()
        delay_toal = df["delay"].sum()

        summary_rows.append({
            "stickiness": f"{k}",
            "iterations": int(iterations_total),
            "contention": round(contention, 3),
            "rank_error" : rank_error_total,
            "delay" : delay_toal
        })

    summary_df = pd.DataFrame(summary_rows)
    output_path = os.path.join(outputs_folder, f"{table_name}.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"Saved summary to {output_path}")




def main():
    """Main function to handle argument parsing and execution."""
    parser = argparse.ArgumentParser(description="Process metrics and rank error files.")
    parser.add_argument('-l', type=str, required=True, help="Path to logs folder.", default='logs')
    parser.add_argument('-p', type=str, required=False, help="Table name.", default='stick_comparison')

    
    args = parser.parse_args()
    
    # Call processing function with arguments
    process_files(args.l, args.p)

if __name__ == "__main__":
    main()
