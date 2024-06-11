import glob
import os
import sys

import numpy as np
import pandas as pd

try:
    task = sys.argv[1]
except IndexError:
    print("Must specify a task as the first commandline argument, e.g.:")
    print("    `python list_metrics.py.py enhancer_annotation`")
    sys.exit()
"""
Get the average and standard deviation of the MCC scores for all CV-folds of each model in the enhancer annotation task
"""
print(f"Listing metrics for {task.upper()}".replace("_", " "))
par_dir = os.path.dirname(os.path.dirname(__file__))

folder = f"{par_dir}/downstream_tasks/{task}/"
for model in os.listdir(folder):
    dfs = glob.glob(f"{folder}/{model}/**/best_model_metrics.csv", recursive=True)
    metric = []
    if len(dfs) == 0:
        continue
    for _df in dfs:
        df = pd.read_csv(_df, header="infer")
        test_col = [n for n in df.columns if n.startswith("test")][1]
        test_metric = test_col.split("_")[-1].upper()
        metric.append(df[test_col].values[0])

    with open(f"{folder}/{model}/summed_metrics.txt", "w") as f:
        f.write(
            f"Embedding: {model:<25} | N runs: {len(metric):>2} | {test_metric} : {np.mean(metric):.4f} ± {np.std(metric):.4f}\n"
        )
    print(
        f"Embedding: {model:<25} | N runs: {len(metric):>2} | {test_metric} : {np.mean(metric):.4f} ± {np.std(metric):.4f}"
    )
