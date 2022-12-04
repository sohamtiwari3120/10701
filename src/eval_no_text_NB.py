import os
import json
import pandas as pd
from datautils import read_patent_splits
from models.baselines.no_text import evaluate_NB

# import warnings
# warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
pd.options.mode.chained_assignment = None  # default='warn'

EXP_PATH = "/home/arnaik/ml_project/10701/experiments"
def write_metrics(metrics: dict, path: str):
    exp_folder = os.path.join(EXP_PATH, path)
    os.makedirs(exp_folder, exist_ok=True)
    metrics_path = os.path.join(exp_folder, "metrics.json")
    json.dump(metrics, open(metrics_path, "w"))

if __name__ == "__main__":
    data = read_patent_splits(
        "../data/patent_acceptance_pred_subs=0.2.jsonl", use_tqdm=True
    )
    for algo in ["gaussian", "multinom", "complement", "bernoulli", "categorical"]:
        print(f"Using algo: {algo}:")
        
        print(f"\x1b[34;1musing all features:\x1b[0m")
        metrics = evaluate_NB(data, algo, feats=["examiner_namer", "inventor_list", "topic_id"])
        write_metrics(metrics, algo+"_all_feats")
        
        print(f"\x1b[34;1mnot using `inventor_list`:\x1b[0m")
        metrics = evaluate_NB(data, algo, feats=["examiner_namer", "topic_id"])
        write_metrics(metrics, algo+"_no_inventor")

        print(f"\x1b[34;1musing `examiner_namer`:\x1b[0m")
        metrics = evaluate_NB(data, algo, feats=["examiner_namer"])
        write_metrics(metrics, algo+"_examiner_namer")
        
        print(f"\x1b[34;1musing `inventor_list`:\x1b[0m")
        metrics = evaluate_NB(data, algo, feats=["inventor_list"])
        write_metrics(metrics, algo+"_inventor_list")
        
        print(f"\x1b[34;1musing `topic_id`:\x1b[0m")
        metrics = evaluate_NB(data, algo, feats=["topic_id"])
        write_metrics(metrics, algo+"_topic_id")
    # F1-micro == accuracy, F1-macro, F1-weighted.