import os
import json
from models.baselines.no_text import evaluate_MLP
from datautils import read_patent_splits

EXP_PATH = "/home/arnaik/ml_project/10701/experiments"
def write_metrics(metrics: dict, path: str):
    exp_folder = os.path.join(EXP_PATH, path)
    os.makedirs(exp_folder, exist_ok=True)
    metrics_path = os.path.join(exp_folder, "metrics.json")
    json.dump(metrics, open(metrics_path, "w"))



if __name__ == "__main__":
    data = read_patent_splits(
        "../../data/patent_acceptance_pred_subs=0.2_no_desc.jsonl", use_tqdm=True
    )
    layers = []
    evaluate_MLP()