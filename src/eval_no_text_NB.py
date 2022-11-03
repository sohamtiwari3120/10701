import os
import json
from datautils import read_patent_splits
from models.baselines.no_text import evaluate_NB

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
    for algo in ["gaussian", "multinom", "complement", "bernoulli", "categorical"]:
        print(f"Using algo: {algo}:")
        
        print(f"using all features:")
        metrics = evaluate_NB(data, algo, feats=["examiner_namer", "inventor_list", "topic_id"])
        write_metrics(metrics, algo+"_all_feats")
        
        print(f"not using `inventor_list`:")
        metrics = evaluate_NB(data, algo, feats=["examiner_namer", "topic_id"])
        write_metrics(metrics, algo+"_no_inventor")

        print(f"using `examiner_namer`:")
        metrics = evaluate_NB(data, algo, feats=["examiner_namer"])
        write_metrics(metrics, algo+"_examiner_namer")
        
        print(f"using `inventor_list`:")
        metrics = evaluate_NB(data, algo, feats=["inventor_list"])
        write_metrics(metrics, algo+"_inventor_list")
        
        print(f"using `topic_id`:")
        metrics = evaluate_NB(data, algo, feats=["topic_id"])
        write_metrics(metrics, algo+"_topic_id")
    # F1-micro == accuracy, F1-macro, F1-weighted.