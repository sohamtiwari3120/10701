import os
import torch
import random
import numpy as np
from datautils import read_patent_splits
from models.baselines.no_text import evaluate_MLP, featurize_data

# seed torch, random and numpy.
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

if __name__ == "__main__":
    data = read_patent_splits(
        "../data/patent_acceptance_pred_subs=0.2.jsonl", use_tqdm=True
    )
    train, val, test = data
    # featurize data:
    train, test, val = featurize_data(
        train, test, val, 
        features=["examiner_namer", 
        "inventor_list", "topic_id"]
    )
    data = [train, val, test]
    LOG_STEPS = 2000
    DEVICE = "cuda:0"
    for layers in [[100], [100, 100], [100, 200, 100, 200, 100]]:
        # all features:
        exp_name = f"MLP_{len(layers)}HL_all_feats"
        os.makedirs(os.path.join("experiments", exp_name), exist_ok=True)
        evaluate_MLP(data, log_step=LOG_STEPS, num_epochs=10, 
        exp_name=exp_name, use_tqdm=True, feats=["examiner_namer", 
        "inventor_list", "topic_id"], device=DEVICE, input_size=12,
        hidden_sizes=layers, num_classes=2)
        
        # every feature one at a time:
        exp_name = f"MLP_{len(layers)}HL_examiner_namer"
        os.makedirs(os.path.join("experiments", exp_name), exist_ok=True)
        evaluate_MLP(data, log_step=LOG_STEPS, num_epochs=10, 
        exp_name=exp_name, use_tqdm=True, feats=["examiner_namer"], 
        device=DEVICE, input_size=1, hidden_sizes=layers, num_classes=2)
        
        exp_name = f"MLP_{len(layers)}HL_inventor_list"
        os.makedirs(os.path.join("experiments", exp_name), exist_ok=True)
        evaluate_MLP(data, log_step=LOG_STEPS, num_epochs=10, 
        exp_name=exp_name, use_tqdm=True, feats=["inventor_list"], 
        device=DEVICE, input_size=10, hidden_sizes=layers, num_classes=2)

        exp_name = f"MLP_{len(layers)}HL_topic_id"
        os.makedirs(os.path.join("experiments", exp_name), exist_ok=True)
        evaluate_MLP(data, log_step=LOG_STEPS, num_epochs=10, 
        exp_name=exp_name, use_tqdm=True, feats=["topic_id"], 
        device=DEVICE, input_size=1, hidden_sizes=layers, num_classes=2)

        # ignoring inventors:
        exp_name = f"MLP_{len(layers)}HL_no_inventor"
        os.makedirs(os.path.join("experiments", exp_name), exist_ok=True)
        evaluate_MLP(data, log_step=LOG_STEPS, num_epochs=10, 
        exp_name=exp_name, use_tqdm=True, device=DEVICE, input_size=2,
        feats=["topic_id", "examiner_namer"], hidden_sizes=layers, num_classes=2)