import os
import torch
import random
import numpy as np
from datautils import read_patent_splits
from models.baselines.text_only import evaluate_BERT

# seed torch, random and numpy.
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

if __name__ == "__main__":
    data = read_patent_splits(
        "../data/patent_acceptance_pred_subs=0.2.jsonl", 
        use_tqdm=True
    )
    LOG_STEPS = 500
    DEVICE = "cuda:0"
    BERT_PATH = "anferico/bert-for-patents" # "bert-base-cased"
    SECTIONS = ["claims"]
    exp_name = f"BERT_{BERT_PATH}_{'_'.join(SECTIONS)}"
    tok_args = {
        "padding": "max_length",
        "max_length": 200,
        "truncation": True,
        "return_tensors": "pt"
    }
    os.makedirs(os.path.join("experiments", exp_name), exist_ok=True)
    evaluate_BERT(data, bert_path=BERT_PATH, sections=SECTIONS, log_step=LOG_STEPS, emb_size=1024,
    num_epochs=10, exp_name=exp_name, use_tqdm=True, batch_size=32, device=DEVICE, **tok_args)