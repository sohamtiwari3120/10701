import os
import torch
import random
import numpy as np
from datautils import read_patent_splits
from models.claims_bert import ClaimsBERT, run_ClaimsBERT

# seed torch, random and numpy.
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

if __name__ == "__main__":
    LOG_STEPS = 2000
    DEVICE = "cuda:0"
    MODEL_ID = 1
    POOL_TYPE = [None, "rand"][0]
    BERT_PATH = ["bert-base-uncased", "anferico/bert-for-patents"][MODEL_ID] # "bert-base-cased"
    BATCH_SIZE = [32, 32][MODEL_ID]
    exp_name = f"BERT_{BERT_PATH.replace('/','_')}_{POOL_TYPE}"
    # tok_args = {
    #     "padding": "max_length",
    #     "max_length": 200,
    #     "truncation": True,
    #     "return_tensors": "pt"
    # }
    os.makedirs(os.path.join("experiments", exp_name), exist_ok=True)
    run_ClaimsBERT(data_dir="/home/arnaik/ml_project/data", bert_path=BERT_PATH, 
                   log_step=LOG_STEPS, num_epochs=10, exp_name=exp_name, 
                   use_tqdm=True, batch_size=BATCH_SIZE, device=DEVICE,
                   pool_type=POOL_TYPE)