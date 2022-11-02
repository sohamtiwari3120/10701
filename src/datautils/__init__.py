import os
import json
import pandas as pd
from typing import *
from tqdm import tqdm
from datetime import datetime as dt

def read_patent_jsonl(path: str, use_tqdm: bool=False) -> List[dict]:
    # list of dicts data to be loaded
    data: List[dict] = []
    with open(path) as f:
        for line in tqdm(f, disable=not(use_tqdm)):
            data.append(json.loads(line.strip()))
    
    return data

def read_patent_splits(path, use_tqdm: bool=False, train_size: float=0.7, 
                       val_size: float=0.1) -> Tuple[pd.DataFrame, 
                       pd.DataFrame, pd.DataFrame]:
    """read train, test, val splits from data, separated based on 
    time.
    """
    df = pd.DataFrame(read_patent_jsonl(path, use_tqdm=use_tqdm))
    # convert filing date to datetime.
    df["filing_date"] = df["filing_date"].apply(lambda x: dt.strptime(x, "%Y%m%d"))
    # sort by filing dates.
    time_sorted = df.sort_values(by=["filing_date"]).reset_index(drop=True)
    # partition/split data into train/val/test by filing dates.
    N = len(time_sorted)
    N_train = int(N*train_size)
    N_val = int(N*val_size)
    # split into train, test and val.
    train = time_sorted[: N_train].reset_index(drop=True)
    val = time_sorted[N_train : N_train + N_val].reset_index(drop=True)
    test = time_sorted[N_train + N_val :].reset_index(drop=True)

    return train, val, test