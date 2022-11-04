import os
import json
import torch
import pandas as pd
from typing import *
from tqdm import tqdm
from datetime import datetime as dt
from torch.utils.data import Dataset

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
    df = df[df["decision"] != "PENDING"].reset_index(drop=True)
    print(f"{len(df)} instances after dropping PENDING patent decisions")
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

class TextFeatures(Dataset):
    """Dataset class to handle text only inputs. We try using varioius parts of the patent like the 
    claim, title, summary, abstract, background"""
    def __init__(self, df: pd.DataFrame, sections: List[str], tokenizer=None, **tok_args): 
        self.data = df.reset_index(drop=True)
        self.sections = sections
        assert tokenizer is not None, "You need to pass a tokenizer"
        self.tok_args = tok_args
        self.tokenizer = tokenizer
        # process the title data (from all CAPS to title case).
        self.data["title"] = self.data["title"].apply(lambda x: x.title())
        self.label_map = {"REJECTED": 0, "ACCEPTED": 1}
        self.data["decision"] = self.data["decision"].apply(lambda x: float(self.label_map[x]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int) -> dict:
        feats = {"label": torch.as_tensor(self.data["decision"][i])}
        for section in self.sections:
            tok_dict = self.tokenizer(self.data[section][i], **self.tok_args)
            for key, value in tok_dict.items():
                feats[section+"_"+key] = value[0]
        
        return feats

class CategFeatures(Dataset):
    """Dataset class to handle no-text or feature only data
    with categorical features like inventor_list, examiner_name, topic_id etc."""
    def __init__(self, df: pd.DataFrame, feats: List[str]):
        self.data = df.reset_index(drop=True)
        print(feats)
        print(self.data)
        self.feats = feats
        if "inventor_list" in feats:
            i = feats.index("inventor_list")
            feats.pop(i)
            for j in range(10): feats.append(f"inventor_{j}")
        self.label_map = {"REJECTED": 0, "ACCEPTED": 1}
        self.data["decision"] = self.data["decision"].apply(lambda x: self.label_map[x])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        feats = []
        for col in self.feats:
            feats.append(float(self.data[col][i]))

        return [torch.as_tensor(feats), torch.as_tensor(self.data["decision"][i])]