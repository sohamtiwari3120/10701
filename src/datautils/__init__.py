import pickle
import os
import re
import json
import torch
import pandas as pd
from typing import *
from tqdm import tqdm
from datetime import datetime as dt
from torch.utils.data import Dataset
from glob import glob
import numpy as np

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

def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data
    
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

def parse_claims(text: str):
    words = text.split()
    regex = re.compile("^[0-9]+-[0-9]+\.$")
    regex2 = re.compile("^[0-9]+-[0-9]+\)$")
    regex3 = re.compile("^[0-9]+\.-[0-9]+\.$")
    regex4 = re.compile("^[0-9]+-[0-9]+\.\)$")
    claims, ctr, curr_claim = [], 1, []
    for word in words:
        word = word.strip()
        if word == f"{ctr}." or word == f"{ctr})" or word == f"{ctr}.)": # print("found point")
            if len(curr_claim) != 0: 
                claims.append(" ".join(curr_claim))
            curr_claim = []
            ctr += 1
        elif word.startswith(f"{ctr}-") and regex.match(word): # print("found range")
            if len(curr_claim) != 0: 
                claims.append(" ".join(curr_claim))
            curr_claim = []
            try: ctr = int(word.split("-")[-1].split(".")[0])+1
            except ValueError: print(word)
        elif word.startswith(f"{ctr}-") and regex4.match(word): # print("found range")
            if len(curr_claim) != 0: 
                claims.append(" ".join(curr_claim))
            curr_claim = []
            try: ctr = int(word.split("-")[-1].split(".)")[0])+1
            except ValueError: print(word)
        elif word.startswith(f"{ctr}-") and regex2.match(word): # print("found range")
            if len(curr_claim) != 0: 
                claims.append(" ".join(curr_claim))
            curr_claim = []
            try: ctr = int(word.split("-")[-1].split(")")[0])+1
            except ValueError: print(word)
        elif word.startswith(f"{ctr}.-") and regex3.match(word): # print("found range2")
            if len(curr_claim) != 0: 
                claims.append(" ".join(curr_claim))
            curr_claim = []
            try: ctr = int(word.split("-")[-1].split(".")[0])+1
            except ValueError: print(word)
        else: curr_claim.append(word)
    if len(curr_claim) > 0:
        claims.append(" ".join(curr_claim))

    return claims

class ClaimsFeatures(Dataset):
    """Dataset to represent the list of claims."""
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

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
 
    if (a_set & b_set):
        return a_set & b_set

    else:
        print("No common elements")
        return {}

class ClaimsMaskedDataset(Dataset):
    """Custom PyTorch Dataset class. Maps the claims of each patent application to the corresponding extracted keyphrases and masks the the top n% most relevant keyphrases from the claims and return them.
    """
    def __init__(self, keyphrases_dir: str, claims_path: str, mask_ratio: float = 0.2) -> None:
        """Accepts as input the path to the directories containing the keyphrases and the path to claims json file.

        Args:
            keyphrases_dir (str): path to the directories containing the keyphrases
            claims_dir (str): path to claims json file
            mask_ratio (float, optional): Ratio of generated keyphrases to be masked in input. Defaults to 0.2.
        """        
        super().__init__()
        self.keyphrases_dir = keyphrases_dir
        self.claims_path = claims_path
        self.claims_dict = json.load(open(self.claims_path))
        self.patent_id_keyphrase_path = {}
        for keyphrase_filepath in glob(os.path.join(self.keyphrases_dir, "*.jsonl")):
            _, patent_id = os.path.basename(keyphrase_filepath)[:-6].split('_')
            self.patent_id_keyphrase_path[patent_id] = keyphrase_filepath
        self.patent_ids = list(self.claims_dict.keys())


    def __len__(self):
        return len(self.patent_ids)

    def __getitem__(self, index):
        patent_id = self.patent_ids[index]
        data = {"claims": self.claims_dict[patent_id], "original_claims_text":self.claims_dict[patent_id]}
        breakpoint()
        if patent_id not in self.patent_id_keyphrase_path:
            return data
        keyphrases = json.load(open(self.patent_id_keyphrase_path[patent_id]))
        claims_keyphrases, claims_relevance = keyphrases['claims'][0], keyphrases['claims'][1]
        if len(claims_keyphrases) == 0:
            return data
        sorted_indices = np.argsort(np.array(claims_relevance))[::-1]
        claims_relevance = claims_relevance[sorted_indices]
        claims_keyphrases = claims_keyphrases[sorted_indices]
        breakpoint()
        return ""

# main function.
if __name__ == "__main__":
    # data = read_patent_splits("../data/patent_acceptance_pred_subs=0.2.jsonl")
    # all_claims = []
    # for i in tqdm(range(len(data[0]))):
    #     all_claims.append(parse_claims(data[0]["claims"][i]))
    ds = ClaimsMaskedDataset("/home/sohamdit/10701-files/10701/data/keyphrases/", "/home/arnaik/ml_project/data/train_claims.json")
    print(ds.__getitem__(1))