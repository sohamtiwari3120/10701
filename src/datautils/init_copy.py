import os
import re
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

# class ClaimsFeatures(Dataset):
#     """Dataset to represent the list of claims."""
#     def __init__(self, df: pd.DataFrame, feats: List[str]):
#         self.data = df.reset_index(drop=True)
#         print(feats)
#         print(self.data)
#         self.feats = feats
#         if "inventor_list" in feats:
#             i = feats.index("inventor_list")
#             feats.pop(i)
#             for j in range(10): feats.append(f"inventor_{j}")
#         self.label_map = {"REJECTED": 0, "ACCEPTED": 1}
#         self.data["decision"] = self.data["decision"].apply(lambda x: self.label_map[x])
def load_keyphrases(folder: str) -> Dict[str, Tuple[str, float]]:
    keyphrases: dict = {}
    for file in tqdm(os.listdir(folder)):
        id = os.path.splitext(file.split("_")[-1])[0]
        path = os.path.join(folder, file)
        data = json.load(open(path))
        pharses = data["claims"][0]
        scores = data["claims"][1]
        list_of_phrases = []
        for word, score in zip(pharses, scores):
            list_of_phrases.append((word, score))
        keyphrases[id] = list_of_phrases

    return keyphrases

def add_keyphrase_loc_n_counts(claims, keyphrases: List[Tuple[str, float]],
                               tokenizer=None, filter_counts: int=0, 
                               filter_list: list=['claim','claims']) -> List[Tuple[str, float, int]]:
    """What this does:
    1. tokenize claims and keyphrases.
    2. add phrase count (no. of ocurrences) information to keyphrases.
    2. add start index information about keyphrases tokens.
    3. add end index information about keyphrases tokens.
    4. remove words with count less than `fitler_counts`
    5. remove words present in `filter_list`"""
    import re
    keyp_with_counts = []
    assert tokenizer is not None, "please pass a tokenizer"
    for word, score in keyphrases:
        # remove words in `filter_list`.
        if word in filter_list: continue
        phrase_tokens = tokenizer.tokenize(word)
        tokenizer.tokenize(claim)
        count = 0
        for claim in claims:
            matches = list(re.finditer(word, claim.lower()))
            if matches: count += len(matches)
        # remove words less frequent than `filter_counts`.
        if count > filter_counts:
            keyp_with_counts.append((word, score, count))

    return keyp_with_counts

class ClaimsFeatures(Dataset):
    def __init__(self) -> None:
        super(ClaimsFeatures, self).__init__()
        self.data = json.load(open())

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

# main function.
if __name__ == "__main__":
    data = read_patent_splits("../data/patent_acceptance_pred_subs=0.2.jsonl")
    all_claims = []
    for i in tqdm(range(len(data[0]))):
        all_claims.append(parse_claims(data[0]["claims"][i]))