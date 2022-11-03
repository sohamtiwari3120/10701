import json
from tqdm import tqdm
from typing import List
    
def read_jsonl(path: str, use_tqdm: bool=True):
    # list of dicts data to be loaded
    data: List[dict] = []
    with open(path) as f:
        for line in tqdm(f, disable=not(use_tqdm)):
            data.append(json.loads(line.strip()))
    return data