import torch
from typing import *
from torch.utils.data import Dataset

class SplitClaimsDataset(Dataset):
    def __init__(self, ids: List[str], split_claims: dict, 
                 tokenizer, classes: dict, tok_args: dict={
                    "max_length": 150,
                    "truncation": True,
                    "return_tensors": "pt",
                    "padding": "max_length",
                }):
        super(SplitClaimsDataset, self).__init__()
        self.ids = []
        self.data = []
        self.classes = []
        for id in ids:
            if classes[id] == "PENDING": continue
            self.ids.append(id)
            self.classes.append(classes[id])
            self.data.append(split_claims[id])
        self.label_map = {
            "ACCEPTED": 1,
            "REJECTED": 0,
        }
        # self.ids = [id for id in ids if classes[id] != "PENDING"]
        # self.data = [split_claims[id] for id in ids]
        # self.classes = [classes[id] for id in ids]
        self.tokenizer = tokenizer
        self.tok_args = tok_args

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        """create a batch"""
        sub_claims = self.data[index]
        inp_dict = self.tokenizer(sub_claims, **self.tok_args)
        label = torch.as_tensor([self.label_map[self.classes[index]]])

        return inp_dict, label

    def __len__(self):
        return len(self.ids)
