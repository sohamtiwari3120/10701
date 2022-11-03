import os
import json
import torch
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
from datautils import CategFeatures
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB

class ExaminerFeaturizer:
    def __init__(self, feats):
        self.feat_ids = {k: i for i,k in enumerate(
            np.sort(feats.apply(
            lambda x: " ".join(x.split())
        ).unique()))}
        self.N = len(self.feat_ids)

    def __call__(self, feat):
        feat = " ".join(feat.split())
        return self.feat_ids.get(feat, self.N)

class TopicFeaturizer:
    def __init__(self, feats):
        self.feat_ids = {k: i for i,k in enumerate(
            feats.apply(lambda x: x[:4]).unique()
        )}
        self.N = len(self.feat_ids)

    def __call__(self, feat):
        feat = feat[:4]
        return self.feat_ids.get(feat, self.N)

class FeatMLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int]=[], 
                 num_classes: int=3, dropout_rate: float=0.2) -> None:
        super(FeatMLP, self).__init__()
        hidden_sizes = [input_size]+hidden_sizes
        self.layers = []
        self.dropouts = []
        self.activations = []
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(
                hidden_sizes[i],
                hidden_sizes[i+1],
            ))
            self.activations.append(
                nn.GeLU()
            )
            self.dropouts.append(nn.Dropout(dropout_rate))
        self.layers.append(nn.Linear(
            hidden_sizes[-1],
            num_classes,
        ))
        # self.dropouts.append(nn.Dropout(dropout_rate))
        self.activations.append(nn.Sigmoid())

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.dropouts[i](self.activations[i](self.layers[i]))
        x = self.activations[-1](self.layers[-1])

        return x

def lget(l: list, i: int):
    try: return l[i]
    except IndexError: return None

def get_inventor(inv: Union[dict, None]) -> str:
    if inv is None: return inv
    first_name = inv["inventor_name_first"].strip()
    last_name = inv["inventor_name_last"].strip()
    location = f"{inv['inventor_city']}, {inv['inventor_state']}, {inv['inventor_country']}"

    return f"name={first_name} {last_name} & loc={location}"

def featurize_data(data: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    featurizer_map = {
        "examiner_namer": ExaminerFeaturizer,
        "topic_id": TopicFeaturizer,
    }
    for feat_name in features:
        if feat_name not in featurizer_map: continue
        f = featurizer_map[feat_name](data[feat_name])
        data[feat_name] = data[feat_name].apply(lambda x: f(x))
    if "inventor_list" in features:
        # inventor featurization done manually.
        inventor_ids = set()
        for inv_list in tqdm(data["inventor_list"]):
            for inv in inv_list:
                inventor_ids.add(get_inventor(inv))
        inventor_map = {k: i+1 for i,k in enumerate(sorted(inventor_ids))}
        for j in range(10):
            data[f"inventor_{j}"] = data["inventor_list"].apply(lambda x: inventor_map.get(
                get_inventor(
                    lget(x, j)
                ), 0
            ))
        del data["inventor_list"]

    return data.reset_index(drop=True)

def evaluate_NB(data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], log_step: int=2000,
                num_epochs: int=10, batch_size: int=128, lr: float=1e-3, use_tqdm: bool=False,
                feats: List[str]=["examiner_namer", "inventor_list", "topic_id"], 
                exp_name: str="MLP_1L", device: str="cuda:0", **mlp_args):
    train, val, test = data
    label_map = {"REJECTED": 0, "ACCEPTED": 1, "PENDING": 2}
    train_loader = DataLoader(CategFeatures(train), shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(CategFeatures(test), shuffle=False, batch_size=batch_size)
    val_loader = DataLoader(CategFeatures(val), shuffle=False, batch_size=batch_size)
    
    loss_fn = nn.CrossEntropyLoss()
    model = FeatMLP(**mlp_args)
    # move model to GPU/CPU.
    model.to(device)
    config_path = os.path.join("experiments", exp_name)
    with open(config_path, "w") as f:
        json.dump(mlp_args, f)
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-12)
    
    best_F1_macro = 0
    for epoch in range(num_epochs):
        trainbar = tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        desc="training model:",
                        disable=not(use_tqdm))
        batch_losses = []
        for step, batch in trainbar:
            model.train()
            optimizer.zero_grad()
            logits = model(batch[0])
            batch_loss = loss_fn(logits, batch[1])
            # compute gradients and update parameters.
            batch_loss.backward()
            batch_losses.append()
            optimizer.step()
            # validate the model at every 2000 steps.
            if (step+1)%log_step == 0:

    # finally test the best validation model on the test data.

    X_train = featurize_data(pd.concat([train[feats], val[feats]]), feats)
    X_test = featurize_data(test[feats], feats)
    y_train = (pd.concat([train['decision'], val['decision']])).apply(lambda x: label_map[x]).reset_index(drop=True)
    y_test = (test['decision']).apply(lambda x: label_map[x]).reset_index(drop=True)
    pred = model.fit(X_train, y_train).predict(X_test)
    acc = (y_test == pred).sum()/len(y_test)
    f1_micro = f1_score(y_test, pred, average="micro")
    f1_macro = f1_score(y_test, pred, average="macro")
    f1_weighted = f1_score(y_test, pred, average="weighted")
    print(f"accuracy: {acc}, F1: micro: {f1_micro} macro: {f1_macro} weighted: {f1_weighted}")
    print(f"No. mislabeled points: {(y_test != pred).sum()} out of {len(y_test)}")

    return {
        "acc": acc, 
        "f1_micro": f1_micro, 
        "f1_macro": f1_macro, 
        "f1_weighted": f1_weighted
    }

def evaulate_MLP(data, num_layers, use_tqdm: bool=False):


if __name__ == "__main__":
    pass