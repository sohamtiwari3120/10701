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
                 num_classes: int=2, dropout_rate: float=0.2) -> None:
        super(FeatMLP, self).__init__()
        hidden_sizes = [input_size]+hidden_sizes
        for i in range(len(hidden_sizes)-1):
            setattr(self, f"layer{i}", nn.Linear(
                hidden_sizes[i],
                hidden_sizes[i+1],
            ))
            setattr(self, f"activation{i}", nn.GELU())
            setattr(self, f"dropout{i}", nn.Dropout(dropout_rate))
        self.N = len(hidden_sizes)
        if num_classes == 2:
            setattr(self, f"layer{self.N-1}", nn.Linear(hidden_sizes[-1], 1))
            # setattr(self, f"activation{self.N-1}", nn.Sigmoid())
        else:
            setattr(self, f"layer{self.N-1}", nn.Linear(
                hidden_sizes[-1],
                num_classes,
            ))
            # setattr(self, f"activation{self.N-1}", nn.Softmax())

    def forward(self, x):
        for i in range(self.N-1):
            x = getattr(self, f"dropout{i}")(
                getattr(self, f"activation{i}")(
                    getattr(self, f"layer{i}")(x)
                ))
        x = getattr(self, f"layer{self.N-1}")(x)

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

def featurize_data(train: pd.DataFrame, test: pd.DataFrame, 
                   val: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    featurizer_map = {
        "examiner_namer": ExaminerFeaturizer,
        "topic_id": TopicFeaturizer,
    }
    for feat_name in features:
        if feat_name not in featurizer_map: continue
        f = featurizer_map[feat_name](train[feat_name])
        train[feat_name] = train[feat_name].apply(lambda x: f(x))
        test[feat_name] = test[feat_name].apply(lambda x: f(x))
        val[feat_name] = val[feat_name].apply(lambda x: f(x))
    if "inventor_list" in features:
        # inventor featurization done manually.
        inventor_ids = set()
        for inv_list in tqdm(train["inventor_list"]):
            for inv in inv_list:
                inventor_ids.add(get_inventor(inv))
        inventor_map = {k: i+1 for i,k in enumerate(sorted(inventor_ids))}
        for j in range(10):
            train[f"inventor_{j}"] = train["inventor_list"].apply(
                lambda x: inventor_map.get(get_inventor(
                lget(x, j)), 0))
            test[f"inventor_{j}"] = test["inventor_list"].apply(
                lambda x: inventor_map.get(get_inventor(
                lget(x, j)), 0))
            val[f"inventor_{j}"] = val["inventor_list"].apply(
                lambda x: inventor_map.get(get_inventor(
                lget(x, j)), 0))
        del train["inventor_list"]
        del test["inventor_list"]
        del val["inventor_list"]
    # reset indices of dataframes.
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    val = val.reset_index(drop=True)

    return train, test, val

def validate_MLP(model, data_loader, loss_fn, **args) -> Dict[str, Union[float, List[int]]]:
    stats = {
        "acc": 0, "tot": 0,
        "preds": [], "losses": []
    }
    trues = []
    model.eval()
    valbar = tqdm(enumerate(data_loader),
                  total=len(data_loader),
                  desc="validating model:",
                  disable=not(args['use_tqdm']))
    for step, batch in valbar:
        with torch.no_grad():
            for i in range(len(batch)):
                batch[i] = batch[i].to(args['device'])
            logits = model(batch[0]).squeeze()
            stats["tot"] += len(batch[0])
            batch_loss = loss_fn(logits, batch[1].float())
            # compute gradients and update parameters.
            stats["losses"].append(batch_loss.item())
            # compute train accuracy till now.
            preds = (logits>0)
            stats["acc"] += (batch[1] == preds).sum().item()
            stats["preds"] += preds.tolist()
            trues += batch[1].cpu().tolist()
            valbar.set_description(f"bl: {batch_loss:.2f} l: {np.mean(stats['losses']):.2f} acc: {100*stats['acc']/stats['tot']:.2f}")
    stats['acc'] = stats['acc']/stats['tot']
    stats['f1_micro'] = f1_score(trues, stats['preds'], average="micro")
    stats['f1_macro'] = f1_score(trues, stats['preds'], average="macro")
    stats['f1_weighted'] = f1_score(trues, stats['preds'], average="weighted")
    stats["epoch"] = args['epoch']

    return stats

def evaluate_MLP(data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], 
                 log_step: int=2000, num_epochs: int=10, batch_size: int=1024, 
                 lr: float=1e-3, use_tqdm: bool=False, feats: List[str]=["examiner_namer", 
                 "inventor_list", "topic_id"], exp_name: str="MLP_1L", device: str="cuda:0", 
                 **mlp_args):
    train, val, test = data
    # create dataloaders.
    train_loader = DataLoader(CategFeatures(train, feats=feats), 
                              shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(CategFeatures(test, feats=feats), 
                             shuffle=False, batch_size=batch_size)
    val_loader = DataLoader(CategFeatures(val, feats=feats), 
                            shuffle=False, batch_size=batch_size)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(0.554))
    model = FeatMLP(**mlp_args)
    print(model)
    # move model to GPU/CPU.
    model.to(device)
    config_path = os.path.join("experiments", exp_name, "config.json")
    print(f"saving config to: {config_path}")
    mlp_args["batch_size"] = batch_size
    mlp_args["num_epochs"] = num_epochs
    mlp_args["lr"] = lr
    mlp_args["feats"] = feats
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
        matches, tot = 0, 0
        for step, batch in trainbar:
            model.train()
            optimizer.zero_grad()
            for i in range(len(batch)):
                batch[i] = batch[i].to(device)
            logits = model(batch[0]).squeeze()
            tot += len(batch[0])
            batch_loss = loss_fn(logits, batch[1].float())
            # compute gradients and update parameters.
            batch_loss.backward()
            batch_losses.append(batch_loss.item())
            optimizer.step()
            # compute train accuracy till now.
            preds = (logits>0)
            matches += (batch[1] == preds).sum().item()
            trainbar.set_description(f"bl: {batch_loss:.2f} l: {np.mean(batch_losses):.2f} acc: {100*matches/tot:.2f}")
            # validate the model at every 2000 steps.
            if (step+1)%log_step == 0:
                stats_path = os.path.join("experiments", exp_name, f"val_stats_epoch_{epoch}_step_{step}.json")
                stats = validate_MLP(model, data_loader=val_loader, 
                                     loss_fn=loss_fn, device=device, 
                                     use_tqdm=use_tqdm, epoch=epoch)
                with open(stats_path, "w") as f:
                    json.dump(stats, f)
                # save model if it has the best_F1_macro
                if stats['f1_macro'] > best_F1_macro:
                    best_F1_macro = stats['f1_macro']
                    print("\x1b[32;1msaving best model\x1b[0m")
                    torch.save(model.state_dict(), os.path.join("experiments", exp_name, "model.pt"))
                else: print("\x1b[31;1mignoring worse model\x1b[0m")

        stats_path = os.path.join("experiments", exp_name, f"val_stats_epoch_{epoch}_step_{step}.json")
        stats = validate_MLP(model, data_loader=val_loader, 
                                loss_fn=loss_fn, device=device, 
                                use_tqdm=use_tqdm, epoch=epoch)
        with open(stats_path, "w") as f:
            json.dump(stats, f)
        # save model if it has the best_F!_macro
        if stats['f1_macro'] > best_F1_macro:
            best_F1_macro = stats['f1_macro']
            print(f"\x1b[32;1msaving best model with F1-macro: {stats['f1_macro']}\x1b[0m")
            torch.save(model.state_dict(), os.path.join("experiments", exp_name, "model.pt"))
        else: print(f"\x1b[31;1mignoring worse model with F1-macro: {stats['f1_macro']}\x1b[0m")
    
    # finally test the best validation model on the test data.
    def print_test_metrics(test_stats: Dict[str, Union[float, List[int]]]):
        print("\n\x1b[34;1mtest metrics:\x1b[0m")
        for metric, value in test_stats.items():
            if isinstance(value, float):
                print(f"{metric}: {100*value:.2f}")
    test_stats_path = os.path.join("experiments", exp_name, f"test_stats.json")
    test_stats = validate_MLP(model, data_loader=test_loader, 
                              loss_fn=loss_fn, device=device, 
                              use_tqdm=use_tqdm, epoch=-1)
    print_test_metrics(test_stats)
    with open(test_stats_path, "w") as f:
        json.dump(test_stats, f)
    
    return test_stats

def evaluate_NB(data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], 
                algo: str="gaussian", feats: List[str]=["examiner_namer", 
                "inventor_list", "topic_id"]):
    train, val, test = data
    label_map = {"REJECTED": 0, "ACCEPTED": 1}
    class_map = {
        "gaussian": GaussianNB, 
        "multinom": MultinomialNB, 
        "complement": ComplementNB,
        "bernoulli": BernoulliNB,
        "categorical": CategoricalNB,
    }
    assert algo in ["gaussian", "multinom", "complement", "bernoulli", "categorical"], NotImplementedError(f"{algo} doesn't have a Naive Bayes implementation in sklearn")
    model = class_map[algo]()
    train = train[feats+['decision']]
    test = test[feats+['decision']]
    val = val[feats+['decision']]
    train, test, val = featurize_data(train, test, val, features=feats)
    X_train = pd.concat([train, val])
    X_test = test
    y_train = (pd.concat([train['decision'], val['decision']])).apply(lambda x: label_map[x]).reset_index(drop=True)
    y_test = (test['decision']).apply(lambda x: label_map[x]).reset_index(drop=True)
    # remove decision (target variable) from the input.
    del X_train["decision"]
    del X_test["decision"]
    pred = model.fit(X_train, y_train).predict(X_test)
    acc = (y_test == pred).sum()/len(y_test)
    f1_micro = f1_score(y_test, pred, average="micro")
    f1_macro = f1_score(y_test, pred, average="macro")
    f1_weighted = f1_score(y_test, pred, average="weighted")
    print(f"accuracy: {acc}, F1: micro: {f1_micro} macro: {f1_macro} weighted: {f1_weighted}")
    print(f"No. mislabeled points: {(y_test != pred).sum()} out of {len(y_test)}")

    return {"acc": acc, "f1_micro": f1_micro, "f1_macro": f1_macro, "f1_weighted": f1_weighted}

if __name__ == "__main__":
    pass