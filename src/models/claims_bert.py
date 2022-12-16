import os
import json
import torch
import numpy as np
from typing import *
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import f1_score
from datautils import read_patent_jsonl
from datautils.split_claims import SplitClaimsDataset
from transformers import BertTokenizer, BertTokenizerFast, BertModel

# BERT based classifier that can scale to larger claims (claims are split into subclaims)
class ClaimsBERT(nn.Module):
    def __init__(self, bert_path, 
                 mlp_proj_size: int=500, 
                 dropout_rate: float=0.2):
        super(ClaimsBERT, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        if "base" in bert_path:
            emb_size = 768
        else: emb_size = 1024
        self.emb_size = emb_size
        self.mlp = nn.Sequential(
            nn.Linear(self.emb_size, mlp_proj_size),
            nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(mlp_proj_size, 1),
        )

def validate_ClaimsBERT(model, data_loader, loss_fn, **args) -> Dict[str, Union[float, List[int]]]:
    stats = {
        "acc": 0, "tot": 0,
        "preds": [], "losses": []
    }
    print("validating")
    trues = []
    model.eval()
    valbar = tqdm(enumerate(data_loader),
                  total=len(data_loader),
                  desc="validating model:",
                  disable=False)
                #   disable=not(args['use_tqdm']))
    for step, (batch, label) in valbar:
        # if len(batch['input_ids']) > args['batch_size']:
        #     print(f"skipping large batch of {len(batch)} items")
        #     continue
        # print(step)
        with torch.no_grad():
            for key in batch:
                batch[key] = batch[key][:args['batch_size']].to(args['device'])
            label = label.to(args['device'])
            patent_repr = model.bert(**batch).pooler_output.mean(axis=0)
            pool_type = args.get("pool_type")
            if pool_type == "rand":
                attn_mask = batch["attention_mask"]
                rand_bin_mask = torch.randint(0, 2, attn_mask.shape).to(args['device'])*attn_mask
                rand_bin_mask = rand_bin_mask.unsqueeze(dim=-1)
            if pool_type == "rand":
                patent_repr = ((rand_bin_mask * model.bert(**batch).last_hidden_state).mean(axis=1)).mean(axis=0)
            else: patent_repr = model.bert(**batch).pooler_output.mean(axis=0)
            logits = model.mlp(patent_repr)
            stats["tot"] += len(label)
            batch_loss = loss_fn(logits, label.float())
            # compute gradients and update parameters.
            stats["losses"].append(batch_loss.item())
            # compute train accuracy till now.
            preds = (logits>0)
            stats["acc"] += (label == preds).sum().item()
            stats["preds"] += preds.tolist()
            trues += label.cpu().tolist()
            valbar.set_description(f"bl: {batch_loss:.2f} l: {np.mean(stats['losses']):.2f} acc: {100*stats['acc']/stats['tot']:.2f}")
    stats['acc'] = stats['acc']/stats['tot']
    stats['f1_micro'] = f1_score(trues, stats['preds'], average="micro")
    stats['f1_macro'] = f1_score(trues, stats['preds'], average="macro")
    stats['f1_weighted'] = f1_score(trues, stats['preds'], average="weighted")
    stats["epoch"] = args['epoch']
    print("finished validation")

    return stats

def run_ClaimsBERT(data_dir: str="/home/arnaik/ml_project/data", 
                   bert_path: str="bert-base-cased", pool_type=None,
                   log_step: int=10, num_epochs: int=10, batch_size: int=64, 
                   lr: float=1e-5, use_tqdm: bool=False, exp_name: str="subclaims_bert", 
                   device: str="cuda:0", tok_args: dict={
                        "max_length": 150, "truncation": True, 
                        "return_tensors": "pt", "padding": "max_length",
                   }):
    data = read_patent_jsonl(os.path.join(data_dir, "patent_acceptance_pred_subs=0.2.jsonl"))
    classes = {rec['id']: rec['decision'] for rec in data}
    train_ids = json.load(open(os.path.join(data_dir, "train_ids.json")))
    val_ids = json.load(open(os.path.join(data_dir, "val_ids.json")))
    test_ids = json.load(open(os.path.join(data_dir, "test_ids.json")))
    split_claims = json.load(open(os.path.join(data_dir, "train_claims.json")))
    split_claims.update(json.load(open(os.path.join(data_dir, "val_claims.json"))))
    split_claims.update(json.load(open(os.path.join(data_dir, "test_claims.json"))))
    # create tokenizer.
    tokenizer = BertTokenizerFast.from_pretrained(bert_path)
    # create datasets.
    train_data = SplitClaimsDataset(ids=train_ids, split_claims=split_claims, 
                                    classes=classes, tokenizer=tokenizer, 
                                    tok_args=tok_args)
    val_data = SplitClaimsDataset(ids=val_ids, split_claims=split_claims, 
                                  classes=classes, tokenizer=tokenizer,
                                  tok_args=tok_args)
    test_data = SplitClaimsDataset(ids=test_ids, split_claims=split_claims, 
                                   classes=classes, tokenizer=tokenizer,
                                   tok_args=tok_args)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(0.554))
    model = ClaimsBERT(bert_path=bert_path)
    # move model to GPU/CPU.
    model.to(device)
    config_path = os.path.join("experiments", exp_name, "config.json")
    print(config_path)
    # exit()
    print(f"saving config to: {config_path}")
    config_ = {}
    config_.update(tok_args)
    config_["batch_size"] = batch_size
    config_["num_epochs"] = num_epochs
    config_["lr"] = lr
    config_["bert_path"] = bert_path
    with open(config_path, "w") as f:
        json.dump(config_, f)
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-12)
    
    best_F1_macro = 0
    for epoch in range(num_epochs):
        trainbar = tqdm(enumerate(train_data),
                        total=len(train_data),
                        desc="training model:",
                        disable=not(use_tqdm))
        max_batch_size = 0
        batch_losses = []
        matches, tot = 0, 0
        for step, (batch, label) in trainbar:
            # if len(batch['input_ids']) > batch_size: 
            #     for key in batch:
            #         batch[key] = batch[key][:batch_size]
                # print(f"skipping large batch of {len(batch['input_ids'])} items")
                # continue
            model.train()
            optimizer.zero_grad()
            for key in batch:
                batch[key] = batch[key][:batch_size].to(device)
            if pool_type == "rand":
                attn_mask = batch["attention_mask"]
                rand_bin_mask = torch.randint(0, 2, attn_mask.shape).to(device)*attn_mask
                rand_bin_mask = rand_bin_mask.unsqueeze(dim=-1)
                # print(len(batch[key]))
            label = label.to(device)
            if pool_type == "rand":
                patent_repr = ((rand_bin_mask * model.bert(**batch).last_hidden_state).mean(axis=1)).mean(axis=0)
            else: patent_repr = model.bert(**batch).pooler_output.mean(axis=0)
            logits = model.mlp(patent_repr)
            tot += len(label)
            batch_loss = loss_fn(logits, label.float())
            # compute gradients and update parameters.
            batch_loss.backward()
            batch_losses.append(batch_loss.item())
            optimizer.step()
            # compute train accuracy till now.
            preds = (logits>0)
            matches += (label == preds).sum().item()
            max_batch_size = max(max_batch_size, len(batch['input_ids']))
            trainbar.set_description(f"mb: {max_batch_size} bl: {batch_loss:.2f} l: {np.mean(batch_losses):.2f} acc: {100*matches/tot:.2f}")
            # validate the model at every 2000 steps.
            if (step+1)%log_step == 0:
                stats_path = os.path.join("experiments", exp_name, f"val_stats_epoch_{epoch}_step_{step}.json")
                stats = validate_ClaimsBERT(model, data_loader=val_data, 
                                      loss_fn=loss_fn, device=device, 
                                      use_tqdm=use_tqdm, epoch=epoch,
                                      batch_size=batch_size)
                with open(stats_path, "w") as f:
                    json.dump(stats, f)
                # save model if it has the best_F1_macro
                if stats['f1_macro'] > best_F1_macro:
                    best_F1_macro = stats['f1_macro']
                    print("\x1b[32;1msaving best model\x1b[0m")
                    torch.save(model.state_dict(), os.path.join("experiments", exp_name, "model.pt"))
                else: print("\x1b[31;1mignoring worse model\x1b[0m")

        stats_path = os.path.join("experiments", exp_name, f"val_stats_epoch_{epoch}_step_{step}.json")
        stats = validate_ClaimsBERT(model, data_loader=val_data, 
                                    loss_fn=loss_fn, device=device, 
                                    use_tqdm=use_tqdm, epoch=epoch,
                                    batch_size=batch_size)
        with open(stats_path, "w") as f:
            json.dump(stats, f)
        # save model if it has the best_F1_macro
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
    test_stats = validate_ClaimsBERT(model, data_loader=test_data, 
                              loss_fn=loss_fn, device=device, 
                              use_tqdm=use_tqdm, epoch=-1)
    print_test_metrics(test_stats)
    with open(test_stats_path, "w") as f:
        json.dump(test_stats, f)
    
    return test_stats