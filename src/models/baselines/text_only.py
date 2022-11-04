import os
import json
import torch
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
from datautils import TextFeatures
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import BertModel, AutoModel, BertTokenizer, AutoTokenizer

# bert text classifier
class BertBinaryTextClassifier(nn.Module):
    def __init__(self, bert_path: str="bert-base-cased", 
                 sections: List[str]=["claims"], emb_size: int=768,
                 mlp_proj_size: int=500, dropout_rate: float=0.2):
        super(BertBinaryTextClassifier, self).__init__()
        self.emb_size = emb_size
        self.sections = sections
        self.dropout_rate = dropout_rate
        if bert_path.startswith("bert"):
            self.bert = BertModel.from_pretrained(bert_path)
            self.input_keys = ["input_ids", "token_type_ids", "attention_mask"]
        else: 
            self.bert = AutoModel.from_pretrained(bert_path)
            self.input_keys = ["input_ids", "attention_mask"]
        self.mlp = nn.Sequential(
            nn.Linear(len(sections)*self.emb_size, mlp_proj_size),
            nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(mlp_proj_size, 1),
        )

    def forward(self, **args):
        section_embs = []
        for section in self.sections:
            input = {key: args.get(section+"_"+key) for key in self.input_keys}
            section_embs.append(self.bert(**input).pooler_output)
        section_embs = torch.cat(section_embs, dim=-1)
        
        return self.mlp(section_embs)

def validate_BERT(model, data_loader, loss_fn, **args) -> Dict[str, Union[float, List[int]]]:
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
            for key in batch:
                batch[key] = batch[key].to(args['device'])
            logits = model(**batch).squeeze()
            stats["tot"] += len(batch["label"])
            batch_loss = loss_fn(logits, batch["label"].float())
            # compute gradients and update parameters.
            stats["losses"].append(batch_loss.item())
            # compute train accuracy till now.
            preds = (logits>0)
            stats["acc"] += (batch["label"] == preds).sum().item()
            stats["preds"] += preds.tolist()
            trues += batch["label"].cpu().tolist()
            valbar.set_description(f"bl: {batch_loss:.2f} l: {np.mean(stats['losses']):.2f} acc: {100*stats['acc']/stats['tot']:.2f}")
    stats['acc'] = stats['acc']/stats['tot']
    stats['f1_micro'] = f1_score(trues, stats['preds'], average="micro")
    stats['f1_macro'] = f1_score(trues, stats['preds'], average="macro")
    stats['f1_weighted'] = f1_score(trues, stats['preds'], average="weighted")
    stats["epoch"] = args['epoch']

    return stats

def evaluate_BERT(data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], 
                  bert_path: str="bert-base-cased", sections: list=["claims"],
                  log_step: int=10, num_epochs: int=10, batch_size: int=32, 
                  lr: float=1e-5, use_tqdm: bool=False, exp_name: str="MLP_1L", 
                  device: str="cuda:0", emb_size: int=768, **tok_args):
    train, val, test = data
    # create dataloaders.
    if bert_path.startswith("bert"):
        tokenizer = BertTokenizer.from_pretrained(bert_path)
    else: tokenizer = AutoTokenizer.from_pretrained(bert_path)
    train_loader = DataLoader(TextFeatures(train, sections=sections, tokenizer=tokenizer, **tok_args), 
                              shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(TextFeatures(test, sections=sections, tokenizer=tokenizer, **tok_args), 
                            shuffle=False, batch_size=batch_size)
    val_loader = DataLoader(TextFeatures(val, sections=sections, tokenizer=tokenizer, **tok_args), 
                            shuffle=False, batch_size=batch_size)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(0.554))
    model = BertBinaryTextClassifier(bert_path=bert_path, emb_size=emb_size, sections=sections)
    # move model to GPU/CPU.
    model.to(device)
    config_path = os.path.join("experiments", exp_name, "config.json")
    print(f"saving config to: {config_path}")
    tok_args["batch_size"] = batch_size
    tok_args["num_epochs"] = num_epochs
    tok_args["lr"] = lr
    tok_args["bert_path"] = bert_path
    tok_args["sections"] = sections
    with open(config_path, "w") as f:
        json.dump(tok_args, f)
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
            for key in batch:
                batch[key] = batch[key].to(device)
            logits = model(**batch).squeeze()
            tot += len(batch["label"])
            batch_loss = loss_fn(logits, batch["label"].float())
            # compute gradients and update parameters.
            batch_loss.backward()
            batch_losses.append(batch_loss.item())
            optimizer.step()
            # compute train accuracy till now.
            preds = (logits>0)
            matches += (batch["label"] == preds).sum().item()
            trainbar.set_description(f"bl: {batch_loss:.2f} l: {np.mean(batch_losses):.2f} acc: {100*matches/tot:.2f}")
            # validate the model at every 2000 steps.
            if (step+1)%log_step == 0:
                stats_path = os.path.join("experiments", exp_name, f"val_stats_epoch_{epoch}_step_{step}.json")
                stats = validate_BERT(model, data_loader=val_loader, 
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
        stats = validate_BERT(model, data_loader=val_loader, 
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
    test_stats = validate_BERT(model, data_loader=test_loader, 
                              loss_fn=loss_fn, device=device, 
                              use_tqdm=use_tqdm, epoch=-1)
    print_test_metrics(test_stats)
    with open(test_stats_path, "w") as f:
        json.dump(test_stats, f)
    
    return test_stats

if __name__ == "__main__":
    pass