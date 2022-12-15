import argparse
from datautils import load_pickle
from functools import partial
from config import hparams
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import numpy as np
import wandb
import os
import glob
import torch

from datasets import Dataset, load_metric

hp = hparams()

metric_acc = load_metric('accuracy')
metric_f1 = load_metric('f1')
metric_prec = load_metric('precision')
metric_rec = load_metric('recall')

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    # Remove ignored index (special tokens) and convert to labels
    clf_report = classification_report(labels, predictions)
    print(clf_report)
    acc = metric_acc.compute(predictions=predictions, references=labels)
    f1 = metric_f1.compute(predictions=predictions, references=labels)
    prec = metric_prec.compute(predictions=predictions, references=labels)
    rec = metric_rec.compute(predictions=predictions, references=labels)
    # flattened_results = {
    #     # "overall_precision": results["overall_precision"],
    #     # "overall_recall": results["overall_recall"],
    #     "overall_f1": results["overall_f1"],
    #     "overall_accuracy": results["overall_accuracy"],
    # }
    # for k in results.keys():
    #   if(k not in flattened_results.keys()):
    #     flattened_results[k+"_f1"]=results[k]["f1"]

    return {'f1': f1['f1'], 'acc': acc['accuracy'], 'prec': prec['precision'], 'rec':rec['recall']}

def load_into_Dataset(data, topic_id):
    data_hf = Dataset.from_dict(data[topic_id]).rename_column('X_text', 'text').rename_column('y', 'label').remove_columns(['y_labels'])
    return data_hf

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def preprocess_function(examples, tokenizer):
    examples['test'] = 'test'
    return tokenizer(examples["text"], truncation=True)

class ClaimsTokenizedDataset(Dataset):
    """Tokenized claims (from patents) dataset with precomputed attention masks"""

    def __init__(self, tokenized_dir):
        """
        Args:
            tokenized_dir (string): Path to the directory containing precomputed tokenized patents along with corresponding attention masks
        """
        self.tokenized_dir = tokenized_dir
        self.pt_files = glob.glob(os.path.join(self.tokenized_dir, "*.pt"))

    def __len__(self):
        return len(self.pt_files) # number of patent applications

    def __getitem__(self, idx):
        # idx corresponds to patent id
        data = torch.load(self.pt_files[idx])
        claims_tokenized_list = data['claims_tokenized_list']
        attention_masks_list = data['attention_masks_list']
        return sample


def main(args):
    run_name = args.run_name
    train_data = load_pickle(hp.train_pkl)
    val_data = load_pickle(hp.val_pkl)
    # test_data = load_pickle(hp.test_pkl)

    val_topic_ids = list(val_data.keys())
    train_topic_ids = list(train_data.keys())
    topic_ids = intersection(train_topic_ids, val_topic_ids)

    tokenizer = AutoTokenizer.from_pretrained(hp.model_name)
    preprocess_function_partial = partial(preprocess_function, tokenizer=tokenizer)

    train_data_hf = load_into_Dataset(train_data, topic_ids[args.topic_id]).train_test_split(test_size=0.25, shuffle=False)
    # val_data_hf = load_into_Dataset(val_data, topic_ids[0])
    breakpoint()
    val_data_hf = train_data_hf['test']
    train_data_hf = train_data_hf['train']
    
    train_data_hf_tokenized = train_data_hf.map(preprocess_function_partial, batched=True)
    val_data_hf_tokenized = val_data_hf.map(preprocess_function_partial, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(hp.model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=hp.output_dir,
        evaluation_strategy=hp.evaluation_strategy,
        save_strategy=hp.save_strategy,
        learning_rate=hp.learning_rate,
        num_train_epochs=hp.num_train_epochs,
        weight_decay=hp.weight_decay,
        # push_to_hub=hp.push_to_hub,
        report_to=hp.report_to,
        run_name=run_name,
        load_best_model_at_end=hp.load_best_model_at_end,
        metric_for_best_model=hp.metric_for_best_model,
        greater_is_better=hp.greater_is_better,
        per_device_train_batch_size=hp.per_device_train_batch_size
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data_hf_tokenized,
        eval_dataset=val_data_hf_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
                early_stopping_patience=hp.early_stopping_patience, early_stopping_threshold=hp.early_stopping_threshold)]
    )
    # if not args.skip_training:
    #     trainer.train()
    trainer.evaluate()
    # trainer.push_to_hub(commit_message=f"{run_name} - Training complete")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-rn', '--run_name', type=str, required=True)
    parser.add_argument('-st', '--skip_training', action='store_true')
    parser.add_argument('-ti', '--topic_id', type=int, default=1)
    args = parser.parse_args()
    main(args)