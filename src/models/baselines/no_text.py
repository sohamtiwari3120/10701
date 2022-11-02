import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm
from sklearn.metrics import f1_score
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

def evaluate_NB(data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], 
                algo: str="gaussian", feats: List[str]=["examiner_namer", 
                "inventor_list", "topic_id"]):
    train, val, test = data
    label_map = {"REJECTED": 0, "ACCEPTED": 1, "PENDING": 2}
    class_map = {
        "gaussian": GaussianNB, 
        "multinom": MultinomialNB, 
        "complement": ComplementNB,
        "bernoulli": BernoulliNB,
        "categorical": CategoricalNB,
    }
    assert algo in ["gaussian", "multinom", ""], NotImplementedError(f"{algo} doesn't have a Naive Bayes implementation in sklearn")
    model = class_map[algo]()
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

if __name__ == "__main__":
    pass