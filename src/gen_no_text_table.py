import os
import json

NB_algos = ['gaussian', 'multinom', 'complement', 'bernoulli', 'categorical']
features = {
    "all_feats": "E+I+T",
    "examiner_namer": "E",
    "inventor_list": "I",
    "topic_id": "T",
    "no_inventor": "E+T",
}
rows = []
for algo in NB_algos:
    for name, feats in features.items():
        folder = algo+"_"+name
        path = os.path.join("experiments", folder, "metrics.json")
        metrics = json.load(open(path)) # print(feats, metrics)
        rows.append([algo, feats]+[f"{100*v:.2f}" for v in metrics.values()])

print("Model & Features & Accuracy & F1 macro & F1 weighted")
for row in rows: print(" & ".join(row)+" \\\\")