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
for layer in [1,2,5]:
    for name, feats in features.items():
        folder = f"MLP_{layer}HL_"+name
        path = os.path.join("experiments", folder, "test_stats.json")
        metrics = json.load(open(path)) # print(feats, metrics)
        rows.append(
            [f"MLP_{layer}", feats]+
            [f"{100*v:.2f}" for v in [metrics["acc"], metrics["f1_macro"], metrics["f1_weighted"]]]
        )

print("Model & Features & Accuracy & F1 macro & F1 weighted")
for row in rows: print(" & ".join(row)+" \\\\")