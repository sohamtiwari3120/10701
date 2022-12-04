import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_cond_accept_ratio_hist(x, y, cond_var="topic"):
    plt.clf()
    path = f"plots/{cond_var}_cond_accept_ratio_hist.png"
    plt.title(f"Histogram of patent acceptance ratios per {cond_var}")
    plt.xlabel("Range of acceptance ratio")
    plt.ylabel(f"No. of {cond_var}s within acceptance ratio range")
    color = {"topic": "red", "examiner": "green"}.get(cond_var, "blue")
    plt.bar(x, y, color=color)
    plt.xticks(x, x, rotation=90)
    plt.tight_layout()
    plt.savefig(path)
    
DATA_PATH = "../../data/patent_acceptance_pred_subs=0.2.jsonl"
def read_jsonl(path: str):
    recs = []
    i = 0
    with open(path, "r") as f:
        for line in tqdm(f):
            i += 1
            line = line.strip()
            recs.append(json.loads(line))
            # if i == 500000: break
    return recs

data = read_jsonl(DATA_PATH)
df = pd.DataFrame(data)
# conditioned on the topic.
unique_topics = df[df['topic_id'] != ""]['topic_id'].apply(lambda x: x[:4]).unique()
ratio_dist = []
for topic in unique_topics:
    dist = df[df["topic_id"].apply(lambda x: x[:4]) == topic]["decision"].value_counts()
    acc = dist.get('ACCEPTED', 0)
    rej = dist.get('REJECTED', 0)
    n = acc + rej
    ratio = acc / (acc + rej)
    ratio_dist.append(ratio)
print(ratio_dist)
print(np.max(ratio_dist))
print(np.min(ratio_dist))
hist = np.histogram(ratio_dist)
x = [f'{float(hist[1][i]):.1f}-{float(hist[1][i+1]):.1f}' for i in range(0, len(hist[1])-1)]
y = hist[0]
plot_cond_accept_ratio_hist(x, y, "topic")

examiners = df['examiner_namer'].apply(lambda x: " ".join(x.split()))
examiner_dist = {ex: {"acc": 0, "rej": 0} for ex in examiners.unique()}
for i in tqdm(range(len(examiners))):
    ex = examiners[i]
    dec = df["decision"][i]
    if dec == "ACCEPTED": examiner_dist[ex]['acc'] += 1
    elif dec == "REJECTED": examiner_dist[ex]['rej'] += 1
ex_ratio_dist = []
for k, v in tqdm(examiner_dist.items()):
    a = v['acc']
    r = v['rej']
    if a == 0 and r == 0: a = 1
    ex_ratio_dist.append(a/(a+r))
hist = np.histogram(ex_ratio_dist)
x = [f'{float(hist[1][i]):.1f}-{float(hist[1][i+1]):.1f}' for i in range(0, len(hist[1])-1)]
y = hist[0]
plot_cond_accept_ratio_hist(x, y, "topic")
# conditioned on the examiner name.